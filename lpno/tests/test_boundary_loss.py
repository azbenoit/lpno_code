"""Tail-matching loss on the substrate-metal temperature peak.

## Why not a boundary / signed-distance loss on the melt mask

Tried 2026-08-04 and REJECTED before any GPU time, by measurement:

`fl` is not a model output. It is derived from T through
`smooth_liquid_fraction_torch`, which AT THE TIME was a tanh centred on 1898 K.
In float32 `torch.tanh` returns exactly -1.0 for arguments <= -9.011, so
`d(fl)/dT` was **bit-zero for every T below 1717.8 K**. (That saturation was
fixed later the same day by moving to the identical `sigmoid(2x)` form, so the
gradient is now alive -- but only ~7.6e-10 at 1631 K against L2 gradients of
order 1e-1, which does not change the conclusion below.) Model 29125 peaks at 1631 K, 87 K below
that dead zone: a mask-space loss measured `max|dL/dT| = 0.0` with zero live
voxels. It only produces gradient on models that already melt, which is exactly
what it was meant to bootstrap.

Two further defects found in review: the +-32 cell clip made the far field
outweigh the pool ~9429x, pushing the model globally COLDER; and with
`alpha_masking` the gas fill at ~3560 K reads as `fl = 1.0`, so ~40% of the
domain contributed an ungradient-able constant.

A trap worth recording: the first version of this test passed `pred_soft` as a
LEAF tensor, so it only checked differentiability w.r.t. the mask -- never in
doubt -- and passed while the real `fl(T)` chain had exactly zero gradient. Any
test here must run through T.

## What replaced it

`topk_tail_loss` matches the mean of the k hottest substrate-metal cells between
prediction and truth. There is no tanh in the path, so `d/dT` is `1/k`,
scale-free, and lands on exactly the cells whose deficit causes the melt
collapse. It is NOT `melt_weight` (a recorded regression): that reweighted the
pointwise L2 by `1 + w*fl_true` and so was zero wherever truth had no melt,
merely amplifying an error L2 already saw. This is a statistic mismatch on the
tail.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lpno.custom_utils.losses import topk_tail_loss
from lpno.custom_utils.utils import smooth_liquid_fraction_torch

T_S, T_L, REF_T = 1873.0, 1923.0, 3000.0


def test_fl_gradient_survives_the_old_dead_zone():
    """UPDATED 2026-08-04 when `smooth_liquid_fraction_*` moved to sigmoid form.

    This test previously asserted the OPPOSITE -- that `d(fl)/dT` is bit-zero
    below 1717.8 K -- because that was true of the tanh implementation and was
    the measurement that killed the mask-space boundary loss. The tanh was then
    replaced by the algebraically identical `sigmoid(2x)`, which computes the
    small value instead of cancelling two near-equal numbers, so the gradient is
    now alive throughout. The oracle changed because the code intentionally did.

    The rejection of the boundary loss still stands on its other grounds: even
    alive, `d(fl)/dT` at 1631 K is ~7.6e-10 against L2 gradients of order 1e-1,
    and above the liquidus `fl` saturates at 1.0 *because the physics says so*,
    so no fl-based term can ever influence superheat. See
    tests/test_liquid_fraction_sigmoid.py for the forward-equivalence proof.
    """
    for peak_K in (1631.0, 1700.0, 1944.0):
        t = torch.tensor([[peak_K / REF_T]], requires_grad=True)
        smooth_liquid_fraction_torch(t, T_S / REF_T, T_L / REF_T).sum().backward()
        assert float(t.grad.abs().max()) > 0.0, f"gradient died at {peak_K} K"


def _field(peak_K, n=12):
    """Substrate-metal temperatures (normalised) with a controlled peak."""
    t = torch.full((1, 1, n, n, n), 300.0 / REF_T)
    t[0, 0, -1, -1, :] = peak_K / REF_T
    return t


def test_gradient_is_live_where_the_mask_loss_was_dead():
    """Same 1631 K model state; this term must produce real gradient."""
    pred = _field(1631.0).clone().requires_grad_(True)
    true = _field(2200.0)
    mask = torch.ones_like(true, dtype=torch.bool)
    topk_tail_loss(pred, true, mask, k=8).backward()
    assert float(pred.grad.abs().max()) > 0.0


def test_loss_is_zero_when_the_tails_match():
    f = _field(2200.0)
    mask = torch.ones_like(f, dtype=torch.bool)
    assert float(topk_tail_loss(f, f, mask, k=8)) == pytest.approx(0.0)


def test_loss_grows_with_the_peak_deficit():
    true = _field(2200.0)
    mask = torch.ones_like(true, dtype=torch.bool)
    near = topk_tail_loss(_field(2100.0), true, mask, k=8)
    far = topk_tail_loss(_field(1631.0), true, mask, k=8)
    assert float(far) > float(near)


def test_gradient_pushes_the_prediction_up_when_under_predicting():
    """The failure is under-prediction; the gradient must raise T, not lower it."""
    pred = _field(1631.0).clone().requires_grad_(True)
    true = _field(2200.0)
    mask = torch.ones_like(true, dtype=torch.bool)
    topk_tail_loss(pred, true, mask, k=8).backward()
    hottest = pred.grad[0, 0, -1, -1, :]
    assert (hottest < 0).any(), "descending the loss must increase the hottest cells"


def test_mask_excludes_non_metal_cells():
    """Gas carries a ~3560 K boiling fill; including it would make the tail meaningless."""
    pred = _field(1631.0)
    true = _field(2200.0)
    gas = torch.zeros_like(true, dtype=torch.bool)
    gas[0, 0, 0, 0, :] = True  # only a cell that is NOT the hot column
    masked = float(topk_tail_loss(pred, true, gas, k=2))
    everywhere = float(topk_tail_loss(pred, true, torch.ones_like(gas), k=2))
    assert masked != everywhere


def test_empty_mask_contributes_nothing():
    f = _field(2200.0)
    empty = torch.zeros_like(f, dtype=torch.bool)
    assert float(topk_tail_loss(f, f, empty, k=8)) == pytest.approx(0.0)


def test_tail_term_is_disabled_not_silently_gas_weighted(capsys):
    """Without true alpha the term must switch OFF, not fall back to the domain.

    With alpha masking the gas carries a ~3560 K boiling fill in BOTH prediction
    and truth, so a whole-domain top-k would match on both sides, give ~0, and
    silently contribute nothing for an entire 35k-step run.
    """
    import torch
    from lpno.custom_utils.losses import GridGradientLoss

    loss_fn = GridGradientLoss(gradient_weight=0.0, tail_weight=1000.0, tail_k=4,
                               t_solidus=1873/3000, t_liquidus=1923/3000)
    pred = {"T": torch.full((1, 1, 4, 4, 4), 0.5, requires_grad=True)}
    true = {"T": torch.full((1, 1, 4, 4, 4), 0.9)}          # no "alpha" key
    lw = {"T": torch.ones((1, 1, 4, 4, 4))}
    out = loss_fn({}, pred, true, lw, step=10)
    assert "tail term is DISABLED" in capsys.readouterr().out
    # loss must be the plain pointwise term only
    expected = float((lw["T"] * (pred["T"] - true["T"]).pow(2)).sum())
    assert float(out["T"]) == pytest.approx(expected, rel=1e-5)
