"""The alpha mask gate must never become an absorbing dead state.

`fill_where_mask_smooth_torch` gated with ``0.5 * (tanh(k*(m-thr)) + 1)``.
For a confidently-wrong mask (alpha <~ 0 with k=20, thr=0.5) the tanh saturates
to -1.0 and the ``+1`` cancels it **exactly**, so the gate is bit-zero and both
gradient paths die:

- ``d out/d data`` = gate = 0  -> the masked field (T) gets no gradient;
- ``d gate/d alpha`` = 0       -> the mask itself gets no gradient either.

MaskNode also fills `alpha` with ``d=0.0`` wherever the gate is ~0, so a model
whose alpha dips below the threshold has its alpha *overwritten* to 0, which
pins the gate at 0 forever. With Lion (a sign-based optimiser) a zero gradient
is a zero update, so the weights stop moving entirely.

Measured live 2026-07-24 on a latent-128 RIGNO (models 13193/12567): from step
500 — exactly when the mask warmup engages — every validation metric froze
bit-identical for 3500 steps, at two different learning rates. alpha had
collapsed to exactly 0 (mae_alpha 0.7498, rmse_alpha 0.8659 = sqrt(0.7498) to
four decimals: the signature of predicting a constant 0 against a 75%-metal
domain).

`0.5*(tanh(x)+1)` is *mathematically identical* to `sigmoid(2x)`, but sigmoid
computes the small value directly instead of cancelling two near-equal numbers,
so it stays finite and differentiable far into the saturated regime. A small
floor keeps the data path alive even past sigmoid's own underflow.
"""
from __future__ import annotations

import pytest
import torch

from lpno.custom_utils.utils import fill_where_mask_smooth_torch

K, THR, FILL = 20.0, 0.5, 1.041


def _grads(alpha_value: float):
    alpha = torch.tensor([[[alpha_value]]], dtype=torch.float32, requires_grad=True)
    data = torch.tensor([[[0.7]]], dtype=torch.float32, requires_grad=True)
    out = fill_where_mask_smooth_torch(data, alpha, threshold=THR, d=FILL, channel_axis=1, k=K)
    out.sum().backward()
    return float(alpha.grad.abs().sum()), float(data.grad.abs().sum())


@pytest.mark.parametrize("alpha", [0.0, -0.2, -0.5, -1.0])
def test_data_path_survives_a_confidently_wrong_mask(alpha):
    """The masked field must keep a gradient however wrong the mask is."""
    _, d_data = _grads(alpha)
    assert d_data > 0.0, f"data gradient died at alpha={alpha}: the run can never recover"


@pytest.mark.parametrize("alpha", [0.0, -0.2, -0.5])
def test_mask_path_survives_a_confidently_wrong_mask(alpha):
    """alpha must keep a gradient too, or it can never climb back over threshold."""
    d_alpha, _ = _grads(alpha)
    assert d_alpha > 0.0, f"alpha gradient died at alpha={alpha}: absorbing state"


def test_healthy_regime_is_unchanged():
    """The fix must not move the gate where the mask is confident and correct."""
    for alpha, expected in ((1.0, 0.7), (0.9, 0.7)):
        data = torch.tensor([[[0.7]]])
        out = fill_where_mask_smooth_torch(data, torch.tensor([[[alpha]]]), threshold=THR, d=FILL, channel_axis=1, k=K)
        assert out.item() == pytest.approx(expected, abs=1e-4)
    # far below threshold still fills, to within a negligible tolerance
    out = fill_where_mask_smooth_torch(
        torch.tensor([[[0.7]]]), torch.tensor([[[0.0]]]), threshold=THR, d=FILL, channel_axis=1, k=K
    )
    assert out.item() == pytest.approx(FILL, abs=1e-3)


def test_gate_is_monotonic_in_the_mask():
    """Sanity: a higher mask value must never fill more."""
    vals = [
        fill_where_mask_smooth_torch(
            torch.tensor([[[0.7]]]), torch.tensor([[[a]]]), threshold=THR, d=FILL, channel_axis=1, k=K
        ).item()
        for a in (-0.5, 0.0, 0.3, 0.5, 0.7, 1.0)
    ]
    # fill (1.041) > data (0.7), so the blend decreases monotonically toward data
    assert all(b <= a + 1e-6 for a, b in zip(vals, vals[1:])), vals


def test_the_documented_dead_state_is_gone():
    """Regression for the exact configuration that froze RIGNO latent-128.

    alpha pinned at 0.0 by MaskNode's own d=0.0 fill: previously bit-zero
    gradients on both paths, so training could never resume.
    """
    d_alpha, d_data = _grads(0.0)
    assert d_data > 0.0 and d_alpha > 0.0
