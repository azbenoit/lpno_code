"""MaskNode must not mask the field it uses as the mask.

`mask_str` is the model's output keys minus `T_dot`, i.e. `["T", "alpha"]`, and
`mask_str_key` is `"alpha"` — so alpha was gated by itself:

    alpha_out = 0.0 + gate(alpha) * (alpha - 0.0) = gate(alpha) * alpha

That is circular and self-squashing: below the threshold `gate < 1`, so the
product drives alpha further down, which lowers the gate, which lowers alpha.
Combined with the gate saturating (fixed separately in
`test_mask_gate_no_dead_state.py`) it is an absorbing state a run cannot leave —
observed as a latent-128 RIGNO frozen bit-identically from step 500 at two
different learning rates, with alpha collapsed to exactly 0.

Masking the mask by itself is also redundant: for a converged model alpha is
already ~0 or ~1, where `gate(alpha) * alpha` equals alpha, so removing it is a
no-op in the healthy regime and only changes the broken one. The mask still
applies to every other field (T, fl), which is the point of masking.
"""
from __future__ import annotations

import pytest
import torch

from lpno.equations import MaskNode

T_FILL = 1.041


def _node(keys, mask_key="alpha"):
    return MaskNode(keys=keys, threshold=0.5, d=0.0, k=20.0, mask_str_key=mask_key, t_fill=T_FILL)


def _run(alpha_value, t_value=0.7, keys=("T", "alpha", "fl")):
    outvar = {
        "T": torch.tensor([[[t_value]]]),
        "alpha": torch.tensor([[[alpha_value]]]),
        "fl": torch.tensor([[[0.0]]]),
    }
    return _node(list(keys))(outvar)


def test_alpha_passes_through_unmasked():
    """alpha is the mask; it must not be gated by itself."""
    for alpha in (0.9, 0.5, 0.2, 0.0, -0.3):
        out = _run(alpha)
        assert out["alpha"].item() == pytest.approx(alpha, abs=1e-6), (
            f"alpha was modified by its own mask at alpha={alpha}"
        )


def test_other_fields_are_still_masked():
    """The point of masking is unaffected: T still fills outside the metal."""
    out_off = _run(0.0)
    assert out_off["T"].item() == pytest.approx(T_FILL, abs=1e-3)
    out_on = _run(1.0)
    assert out_on["T"].item() == pytest.approx(0.7, abs=1e-4)


def test_no_self_squashing_feedback():
    """Repeatedly applying the node must not drive alpha toward zero.

    The old `gate(alpha) * alpha` shrank alpha on every application, which is
    what made the dead state absorbing.
    """
    alpha = 0.45
    for _ in range(10):
        alpha = _run(alpha)["alpha"].item()
    assert alpha == pytest.approx(0.45, abs=1e-6), f"alpha decayed to {alpha}"


def test_healthy_model_is_unaffected():
    """For a converged model (alpha ~0 or ~1) the change is a no-op."""
    for alpha in (0.0, 1.0):
        out = _run(alpha)
        assert out["alpha"].item() == pytest.approx(alpha, abs=1e-6)


def test_a_different_mask_key_is_also_exempt():
    """The exemption follows mask_str_key, not the literal name 'alpha'."""
    outvar = {"T": torch.tensor([[[0.7]]]), "fl": torch.tensor([[[0.25]]])}
    out = _node(["T", "fl"], mask_key="fl")(outvar)
    assert out["fl"].item() == pytest.approx(0.25, abs=1e-6)


def test_alpha_still_gradients_through_the_other_fields():
    """Removing self-masking must not cut alpha off from the loss entirely."""
    alpha = torch.tensor([[[0.2]]], requires_grad=True)
    outvar = {"T": torch.tensor([[[0.7]]], requires_grad=True), "alpha": alpha}
    out = _node(["T", "alpha"])(outvar)
    out["T"].sum().backward()
    assert alpha.grad is not None and float(alpha.grad.abs().sum()) > 0.0
