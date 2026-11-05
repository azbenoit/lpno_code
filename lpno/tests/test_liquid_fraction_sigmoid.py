"""The liquid fraction must be a sigmoid, not a tanh, for gradient reasons only.

`0.5*(tanh(x)+1)` and `sigmoid(2x)` are the same function algebraically. They are
not the same numerically: float32 `tanh` saturates to exactly -1.0 for
x <= -9.011, so the tanh form had a BIT-ZERO derivative for every T below
1717.8 K. Model 29125 peaks at 1631 K, so every loss routed through `fl` was
silent on exactly the models that needed it (measured 2026-08-04).

This file pins both halves of the change:
  - the gradient is alive where the tanh form was bit-zero, and
  - the forward values are unchanged, so no stored metric or prior ablation
    shifts meaning. That second half is what made the change safe to land.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lpno.custom_utils.utils import (
    smooth_liquid_fraction_numpy,
    smooth_liquid_fraction_torch,
)

T_S, T_L, REF_T = 1873.0, 1923.0, 3000.0
THRESHOLD = 0.5 * (T_S + T_L)  # 1898 K


def _tanh_reference(T, T_s, T_l, beta=2.5):
    """The previous implementation, kept here as the equivalence oracle."""
    phi = (np.asarray(T) - T_s) / (T_l - T_s)
    return 0.5 * (np.tanh(beta * (phi - 0.5)) + 1)


def test_forward_values_are_unchanged():
    """Worst difference must stay at float32 epsilon, or stored metrics shift."""
    T = np.linspace(300.0, 4000.0, 200001) / REF_T
    new = smooth_liquid_fraction_numpy(T, T_S / REF_T, T_L / REF_T)
    old = _tanh_reference(T, T_S / REF_T, T_L / REF_T)
    assert np.max(np.abs(new - old)) < 1e-6


def test_no_threshold_decision_changes():
    """`fl > 0.5` drives every melt metric; not one may flip."""
    T = np.linspace(300.0, 4000.0, 200001) / REF_T
    new = smooth_liquid_fraction_numpy(T, T_S / REF_T, T_L / REF_T) > 0.5
    old = _tanh_reference(T, T_S / REF_T, T_L / REF_T) > 0.5
    assert int((new != old).sum()) == 0


def test_threshold_is_still_the_midpoint():
    assert smooth_liquid_fraction_numpy(
        np.array([THRESHOLD / REF_T]), T_S / REF_T, T_L / REF_T
    )[0] == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("peak_K", [1631.0, 1700.0])
def test_gradient_is_alive_where_tanh_was_bit_zero(peak_K):
    """The whole point: 29125 sits at 1631 K, inside the old dead zone."""
    t = torch.tensor([[peak_K / REF_T]], requires_grad=True)
    smooth_liquid_fraction_torch(t, T_S / REF_T, T_L / REF_T).sum().backward()
    assert float(t.grad.abs().max()) > 0.0


def test_torch_and_numpy_agree():
    """Train and eval must use the same mask, or scores are of a different model."""
    T = np.linspace(1400.0, 2600.0, 5001) / REF_T
    a = smooth_liquid_fraction_numpy(T, T_S / REF_T, T_L / REF_T)
    b = smooth_liquid_fraction_torch(torch.tensor(T), T_S / REF_T, T_L / REF_T).numpy()
    assert np.max(np.abs(a - b)) < 1e-6
