"""Velocity outputs scored only inside the melt pool, engaged after a warmup.

Design (2026-08-09, straight-path-steady velocity arms):

- **Pool mask on both sides.** GT velocity is physically ~0 in solid metal and
  garbage in the gas (with alpha masking the gas carries fill values), so u/v/w
  are scored as ``mask * (pred - true)`` with
  ``mask = fl(T_true) * sigmoid((alpha_true - 0.5)/vel_tau)`` — smooth, from
  TRUE fields, detached. Multiplying both sides by the same mask is identical
  to masking the residual. The alpha factor matters: with alpha masking the gas
  T fill is ~boiling, so ``fl(T_fill)=1`` and a T-only mask would score the
  entire gas region (the exact trap the tail term documents).
- **Detached warmup, not a zero loss.** The aggregator is homoscedastic
  uncertainty: each key's weight is ``exp(-log_var)`` with
  ``d/dlog_var = 1 - exp(-log_var)*L``. A loss held at literal zero for 4k
  steps drives ``log_var`` down every step, and the weight is enormous when
  the term finally engages. Instead the velocity loss VALUE is computed from
  step 0 (so sigma calibrates to its true scale) but reaches the model as
  ``s*L + (1-s)*L.detach()`` — no model gradient until ``vel_warmup_steps``,
  then a linear ramp over ``vel_ramp_steps``.
- **No H1 gradient term on velocity keys.** The mask edge would manufacture
  spatial-gradient mismatch at the pool boundary that neither field has.
"""

from __future__ import annotations

import pytest
import torch

from lpno.custom_utils.losses import GridGradientLoss

T_S, T_L, REF = 1873.0 / 3000, 1923.0 / 3000, 3000.0
N = 8


def _loss(vel_warmup=0, vel_ramp=0, gradient_weight=0.0):
    return GridGradientLoss(
        gradient_weight=gradient_weight,
        t_solidus=T_S,
        t_liquidus=T_L,
        vel_warmup_steps=vel_warmup,
        vel_ramp_steps=vel_ramp,
    )


def _fields(with_gas=False):
    """T with a molten corner column, metal alpha, velocity present only there."""
    T = torch.full((1, 1, N, N, N), 300.0 / REF)
    T[0, 0, -1, -1, :] = 2200.0 / REF          # molten column
    alpha = torch.ones_like(T)
    if with_gas:
        alpha[0, 0, 0, 0, :] = 0.0             # gas cells...
        T[0, 0, 0, 0, :] = 3400.0 / REF        # ...carrying a boiling fill
    u_true = torch.zeros_like(T)
    u_true[0, 0, -1, -1, :] = 0.5              # flow only in the pool
    lw = {k: torch.ones_like(T) for k in ("T", "u")}
    return T, alpha, u_true, lw


def _forward(loss_fn, step, with_gas=False, u_pred_val=0.1):
    T, alpha, u_true, lw = _fields(with_gas)
    u_pred = torch.full_like(u_true, u_pred_val).requires_grad_(True)
    out = loss_fn({}, {"T": T.clone(), "u": u_pred}, {"T": T, "alpha": alpha, "u": u_true},
                  {"T": lw["T"], "u": lw["u"]}, step=step)
    return out, u_pred


def test_gradient_lands_only_in_the_pool():
    out, u_pred = _forward(_loss(), step=0)
    out["u"].backward()
    g = u_pred.grad[0, 0]
    assert float(g[-1, -1, :].abs().min()) > 1e-8, "no gradient inside the pool"
    assert float(g[0, 1, :].abs().max()) < 1e-12, "gradient leaked into cold solid"


def test_gas_boiling_fill_is_excluded_by_the_alpha_factor():
    """fl(T_fill)=1 in the gas; without the alpha factor it would be scored."""
    out, u_pred = _forward(_loss(), step=0, with_gas=True)
    out["u"].backward()
    assert float(u_pred.grad[0, 0, 0, 0, :].abs().max()) < 1e-9, "gas cells scored"


def test_warmup_detaches_the_model_but_keeps_the_value():
    """The homoscedastic sigma needs the value; the model must get no gradient."""
    out, u_pred = _forward(_loss(vel_warmup=4000, vel_ramp=1000), step=100)
    assert float(out["u"]) > 0.0, "value must be present during warmup"
    out["u"].backward()
    assert float(u_pred.grad.abs().max()) == 0.0, "model gradient during warmup"


def test_ramp_reaches_the_full_gradient():
    out_w, u_w = _forward(_loss(vel_warmup=4000, vel_ramp=1000), step=5000)
    out_0, u_0 = _forward(_loss(), step=0)
    out_w["u"].backward(); out_0["u"].backward()
    assert torch.allclose(u_w.grad, u_0.grad), "post-ramp gradient != no-warmup gradient"


def test_mid_ramp_gradient_is_scaled():
    out_m, u_m = _forward(_loss(vel_warmup=4000, vel_ramp=1000), step=4500)
    out_0, u_0 = _forward(_loss(), step=0)
    out_m["u"].backward(); out_0["u"].backward()
    assert torch.allclose(u_m.grad, 0.5 * u_0.grad, rtol=1e-5)


def test_missing_alpha_raises_loudly():
    T, alpha, u_true, lw = _fields()
    loss_fn = _loss()
    with pytest.raises(ValueError, match="alpha"):
        loss_fn({}, {"T": T.clone(), "u": u_true.clone()}, {"T": T, "u": u_true},
                {"T": lw["T"], "u": lw["u"]}, step=0)


def test_velocity_keys_skip_the_h1_gradient_term():
    """gradient_weight must change T's loss but not u's."""
    a, _ = _forward(_loss(gradient_weight=0.0), step=0)
    b, _ = _forward(_loss(gradient_weight=5.0), step=0)
    assert float(a["u"]) == pytest.approx(float(b["u"]))
    assert float(b["T"]) >= float(a["T"])


def test_non_velocity_keys_are_untouched():
    """With no velocity keys the loss must behave exactly as before."""
    T, alpha, _, lw = _fields()
    plain = GridGradientLoss(gradient_weight=0.0)
    vel = _loss(vel_warmup=4000)
    pred = (T + 0.01).clone()
    a = plain({}, {"T": pred}, {"T": T}, {"T": lw["T"]}, step=0)
    b = vel({}, {"T": pred}, {"T": T}, {"T": lw["T"]}, step=0)
    assert float(a["T"]) == pytest.approx(float(b["T"]))
