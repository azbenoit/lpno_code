"""Tests for the autoregressive rollout-stability mitigations.

Covers the three knobs added to fight rollout error accumulation:
  #1 fed-back state clamp (_apply_prediction_postprocessing),
  #2 per-batch pushforward noise (NoisyPrevGridDataset),
  #4 multi-step unrolled training loss (build via unroll_steps>1).

All default to the pre-existing one-step behaviour, so the "off" paths here are
regression guards.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import _apply_prediction_postprocessing


def _run_cfg(**over):
    base = dict(
        alpha_masking=False,
        material_params={"T_s": 1900.0, "T_l": 1950.0},
        ref_scales={"T": 3000.0},
        autoregressive_clamp_feedback=False,
        autoregressive_clamp_t_min=273.15 / 3000.0,
        autoregressive_clamp_t_max=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _preds():
    # deliberately unphysical fed-back state, like the DeepONet rollout drift
    return {
        "alpha": np.array([[[[[1.28, -0.1]]]]], dtype=np.float32),
        "T": np.array([[[[[-0.5, 2.0]]]]], dtype=np.float32),
    }


def test_clamp_on_bounds_alpha_and_floors_T():
    cfg = _run_cfg(autoregressive_clamp_feedback=True)
    out = _apply_prediction_postprocessing(_preds(), cfg, target_keys=["T", "alpha"])
    assert out["alpha"].min() >= 0.0 and out["alpha"].max() <= 1.0
    assert out["T"].min() >= 273.15 / 3000.0 - 1e-6
    # no upper cap by default: the legitimately-high cell survives
    assert out["T"].max() == pytest.approx(2.0)


def test_clamp_off_is_passthrough():
    cfg = _run_cfg(autoregressive_clamp_feedback=False)
    out = _apply_prediction_postprocessing(_preds(), cfg, target_keys=["T", "alpha"])
    assert out["alpha"].max() == pytest.approx(1.28)
    assert out["T"].min() == pytest.approx(-0.5)


def test_clamp_t_max_caps_superheat():
    cfg = _run_cfg(autoregressive_clamp_feedback=True, autoregressive_clamp_t_max=1.041)
    out = _apply_prediction_postprocessing(_preds(), cfg, target_keys=["T", "alpha"])
    assert out["T"].max() <= 1.041 + 1e-6


# ---- #2 per-batch pushforward noise -------------------------------------------------

torch = pytest.importorskip("torch")
from lpno.models.pipeline.preprocessing_datasets import NoisyPrevGridDataset  # noqa: E402


def _grid_ds():
    inv = {
        "ctx": np.ones((2, 1, 2, 2, 2), dtype=np.float32),
        "T_prev": np.ones((2, 1, 2, 2, 2), dtype=np.float32),
    }
    out = {"T": np.ones((2, 1, 2, 2, 2), dtype=np.float32)}
    return inv, out


def test_pushforward_noise_perturbs_only_prev_and_resamples():
    inv, out = _grid_ds()
    ds = NoisyPrevGridDataset(inv, out, prev_keys=["T_prev"], noise_std=0.1, seed=0)
    a_in, _, _ = ds[0]
    b_in, _, _ = ds[0]
    # fresh noise each access on the prev key...
    assert not torch.allclose(a_in["T_prev"], b_in["T_prev"])
    # ...but context is untouched and stays clean
    assert torch.allclose(a_in["ctx"], torch.ones_like(a_in["ctx"]))
    # perturbation magnitude is in the right ballpark (std ~ noise_std)
    assert 0.02 < float((a_in["T_prev"] - 1.0).std()) < 0.5


def test_pushforward_noise_zero_is_passthrough():
    inv, out = _grid_ds()
    ds = NoisyPrevGridDataset(inv, out, prev_keys=["T_prev"], noise_std=0.0, seed=0)
    a_in, _, _ = ds[0]
    assert torch.allclose(a_in["T_prev"], torch.ones_like(a_in["T_prev"]))


# ---- #4 multi-step unrolled loss ----------------------------------------------------

from lpno.models.pipeline.preprocessing_datasets import (  # noqa: E402
    _build_autoregressive_window_pairs,
)
from lpno.custom_utils.ar_constraints import UnrolledGridConstraint  # noqa: E402


def _temporal_var():
    shape = (2, 4, 1, 2, 2, 2)  # (n_batch, n_time, C, X, Y, Z)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return {"x": base.copy(), "T": base * 2.0, "alpha": base * 3.0, "fl": base * 0.5}


def test_window_pairs_shapes_and_frames():
    var = _temporal_var()
    inv, out = _build_autoregressive_window_pairs(
        var,
        context_keys=["x"],
        state_keys=["T", "alpha"],
        state_prev_suffix="_prev",
        unroll_steps=2,
        extra_target_keys=["fl"],
    )
    n_batch, n_time, k = 2, 4, 2
    n_windows = n_time - k  # 2
    N = n_batch * n_windows  # 4

    def flat(arr, start):
        return arr[:, start : start + n_windows].reshape(N, 1, 2, 2, 2)

    # base context = GT[t0+1]; future context ar1 = GT[t0+2]
    np.testing.assert_allclose(inv["x"], flat(var["x"], 1))
    np.testing.assert_allclose(inv["x__ar1"], flat(var["x"], 2))
    # seed prev = GT[t0]
    np.testing.assert_allclose(inv["T_prev"], flat(var["T"], 0))
    # base target = GT[t0+1]; future target rides in ds_input = GT[t0+2]
    np.testing.assert_allclose(out["T"], flat(var["T"], 1))
    np.testing.assert_allclose(inv["T__tgt_ar1"], flat(var["T"], 2))
    # only base producible keys drive the Graph
    assert set(out.keys()) == {"T", "alpha", "fl"}
    assert "fl__tgt_ar1" in inv and "alpha__tgt_ar1" in inv


def _toy_constraint(detach):
    """UnrolledGridConstraint with __init__ bypassed and a toy context-aware model
    T_next = T_prev + w*x, so the k-step feedback and its gradient are analytic."""
    c = UnrolledGridConstraint.__new__(UnrolledGridConstraint)
    c._ctx_keys = ["x"]
    c._state_keys = ["T"]
    c._prev_suffix = "_prev"
    c._k = 2
    c._detach = detach
    c._unroll_preds = None
    c._step_invars = None
    w = torch.nn.Parameter(torch.tensor([[1.0]]))
    c.model = lambda inv: {"T": inv["T_prev"] + w * inv["x"]}
    c._loss = lambda invar, pred, true, lam, step: {
        key: ((pred[key] - true[key]) ** 2).sum() for key in pred
    }
    return c, w


def test_unrolled_feedback_chain():
    c, _ = _toy_constraint(detach=False)
    c._input_vars = {
        "x": torch.ones(1, 1),
        "x__ar1": torch.full((1, 1), 2.0),
        "T_prev": torch.zeros(1, 1),
        "T__tgt_ar1": torch.full((1, 1), 3.0),
    }
    c._target_vars = {"T": torch.ones(1, 1)}
    c.forward()
    p0, p1 = c._unroll_preds
    assert float(p0["T"].detach()) == 1.0  # seed 0 + x 1
    assert float(p1["T"].detach()) == 3.0  # p0 1 + x__ar1 2 -> prev was fed back
    assert float(c.loss(0)["T"].detach()) == 0.0  # both steps hit their targets


@pytest.mark.parametrize("detach,expected_grad", [(False, 5.0), (True, 3.0)])
def test_unrolled_loss_gradient_bptt_vs_detach(detach, expected_grad):
    # w=1: loss0=w^2 (grad 2), loss1=(2w)^2 with BPTT (grad 8) or (const+w)^2 detached
    # (grad 4); total=(loss0+loss1)/2 -> d/dw = (2+8)/2=5 (BPTT) or (2+4)/2=3 (detach).
    c, w = _toy_constraint(detach=detach)
    c._input_vars = {
        "x": torch.ones(1, 1),
        "x__ar1": torch.ones(1, 1),
        "T_prev": torch.zeros(1, 1),
        "T__tgt_ar1": torch.zeros(1, 1),
    }
    c._target_vars = {"T": torch.zeros(1, 1)}
    c.forward()
    c.loss(0)["T"].backward()
    assert w.grad is not None
    assert float(w.grad) == pytest.approx(expected_grad)
