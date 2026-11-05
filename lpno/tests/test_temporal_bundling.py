"""Temporal bundling (RIGNO-style): one forward predicts frames t+1..t+B.

Upstream RIGNO's AutoregressiveStepper emits `num_steps_direct` frames from a
single input snapshot and then jumps (stepping.py:369-404), so a rollout
composes ~B x fewer times and error compounds ~B x less. Here: the AR window
dataset carries the future frames as bundled outvar keys `{key}_bnd{i}`, the
model widens its output head, and the rollout loop advances B frames per
forward. `bundle_steps=1` leaves every existing path byte-identical.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lpno.models.pipeline.preprocessing_datasets import _ar_window_sample

CONTEXT = ["P", "V_scan", "x"]
STATE = ["T", "alpha"]
EXTRA = ["fl"]
SUFFIX = "_prev"


def _temporal_var(n_time: int = 8) -> dict[str, np.ndarray]:
    shape = (1, n_time, 1, 2, 2, 2)
    out = {}
    for i, key in enumerate(CONTEXT + STATE + EXTRA):
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        out[key] = base + 100.0 * i
    return out


def _frames(var: dict[str, np.ndarray], t0: int, count: int) -> list[dict[str, np.ndarray]]:
    return [{key: arr[0, t0 + j] for key, arr in var.items()} for j in range(count)]


def test_ar_window_sample_bundle_mode_keys_and_frames() -> None:
    var = _temporal_var()
    B = 3
    invar, outvar = _ar_window_sample(
        _frames(var, 0, B + 1),
        context_keys=CONTEXT,
        state_keys=STATE,
        state_prev_suffix=SUFFIX,
        unroll_steps=1,
        extra_target_keys=EXTRA,
        bundle_steps=B,
    )
    # invar carries step-0 context + per-bundled-frame future context (see
    # test_bundle_invar_carries_future_context), but never fed-back targets.
    assert not any("__tgt_ar" in k for k in invar)
    for key in CONTEXT:
        np.testing.assert_allclose(invar[key], var[key][0, 1])
    for key in STATE:
        np.testing.assert_allclose(invar[f"{key}{SUFFIX}"], var[key][0, 0])
    # outvar: base frame plus bundled future frames for the state keys only.
    for key in STATE:
        np.testing.assert_allclose(outvar[key], var[key][0, 1])
        for i in range(1, B):
            np.testing.assert_allclose(outvar[f"{key}_bnd{i}"], var[key][0, 1 + i])
    np.testing.assert_allclose(outvar["fl"], var["fl"][0, 1])
    assert not any(k.startswith("fl_bnd") for k in outvar)


def test_bundle_invar_carries_future_context() -> None:
    """Frames t+2..t+B need the beam where it WILL be, not where it is.

    Measured on the kh2579 data: the laser moves 0.57 beam radii across a
    4-frame bundle, and the path is random — so predicting those frames from
    the step-0 context alone is ill-posed (bundle4 run: flat train loss, val
    error rising). The scan path is planned and known at inference, so the
    future exogenous context is legitimately an input.
    """
    var = _temporal_var()
    B = 3
    invar, _ = _ar_window_sample(
        _frames(var, 0, B + 1),
        context_keys=CONTEXT,
        state_keys=STATE,
        state_prev_suffix=SUFFIX,
        unroll_steps=1,
        extra_target_keys=EXTRA,
        bundle_steps=B,
    )
    for key in CONTEXT:
        np.testing.assert_allclose(invar[key], var[key][0, 1])
        for i in range(1, B):
            np.testing.assert_allclose(invar[f"{key}__ar{i}"], var[key][0, 1 + i])
    # State is still predicted, never handed over: only the seed prev exists.
    for key in STATE:
        assert f"{key}__ar1" not in invar
        assert not any(k.startswith(f"{key}__tgt") for k in invar)


def test_ar_window_sample_bundle_rejects_unrolling() -> None:
    var = _temporal_var()
    with pytest.raises(ValueError):
        _ar_window_sample(
            _frames(var, 0, 4),
            context_keys=CONTEXT,
            state_keys=STATE,
            state_prev_suffix=SUFFIX,
            unroll_steps=3,
            bundle_steps=3,
        )


# ---- rollout: reuse the stubbed postprocessing loader from the rollout tests ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_autoregressive_rollout import _load_postprocessing_module  # noqa: E402


class _BundleRolloutModel(torch.nn.Module):
    """T frames: prev + x (frame 1) and prev + 2x (frame 2); alpha passthrough."""

    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, invar):
        # Frame 2 uses the FUTURE context x__ar1, proving it is wired through.
        return {
            "T": invar["T_prev"] + invar["x"],
            "T_bnd1": invar["T_prev"] + invar["x"] + invar["x__ar1"],
            "alpha": invar["alpha_prev"],
            "alpha_bnd1": invar["alpha_prev"],
        }


def _make_bundle_run_cfg():
    return SimpleNamespace(
        autoregressive_enabled=True,
        autoregressive_eval_use_gt_init=True,
        autoregressive_context_keys=["x"],
        autoregressive_state_keys=["T", "alpha"],
        autoregressive_state_prev_suffix="_prev",
        autoregressive_eval_horizon_k=10,
        autoregressive_bundle_steps=2,
        inference_batch_size=2,
        alpha_masking=False,
        material_params={"T_s": 0.2, "T_l": 0.8},
        ref_scales={"T": 1.0},
    )


def test_rollout_bundle_jump_even_horizon(tmp_path: Path):
    module = _load_postprocessing_module()
    # T=5, x forcing 0..4. jump0: T=0+x1=1, T_bnd1=0+x1+x2=3 (state->3);
    # jump1: T=3+x3=6, T_bnd1=3+x3+x4=10.
    x = np.arange(5, dtype=np.float32).reshape(1, 5, 1, 1, 1, 1)
    T = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32).reshape(1, 5, 1, 1, 1, 1)
    alpha = np.ones((1, 5, 1, 1, 1, 1), dtype=np.float32)
    fl = np.ones((1, 5, 1, 1, 1, 1), dtype=np.float32)
    trajectory_var = {"x": x, "T": T, "alpha": alpha, "fl": fl}

    out_dir = tmp_path / "fno_eval"
    metrics = module.run_autoregressive_rollout_and_metrics(
        model=_BundleRolloutModel(),
        trajectory_var=trajectory_var,
        run_cfg=_make_bundle_run_cfg(),
        output_dir=out_dir,
    )
    preds = np.load(out_dir / "rollout_predictions.npz")
    np.testing.assert_allclose(preds["T"].reshape(1, 4), [[1.0, 3.0, 6.0, 10.0]])
    assert metrics.errors["T"]["abs_err_mean"] == pytest.approx(0.0)


def test_rollout_bundle_jump_partial_last_bundle(tmp_path: Path):
    module = _load_postprocessing_module()
    # T=4 -> 3 rollout frames with B=2: full jump (1, 3) then a partial jump
    # taking only the first bundled frame (prev=3, x3=3 -> 6).
    x = np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1, 1, 1)
    T = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32).reshape(1, 4, 1, 1, 1, 1)
    alpha = np.ones((1, 4, 1, 1, 1, 1), dtype=np.float32)
    fl = np.ones((1, 4, 1, 1, 1, 1), dtype=np.float32)
    trajectory_var = {"x": x, "T": T, "alpha": alpha, "fl": fl}

    out_dir = tmp_path / "fno_eval"
    module.run_autoregressive_rollout_and_metrics(
        model=_BundleRolloutModel(),
        trajectory_var=trajectory_var,
        run_cfg=_make_bundle_run_cfg(),
        output_dir=out_dir,
    )
    preds = np.load(out_dir / "rollout_predictions.npz")
    np.testing.assert_allclose(preds["T"].reshape(1, 3), [[1.0, 3.0, 6.0]])
