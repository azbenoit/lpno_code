"""Teacher-forced rollout: feed GROUND TRUTH prev each step, not the prediction.

Without this the only baseline for a free-running rollout curve is its own step-1
error held flat, which misrepresents the comparison -- late frames are harder than
early ones, so a real teacher-forced reference fluctuates with frame difficulty.
"""
import numpy as np
import pytest
import torch

from lpno.models.pipeline import postprocessing as pp


class _AddOne(torch.nn.Module):
    """Predicts prev + 1. Free-running compounds (+1,+2,+3...); forced never does."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))  # device probe only

    def forward(self, invar):
        return {"T": invar["T_prev"] + 1.0}


def _run(monkeypatch, teacher_forcing):
    # GT is the constant 0 trajectory, so every forced step predicts exactly 1.0.
    n_time = 5
    z = np.zeros((1, n_time, 1, 2, 2, 2), dtype=np.float32)
    traj = {"T": z.copy(), "beam_dist": z.copy(), "fl": z.copy()}

    class RC:
        autoregressive_enabled = True
        autoregressive_eval_use_gt_init = True
        autoregressive_context_keys = ["beam_dist"]
        autoregressive_state_keys = ["T"]
        autoregressive_state_prev_suffix = "_prev"
        autoregressive_teacher_forcing = teacher_forcing
        autoregressive_bundle_steps = 1
        autoregressive_eval_horizon_k = 2
        data_layout = "grid"
        inference_batch_size = 1
        ref_scales = {"T": 1.0}

    monkeypatch.setattr(pp, "_apply_prediction_postprocessing", lambda p, *a, **k: p)
    collect = {}
    pp.run_autoregressive_rollout_and_metrics(
        model=_AddOne(), trajectory_var=traj, run_cfg=RC(),
        output_dir=None, write_files=False, collect=collect,
    )
    return collect["errors_full"]["T"]["abs_err_mean"]


def test_free_running_compounds(monkeypatch):
    """Errors are +1,+2,+3,+4 against a zero GT -> mean 2.5."""
    assert _run(monkeypatch, False) == pytest.approx(2.5)


def test_teacher_forced_does_not_compound(monkeypatch):
    """Every step restarts from GT (0), so every error is exactly 1.0 -- no drift."""
    assert _run(monkeypatch, True) == pytest.approx(1.0)
