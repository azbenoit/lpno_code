"""Per-step curves must cover EVERY case, not just the first.

The lazy rollout runs one case at a time and passes `write_files=(n_done == 0)`,
so `rollout_errors_per_step.csv` held only the first non-degenerate case while
`rollout_full` aggregated all of them. Averaging a curve over the split was
therefore impossible, and reading the single-case curve as the split's behaviour
is actively wrong -- on ds 55017 that case scored 53.2 K against a 9.1 K median.
"""
import csv
import numpy as np
import pytest
import torch

from lpno.models.pipeline import postprocessing as pp


class _Decay(torch.nn.Module):
    """Prediction = prev * 0.5, so each case has its own error trajectory."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, invar):
        return {"T": invar["T_prev"] * 0.5}


class _RC:
    autoregressive_enabled = True
    autoregressive_eval_use_gt_init = True
    autoregressive_context_keys = ["beam_dist"]
    autoregressive_state_keys = ["T"]
    autoregressive_state_prev_suffix = "_prev"
    autoregressive_teacher_forcing = False
    autoregressive_bundle_steps = 1
    autoregressive_eval_horizon_k = 2
    data_layout = "grid"
    inference_batch_size = 1
    ref_scales = {"T": 1.0}


def _case(nt, fill):
    a = np.full((1, nt, 1, 2, 2, 2), float(fill), dtype=np.float32)
    return {"T": a.copy(), "beam_dist": a.copy(), "fl": a.copy()}


def test_per_step_csv_covers_every_case(monkeypatch, tmp_path):
    """Two cases of different length -> both must appear, tagged by case."""
    monkeypatch.setattr(pp, "_apply_prediction_postprocessing", lambda p, *a, **k: p)
    cases = {0: _case(5, 1.0), 1: _case(4, 2.0)}

    def frame_reader(case_idx, time_idx):
        return {k: v[0, time_idx] for k, v in cases[case_idx].items()}

    pp.run_lazy_autoregressive_rollout_and_metrics(
        model=_Decay(), frame_reader=frame_reader, case_indices=[0, 1],
        case_lengths={0: 5, 1: 4}, run_cfg=_RC(), output_dir=tmp_path,
    )

    rows = list(csv.DictReader((tmp_path / "rollout_errors_per_step.csv").open()))
    assert rows, "no per-step csv written"
    assert "case_index" in rows[0], f"csv has no case column: {list(rows[0])}"
    seen = {r["case_index"] for r in rows}
    assert seen == {"0", "1"}, f"expected both cases in the per-step csv, got {seen}"
