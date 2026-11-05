"""Lazy per-case AR rollout eval: streamed twin of the eager rollout metrics.

The AiiDA eval lane stages the lazy arraydir, where var_train/var_val are None
and the eager rollout crashed (TypeError on `key in None`). The lazy twin
materializes one case at a time and frame-weight-combines the per-case stats.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_autoregressive_rollout import _load_postprocessing_module  # noqa: E402


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, invar):
        return {"T": invar["T_prev"] + invar["x"], "alpha": invar["alpha_prev"]}


def _run_cfg():
    return SimpleNamespace(
        autoregressive_enabled=True,
        autoregressive_eval_use_gt_init=True,
        autoregressive_context_keys=["x"],
        autoregressive_state_keys=["T", "alpha"],
        autoregressive_state_prev_suffix="_prev",
        autoregressive_eval_horizon_k=10,
        inference_batch_size=2,
        alpha_masking=False,
        material_params={"T_s": 0.2, "T_l": 0.8},
        ref_scales={"T": 1.0},
    )


def _case(n_time: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (1, n_time, 1, 2, 2, 2)
    return {
        "x": rng.normal(size=shape).astype(np.float32),
        "T": rng.normal(size=shape).astype(np.float32),
        "alpha": rng.random(shape).astype(np.float32),
        "fl": rng.random(shape).astype(np.float32),
    }


def test_lazy_rollout_matches_frame_weighted_eager(tmp_path: Path):
    module = _load_postprocessing_module()
    cases = {0: _case(5, seed=1), 1: _case(4, seed=2)}
    lengths = {0: 5, 1: 4}

    def frame_reader(*, case_idx: int, time_idx: int):
        # Mimics _read_processed_frame: (1, C, ...) arrays per key.
        return {k: v[:, time_idx] for k, v in cases[case_idx].items()}

    lazy_dir = tmp_path / "lazy"
    metrics = module.run_lazy_autoregressive_rollout_and_metrics(
        model=_Model(),
        frame_reader=frame_reader,
        case_indices=[0, 1],
        case_lengths=lengths,
        run_cfg=_run_cfg(),
        output_dir=lazy_dir,
    )

    # Reference: eager per case, frame-weighted mean of abs_err_mean.
    ref = {}
    for idx, var in cases.items():
        collected = {}
        module.run_autoregressive_rollout_and_metrics(
            model=_Model(), trajectory_var=var, run_cfg=_run_cfg(),
            output_dir=tmp_path / f"eager_{idx}", write_files=False, collect=collected)
        ref[idx] = collected
    for key in ("T", "alpha"):
        w0, w1 = ref[0]["n_frames"], ref[1]["n_frames"]
        expect_mae = (
            w0 * ref[0]["errors_full"][key]["abs_err_mean"]
            + w1 * ref[1]["errors_full"][key]["abs_err_mean"]
        ) / (w0 + w1)
        assert metrics.errors[key]["abs_err_mean"] == pytest.approx(expect_mae, rel=1e-6)
        expect_rmse = np.sqrt(
            (w0 * ref[0]["errors_full"][key]["rmse"] ** 2
             + w1 * ref[1]["errors_full"][key]["rmse"] ** 2) / (w0 + w1))
        assert metrics.errors[key]["rmse"] == pytest.approx(expect_rmse, rel=1e-6)

    # Artifacts: aggregates + per-case rows + first-case debug dumps.
    assert (lazy_dir / "model_errors_rollout_full.txt").is_file()
    assert (lazy_dir / "model_errors_rollout_k.txt").is_file()
    assert (lazy_dir / "rollout_per_case.csv").is_file()
    assert (lazy_dir / "rollout_predictions.npz").is_file()
    csv_text = (lazy_dir / "rollout_per_case.csv").read_text(encoding="utf-8")
    assert csv_text.count("\nfull,") == 0  # scope is a column, sanity on layout
    assert len(csv_text.strip().splitlines()) > 4  # header + rows for 2 cases x scopes x keys


def test_lazy_rollout_write_files_off_writes_nothing(tmp_path: Path):
    module = _load_postprocessing_module()
    var = _case(4, seed=3)
    out = tmp_path / "nothing"
    module.run_autoregressive_rollout_and_metrics(
        model=_Model(), trajectory_var=var, run_cfg=_run_cfg(),
        output_dir=out, write_files=False)
    assert not out.exists()
