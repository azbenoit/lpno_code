"""run_lazy_dataset_metrics must report substrate/surface z-region stats that
match the batched region_error_stats -- computed by streaming (raw sums per
batch), so the memory-safe eval carries the same melt-region breakdown the eager
path would, without holding whole fields.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lpno.models.pipeline.postprocessing import (
    region_error_stats,
    run_lazy_dataset_metrics,
)


class _ZeroT(torch.nn.Module):
    """Predicts T=0 everywhere, so error == -truth (regions are easy to check)."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))  # gives the module a device

    def forward(self, invar):
        return {"T": torch.zeros_like(invar["x"])}


class _DS(torch.utils.data.Dataset):
    def __init__(self, z1d, truths):
        # grid (C=1, nx=2, ny=2, nz); z broadcast along the last (z) axis.
        self.z = np.broadcast_to(z1d, (1, 2, 2, z1d.size)).astype(np.float32)
        self.truths = truths

    def __len__(self):
        return len(self.truths)

    def __getitem__(self, i):
        invar = {"x": torch.from_numpy(np.zeros_like(self.z)), "z": torch.from_numpy(self.z.copy())}
        return invar, {"T": torch.from_numpy(self.truths[i].astype(np.float32))}, 0


def test_lazy_region_metrics_match_batched(tmp_path):
    z1d = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)  # substrate=idx0-2, surface=idx2 (z=0)
    rng = np.random.default_rng(0)
    truths = [rng.standard_normal((1, 2, 2, 4)).astype(np.float32) for _ in range(3)]
    ds = _DS(z1d, truths)

    run_cfg = SimpleNamespace(
        inference_batch_size=1,
        material_params={"T_s": 1.0, "T_l": 2.0},
        ref_scales={"T": 1.0},
        alpha_masking=False,
    )
    m = run_lazy_dataset_metrics(_ZeroT(), ds, run_cfg, tmp_path, ["T"])
    T = m.errors["T"]

    # Reference: batched region_error_stats over all samples stacked (pred=0).
    true_all = np.concatenate(truths, axis=0)[None]  # (1, N, 2, 2, 4)
    z_all = np.broadcast_to(z1d, true_all.shape)
    ref = region_error_stats(np.zeros_like(true_all), true_all, z_all)

    for tag in ("substrate", "surface"):
        assert f"{tag}_abs_err_mean" in T, f"missing {tag} region metric"
        assert abs(T[f"{tag}_abs_err_mean"] - ref[f"{tag}_abs_err_mean"]) < 1e-5
        assert abs(T[f"{tag}_rmse"] - ref[f"{tag}_rmse"]) < 1e-5
    # substrate (3 of 4 z-slices) differs from whole-field volume => real split.
    assert abs(T["substrate_abs_err_mean"] - T["abs_err_mean"]) > 1e-6


def test_streaming_std_matches_the_materialized_std(tmp_path):
    """Regression: streaming eval wrote abs_err_std/rel_err_std/iou_std as literal 0.0.

    The accumulators only tracked sums, so a reader comparing std across eval
    modes saw false zero variance for every streaming run while the materialized
    path reported a real number. sq_err_sum is already the sum of squared
    absolute errors, so the population std needs no new accumulator.
    """
    z1d = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
    rng = np.random.default_rng(7)
    truths = [rng.standard_normal((1, 2, 2, 4)).astype(np.float32) for _ in range(3)]
    ds = _DS(z1d, truths)

    run_cfg = SimpleNamespace(
        inference_batch_size=1,
        material_params={"T_s": 1.0, "T_l": 2.0},
        ref_scales={"T": 1.0},
        alpha_masking=False,
    )
    T = run_lazy_dataset_metrics(_ZeroT(), ds, run_cfg, tmp_path, ["T"]).errors["T"]

    # The model predicts 0, so err == -truth over every voxel of every sample.
    err = np.concatenate(truths, axis=0).ravel()
    abs_err = np.abs(err)
    denom = float(np.mean(np.concatenate(truths, axis=0))) + 1e-12

    assert T["abs_err_std"] == pytest.approx(float(abs_err.std()), rel=1e-5, abs=1e-6)
    assert T["rel_err_std"] == pytest.approx(float(abs_err.std() / denom), rel=1e-5, abs=1e-6)
    assert T["abs_err_std"] > 0.0, "a random error field must have nonzero spread"
