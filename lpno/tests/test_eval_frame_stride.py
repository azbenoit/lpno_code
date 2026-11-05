"""PINO_EVAL_FRAME_STRIDE scores every Nth frame, and says so in the metrics.

An eval deliberately ignores custom.val_max_cases so a scored split is never a
subset (_effective_val_max_cases returns 0 when eval_only). That makes the full
5-11k-frame split the only thing an eval can score, which is hours for a peak
statistic like tail_deficit_K that converges far earlier. The stride is the
opt-in escape hatch -- and because it produces a number that LOOKS like a split
score, the stamp asserted here is the feature, not decoration.
"""

from types import SimpleNamespace

import numpy as np
import torch

from lpno.models.pipeline.postprocessing import run_lazy_dataset_metrics


class _ZeroT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, invar):
        return {"T": torch.zeros_like(invar["x"])}


class _DS(torch.utils.data.Dataset):
    def __init__(self, n):
        z1d = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
        self.z = np.broadcast_to(z1d, (1, 2, 2, z1d.size)).astype(np.float32)
        rng = np.random.default_rng(0)
        self.truths = [rng.standard_normal((1, 2, 2, 4)).astype(np.float32) for _ in range(n)]

    def __len__(self):
        return len(self.truths)

    def __getitem__(self, i):
        invar = {"x": torch.from_numpy(np.zeros_like(self.z)), "z": torch.from_numpy(self.z.copy())}
        return invar, {"T": torch.from_numpy(self.truths[i])}, 0


def _run_cfg():
    return SimpleNamespace(
        inference_batch_size=1,
        material_params={"T_s": 1.0, "T_l": 2.0},
        ref_scales={"T": 1.0},
        alpha_masking=False,
    )


def test_no_stride_scores_every_frame_and_adds_no_stamp(tmp_path, monkeypatch):
    monkeypatch.delenv("PINO_EVAL_FRAME_STRIDE", raising=False)
    m = run_lazy_dataset_metrics(_ZeroT(), _DS(6), _run_cfg(), tmp_path, ["T"])
    # No stamp means "this is a full-split score" -- downstream reads it that way.
    assert "eval_frame_stride" not in m.errors


def test_stride_scores_subset_and_stamps_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_EVAL_FRAME_STRIDE", "3")
    m = run_lazy_dataset_metrics(_ZeroT(), _DS(6), _run_cfg(), tmp_path, ["T"])
    stamp = m.errors["eval_frame_stride"]
    assert stamp["stride"] == 3.0
    assert stamp["frames_scored"] == 2.0  # frames 0 and 3 of 6
    assert "T" in m.errors  # the real metrics still come through


def test_stride_of_one_is_the_full_split(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_EVAL_FRAME_STRIDE", "1")
    m = run_lazy_dataset_metrics(_ZeroT(), _DS(4), _run_cfg(), tmp_path, ["T"])
    assert "eval_frame_stride" not in m.errors
