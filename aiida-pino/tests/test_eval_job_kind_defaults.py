"""Eval resource and duration defaults derive from the JOB KIND, never inheritance.

Regression origin (2026-08-06): an eval inherited its training profile's
host_ram_mb=49152 -- the figure sized for the training's 57 GB graph-build peak --
while itself using ~5.6 GB of VRAM, and the booking blocked the paired control training
for hours. Separately, the automatic post-training eval ran the full test split
(22 cases x ~400 frames ~ 2.9 h measured) when a 40-minute health readout was wanted.
"""
import pytest

from aiida_pino.workflows.evaluate_model import (
    EVAL_FRAMES_PER_CASE,
    EVAL_HOST_RAM_MB,
    EVAL_SEC_PER_FRAME,
    EVAL_TIME_BUDGET_S,
    auto_eval_stride,
    dataset_total_frames,
)


def test_dense_cases_stride_to_25_frames_per_case():
    """400 stored frames/case -> stride 16 -> 25 scored per case (the chosen default)."""
    stride = auto_eval_stride(400, 400 * 8)
    assert stride == 16
    assert 400 // stride == 25


def test_sparse_cases_run_full():
    """A case with <= 25 frames is never subset."""
    assert auto_eval_stride(21, 21 * 8) == 1
    assert auto_eval_stride(25, 25 * 22) == 1


def test_budget_still_caps_many_case_datasets():
    """The per-case rule alone could exceed the wall clock on a huge case count;
    the 40-min budget stays as the secondary cap."""
    # 500 cases x 400 frames: per-case stride 16 leaves 12500 frames ~ 4.1 h
    stride = auto_eval_stride(400, 400 * 500)
    scored = (400 * 500) / stride
    assert scored * EVAL_SEC_PER_FRAME <= EVAL_TIME_BUDGET_S


def test_unknown_size_never_subsets():
    """A wrong stride silently degrades a score; unknown size means full split."""
    assert auto_eval_stride(None, None) == 1
    assert auto_eval_stride(0, 0) == 1


def test_host_ram_constant_is_sane():
    """16 GB: generous for a sequential reader, nowhere near a training's 49 GB."""
    assert 8192 <= EVAL_HOST_RAM_MB <= 24576


class _FakeNode:
    def __init__(self, attrs):
        class _B:
            class attributes:
                all = attrs
        self.base = _B()


def test_dataset_total_frames_reads_durable_attributes():
    node = _FakeNode({'simulation_count': 22, 'time_steps_max': 400})
    assert dataset_total_frames(node) == 8800


def test_dataset_total_frames_unknown_is_none():
    assert dataset_total_frames(_FakeNode({})) is None


def test_full_frames_is_an_accepted_profile_key():
    from aiida_pino.utils.model_registry import validate_evaluation_profile
    out = validate_evaluation_profile({'full_frames': True})
    assert out['full_frames'] is True


def test_frames_per_case_target_is_25():
    assert EVAL_FRAMES_PER_CASE == 25
