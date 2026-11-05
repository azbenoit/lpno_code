"""The best-checkpoint keeper must reach validators that OVERRIDE save_results.

The keeper patched only GridValidator/PointwiseValidator, but solve_domain
builds StreamingGridValidator (A35), whose own save_results shadows the
patched base method. Result: neither rigno2579 fp32 arm wrote
u_network.best.pth -- 32k steps, 130 validations, zero snapshots, and the
gaot_cond_32k_ctrl divergence protection silently not in force (2026-08-10).
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("physicsnemo.sym.domain.validator")

from physicsnemo.sym.domain.validator import GridValidator  # noqa: E402

from lpno.models.pipeline.best_checkpoint import (  # noqa: E402
    BEST_META_NAME,
    BEST_WEIGHTS_NAME,
    best_val_checkpoint,
)


class _OverridingValidator(GridValidator):
    """Same shape as StreamingGridValidator: save_results in its OWN __dict__."""

    def __init__(self, losses_per_call):
        # deliberately no super().__init__: instances in the real failure were
        # built long before the keeper entered; only the class dict matters.
        self._losses = list(losses_per_call)
        self._call = 0

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        losses = self._losses[min(self._call, len(self._losses) - 1)]
        self._call += 1
        return losses


def test_keeper_snapshots_through_a_save_results_override(tmp_path):
    model = torch.nn.Linear(2, 1)
    validator = _OverridingValidator([
        {"mae_T": 5.0},   # first validation: becomes best
        {"mae_T": 9.0},   # worse: must NOT overwrite
        {"mae_T": 3.0},   # better: must overwrite
    ])

    with best_val_checkpoint(cfg=None, model=model, network_dir=tmp_path):
        for step in (100, 200, 300):
            validator.save_results("val", str(tmp_path), None, "", step)

    weights = tmp_path / BEST_WEIGHTS_NAME
    meta = json.loads((tmp_path / BEST_META_NAME).read_text())
    assert weights.is_file(), "the override shadowed the patch before the fix"
    assert meta["metric"] == "mae_T" and meta["value"] == 3.0 and meta["step"] == 300

    # the patch is temporary: outside the context the class is restored
    assert not getattr(_OverridingValidator.save_results, "_best_checkpoint_patched", False)


def test_keeper_never_counts_a_diverged_validation(tmp_path):
    model = torch.nn.Linear(2, 1)
    validator = _OverridingValidator([
        {"mae_T": 4.0},
        {"mae_T": float("nan")},
        {"mae_T": float("inf")},
    ])
    with best_val_checkpoint(cfg=None, model=model, network_dir=tmp_path):
        for step in (1, 2, 3):
            validator.save_results("val", str(tmp_path), None, "", step)
    meta = json.loads((tmp_path / BEST_META_NAME).read_text())
    assert meta["value"] == 4.0 and meta["step"] == 1
