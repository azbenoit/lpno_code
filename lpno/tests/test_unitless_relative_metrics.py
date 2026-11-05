"""Regression: per-sample relative error divides by that sample's own field mean.

For fl/alpha the denominator is the sample's spatial mean, so a low-melt sample
with mean ~1e-3 (or 0) blows its relative error up to 1e3-1e9. Those values were
written to the val-error CSV and averaged across folds, making the relative-error
column meaningless for exactly the samples a melt model is judged on.

fl and alpha are already dimensionless fractions in [0, 1], so an absolute error
IS the relative one -- there is nothing for a relative metric to normalize away.
The fix is to stop reporting relative metrics for them at all rather than pick a
different denominator. IoU and absolute/RMSE metrics are unaffected.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lpno.models.pipeline.postprocessing import (
    _UNITLESS_KEYS,
    _compute_error_stats,
    region_error_stats,
    write_val_error_csv,
)

_REL = ("rel_err_mean", "rel_err_std", "rel_rmse")


def _pair(value_true: float, value_pred: float):
    return (
        np.full((2, 1, 2, 2, 2), value_pred, dtype=np.float32),
        np.full((2, 1, 2, 2, 2), value_true, dtype=np.float32),
    )


def test_fl_and_alpha_are_the_unitless_keys():
    assert set(_UNITLESS_KEYS) == {"fl", "alpha"}


def test_no_relative_stats_for_unitless_keys():
    pred, true = _pair(1e-3, 2e-3)  # a near-zero-mean, low-melt sample
    for key in ("fl", "alpha"):
        stats = _compute_error_stats({key: pred}, {key: true}, metric_keys=[key])[key]
        assert not any(name in stats for name in _REL), f"{key}: {sorted(stats)}"
        # The metrics that do mean something for a fraction stay.
        assert stats["abs_err_mean"] > 0
        assert "iou_mean" in stats


def test_dimensional_keys_keep_their_relative_stats():
    pred, true = _pair(300.0, 310.0)
    stats = _compute_error_stats({"T": pred}, {"T": true}, metric_keys=["T"])["T"]
    assert all(name in stats for name in _REL)
    assert abs(stats["rel_err_mean"] - 10.0 / 300.0) < 1e-6


def test_region_stats_drop_the_relative_column_when_asked():
    pred, true = _pair(1e-3, 2e-3)
    z = np.linspace(-1.0, 1.0, 2, dtype=np.float32)[None, None, None, None, :]
    z = np.broadcast_to(z, pred.shape)
    rel_on = region_error_stats(pred, true, z)
    rel_off = region_error_stats(pred, true, z, relative=False)
    assert "substrate_rel_err_mean" in rel_on
    assert "substrate_rel_err_mean" not in rel_off
    assert "substrate_abs_err_mean" in rel_off


def test_val_csv_leaves_the_relative_cell_blank(tmp_path: Path):
    rows = [
        {"fold": 0, "global_index": 0, "local_val_index": 0, "P": 1.0, "V_scan": 1.0,
         "output_key": "fl", "abs_err_mean": 0.001, "rel_err_mean": "", "iou": 0.9},
        {"fold": 0, "global_index": 0, "local_val_index": 0, "P": 1.0, "V_scan": 1.0,
         "output_key": "T", "abs_err_mean": 0.01, "rel_err_mean": 0.02, "iou": ""},
    ]
    out = tmp_path / "val_errors_by_sample.csv"
    write_val_error_csv(rows, out)
    with out.open(encoding="utf-8") as f:
        parsed = {row["output_key"]: row for row in csv.DictReader(f)}
    assert parsed["fl"]["rel_err_mean"] == ""
    assert float(parsed["T"]["rel_err_mean"]) == 0.02
