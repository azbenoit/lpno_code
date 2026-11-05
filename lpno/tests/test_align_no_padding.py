"""Regression: a short prediction was edge-padded up to the truth shape.

``np.pad(..., mode="edge")`` replicated the boundary prediction values into the
missing band, and those invented voxels were then differenced against real truth
in abs/rmse/rel/IoU. Predictions and truth are now center-cropped to their common
overlap instead, so every compared voxel exists on both sides.
"""
from __future__ import annotations

import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import (
    _align_preds_to_truth_shapes,
    _center_crop_numpy,
    region_error_stats,
)


def test_short_prediction_crops_truth_instead_of_padding() -> None:
    truth = {"T": np.arange(6 * 4 * 4, dtype=np.float64).reshape(1, 1, 6, 4, 4)}
    preds = {"T": np.zeros((1, 1, 4, 4, 4))}

    aligned_preds, aligned_truth = _align_preds_to_truth_shapes(preds, truth, ["T"])

    assert aligned_preds["T"].shape == (1, 1, 4, 4, 4)
    assert aligned_truth["T"].shape == (1, 1, 4, 4, 4)
    # The truth band with no prediction behind it is dropped, not matched
    # against a replicated prediction value.
    np.testing.assert_array_equal(aligned_truth["T"], truth["T"][:, :, 1:5])


def test_long_prediction_still_crops_to_truth() -> None:
    truth = {"T": np.zeros((1, 1, 4, 4, 4))}
    preds = {"T": np.arange(6 * 4 * 4, dtype=np.float64).reshape(1, 1, 6, 4, 4)}

    aligned_preds, aligned_truth = _align_preds_to_truth_shapes(preds, truth, ["T"])

    assert aligned_truth["T"].shape == (1, 1, 4, 4, 4)
    np.testing.assert_array_equal(aligned_preds["T"], preds["T"][:, :, 1:5])


def test_center_crop_refuses_to_invent_voxels() -> None:
    with pytest.raises(ValueError, match="smaller than target"):
        _center_crop_numpy(np.zeros((1, 1, 3, 4, 4)), (4, 4, 4))


def test_region_stats_align_z_field_to_cropped_data() -> None:
    y_true = np.ones((1, 1, 2, 2, 4))
    y_pred = y_true + 1.0
    z_field = np.broadcast_to(
        np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]), (1, 1, 2, 2, 6)
    )

    stats = region_error_stats(y_pred, y_true, z_field)

    # z is center-cropped to [-2, -1, 0, 1]: three substrate slices, surface at 0.
    assert stats["substrate_abs_err_mean"] == pytest.approx(1.0)
    assert stats["surface_abs_err_mean"] == pytest.approx(1.0)
