"""RMSE >> MAE must be attributable, not just visible.

`gaot_mat_steady_20k` reported T MAE 17.75 K against T RMSE 6279 K -- a ratio of 354 --
while its own in-training validation on 5 cases read 47.7 K. Nothing in the eval
recorded a prediction range, so there was no way to tell a shape error from a numerical
blow-up, or to find the offending case, without re-running inference by hand.
"""
import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import _track_prediction_extremes


def test_records_the_outlier_location_and_its_alpha():
    pred = np.zeros((2, 1, 3, 3), dtype=np.float32)
    pred[1, 0, 2, 1] = 740.0                    # the blow-up
    true = np.zeros_like(pred)
    alpha = np.full_like(pred, 0.9)              # inside the metal region
    acc: dict = {}
    _track_prediction_extremes(acc, pred, true, batch_idx=7, aux={"alpha": alpha})
    assert acc["pred_max"] == 740.0
    assert acc["pred_max_batch"] == 7
    assert acc["pred_max_loc"] == [1, 0, 2, 1]
    assert acc["pred_max_truth"] == 0.0
    # alpha > 0.5 means the masking fill passed this value through untouched.
    assert acc["pred_max_alpha"] == pytest.approx(0.9)


def test_running_max_keeps_the_worst_across_batches():
    acc: dict = {}
    for i, v in enumerate((3.0, 91.0, 12.0)):
        pred = np.full((1, 1, 2, 2), v, dtype=np.float32)
        _track_prediction_extremes(acc, pred, np.zeros_like(pred), batch_idx=i)
    assert acc["pred_max"] == 91.0 and acc["pred_max_batch"] == 1


def test_min_is_tracked_too():
    acc: dict = {}
    pred = np.array([[[[-4.0, 2.0]]]], dtype=np.float32)
    _track_prediction_extremes(acc, pred, np.zeros_like(pred), batch_idx=0)
    assert acc["pred_min"] == -4.0


def test_empty_batch_is_a_no_op():
    acc: dict = {}
    _track_prediction_extremes(acc, np.zeros((0, 1, 2, 2)), np.zeros((0, 1, 2, 2)), batch_idx=0)
    assert acc == {}


def test_mismatched_aux_is_ignored_not_crashing():
    """z can be absent or a different shape; the tracker must not fail on it."""
    acc: dict = {}
    pred = np.ones((1, 1, 2, 2), dtype=np.float32)
    _track_prediction_extremes(acc, pred, np.zeros_like(pred), batch_idx=0,
                               aux={"z": np.ones((5,)), "alpha": None})
    assert acc["pred_max"] == 1.0 and "pred_max_z" not in acc
