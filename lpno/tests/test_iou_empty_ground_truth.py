"""IoU over sparse targets must not average empty-ground-truth frames.

With `time_as_batch=true` each batch entry is one time frame, and early frames have
no melt at all. `iou_per_batch` returns 1.0 for both-empty (correct for a single
comparison) but exactly 0.0 the moment the prediction has ANY cell above threshold.
Averaging those frames yields bimodal noise, not accuracy.

Measured 2026-07-30 on a real frame from eval 22352 (`gaot_cond150_fullres_20k`):
GT `fl` identically zero (range [0.0, 0.0]), prediction 4 cells > 0.5 out of 600,000
-> IoU 0.0, the worst possible score, for 0.0007% false positives against no melt.
Aggregated to `fl.iou_mean` 0.539 +/- 0.413, which was misread as an architecture
weakness. It is a melt-volume regime effect, not a model defect.
"""
import numpy as np

from lpno.custom_utils.utils import (
    iou_per_batch,
    iou_per_batch_with_support,
)
from lpno.models.pipeline.postprocessing import _iou_stats


def _frames(n_empty: int, n_real: int, shape=(6, 5, 4)):
    """n_empty frames with zero GT, n_real frames with a solid GT block."""
    pred = np.zeros((n_empty + n_real, 1, *shape), dtype=np.float32)
    true = np.zeros_like(pred)
    for i in range(n_empty):
        pred[i, 0, 0, 0, 0] = 1.0          # a single stray cell, GT stays empty
    for j in range(n_empty, n_empty + n_real):
        true[j, 0, :3] = 1.0
        pred[j, 0, :3] = 1.0               # perfect on the frames that have melt
    return pred, true


def test_one_stray_cell_zeroes_an_empty_gt_frame():
    """The mechanism, reproduced exactly."""
    pred, true = _frames(n_empty=1, n_real=0)
    assert true.max() == 0.0
    assert (pred > 0.5).sum() == 1
    assert iou_per_batch(pred, true)[0] == 0.0


def test_legacy_mean_is_dragged_down_by_empty_frames():
    pred, true = _frames(n_empty=3, n_real=1)
    legacy = float(iou_per_batch(pred, true).mean())
    assert legacy < 0.3, legacy          # 3 zeros + 1 perfect -> 0.25
    stats = _iou_stats(pred, true)
    # The frames that actually contain melt were predicted perfectly.
    assert stats["iou_mean"] == 1.0
    assert stats["iou_support_frames"] == 1
    assert stats["iou_empty_gt_frames"] == 3
    assert stats["iou_empty_gt_false_positive_frames"] == 3
    # legacy value preserved, so older numbers stay comparable
    assert abs(stats["iou_mean_all_frames"] - legacy) < 1e-9


def test_support_mask_identifies_frames_with_melt():
    pred, true = _frames(n_empty=2, n_real=3)
    _, support = iou_per_batch_with_support(pred, true)
    assert support.tolist() == [False, False, True, True, True]


def test_all_empty_gt_reports_nan_not_a_fake_score():
    """If nothing in the batch has melt there is no shape to score; say so rather
    than emit a number that reads like accuracy."""
    pred, true = _frames(n_empty=4, n_real=0)
    stats = _iou_stats(pred, true)
    assert np.isnan(stats["iou_mean"])
    assert stats["iou_support_frames"] == 0
    assert stats["iou_empty_gt_false_positive_frames"] == 4


def test_dense_target_is_unaffected():
    """alpha covers ~75% of the domain, so its IoU must be unchanged by this fix."""
    rng = np.random.default_rng(0)
    true = (rng.random((5, 1, 6, 5, 4)) > 0.25).astype(np.float32)
    pred = true.copy()
    stats = _iou_stats(pred, true)
    assert stats["iou_mean"] == 1.0
    assert stats["iou_empty_gt_frames"] == 0
    assert stats["iou_mean_all_frames"] == 1.0
