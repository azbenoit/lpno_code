"""Surface-distance metrics for the melt and free-surface interfaces.

Motivation (2026-08-04). IoU and volume error are volume-dependent: they say how
much melt was missed, not how far the predicted interface sits from the true one.
On a thin shell around a keyhole a one-cell displacement can halve IoU while the
surface is essentially right, and conversely a model can hold a good volume while
placing the boundary badly. It was asked for twice before being built
("Moving RMSE needs a finer mesh or an interface-aware loss/representation",
2026-07-28).

`surface_distance_stats` reports the symmetric mean surface distance and the 95th
percentile Hausdorff distance between two binary masks, in micrometres.

The degenerate case matters here and is reported explicitly rather than being
allowed to become a silent NaN: model 29125 predicts NO melt at all, so its melt
surface is empty and no finite distance exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import surface_distance_stats

CELL_UM = 5.0


def _slab(x0, x1):
    """A (1,1,20,8,8) block that is True for x in [x0, x1)."""
    m = np.zeros((1, 1, 20, 8, 8), dtype=bool)
    m[0, 0, x0:x1] = True
    return m


def test_identical_masks_have_zero_distance():
    m = _slab(5, 12)
    s = surface_distance_stats(m, m, cell_um=CELL_UM)
    assert s["surface_msd_um"] == pytest.approx(0.0)
    assert s["surface_hd95_um"] == pytest.approx(0.0)
    assert s["surface_pred_empty"] is False


def test_one_cell_shift_costs_about_one_cell():
    """The property IoU lacks: a small displacement reads as a small distance."""
    s = surface_distance_stats(_slab(6, 13), _slab(5, 12), cell_um=CELL_UM)
    assert 0 < s["surface_msd_um"] <= 2 * CELL_UM


def test_distance_grows_with_displacement():
    near = surface_distance_stats(_slab(6, 13), _slab(5, 12), cell_um=CELL_UM)
    far = surface_distance_stats(_slab(12, 19), _slab(5, 12), cell_um=CELL_UM)
    assert far["surface_msd_um"] > near["surface_msd_um"]


def test_less_volume_dependent_than_iou():
    """The actual claim: for one fixed displacement, IoU depends hard on volume.

    Both pairs below are displaced by exactly one cell. IoU swings from 0.33 to
    0.85 purely because the enclosed volume changed, which is why a thin melt
    shell scores badly on IoU even when its interface is nearly right. Surface
    distance stays bounded by the physical displacement in both cases. It is NOT
    invariant to volume -- on a thick block much of the surface nearly coincides
    after the shift -- but it is interpretable in micrometres, which IoU is not.
    """

    def iou(a, b):
        return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()

    thin_p, thin_t = _slab(6, 8), _slab(5, 7)
    thick_p, thick_t = _slab(6, 18), _slab(5, 17)
    iou_thin, iou_thick = iou(thin_p, thin_t), iou(thick_p, thick_t)
    msd_thin = surface_distance_stats(thin_p, thin_t, cell_um=CELL_UM)["surface_msd_um"]
    msd_thick = surface_distance_stats(thick_p, thick_t, cell_um=CELL_UM)["surface_msd_um"]

    assert iou_thick > 2 * iou_thin, "IoU should swing hard with volume here"
    # Both surfaces are one cell apart, so both distances stay within one cell.
    assert msd_thin <= CELL_UM and msd_thick <= CELL_UM
    # And the distances are far closer to each other than the IoUs are.
    assert (max(msd_thin, msd_thick) / min(msd_thin, msd_thick)) < (iou_thick / iou_thin)


def test_empty_prediction_is_flagged_not_silently_nan():
    """Model 29125 predicts zero melt: report the failure, do not hide it."""
    empty = np.zeros((1, 1, 20, 8, 8), dtype=bool)
    s = surface_distance_stats(empty, _slab(5, 12), cell_um=CELL_UM)
    assert s["surface_pred_empty"] is True
    assert np.isfinite(s["surface_msd_um"]), "must report a finite sentinel, not NaN"
    assert s["surface_msd_um"] > 0


def test_empty_truth_and_empty_prediction_agree():
    empty = np.zeros((1, 1, 20, 8, 8), dtype=bool)
    s = surface_distance_stats(empty, empty, cell_um=CELL_UM)
    assert s["surface_msd_um"] == pytest.approx(0.0)
    assert s["surface_pred_empty"] is True


def test_hd95_is_below_the_worst_case():
    """The 95th percentile must ignore a single far outlier voxel."""
    pred = _slab(5, 12).copy()
    pred[0, 0, 19, 0, 0] = True  # one stray voxel far away
    s = surface_distance_stats(pred, _slab(5, 12), cell_um=CELL_UM)
    assert s["surface_hd95_um"] < s["surface_hd_max_um"]


def test_anisotropic_spacing_is_honoured():
    """A [2,2,1] stride leaves dz=10 um but dx=dy=20 um.

    Using one z-derived cell size under-reports x/y distances by 2x, which is
    what every strided RIGNO eval did before 2026-08-04.
    """
    iso = surface_distance_stats(_slab(6, 13), _slab(5, 12), cell_um=10.0)
    aniso = surface_distance_stats(_slab(6, 13), _slab(5, 12), cell_um=(20.0, 20.0, 10.0))
    # the displacement is along x, so doubling dx must roughly double the distance
    assert aniso["surface_msd_um"] > 1.5 * iso["surface_msd_um"]
