"""region_error_stats: substrate-only (z<=0) and z=0-surface-plane error metrics.

The random-path grids carry a gas/void layer above z=0 whose cells are masked
to ~zero error in both prediction and ground truth; including them deflates
volume-averaged metrics. Substrate stats exclude that region; surface stats
report the slice nearest z=0 from below.
"""

import numpy as np

from lpno.models.pipeline.postprocessing import region_error_stats


def _grid(z_centers, b=2, c=1, nx=3, ny=3):
    z1 = np.asarray(z_centers, dtype=np.float64)
    z = np.broadcast_to(z1, (b, c, nx, ny, z1.size)).copy()
    true = np.full((b, c, nx, ny, z1.size), 2.0)
    return z, true


def test_void_error_excluded_from_substrate():
    z, true = _grid([-3.0, -2.0, -1.0, 1.0])
    pred = true.copy()
    pred[..., 3] += 5.0  # error only in the void slice above z=0
    s = region_error_stats(pred, true, z)
    assert s["substrate_abs_err_mean"] == 0.0
    assert s["substrate_rmse"] == 0.0
    assert s["substrate_rel_err_mean"] == 0.0


def test_surface_is_slice_nearest_zero_from_below():
    z, true = _grid([-3.0, -2.0, -1.0, 1.0])
    pred = true.copy()
    pred[..., 2] += 4.0  # the z=-1 slice is the surface
    s = region_error_stats(pred, true, z)
    assert s["surface_abs_err_mean"] == 4.0
    assert s["surface_rmse"] == 4.0
    assert abs(s["surface_rel_err_mean"] - 2.0) < 1e-9  # MAE 4 / mean(true) 2 (+eps guard)


def test_z_exactly_zero_counts_as_substrate_surface():
    # Straight-line grids end at z=0: top slice is the surface, nothing dropped.
    z, true = _grid([-2.0, -1.0, 0.0])
    pred = true + 1.0
    s = region_error_stats(pred, true, z)
    assert s["substrate_abs_err_mean"] == 1.0  # equals the full-volume MAE
    assert s["surface_abs_err_mean"] == 1.0


def test_all_void_returns_empty():
    z, true = _grid([1.0, 2.0])
    assert region_error_stats(true, true, z) == {}
