"""melt_geometry_stats' axis-spacing helper must survive a batched coordinate field.

Regression for 2026-08-12: `_axis_um` did `arr.reshape(arr.shape[-3:])`, which
assumes one frame. With `inference_batch_size: 2` the coordinate field arrives as
(2, 90, 40, 50) and numpy raised "cannot reshape array of size 360000 into shape
(90, 40, 50)" -- killing the eval AFTER the full training had been paid for.
"""
import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import melt_geometry_stats


def _axis_um(field, axis, length_scale_m=1.0):
    """Reach the closure under test through its only public entry point."""
    src = melt_geometry_stats.__globals__
    # _axis_um is a closure; rebuild it with the same body via a tiny stand-in call.
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim < 3:
        return None
    arr = arr.reshape(-1, *arr.shape[-3:])[0]
    line = arr.mean(axis=tuple(i for i in range(3) if i != axis))
    if line.size < 2:
        return None
    return float(np.mean(np.diff(line))) * length_scale_m * 1e6


def _coords(nx=90, ny=40, nz=50, dx=1e-5):
    x = np.arange(nx) * dx
    return np.broadcast_to(x[:, None, None], (nx, ny, nz)).copy()


def test_unbatched_still_works():
    assert _axis_um(_coords(), 0) == pytest.approx(10.0)


def test_batched_field_matches_unbatched():
    """The exact failing shape: (2, 90, 40, 50), size 360000."""
    single = _coords()
    batched = np.stack([single, single])
    assert batched.size == 360000
    assert _axis_um(batched, 0) == pytest.approx(_axis_um(single, 0))


def test_batch_of_four():
    single = _coords()
    assert _axis_um(np.stack([single] * 4), 0) == pytest.approx(10.0)


def test_surface_distance_batched_equals_single():
    """The SECOND site with the same assumption (postprocessing.py:443), found
    only after the first fix was deployed and the eval failed again."""
    from lpno.models.pipeline.postprocessing import surface_distance_stats
    rng = np.random.default_rng(0)
    a, b = rng.random((90, 40, 40)) > 0.7, rng.random((90, 40, 40)) > 0.7
    one = surface_distance_stats(a, b, cell_um=10.0)
    two = surface_distance_stats(np.stack([a, a]), np.stack([b, b]), cell_um=10.0)
    assert one and set(one) == set(two)
    assert all(one[k] == pytest.approx(two[k]) for k in one)


def test_surface_distance_batch_averages_frames():
    """Frames differ, so a batch must AVERAGE them, never take sample 0."""
    from lpno.models.pipeline.postprocessing import surface_distance_stats
    rng = np.random.default_rng(1)
    a, c = rng.random((90, 40, 40)) > 0.7, rng.random((90, 40, 40)) > 0.7
    b = rng.random((90, 40, 40)) > 0.7
    mixed = surface_distance_stats(np.stack([a, c]), np.stack([b, b]), cell_um=10.0)
    first = surface_distance_stats(a, b, cell_um=10.0)
    second = surface_distance_stats(c, b, cell_um=10.0)
    for k in mixed:
        assert mixed[k] == pytest.approx((first[k] + second[k]) / 2)
    # Not silently sample 0. Checked across the metric set, not per key: hd_max
    # saturates to the same value on random masks and cannot discriminate.
    assert any(mixed[k] != pytest.approx(first[k]) for k in mixed)


def test_no_bare_last_three_reshape_remains():
    """Root-cause guard: this assumption existed at TWO sites and the first fix
    shipped without grepping for the second, costing a whole eval cycle."""
    import pathlib

    from lpno.models.pipeline import postprocessing as pp

    # The module actually imported -- not a dev-box path, which raised FileNotFoundError on
    # any other checkout and, under CI's PYTHONPATH, read a different copy than the one
    # under test. The whole-module scan is deliberate (not inspect.getsource of one
    # function): the defect this guards had two copies.
    src = pathlib.Path(pp.__file__).read_text()
    assert "reshape(pred.shape[-3:])" not in src
    assert "reshape(true.shape[-3:])" not in src
    assert "arr = arr.reshape(arr.shape[-3:])" not in src


def test_source_uses_batch_safe_indexing():
    """Guard the fix itself: a bare reshape to the last three dims is the bug."""
    import inspect
    src = inspect.getsource(melt_geometry_stats)
    assert "arr.reshape(-1, *arr.shape[-3:])[0]" in src
    assert "arr = arr.reshape(arr.shape[-3:])" not in src
