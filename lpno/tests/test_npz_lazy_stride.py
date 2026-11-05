"""Spatial striding in the lazy per-case NPZ store.

RIGNO's cost scales with the number of grid points, and the full 150x100x40
FLOW-3D grid does not fit the 24 GB card (2026-07-14). Striding at
read time gives the validated 75x50x40 training grid without writing a second
copy of the dataset: an aiida-pino training profile sets ``PINO_LAZY_STRIDE``
and the store slices every case as it decompresses it.

The contract under test: a strided read equals the plain numpy slice, and the
coordinate vectors are strided identically so the geometry still matches the
field values (RIGNO builds its graph from those coordinates).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lpno.custom_utils import npz_datasource as nds
from lpno.models.pipeline.npz_lazy_store import (
    NpzDirStore,
    maybe_build_npz_store,
)

from test_npz_lazy_store import _write_cases


def _store(tmp_path, **kw):
    return NpzDirStore(
        tmp_path,
        requested_fields=["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"],
        path_channels=list(nds.PATH_CHANNEL_NAMES),
        material_channels=list(nds.MATERIAL_CHANNEL_NAMES),
        **kw,
    )


def _run_cfg(tmp_path):
    return SimpleNamespace(
        data_directory_path=str(tmp_path),
        vtk_fields=["Temperature", "Pressure"],
        keys=SimpleNamespace(str_fno_input_keys=["V_scan", "P", "x", "y", "z"]),
        moving_ref_frame=False,
        time_as_batch=True,
        time_as_dim=False,
        include_time=True,
        lazy_npz=True,
    )


def test_default_stride_is_a_noop(tmp_path):
    """No stride argument must behave exactly as before this feature existed."""
    nx, ny, nz, _ = _write_cases(tmp_path)
    plain = _store(tmp_path)
    unit = _store(tmp_path, spatial_stride=(1, 1, 1))
    assert plain.spatial_shape() == unit.spatial_shape() == (nx, ny, nz)
    a = plain.read_case(0, time_idx=2)
    b = unit.read_case(0, time_idx=2)
    assert set(a) == set(b)
    for key in a:
        assert np.array_equal(a[key], b[key]), key


def test_spatial_shape_reflects_stride(tmp_path):
    _write_cases(tmp_path, nx=10, ny=8, nz=6)
    s = _store(tmp_path, spatial_stride=(2, 2, 1))
    assert s.spatial_shape() == (5, 4, 6)
    # array_shape() is derived from the same metadata and must agree
    assert s.array_shape("Temperature")[-3:] == (5, 4, 6)


def test_uneven_dimension_rounds_up(tmp_path):
    """A stride that does not divide the axis keeps ceil(n/s) points, like numpy."""
    _write_cases(tmp_path, nx=10, ny=8, nz=6)
    s = _store(tmp_path, spatial_stride=(3, 3, 4))
    assert s.spatial_shape() == (4, 3, 2)  # len(range(0,10,3)), (0,8,3), (0,6,4)
    assert s.read_case(0, time_idx=0)["Pressure"].shape == (1, 1, 4, 3, 2)


def test_strided_values_match_numpy_slice(tmp_path):
    """The strided read is exactly the unstrided read sliced the same way."""
    _write_cases(tmp_path, nx=10, ny=8, nz=6)
    full = _store(tmp_path).read_case(0, time_idx=3)
    strided = _store(tmp_path, spatial_stride=(2, 2, 3)).read_case(0, time_idx=3)
    for key in ("Pressure", "Fraction Of Fluid", "Velocity"):
        expected = full[key][:, :, ::2, ::2, ::3]
        assert strided[key].shape == expected.shape, key
        assert np.array_equal(strided[key], expected), key


def test_coordinates_stride_with_the_fields(tmp_path):
    """Geometry must stay consistent: coords sample the same cells as the fields."""
    _write_cases(tmp_path, nx=10, ny=8, nz=6)
    full = _store(tmp_path).read_case(0, time_idx=1)
    strided = _store(tmp_path, spatial_stride=(2, 2, 1)).read_case(0, time_idx=1)
    for key in ("x", "y", "z"):
        assert np.array_equal(strided[key], full[key][:, :, ::2, ::2, ::1]), key
    # x varies along axis 2 only, so its strided distinct values are x[::2]
    assert np.array_equal(
        np.unique(strided["x"]), np.unique(full["x"][:, :, ::2, :, :])
    )


def test_path_channels_built_on_the_strided_grid(tmp_path):
    """Dose/beam-distance channels are spatial fields and must match the stride."""
    _write_cases(tmp_path, nx=10, ny=8, nz=6)
    s = _store(tmp_path, spatial_stride=(2, 2, 1))
    raw = s.read_case(0, time_idx=4)
    for key in nds.PATH_CHANNEL_NAMES:
        assert raw[key].shape == (1, 1, 5, 4, 6), key
    assert np.all(np.isfinite(raw["beam_dist"]))


def test_full_case_read_strides_every_frame(tmp_path):
    """time_idx=None returns the whole trajectory, strided on the spatial axes."""
    nx, ny, nz, nt = _write_cases(tmp_path, nx=10, ny=8, nz=6)
    raw = _store(tmp_path, spatial_stride=(2, 2, 1)).read_case(0)
    assert raw["Temperature"].shape == (1, nt, 1, 5, 4, 6)
    assert raw["Velocity"].shape == (1, nt, 3, 5, 4, 6)


def test_stride_of_one_per_axis_is_allowed_and_partial(tmp_path):
    """Striding only x is legitimate (the z axis is already thin in these grids)."""
    _write_cases(tmp_path, nx=10, ny=8, nz=6)
    s = _store(tmp_path, spatial_stride=(2, 1, 1))
    assert s.spatial_shape() == (5, 8, 6)


@pytest.mark.parametrize("bad", [(0, 1, 1), (-2, 1, 1), (1, 1)])
def test_invalid_stride_is_rejected(tmp_path, bad):
    _write_cases(tmp_path)
    with pytest.raises(ValueError):
        _store(tmp_path, spatial_stride=bad)


def test_env_var_sets_the_stride(tmp_path, monkeypatch):
    """aiida-pino passes the stride as an env var in the training profile."""
    _write_cases(tmp_path, nx=10, ny=8, nz=6)
    monkeypatch.setenv("PINO_LAZY_STRIDE", "2,2,1")
    store = maybe_build_npz_store(tmp_path, _run_cfg(tmp_path))
    assert store is not None
    assert store.spatial_shape() == (5, 4, 6)


def test_absent_env_var_keeps_the_full_grid(tmp_path, monkeypatch):
    """The feature is opt-in: unset means train on the full grid, as before."""
    nx, ny, nz, _ = _write_cases(tmp_path, nx=10, ny=8, nz=6)
    monkeypatch.delenv("PINO_LAZY_STRIDE", raising=False)
    store = maybe_build_npz_store(tmp_path, _run_cfg(tmp_path))
    assert store is not None
    assert store.spatial_shape() == (nx, ny, nz)


def test_malformed_env_var_fails_loudly(tmp_path, monkeypatch):
    """A typo must not silently train on the wrong grid."""
    _write_cases(tmp_path)
    monkeypatch.setenv("PINO_LAZY_STRIDE", "2,2")
    with pytest.raises(ValueError):
        maybe_build_npz_store(tmp_path, _run_cfg(tmp_path))
