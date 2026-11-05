"""Regression tests: a source grid whose resolution differs from the configured
``grid_params`` must raise instead of being silently KNN-interpolated onto the
wrong size.

The failure this guards against: a case exported at a different resolution than
the rest of a sweep (or a truncated/misexported case) forms a *complete* uniform
grid, so ``_detect_uniform_grid`` used to just return ``None`` and the field was
silently resampled onto the config grid, poisoning training without a trace.

Reuses the real-torch/scipy loader from test_openfoam_datasource_audit_fixes so
the KNN interpolation fallback is exercised for real on CPU.
"""

from pathlib import Path

import numpy as np
import pytest

from test_openfoam_datasource_audit_fixes import _grid_params, _load_module


def _uniform_grid_coords(nx: int, ny: int, nz: int) -> np.ndarray:
    """(nx*ny*nz, 3) cell centres on a complete uniform grid over the unit cube."""
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    zs = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)


def _make_grid_ds(module, tmp_path: Path, grid_shape):
    (tmp_path / "caseA_1.vtk").write_text("")
    coords = _uniform_grid_coords(*grid_shape)
    temp = np.arange(coords.shape[0], dtype=np.float32)

    def _fake_read_file(filename):
        return {
            "coordinates": coords.copy(),
            "metadata": {"power": 10.0, "velocity": 0.1, "time_s": 1e-6},
            "Temperature": temp.copy(),
        }

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    ds.read_file = _fake_read_file
    return ds


def _fake_scalar(coords, values, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz, **kwargs):
    gx = np.linspace(Xmin, Xmax, nx, dtype=np.float32)
    gy = np.linspace(Ymin, Ymax, ny, dtype=np.float32)
    gz = np.linspace(Zmin, Zmax, nz, dtype=np.float32)
    return gx, gy, gz, np.full((nx, ny, nz), float(values[0]), dtype=np.float32)


def test_wrong_size_uniform_grid_raises(tmp_path: Path) -> None:
    module = _load_module()
    ds = _make_grid_ds(module, tmp_path, (6, 3, 3))  # complete grid, wrong resolution

    with pytest.raises(module.GridResolutionMismatchError) as excinfo:
        ds.parse_directory(
            grid_params=_grid_params((4, 2, 2)),
            requested_fields=["Temperature"],
            time_axis=False,
            force_recompute=True,
            flip_axes=False,
        )
    msg = str(excinfo.value)
    assert "6x3x3" in msg
    assert "4x2x2" in msg


def test_wrong_size_grid_interpolates_when_allowed(tmp_path: Path) -> None:
    module = _load_module()
    ds = _make_grid_ds(module, tmp_path, (6, 3, 3))
    # Keep the real _detect_uniform_grid (which now must return None when allowed);
    # only stub the heavy resample so the override path is unit-testable on CPU.
    ds.crop_and_resample_scalar = _fake_scalar

    var = ds.parse_directory(
        grid_params=_grid_params((4, 2, 2)),
        requested_fields=["Temperature"],
        time_axis=False,
        force_recompute=True,
        flip_axes=False,
        allow_grid_interpolation=True,
    )
    assert var["Temperature"].shape == (1, 1, 4, 2, 2)


def test_scattered_points_still_interpolate_without_raising(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "caseA_1.vtk").write_text("")
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.05, 0.95, size=(40, 3)).astype(np.float32)
    temp = rng.uniform(300.0, 400.0, size=40).astype(np.float32)

    def _fake_read_file(filename):
        return {
            "coordinates": coords.copy(),
            "metadata": {"power": 10.0, "velocity": 0.1, "time_s": 1e-6},
            "Temperature": temp.copy(),
        }

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    ds.read_file = _fake_read_file
    orig = ds.crop_and_resample_scalar

    def _cpu_scalar(*args, **kwargs):
        kwargs["device"] = "cpu"
        return orig(*args, **kwargs)

    ds.crop_and_resample_scalar = _cpu_scalar

    # Genuinely scattered data is not a complete grid, so it must interpolate as
    # before (the default) with no error.
    var = ds.parse_directory(
        grid_params=_grid_params((3, 3, 3)),
        requested_fields=["Temperature"],
        time_axis=False,
        force_recompute=True,
        flip_axes=False,
    )
    assert var["Temperature"].shape == (1, 1, 3, 3, 3)
