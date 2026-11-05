"""Regression tests for the 2026-07 aiida-pino datasource audit fixes.

Covers, in openFoam_datasource.VTKOpenFOAMDataSource:
- unknown frame time / non-finite field values must raise, not
  silently poison the training tensors with NaN.
Unlike tests/test_openfoam_datasource_moving_ref_frame.py this loader keeps
the real torch/scipy (needed for the KNN resampling assertions) and stubs
only pyvista and physicsnemo_curator.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/lpno/custom_utils/openFoam_datasource.py"
MODULE_NAME = "tests._openfoam_datasource_audit_under_test"


def _evict_stub(name: str) -> None:
    """Drop a bare ModuleType stub another test left in sys.modules."""
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__spec__", None) is None and getattr(mod, "__file__", None) is None:
        del sys.modules[name]


def _load_module():
    for name in ("torch", "scipy", "scipy.spatial"):
        _evict_stub(name)
    import scipy.spatial  # noqa: F401  (real, for cKDTree)
    import torch  # noqa: F401  (real, for the KNN torch path)

    pyvista_mod = types.ModuleType("pyvista")
    pyvista_mod.read = lambda path: None
    sys.modules["pyvista"] = pyvista_mod

    pkg = types.ModuleType("physicsnemo_curator")
    etl_mod = types.ModuleType("physicsnemo_curator.etl")
    data_sources_mod = types.ModuleType("physicsnemo_curator.etl.data_sources")

    class _DummyDataSource:
        def __init__(self, cfg=None):
            self.cfg = cfg

    data_sources_mod.DataSource = _DummyDataSource
    etl_mod.data_sources = data_sources_mod
    pkg.etl = etl_mod
    sys.modules["physicsnemo_curator"] = pkg
    sys.modules["physicsnemo_curator.etl"] = etl_mod
    sys.modules["physicsnemo_curator.etl.data_sources"] = data_sources_mod

    sys.modules.pop(MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load openFoam_datasource module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _install_lightweight_resampling(ds):
    def _fake_scalar(coords, values, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz, **kwargs):
        gx = np.linspace(Xmin, Xmax, nx, dtype=np.float32)
        gy = np.linspace(Ymin, Ymax, ny, dtype=np.float32)
        gz = np.linspace(Zmin, Zmax, nz, dtype=np.float32)
        grid = np.full((nx, ny, nz), float(values[0]), dtype=np.float32)
        return gx, gy, gz, grid

    ds.crop_and_resample_scalar = _fake_scalar
    ds._detect_uniform_grid = lambda *args, **kwargs: None


def _grid_params(resolution=(4, 2, 2)):
    return {
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
        "z_range": [0.0, 1.0],
        "resolution": list(resolution),
    }


# --------------------------------------------------------------------------- #
# NaN time fill / non-finite field values
# --------------------------------------------------------------------------- #

def test_read_file_rejects_non_finite_field_values(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "bad_case.vtk").write_text("")

    class _FakeVTKDataset:
        def cell_centers(self):
            return types.SimpleNamespace(points=np.zeros((2, 3), dtype=np.float32))

    module.pv.read = lambda path: _FakeVTKDataset()

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    ds._collect_cell_fields_pyvista = lambda dataset: {
        "Temperature": np.array([300.0, np.nan], dtype=np.float32),
    }

    with pytest.raises(ValueError) as excinfo:
        ds.read_file("bad_case")
    msg = str(excinfo.value)
    assert "Temperature" in msg
    assert "bad_case.vtk" in msg


def test_time_axis_missing_time_token_raises_instead_of_nan(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "caseA.vtk").write_text("")

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    _install_lightweight_resampling(ds)

    def _fake_read_file(filename):
        return {
            "coordinates": np.zeros((8, 3), dtype=np.float32),
            "metadata": {"power": 10.0, "velocity": 0.1, "time_s": None},
            "Temperature": np.full((8,), 300.0, dtype=np.float32),
        }

    ds.read_file = _fake_read_file

    with pytest.raises(ValueError) as excinfo:
        ds.parse_directory(
            grid_params=_grid_params(),
            requested_fields=["Temperature"],
            time_axis=True,
            force_recompute=True,
            flip_axes=False,
        )
    msg = str(excinfo.value)
    assert "caseA" in msg
    assert "time" in msg.lower()


def test_readonly_cache_write_does_not_crash_parse(tmp_path: Path) -> None:
    """A read-only data dir (e.g. a shared dataset in CI) must not abort a parse:
    the var_*.npz cache is an optimization, so a failed write is warned and skipped."""
    module = _load_module()
    (tmp_path / "caseA_1.vtk").write_text("")

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    _install_lightweight_resampling(ds)
    ds.read_file = lambda filename: {
        "coordinates": np.zeros((8, 3), dtype=np.float32),
        "metadata": {"power": 10.0, "velocity": 0.1, "time_s": 1e-6},
        "Temperature": np.full((8,), 300.0, dtype=np.float32),
    }

    # Simulate a read-only data dir: the cache write raises like it does in CI.
    original_savez = module.np.savez_compressed

    def _deny(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    module.np.savez_compressed = _deny
    try:
        with pytest.warns(UserWarning, match="cache"):
            var = ds.parse_directory(
                grid_params=_grid_params(),
                requested_fields=["Temperature"],
                time_axis=False,
                force_recompute=True,
                flip_axes=False,
            )
    finally:
        module.np.savez_compressed = original_savez
    # Parse completed and returned the computed grid despite the failed cache write.
    assert var["Temperature"].shape == (1, 1, 4, 2, 2)


# --------------------------------------------------------------------------- #
# var_*.npz cache key must include grid geometry and flip_axes
# --------------------------------------------------------------------------- #

def _fake_read_file_static(filename):
    return {
        "coordinates": np.zeros((8, 3), dtype=np.float32),
        "metadata": {"power": 10.0, "velocity": 0.1, "time_s": 1e-6},
        "Temperature": np.full((8,), 300.0, dtype=np.float32),
    }


def test_cache_key_includes_grid_geometry_and_flip(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "caseA_1.vtk").write_text("")

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    ds.read_file = _fake_read_file_static
    _install_lightweight_resampling(ds)

    common = dict(requested_fields=["Temperature"], time_axis=False, flip_axes=False)
    var_a = ds.parse_directory(grid_params=_grid_params((4, 2, 2)), **common)
    assert var_a["Temperature"].shape == (1, 1, 4, 2, 2)

    # Different resolution, no force_recompute: a stale-cache hit would return
    # the (4,2,2) arrays; a keyed cache must recompute at (3,2,2).
    var_b = ds.parse_directory(grid_params=_grid_params((3, 2, 2)), **common)
    assert var_b["Temperature"].shape == (1, 1, 3, 2, 2)

    # Different x-range, same resolution: same shape, different geometry.
    shifted = _grid_params((3, 2, 2))
    shifted["x_range"] = [0.0, 2.0]
    var_c = ds.parse_directory(grid_params=shifted, **common)
    assert np.isclose(float(var_c["x"].max()), 2.0)

    # Different flip_axes: distinct cache again.
    ds.parse_directory(grid_params=shifted, requested_fields=["Temperature"], time_axis=False, flip_axes=True)

    caches = sorted(p.name for p in tmp_path.glob("var_*.npz"))
    assert len(caches) == 4, caches
    assert all(name.startswith("var_1_") for name in caches)

    # Same geometry again must still hit its own cache (name is stable).
    var_c2 = ds.parse_directory(grid_params=shifted, **common)
    assert var_c2["Temperature"].shape == (1, 1, 3, 2, 2)
    assert sorted(p.name for p in tmp_path.glob("var_*.npz")) == caches


# --------------------------------------------------------------------------- #
# load_all_fields silently drops fields absent from the first file
# --------------------------------------------------------------------------- #

def test_load_all_fields_warns_about_fields_only_in_later_files(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "caseA_1.vtk").write_text("")
    (tmp_path / "caseB_1.vtk").write_text("")

    def _fake_read_file(filename):
        data = {
            "coordinates": np.zeros((8, 3), dtype=np.float32),
            "metadata": {"power": 10.0, "velocity": 0.1, "time_s": 1e-6},
            "Temperature": np.full((8,), 300.0, dtype=np.float32),
        }
        if "caseB" in str(filename):
            data["ExtraField"] = np.ones((8,), dtype=np.float32)
        return data

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    ds.read_file = _fake_read_file
    _install_lightweight_resampling(ds)

    with pytest.warns(UserWarning, match="ExtraField") as record:
        var = ds.parse_directory(
            grid_params=_grid_params(),
            requested_fields=None,  # load_all_fields path
            time_axis=False,
            force_recompute=True,
            flip_axes=False,
        )

    # One consolidated warning, naming an example file; parse still succeeds
    # with the first file's channel set.
    dropped_warnings = [w for w in record if "ExtraField" in str(w.message)]
    assert len(dropped_warnings) == 1
    assert "caseB" in str(dropped_warnings[0].message)
    assert var["Temperature"].shape == (2, 1, 4, 2, 2)
    assert "ExtraField" not in var


# --------------------------------------------------------------------------- #
# categorical fields must be nearest-neighbour (k=1), not KNN-blended
# --------------------------------------------------------------------------- #

def _make_categorical_ds(module, tmp_path: Path):
    """Datasource over one scattered-point file with a discrete {0,3} 'flag' field."""
    (tmp_path / "caseA_1.vtk").write_text("")
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.05, 0.95, size=(40, 3)).astype(np.float32)
    flag = np.where(np.arange(40) % 2 == 0, 0.0, 3.0).astype(np.float32)

    def _fake_read_file(filename):
        return {
            "coordinates": coords.copy(),
            "metadata": {"power": 10.0, "velocity": 0.1, "time_s": 1e-6},
            "flag": flag.copy(),
        }

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    ds.read_file = _fake_read_file

    # Real KNN resampling, forced onto CPU (unit tests do not touch CUDA).
    orig_scalar = ds.crop_and_resample_scalar

    def _cpu_scalar(*args, **kwargs):
        kwargs["device"] = "cpu"
        return orig_scalar(*args, **kwargs)

    ds.crop_and_resample_scalar = _cpu_scalar
    return ds


def test_categorical_fields_resample_with_pure_nearest_neighbour(tmp_path: Path) -> None:
    module = _load_module()
    ds = _make_categorical_ds(module, tmp_path)
    common = dict(
        grid_params=_grid_params((3, 3, 3)),
        requested_fields=["flag"],
        time_axis=False,
        force_recompute=True,
        flip_axes=False,
    )

    blended = ds.parse_directory(**common)["flag"]
    is_discrete = np.isclose(blended, 0.0) | np.isclose(blended, 3.0)
    assert not is_discrete.all(), "default k=8 blending should produce intermediate values"

    categorical = ds.parse_directory(categorical_fields={"flag"}, **common)["flag"]
    is_discrete = np.isclose(categorical, 0.0) | np.isclose(categorical, 3.0)
    assert is_discrete.all(), "categorical resampling must keep only original discrete values"
