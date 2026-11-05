import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/lpno/custom_utils/openFoam_datasource.py"


_STUBBED_MODULE_NAMES = (
    "pyvista",
    "torch",
    "scipy",
    "scipy.spatial",
    "physicsnemo_curator",
    "physicsnemo_curator.etl",
    "physicsnemo_curator.etl.data_sources",
)


def _load_openfoam_module():
    # Save the real modules so the bare stubs below (notably a torch with no
    # .cuda) do not leak into later tests via sys.modules; restored after load
    # in the finally, mirroring test_pointwise_grid_restore's pattern.
    original_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}

    pyvista_mod = types.ModuleType("pyvista")
    pyvista_mod.read = lambda path: None
    sys.modules["pyvista"] = pyvista_mod

    torch_mod = types.ModuleType("torch")
    sys.modules["torch"] = torch_mod

    scipy_mod = types.ModuleType("scipy")
    scipy_spatial_mod = types.ModuleType("scipy.spatial")

    class _DummyKDTree:
        def __init__(self, *args, **kwargs):
            pass

        def query(self, *args, **kwargs):
            raise RuntimeError("cKDTree.query should not be reached in this test")

    scipy_spatial_mod.cKDTree = _DummyKDTree
    scipy_mod.spatial = scipy_spatial_mod
    sys.modules["scipy"] = scipy_mod
    sys.modules["scipy.spatial"] = scipy_spatial_mod

    physicsnemo_curator_pkg = types.ModuleType("physicsnemo_curator")
    etl_mod = types.ModuleType("physicsnemo_curator.etl")
    data_sources_mod = types.ModuleType("physicsnemo_curator.etl.data_sources")

    class _DummyDataSource:
        def __init__(self, cfg=None):
            self.cfg = cfg

    data_sources_mod.DataSource = _DummyDataSource
    etl_mod.data_sources = data_sources_mod
    physicsnemo_curator_pkg.etl = etl_mod
    sys.modules["physicsnemo_curator"] = physicsnemo_curator_pkg
    sys.modules["physicsnemo_curator.etl"] = etl_mod
    sys.modules["physicsnemo_curator.etl.data_sources"] = data_sources_mod

    module_name = "tests._openfoam_datasource_under_test"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load openFoam_datasource module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


def _install_lightweight_resampling(ds):
    def _fake_scalar(coords, values, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz, **kwargs):
        gx = np.linspace(Xmin, Xmax, nx, dtype=np.float32)
        gy = np.linspace(Ymin, Ymax, ny, dtype=np.float32)
        gz = np.linspace(Zmin, Zmax, nz, dtype=np.float32)
        fill = float(values[0])
        grid = np.full((nx, ny, nz), fill, dtype=np.float32)
        return gx, gy, gz, grid

    ds.crop_and_resample_scalar = _fake_scalar
    ds._detect_uniform_grid = lambda *args, **kwargs: None


def _fake_read_file(filename: str | Path):
    stem = Path(filename).stem
    step = int(stem.rsplit("_", 1)[-1])
    coords = np.zeros((8, 3), dtype=np.float32)
    temperature = np.full((8,), float(step), dtype=np.float32)
    return {
        "coordinates": coords,
        "metadata": {"power": 10.0, "velocity": 0.1, "time_s": float(step) * 1e-6},
        "Temperature": temperature,
    }


def test_parse_directory_disables_moving_ref_frame_without_shift_or_crop(tmp_path: Path) -> None:
    module = _load_openfoam_module()
    for step in range(3):
        (tmp_path / f"caseA_{step}.vtk").write_text("")

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    ds.read_file = _fake_read_file
    _install_lightweight_resampling(ds)

    grid_params = {
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
        "z_range": [0.0, 1.0],
        "resolution": [4, 2, 2],
    }
    var = ds.parse_directory(
        grid_params=grid_params,
        requested_fields=["Temperature"],
        time_axis=True,
        moving_ref_frame=False,
        moving_frame_steps=3,
        avg_window_size=2,
        force_recompute=True,
        flip_axes=False,
    )

    assert var["Temperature"].shape == (1, 3, 1, 4, 2, 2)
    assert np.allclose(var["Temperature"][0, :, 0, 0, 0, 0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    # Cache name carries a geometry/flip token (A2).
    assert len(list(tmp_path.glob("var_1_time_noref_*.npz"))) == 1


def test_parse_time_supports_decimal_and_legacy_integer_suffixes() -> None:
    module = _load_openfoam_module()

    assert module.VTKOpenFOAMDataSource._parse_time("case_0.00123") == 0.00123
    assert np.isclose(module.VTKOpenFOAMDataSource._parse_time("case_123"), 123e-6)


def test_parse_directory_uses_distinct_cache_when_moving_ref_frame_enabled(tmp_path: Path) -> None:
    module = _load_openfoam_module()
    for step in range(3):
        (tmp_path / f"caseA_{step}.vtk").write_text("")

    ds = module.VTKOpenFOAMDataSource(input_dir=str(tmp_path))
    ds.read_file = _fake_read_file
    _install_lightweight_resampling(ds)

    grid_params = {
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
        "z_range": [0.0, 1.0],
        "resolution": [4, 2, 2],
    }
    ds.parse_directory(
        grid_params=grid_params,
        requested_fields=["Temperature"],
        time_axis=True,
        moving_ref_frame=False,
        moving_frame_steps=3,
        avg_window_size=1,
        force_recompute=True,
        flip_axes=False,
    )
    var = ds.parse_directory(
        grid_params=grid_params,
        requested_fields=["Temperature"],
        time_axis=True,
        moving_ref_frame=True,
        moving_frame_steps=2,
        avg_window_size=1,
        force_recompute=True,
        flip_axes=False,
    )

    assert var["Temperature"].shape == (1, 2, 1, 3, 2, 2)
    assert np.allclose(var["Temperature"][0, :, 0, 0, 0, 0], np.array([1.0, 2.0], dtype=np.float32))
    # Cache names carry a geometry/flip token (A2).
    assert len(list(tmp_path.glob("var_1_time_steps2_avg1_*.npz"))) == 1
    assert len(list(tmp_path.glob("var_1_time_noref_*.npz"))) == 1
