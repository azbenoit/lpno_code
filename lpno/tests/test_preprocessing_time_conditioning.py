import importlib.util
import os
import sys
import types
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/lpno/models/pipeline/preprocessing_datasets.py"


def _install_preprocessing_stubs(
    *,
    parse_return: dict[str, np.ndarray] | None = None,
    fill_delta: float = 0.0,
):
    physicsnemo_pkg = types.ModuleType("physicsnemo")
    sym_mod = types.ModuleType("physicsnemo.sym")
    dataset_mod = types.ModuleType("physicsnemo.sym.dataset")
    dataset_mod.__path__ = []
    dataset_dataset_mod = types.ModuleType("physicsnemo.sym.dataset.dataset")

    class _DummyDictGridDataset:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyDataset:
        @staticmethod
        def _to_tensor_dict(value):
            return value

    dataset_mod.DictGridDataset = _DummyDictGridDataset
    dataset_mod.DictPointwiseDataset = _DummyDictGridDataset
    dataset_dataset_mod.Dataset = _DummyDataset
    sym_mod.dataset = dataset_mod
    physicsnemo_pkg.sym = sym_mod
    sys.modules.setdefault("physicsnemo", physicsnemo_pkg)
    sys.modules.setdefault("physicsnemo.sym", sym_mod)
    sys.modules["physicsnemo.sym.dataset"] = dataset_mod
    sys.modules["physicsnemo.sym.dataset.dataset"] = dataset_dataset_mod

    sklearn_pkg = types.ModuleType("sklearn")
    model_selection_mod = types.ModuleType("sklearn.model_selection")

    class _DummyKFold:
        def __init__(self, *args, **kwargs):
            pass

    model_selection_mod.KFold = _DummyKFold
    sklearn_pkg.model_selection = model_selection_mod
    sys.modules.setdefault("sklearn", sklearn_pkg)
    sys.modules["sklearn.model_selection"] = model_selection_mod

    custom_utils_pkg = types.ModuleType("lpno.custom_utils")
    custom_utils_pkg.__path__ = [str(ROOT / "src/lpno/custom_utils")]  # REAL path: an
    # empty one made unstubbed submodules (npz_datasource) unimportable, so the
    # file broke ITSELF in any subset run where the real package was not
    # already imported (collection-time load, 2026-08-14)
    sys.modules.setdefault("lpno.custom_utils", custom_utils_pkg)

    dataset_loading = types.ModuleType("lpno.custom_utils.dataset_loading")

    def _dummy_get_c_piecewise_func(_):
        return (lambda t: t, lambda t: t, lambda t: t)

    dataset_loading.get_c_piecewise_func = _dummy_get_c_piecewise_func
    sys.modules["lpno.custom_utils.dataset_loading"] = dataset_loading

    openfoam_ds = types.ModuleType("lpno.custom_utils.openFoam_datasource")

    class _DummyDataSource:
        last_args = None
        last_kwargs = None

        def parse_directory(self, *args, **kwargs):
            _DummyDataSource.last_args = args
            _DummyDataSource.last_kwargs = kwargs
            if parse_return is None:
                raise RuntimeError("parse_directory return data was not configured for this test")
            return parse_return

    openfoam_ds.VTKOpenFOAMDataSource = _DummyDataSource
    sys.modules["lpno.custom_utils.openFoam_datasource"] = openfoam_ds

    utils_mod = types.ModuleType("lpno.custom_utils.utils")

    def _fill_where_mask_smooth_np(data, *args, **kwargs):
        return np.asarray(data) + fill_delta

    def _make_interface_sharp(alpha, threshold=0.5):
        return np.where(np.asarray(alpha) >= threshold, 1.0, 0.0)

    def _smooth_liquid_fraction_numpy(T, T_s, T_l):
        T = np.asarray(T, dtype=float)
        phi = (T - float(T_s)) / (float(T_l) - float(T_s))
        return np.clip(phi, 0.0, 1.0)

    def _split_dict(d, idx):
        return {k: v[idx] for k, v in d.items()}

    utils_mod.boiling_temperature = lambda material_params: 3123.0
    utils_mod.t_boil_from_manifest = lambda manifest: None
    utils_mod.fill_where_mask_smooth_np = _fill_where_mask_smooth_np
    utils_mod.make_interface_sharp = _make_interface_sharp
    utils_mod.smooth_liquid_fraction_numpy = _smooth_liquid_fraction_numpy
    utils_mod.split_dict = _split_dict
    # Real H* numerator (see the sibling stub): scaling behaviour must not be faked.
    def _h_norm_raw(power, v_scan, beam_radius=None, material_factor=None):
        value = np.asarray(power, dtype=np.float64) / np.sqrt(
            np.abs(np.asarray(v_scan, dtype=np.float64))
        )
        if beam_radius is not None:
            r0 = np.maximum(np.abs(np.asarray(beam_radius, dtype=np.float64)), 1e-12)
            value = value / np.sqrt(r0 ** 3)
        return value

    utils_mod.h_norm_raw = _h_norm_raw
    # Kept in sync with the real module: production code imports this, so a stub
    # without it breaks collection rather than the behaviour under test.
    def _king_material_factor(rho, cp, k, latent, T_l, T_0=293.0, A=1.0):
        rho = np.asarray(rho, dtype=np.float64); cp = np.asarray(cp, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64); latent = np.asarray(latent, dtype=np.float64)
        T_l = np.asarray(T_l, dtype=np.float64)
        d = np.maximum(k / np.maximum(rho * cp, 1e-12), 1e-12)
        return np.maximum(rho * (cp * np.maximum(T_l - float(T_0), 1.0) + latent)
                          * np.sqrt(np.pi * d), 1e-30) / np.maximum(A, 1e-6)
    utils_mod.king_material_factor = _king_material_factor
    sys.modules["lpno.custom_utils.utils"] = utils_mod

    package_pkg = types.ModuleType("lpno")
    package_pkg.__path__ = [str(ROOT / "src/lpno")]
    sys.modules.setdefault("lpno", package_pkg)

    models_pkg = types.ModuleType("lpno.models")
    models_pkg.__path__ = [str(ROOT / "src/lpno/models")]
    sys.modules.setdefault("lpno.models", models_pkg)

    pipeline_pkg = types.ModuleType("lpno.models.pipeline")
    pipeline_pkg.__path__ = [str(ROOT / "src/lpno/models/pipeline")]
    sys.modules.setdefault("lpno.models.pipeline", pipeline_pkg)

    dataclasses_mod = types.ModuleType("lpno.models.pipeline.dataclasses")
    dataclasses_mod.DatasetBundle = object
    dataclasses_mod.KeyConfig = object
    dataclasses_mod.RunConfig = object
    dataclasses_mod.SplitSpec = object
    sys.modules["lpno.models.pipeline.dataclasses"] = dataclasses_mod

    return _DummyDataSource


#: Every sys.modules key the stub installer touches. Same save/restore as
#: test_autoregressive_rollout (c81e826) and test_pointwise_grid_restore: an
#: unrestored stub survives for the REST of the suite -- the dataclasses stub
#: here (RunConfig = object) broke test_val_frames_per_case with "must be
#: called with a dataclass type or instance" once 2026-08-12's new tests
#: shifted the alphabetical order (2026-08-14).
_STUBBED_MODULE_NAMES = (
    "physicsnemo",
    "physicsnemo.sym",
    "physicsnemo.sym.dataset",
    "physicsnemo.sym.dataset.dataset",
    "sklearn",
    "sklearn.model_selection",
    "lpno.custom_utils",
    "lpno.custom_utils.dataset_loading",
    "lpno.custom_utils.openFoam_datasource",
    "lpno.custom_utils.utils",
    "lpno.models.pipeline.dataclasses",
)
_ORIGINAL_MODULES: dict = {}


def _load_preprocessing_module(
    *,
    parse_return: dict[str, np.ndarray] | None = None,
    fill_delta: float = 0.0,
):
    for name in _STUBBED_MODULE_NAMES:
        _ORIGINAL_MODULES[name] = sys.modules.get(name)
    data_source_cls = _install_preprocessing_stubs(parse_return=parse_return, fill_delta=fill_delta)
    module_name = "lpno.models.pipeline.preprocessing_datasets"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load preprocessing_datasets module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # The loaded module keeps its bound stub references; the global namespace
    # must not. Restore what was there, drop what was not.
    for name in _STUBBED_MODULE_NAMES:
        original = _ORIGINAL_MODULES.get(name)
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    return module, data_source_cls


def _make_run_cfg(*, include_time=True, time_as_batch=True, alpha_masking=False):
    return SimpleNamespace(
        software="flow3D",
        material_params={"T_s": 1873.0, "T_l": 1923.0},
        pde_weight=1.0,
        sharp_alpha=False,
        alpha_masking=alpha_masking,
        pde_loss=False,
        vof_pde_loss=False,
        include_time=include_time,
        time_as_batch=time_as_batch,
        use_2d=False,
        time_as_dim=False,
        time_avg_window=3,
        moving_ref_frame=True,
        moving_frame_steps=3,
        ref_scales={"L": 1e-4, "T": 3000.0, "V": 0.1, "P": 10.0, "Rho": 4.506e3, "H": 203.0},
        keys=SimpleNamespace(str_fno_input_keys=["x", "y", "z", "V_scan", "P", "t_hat"]),
        grid_params={"x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "z_range": [0.0, 1.0], "resolution": [2, 2, 2]},
        super_res_factor=1,
        vtk_fields=["Temperature", "Fraction Of Fluid", "Pressure", "Velocity"],
        data_directory_path="unused",
        param_dir="unused",
        laser_params={},
    )


def _make_flow3d_var(*, with_time_axis=False):
    B, T, nx, ny, nz = 2, 3, 2, 2, 2
    shape_scalar = (B, T, 1, nx, ny, nz) if with_time_axis else (B, 1, nx, ny, nz)
    shape_vector = (B, T, 3, nx, ny, nz) if with_time_axis else (B, 3, nx, ny, nz)
    base_time = np.linspace(1.0e-6, 3.0e-6, T, dtype=np.float32).reshape(1, T, 1, 1, 1, 1)
    if with_time_axis:
        time_s = np.broadcast_to(base_time, shape_scalar).copy()
    else:
        time_s = np.full(shape_scalar, 2.0e-6, dtype=np.float32)
    return {
        "Temperature": np.full(shape_scalar, 1900.0, dtype=np.float32),
        "Velocity": np.full(shape_vector, 0.25, dtype=np.float32),
        "Fraction Of Fluid": np.ones(shape_scalar, dtype=np.float32),
        "Pressure": np.zeros(shape_scalar, dtype=np.float32),
        "x": np.ones(shape_scalar, dtype=np.float32),
        "y": np.ones(shape_scalar, dtype=np.float32),
        "z": np.ones(shape_scalar, dtype=np.float32),
        "V_scan": np.full(shape_scalar, 0.1, dtype=np.float32),
        "P": np.full(shape_scalar, 10.0, dtype=np.float32),
        "time_s": time_s,
    }


def test_flatten_time_axis_6d_to_5d():
    module, _ = _load_preprocessing_module()
    arr = np.arange(2 * 3 * 1 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 1, 2, 2, 2)
    out = module._flatten_time_axis({"A": arr})
    assert out["A"].shape == (6, 1, 2, 2, 2)
    assert np.array_equal(out["A"][0], arr[0, 0])
    assert np.array_equal(out["A"][1], arr[0, 1])
    assert np.array_equal(out["A"][3], arr[1, 0])


def test_preprocess_var_builds_t_hat():
    module, _ = _load_preprocessing_module()
    run_cfg = _make_run_cfg(include_time=True, time_as_batch=True, alpha_masking=False)
    var = _make_flow3d_var(with_time_axis=False)
    out = module._preprocess_var(var=var, run_cfg=run_cfg, h_norm_mode="max", set_runtime_params=False)
    expected = out["time_s"] * (run_cfg.ref_scales["V"] / run_cfg.ref_scales["L"])
    assert "t_hat" in out
    assert np.allclose(out["t_hat"], expected)


def test_time_keys_not_alpha_masked():
    module, _ = _load_preprocessing_module(fill_delta=123.0)
    run_cfg = _make_run_cfg(include_time=True, time_as_batch=True, alpha_masking=True)
    var = _make_flow3d_var(with_time_axis=False)
    time_before = var["time_s"].copy()
    out = module._preprocess_var(var=var, run_cfg=run_cfg, h_norm_mode="max", set_runtime_params=False)
    assert np.array_equal(out["time_s"], time_before)
    expected = time_before * (run_cfg.ref_scales["V"] / run_cfg.ref_scales["L"])
    assert np.allclose(out["t_hat"], expected)


def test_load_and_preprocess_time_as_batch_calls_datasource_correctly():
    parse_return = _make_flow3d_var(with_time_axis=True)
    module, data_source_cls = _load_preprocessing_module(parse_return=parse_return)
    run_cfg = _make_run_cfg(include_time=True, time_as_batch=True, alpha_masking=False)
    with mock.patch.dict(os.environ, {"PINO_ALLOW_VTK": "1"}):  # legacy VTK path; gate tested separately
        out = module.load_and_preprocess_data(run_cfg=run_cfg, input_dir="dummy", set_runtime_params=False)
    assert data_source_cls.last_kwargs is not None
    assert data_source_cls.last_kwargs["time_axis"] is True
    # time_as_batch must not temporally average; it keeps time snapshots.
    assert data_source_cls.last_kwargs["avg_window_size"] == 1
    assert data_source_cls.last_kwargs["moving_ref_frame"] is True
    assert data_source_cls.last_kwargs["moving_frame_steps"] == run_cfg.moving_frame_steps
    # B*T = 2*3 after flattening.
    assert out["T"].shape[0] == 6
    assert out["t_hat"].shape[0] == 6


def test_load_and_preprocess_time_as_dim_keeps_moving_frame_shift_enabled():
    parse_return = _make_flow3d_var(with_time_axis=True)
    module, data_source_cls = _load_preprocessing_module(parse_return=parse_return)
    run_cfg = _make_run_cfg(include_time=True, time_as_batch=False, alpha_masking=False)
    run_cfg.time_as_dim = True
    run_cfg.use_2d = True
    with mock.patch.dict(os.environ, {"PINO_ALLOW_VTK": "1"}):  # legacy VTK path; gate tested separately
        out = module.load_and_preprocess_data(run_cfg=run_cfg, input_dir="dummy", set_runtime_params=False)
    assert data_source_cls.last_kwargs is not None
    assert data_source_cls.last_kwargs["time_axis"] is True
    # time_as_dim keeps moving-frame shift/crop while disabling temporal averaging.
    assert data_source_cls.last_kwargs["avg_window_size"] == 1
    assert data_source_cls.last_kwargs["moving_ref_frame"] is True
    assert data_source_cls.last_kwargs["moving_frame_steps"] == run_cfg.moving_frame_steps
    # Converted layout is (B,C,X,T,Z) after middle-y slicing.
    assert out["T"].ndim == 5
    assert out["T"].shape[3] == 3


def test_load_and_preprocess_disables_moving_ref_frame_with_effective_steps_one():
    parse_return = _make_flow3d_var(with_time_axis=True)
    module, data_source_cls = _load_preprocessing_module(parse_return=parse_return)
    run_cfg = _make_run_cfg(include_time=True, time_as_batch=True, alpha_masking=False)
    run_cfg.moving_ref_frame = False
    run_cfg.moving_frame_steps = 7
    with mock.patch.dict(os.environ, {"PINO_ALLOW_VTK": "1"}):  # legacy VTK path; gate tested separately
        out = module.load_and_preprocess_data(run_cfg=run_cfg, input_dir="dummy", set_runtime_params=False)
    assert data_source_cls.last_kwargs is not None
    assert data_source_cls.last_kwargs["moving_ref_frame"] is False
    assert data_source_cls.last_kwargs["moving_frame_steps"] == 1
    # Time-as-batch still keeps all timesteps before flattening.
    assert out["T"].shape[0] == 6


def test_load_and_preprocess_autoregressive_preserves_time_axis():
    parse_return = _make_flow3d_var(with_time_axis=True)
    module, data_source_cls = _load_preprocessing_module(parse_return=parse_return)
    run_cfg = _make_run_cfg(include_time=True, time_as_batch=True, alpha_masking=False)
    run_cfg.autoregressive_enabled = True
    with mock.patch.dict(os.environ, {"PINO_ALLOW_VTK": "1"}):  # legacy VTK path; gate tested separately
        out = module.load_and_preprocess_data(run_cfg=run_cfg, input_dir="dummy", set_runtime_params=False)
    assert data_source_cls.last_kwargs is not None
    assert data_source_cls.last_kwargs["time_axis"] is True
    # AR mode keeps temporal axis instead of flattening to B*T.
    assert out["T"].shape[:2] == (2, 3)
    assert out["t_hat"].shape[:2] == (2, 3)


def test_build_autoregressive_pairs_alignment():
    module, _ = _load_preprocessing_module()
    B, T = 2, 3
    x = np.arange(B * T * 1 * 2 * 2 * 2, dtype=np.float32).reshape(B, T, 1, 2, 2, 2)
    T_field = np.arange(B * T * 1 * 2 * 2 * 2, dtype=np.float32).reshape(B, T, 1, 2, 2, 2) + 100.0
    alpha = np.arange(B * T * 1 * 2 * 2 * 2, dtype=np.float32).reshape(B, T, 1, 2, 2, 2) + 200.0
    fl = np.arange(B * T * 1 * 2 * 2 * 2, dtype=np.float32).reshape(B, T, 1, 2, 2, 2) + 300.0
    var = {"x": x, "T": T_field, "alpha": alpha, "fl": fl}

    ds_input, ds_output = module._build_autoregressive_pairs(
        var=var,
        context_keys=["x"],
        state_keys=["T", "alpha"],
        state_prev_suffix="_prev",
        extra_target_keys=["fl"],
    )

    assert ds_input["x"].shape[0] == B * (T - 1)
    assert ds_input["T_prev"].shape[0] == B * (T - 1)
    assert ds_output["T"].shape[0] == B * (T - 1)
    assert ds_output["fl"].shape[0] == B * (T - 1)

    # pair index 0 uses previous state at t=0 and context/targets at t=1
    assert np.array_equal(ds_input["x"][0], x[0, 1])
    assert np.array_equal(ds_input["T_prev"][0], T_field[0, 0])
    assert np.array_equal(ds_input["alpha_prev"][0], alpha[0, 0])
    assert np.array_equal(ds_output["T"][0], T_field[0, 1])
    assert np.array_equal(ds_output["alpha"][0], alpha[0, 1])
    assert np.array_equal(ds_output["fl"][0], fl[0, 1])
