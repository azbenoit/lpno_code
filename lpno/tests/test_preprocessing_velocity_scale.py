import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/lpno/models/pipeline/preprocessing_datasets.py"


def _install_preprocessing_stubs() -> None:
    physicsnemo_pkg = types.ModuleType("physicsnemo")
    sym_mod = types.ModuleType("physicsnemo.sym")
    dataset_mod = types.ModuleType("physicsnemo.sym.dataset")

    class _DummyDictGridDataset:
        def __init__(self, *args, **kwargs):
            pass

    dataset_mod.DictGridDataset = _DummyDictGridDataset
    sym_mod.dataset = dataset_mod
    # npz_datasource (now importable via the real package __path__) pulls
    # lazy_runtime, which imports physicsnemo.sym.dataset.dataset.Dataset --
    # the sibling time_conditioning file always stubbed it; this one did not.
    dataset_dataset_mod = types.ModuleType("physicsnemo.sym.dataset.dataset")

    class _DummyDataset:  # noqa: D401 - matching the sibling stub
        pass

    dataset_dataset_mod.Dataset = _DummyDataset
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
    # already imported (collection-time load, 2026-08-14)  # namespace marker
    sys.modules.setdefault("lpno.custom_utils", custom_utils_pkg)

    dataset_loading = types.ModuleType("lpno.custom_utils.dataset_loading")

    def _dummy_get_c_piecewise_func(_):
        return (lambda t: t, lambda t: t, lambda t: t)

    dataset_loading.get_c_piecewise_func = _dummy_get_c_piecewise_func
    sys.modules["lpno.custom_utils.dataset_loading"] = dataset_loading

    openfoam_ds = types.ModuleType("lpno.custom_utils.openFoam_datasource")

    class _DummyDataSource:
        def parse_directory(self, *args, **kwargs):
            raise RuntimeError("parse_directory should not be called in this unit test")

    openfoam_ds.VTKOpenFOAMDataSource = _DummyDataSource
    sys.modules["lpno.custom_utils.openFoam_datasource"] = openfoam_ds

    utils_mod = types.ModuleType("lpno.custom_utils.utils")

    def _fill_where_mask_smooth_np(data, *args, **kwargs):
        return np.asarray(data)

    def _make_interface_sharp(alpha, threshold=0.5):
        return np.where(np.asarray(alpha) >= threshold, 1.0, 0.0)

    def _smooth_liquid_fraction_numpy(T, T_s, T_l):
        T = np.asarray(T, dtype=float)
        phi = (T - float(T_s)) / (float(T_l) - float(T_s))
        return np.clip(phi, 0.0, 1.0)

    def _split_dict(d, idx):
        return {k: v[idx] for k, v in d.items()}

    def _boiling_temperature(material_params):
        return 3123.0

    utils_mod.boiling_temperature = _boiling_temperature
    # material-driven fill (2026-08-20): the stub declines, keeping legacy behaviour
    utils_mod.t_boil_from_manifest = lambda manifest: None
    utils_mod.fill_where_mask_smooth_np = _fill_where_mask_smooth_np
    utils_mod.make_interface_sharp = _make_interface_sharp
    utils_mod.smooth_liquid_fraction_numpy = _smooth_liquid_fraction_numpy
    utils_mod.split_dict = _split_dict
    # The REAL H* numerator, not a stub: this file exercises input SCALING, and a
    # simplified H_norm would hide exactly the class of defect the shared definition
    # exists to prevent (numerator and denominator disagreeing, 2026-08-21).
    def _h_norm_raw(power, v_scan, beam_radius=None, material_factor=None):
        value = np.asarray(power, dtype=np.float64) / np.sqrt(
            np.abs(np.asarray(v_scan, dtype=np.float64))
        )
        if beam_radius is not None:
            r0 = np.maximum(np.abs(np.asarray(beam_radius, dtype=np.float64)), 1e-12)
            value = value / np.sqrt(r0 ** 3)
        if material_factor is not None:
            value = value / np.maximum(np.asarray(material_factor, dtype=np.float64), 1e-30)
        return value

    # Real form for the same reason as `_h_norm_raw` above: the material half is the
    # other way numerator and denominator can silently disagree.
    def _king_material_factor(rho, cp, k, latent, T_l, T_0=293.0, A=1.0):
        rho = np.asarray(rho, dtype=np.float64); cp = np.asarray(cp, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64); latent = np.asarray(latent, dtype=np.float64)
        T_l = np.asarray(T_l, dtype=np.float64)
        diffusivity = np.maximum(k / np.maximum(rho * cp, 1e-12), 1e-12)
        sensible = cp * np.maximum(T_l - float(T_0), 1.0) + latent
        return np.maximum(rho * sensible * np.sqrt(np.pi * diffusivity), 1e-30) / np.maximum(A, 1e-6)

    utils_mod.h_norm_raw = _h_norm_raw
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


# The stubs above replace real modules in sys.modules (custom_utils.utils among
# them), and the loaded module keeps its own references to them, so they must not
# outlive the load: a leftover stub makes every later test that imports the real
# module fail with "cannot import name ... (unknown location)".
_STUBBED_MODULES = (
    # The PARENTS too: restoring only the leaf left a bare `sklearn` stub (a
    # module, not a package) in sys.modules, so the next real import of
    # sklearn.model_selection failed with "'sklearn' is not a package".
    "physicsnemo",
    "physicsnemo.sym",
    "sklearn",
    "lpno.custom_utils",
    "physicsnemo.sym.dataset",
    "physicsnemo.sym.dataset.dataset",
    "sklearn.model_selection",
    "lpno.custom_utils.dataset_loading",
    "lpno.custom_utils.openFoam_datasource",
    "lpno.custom_utils.utils",
    "lpno.models.pipeline.dataclasses",
    "lpno.models.pipeline.preprocessing_datasets",
)


def _load_preprocess_var():
    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    _install_preprocessing_stubs()
    module_name = "lpno.models.pipeline.preprocessing_datasets"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load preprocessing_datasets module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module._preprocess_var


_preprocess_var = _load_preprocess_var()


def _base_var():
    b, nx, ny, nz = 1, 2, 2, 2
    shape = (b, 1, nx, ny, nz)
    return {
        "Temperature": np.full(shape, 1900.0, dtype=np.float32),
        "Velocity": np.full((b, 3, nx, ny, nz), 0.25, dtype=np.float32),
        "Fraction Of Fluid": np.ones(shape, dtype=np.float32),
        "Pressure": np.zeros(shape, dtype=np.float32),
        "x": np.ones(shape, dtype=np.float32),
        "y": np.ones(shape, dtype=np.float32),
        "z": np.ones(shape, dtype=np.float32),
        "V_scan": np.full(shape, 0.1, dtype=np.float32),
        "P": np.full(shape, 10.0, dtype=np.float32),
    }


def _run_cfg(ref_scales):
    return SimpleNamespace(
        software="flow3D",
        material_params={"T_s": 1873.0, "T_l": 1923.0},
        pde_weight=1.0,
        sharp_alpha=False,
        alpha_masking=False,
        pde_loss=False,
        vof_pde_loss=False,
        ref_scales=ref_scales,
    )


def test_preprocess_var_uses_v_melt_for_meltpool_velocity():
    ref_scales = {
        "L": 1e-4,
        "T": 3000.0,
        "V": 0.1,
        "V_melt": 0.25,
        "P": 10.0,
        "Rho": 4.506e3,
        "H": 203.0,
    }
    out = _preprocess_var(
        var=_base_var(),
        run_cfg=_run_cfg(ref_scales),
        h_norm_mode="max",
        set_runtime_params=False,
    )

    # V_scan is normalized by scan scale V.
    assert np.isclose(float(out["V_scan"][0, 0, 0, 0, 0]), 1.0)
    # U/u/v/w are normalized by V_melt.
    assert np.isclose(float(out["U"][0, 0, 0, 0, 0]), 1.0)
    assert np.isclose(float(out["u"][0, 0, 0, 0, 0]), 1.0)
    assert np.isclose(float(out["v"][0, 0, 0, 0, 0]), 1.0)
    assert np.isclose(float(out["w"][0, 0, 0, 0, 0]), 1.0)


def test_preprocess_var_velocity_falls_back_to_v_without_v_melt():
    ref_scales = {
        "L": 1e-4,
        "T": 3000.0,
        "V": 0.1,
        "P": 10.0,
        "Rho": 4.506e3,
        "H": 203.0,
    }
    out = _preprocess_var(
        var=_base_var(),
        run_cfg=_run_cfg(ref_scales),
        h_norm_mode="max",
        set_runtime_params=False,
    )

    # Fallback behavior remains previous default (divide U by V).
    assert np.isclose(float(out["U"][0, 0, 0, 0, 0]), 2.5)


def _multi_case_var(p_values, v=0.1):
    """Build a var dict with one case per entry in ``p_values`` (varying P, fixed V)."""
    b, nx, ny, nz = len(p_values), 2, 2, 2
    shape = (b, 1, nx, ny, nz)
    pressure = np.zeros(shape, dtype=np.float32)
    for i, p in enumerate(p_values):
        pressure[i] = float(p)
    return {
        "Temperature": np.full(shape, 1900.0, dtype=np.float32),
        "Velocity": np.full((b, 3, nx, ny, nz), 0.25, dtype=np.float32),
        "Fraction Of Fluid": np.ones(shape, dtype=np.float32),
        "Pressure": np.zeros(shape, dtype=np.float32),
        "x": np.ones(shape, dtype=np.float32),
        "y": np.ones(shape, dtype=np.float32),
        "z": np.ones(shape, dtype=np.float32),
        "V_scan": np.full(shape, float(v), dtype=np.float32),
        "P": pressure,
    }


def _run_cfg_runtime(ref_scales):
    # Like _run_cfg but carries the runtime-param containers the authoritative
    # train pass (set_runtime_params=True) mutates. c is a float so
    # _define_material_properties skips the piecewise specific_heat.txt read.
    return SimpleNamespace(
        software="flow3D",
        param_dir="/nonexistent",
        material_params={"T_s": 1873.0, "T_l": 1923.0, "c": 0.5},
        laser_params={},
        pde_weight=1.0,
        sharp_alpha=False,
        alpha_masking=False,
        pde_loss=False,
        vof_pde_loss=False,
        ref_scales=ref_scales,
    )


def test_val_pass_reuses_train_h_norm_denom():
    # BUG-04 regression: train and val splits with DIFFERENT max(P/sqrt|V|) must
    # still map the same physical (P, V_scan) to the same normalized H_norm. The
    # denom is fitted on the authoritative train pass and reused on val/eval.
    ref_scales = {"L": 1e-4, "T": 3000.0, "V": 0.1, "V_melt": 0.25, "P": 10.0, "Rho": 4.506e3}
    cfg = _run_cfg_runtime(ref_scales)  # shared across both passes, like _run_standard

    # Shared physical point is case 0 (P=10, V=0.1) in both splits; the splits
    # differ only in their hottest case, so their per-split max(P/sqrt|V|) differ.
    train_out = _preprocess_var(
        var=_multi_case_var([10.0, 20.0]), run_cfg=cfg,
        h_norm_mode="max", set_runtime_params=True,
    )
    val_out = _preprocess_var(
        var=_multi_case_var([10.0, 40.0]), run_cfg=cfg,
        h_norm_mode="max", set_runtime_params=False,
    )

    h_train_shared = float(train_out["H_norm"][0, 0, 0, 0, 0])
    h_val_shared = float(val_out["H_norm"][0, 0, 0, 0, 0])
    # Both use the train denom (max = 20/sqrt(0.1)); the shared P=10 case -> 0.5.
    assert np.isclose(h_train_shared, 0.5), h_train_shared
    assert np.isclose(h_val_shared, h_train_shared), (h_val_shared, h_train_shared)
