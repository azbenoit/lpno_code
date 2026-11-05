import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/lpno/models/pipeline/postprocessing.py"


def _install_postprocessing_stubs() -> None:
    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class _Module:
        pass

    class _InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    nn_mod.Module = _Module
    torch_mod.nn = nn_mod
    torch_mod.Tensor = object
    torch_mod.device = lambda arg=None: arg
    torch_mod.inference_mode = lambda: _InferenceMode()
    torch_mod.from_numpy = lambda array: array
    torch_mod.cuda = SimpleNamespace(is_available=lambda: False)
    data_mod.DataLoader = object
    utils_mod.data = data_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod

    custom_utils_pkg = types.ModuleType("lpno.custom_utils")
    custom_utils_pkg.__path__ = []
    sys.modules.setdefault("lpno.custom_utils", custom_utils_pkg)

    inferencer_mod = types.ModuleType("lpno.custom_utils.inferencer")
    inferencer_mod.FNOGridInferencer = object
    sys.modules["lpno.custom_utils.inferencer"] = inferencer_mod

    helper_utils_mod = types.ModuleType("lpno.custom_utils.utils")
    helper_utils_mod.boiling_fill_value = lambda material_params, ref_scales: 3123.0 / float(ref_scales["T"])
    # postprocessing also imports the per-case fill helper (custom.per_case_t_boil, 2026-09-04);
    # no mat_Tv channel under this stub, so it is the same scalar.
    helper_utils_mod.t_fill_field = lambda run_cfg, tensors=None: 3123.0 / float(run_cfg.ref_scales["T"])
    helper_utils_mod.fill_where_mask_smooth_np = lambda data, *args, **kwargs: np.asarray(data)
    helper_utils_mod.iou_per_batch = lambda a, b, threshold=0.5: np.ones((np.asarray(a).shape[0],), dtype=np.float32)
    helper_utils_mod.smooth_liquid_fraction_numpy = lambda T, T_s, T_l: np.asarray(T)
    sys.modules["lpno.custom_utils.utils"] = helper_utils_mod

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

    @dataclass
    class _InferenceMetrics:
        errors: dict[str, dict[str, float]]
        val_rows: list[dict[str, object]]

    dataclasses_mod.InferenceMetrics = _InferenceMetrics
    dataclasses_mod.RunConfig = object
    sys.modules["lpno.models.pipeline.dataclasses"] = dataclasses_mod


_STUBBED_MODULE_NAMES = (
    "torch",
    "torch.nn",
    "torch.utils",
    "torch.utils.data",
    "lpno.custom_utils",
    "lpno.custom_utils.inferencer",
    "lpno.custom_utils.utils",
    "lpno",
    "lpno.models",
    "lpno.models.pipeline",
    "lpno.models.pipeline.dataclasses",
)
_ORIGINAL_MODULES: dict[str, object | None] = {}


def _load_postprocessing_module():
    for name in _STUBBED_MODULE_NAMES:
        _ORIGINAL_MODULES[name] = sys.modules.get(name)
    _install_postprocessing_stubs()
    module_name = "lpno.models.pipeline.postprocessing"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load postprocessing module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in _STUBBED_MODULE_NAMES:
        original = _ORIGINAL_MODULES.get(name)
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    return module


def test_restore_pointwise_grid_layout_3d():
    module = _load_postprocessing_module()
    run_cfg = SimpleNamespace(use_2d=False, time_as_dim=False, nx_effective=3, ny_effective=2, nz_effective=2)

    structured = np.arange(2 * 1 * 3 * 2 * 2, dtype=np.float32).reshape(2, 1, 3, 2, 2)
    flattened = np.moveaxis(structured, 1, -1).reshape(-1, 1)
    restored = module._restore_pointwise_grid_layout({"T": flattened}, run_cfg, keys=["T"])

    assert restored["T"].shape == structured.shape
    assert np.array_equal(restored["T"], structured)


def test_restore_pointwise_grid_layout_2d():
    module = _load_postprocessing_module()
    run_cfg = SimpleNamespace(use_2d=True, time_as_dim=False, nx_effective=4, ny_effective=1, nz_effective=3)

    structured = np.arange(2 * 1 * 4 * 3, dtype=np.float32).reshape(2, 1, 4, 3)
    flattened = np.moveaxis(structured, 1, -1).reshape(-1, 1)
    restored = module._restore_pointwise_grid_layout({"T": flattened}, run_cfg, keys=["T"])

    assert restored["T"].shape == structured.shape
    assert np.array_equal(restored["T"], structured)


def test_restore_pointwise_grid_layout_uses_grid_resolution_when_nx_effective_is_stale():
    module = _load_postprocessing_module()
    run_cfg = SimpleNamespace(
        use_2d=False,
        time_as_dim=False,
        include_time=True,
        moving_ref_frame=True,
        moving_frame_steps=30,
        super_res_factor=1,
        grid_params={"resolution": (90, 40, 30)},
        nx_effective=60,
        ny_effective=40,
        nz_effective=30,
    )

    structured = np.arange(2 * 1 * 61 * 40 * 30, dtype=np.float32).reshape(2, 1, 61, 40, 30)
    flattened = np.moveaxis(structured, 1, -1).reshape(-1, 1)
    restored = module._restore_pointwise_grid_layout({"T": flattened}, run_cfg, keys=["T"])

    assert restored["T"].shape == structured.shape
    assert np.array_equal(restored["T"], structured)
