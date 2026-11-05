import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test envs.
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch is required for autoregressive rollout tests")


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/lpno/models/pipeline/postprocessing.py"


# Every sys.modules key the stub installer touches. test_pointwise_grid_restore
# got this save/restore treatment earlier; this file kept polluting: its utils
# stub (4 functions) stayed installed for the REST OF THE SUITE, so any later
# test importing anything else from the real custom_utils.utils died with
# ImportError -- first victim: test_late_param_group (2026-08-10).
_STUBBED_MODULE_NAMES = (
    "lpno.custom_utils",
    "lpno.custom_utils.inferencer",
    "lpno.custom_utils.utils",
    "lpno",
    "lpno.models",
    "lpno.models.pipeline",
    "lpno.models.pipeline.dataclasses",
)
_ORIGINAL_MODULES: dict = {}


def _install_postprocessing_stubs():
    custom_utils_pkg = types.ModuleType("lpno.custom_utils")
    custom_utils_pkg.__path__ = []
    sys.modules.setdefault("lpno.custom_utils", custom_utils_pkg)

    inferencer_mod = types.ModuleType("lpno.custom_utils.inferencer")

    class _DummyFNOGridInferencer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            raise RuntimeError("FNOGridInferencer should not be used in autoregressive rollout tests.")

    inferencer_mod.FNOGridInferencer = _DummyFNOGridInferencer
    sys.modules["lpno.custom_utils.inferencer"] = inferencer_mod

    utils_mod = types.ModuleType("lpno.custom_utils.utils")

    def _fill_where_mask_smooth_np(data, *args, **kwargs):
        return np.asarray(data)

    def _iou_per_batch(a, b, threshold=0.5):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if a_arr.shape != b_arr.shape:
            raise ValueError("Shape mismatch in IoU stub")
        return np.ones((a_arr.shape[0],), dtype=np.float32)

    def _smooth_liquid_fraction_numpy(T, T_s, T_l):
        T_arr = np.asarray(T, dtype=float)
        frac = (T_arr - float(T_s)) / (float(T_l) - float(T_s))
        return np.clip(frac, 0.0, 1.0)

    def _boiling_fill_value(material_params, ref_scales):
        return 3123.0 / float(ref_scales["T"])

    utils_mod.boiling_fill_value = _boiling_fill_value
    # postprocessing now imports the per-case fill helper too (custom.per_case_t_boil,
    # 2026-09-04); under this stub there is no mat_Tv channel, so it is the scalar.
    utils_mod.t_fill_field = lambda run_cfg, tensors=None: _boiling_fill_value(
        run_cfg.material_params, run_cfg.ref_scales)
    utils_mod.fill_where_mask_smooth_np = _fill_where_mask_smooth_np
    utils_mod.iou_per_batch = _iou_per_batch
    utils_mod.smooth_liquid_fraction_numpy = _smooth_liquid_fraction_numpy
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

    @dataclass
    class _InferenceMetrics:
        errors: dict[str, dict[str, float]]
        val_rows: list[dict[str, object]]

    dataclasses_mod.InferenceMetrics = _InferenceMetrics
    dataclasses_mod.RunConfig = object
    sys.modules["lpno.models.pipeline.dataclasses"] = dataclasses_mod


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
    # The loaded module holds bound references to the stubs it imported; the
    # global namespace must not. Restore what was there, drop what was not.
    for name in _STUBBED_MODULE_NAMES:
        original = _ORIGINAL_MODULES.get(name)
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    return module


if torch is not None:
    class _ContextAwareRolloutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, invar):
            return {
                "T": invar["T_prev"] + invar["x"],
                "alpha": invar["alpha_prev"],
            }


def _make_run_cfg():
    return SimpleNamespace(
        autoregressive_enabled=True,
        autoregressive_eval_use_gt_init=True,
        autoregressive_context_keys=["x"],
        autoregressive_state_keys=["T", "alpha"],
        autoregressive_state_prev_suffix="_prev",
        autoregressive_eval_horizon_k=10,
        inference_batch_size=2,
        alpha_masking=False,
        material_params={"T_s": 0.2, "T_l": 0.8},
        ref_scales={"T": 1.0},
    )


if torch is not None:
    class _PointwiseRolloutModel(torch.nn.Module):
        """Pointwise twin of the context-aware model: requires flattened (N, C) inputs."""

        def __init__(self):
            super().__init__()
            self._dummy = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, invar):
            if invar["x"].ndim != 2:
                raise AssertionError(
                    f"pointwise model expects flattened (N, C) inputs; got ndim={invar['x'].ndim}"
                )
            return {
                "T": invar["T_prev"] + invar["x"],
                "alpha": invar["alpha_prev"],
            }


def test_autoregressive_rollout_uses_context_each_step(tmp_path: Path):
    module = _load_postprocessing_module()

    # B=1, T=4, scalar grid. x is per-step forcing: 1,2,3 at t=0,1,2.
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 4, 1, 1, 1, 1)
    T = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32).reshape(1, 4, 1, 1, 1, 1)
    alpha = np.ones((1, 4, 1, 1, 1, 1), dtype=np.float32)
    fl = np.ones((1, 4, 1, 1, 1, 1), dtype=np.float32)
    trajectory_var = {"x": x, "T": T, "alpha": alpha, "fl": fl}

    out_dir = tmp_path / "fno_eval"
    metrics = module.run_autoregressive_rollout_and_metrics(
        model=_ContextAwareRolloutModel(),
        trajectory_var=trajectory_var,
        run_cfg=_make_run_cfg(),
        output_dir=out_dir,
    )

    preds = np.load(out_dir / "rollout_predictions.npz")
    pred_T = preds["T"].reshape(1, 3)
    assert np.allclose(pred_T, np.array([[1.0, 3.0, 6.0]], dtype=np.float32))
    assert "T" in metrics.errors
    assert metrics.errors["T"]["abs_err_mean"] == 0.0

    assert (out_dir / "model_errors_rollout_full.txt").is_file()
    assert (out_dir / "model_errors_rollout_k.txt").is_file()
    assert (out_dir / "ground_truth_rollout.npz").is_file()
    assert (out_dir / "rollout_errors_per_step.csv").is_file()


def test_autoregressive_rollout_pointwise_layout(tmp_path: Path):
    module = _load_postprocessing_module()

    # B=1, T=4, C=1, spatial (2, 1, 1): two voxels with distinct trajectories.
    x = np.zeros((1, 4, 1, 2, 1, 1), dtype=np.float32)
    x[0, :, 0, 0, 0, 0] = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    x[0, :, 0, 1, 0, 0] = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    T = np.zeros_like(x)
    T[0, :, 0, 0, 0, 0] = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
    T[0, :, 0, 1, 0, 0] = np.array([0.0, 10.0, 30.0, 60.0], dtype=np.float32)
    alpha = np.ones_like(x)
    fl = np.ones_like(x)
    trajectory_var = {"x": x, "T": T, "alpha": alpha, "fl": fl}

    run_cfg = _make_run_cfg()
    run_cfg.data_layout = "pointwise"

    out_dir = tmp_path / "fno_eval"
    metrics = module.run_autoregressive_rollout_and_metrics(
        model=_PointwiseRolloutModel(),
        trajectory_var=trajectory_var,
        run_cfg=run_cfg,
        output_dir=out_dir,
    )

    preds = np.load(out_dir / "rollout_predictions.npz")
    assert preds["T"].shape == (1, 3, 1, 2, 1, 1)
    np.testing.assert_allclose(preds["T"][0, :, 0, 0, 0, 0], np.array([1.0, 3.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(preds["T"][0, :, 0, 1, 0, 0], np.array([10.0, 30.0, 60.0], dtype=np.float32))
    assert metrics.errors["T"]["abs_err_mean"] == 0.0


def test_autoregressive_rollout_masks_fl_and_scores_iou_per_frame(tmp_path: Path):
    module = _load_postprocessing_module()

    fl_mask_calls = []
    iou_shapes = []

    def _recording_fill(data, mask, threshold=0.5, d=0.0, k=20, inplace=False):
        fl_mask_calls.append((np.asarray(data).shape, float(d)))
        return np.asarray(data)

    def _recording_iou(a, b, threshold=0.5):
        iou_shapes.append(np.asarray(a).shape)
        return np.ones((np.asarray(a).shape[0],), dtype=np.float32)

    module.fill_where_mask_smooth_np = _recording_fill
    module.iou_per_batch = _recording_iou

    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 4, 1, 1, 1, 1)
    T = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32).reshape(1, 4, 1, 1, 1, 1)
    alpha = np.ones((1, 4, 1, 1, 1, 1), dtype=np.float32)
    fl = np.ones((1, 4, 1, 1, 1, 1), dtype=np.float32)
    trajectory_var = {"x": x, "T": T, "alpha": alpha, "fl": fl}

    run_cfg = _make_run_cfg()
    run_cfg.alpha_masking = True

    module.run_autoregressive_rollout_and_metrics(
        model=_ContextAwareRolloutModel(),
        trajectory_var=trajectory_var,
        run_cfg=run_cfg,
        output_dir=tmp_path / "fno_eval",
    )

    # Per step, alpha masking must touch T (boiling fill), alpha (zero fill),
    # AND the derived fl (zero fill), matching the direct grid eval convention:
    # 3 calls x 3 steps. Without the fl fix this is 6 total / 3 zero-fills.
    zero_fill_calls = [c for c in fl_mask_calls if c[1] == 0.0]
    assert len(fl_mask_calls) == 9, fl_mask_calls
    assert len(zero_fill_calls) == 6, fl_mask_calls

    # IoU must be computed per frame (leading dim B*(T-1)=3), not per trajectory.
    assert iou_shapes and all(shape[0] == 3 for shape in iou_shapes), iou_shapes
