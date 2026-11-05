"""An evaluation scales H_norm by the TRAINING split's maximum, never the scored split's.

2026-10-30. H_norm = (P / sqrt|V|) / denom with denom the maximum over the training split
(BUG-04). Nothing carried that number past the training run, so a standalone evaluation
(`custom.data_dir` = the scored split) refitted it on the scored split: every published
single-alloy evaluation fed its model H_norm scaled by train max / scored max (0.918 to
1.445 on the test splits, up to 3.99 on the one-case figure evaluations, where the scored
case reads exactly 1.0). These pin each path of the fix:

- training writes the denominator it used beside its weights (lazy and eager loaders);
- an evaluation of a model carrying that record uses it;
- an evaluation of an older model recomputes it from the training split the model's
  archived config names, when readable;
- otherwise it says so loudly, falls back to the scored split, and the stored metrics
  flag it.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import lpno.models.ns_energy_pipeline as pipeline
import lpno.models.pipeline.config_loading as config_loading
import lpno.models.pipeline.preprocessing_datasets as prep
from lpno.custom_utils.utils import h_norm_raw
from lpno.models.pipeline import h_norm_record as rec
from lpno.models.pipeline.dataclasses import SplitSpec
from lpno.models.pipeline.lazy_runtime import (
    RUNTIME_FORMAT_V2,
    ArrayDirCollectionStore,
    compute_h_norm_denom,
)
from lpno.models.pipeline.postprocessing import _write_metric_file

# Training split max at (200 W, 0.5 m/s); the scored split peaks at half that.
TRAIN = [(100.0, 0.5), (200.0, 0.5), (150.0, 1.0)]
TEST = [(100.0, 0.5), (80.0, 0.8)]
TRAIN_MAX = 200.0 / np.sqrt(0.5)
TEST_MAX = 100.0 / np.sqrt(0.5)


def _write_collection(root: Path, cases) -> Path:
    for i, (power, speed) in enumerate(cases):
        member = root / f"case_{i}"
        member.mkdir(parents=True)
        arrays = {"Temperature": np.full((1, 4, 3, 2), 300.0, np.float32),
                  "P": np.full((1, 4, 3, 2), power, np.float32),
                  "V_scan": np.full((1, 4, 3, 2), speed, np.float32)}
        for name, arr in arrays.items():
            np.save(member / f"{name}.npy", arr)
        (member / "manifest.json").write_text(json.dumps({
            "format": RUNTIME_FORMAT_V2, "source_kind": "simulation",
            "simulation_metadata": {"case_name": member.name},
            "array_entries": [{"original_name": n, "filename": f"{n}.npy", "shape": list(a.shape)}
                              for n, a in arrays.items()],
        }))
    (root / "manifest.json").write_text(json.dumps({
        "format": RUNTIME_FORMAT_V2, "source_kind": "dataset_collection",
        "members": [{"relative_path": f"case_{i}"} for i in range(len(cases))]}))
    return root


def _lazy_run_cfg(network_dir: Path, *, eval_only: bool, h_norm_denom=None):
    return SimpleNamespace(
        eval_only=eval_only, val_max_cases=0, val_frames_per_case=0, seed=42,
        time_as_dim=False, time_as_batch=False, use_2d=False, autoregressive_enabled=False,
        case_coherent_sampling=False, h_norm_denom=h_norm_denom, h_norm_include_radius=False,
        h_norm_include_material=False, network_base_dir=str(network_dir),
        cfg=OmegaConf.create({"network_dir": str(network_dir)}),
    )


def _build_lazy(store, run_cfg, monkeypatch):
    """Run `_build_lazy_datasets` and return the denominator its sample builders got."""
    seen = {}
    monkeypatch.setattr(prep, "_prepare_runtime_reference",
                        lambda store, run_cfg, idx, *, h_norm_denom: seen.setdefault("denom", h_norm_denom))
    keys = SimpleNamespace(str_fno_input_keys=["P", "V_scan", "H_norm"], str_fno_output_keys=["T"])
    n = store.sample_count()
    prep._build_lazy_datasets(var=store, key_cfg=keys,
                              split=SplitSpec(fold=0, train_idx=np.arange(n), val_idx=np.arange(n)),
                              run_cfg=run_cfg)
    return seen["denom"]


# --------------------------------------------------------------------------- training


def test_lazy_training_records_the_denominator_it_fitted(tmp_path, monkeypatch):
    store = ArrayDirCollectionStore(_write_collection(tmp_path / "train", TRAIN))
    run_dir = tmp_path / "run"
    used = _build_lazy(store, _lazy_run_cfg(run_dir, eval_only=False), monkeypatch)
    assert np.isclose(used, TRAIN_MAX)
    record = json.loads((run_dir / rec.H_NORM_RECORD_NAME).read_text())
    assert np.isclose(record["h_norm_denom"], TRAIN_MAX)
    assert record["n_cases"] == len(TRAIN)
    assert record["material_aware"] is False and record["radius_aware"] is False


def test_eval_never_writes_a_record_into_the_model(tmp_path, monkeypatch):
    store = ArrayDirCollectionStore(_write_collection(tmp_path / "test", TEST))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _build_lazy(store, _lazy_run_cfg(model_dir, eval_only=True, h_norm_denom=TRAIN_MAX), monkeypatch)
    assert not (model_dir / rec.H_NORM_RECORD_NAME).exists()


def test_eager_training_records_the_stashed_denominator(tmp_path, monkeypatch):
    """The eager loader fits the denominator in `_preprocess_var` and stashes it on
    laser_params; `_run_standard` records that value."""
    run_cfg = SimpleNamespace(
        wb_project="p", run_name="r", seed=0, data_directory_path="d", val_directory_path="v",
        keys=SimpleNamespace(str_fno_output_keys=["T", "alpha"]),
        pde_loss=False, vof_pde_loss=False, diff_loss=False, material_params={},
        laser_params={}, ref_scales={}, alpha_masking=True, eval_only=False, dx=1.0, dy=1.0,
        dz=1.0, diff_weight=1.0, debug=False, h_norm_denom=None)

    def _load(run_cfg, input_dir=None, h_norm_mode="max", set_runtime_params=True):
        if set_runtime_params:
            run_cfg.laser_params["h_norm_denom"] = TRAIN_MAX
        return {"var": np.zeros(1)}

    for name, value in {
        "setup_wandb_run": lambda *a, **k: None, "load_and_preprocess_data": _load,
        "print_dict_shapes": lambda *a, **k: None, "sample_count": lambda *a, **k: 3,
        "get_splits": lambda *a, **k: [SimpleNamespace(train_idx=[0], val_idx=[1])],
        "build_datasets": lambda *a, **k: SimpleNamespace(ds_input=None, ds_output=None),
        "build_model_and_nodes": lambda *a, **k: SimpleNamespace(model=None),
        "build_domain": lambda *a, **k: None, "solve_domain": lambda *a, **k: None,
        "assert_init_checkpoint_loadable": lambda *a, **k: None,
    }.items():
        monkeypatch.setattr(pipeline, name, value)
    run_dir = tmp_path / "eager_run"
    pipeline._run_standard(SimpleNamespace(network_dir=str(run_dir)), run_cfg)
    record = json.loads((run_dir / rec.H_NORM_RECORD_NAME).read_text())
    assert np.isclose(record["h_norm_denom"], TRAIN_MAX)


# --------------------------------------------------------------------------- evaluation


def _record(model_dir: Path, denom: float, **extra) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / rec.H_NORM_RECORD_NAME).write_text(json.dumps(
        {"format": 1, "h_norm_denom": denom, "radius_aware": False, "material_aware": False, **extra}))


def _resolve(model_dir, **kw):
    args = dict(network_dir=model_dir, explicit=None, include_radius=False,
                include_material=False, time_as_batch=False)
    args.update(kw)
    return rec.resolve_eval_denominator(**args)


def test_eval_uses_the_models_record(tmp_path):
    _record(tmp_path / "model", TRAIN_MAX)
    res = _resolve(tmp_path / "model")
    assert res.source == rec.SOURCE_RECORD and np.isclose(res.denom, TRAIN_MAX)


def test_an_imported_models_record_is_found_one_level_up(tmp_path):
    """Imported models keep their files under `model/`; the archived config is searched in
    the same three places."""
    _record(tmp_path / "bundle", TRAIN_MAX)
    (tmp_path / "bundle" / "model").mkdir()
    assert np.isclose(_resolve(tmp_path / "bundle" / "model").denom, TRAIN_MAX)


def test_legacy_model_recomputes_from_its_training_split(tmp_path):
    train_dir = _write_collection(tmp_path / "train", TRAIN)
    model_dir = tmp_path / "legacy_model"
    (model_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"custom": {"data_dir": str(train_dir)}}),
                   model_dir / ".hydra" / "config.yaml")
    res = _resolve(model_dir)
    assert res.source == rec.SOURCE_RECOMPUTED
    expected = compute_h_norm_denom(ArrayDirCollectionStore(train_dir), range(len(TRAIN)), time_as_batch=False)
    assert np.isclose(res.denom, expected) and np.isclose(res.denom, TRAIN_MAX)


def test_unrecoverable_model_falls_back_loudly(tmp_path, capsys):
    model_dir = tmp_path / "orphan"
    (model_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"custom": {"data_dir": str(tmp_path / "gone")}}),
                   model_dir / ".hydra" / "config.yaml")
    res = _resolve(model_dir)
    assert res.denom is None and res.source == rec.SOURCE_SCORED_SPLIT
    out = capsys.readouterr().out
    assert "NO TRAINING NORMALIZATION FOR H_norm" in out and "not comparable" in out


def test_explicit_override_wins_and_a_disagreement_is_announced(tmp_path, capsys):
    _record(tmp_path / "model", TRAIN_MAX)
    res = _resolve(tmp_path / "model", explicit=TEST_MAX)
    assert res.source == rec.SOURCE_EXPLICIT and np.isclose(res.denom, TEST_MAX)
    assert "overrides the denominator this model trained with" in capsys.readouterr().out


def test_a_record_fitted_under_another_convention_is_refused(tmp_path):
    _record(tmp_path / "model", TRAIN_MAX, material_aware=True)
    with pytest.raises(ValueError, match="material_aware"):
        _resolve(tmp_path / "model")


def test_eval_lazy_build_scales_by_the_training_denominator(tmp_path, monkeypatch):
    """The scored split peaks at half the training max. The builders must receive the
    training value; before the fix they received TEST_MAX."""
    store = ArrayDirCollectionStore(_write_collection(tmp_path / "test", TEST))
    _record(tmp_path / "model", TRAIN_MAX)
    res = _resolve(tmp_path / "model")
    run_cfg = _lazy_run_cfg(tmp_path / "model", eval_only=True, h_norm_denom=res.denom)
    used = _build_lazy(store, run_cfg, monkeypatch)
    assert np.isclose(used, TRAIN_MAX) and not np.isclose(used, TEST_MAX)
    # the scored case at (100 W, 0.5 m/s) reaches the network at 0.5, as in training
    assert np.isclose(h_norm_raw(100.0, 0.5) / used, 0.5)


# ------------------------------------------------------------ the config loader wires it in


def _base_custom():
    return {
        "name": "test", "software": "openFoam",
        "geometry": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "z_range": [0.0, 1.0],
                     "resolution": [10, 8, 6]},
        "ref_scales": {"L": 1.0, "T": 2.0, "V": 3.0, "P": 4.0, "Rho": 5.0, "H": 6.0},
        "material_params": {"T_s": 1800, "T_l": 1900, "K": {"coeffs": [1.0, 2.0]}, "K_g": 0.1},
        "laser": {"x0": 0.0, "z0": 0.0},
        "vtk_fields": ["Temperature", "Fraction Of Fluid", "Pressure", "Velocity"],
        "FNO_input_keys": ["x", "y", "z"], "FNO_output_keys": ["T", "alpha"],
        "Energy_input_keys": ["u", "v", "w"], "Energy_output_keys": ["energy"],
        "VOF_input_keys": ["u", "v", "w"], "VOF_output_keys": ["VOF"],
        "Diff_input_keys": [], "Diff_output_keys": ["T_diffTerm"],
        "include_time": False, "time_as_batch": False, "infer_velocities": False,
        "inference_batch_size": 1, "moving_ref_frame": True, "moving_frame_steps": 1,
    }


@pytest.fixture
def _data_root(tmp_path, monkeypatch):
    root = tmp_path / "data_root"
    for sub in (config_loading._DEFAULT_TRAIN_DATA_SUBDIR, config_loading._DEFAULT_VAL_DATA_SUBDIR,
                config_loading._DEFAULT_TEST_DATA_SUBDIR, config_loading._DEFAULT_PARAM_DATA_SUBDIR):
        (root / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PINO_SIMULATIONS_DATA_ROOT", str(root))
    monkeypatch.setattr("sys.argv", ["eval"])
    return root


def _eval_cfg(model_dir: Path, model_custom: dict | None = None, custom: dict | None = None):
    (model_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"run_mode": "train", "network_dir": ".",
                                     "custom": {**_base_custom(), **(model_custom or {})},
                                     "arch": {"fno": {"padding": 1, "fno_modes": [3, 3, 3]}}}),
                   model_dir / ".hydra" / "config.yaml")
    return OmegaConf.create({"custom": {**_base_custom(), **(custom or {})}, "network_dir": str(model_dir),
                             "run_mode": "eval", "batch_size": {"validation": 1},
                             "arch": {"fno": {"padding": 1, "fno_modes": [3, 3, 3]}}})


def test_load_cfg_and_paths_takes_the_record_at_eval(tmp_path, _data_root):
    model_dir = tmp_path / "model"
    _record(model_dir, TRAIN_MAX)
    run_cfg = config_loading.load_cfg_and_paths(_eval_cfg(model_dir))
    assert np.isclose(run_cfg.h_norm_denom, TRAIN_MAX)
    assert run_cfg.h_norm_denom_source == rec.SOURCE_RECORD


def test_load_cfg_and_paths_recomputes_for_a_legacy_model(tmp_path, _data_root):
    train_dir = _write_collection(tmp_path / "train", TRAIN)
    run_cfg = config_loading.load_cfg_and_paths(
        _eval_cfg(tmp_path / "model", model_custom={"data_dir": str(train_dir)}))
    assert np.isclose(run_cfg.h_norm_denom, TRAIN_MAX)
    assert run_cfg.h_norm_denom_source == rec.SOURCE_RECOMPUTED


def test_load_cfg_and_paths_keeps_an_inherited_explicit_denominator(tmp_path, _data_root):
    """The two-alloy models trained with custom.h_norm_denom and their evals inherit it."""
    run_cfg = config_loading.load_cfg_and_paths(
        _eval_cfg(tmp_path / "model", model_custom={"h_norm_denom": 1.25e-5}))
    assert run_cfg.h_norm_denom == pytest.approx(1.25e-5)
    assert run_cfg.h_norm_denom_source == rec.SOURCE_EXPLICIT


def test_training_load_resolves_nothing(tmp_path, _data_root):
    cfg = _eval_cfg(tmp_path / "model")
    cfg.run_mode = "train"
    cfg.network_dir = str(tmp_path / "net")
    run_cfg = config_loading.load_cfg_and_paths(cfg)
    assert run_cfg.h_norm_denom is None and run_cfg.h_norm_denom_source is None


# ------------------------------------------------------------------ what the eval stores


def test_metrics_record_the_denominator_and_flag_a_scored_split_fit(tmp_path):
    def _written(run_cfg):
        path = tmp_path / "model_errors.txt"
        _write_metric_file(path, {"T": {"abs_err_mean": 0.01}}, run_cfg=run_cfg)
        return path.read_text()

    text = _written(SimpleNamespace(eval_only=True, h_norm_denom=TRAIN_MAX,
                                    h_norm_denom_source=rec.SOURCE_RECORD, laser_params={}))
    assert "[h_norm]" in text and f"denom: {TRAIN_MAX}" in text and "scored_split: 0.0" in text
    text = _written(SimpleNamespace(eval_only=True, h_norm_denom=None,
                                    h_norm_denom_source=rec.SOURCE_SCORED_SPLIT,
                                    laser_params={"h_norm_denom": TEST_MAX}))
    assert f"denom: {TEST_MAX}" in text and "scored_split: 1.0" in text
    assert "[h_norm]" not in _written(SimpleNamespace(eval_only=False))


def test_a_radius_aware_record_passes_the_explicit_denominator_guard():
    """`_preprocess_var` refuses custom.h_norm_denom with h_norm_include_radius because a
    user-supplied value is radius-blind. A record fitted radius-aware is not."""
    var = {"T": np.full((1, 1, 2, 2, 2), 300.0, np.float32), "alpha.metal": np.ones((1, 1, 2, 2, 2), np.float32),
           "P": np.full((1, 1, 2, 2, 2), 100.0, np.float32), "V_scan": np.full((1, 1, 2, 2, 2), 0.5, np.float32),
           "beam_radius": np.full((1, 1, 2, 2, 2), 5e-5, np.float32),
           "x": np.zeros((1, 1, 2, 2, 2), np.float32), "y": np.zeros((1, 1, 2, 2, 2), np.float32),
           "z": np.zeros((1, 1, 2, 2, 2), np.float32)}
    denom = float(h_norm_raw(200.0, 0.5, 5e-5))
    base = dict(software="none", sharp_alpha=False, alpha_masking=False, laser_params={},
                pde_loss=False, vof_pde_loss=False, pde_weight=1.0, param_dir="/nonexistent",
                material_params={"T_s": 1873.0, "T_l": 1923.0, "c": 0.5}, h_norm_include_radius=True,
                h_norm_denom=denom, ref_scales={"L": 1e-4, "T": 3000.0, "V": 0.1, "P": 10.0, "Rho": 4420.0})
    with pytest.raises(ValueError, match="radius-BLIND"):
        prep._preprocess_var(dict(var), SimpleNamespace(**base), h_norm_mode="max",
                             set_runtime_params=False, h_norm_denom=denom)
    out = prep._preprocess_var(dict(var), SimpleNamespace(**base, h_norm_denom_radius_aware=True),
                               h_norm_mode="max", set_runtime_params=False, h_norm_denom=denom)
    assert np.isclose(float(out["H_norm"].reshape(-1)[0]), 0.5)
