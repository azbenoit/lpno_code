import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import lpno.models.pipeline.config_loading as config_loading
from lpno.models.ns_energy_pipeline import _collect_sim_param_overrides, load_cfg_and_paths


# pyright: reportArgumentType=false

def _create_default_data_root(data_root: Path) -> Path:
    for relative_path in (
        config_loading._DEFAULT_TRAIN_DATA_SUBDIR,
        config_loading._DEFAULT_VAL_DATA_SUBDIR,
        config_loading._DEFAULT_TEST_DATA_SUBDIR,
        config_loading._DEFAULT_PARAM_DATA_SUBDIR,
    ):
        (data_root / relative_path).mkdir(parents=True, exist_ok=True)
    return data_root


@pytest.fixture(autouse=True)
def _default_data_root_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = _create_default_data_root(tmp_path / "data_root")
    monkeypatch.setenv("PINO_SIMULATIONS_DATA_ROOT", str(data_root))
    return data_root


def _base_custom_sim() -> dict:
    return {
        "name": "test",
        "software": "openFoam",
        "geometry": {
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "z_range": [0.0, 1.0],
            "resolution": [10, 8, 6],
        },
        "ref_scales": {"L": 1.0, "T": 2.0, "V": 3.0, "P": 4.0, "Rho": 5.0, "H": 6.0},
        "material_params": {"T_s": 1800, "T_l": 1900, "K": {"coeffs": [1.0, 2.0]}, "K_g": 0.1},
        "laser": {"x0": 0.0, "z0": 0.0},
        "vtk_fields": ["Temperature", "Fraction Of Fluid", "Pressure", "Velocity"],
        "FNO_input_keys": ["x", "y", "z"],
        "FNO_output_keys": ["T", "alpha"],
        "Energy_input_keys": ["u", "v", "w"],
        "Energy_output_keys": ["energy"],
        "VOF_input_keys": ["u", "v", "w"],
        "VOF_output_keys": ["VOF"],
        "Diff_input_keys": [],
        "Diff_output_keys": ["T_diffTerm"],
        "include_time": False,
        "time_as_batch": False,
        "infer_velocities": False,
        "inference_batch_size": 1,
        "moving_ref_frame": True,
        "moving_frame_steps": 30,
    }


def _make_cfg(tmp_path: Path, custom_overrides: dict | None = None, extra: dict | None = None):
    custom = _base_custom_sim()
    if custom_overrides:
        custom = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(custom), OmegaConf.create(custom_overrides)),
            resolve=True,
        )

    cfg_dict = {
        "custom": custom,
        "network_dir": str(tmp_path / "net"),
        "run_mode": "train",
        "batch_size": {"validation": 1},
        "arch": {
            "fno": {
                "padding": 1,
                "fno_modes": [3, 3, 3],
            }
        },
    }
    if extra:
        cfg_dict = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(cfg_dict), OmegaConf.create(extra)),
            resolve=True,
        )
    return OmegaConf.create(cfg_dict)


def _write_model_hydra_config(
    run_dir: Path,
    custom_overrides: dict | None = None,
    extra: dict | None = None,
) -> None:
    custom = _base_custom_sim()
    if custom_overrides:
        custom = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(custom), OmegaConf.create(custom_overrides)),
            resolve=True,
        )

    model_cfg = {
        "run_mode": "train",
        "network_dir": ".",
        "custom": custom,
        "arch": {
            "fno": {
                "padding": 1,
                "fno_modes": [3, 3, 3],
            }
        },
        "hydra": {"run": {"dir": "outputs/fake"}},
    }
    if extra:
        model_cfg = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(model_cfg), OmegaConf.create(extra)),
            resolve=True,
        )

    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(model_cfg), f=hydra_dir / "config.yaml")


def test_collect_sim_param_overrides_prefers_custom_over_top_level() -> None:
    cfg = OmegaConf.create(
        {
            "custom": {
                "geometry": {"resolution": [8, 4, 2]},
                "material_params": {"T_s": 1900},
                "FNO_input_keys": ["x", "y", "z"],
            },
            "geometry": {"resolution": [100, 100, 100]},
            "material_params": {"T_s": 9999},
        }
    )

    overrides = _collect_sim_param_overrides(cfg)

    assert overrides["geometry"]["resolution"] == [8, 4, 2]
    assert overrides["material_params"]["T_s"] == 1900
    assert overrides["FNO_input_keys"] == ["x", "y", "z"]


def test_collect_sim_param_overrides_falls_back_to_top_level() -> None:
    cfg = OmegaConf.create(
        {
            "geometry": {"resolution": [8, 4, 2]},
            "material_params": {"T_s": 1900},
            "FNO_input_keys": ["x", "y", "z"],
            "unknown_key": "ignored",
        }
    )

    overrides = _collect_sim_param_overrides(cfg)

    assert "geometry" in overrides
    assert "material_params" in overrides
    assert "FNO_input_keys" in overrides
    assert "unknown_key" not in overrides


def test_load_cfg_and_paths_uses_custom_as_primary_source(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "geometry": {"resolution": [12, 9, 7]},
            "material_params": {"T_s": 1950},
        },
        extra={
            "geometry": {"resolution": [2, 2, 2]},
            "material_params": {"T_s": -1},
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.grid_params["resolution"] == (12, 9, 7)
    assert run_cfg.material_params["T_s"] == 1950
    assert run_cfg.material_params["T_l"] == 1900
    assert run_cfg.time_as_batch is False
    assert run_cfg.moving_ref_frame is True
    assert run_cfg.moving_frame_steps == 30
    assert run_cfg.inference_batch_size == 1
    assert run_cfg.nx_effective == 12
    assert run_cfg.keys.str_fno_output_keys == ["T", "alpha"]


def test_load_cfg_and_paths_adds_velocity_outputs_when_enabled(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, custom_overrides={"infer_velocities": True})

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.keys.str_fno_output_keys == ["T", "alpha", "u", "v", "w"]


def test_load_cfg_and_paths_ignores_top_level_infer_velocities_when_custom_false(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={"infer_velocities": False},
        extra={"infer_velocities": True},
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.keys.str_fno_output_keys == ["T", "alpha"]


def test_load_cfg_and_paths_requires_include_time_for_t_hat(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": False,
            "FNO_input_keys": ["x", "y", "z", "t_hat"],
        },
    )

    try:
        load_cfg_and_paths(cfg)
    except ValueError as exc:
        assert "t_hat" in str(exc)
    else:
        raise AssertionError("Expected ValueError when t_hat is configured with include_time=False")


def test_load_cfg_and_paths_auto_adds_t_hat_when_time_as_batch_enabled(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.time_as_batch is True
    assert "t_hat" in run_cfg.keys.str_fno_input_keys


def test_load_cfg_and_paths_autoregressive_adds_prev_state_inputs(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True},
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.autoregressive_enabled is True
    assert run_cfg.autoregressive_state_prev_suffix == "_prev"
    assert run_cfg.autoregressive_eval_horizon_k == 10
    assert run_cfg.autoregressive_eval_use_gt_init is True
    assert run_cfg.autoregressive_context_keys == ["x", "y", "z", "t_hat"]
    assert run_cfg.autoregressive_state_keys == ["T", "alpha"]
    assert "T_prev" in run_cfg.keys.str_fno_input_keys
    assert "alpha_prev" in run_cfg.keys.str_fno_input_keys


def test_load_cfg_and_paths_defaults_model_family_to_fno(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.model_family == "fno"
    assert run_cfg.data_layout == "grid"


def test_load_cfg_and_paths_keeps_fno_mutation_gated_to_fno(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={"model_family": "unet"},
        extra={"arch": {"fno": {"padding": 3, "fno_modes": [9, 7, 5], "dimension": 99}}},
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.model_family == "unet"
    assert run_cfg.data_layout == "grid"
    assert cfg.arch.fno.padding == 3
    assert cfg.arch.fno.fno_modes == [9, 7, 5]
    assert cfg.arch.fno.dimension == 99


def test_load_cfg_and_paths_derives_deeponet_branch_and_trunk_keys(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "model_family": "deeponet",
            "FNO_input_keys": ["V_scan", "P", "H_norm", "x", "y", "z"],
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.model_family == "deeponet"
    assert run_cfg.data_layout == "pointwise"
    assert run_cfg.deeponet_branch_keys == ["V_scan", "P", "H_norm"]
    assert run_cfg.deeponet_trunk_keys == ["x", "y", "z"]


@pytest.mark.parametrize(
    ("custom_overrides", "message"),
    [
        ({"model_family": "fully_connected", "pde_loss": True}, "pde_loss"),
        (
            {
                "model_family": "fully_connected",
                "include_time": True,
                "time_as_batch": True,
                "autoregressive": {"enabled": True},
            },
            "autoregressive.enabled",
        ),
        ({"model_family": "fully_connected", "include_time": True, "time_as_dim": True}, "time_as_dim"),
    ],
)
def test_load_cfg_and_paths_rejects_non_fno_unsupported_modes(
    tmp_path: Path,
    custom_overrides: dict,
    message: str,
) -> None:
    cfg = _make_cfg(tmp_path, custom_overrides=custom_overrides)

    with pytest.raises(ValueError, match=message):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_autoregressive_requires_include_time(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": False,
            "time_as_batch": True,
            "autoregressive": {"enabled": True, "require_time": True},
        },
    )

    with pytest.raises(ValueError, match="requires include_time=True"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_autoregressive_requires_time_as_batch(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": False,
            "autoregressive": {"enabled": True},
        },
    )

    with pytest.raises(ValueError, match="requires time_as_batch=True"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_autoregressive_requires_data_only_mode(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "moving_frame_steps": 1,
            "pde_loss": True,
            "autoregressive": {"enabled": True},
        },
    )

    with pytest.raises(ValueError, match="data-only mode"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_autoregressive_rejects_prev_key_collision(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "moving_frame_steps": 1,
            "FNO_input_keys": ["x", "y", "z", "T_prev"],
            "autoregressive": {"enabled": True, "state_prev_suffix": "_prev"},
        },
    )

    with pytest.raises(ValueError, match="input key collision"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_disables_moving_ref_frame_with_effective_step_one(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "moving_ref_frame": False,
            "moving_frame_steps": 4,
            "geometry": {"resolution": [12, 8, 6]},
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.moving_ref_frame is False
    assert run_cfg.moving_frame_steps == 1
    # Disabling the moving reference frame preserves the original x resolution.
    assert run_cfg.nx_effective == 12


@pytest.mark.parametrize(
    ("custom_overrides", "extra"),
    (
        ({"infer_time": True}, None),
        ({"time_sampling_mode": "per_timestep"}, None),
        (None, {"infer_time": True}),
        (None, {"time_sampling_mode": "per_timestep"}),
    ),
)
def test_load_cfg_and_paths_errors_on_legacy_time_keys(
    tmp_path: Path,
    custom_overrides: dict | None,
    extra: dict | None,
) -> None:
    cfg = _make_cfg(tmp_path, custom_overrides=custom_overrides, extra=extra)

    with pytest.raises(ValueError, match="Legacy temporal config keys are no longer supported"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_errors_on_legacy_time_keys_in_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path)
    monkeypatch.setattr(
        config_loading,
        "_parse_explicit_cli_override_paths",
        lambda argv=None: ({"custom.infer_time"}, set()),
    )

    with pytest.raises(ValueError, match="Legacy temporal config keys are no longer supported"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_errors_on_deprecated_top_level_sim_params(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, extra={"sim_params": "src/conf/sim_params/flow3D_params.yaml"})

    try:
        load_cfg_and_paths(cfg)
    except ValueError as exc:
        assert "sim_params" in str(exc)
        assert "deprecated" in str(exc)
    else:
        raise AssertionError("Expected ValueError when deprecated sim_params is provided.")


def test_load_cfg_and_paths_errors_on_deprecated_custom_sim_params(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, extra={"custom": {"sim_params": "src/conf/sim_params/flow3D_params.yaml"}})

    try:
        load_cfg_and_paths(cfg)
    except ValueError as exc:
        assert "sim_params" in str(exc)
        assert "deprecated" in str(exc)
    else:
        raise AssertionError("Expected ValueError when deprecated custom.sim_params is provided.")


def test_profile_configs_have_required_custom_fields_and_load(tmp_path: Path) -> None:
    profile_cases = [
        ("config_NS_energy_PINO_flow3D.yaml", "flow3D"),
        ("config_NS_energy_PINO_lbf_params_VP_coarse.yaml", "openFoam"),
        ("config_NS_energy_PINO_abaqus.yaml", "abaqus"),
    ]

    for cfg_name, expected_software in profile_cases:
        cfg_path = Path("src/lpno/conf/hydra") / cfg_name
        cfg = OmegaConf.load(cfg_path)
        cfg.network_dir = str(tmp_path / cfg_name.replace(".yaml", ""))
        run_cfg = load_cfg_and_paths(cfg)

        assert run_cfg.software == expected_software
        assert run_cfg.grid_params is not None
        assert run_cfg.ref_scales is not None
        assert run_cfg.keys.str_fno_input_keys


def test_eval_load_cfg_and_paths_uses_model_hydra_defaults(tmp_path: Path) -> None:
    run_dir = tmp_path / "trained_run"
    _write_model_hydra_config(
        run_dir,
        custom_overrides={
            "geometry": {"resolution": [16, 12, 10]},
            "material_params": {"T_s": 1777},
            "FNO_output_keys": ["T", "alpha", "u"],
            "include_time": False,
            "moving_frame_steps": 1,
        },
        extra={"arch": {"fno": {"fno_modes": [4, 4, 4]}}},
    )
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "geometry": {"resolution": [10, 8, 6]},
            "moving_frame_steps": 1,
        },
        extra={"run_mode": "eval", "network_dir": str(run_dir)},
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.grid_params["resolution"] == (16, 12, 10)
    assert run_cfg.material_params["T_s"] == 1777
    assert run_cfg.cfg.arch.fno.fno_modes == (4, 4, 4)
    assert run_cfg.keys.str_fno_output_keys == ["T", "alpha", "u"]


def test_eval_load_cfg_and_paths_uses_grandparent_hydra_fallback(tmp_path: Path) -> None:
    run_root = tmp_path / "FNO_nested_run"
    _write_model_hydra_config(
        run_root,
        custom_overrides={
            "geometry": {"resolution": [14, 11, 9]},
            "moving_frame_steps": 1,
        },
    )
    nested_network_dir = run_root / "outputs" / "FNO_nested_run"
    nested_network_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "geometry": {"resolution": [10, 8, 6]},
            "moving_frame_steps": 1,
        },
        extra={"run_mode": "eval", "network_dir": str(nested_network_dir)},
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.grid_params["resolution"] == (14, 11, 9)


def test_eval_load_cfg_and_paths_prefers_grandparent_hydra_over_parent(tmp_path: Path) -> None:
    run_root = tmp_path / "FNO_nested_run"
    _write_model_hydra_config(
        run_root,
        custom_overrides={
            "geometry": {"resolution": [14, 11, 9]},
            "moving_frame_steps": 1,
        },
    )
    parent_outputs = run_root / "outputs"
    _write_model_hydra_config(
        parent_outputs,
        custom_overrides={
            "geometry": {"resolution": [30, 30, 30]},
            "moving_frame_steps": 1,
        },
    )
    nested_network_dir = parent_outputs / "FNO_nested_run"
    nested_network_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "geometry": {"resolution": [10, 8, 6]},
            "moving_frame_steps": 1,
        },
        extra={"run_mode": "eval", "network_dir": str(nested_network_dir)},
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.grid_params["resolution"] == (14, 11, 9)


def test_eval_load_cfg_and_paths_cli_overrides_win_over_model_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "trained_run"
    _write_model_hydra_config(run_dir, custom_overrides={"geometry": {"resolution": [20, 20, 20]}})
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "geometry": {"resolution": [12, 9, 7]},
            "moving_frame_steps": 1,
        },
        extra={"run_mode": "eval", "network_dir": str(run_dir)},
    )
    monkeypatch.setattr(
        config_loading,
        "_parse_explicit_cli_override_paths",
        lambda argv=None: ({"custom.geometry.resolution"}, set()),
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.grid_params["resolution"] == (12, 9, 7)


def test_eval_load_cfg_and_paths_defaults_dataset_to_test_even_if_model_hydra_has_data_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "trained_run"
    _write_model_hydra_config(
        run_dir,
        custom_overrides={
            "data_dir": "/tmp/model/train",
            "val_data_dir": "/tmp/model/val",
            "moving_frame_steps": 1,
        },
    )
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={"moving_frame_steps": 1},
        extra={"run_mode": "eval", "network_dir": str(run_dir)},
    )

    run_cfg = load_cfg_and_paths(cfg)
    expected_test_dir = str(
        (tmp_path / "data_root" / config_loading._DEFAULT_TEST_DATA_SUBDIR).resolve()
    )

    assert run_cfg.data_directory_path == expected_test_dir
    assert run_cfg.val_directory_path == expected_test_dir


def test_eval_load_cfg_and_paths_warns_and_falls_back_when_model_hydra_missing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run_dir = tmp_path / "trained_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={"moving_frame_steps": 1},
        extra={"run_mode": "eval", "network_dir": str(run_dir)},
    )

    run_cfg = load_cfg_and_paths(cfg)
    captured = capsys.readouterr()

    assert run_cfg.grid_params["resolution"] == (10, 8, 6)
    assert "Warning: Eval defaults not loaded from model .hydra config" in captured.out


def test_eval_load_cfg_and_paths_migrates_legacy_model_hydra_time_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    run_dir = tmp_path / "trained_run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    legacy_model_cfg = OmegaConf.create(
        {
            "run_mode": "train",
            "network_dir": ".",
            "custom": {
                "name": "legacy",
                "software": "openFoam",
                "geometry": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "z_range": [0.0, 1.0], "resolution": [10, 8, 6]},
                "ref_scales": {"L": 1.0, "T": 2.0, "V": 3.0, "P": 4.0, "Rho": 5.0, "H": 6.0},
                "material_params": {"T_s": 1800, "T_l": 1900, "K": {"coeffs": [1.0, 2.0]}, "K_g": 0.1},
                "laser": {"x0": 0.0, "z0": 0.0},
                "vtk_fields": ["Temperature", "Fraction Of Fluid", "Pressure", "Velocity"],
                "FNO_input_keys": ["x", "y", "z"],
                "FNO_output_keys": ["T", "alpha"],
                "Energy_input_keys": ["u", "v", "w"],
                "Energy_output_keys": ["energy"],
                "VOF_input_keys": ["u", "v", "w"],
                "VOF_output_keys": ["VOF"],
                "Diff_input_keys": [],
                "Diff_output_keys": ["T_diffTerm"],
                "include_time": True,
                "infer_time": True,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
            },
            "arch": {"fno": {"padding": 1, "fno_modes": [3, 3, 3]}},
            "hydra": {"run": {"dir": "outputs/fake"}},
        }
    )
    OmegaConf.save(config=legacy_model_cfg, f=hydra_dir / "config.yaml")
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={"include_time": True, "moving_frame_steps": 1},
        extra={"run_mode": "eval", "network_dir": str(run_dir)},
    )

    run_cfg = load_cfg_and_paths(cfg)
    captured = capsys.readouterr()

    assert run_cfg.time_as_batch is True
    assert "t_hat" in run_cfg.keys.str_fno_input_keys
    assert "Migrated legacy model temporal keys to custom.time_as_batch" in captured.out


def test_load_cfg_and_paths_uses_env_data_root_for_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data_root"
    train_dir = data_root / "flow3D" / "original_straight_path_data" / "train"
    val_dir = data_root / "flow3D" / "original_straight_path_data" / "val"
    test_dir = data_root / "flow3D" / "original_straight_path_data" / "test"
    param_dir = data_root / "abaqus_data" / "Ti64_params"
    for path in (train_dir, val_dir, test_dir, param_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PINO_SIMULATIONS_DATA_ROOT", str(data_root))
    cfg = _make_cfg(tmp_path)

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.data_directory_path == str(train_dir)
    assert run_cfg.val_directory_path == str(val_dir)
    assert run_cfg.param_dir == str(param_dir)


def test_load_cfg_and_paths_errors_when_default_data_root_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PINO_SIMULATIONS_DATA_ROOT", raising=False)
    monkeypatch.setattr(config_loading, "_discover_repo_data_root", lambda: None)
    cfg = _make_cfg(tmp_path)

    with pytest.raises(ValueError, match="Default dataset paths could not be resolved"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_accepts_rollout_mode_for_fno(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "rollout"},
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.autoregressive_mode == "rollout"
    assert run_cfg.autoregressive_state_mask_keys == []
    assert "T_prev" in run_cfg.keys.str_fno_input_keys
    assert "alpha_prev" in run_cfg.keys.str_fno_input_keys
    assert "T_prev_mask" not in run_cfg.keys.str_fno_input_keys
    assert run_cfg.autoregressive_prev_noise_std == 0.0


def test_load_cfg_and_paths_accepts_rollout_mode_for_rigno(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "model_family": "rigno",
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "rollout"},
        },
        extra={"arch": {"rigno": {"dimension": 3}}},
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.model_family == "rigno"
    assert run_cfg.autoregressive_mode == "rollout"
    assert "T_prev" in run_cfg.keys.str_fno_input_keys
    assert "alpha_prev" in run_cfg.keys.str_fno_input_keys
    assert run_cfg.autoregressive_state_mask_keys == []


def test_load_cfg_and_paths_accepts_rollout_mode_for_deeponet(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "model_family": "deeponet",
            "FNO_input_keys": ["V_scan", "P", "x", "y", "z"],
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "rollout"},
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.model_family == "deeponet"
    assert run_cfg.data_layout == "pointwise"
    assert "T_prev" in run_cfg.deeponet_branch_keys
    assert "alpha_prev" in run_cfg.deeponet_branch_keys
    assert run_cfg.deeponet_trunk_keys == ["x", "y", "z"]


def test_load_cfg_and_paths_rollout_mode_routes_prev_state_to_trunk_when_asked(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "model_family": "deeponet",
            "FNO_input_keys": ["V_scan", "P", "x", "y", "z"],
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "deeponet_trunk_extra_keys": ["T_prev", "alpha_prev"],
            "autoregressive": {"enabled": True, "mode": "rollout"},
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert "T_prev" in run_cfg.deeponet_trunk_keys
    assert "alpha_prev" in run_cfg.deeponet_trunk_keys
    assert "T_prev" not in run_cfg.deeponet_branch_keys


def test_load_cfg_and_paths_masked_patch_remains_fno_only(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "model_family": "rigno",
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "masked_patch"},
        },
        extra={"arch": {"rigno": {"dimension": 3}}},
    )

    with pytest.raises(ValueError, match="masked_patch"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_rejects_unknown_autoregressive_mode(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "recursive"},
        },
    )

    with pytest.raises(ValueError, match="autoregressive.mode"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_rollout_mode_skips_masked_patch_runtime_defaults(tmp_path: Path) -> None:
    from omegaconf import OmegaConf as _OC

    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "rollout"},
        },
    )

    load_cfg_and_paths(cfg)

    assert _OC.select(cfg, "custom.alpha_masking") is None
    assert _OC.select(cfg, "training.grad_clip_max_norm") is None


def test_load_cfg_and_paths_autoregressive_prev_noise_std(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "rollout", "prev_noise_std": 0.05},
        },
    )

    run_cfg = load_cfg_and_paths(cfg)

    assert run_cfg.autoregressive_prev_noise_std == pytest.approx(0.05)


def test_load_cfg_and_paths_rejects_negative_prev_noise_std(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "rollout", "prev_noise_std": -0.1},
        },
    )

    with pytest.raises(ValueError, match="prev_noise_std"):
        load_cfg_and_paths(cfg)


def test_load_cfg_and_paths_rollout_rejects_disabled_gt_init(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={
            "include_time": True,
            "time_as_batch": True,
            "time_avg_window": 1,
            "moving_frame_steps": 1,
            "autoregressive": {"enabled": True, "mode": "rollout", "eval_use_gt_init": False},
        },
    )

    with pytest.raises(ValueError, match="eval_use_gt_init"):
        load_cfg_and_paths(cfg)


def test_eval_does_not_require_a_val_subdir_it_never_uses(tmp_path: Path) -> None:
    """Regression: eval resolved the default val/ path and then threw it away.

    `_resolve_configured_data_path` raises when the default VAL subdir is
    missing, but the very next line overwrote the result with the eval data
    path -- so a data root with no val/ crashed the run over a value that was
    never read. Eval validates against the eval set itself.
    """
    data_root = tmp_path / "data_root"
    _create_default_data_root(data_root)
    shutil.rmtree(data_root / config_loading._DEFAULT_VAL_DATA_SUBDIR)

    run_dir = tmp_path / "trained_run"
    _write_model_hydra_config(run_dir, custom_overrides={"moving_frame_steps": 1})
    cfg = _make_cfg(
        tmp_path,
        custom_overrides={"moving_frame_steps": 1},
        extra={"run_mode": "eval", "network_dir": str(run_dir)},
    )

    run_cfg = load_cfg_and_paths(cfg)

    expected = str((data_root / config_loading._DEFAULT_TEST_DATA_SUBDIR).resolve())
    assert run_cfg.data_directory_path == expected
    assert run_cfg.val_directory_path == expected


def test_training_still_requires_the_val_subdir(tmp_path: Path) -> None:
    """The other half: training really does read val/, so it must still fail."""
    data_root = tmp_path / "data_root"
    _create_default_data_root(data_root)
    shutil.rmtree(data_root / config_loading._DEFAULT_VAL_DATA_SUBDIR)

    cfg = _make_cfg(tmp_path, custom_overrides={"moving_frame_steps": 1})

    with pytest.raises(ValueError, match="validation data"):
        load_cfg_and_paths(cfg)
