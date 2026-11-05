from importlib import resources

from lpno.models.argv_custom_remap import remap_plus_args_to_custom


def test_remap_known_custom_key_removes_plus() -> None:
    args = ["+name=run1"]

    remapped = remap_plus_args_to_custom(args, default_config_name="config_NS_energy_PINO")

    assert remapped == ["custom.name=run1"]


def test_remap_known_nested_custom_key_removes_plus() -> None:
    args = ["+material_params.T_s=1900"]

    remapped = remap_plus_args_to_custom(args, default_config_name="config_NS_energy_PINO")

    assert remapped == ["custom.material_params.T_s=1900"]


def test_remap_unknown_key_keeps_plus_on_custom_prefix() -> None:
    args = ["+unknown_key=true"]

    remapped = remap_plus_args_to_custom(args, default_config_name="config_NS_energy_PINO")

    assert remapped == ["+custom.unknown_key=true"]


def test_remap_custom_prefixed_key_obeys_declared_rule() -> None:
    args = ["+custom.time_as_batch=true"]

    remapped = remap_plus_args_to_custom(args, default_config_name="config_NS_energy_PINO")

    assert remapped == ["custom.time_as_batch=true"]


def test_remap_legacy_time_key_is_not_declared_anymore() -> None:
    args = ["+custom.infer_time=true"]

    remapped = remap_plus_args_to_custom(args, default_config_name="config_NS_energy_PINO")

    assert remapped == ["+custom.infer_time=true"]


def test_remap_respects_selected_config_name() -> None:
    args = ["--config-name=config_NS_energy_PINO_abaqus", "+laser.eta=0.2"]

    remapped = remap_plus_args_to_custom(args, default_config_name="config_NS_energy_PINO")

    assert remapped == ["--config-name=config_NS_energy_PINO_abaqus", "custom.laser.eta=0.2"]


def test_remap_legacy_sim_params_to_custom_plus_form() -> None:
    args = ["+sim_params=src/conf/sim_params/flow3D_params.yaml"]

    remapped = remap_plus_args_to_custom(args, default_config_name="config_NS_energy_PINO")

    assert remapped == ["+custom.sim_params=src/conf/sim_params/flow3D_params.yaml"]


def test_packaged_hydra_config_is_discoverable() -> None:
    cfg_path = resources.files("lpno").joinpath("conf", "hydra", "config_NS_energy_PINO.yaml")
    assert cfg_path.is_file()
