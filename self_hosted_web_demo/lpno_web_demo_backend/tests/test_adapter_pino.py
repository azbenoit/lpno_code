from __future__ import annotations

from pathlib import Path

import pytest

from app.adapter_pino import PinoRuntimeAdapter


def test_find_hydra_config_path_prefers_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "trained_run"
    hydra = run_dir / ".hydra"
    hydra.mkdir(parents=True)
    config = hydra / "config.yaml"
    config.write_text("arch: {}\n", encoding="utf-8")

    found = PinoRuntimeAdapter._find_hydra_config_path(run_dir)

    assert found == config


def test_find_hydra_config_path_falls_back_to_parent_and_grandparent(tmp_path: Path) -> None:
    run_root = tmp_path / "trained_run"
    checkpoint_dir = run_root / "outputs" / "trained_run"
    checkpoint_dir.mkdir(parents=True)

    hydra = run_root / ".hydra"
    hydra.mkdir(parents=True)
    config = hydra / "config.yaml"
    config.write_text("arch: {}\n", encoding="utf-8")

    found = PinoRuntimeAdapter._find_hydra_config_path(checkpoint_dir)

    assert found == config


def test_find_hydra_config_path_returns_none_when_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "trained_run" / "outputs" / "trained_run"
    run_dir.mkdir(parents=True)

    found = PinoRuntimeAdapter._find_hydra_config_path(run_dir)

    assert found is None


def test_prepare_cfg_and_template_reports_all_fallback_locations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = PinoRuntimeAdapter()
    run_dir = tmp_path / "trained_run" / "outputs" / "trained_run"
    run_dir.mkdir(parents=True)
    sim_params = tmp_path / "flow3D_params.yaml"
    sim_params.write_text("geometry: {}\nref_scales: {}\n", encoding="utf-8")

    class DummySettings:
        runtime_run_dir = str(run_dir)
        runtime_sim_params_path = str(sim_params)

    imports = {
        "yaml": object(),
        "OmegaConf": object(),
        "coerce_numbers": lambda value: value,
        "register_custom_hydra_inputs": lambda: None,
    }

    with pytest.raises(FileNotFoundError, match="Hydra config not found under"):
        adapter._prepare_cfg_and_template(
            settings=DummySettings(), imports=imports, run_dir=run_dir, sim_params_path=sim_params
        )
