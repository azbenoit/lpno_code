"""A runner that would ignore the config's physics must refuse, not render.

Round 5's blocker: the console left `flow3d_scripts_root` unset, so its runs
fell through to the bundled `external/flow3D_scripts` submodule — pinned at
2026-07-01, a month before the calibration campaign. That checkout does not
reject `recoil_model` / `void_vapor_pressure_pa`; it **ignores** them, rendering
`pv0=50650` where the config asked for `101325`. The run succeeds, the data
looks fine, and it is pre-calibration physics filed beside calibrated data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiida_pino.scripts.flow3d_run_adapter import (
    RUNNER_RELATIVE,
    assert_runner_supports_config,
    runner_identity,
    runner_supports,
)

CALIBRATED = {
    "simulation": {"recoil_model": 2, "void_vapor_pressure_pa": 101325.0},
    "laser": {"fresnel_from_material": True},
}


def _fake_runner(root: Path, *, supports: bool) -> Path:
    (root / RUNNER_RELATIVE.parent).mkdir(parents=True, exist_ok=True)
    body = "def main():\n    pass\n"
    if supports:
        body += "# handles recoil_model, void_vapor_pressure_pa, fresnel_from_material\n"
    (root / RUNNER_RELATIVE).write_text(body, encoding="utf-8")
    return root


def test_a_runner_missing_the_keys_is_refused(tmp_path):
    stale = _fake_runner(tmp_path / "stale", supports=False)
    with pytest.raises(ValueError) as excinfo:
        assert_runner_supports_config(stale, CALIBRATED, source="external/flow3D_scripts submodule")
    message = str(excinfo.value)
    # The message must name every unsupported key, where the runner came from,
    # and what to do — this failure is otherwise invisible.
    for key in ("simulation.recoil_model", "simulation.void_vapor_pressure_pa", "laser.fresnel_from_material"):
        assert key in message
    assert "IGNORE" in message and "external/flow3D_scripts submodule" in message
    assert "F3D_SCRIPTS_ROOT" in message


def test_a_current_runner_passes(tmp_path):
    current = _fake_runner(tmp_path / "current", supports=True)
    assert_runner_supports_config(current, CALIBRATED)  # must not raise


def test_a_config_without_calibrated_keys_is_unaffected(tmp_path):
    """Legacy configs must keep running on whatever runner resolves."""
    stale = _fake_runner(tmp_path / "stale", supports=False)
    assert_runner_supports_config(stale, {"simulation": {"finish_time_s": 1e-3}, "laser": {"power_w": 300}})


def test_runner_identity_records_what_rendered_the_case(tmp_path):
    """The summary recorded aiida_pino and lpno shas but not this."""
    current = _fake_runner(tmp_path / "current", supports=True)
    identity = runner_identity(current, "F3D_SCRIPTS_ROOT")
    assert identity["root"] == str(current)
    assert identity["resolved_from"] == "F3D_SCRIPTS_ROOT"
    assert identity["supports"] == {
        "fresnel_from_material": True,
        "recoil_model": True,
        "void_vapor_pressure_pa": True,
    }


def test_runner_supports_is_false_for_a_missing_runner(tmp_path):
    assert runner_supports(tmp_path / "nowhere", "recoil_model") is False
