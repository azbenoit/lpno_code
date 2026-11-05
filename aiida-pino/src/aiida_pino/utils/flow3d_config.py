"""Front-door checks for a flow3D_scripts run config, at submit time.

`submit_flow3d_generate` validated `run_config` for truthiness and stored the
dict verbatim, while `execution_profile` and `dataset_profile` next to it reject
unknown keys. A config of `{"schema_version": 1}` passed every submit-time
check — and the console's own form shipped `schema_version: 3`, which
`run_rectangular_case.py` rejects outright ("schema_version must be exactly 1").
Clicking Submit on the untouched form therefore created a workchain and a
calcjob that could not succeed; with `prepare_only` off it queued a malformed
case behind the seat, days deep, to fail when it reached the front.

These checks mirror the runner's own front door
(`flow3D_scripts/python/run_rectangular_case.py`, `load_config`). They are
deliberately shallow — the runner stays the authority on physics and units — and
only catch what makes a run pointless before it costs a queue slot.
"""

from __future__ import annotations

import re
from typing import Any

# run_rectangular_case.py: CASE_NAME
_CASE_NAME = re.compile(r"[A-Za-z0-9_-]+")
_REQUIRED_SECTIONS = ("geometry", "mesh", "laser", "simulation")


def default_runner_root() -> str:
    """The flow3D_scripts checkout to use when nobody named one.

    The sibling checkout, which is what 1175 of 1178 `Flow3DRunCalculation`s
    pin. Left to itself the adapter falls through to the vendored
    `external/flow3D_scripts` submodule, pinned behind master, which *ignores*
    `recoil_model` and `void_vapor_pressure_pa` instead of rejecting them — so a
    config asking for calibrated physics renders pre-calibration physics, and
    the first sign is exit 410 on every child minutes after a clean-looking
    submit. That cost one batch in full.

    Mirrors `scripts/flow3d_run_adapter.resolve_runner_root`, which cannot
    import this: the adapter is uploaded and executed standalone.
    """
    from pathlib import Path

    try:
        import aiida_pino
    except ImportError:  # pragma: no cover
        return ""
    package_file = getattr(aiida_pino, "__file__", None)
    if not package_file:
        return ""
    sibling = Path(package_file).resolve().parents[2].parent / "flow3D_scripts"
    return str(sibling) if (sibling / "python" / "run_rectangular_case.py").is_file() else ""


def validate_with_runner(config: Any, runner_root: Any, *, where: str = "run_config") -> None:
    """Run the runner's OWN front door over this config, in-process.

    The shallow checks below are a copy of the runner's rules, and a copy
    drifts: measured 2026-08-05, eight configs the runner refuses in under
    1.3 ms were accepted here — a missing material file, a mesh coarser than the
    beam, an end point off the plate, negative power, T0 = 0 K,
    `recoil_model: 7`, a finish time shorter than the laser travel. Each cost a
    round trip of minutes to learn, ending in "runner exited with code 2".

    `load_and_validate` is pure: it reads the config and the material JSON and
    returns `(case_name, replacements, manifest)` without writing anything or
    touching a licence. Importing the runner by path keeps ONE authority for
    what a legal config is, instead of two that agree until they do not.

    Best effort by design: if the runner cannot be imported (missing checkout,
    incompatible version) the caller still has the shallow checks.
    """
    import importlib.util
    import json
    import sys
    import tempfile
    from pathlib import Path

    runner = Path(runner_root) / "python" / "run_rectangular_case.py"
    if not runner.is_file():
        return
    try:
        spec = importlib.util.spec_from_file_location("_flow3d_runner_validation", runner)
        module = importlib.util.module_from_spec(spec)
        # The runner imports siblings from its own package directory.
        sys.path.insert(0, str(runner.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(runner.parent))
        load_and_validate = module.load_and_validate
        config_error = getattr(module, "ConfigError", ValueError)
    except Exception:  # noqa: BLE001 - the runner is the authority, not a dependency
        return

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        try:
            load_and_validate(path, None)
        except config_error as exc:
            raise ValueError(f"{where}: {exc}") from exc
        except Exception:  # noqa: BLE001 - anything else is the runner's problem, not the user's
            return


def validate_run_config(config: Any, *, where: str = "run_config") -> None:
    """Raise ``ValueError`` when the runner would reject this config on sight."""
    if not isinstance(config, dict):
        raise ValueError(f"{where} must be a JSON object (a full flow3D_scripts config)")
    version = config.get("schema_version")
    if version != 1:
        raise ValueError(
            f"{where}.schema_version must be exactly 1 (got {version!r}); "
            "flow3D_scripts rejects anything else before the solve starts"
        )
    case_name = config.get("case_name")
    if case_name is not None and not (isinstance(case_name, str) and _CASE_NAME.fullmatch(case_name)):
        raise ValueError(f"{where}.case_name must use letters, digits, _ and - only (got {case_name!r})")
    missing = [name for name in _REQUIRED_SECTIONS if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(
            f"{where} is missing required section(s): {', '.join(missing)}. "
            "A run config needs geometry, mesh, laser and simulation objects."
        )
