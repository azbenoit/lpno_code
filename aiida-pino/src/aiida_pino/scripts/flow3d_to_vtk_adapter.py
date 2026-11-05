"""Standalone adapter for Flow3D flask to VTK conversion."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
from pathlib import Path

try:
    from aiida_pino.scripts.live_tail import run_with_live_tail
except ImportError:
    _LIVE_TAIL_SPEC = importlib.util.spec_from_file_location("aiida_pino_live_tail", Path(__file__).with_name("live_tail.py"))
    if _LIVE_TAIL_SPEC is None or _LIVE_TAIL_SPEC.loader is None:
        raise
    _LIVE_TAIL_MODULE = importlib.util.module_from_spec(_LIVE_TAIL_SPEC)
    _LIVE_TAIL_SPEC.loader.exec_module(_LIVE_TAIL_MODULE)
    run_with_live_tail = _LIVE_TAIL_MODULE.run_with_live_tail

PACKAGE_INSTALL_HINT = (
    "`lpno` is not installed in this Code environment. "
    "Install the sibling checkout with `pip install -e /path/to/lpno`."
)


def load_flow3d_functions(script_path: Path) -> tuple:
    """Load selected definitions from the existing non-CLI-safe Flow3D script."""
    source = script_path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(script_path))
    selected = []
    for node in parsed.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "save_fluid_volume":
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace: dict = {}
    exec(compile(module, str(script_path), "exec"), namespace, namespace)
    return namespace["save_fluid_volume"], re.compile(r"^(?P<P>\d{2,3})W_(?P<V>\d{1,4})mps_line_track$")


def resolve_flow3d_script_path() -> Path:
    """Resolve the installed Flow3D converter source file without importing it."""
    spec = importlib.util.find_spec("lpno.custom_utils.Flow3D_flask_to_vtk")
    if spec is None or spec.origin is None:
        raise SystemExit(PACKAGE_INSTALL_HINT)
    return Path(spec.origin).resolve()


def legacy_output_name(case_name: str, case_match: re.Match[str] | None, legacy_case_re: re.Pattern[str]) -> str:
    """Produce the current PINO VTK basename."""
    match = legacy_case_re.match(case_name)
    if not match:
        return case_name
    power = match.group("P")
    velocity_raw = match.group("V")
    velocity_formatted = f"{velocity_raw}p0" if len(velocity_raw) == 1 else f"{velocity_raw[0]}p{velocity_raw[1:]}"
    return f"results_V_scan_{velocity_formatted}_laser_power_{power}"


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-file", required=True)
    args = parser.parse_args()

    source_path = Path(args.source_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    profile = json.loads(Path(args.profile_file).read_text(encoding="utf-8"))

    converted: list[dict] = []
    skipped: list[str] = []
    failed: list[dict] = []
    try:
        missing_keys = sorted(key for key in ("case_regex", "num_timesteps", "output_basename_mode") if key not in profile)
        if missing_keys:
            raise ValueError(f"Conversion profile is missing required keys: {', '.join(missing_keys)}")
        flow3d_script = resolve_flow3d_script_path()

        save_fluid_volume, legacy_case_re = load_flow3d_functions(flow3d_script)
        case_regex = re.compile(str(profile["case_regex"]))
        num_timesteps = int(profile["num_timesteps"])
        output_mode = str(profile["output_basename_mode"])

        output_dir.mkdir(parents=True, exist_ok=True)

        for case_path in sorted(path for path in source_path.iterdir() if path.is_dir()):
            match = case_regex.match(case_path.name)
            if not match:
                skipped.append(case_path.name)
                continue

            input_file = case_path / f"flsgrf1.{case_path.name}"
            if not input_file.exists():
                skipped.append(case_path.name)
                continue

            if output_mode == "case_name":
                output_basename = case_path.name
            else:
                output_basename = legacy_output_name(case_path.name, match, legacy_case_re)

            case_output_dir = output_dir / case_path.name if num_timesteps > 1 else output_dir
            case_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                save_fluid_volume(
                    home_dir=str(source_path),
                    case_name=case_path.name,
                    output_dir=str(case_output_dir),
                    output_filename=output_basename,
                    num_timesteps=num_timesteps,
                )
            except Exception as exc:  # noqa: BLE001 - isolate one bad case from the batch
                failed.append({"case_name": case_path.name, "error": str(exc)})
                continue
            converted.append({"case_name": case_path.name, "output_dir": str(case_output_dir), "output_basename": output_basename})

        if not converted:
            skip_note = f" Skipped cases: {', '.join(skipped)}." if skipped else ""
            fail_note = f" Failed cases: {', '.join(entry['case_name'] for entry in failed)}." if failed else ""
            raise ValueError(
                f"Flow3D conversion produced zero converted cases under `{source_path}`.{skip_note}{fail_note} "
                "Check `case_regex`, the `flsgrf1.<case>` file naming, and the per-case errors."
            )

        summary = {
            "status": "ok",
            "source_path": str(source_path),
            "output_dir": str(output_dir),
            "profile": profile,
            "converted_cases": converted,
            "skipped_cases": skipped,
            "failed_cases": failed,
        }
        Path("flow3d_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        failure_summary = {
            "status": "error",
            "source_path": str(source_path),
            "output_dir": str(output_dir),
            "profile": profile,
            "converted_cases": converted,
            "skipped_cases": skipped,
            "failed_cases": failed,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        Path("flow3d_summary.json").write_text(json.dumps(failure_summary, indent=2, sort_keys=True), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(run_with_live_tail(_main))
