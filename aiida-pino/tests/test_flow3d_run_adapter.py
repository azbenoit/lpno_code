"""Unit tests for the FLOW-3D run adapter (`flow3d_run_adapter.py`).

Pure filesystem tests with a monkeypatched subprocess launcher: no AiiDA
profile, no FLOW-3D, no license. The real subprocess path is covered by the
fake-runner integration test in `tests/test_workflows.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aiida_pino.scripts import flow3d_run_adapter as adapter
from aiida_pino.utils.profiles import validate_flow3d_run_execution_profile

CASE = "case_unit"


def _runner_root(root: Path) -> Path:
    (root / "python").mkdir(parents=True, exist_ok=True)
    (root / "python" / "run_rectangular_case.py").write_text("# stub\n", encoding="utf-8")
    return root


def _fabricate_run_dir(case_dir: Path, *, with_npz: bool = True, runhyd_lines: int = 300) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_manifest.json").write_text(
        json.dumps({"case_name": case_dir.name, "input": {"laser": {"power_w": 250.0}}}), encoding="utf-8"
    )
    (case_dir / "summary.md").write_text("# summary\n", encoding="utf-8")
    (case_dir / "path.png").write_bytes(b"\x89PNG fake")
    (case_dir / f"prepin.{case_dir.name}").write_text("prepin\n", encoding="utf-8")
    (case_dir / "runhyd.log").write_text("\n".join(f"line {i}" for i in range(runhyd_lines)) + "\n", encoding="utf-8")
    (case_dir / f"flsgrf1.{case_dir.name}").write_text("BIG", encoding="utf-8")
    if with_npz:
        (case_dir / "npz").mkdir(exist_ok=True)
        (case_dir / "npz" / f"{case_dir.name}.npz").write_bytes(b"npz payload")


def _run_main(monkeypatch, sandbox: Path, profile: dict, *, run_config: dict | None = None) -> None:
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "run_config.json").write_text(json.dumps(run_config or {"laser": {"power_w": 250.0}}), encoding="utf-8")
    (sandbox / "execution_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.chdir(sandbox)
    # Keep argv assertions deterministic: the systemd scope path has its own
    # tests in test_adapter_locks.py.
    monkeypatch.setattr(adapter, "systemd_scope_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["flow3d_run_adapter.py", "--config-file", "run_config.json", "--profile-file", "execution_profile.json"],
    )
    adapter._main()


# --- command construction -----------------------------------------------------


def test_command_construction_and_no_license_leakage(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")
    out_root = tmp_path / "runs"
    recorded: list[list[str]] = []

    def fake_run(command, **kwargs):
        recorded.append(command)
        _fabricate_run_dir(out_root / CASE)

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    monkeypatch.setenv("F3D_LICENSE", "2700@secret-host")
    profile = {
        "flow3d_scripts_root": str(root),
        "out_root": str(out_root),
        "case_name": CASE,
        "cores": 4,
        "threads_per_rank": 2,
        "overwrite": True,
        "export": "npz",
        "flow3d_env": "/opt/flow3d/flow3dvars.sh",
        "post_home": "/opt/flow3d/post",
    }
    validate_flow3d_run_execution_profile(profile)

    _run_main(monkeypatch, tmp_path / "sandbox", profile)

    (command,) = recorded
    assert command[0] == sys.executable
    assert command[1] == str(root / "python" / "run_rectangular_case.py")
    # The staged config is passed as an absolute path.
    assert command[2] == str(tmp_path / "sandbox" / "run_config.json")
    assert Path(command[2]).is_absolute()
    for flag, value in (
        ("--case-name", CASE),
        ("--out-root", str(out_root)),
        ("--export", "npz"),
        ("--cores", "4"),
        ("--threads-per-rank", "2"),
        ("--flow3d-env", "/opt/flow3d/flow3dvars.sh"),
        ("--post-home", "/opt/flow3d/post"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--overwrite" in command
    assert "--prepare-only" not in command
    # The license endpoint never reaches the command line or the summary.
    joined = " ".join(command)
    assert "license" not in joined.lower() and "secret-host" not in joined
    summary_text = (tmp_path / "sandbox" / "flow3d_run_summary.json").read_text(encoding="utf-8")
    assert "secret-host" not in summary_text


def test_defaults_out_root_and_export(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")
    recorded: list[list[str]] = []

    def fake_run(command, **kwargs):
        recorded.append(command)
        _fabricate_run_dir(root / "runs" / CASE)

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    _run_main(monkeypatch, tmp_path / "sandbox", {"flow3d_scripts_root": str(root), "case_name": CASE})

    (command,) = recorded
    assert command[command.index("--out-root") + 1] == str(root / "runs")
    assert command[command.index("--export") + 1] == "npz"


def test_keep_flsgrf_is_passed_through_and_off_by_default(monkeypatch, tmp_path: Path) -> None:
    # The runner deletes the multi-GB flsgrf once the NPZ export completes; AiiDA runs
    # must inherit that cleanup, and only opt out when a re-export needs the result.
    root = _runner_root(tmp_path / "runner")
    recorded: list[list[str]] = []

    def fake_run(command, **kwargs):
        recorded.append(command)
        _fabricate_run_dir(root / "runs" / CASE)

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    base = {"flow3d_scripts_root": str(root), "case_name": CASE}
    _run_main(monkeypatch, tmp_path / "default", dict(base))
    _run_main(monkeypatch, tmp_path / "kept", {**base, "keep_flsgrf": True})

    default_command, kept_command = recorded
    assert "--keep-flsgrf" not in default_command
    assert "--keep-flsgrf" in kept_command


def test_prepare_only_flag_and_summary(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")
    out_root = tmp_path / "runs"

    def fake_run(command, **kwargs):
        case_dir = out_root / CASE
        case_dir.mkdir(parents=True)
        (case_dir / "case_manifest.json").write_text(json.dumps({"case_name": CASE}), encoding="utf-8")
        (case_dir / "summary.md").write_text("# prepared\n", encoding="utf-8")

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    _run_main(
        monkeypatch,
        tmp_path / "sandbox",
        {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": CASE, "prepare_only": True},
    )

    summary = json.loads((tmp_path / "sandbox" / "flow3d_run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["prepare_only"] is True
    assert summary["npz_file"] is None
    assert summary["manifest"] == {"case_name": CASE}


# --- runner root resolution ----------------------------------------------------


def test_root_resolution_prefers_profile_over_env(monkeypatch, tmp_path: Path) -> None:
    profile_root = _runner_root(tmp_path / "profile_root")
    env_root = _runner_root(tmp_path / "env_root")
    monkeypatch.setenv("F3D_SCRIPTS_ROOT", str(env_root))
    assert adapter.resolve_runner_root({"flow3d_scripts_root": str(profile_root)}) == profile_root
    assert adapter.resolve_runner_root({}) == env_root


def test_root_resolution_falls_back_to_submodule(monkeypatch, tmp_path: Path) -> None:
    """Changed deliberately: the submodule is the LAST resort, not the first.

    This asserted that an unset root lands on `external/flow3D_scripts`. That is
    the behaviour that cost one batch in full: the vendored submodule is
    pinned behind master and *ignores* `recoil_model` and
    `void_vapor_pressure_pa` rather than rejecting them, so a config asking for
    calibrated physics rendered pre-calibration physics. The sibling checkout
    now comes first (11 occurrences of `recoil_model` against 0 in the
    submodule). The submodule is still reachable, which is what this now pins --
    with no sibling present, since this repo has a real one beside it.
    """
    monkeypatch.delenv("F3D_SCRIPTS_ROOT", raising=False)
    repo = tmp_path / "aiida-pino"
    submodule = repo / "external" / "flow3D_scripts"
    (submodule / adapter.RUNNER_RELATIVE.parent).mkdir(parents=True)
    (submodule / adapter.RUNNER_RELATIVE).write_text("# runner\n")
    monkeypatch.setattr(adapter, "_default_repo_root", lambda: repo)
    assert adapter.resolve_runner_root({}) == submodule.resolve()


def test_root_resolution_prefers_the_sibling_checkout(monkeypatch) -> None:
    """The real repo has a sibling checkout, so an unset root must land there."""
    monkeypatch.delenv("F3D_SCRIPTS_ROOT", raising=False)
    repo_root = Path(__file__).resolve().parents[1]
    sibling = repo_root.parent / "flow3D_scripts"
    if not (sibling / adapter.RUNNER_RELATIVE).is_file():
        pytest.skip("no sibling flow3D_scripts checkout on this machine")
    assert adapter.resolve_runner_root({}) == sibling.resolve()


def test_root_resolution_error_names_all_three_options(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("F3D_SCRIPTS_ROOT", raising=False)
    monkeypatch.setattr(adapter, "_default_repo_root", lambda: None)
    with pytest.raises(ValueError) as excinfo:
        adapter.resolve_runner_root({})
    message = str(excinfo.value)
    for expected in ("flow3d_scripts_root", "F3D_SCRIPTS_ROOT", "external/flow3D_scripts"):
        assert expected in message


def test_explicit_root_without_runner_script_is_a_hard_error(monkeypatch, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="run_rectangular_case.py"):
        adapter.resolve_runner_root({"flow3d_scripts_root": str(empty)})


# --- artifact curation ----------------------------------------------------------


def test_artifact_curation_small_files_only(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")
    out_root = tmp_path / "runs"
    monkeypatch.setattr(
        adapter, "run_streamed_subprocess", lambda command, **kwargs: _fabricate_run_dir(out_root / CASE)
    )
    _run_main(
        monkeypatch,
        tmp_path / "sandbox",
        {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": CASE},
    )

    artifacts = tmp_path / "sandbox" / "run_artifacts"
    names = sorted(p.name for p in artifacts.iterdir())
    assert names == sorted(["case_manifest.json", "summary.md", "path.png", f"prepin.{CASE}", "runhyd_tail.log"])
    # The tail keeps only the last 200 lines of the 300-line log.
    tail_lines = (artifacts / "runhyd_tail.log").read_text(encoding="utf-8").splitlines()
    assert len(tail_lines) == 200
    assert tail_lines[0] == "line 100"
    # Big artifacts stay on disk under out_root and are referenced by path.
    summary = json.loads((tmp_path / "sandbox" / "flow3d_run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["returncode"] == 0
    assert summary["case_dir"] == str(out_root / CASE)
    assert summary["npz_file"] == str(out_root / CASE / "npz" / f"{CASE}.npz")
    assert summary["manifest"]["input"]["laser"]["power_w"] == 250.0
    assert (out_root / CASE / f"flsgrf1.{CASE}").is_file()


# --- failure path -----------------------------------------------------------------


def test_runner_failure_writes_error_summary_and_reraises(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")

    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        _run_main(
            monkeypatch,
            tmp_path / "sandbox",
            {"flow3d_scripts_root": str(root), "case_name": CASE, "out_root": str(tmp_path / "runs")},
        )

    summary = json.loads((tmp_path / "sandbox" / "flow3d_run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["returncode"] == 2
    assert "exited with code 2" in summary["error_message"]


# --- execution profile validation ---------------------------------------------------


def test_execution_profile_rejects_license_and_unknown_keys() -> None:
    with pytest.raises(ValueError, match="license"):
        validate_flow3d_run_execution_profile({"license": "2700@host"})
    with pytest.raises(ValueError, match="license"):
        validate_flow3d_run_execution_profile({"F3D_LICENSE_ENDPOINT": "2700@host"})
    with pytest.raises(ValueError, match="unknown keys"):
        validate_flow3d_run_execution_profile({"case": "typo"})
    with pytest.raises(ValueError, match="cores"):
        validate_flow3d_run_execution_profile({"cores": 0})
    with pytest.raises(ValueError, match="export"):
        validate_flow3d_run_execution_profile({"export": "hdf5"})
    validate_flow3d_run_execution_profile({})
    validate_flow3d_run_execution_profile({"case_name": "a", "cores": 2, "export": "vtk", "prepare_only": True})


# --- watchdog and wall timeout ---------------------------------------------------


def test_execution_profile_validates_watchdog_keys() -> None:
    validate_flow3d_run_execution_profile({"timeout_s": 7200, "watchdog": True, "stall_s": 600.0})
    with pytest.raises(ValueError, match="timeout_s"):
        validate_flow3d_run_execution_profile({"timeout_s": 0})
    with pytest.raises(ValueError, match="stall_s"):
        validate_flow3d_run_execution_profile({"stall_s": -1})
    with pytest.raises(ValueError, match="watchdog"):
        validate_flow3d_run_execution_profile({"watchdog": "yes"})


def test_timeout_prefix_wraps_runner_command(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")
    out_root = tmp_path / "runs"
    recorded: list[list[str]] = []

    def fake_run(command, **kwargs):
        recorded.append(command)
        _fabricate_run_dir(out_root / CASE)

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    _run_main(
        monkeypatch,
        tmp_path / "sandbox",
        {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": CASE, "timeout_s": 7200},
    )

    (command,) = recorded
    assert command[0] == "timeout"
    assert command[1] == f"--kill-after={adapter.WATCHDOG_KILL_GRACE_S}"
    assert command[2] == "7200"
    assert command[3] == sys.executable


def test_wall_timeout_return_code_is_labeled(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")

    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(124, command)

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        _run_main(
            monkeypatch,
            tmp_path / "sandbox",
            {"flow3d_scripts_root": str(root), "case_name": CASE, "out_root": str(tmp_path / "runs"), "timeout_s": 60},
        )

    summary = json.loads((tmp_path / "sandbox" / "flow3d_run_summary.json").read_text(encoding="utf-8"))
    assert "wall timeout" in summary["error_message"]


def test_monitor_sidecar_spawned_with_runner_pid_and_hd3msg(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")
    (root / "python" / "monitor_run.py").write_text("# stub monitor\n", encoding="utf-8")
    out_root = tmp_path / "runs"
    seen: dict = {}

    def fake_launch(command, monitor_args):
        seen["command"] = command
        seen["monitor"] = monitor_args(4242)
        _fabricate_run_dir(out_root / CASE)

    monkeypatch.setattr(adapter, "_launch_with_watchdog", fake_launch)
    _run_main(
        monkeypatch,
        tmp_path / "sandbox",
        {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": CASE, "stall_s": 450},
    )

    monitor = seen["monitor"]
    assert monitor[0] == sys.executable
    assert monitor[1] == str(root / "python" / "monitor_run.py")
    assert monitor[2] == str(out_root / CASE / f"hd3msg.{CASE}")
    assert monitor[monitor.index("--until-pid") + 1] == "4242"
    assert monitor[monitor.index("--stall-s") + 1] == "450.0"


def test_watchdog_skipped_for_prepare_only_and_optout(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")
    (root / "python" / "monitor_run.py").write_text("# stub monitor\n", encoding="utf-8")
    out_root = tmp_path / "runs"
    recorded: list[list[str]] = []

    def fake_run(command, **kwargs):
        recorded.append(command)
        case_dir = out_root / CASE
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case_manifest.json").write_text(json.dumps({"case_name": CASE}), encoding="utf-8")

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    # prepare_only: no monitor even though monitor_run.py exists.
    _run_main(
        monkeypatch,
        tmp_path / "sandbox_a",
        {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": CASE, "prepare_only": True},
    )
    # watchdog=False: explicit opt-out on a real solve.
    _run_main(
        monkeypatch,
        tmp_path / "sandbox_b",
        {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": CASE, "watchdog": False, "overwrite": True},
    )
    assert len(recorded) == 2


def test_stale_progress_files_removed_before_watched_solve(monkeypatch, tmp_path: Path) -> None:
    root = _runner_root(tmp_path / "runner")
    (root / "python" / "monitor_run.py").write_text("# stub monitor\n", encoding="utf-8")
    out_root = tmp_path / "runs"
    case_dir = out_root / CASE
    case_dir.mkdir(parents=True)
    stale_hd3 = case_dir / f"hd3msg.{CASE}"
    stale_log = case_dir / "runhyd.log"
    stale_hd3.write_text("old progress\n", encoding="utf-8")
    stale_log.write_text("old log\n", encoding="utf-8")

    def fake_launch(command, monitor_args):
        # By launch time the previous attempt's progress files must be gone,
        # or the monitor anchors its stall baseline on stale data.
        assert not stale_hd3.exists()
        assert not stale_log.exists()
        _fabricate_run_dir(case_dir)

    monkeypatch.setattr(adapter, "_launch_with_watchdog", fake_launch)
    _run_main(
        monkeypatch,
        tmp_path / "sandbox",
        {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": CASE, "overwrite": True},
    )


def test_runner_env_is_sanitized_but_license_survives(monkeypatch, tmp_path: Path) -> None:
    # A daemon started from a conda shell leaks PYTHONPATH/LD_LIBRARY_PATH into
    # the runner, and FLOW-3D POST's embedded pvpython then imports the wrong
    # vtk bindings and aborts (alsi_c000, 2026-07-20). The adapter must strip
    # interpreter-poisoning variables while leaving license config untouched.
    root = _runner_root(tmp_path / "runner")
    out_root = tmp_path / "runs"
    seen: dict = {}

    def fake_run(command, **kwargs):
        seen["env"] = kwargs.get("env")
        _fabricate_run_dir(out_root / CASE)

    monkeypatch.setattr(adapter, "run_streamed_subprocess", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/some/conda/site-packages")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/some/openfoam/lib")
    monkeypatch.setenv("F3D_LICENSE", "2700@secret-host")
    _run_main(
        monkeypatch,
        tmp_path / "sandbox",
        {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": CASE},
    )

    env = seen["env"]
    assert env is not None
    for poison in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        assert poison not in env
    assert env["F3D_LICENSE"] == "2700@secret-host"


def test_sanitized_env_helper_only_removes_poison_vars(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/x")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LM_LICENSE_FILE", "2700@host")
    env = adapter.sanitized_runner_env()
    assert "PYTHONPATH" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["LM_LICENSE_FILE"] == "2700@host"


def test_launch_with_watchdog_streams_and_raises_on_failure(capsys) -> None:
    idle_monitor = lambda pid: [sys.executable, "-c", "import time; time.sleep(60)"]  # noqa: E731
    adapter._launch_with_watchdog([sys.executable, "-c", "print('runner-out')"], idle_monitor)
    assert "runner-out" in capsys.readouterr().out

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        adapter._launch_with_watchdog([sys.executable, "-c", "raise SystemExit(5)"], idle_monitor)
    assert excinfo.value.returncode == 5
