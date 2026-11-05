"""Command-construction tests for the GPU seat lock and FLOW-3D resource scope.

Pure argv tests: no GPU, no solver, no daemon. The gpu_spool seat prefix and
the systemd-run scope prefix are asserted on the built command lists only.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from aiida_pino.scripts import eval_adapter, flow3d_run_adapter, kfold_adapter, train_adapter
from aiida_pino.calculations.base import runtime_fields_from_training_profile
from aiida_pino.utils.profiles import load_training_profile_defaults, validate_flow3d_run_execution_profile

CWD = Path("/tmp/aiida_workdir")


def _training_profile(cuda_device: str = "1") -> dict:
    profile = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    profile["run_name"] = "run"
    profile["cuda_device"] = cuda_device
    return profile


def _fake_find_spec(available: bool):
    return lambda name: object() if available else None


# --- gpu_spool seat wrap (train / eval / kfold) --------------------------------


@pytest.mark.parametrize("adapter", [train_adapter, eval_adapter, kfold_adapter])
def test_seat_wrap_prefixes_gpu_spool_with_profile_gpu(monkeypatch, adapter) -> None:
    monkeypatch.setattr(adapter.importlib.util, "find_spec", _fake_find_spec(True))
    inner = [sys.executable, "-u", "-m", "some.module", "custom.name=run"]
    wrapped = adapter.wrap_with_gpu_seat(inner, _training_profile("1"), cwd=CWD)

    assert wrapped[:7] == [sys.executable, "-m", "lpno.gpu_spool", "seat", "--gpu", "1", "--"]
    # sh shim: logs the acquired seat and restores the job's working directory.
    assert wrapped[7:9] == ["sh", "-c"]
    expected_gpu = "$CUDA_VISIBLE_DEVICES" if adapter in (train_adapter, eval_adapter) else "1"
    assert f"[gpu-seat] acquired gpu {expected_gpu}" in wrapped[9]
    assert str(CWD) in wrapped[9]
    # The original command is passed through untouched behind the shim.
    assert wrapped[-len(inner):] == inner


def test_seat_wrap_defaults_to_gpu_zero(monkeypatch) -> None:
    monkeypatch.setattr(train_adapter.importlib.util, "find_spec", _fake_find_spec(True))
    for profile in ({}, {"cuda_device": ""}, {"cuda_device": None}):
        wrapped = train_adapter.wrap_with_gpu_seat(["cmd"], profile, cwd=CWD)
        assert wrapped[wrapped.index("--gpu") + 1] == "auto"


def test_train_command_keeps_cuda_device_override_matching_seat_gpu(monkeypatch) -> None:
    """The seat pins CUDA_VISIBLE_DEVICES, but the entrypoint re-routes by the
    physical index in custom.cuda_device, so the override must equal --gpu."""
    monkeypatch.setattr(train_adapter.importlib.util, "find_spec", _fake_find_spec(True))
    profile = _training_profile("1")
    command = train_adapter.build_train_command(profile, Path("/tmp/t"), Path("/tmp/v"), Path("/tmp/o"))
    wrapped = train_adapter.wrap_with_gpu_seat(command, profile, cwd=CWD)
    assert wrapped[wrapped.index("--gpu") + 1] == "1"
    assert "custom.cuda_device=1" in wrapped


def test_runtime_fields_parse_hydra_vtk_override() -> None:
    assert runtime_fields_from_training_profile(
        {"hydra_overrides": ['custom.vtk_fields=["Temperature","Fraction Of Fluid"]']}
    ) == ["Temperature", "Fraction Of Fluid"]


def test_eval_command_seat_gpu_matches_override(monkeypatch) -> None:
    monkeypatch.setattr(eval_adapter.importlib.util, "find_spec", _fake_find_spec(True))
    profile = _training_profile("0")
    command = eval_adapter.build_eval_command(profile, Path("/tmp/d"), Path("/tmp/t"))
    wrapped = eval_adapter.wrap_with_gpu_seat(command, profile, cwd=CWD)
    assert wrapped[wrapped.index("--gpu") + 1] == "0"
    assert "custom.cuda_device=0" in wrapped


@pytest.mark.parametrize("adapter", [train_adapter, eval_adapter, kfold_adapter])
def test_seat_wrap_falls_back_when_gpu_spool_missing(monkeypatch, capsys, adapter) -> None:
    monkeypatch.setattr(adapter.importlib.util, "find_spec", _fake_find_spec(False))
    inner = ["python", "-m", "some.module"]
    assert adapter.wrap_with_gpu_seat(inner, _training_profile("0"), cwd=CWD) == inner
    assert "WARNING" in capsys.readouterr().out


def test_seat_wrap_passes_multi_gpu_values_through_unwrapped(monkeypatch, capsys) -> None:
    monkeypatch.setattr(train_adapter.importlib.util, "find_spec", _fake_find_spec(True))
    inner = ["python", "-m", "some.module"]
    assert train_adapter.wrap_with_gpu_seat(inner, {"cuda_device": "0,1"}, cwd=CWD) == inner
    assert "WARNING" in capsys.readouterr().out


# --- FLOW-3D systemd resource scope --------------------------------------------


def test_resource_scope_prefix_computes_cpuquota_from_cores() -> None:
    command, state = flow3d_run_adapter.apply_resource_scope(
        ["python", "runner.py"], {"cores": 4}, probe=lambda: True
    )
    assert state == "active"
    assert command[:4] == ["systemd-run", "--user", "--scope", "--nice=10"]
    assert command[command.index("CPUQuota=400%") - 1] == "-p"
    assert command[command.index("--") + 1:] == ["python", "runner.py"]


def test_resource_scope_sets_no_hard_memory_cap_by_default() -> None:
    """A MemoryMax OOM-kills healthy long solves, so the CPU cap ships alone.

    flow3D_scripts' own batch wrappers drop the memory cap on purpose: measured
    2026-07-14, FLOW-3D's ~13 GB flsgrf write-back page cache is charged to the
    cgroup while anon RSS stays ~1.2 GB, so a 16G ceiling kills long runs that are
    behaving perfectly. CPUQuota is what actually protects interactive SSH.
    """
    command, state = flow3d_run_adapter.apply_resource_scope(["runner"], {"cores": 10}, probe=lambda: True)
    assert state == "active"
    assert "CPUQuota=1000%" in command
    assert not any(part.startswith("Memory") for part in command)


def test_resource_scope_memory_max_is_opt_in() -> None:
    command, _ = flow3d_run_adapter.apply_resource_scope(
        ["runner"], {"cores": 4, "memory_max": "16G"}, probe=lambda: True
    )
    assert "MemoryMax=16G" in command
    assert command[command.index("MemoryMax=16G") - 1] == "-p"


def test_resource_scope_defaults_to_the_core_cap() -> None:
    command, state = flow3d_run_adapter.apply_resource_scope(["runner"], {}, probe=lambda: True)
    assert state == "active"
    assert "CPUQuota=1000%" in command


def test_resource_scope_falls_back_when_probe_fails(capsys) -> None:
    command, state = flow3d_run_adapter.apply_resource_scope(["runner"], {"cores": 4}, probe=lambda: False)
    assert state == "unavailable"
    assert command == ["runner"]
    assert "WARNING" in capsys.readouterr().out


def test_resource_scope_false_disables_prefix() -> None:
    command, state = flow3d_run_adapter.apply_resource_scope(["runner"], {"resource_scope": False}, probe=lambda: True)
    assert state == "disabled"
    assert command == ["runner"]


def test_resource_scope_profile_key_is_validated() -> None:
    validate_flow3d_run_execution_profile({"resource_scope": False})
    validate_flow3d_run_execution_profile({"resource_scope": True, "cores": 4})
    with pytest.raises(ValueError, match="resource_scope"):
        validate_flow3d_run_execution_profile({"resource_scope": "yes"})


def test_main_records_scope_state_in_summary(monkeypatch, tmp_path: Path) -> None:
    import json

    root = tmp_path / "runner"
    (root / "python").mkdir(parents=True)
    (root / "python" / "run_rectangular_case.py").write_text("# stub\n", encoding="utf-8")
    out_root = tmp_path / "runs"
    recorded: list[list[str]] = []

    def fake_run(command, **kwargs):
        recorded.append(command)
        case_dir = out_root / "case_locks"
        case_dir.mkdir(parents=True)
        (case_dir / "case_manifest.json").write_text(json.dumps({"case_name": "case_locks"}), encoding="utf-8")

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "run_config.json").write_text("{}", encoding="utf-8")
    profile = {"flow3d_scripts_root": str(root), "out_root": str(out_root), "case_name": "case_locks", "cores": 2}
    (sandbox / "execution_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.chdir(sandbox)
    monkeypatch.setattr(flow3d_run_adapter, "run_streamed_subprocess", fake_run)
    monkeypatch.setattr(flow3d_run_adapter, "systemd_scope_available", lambda: True)
    monkeypatch.setattr(
        sys, "argv", ["flow3d_run_adapter.py", "--config-file", "run_config.json", "--profile-file", "execution_profile.json"]
    )
    flow3d_run_adapter._main()

    (command,) = recorded
    assert command[0] == "systemd-run"
    assert "CPUQuota=200%" in command
    summary = json.loads((sandbox / "flow3d_run_summary.json").read_text(encoding="utf-8"))
    assert summary["resource_scope"] == "active"


def test_flow3d_pause_gate_blocks_before_timeout_clock(tmp_path, monkeypatch, capsys):
    """The pause wait must block, so it can run before the `timeout` wrapper.

    Regression for 2026-07-27: the runner's own pause wait sits *inside*
    `timeout`, so a 40 h disk-guard pause burned the whole solve budget and
    killed cases mck_275..mck_294 minutes after they finally started solving.
    """
    import threading

    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path))
    sentinel = tmp_path / "PAUSED"
    sentinel.write_text("held for test", encoding="utf-8")

    done = threading.Event()

    def _wait():
        flow3d_run_adapter.wait_while_paused(poll_s=0.05)
        done.set()

    threading.Thread(target=_wait, daemon=True).start()
    assert not done.wait(0.4), "returned while PAUSED was still present"

    sentinel.unlink()
    assert done.wait(5), "did not return after PAUSED was cleared"
    assert "held for test" in capsys.readouterr().out


def test_flow3d_pause_gate_is_noop_without_sentinel(tmp_path, monkeypatch):
    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path))
    flow3d_run_adapter.wait_while_paused(poll_s=30.0)  # returns immediately


def test_flow3d_hold_seat_serializes_a_simultaneous_batch(tmp_path, monkeypatch):
    """Concurrent adapters must take the seat one at a time.

    Regression for 2026-07-27: the seat wait lived *inside* the `timeout` wrapper,
    so submitting 20 requeues at once started 20 clocks and the backlog died
    waiting. Acquiring before the wrapper makes a simultaneous batch serialize
    with no budget running.
    """
    import threading

    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path))
    order = []

    def worker(i):
        with flow3d_run_adapter.hold_seat():
            order.append(("in", i))
            time.sleep(0.2)
            order.append(("out", i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert len(order) == 6
    # strict in/out alternation == never two holders at once
    for i in range(0, len(order), 2):
        assert order[i][0] == "in" and order[i + 1][0] == "out", order
        assert order[i][1] == order[i + 1][1], order


def test_flow3d_hold_seat_releases_on_exit(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path))
    with flow3d_run_adapter.hold_seat():
        pass
    fd = os.open(tmp_path / ".seat.lock", os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
