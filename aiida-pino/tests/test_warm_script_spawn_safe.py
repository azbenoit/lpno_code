"""The warm job's script is the `__main__` of a process that starts a SPAWN pool.

Every spawned child re-imports `__main__`. A script whose work sits at module level therefore
runs the export AGAIN in each child, and the children then block on the `.materialise.lock`
flock their parent holds while it waits for them -- a deadlock reproduced twice on 2026-09-14
(self-improve, /proc: parent in futex wait holding the lock, both spawn children in
locks_lock_inode_wait on the same file). Importing the script must run nothing.
"""

from __future__ import annotations

import runpy
import sys
import types

import pytest

from aiida_pino.calculations.warm_runtime_cache import warm_script


def _fake_aiida(monkeypatch, calls):
    aiida = types.ModuleType("aiida")
    aiida.load_profile = lambda *a, **k: calls.append("load_profile")
    aiida.orm = types.SimpleNamespace(load_node=lambda uuid: f"node:{uuid}")
    base = types.ModuleType("aiida_pino.calculations.base")

    def warm(node, runtime_fields=None, transport_type=None):
        calls.append(("warm", node, runtime_fields, transport_type))
        return "/cache/x"

    base.warm_dataset_runtime_cache = warm
    monkeypatch.setitem(sys.modules, "aiida", aiida)
    monkeypatch.setitem(sys.modules, "aiida_pino.calculations.base", base)


def test_importing_the_script_as_a_spawn_child_runs_no_export(tmp_path, monkeypatch):
    calls = []
    _fake_aiida(monkeypatch, calls)
    script = tmp_path / "warm_runtime_cache.py"
    script.write_text(warm_script("abc-uuid", ["T"], "warm_result.json"))
    runpy.run_path(str(script), run_name="__mp_main__")  # what multiprocessing's spawn does
    assert calls == []
    assert not (tmp_path / "warm_result.json").exists()


def test_running_the_script_as_main_does_the_export_once(tmp_path, monkeypatch):
    calls = []
    _fake_aiida(monkeypatch, calls)
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "warm_runtime_cache.py"
    script.write_text(warm_script("abc-uuid", None, "warm_result.json"))
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(script), run_name="__main__")
    assert info.value.code == 0
    assert calls == ["load_profile", ("warm", "node:abc-uuid", None, "core.local")]
    assert (tmp_path / "warm_result.json").read_text().strip() == '{"cache": "/cache/x"}'


def test_a_failing_export_exits_nonzero_without_the_marker(tmp_path, monkeypatch):
    calls = []
    _fake_aiida(monkeypatch, calls)
    sys.modules["aiida_pino.calculations.base"].warm_dataset_runtime_cache = lambda *a, **k: 1 / 0
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "warm_runtime_cache.py"
    script.write_text(warm_script("abc-uuid", None, "warm_result.json"))
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(script), run_name="__main__")
    assert info.value.code == 1
    assert not (tmp_path / "warm_result.json").exists()
