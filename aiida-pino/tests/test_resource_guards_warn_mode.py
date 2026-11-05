"""Memory and disk guards are one switch: enforce here, warn in the public export.

Since 2026-09-14 the open-source export turns the memory and disk guards into warnings:
they complicate use, and memory limits are hardware dependent. The thresholds are the
original deployment's measurements; the code keeps them,
but `AIIDA_PINO_RESOURCE_GUARDS=warn` (the export's default, rewritten at assembly) turns
every refusal and clamp into a printed warning and proceeds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiida_pino.utils import guards
from aiida_pino.utils.export_pool import EXPORT_WORKERS_MAX, export_worker_count

GB = 1e9


def test_the_default_is_one_literal_the_export_can_rewrite():
    """The default is one literal, so a site that wants `enforce` changes one line; this pins
    the shape, not the value."""
    src = open(guards.__file__, encoding="utf-8").read()
    assert src.count("RESOURCE_GUARDS_DEFAULT = ") == 1
    assert guards.RESOURCE_GUARDS_DEFAULT in ("enforce", "warn")


def test_enforce_raises_and_warn_prints_then_proceeds(monkeypatch, capsys):
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "enforce")
    with pytest.raises(ValueError, match="too small"):
        guards.resource_guard("too small", exc=ValueError)
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "warn")
    assert guards.resource_guard("too small", exc=ValueError) is True
    out = capsys.readouterr().out
    assert "WARNING" in out and "too small" in out and "AIIDA_PINO_RESOURCE_GUARDS=enforce" in out


def test_an_unknown_mode_falls_back_to_the_default(monkeypatch, capsys):
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "maybe")
    assert guards.resource_guard_mode() == guards.RESOURCE_GUARDS_DEFAULT
    assert "maybe" in capsys.readouterr().out


def test_export_workers_are_clamped_by_memory_only_when_enforced(monkeypatch, capsys):
    monkeypatch.delenv("AIIDA_PINO_EXPORT_WORKERS", raising=False)
    kw = dict(cpu_count=20, n_members=61, per_member_bytes=23 * GB, available_bytes=101 * GB)
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "enforce")
    assert export_worker_count(**kw) == 3  # (101 - 20) // 23
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "warn")
    assert export_worker_count(**kw) == min(EXPORT_WORKERS_MAX, 20 - 4)
    out = capsys.readouterr().out
    assert "WARNING" in out and "AIIDA_PINO_EXPORT_WORKERS" in out and "23 GB per member" in out
    # An explicit override is honoured as written in warn mode, with the same warning.
    monkeypatch.setenv("AIIDA_PINO_EXPORT_WORKERS", "6")
    assert export_worker_count(**kw) == 6
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "enforce")
    assert export_worker_count(**kw) == 3


def test_the_disk_fit_check_warns_and_returns_at_once_in_warn_mode(monkeypatch, tmp_path, capsys):
    from aiida_pino.utils import runtime_cache as rc

    monkeypatch.setattr(rc, "estimated_derived_bytes", lambda members, profile: 10**15)  # a petabyte
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "warn")
    rc.refuse_build_that_cannot_fit(tmp_path, [SimpleNamespace()], {}, label="ds1")  # no raise, no wait
    assert "may fill the disk" in capsys.readouterr().out
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "enforce")
    monkeypatch.setattr(rc, "RUNTIME_BUILD_WAIT_S", 0)
    with pytest.raises(RuntimeError, match="refusing to materialise"):
        rc.refuse_build_that_cannot_fit(tmp_path, [SimpleNamespace()], {}, label="ds1")


def test_the_host_ram_floor_warns_instead_of_rejecting_in_warn_mode(monkeypatch, capsys):
    from aiida_pino.utils.profiles import DEFAULT_TRAINING_PROFILE, validate_training_profile

    too_small = {**DEFAULT_TRAINING_PROFILE, "host_ram_mb": 1024,
                 "env": {**(DEFAULT_TRAINING_PROFILE.get("env") or {}), "PINO_LOADER_WORKERS": "4"}}
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "enforce")
    with pytest.raises(ValueError, match="host_ram_mb=1024 is too small"):
        validate_training_profile(too_small)
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "warn")
    validated = validate_training_profile(too_small)
    assert validated["host_ram_mb"] == 1024
    assert "host_ram_mb=1024 is too small" in capsys.readouterr().out
