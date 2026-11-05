"""Admission must leave cores for the OS, not just VRAM and host RAM.

A shared machine should keep >=4 cores and >=20 GB free for the OS and interactive
logins; a heavy job stack that leaves nothing for them can make the machine unreachable. gpu_spool enforced the memory half (VRAM budget, host-RAM ledger)
and nothing at all on CPU -- which only stayed safe by accident, because an over-booked
host-RAM ledger happened to cap concurrency too.

Right-sizing the RAM declarations on 2026-08-20 (24576 -> 8192 for arms measured at
2.6 GiB) removed that accident: bookings stopped being the limit, and concurrency became
free to climb until the cores ran out. Measured that evening: a FLOW-3D solve at 9.9 cores
plus four trainings at ~1.05 each = 14.1 of 20, with the queue still admitting.

The gate is a poll-loop refusal, not a hard failure: `seat` retries, so a busy box delays
a run rather than killing it (the flow3D_scripts "disk floor must WAIT, not refuse" rule).
"""
from __future__ import annotations

import pytest

from lpno import gpu_spool


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # conftest disables the gate repo-wide so no test depends on the box's load; this
    # file is the one that exercises it, so it opts back in to the real default.
    monkeypatch.delenv("PINO_SPOOL_MIN_FREE_CORES", raising=False)


def test_default_reserve_matches_the_documented_floor():
    assert gpu_spool.min_free_cores() == 4


def test_env_overrides_the_floor(monkeypatch):
    monkeypatch.setenv("PINO_SPOOL_MIN_FREE_CORES", "2")
    assert gpu_spool.min_free_cores() == 2


def test_zero_disables_the_gate(monkeypatch):
    monkeypatch.setenv("PINO_SPOOL_MIN_FREE_CORES", "0")
    assert gpu_spool.min_free_cores() == 0
    assert gpu_spool.cpu_headroom_ok(load=19.0, cores=20) is True


def test_a_loaded_box_is_refused():
    # 20 cores, load 17 -> 3 free, under the floor of 4
    assert gpu_spool.cpu_headroom_ok(load=17.0, cores=20) is False


def test_an_idle_box_is_admitted():
    assert gpu_spool.cpu_headroom_ok(load=2.0, cores=20) is True


def test_exactly_at_the_floor_is_admitted():
    # 16.0 load on 20 cores leaves exactly 4 -- the floor is a minimum, not a strict bound
    assert gpu_spool.cpu_headroom_ok(load=16.0, cores=20) is True


def test_unknown_load_admits_rather_than_deadlocks():
    """A box that cannot report load must not become permanently unschedulable."""
    assert gpu_spool.cpu_headroom_ok(load=None, cores=20) is True
    assert gpu_spool.cpu_headroom_ok(load=1.0, cores=None) is True


def test_reserve_refuses_when_cores_are_short(monkeypatch, tmp_path):
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("PINO_SPOOL_VRAM_MB", "10000")
    monkeypatch.setenv("PINO_SPOOL_CHECK_FREE", "0")
    monkeypatch.setenv("PINO_SPOOL_DEFAULT_RAM_MB", "0")
    monkeypatch.setattr(gpu_spool, "_load_1min", lambda: 19.0)
    monkeypatch.setattr(gpu_spool, "_cpu_count", lambda: 20)
    gpu_spool.ensure_dirs(tmp_path / "spool", 0)
    assert gpu_spool.reserve(tmp_path / "spool", 0, "crowded", 1000) is None


def test_reserve_admits_when_cores_are_free(monkeypatch, tmp_path):
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("PINO_SPOOL_VRAM_MB", "10000")
    monkeypatch.setenv("PINO_SPOOL_CHECK_FREE", "0")
    monkeypatch.setenv("PINO_SPOOL_DEFAULT_RAM_MB", "0")
    monkeypatch.setattr(gpu_spool, "_load_1min", lambda: 3.0)
    monkeypatch.setattr(gpu_spool, "_cpu_count", lambda: 20)
    gpu_spool.ensure_dirs(tmp_path / "spool", 0)
    got = gpu_spool.reserve(tmp_path / "spool", 0, "roomy", 1000)
    assert got is not None
    gpu_spool._drop(*got)
