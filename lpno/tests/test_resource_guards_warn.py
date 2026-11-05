"""Memory guards are warnings in the public cut (2026-09-14: "all the memory and disk
guards should just be warnings ... memory limits are hardware dependent anyways").

One switch, `PINO_RESOURCE_GUARDS`: `enforce` (this workstation's default) refuses or waits
exactly as before; `warn` prints `[resource-guard] WARNING ...` and admits. The public copy
flips the module literal `RESOURCE_GUARDS_DEFAULT`, so nothing here depends on the env of the
box that runs the suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from lpno import gpu_spool as gs

BUDGET_MB = 1000


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("PINO_SPOOL_VRAM_MB", str(BUDGET_MB))
    monkeypatch.setenv("PINO_SPOOL_CHECK_FREE", "0")
    monkeypatch.setenv("PINO_SPOOL_DEFAULT_RAM_MB", "0")
    monkeypatch.delenv("PINO_SPOOL_DEFAULT_VRAM_MB", raising=False)
    monkeypatch.setenv("PINO_RESOURCE_GUARDS", "enforce")


def test_the_literal_is_the_default_and_the_env_overrides_it(monkeypatch):
    # The private repo carries "enforce"; the public assembler rewrites the literal to
    # "warn". Either way the env wins, which is what the rest of the suite relies on.
    assert gs.RESOURCE_GUARDS_DEFAULT in ("enforce", "warn")
    monkeypatch.setenv("PINO_RESOURCE_GUARDS", "enforce")
    assert gs.resource_guard(True, "x") is True
    assert gs.resource_guard(False, "x") is False
    monkeypatch.setenv("PINO_RESOURCE_GUARDS", "warn")
    assert gs.resource_guard(True, "x") is False


def test_warn_mode_admits_past_the_vram_budget(tmp_path: Path, monkeypatch, capsys):
    a = gs.reserve(tmp_path, 0, "a", 400)
    b = gs.reserve(tmp_path, 0, "b", 400)
    assert gs.reserve(tmp_path, 0, "c", 400) is None  # enforce: 1200 > 1000
    monkeypatch.setenv("PINO_RESOURCE_GUARDS", "warn")
    c = gs.reserve(tmp_path, 0, "c", 400)
    assert c is not None  # warn: admitted, and the ledger still records it
    assert gs.reserved_mb(0) == 1200
    assert "[resource-guard] WARNING" in capsys.readouterr().out
    for h in (a, b, c):
        gs._drop(*h)


def test_warn_mode_admits_past_host_ram(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(gs, "ram_available_mb", lambda: 1000)
    monkeypatch.setattr(gs, "ram_total_mb", lambda: 2000)
    monkeypatch.setattr(gs, "ram_headroom_mb", lambda: 0)
    assert gs.reserve(tmp_path, 0, "big", 100, ram_mb=5000) is None  # enforce
    monkeypatch.setenv("PINO_RESOURCE_GUARDS", "warn")
    held = gs.reserve(tmp_path, 0, "big", 100, ram_mb=5000)
    assert held is not None
    out = capsys.readouterr().out
    assert out.count("[resource-guard] WARNING") >= 1 and "not enforced" in out
    gs._drop(*held)


def test_warn_mode_does_not_refuse_an_unsatisfiable_ram_booking(monkeypatch, capsys):
    # `_run_seat` raises SystemExit under enforce when --ram-mb exceeds MemTotal; the
    # decision goes through the same helper, which is what this pins.
    monkeypatch.setenv("PINO_RESOURCE_GUARDS", "warn")
    assert gs.resource_guard(True, "--ram-mb 5000 exceeds the machine's 2000 MiB") is False
    assert "not enforced" in capsys.readouterr().out
