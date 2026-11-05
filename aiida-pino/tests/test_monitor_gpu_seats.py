"""The GPU panel must show seat holders and seat waiters, not just the spool queue.

`gpu_spool_snapshot` reads `~/.pino/spool/gpu*/{queue,running}` — the *spooler
worker's* own task list. AiiDA trainings never go there: they call
`gpu_spool seat`, which takes a lease under `/var/tmp/pino_gpu_leases/gpu*/`.
The result, observed 2026-07-23: the console read "gpu0 idle, 3 queued /
gpu1 idle, 0 queued" while three seat holders occupied 21 GB and two
AiiDA trainings had been waiting hours for a seat on those cards.

`attach_aiida_to_spool` half-covers the first case by counting AiiDA processes
already on the card, but a job *blocked before CUDA init* has no process, so it
is invisible everywhere.

The lease ledger is the authoritative view of what holds a card, so read it.
"""
from __future__ import annotations

import json
import os

import pytest

from aiida_pino.console.monitor import gpu_seat_snapshot


def _lease(dirpath, pid, name, vram, ram=0):
    dirpath.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "vram_mb": vram, "ram_mb": ram, "pid": pid,
               "user": "tester", "since": "2026-07-23T12:00:00", "exclusive": False}
    (dirpath / f"{pid}-1.lease").write_text(json.dumps(payload), encoding="utf-8")


def test_seat_holders_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    _lease(tmp_path / "gpu0", os.getpid(), "seat:sh", 11000, 6000)
    rows = {r["gpu"]: r for r in gpu_seat_snapshot()}
    assert rows["gpu0"]["seat_count"] == 1
    assert rows["gpu0"]["seat_vram_mb"] == 11000
    assert rows["gpu0"]["seats"][0]["name"] == "seat:sh"


def test_two_holders_sum_their_reservations(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    _lease(tmp_path / "gpu0", os.getpid(), "seat:sh", 11000)
    _lease(tmp_path / "gpu0", os.getppid(), "seat:sh", 11000)
    row = next(r for r in gpu_seat_snapshot() if r["gpu"] == "gpu0")
    assert row["seat_count"] == 2
    assert row["seat_vram_mb"] == 22000


def test_a_card_with_no_leases_reports_zero_not_missing(tmp_path, monkeypatch):
    """A quiet card must still appear, so the panel never silently drops a GPU."""
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    (tmp_path / "gpu1").mkdir(parents=True)
    row = next(r for r in gpu_seat_snapshot() if r["gpu"] == "gpu1")
    assert row["seat_count"] == 0
    assert row["seat_vram_mb"] == 0
    assert row["seats"] == []


def test_dead_holder_is_not_counted(tmp_path, monkeypatch):
    """A lease whose pid is gone is stale; counting it would show phantom load."""
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    _lease(tmp_path / "gpu0", 999_999_999, "seat:ghost", 11000)
    row = next(r for r in gpu_seat_snapshot() if r["gpu"] == "gpu0")
    assert row["seat_count"] == 0


def test_missing_ledger_is_not_an_error(tmp_path, monkeypatch):
    """No ledger (fresh box, or gpu_spool absent) must not break the panel."""
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path / "nope"))
    assert gpu_seat_snapshot() == []


def test_corrupt_lease_does_not_break_the_panel(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    (tmp_path / "gpu0").mkdir(parents=True)
    (tmp_path / "gpu0" / f"{os.getpid()}-1.lease").write_text("{not json", encoding="utf-8")
    row = next(r for r in gpu_seat_snapshot() if r["gpu"] == "gpu0")
    assert row["seat_count"] >= 0  # counted or skipped, but never raises


def test_waiting_jobs_are_surfaced(tmp_path, monkeypatch):
    """A job blocked on a seat has no process and no queue entry -- the panel
    must still say something is waiting, or a queue looks idle while it is not."""
    from aiida_pino.console.monitor import attach_seats_to_spool

    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    _lease(tmp_path / "gpu0", os.getpid(), "seat:sh", 11000)
    rows = [{"gpu": "gpu0", "running": [], "queued_count": 0, "queued": []}]
    merged = attach_seats_to_spool(rows)
    assert merged[0]["seat_count"] == 1
    assert merged[0]["seat_vram_mb"] == 11000
    assert merged[0]["busy"] is True, "a card with a seat holder is not idle"


def _waiting(dirpath, pid, name, vram, ram=0):
    dirpath.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "vram_mb": vram, "ram_mb": ram, "pid": pid,
               "user": "tester", "since": "2026-07-23T12:00:00", "priority": False}
    (dirpath / f"{pid}-2.waiting").write_text(json.dumps(payload), encoding="utf-8")


def test_waiters_are_reported(tmp_path, monkeypatch):
    """gpu_spool now publishes a .waiting entry for a blocked seat."""
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    _waiting(tmp_path / "gpu1", os.getpid(), "seat:sh", 19000)
    row = next(r for r in gpu_seat_snapshot() if r["gpu"] == "gpu1")
    assert row["waiting_count"] == 1
    assert row["waiting"][0]["vram_mb"] == 19000


def test_a_card_with_only_waiters_is_not_idle(tmp_path, monkeypatch):
    """The exact 2026-07-23 case: no lease, no spool queue, but work pending."""
    from aiida_pino.console.monitor import attach_seats_to_spool

    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    _waiting(tmp_path / "gpu1", os.getpid(), "seat:sh", 19000)
    rows = attach_seats_to_spool([{"gpu": "gpu1", "running": [], "queued_count": 0, "queued": []}])
    assert rows[0]["waiting_count"] == 1
    assert rows[0]["pending"] is True, "a card with a blocked run has work pending"


def test_dead_waiter_is_not_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path))
    _waiting(tmp_path / "gpu1", 999_999_999, "seat:ghost", 19000)
    row = next(r for r in gpu_seat_snapshot() if r["gpu"] == "gpu1")
    assert row["waiting_count"] == 0
