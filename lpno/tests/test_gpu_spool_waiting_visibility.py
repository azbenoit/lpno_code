"""A run blocked on a seat announces itself in the ledger.

`gpu_spool seat` books a `.lease` once it wins a card, and a `--priority` run
also drops a `.want` entry so workers hold off. A plain waiter recorded nothing
at all, so a job blocked for hours was invisible to every observer: it is not in
the spooler queue (it never queued), it has no GPU process yet (it blocks before
CUDA init), and it holds no lease. Observed 2026-07-23 — the console read
"gpu0 idle, 3 queued / gpu1 idle, 0 queued" while two AiiDA trainings had been
waiting hours for seats on those cards.

`.waiting` entries fix that and are deliberately **inert**: admission consults
`.lease` (budget) and `.want` (priority yielding) and must keep ignoring
`.waiting`, or every waiter would silently acquire priority semantics.

GPU-free: the ledger is redirected into tmp_path and the budget pinned.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from lpno import gpu_spool as gs

SPOOL = None

BUDGET_MB = 1000


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path: Path, monkeypatch):
    global SPOOL
    SPOOL = tmp_path / "spool"
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("PINO_SPOOL_VRAM_MB", str(BUDGET_MB))
    monkeypatch.setenv("PINO_SPOOL_CHECK_FREE", "0")
    monkeypatch.setenv("PINO_SPOOL_DEFAULT_RAM_MB", "0")
    monkeypatch.delenv("PINO_SPOOL_DEFAULT_VRAM_MB", raising=False)


def test_no_waiters_reports_empty():
    gs.ensure_dirs(SPOOL, 0)
    assert gs.waiting_entries(0) == []


def test_a_blocked_seat_announces_itself(tmp_path):
    """Fill the card, then start a seat that cannot fit and check it is visible."""
    gs.ensure_dirs(SPOOL, 0)
    holder = gs.reserve(SPOOL, 0, "holder", BUDGET_MB)
    assert holder is not None, "the holder should have taken the whole budget"

    done = threading.Event()

    def _blocked():
        gs._run_seat(SPOOL, 0, vram_mb=BUDGET_MB, ram_mb=0, priority=False,
                    command=["true"], poll=0.05)
        done.set()

    t = threading.Thread(target=_blocked, daemon=True)
    t.start()
    deadline = time.time() + 10
    seen = []
    while time.time() < deadline and not seen:
        seen = gs.waiting_entries(0)
        time.sleep(0.05)

    assert seen, "a blocked seat must leave a .waiting entry"
    assert seen[0]["vram_mb"] == BUDGET_MB
    assert "name" in seen[0] and "pid" in seen[0]

    gs._drop(*holder)          # release the card
    t.join(timeout=20)
    assert done.is_set(), "the waiter should have acquired the seat once freed"
    assert gs.waiting_entries(0) == [], "the .waiting entry must be cleaned up"


def test_waiting_entries_do_not_block_admission():
    """Inert by design: unlike `.want`, a `.waiting` entry must not gate a run.

    If admission ever consulted `.waiting`, every ordinary waiter would acquire
    the priority semantics that only `--priority` is meant to have.
    """
    gs.ensure_dirs(SPOOL, 0)
    path = gs._entry_path(0, ".waiting")
    fd = gs._hold(path, {"name": "ghost", "pid": 1, "vram_mb": BUDGET_MB})
    try:
        booked = gs.reserve(SPOOL, 0, "newcomer", BUDGET_MB)
        assert booked is not None, "a .waiting entry must not reserve budget"
        gs._drop(*booked)
    finally:
        gs._drop(fd, path)


def test_waiting_entries_do_not_consume_budget():
    gs.ensure_dirs(SPOOL, 0)
    path = gs._entry_path(0, ".waiting")
    fd = gs._hold(path, {"name": "ghost", "pid": 1, "vram_mb": BUDGET_MB})
    try:
        assert gs.reserved_mb(0) == 0, "only .lease entries book VRAM"
    finally:
        gs._drop(fd, path)


def test_a_seat_that_fits_immediately_leaves_nothing_behind():
    gs.ensure_dirs(SPOOL, 0)
    rc = gs._run_seat(SPOOL, 0, vram_mb=10, ram_mb=0, priority=False,
                     command=["true"], poll=0.05)
    assert rc == 0
    assert gs.waiting_entries(0) == []
    assert gs.reserved_mb(0) == 0


def test_unbooked_tenants_climb_the_process_tree():
    """A CUDA child of a lease-holding worker is booked; a stranger is not."""
    ppid = {300: 200, 200: 100, 100: 1, 999: 1}  # 100 holds the lease
    apps = [(300, "python", 4000), (999, "web_server", 1642)]
    ghosts = gs.unbooked_tenants(apps, {100}, ppid_of=lambda p: ppid.get(p, 0))
    assert ghosts == [(999, "web_server", 1642)]


def test_unbooked_tenants_survive_a_ppid_cycle():
    ppid = {7: 8, 8: 7}  # corrupt /proc reading must not hang status
    assert gs.unbooked_tenants([(7, "x", 10)], {5}, ppid_of=lambda p: ppid.get(p, 0)) \
        == [(7, "x", 10)]


# --------------------------------------------------------------------------- #
# 2026-08-20: dead .waiting files accumulated for weeks (235 on the primary box,
# oldest 07-27) because waiting_entries(prune=True) had no production caller.
# The prune belongs in the ADMIT path -- _entries' contract says stale entries
# are deleted "only from inside the admit lock", so a read-only caller like
# status must never unlink (it could race a file another process is mid-creating).
# --------------------------------------------------------------------------- #
def _dead_waiting_file(gpu: int, name: str) -> Path:
    """A .waiting entry whose owner is gone: valid JSON, nobody holds the flock."""
    path = gs.lease_dir(gpu) / f"{name}.waiting"
    path.write_text(json.dumps({"name": name, "vram_mb": 1, "pid": 1}), encoding="utf-8")
    return path


def test_an_admission_attempt_prunes_dead_waiting_files():
    gs.ensure_dirs(SPOOL, 0)
    dead = _dead_waiting_file(0, "long-gone")
    got = gs.reserve(SPOOL, 0, "newcomer", 1)
    assert got is not None
    try:
        assert not dead.exists(), ".waiting of a dead process must be pruned by admission"
    finally:
        gs._drop(*got)


def test_admission_prune_keeps_a_live_waiter():
    import fcntl
    import os as _os

    gs.ensure_dirs(SPOOL, 0)
    live = gs.lease_dir(0) / "still-here.waiting"
    live.write_text(json.dumps({"name": "still-here", "vram_mb": 1, "pid": _os.getpid()}),
                    encoding="utf-8")
    fd = _os.open(live, _os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        got = gs.reserve(SPOOL, 0, "newcomer", 1)
        assert got is not None
        gs._drop(*got)
        assert live.exists(), "a held .waiting entry must survive admission pruning"
    finally:
        _os.close(fd)


def test_status_shows_live_waiters_but_never_unlinks_dead_ones(capsys):
    """status is read-only: it must NAME live blocked runs (2026-07-23: two trainings
    sat invisible for hours) and must not delete even a dead entry -- pruning outside
    the admit lock could unlink a file another process is mid-creating. Stale entries
    are (rightly) not displayed either; they just wait for the next admission prune."""
    import fcntl
    import os as _os

    gs.ensure_dirs(SPOOL, 0)
    dead = _dead_waiting_file(0, "ghost")
    live = gs.lease_dir(0) / "patient-run.waiting"
    live.write_text(json.dumps({"name": "patient-run", "vram_mb": 1, "pid": _os.getpid()}),
                    encoding="utf-8")
    fd = _os.open(live, _os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        gs.status(SPOOL, [0])
        out = capsys.readouterr().out
        assert dead.exists(), "status must not unlink a stale entry"
        assert "patient-run" in out, "a live blocked run must be named in status"
    finally:
        _os.close(fd)
