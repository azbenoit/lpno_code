"""A seat request must never block a card forever (2026-07-28).

CPU-only and license-free. CI booked ``--ram-mb 92000`` on a 125 GB box that
routinely has ~84 GB free. ``reserve`` clamps an oversized *VRAM* request so it
cannot queue for ever, but host RAM had no equivalent: the request simply waited,
and because it was ``--priority`` its ``.want`` marker also stopped every worker
from starting anything new. The card sat idle, the queue sat blocked, and there
was no max wait, no yield, and no admission check against the machine's total
capacity. Three escapes, tested here.
"""
import os

import pytest

from lpno import gpu_spool


def test_request_above_machine_total_is_refused_immediately(tmp_path, monkeypatch):
    """A booking larger than the box can *ever* provide must fail fast.

    MemAvailable fluctuates, so waiting on it is legitimate; MemTotal does not.
    A request above it is unsatisfiable no matter what finishes, which is a
    deadlock rather than a queue.
    """
    monkeypatch.setattr(gpu_spool, "ram_total_mb", lambda: 8000)
    with pytest.raises(SystemExit) as err:
        gpu_spool._run_seat(tmp_path, 0, ["true"], ram_mb=99000, poll=0.01)
    msg = str(err.value)
    assert "99000" in msg and "8000" in msg, msg
    # names the machine limit it can never clear, not a generic timeout
    assert "total" in msg.lower()


def test_request_within_total_is_not_refused(tmp_path, monkeypatch):
    """Below MemTotal the request is legitimate even if nothing is free now --
    another job finishing can satisfy it, so it must queue, not fail."""
    monkeypatch.setattr(gpu_spool, "ram_total_mb", lambda: 128000)
    monkeypatch.setattr(gpu_spool, "ram_available_mb", lambda: 1000)
    # max_wait_s makes it give up rather than hang the test
    with pytest.raises(SystemExit) as err:
        gpu_spool._run_seat(tmp_path, 0, ["true"], ram_mb=92000, poll=0.01, max_wait_s=0.05)
    # It queued and then timed out -- it must NOT have been refused at admission,
    # which is the difference between "wait for capacity" and "impossible".
    assert "refusing" not in str(err.value).lower(), str(err.value)
    assert "gave up after waiting" in str(err.value), str(err.value)


def test_max_wait_gives_up_with_an_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_spool, "ram_total_mb", lambda: 128000)
    monkeypatch.setattr(gpu_spool, "ram_available_mb", lambda: 10)
    with pytest.raises(SystemExit) as err:
        gpu_spool._run_seat(tmp_path, 0, ["true"], ram_mb=92000, poll=0.01, max_wait_s=0.05)
    msg = str(err.value)
    assert "--ram-mb" in msg or "ram" in msg.lower()


def test_priority_want_marker_is_released_while_still_waiting(tmp_path, monkeypatch):
    """The yield: a priority run that cannot be satisfied must stop starving the
    queue. It may keep waiting, but it must not hold `.want` indefinitely --
    that is what left the card idle *and* the workers blocked.
    """
    monkeypatch.setattr(gpu_spool, "ram_total_mb", lambda: 128000)
    monkeypatch.setattr(gpu_spool, "ram_available_mb", lambda: 10)
    seen = {}

    real_drop = gpu_spool._drop

    def spy_drop(fd, path):
        if path.suffix == ".want":
            seen["want_dropped"] = True
        return real_drop(fd, path)

    monkeypatch.setattr(gpu_spool, "_drop", spy_drop)
    with pytest.raises(SystemExit):
        gpu_spool._run_seat(tmp_path, 0, ["true"], ram_mb=92000, poll=0.01,
                            priority=True, max_wait_s=0.2, priority_yield_s=0.01)
    assert seen.get("want_dropped"), "priority .want was never released"
    # and it is gone from disk, so workers can proceed
    assert not list((tmp_path / "gpu0").glob("*.want")), "priority marker still on disk"


def test_ram_total_mb_reads_meminfo():
    total = gpu_spool.ram_total_mb()
    assert total is None or total > 0
    if total is not None and os.path.exists("/proc/meminfo"):
        # sanity: total must be at least what is currently available
        avail = gpu_spool.ram_available_mb()
        if avail is not None:
            assert total >= avail


def test_zero_max_wait_means_wait_forever(tmp_path, monkeypatch):
    """0 keeps the old behaviour for anyone who wants it -- but it is not the
    default, because the default is what CI got bitten by."""
    monkeypatch.setattr(gpu_spool, "ram_total_mb", lambda: 128000)
    calls = {"n": 0}

    def fake_reserve(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt  # stand-in for "eventually it fits"
        return None

    monkeypatch.setattr(gpu_spool, "reserve", fake_reserve)
    with pytest.raises(KeyboardInterrupt):
        gpu_spool._run_seat(tmp_path, 0, ["true"], ram_mb=1000, poll=0.001, max_wait_s=0)
    assert calls["n"] > 3, "max_wait_s=0 should keep retrying"
