"""A queued run can say it should go next, without pausing everything else.

Reported 2026-08-06: "run this in priority" had exactly one implementation --
pause two unrelated sweeps by hand. That is coarse (it stops a whole sweep
rather than deprioritising it) and it leaves genuinely dangerous state: if whoever
paused them stops, the paused batches sit indefinitely with nothing scheduled to
resume them.

The seat is a plain `flock`, so waiters race and the kernel picks. Ordering is
strict and cooperative: a run stands aside while any live, fresh,
higher-priority run is queued. Starvation is bounded by ticket validity, not by
a clock -- see `TICKET_MAX_AGE_S`.
"""

from __future__ import annotations

import json
import os

import pytest

from aiida_pino.scripts.flow3d_run_adapter import (
    TICKET_MAX_AGE_S,
    _live_waiters,
    _register_waiter,
    _waiting_dir,
)
from aiida_pino.utils.profiles import validate_flow3d_run_execution_profile


def test_a_waiter_announces_what_it_is_worth(tmp_path):
    ticket = _register_waiter(tmp_path, 5)
    assert ticket is not None and ticket.is_file()
    row = json.loads(ticket.read_text())
    assert row["priority"] == 5 and row["pid"] == os.getpid()
    assert [w["priority"] for w in _live_waiters(tmp_path)] == [5]


def test_a_dead_waiters_ticket_is_pruned(tmp_path):
    """A crashed job must not hold the queue hostage from beyond the grave."""
    stale = _waiting_dir(tmp_path) / "999999999.json"
    stale.write_text(json.dumps({"pid": 999999999, "priority": 99, "since": 0}))
    assert _live_waiters(tmp_path) == []
    assert not stale.exists(), "a ticket whose process is gone must be removed"


def test_a_corrupt_ticket_is_ignored_not_fatal(tmp_path):
    (_waiting_dir(tmp_path) / "bad.json").write_text("{not json")
    _register_waiter(tmp_path, 3)
    assert [w["priority"] for w in _live_waiters(tmp_path)] == [3]


def test_starvation_is_bounded_by_ticket_validity_not_a_clock():
    """Changed deliberately: the one-hour deference budget is gone.

    It duplicated the ticket age cap and was the worse of the two bounds,
    because it made `seat_priority=9` lose the seat to a priority-0 run that had
    been queuing an hour -- the opposite of what the field says, and invisible
    unless you read the log. The FLOW-3D seat has no admitted-but-blocked state
    (a ticket-holder is sitting on the flock; if it can take the lock it can
    run), so the only way a higher-priority ticket blocks anything forever is a
    dead, zombied or stopped owner -- which the pid check and this cap handle.
    """
    import aiida_pino.scripts.flow3d_run_adapter as adapter

    assert TICKET_MAX_AGE_S > 0
    assert not hasattr(adapter, "MAX_DEFER_S"), "the duplicate time bound must stay gone"


def test_seat_priority_is_an_accepted_profile_key():
    validate_flow3d_run_execution_profile({"seat_priority": 10})
    validate_flow3d_run_execution_profile({"seat_priority": -1})
    validate_flow3d_run_execution_profile({})
    with pytest.raises(ValueError, match="seat_priority"):
        validate_flow3d_run_execution_profile({"seat_priority": "high"})
    with pytest.raises(ValueError, match="seat_priority"):
        validate_flow3d_run_execution_profile({"seat_priority": True})


def test_priority_orders_the_waiters_for_real(tmp_path, monkeypatch):
    """Three processes queue behind a holder; the highest priority goes first.

    Measured 5/5 trials as `blocker -> p9 -> p5 -> p1` before this was written
    down. A running holder is NEVER preempted -- it took the seat through the
    non-blocking fast path and never enters the wait loop -- because killing a
    solve that is already burning licence time to start a "more important" one
    wastes both. Priority orders the queue, it does not interrupt work.
    """
    import multiprocessing as mp

    from aiida_pino.scripts import flow3d_run_adapter as adapter

    def worker(lock_dir, priority, label, delay, results):
        adapter.spool_dir = lambda: __import__("pathlib").Path(lock_dir)
        __import__("time").sleep(delay)
        with adapter.hold_seat(priority):
            results.append((label, __import__("time").time()))
            __import__("time").sleep(0.5)

    # fork copies parent memory, so turning the poll interval down here is
    # inherited by every child. At the shipped 5.0 s these two tests cost 26 s
    # of a 71 s fast tier and the wait buys nothing: 15/15 identical ordering.
    monkeypatch.setattr(adapter, "SEAT_POLL_S", 0.05)
    ctx = mp.get_context("fork")
    with ctx.Manager() as manager:
        results = manager.list()
        entries = [("blocker", 0, 0.0), ("p1", 1, 0.3), ("p9", 9, 0.3), ("p5", 5, 0.3)]
        procs = [ctx.Process(target=worker, args=(str(tmp_path), pri, label, delay, results))
                 for label, pri, delay in entries]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(120)
        order = [row[0] for row in sorted(results, key=lambda row: row[1])]

    assert len(order) == 4, f"every waiter must get through, got {order}"
    assert order[0] == "blocker", "the process already holding the seat is not preempted"
    assert order[1:] == ["p9", "p5", "p1"], f"waiters must drain by priority, got {order}"


def test_the_deference_budget_is_not_spent_by_the_current_holder(tmp_path, monkeypatch):
    """A long holder must not cost the high-priority waiter its place.

    This began as a regression test for the one-hour deference budget being
    spent while merely queuing -- 43.5% of all seat wall-clock sits inside holds
    longer than that budget, so a waiter routinely burned it before yielding
    once. The budget is gone entirely now (ordering is strict), which makes the
    property unconditional rather than conditional: however long the current
    holder keeps the seat, the higher priority still goes next.
    """
    import multiprocessing as mp

    from aiida_pino.scripts import flow3d_run_adapter as adapter

    def worker(lock_dir, priority, label, delay, results, budget):
        import pathlib
        import time as _time

        adapter.spool_dir = lambda: pathlib.Path(lock_dir)
        _time.sleep(delay)
        with adapter.hold_seat(priority):
            results.append((label, _time.time()))
            _time.sleep(0.3)

    # fork copies parent memory, so turning the poll interval down here is
    # inherited by every child. At the shipped 5.0 s these two tests cost 26 s
    # of a 71 s fast tier and the wait buys nothing: 15/15 identical ordering.
    monkeypatch.setattr(adapter, "SEAT_POLL_S", 0.05)
    ctx = mp.get_context("fork")
    with ctx.Manager() as manager:
        results = manager.list()
        entries = [("holder", 0, 0.0, 4.0), ("low", 1, 0.4, 0.3), ("HIGH", 9, 0.4, 0.3)]
        procs = []
        for label, pri, delay, hold in entries:
            target = worker if label != "holder" else worker
            procs.append(ctx.Process(target=target, args=(str(tmp_path), pri, label, delay, results, 2.0)))
        # The holder keeps the seat far longer than the waiters take to queue.
        procs[0] = ctx.Process(
            target=_long_holder, args=(str(tmp_path), str(tmp_path), results, 4.0)
        )
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(120)
        order = [row[0] for row in sorted(results, key=lambda row: row[1])]

    assert order[0] == "holder"
    assert order[1] == "HIGH", f"the budget was spent waiting, not deferring: {order}"


def _long_holder(lock_dir, _unused, results, hold_s):
    import pathlib
    import time as _time

    from aiida_pino.scripts import flow3d_run_adapter as adapter

    adapter.spool_dir = lambda: pathlib.Path(lock_dir)
    with adapter.hold_seat(0):
        results.append(("holder", _time.time()))
        _time.sleep(hold_s)


def test_the_two_priorities_are_different_mechanisms():
    """FLOW-3D and GPU priority are not the same knob, and must not be confused.

    `execution_profile.seat_priority` is an INTEGER ordering the FLOW-3D licence
    seat queue (higher first, bounded deference, no preemption).
    `training_profile.gpu_seat_priority` is a BOOLEAN queue-jump on gpu_spool's
    GPU seat, whose own contract trades a deadline for the jump.

    `train_adapter.wrap_with_gpu_seat` had always read the GPU one and appended
    `--priority`, but the key was declared only on the EVALUATION profile, so
    the console's unknown-key check refused it on a training: working code that
    no caller could reach.
    """
    from aiida_pino.scripts.train_adapter import wrap_with_gpu_seat
    from aiida_pino.utils.profiles import DEFAULT_TRAINING_PROFILE

    assert "gpu_seat_priority" in DEFAULT_TRAINING_PROFILE
    assert DEFAULT_TRAINING_PROFILE["gpu_seat_priority"] is False, "opt-in, never on by default"

    wrapped = wrap_with_gpu_seat(["python", "train.py"], {"cuda_device": "0", "gpu_seat_priority": True})
    assert "--priority" in wrapped
    assert "--priority" not in wrap_with_gpu_seat(["python", "train.py"], {"cuda_device": "0"})

    # The FLOW-3D key is an integer and belongs to the execution profile; it is
    # not a training key and must not be silently accepted as one.
    validate_flow3d_run_execution_profile({"seat_priority": 7})
    assert "seat_priority" not in DEFAULT_TRAINING_PROFILE


def test_an_unknown_key_is_still_refused():
    """The new key must not have widened the profile to anything at all."""
    with pytest.raises(ValueError, match="unknown keys"):
        validate_flow3d_run_execution_profile({"seat_priorty": 3})
