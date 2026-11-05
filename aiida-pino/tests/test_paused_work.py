"""Parked work has to be visible, because priority cannot park anything.

`seat_priority` reorders a queue; it does not hold work back. Ordering is now
strict — a run stands aside while any live, fresh, higher-priority run is
queued — but that still parks nothing, because parking means "stay out of the
way of everything, including work that has not been submitted yet", and a queue
position cannot express that.

Which leaves pausing, whose hazard is being forgotten. Measured on the authors' database
2026-08-07: 4 paused processes, **all four with no description**, one of them
showing `Killed through verdi process kill` while still counted as
paused 38.4 hours later — plus 321 held spool tasks in two queues that no view
put beside them.
"""

from __future__ import annotations

from aiida_pino.console.config import Settings
from aiida_pino.console.service import AiidaConsoleService


def test_paused_work_reports_both_places_and_flags_the_undescribed(aiida_profile_clean, monkeypatch):
    service = AiidaConsoleService(Settings())

    # The spool half is a fallback path, not a dependency: stub it so the test
    # does not read the real ~/.flow3d/spool.
    monkeypatch.setattr(
        service, "flow3d_queue_status",
        lambda: {"held": [{"name": "queue_held_alsi", "count": 305},
                          {"name": "queue_held_mb40_domain", "count": 16}]},
    )
    out = service.paused_work()

    assert out["held_task_count"] == 321, "held spool tasks must be counted, not just listed"
    assert out["paused_count"] == 0, "a clean profile has nothing paused"
    assert out["undescribed"] == []


def test_a_terminated_process_is_not_parked_work(aiida_profile_clean, monkeypatch):
    """`paused` is never cleared on termination, so KILLED nodes read as paused.

    One process reported "paused for 38.4 hours" while its own status said `Killed
    through verdi process kill`. Listing finished work as parked is how the
    genuinely parked items get lost in the noise.
    """
    service = AiidaConsoleService(Settings())
    monkeypatch.setattr(service, "flow3d_queue_status", lambda: {"held": []})

    class _Node:
        pk = 1
        paused = True
        is_terminated = True
        process_label = "Flow3DBatchWorkChain"
        label = description = process_status = None
        mtime = None

    seen = [_Node()]
    monkeypatch.setattr(
        "aiida.orm.QueryBuilder",
        lambda *a, **k: type("Q", (), {
            "append": lambda self, *a, **k: self,
            "all": lambda self, flat=False: seen,
        })(),
    )
    assert service.paused_work()["paused_count"] == 0


def test_a_broken_spool_does_not_break_the_audit(aiida_profile_clean, monkeypatch):
    """The spool is the manual fallback; its absence must not hide AiiDA's half."""
    service = AiidaConsoleService(Settings())

    def _boom():
        raise OSError("no spool here")

    monkeypatch.setattr(service, "flow3d_queue_status", _boom)
    out = service.paused_work()
    assert out["held_spool_queues"] == []
    assert out["held_task_count"] == 0
    assert "paused_processes" in out


def test_priority_cannot_park_work():
    """The fact that makes "resume at low priority" the wrong tool.

    Deference is evaluated per seat acquisition against the tickets on disk at
    that moment, so a low-priority run holds back only for work already queued.
    Nothing about it means "wait until told otherwise", which is what parking a
    sweep needs — that is what pausing is for, and why it needs to be visible.
    """
    import inspect

    from aiida_pino.scripts import flow3d_run_adapter as adapter

    source = inspect.getsource(adapter._wait_for_seat)
    assert "_better_waiters(lock_dir, priority)" in source, "deference reads the live queue"
    # No standing state: each acquisition starts from whatever is queued now.
    assert "waited_since = None" in source
