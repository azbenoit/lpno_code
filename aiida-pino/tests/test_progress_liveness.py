"""A step count is what a run REACHED, never proof it is still running.

2026-08-27, spotted from the Monitor itself: a compute machine showed 0% utilisation and
"no compute processes" on both cards, while two of its runs displayed a moving step count and
a beam-box MAE. Both facts were true of different things -- the GPU panel read the box, the
rows read a LOG, and a stopped job's log keeps its final step forever. Before the remote tail
existed those rows said "starting...", which was useless but not misleading.

The age of the log is not enough on its own, and the two failure directions are why:

  - an AR run at step 800/32000 after 14.7 h logs about every 66 s, so any threshold short
    enough to catch a run that stopped ten minutes ago calls that live run stale;
  - a run that finished ten minutes ago has a log younger than that same threshold.

So the authoritative signal is whether the run holds a GPU process -- which the monitor
already collects, with `run_pk` per process. `on_gpu is False` counts only when a snapshot of
that machine actually came back; otherwise the age is the fallback.
"""

from __future__ import annotations

from aiida_pino.console.service import classify_run


def test_a_run_holding_a_gpu_is_running():
    category, phase = classify_run("PinoTrainCalculation", {"step": 800}, on_gpu=True, has_seat=False)
    assert (category, phase) == ("gpu", "running")


def test_waiting_for_a_seat_is_queued_even_with_a_step():
    """A run that half-started and then logged `waiting for GPU seat` is queued."""
    progress = {"step": 10, "waiting_for_gpu": True}
    assert classify_run("PinoTrainCalculation", progress, on_gpu=False, has_seat=False)[1] == "queued"


def test_the_monitor_stamps_the_gpu_fact_onto_progress():
    """Structural: the row must carry both the fact and whether it is KNOWN.

    `on_gpu False` means "not running" only when some box answered; with no snapshot at all
    it means "unknown", and treating those the same would mark every row stale the moment a
    machine stopped answering.
    """
    import inspect

    from aiida_pino.console import service

    src = inspect.getsource(service)
    assert 'progress["on_gpu"] = bool(on_gpu)' in src
    assert 'progress["gpu_snapshot_seen"]' in src


def test_the_cell_prefers_the_gpu_fact_over_the_age():
    """The frontend rule, pinned so a later edit cannot quietly go back to age-only."""
    from pathlib import Path

    src = Path("web/public/app.js").read_text()
    assert "p.gpu_snapshot_seen && p.on_gpu === false" in src
    assert "const STALE_AFTER_S = 900" in src, "the fallback threshold must stay long"
    # a stale row must not show a beam-box MAE: the metric would be as old as the step
    assert "if (!bb || stale) return bar;" in src
