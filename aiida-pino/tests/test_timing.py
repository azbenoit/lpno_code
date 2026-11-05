"""Timing that survives a shared box.

Wall clock here is not a property of the operation. Measured while writing this:
both TITAN RTXs at 100% and 78%, load 26.7 on 20 cores, two runs queued for the
FLOW-3D seat. Two defences, both measured on the live database:

* **normalise by work done** — seconds per solver cycle rather than seconds.
  Relative IQR falls from **125% to 51%** on 1171 FLOW-3D runs and from
  **351% to 54%** on the trainings;
* **record the contention** and filter on it, rather than averaging an idle box
  together with a saturated one.

The snapshot costs 67 ms and is taken twice per run — 0.04% of the 348 s median
solve, which is the whole reason it can be always-on.
"""

from __future__ import annotations

from aiida_pino.parsers.common import stamp_runtime_extras
from aiida_pino.scripts.live_tail import contention_snapshot
from aiida_pino.utils.timing import _quartiles


def test_the_snapshot_is_cheap_and_never_raises():
    import time

    started = time.perf_counter()
    snapshot = contention_snapshot()
    elapsed = time.perf_counter() - started
    assert snapshot["at"], "a witness with no timestamp cannot be matched to a run"
    # Generous: the budget is one nvidia-smi call. If this ever approaches the
    # length of a short operation, it stops being free and must become optional.
    assert elapsed < 5.0, f"contention snapshot took {elapsed:.2f}s"


def test_quartiles_report_spread_not_just_a_middle():
    stats = _quartiles([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats["median"] == 3.0
    assert stats["max"] == 100.0
    assert stats["n"] == 5
    # A single 57-hour solve among 348-second ones must not move the middle.
    assert stats["median"] < 10.0
    single = _quartiles([7.0])
    assert single["median"] == single["p25"] == single["p75"] == 7.0
    assert single["iqr"] == 0.0


def test_solo_is_recorded_from_both_ends(aiida_profile_clean):
    """One quiet sample is not evidence; a run is only solo if it stayed solo."""
    from aiida import orm

    def _stamp(start, end):
        node = orm.Dict(dict={}).store()
        stamp_runtime_extras(node, {"runtime": {
            "wall_seconds": 10.0, "contention_start": start, "contention_end": end,
        }})
        return node.base.extras.get("solo_run", None)

    quiet = {"loadavg": [2.0, 2.0, 2.0], "cpu_count": 20, "gpu_busy": 0}
    busy = {"loadavg": [40.0, 40.0, 40.0], "cpu_count": 20, "gpu_busy": 2}

    assert _stamp(quiet, quiet) is True
    assert _stamp(quiet, busy) is False, "it got busy during the run"
    assert _stamp(busy, quiet) is False, "it started under load"
    # No witness at all must read as unknown, never as solo.
    assert _stamp({}, {}) is None


def test_work_counts_land_on_the_calcjob(aiida_profile_clean):
    """`cycles` is what makes the contention-independent metric possible."""
    from aiida import orm

    node = orm.Dict(dict={}).store()
    stamp_runtime_extras(node, {
        "runtime": {"wall_seconds": 100.0},
        # The runner writes cycles as a STRING; a numeric check drops them all.
        "manifest": {"run": {"elapsed_seconds": 90.0, "cycles": "1200"}},
    })
    assert node.base.extras.get("cycles") == 1200
    assert node.base.extras.get("solver_elapsed_seconds") == 90.0
