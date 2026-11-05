"""How long each kind of operation takes, without pretending the box is idle.

Wall clock on a shared workstation is not a property of the operation. Measured
while writing this: both TITAN RTXs at 100% and 78%, load 26.7 on 20 cores, two
runs queued for the FLOW-3D seat. Any average over that mixture is a number
about the week, not about the code.

There are two honest ways out, and this uses both.

**Normalise by the work done.** FLOW-3D reports its own `cycles`, training
reports `steps_trained`. Cost per unit of work absorbs a solve that simply had
more to do, which raw wall clock cannot.

**Record the contention and filter on it.** Every run stamps `loadavg_start`,
`loadavg_end`, `gpu_busy_*` and `solo_run` (see
`parsers/common.stamp_runtime_extras`). `solo_run` means at most one GPU busy
and load at or below one per core at both ends, so `--solo` compares only runs
that were effectively alone. Nothing here reconstructs that after the fact: a
run without the witness is reported as unknown rather than folded in.

Reported as median and inter-quartile range, never a mean. One 57-hour solve in
a set of 348-second ones moves a mean and does not move a median.

    python -m aiida_pino.utils.timing
    python -m aiida_pino.utils.timing --solo
"""

from __future__ import annotations

import statistics
from typing import Any

from aiida import orm

#: Operations worth timing, and the field that says how much work each did.
OPERATIONS = {
    "Flow3DRunCalculation": ("solver_elapsed_seconds", "cycles", "s/cycle"),
    "PinoTrainCalculation": ("runtime_wall_seconds", "steps_trained", "s/step"),
    "PinoEvalCalculation": ("runtime_wall_seconds", None, None),
    "RawFlow3DToDatasetCalculation": ("runtime_wall_seconds", None, None),
}


def _quartiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    if len(ordered) >= 4:
        lower = statistics.median(ordered[: len(ordered) // 2])
        upper = statistics.median(ordered[(len(ordered) + 1) // 2 :])
    else:
        lower = upper = median
    return {"median": round(median, 3), "p25": round(lower, 3), "p75": round(upper, 3),
            "iqr": round(upper - lower, 3), "min": round(ordered[0], 3),
            "max": round(ordered[-1], 3), "n": len(ordered)}


def report(*, solo_only: bool = False) -> dict[str, Any]:
    """Median and IQR per operation, optionally restricted to uncontended runs."""
    out: dict[str, Any] = {"solo_only": solo_only, "operations": {}}
    for label, (time_key, work_key, unit) in OPERATIONS.items():
        walls: list[float] = []
        normalized: list[float] = []
        witnessed = solo = uninstrumented = 0
        for extras in orm.QueryBuilder().append(
            orm.CalcJobNode, filters={"attributes.process_label": label}, project=["extras"]
        ).all(flat=True):
            extras = extras or {}
            if extras.get("solo_run") is not None:
                witnessed += 1
                solo += int(bool(extras["solo_run"]))
            if solo_only and not extras.get("solo_run"):
                continue
            seconds = extras.get(time_key)
            if not isinstance(seconds, (int, float)) or seconds <= 0:
                uninstrumented += 1
                continue
            walls.append(float(seconds))
            work = extras.get(work_key) if work_key else None
            if isinstance(work, (int, float)) and work > 0:
                normalized.append(float(seconds) / float(work))
        entry: dict[str, Any] = {
            "runs_timed": len(walls),
            # Jobs of this kind carrying no meter reading. Import jobs are ALL in
            # this bucket: raw_flow3d_to_dataset_adapter.py never starts a
            # RuntimeMeter and parsers/vtk_to_dataset.py is a plain Parser, so
            # stamp_runtime_extras never fires (0 of 1418 on 2026-08-09). Saying
            # so beats printing `runs_timed: 0` with no explanation.
            #
            # Node timestamps do NOT substitute: mtime is bumped by any extras
            # write (one backfill re-dated every eval node), and output ctime
            # minus job ctime includes scheduler queueing -- calibrated against
            # 589 metered solver jobs it runs 2.5x the measured wall time at the
            # median, 12x at p90.
            "not_instrumented": uninstrumented,
            "with_a_contention_witness": witnessed,
            "of_those_solo": solo,
        }
        if walls:
            entry["wall_seconds"] = _quartiles(walls)
        if normalized:
            entry[f"per_unit ({unit})"] = _quartiles(normalized)
        out["operations"][label] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solo", action="store_true",
                        help="only runs that were effectively alone on the box")
    parser.add_argument("--profile", default=None, help="AiiDA profile (default: the configured default profile)")
    args = parser.parse_args(argv)

    from aiida import load_profile

    load_profile(args.profile)
    print(json.dumps(report(solo_only=args.solo), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
