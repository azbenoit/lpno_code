"""Compute melt-pool statistics in the background as datasets land.

Why a separate worker rather than a hook in `Flow3DBatchWorkChain.finalize`:
`compute_case_metrics` loads whole field stacks and takes 30-120 s per case
(`service.py` measures 67 s for a 119-frame case), so a 300-case sweep is hours
of work. On 2026-08-05 a multi-hour dataset staging run INSIDE the daemon
worker's event loop made `verdi daemon stop`, `process kill` and the heartbeat
all time out for six hours, because RPC shares that loop. Putting this work
anywhere near the daemon would reproduce that exactly. It runs detached instead,
one case at a time, and can be stopped at any moment.

The pure functions here (`needs_stats`, `compute_one`, `drain`) hold the policy
and are unit-tested without AiiDA; `main()` is the thin polling loop around them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from typing import Any, Callable, Iterable, Sequence

#: Set on a simulation whose fields cannot yield melt-pool metrics at all, so the
#: worker stops offering it. Training-profile datasets carry no `Melt Region`,
#: and without this the worker retries those cases every cycle and never reaches
#: real work. Stamped with the version that failed, so a later metric revision
#: reconsiders them.
UNAVAILABLE_KEY = "meltpool_stats_unavailable"

#: Skip a case needing more than this many GiB of raw float32 field stacks.
#: 2026-08-21: the worker had restarted 16 times without once logging
#: `computed N/50`. `_pending` returns the newest pending cases in a fixed
#: order, so one case too big to hold was re-picked after every restart,
#: OOM-killed the process, and `Restart=always` returned it to the same case --
#: a poison pill that cost the other 2451 pending cases indefinitely. Measured
#: over 400 pending: median 0.46 GiB, p95 2.31 GiB, one case at
#: 13.86 GiB before `compute_case_metrics` adds its metal-subset and boolean
#: copies. 6 GiB steps over that tail and keeps everything else.
DEFAULT_MAX_STACK_GIB = 6.0


def full_stack_fields() -> tuple[str, ...]:
    from aiida_pino.console import viz

    return tuple(viz.FULL_STACK_FIELDS)


def estimated_stack_gib(sim) -> float | None:
    """GiB of field stacks `compute_case_metrics` would hold at once.

    ``None`` when the node records no shapes. Unknown is not the same as huge:
    skipping on unknown would strand every case with incomplete metadata, so an
    unsized case is attempted rather than tombstoned.
    """
    try:
        aliases = sim.base.attributes.get("array_aliases", None) or {}
        shapes = sim.base.attributes.get("array_shapes", None) or {}
    except Exception:  # noqa: BLE001 - attributes may be absent entirely
        return None
    if not shapes:
        return None
    total = 0
    for field in full_stack_fields():
        shape = shapes.get(aliases.get(field, field)) or shapes.get(field)
        if not shape:
            continue
        count = 1
        for dim in shape:
            if dim:
                count *= int(dim)
        total += 4 * count  # float32
    return total / float(2 ** 30)


def stats_version() -> int:
    from aiida_pino.console import viz

    return int(viz.STATS_VERSION)


def _stats_key() -> str:
    from aiida_pino.console import viz

    return str(viz.STATS_EXTRA_KEY)


def needs_stats(sim, *, version: int | None = None,
                max_stack_gib: float | None = None) -> bool:
    """True when this simulation owes a (re)compute.

    Absent stats and stale stats both qualify -- a STATS_VERSION bump is exactly
    when a recompute is owed. A tombstone suppresses it, but only for the version
    that failed.
    """
    v = stats_version() if version is None else version
    try:
        cached = sim.base.extras.get(_stats_key(), None)
    except Exception:  # noqa: BLE001 - extras may be absent entirely
        cached = None
    if isinstance(cached, dict) and cached.get("version") == v:
        return False
    try:
        dead = sim.base.extras.get(UNAVAILABLE_KEY, None)
    except Exception:  # noqa: BLE001
        dead = None
    if isinstance(dead, dict) and dead.get("version") == v:
        # A size skip is provisional -- it records the budget that rejected the
        # case, so raising the budget offers it again. A missing-field skip
        # records no budget and stays permanent for this version.
        limit = dead.get("max_stack_gib")
        budget = DEFAULT_MAX_STACK_GIB if max_stack_gib is None else max_stack_gib
        if limit is None or float(limit) >= float(budget):
            return False
    return True


def compute_one(sim, *, compute: Callable[[Any], dict] | None = None,
                version: int | None = None,
                max_stack_gib: float | None = None) -> bool:
    """Compute and store stats for one simulation. True when stats were written.

    A `KeyError` means the case lacks a field the metrics need -- a normal,
    permanent state for reduced-field datasets, not a failure. It is tombstoned
    rather than raised so the drain keeps moving and never revisits it.
    """
    v = stats_version() if version is None else version
    budget = DEFAULT_MAX_STACK_GIB if max_stack_gib is None else float(max_stack_gib)
    est = estimated_stack_gib(sim)
    if budget and est is not None and est > budget:
        # Tombstone WITHOUT calling compute: loading it is the thing that kills
        # the process, so the guard has to sit in front of the load.
        sim.base.extras.set(UNAVAILABLE_KEY, {
            "version": v,
            "reason": f"field stacks too large: {est:.2f} GiB > {budget:.2f} GiB budget",
            "max_stack_gib": budget, "est_stack_gib": round(est, 3),
            "at": _now()})
        return False
    if compute is None:
        from aiida_pino.console import viz

        def compute(s):  # noqa: E306
            return viz.get_or_compute_case_stats(s, recompute=True)

    try:
        stats = compute(sim)
    except KeyError as exc:
        sim.base.extras.set(UNAVAILABLE_KEY, {
            "version": v, "reason": f"missing field {exc.args[0]}",
            "at": _now()})
        return False
    if not isinstance(stats, dict):
        return False
    sim.base.extras.set(_stats_key(), stats)
    return True


def drain(sims: Iterable, *, compute: Callable[[Any], dict] | None = None,
          should_stop: Callable[[], bool] | None = None,
          max_stack_gib: float | None = None,
          log: Callable[[str], None] | None = None) -> int:
    """Compute stats for every simulation that needs them. Returns the count done.

    One bad case must not abort the batch: a sweep is hours of work and losing it
    to a single unexpected error would mean the whole thing is retried by hand.
    """
    done = 0
    for sim in sims:
        if should_stop is not None and should_stop():
            break
        if not needs_stats(sim, max_stack_gib=max_stack_gib):
            continue
        try:
            if compute_one(sim, compute=compute, max_stack_gib=max_stack_gib):
                done += 1
        except Exception as exc:  # noqa: BLE001
            if log:
                log(f"case {getattr(sim, 'pk', '?')} failed: "
                    f"{type(exc).__name__}: {exc}")
    return done


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _pending(limit: int, *, max_stack_gib: float | None = None) -> Sequence:
    """Simulations owing a compute, newest first so a fresh sweep is served first."""
    from aiida.orm import QueryBuilder
    from aiida_pino.data import PinoSimulationData

    q = QueryBuilder().append(PinoSimulationData, project=["*"])
    q.order_by({PinoSimulationData: {"ctime": "desc"}})
    out = []
    for (sim,) in q.iterall():
        if needs_stats(sim, max_stack_gib=max_stack_gib):
            out.append(sim)
        if len(out) >= limit:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=None, help="AiiDA profile (default: the configured default profile)")
    ap.add_argument("--poll-s", type=float, default=300.0)
    ap.add_argument("--batch", type=int, default=25,
                    help="cases per cycle; keeps one pass short so --once is useful")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--max-stack-gib", type=float, default=DEFAULT_MAX_STACK_GIB,
                    help="skip cases whose field stacks exceed this (0 disables)")
    ap.add_argument("--log", default=os.path.expanduser(
        "~/.flow3d/watch/stats_autocompute.log"))
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        line = f"{_now()}  {msg}"
        try:
            os.makedirs(os.path.dirname(args.log), exist_ok=True)
            with open(args.log, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:  # noqa: BLE001
            pass
        print(line, flush=True)

    from aiida import load_profile

    load_profile(args.profile)
    # Never compete with a FLOW-3D solve or a training run for CPU.
    try:
        os.nice(15)
    except Exception:  # noqa: BLE001
        pass
    log(f"started (version {stats_version()}, batch {args.batch}, "
        f"poll {args.poll_s:.0f}s, max stack {args.max_stack_gib:g} GiB, nice 15)")

    while True:
        try:
            todo = _pending(args.batch, max_stack_gib=args.max_stack_gib)
            if todo:
                log(f"{len(todo)} case(s) pending; computing")
                n = drain(todo, max_stack_gib=args.max_stack_gib, log=log)
                log(f"computed {n}/{len(todo)}")
            elif args.once:
                log("nothing pending")
        except Exception as exc:  # noqa: BLE001 - a DB blip must not kill the worker
            log(f"cycle error: {type(exc).__name__}: {exc}")
        if args.once:
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
