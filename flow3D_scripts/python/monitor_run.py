#!/usr/bin/env python3
"""Runtime stall / divergence monitor for a FLOW-3D solve.

Watches a case's solver progress table -- pass the solver's own **hd3msg**
file, NOT ``runhyd.log`` (the latter is a block-buffered stdout capture that lags
the solver and triggers false stalls; hd3msg is written directly) -- and aborts
the solve when it
stops making progress or blows up, so a stuck or diverging run (a variable-speed
dwell that starts to drill, a time-step collapse) frees the solver lock
instead of burning the whole per-case timeout. The queue then moves on.

Run it as a sidecar to the solver (see the ti64_varspeed150 wrapper); on
detection it SIGKILLs the FLOW-3D solver chain. Progress rows in ``runhyd.log``
are (columns: ``sim_time cycle delt dt_stbl iter res vol %loss ...``)::

   4.47320E-04  1553  2.73E-07  2.73E-07/st  2  8.06E-15  4.5000E-10 +4.60E-08 ...

Abort signals:
  * **stall** -- ``sim_time`` has not advanced by ``--min-progress`` seconds of
    simulated time within ``--stall-s`` wall seconds (covers hangs and a coarse
    time-step collapse).
  * **divergence** -- ``delt`` has collapsed below ``--delt-min`` once past
    ``--min-cycle`` (catches a fast dt-collapse that still creeps ``sim_time``
    forward and so would slip past the stall check), or a non-finite / overflowed
    (``NaN``/``Inf``/``****``) value appears in the log.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import signal
import sys
import time
from dataclasses import dataclass

# runhyd.log parsing matches check_run.py: sim_time=t[0], cycle=t[1], delt=t[2],
# %loss=t[7]. Kept in sync deliberately (same column contract).
# NaN/Inf tokens never occur in the FLOW-3D startup banner, so they are matched
# anywhere. The '****' Fortran field-overflow marker, however, IS the banner's own
# decoration (rows of '*' and '**** executing ...' head every runhyd.log), so it is
# checked separately and only inside a data row -- see has_naninf().
_NANINF = re.compile(r"\bnan\b|[-+]?\binf(inity)?\b", re.IGNORECASE)
_OVERFLOW = re.compile(r"\*{4,}")


@dataclass
class Progress:
    sim_time: float
    cycle: float
    delt: float
    pct_loss: float


def parse_progress(text: str) -> list[Progress]:
    """Solver progress rows from runhyd.log text; header/banner lines are skipped."""
    rows: list[Progress] = []
    for line in text.splitlines():
        t = line.split()
        if len(t) < 8:
            continue
        try:
            rows.append(Progress(float(t[0]), float(t[1]), float(t[2]), float(t[7])))
        except ValueError:
            continue  # header row / non-numeric column
    return rows


def has_naninf(text: str) -> bool:
    """True if runhyd.log shows a NaN/Inf token, or an overflowed (****) field in a
    solver PROGRESS row.

    The '****' marker is restricted to data rows (first column parses as a numeric
    sim_time): the FLOW-3D startup banner is full of decorative '****' separators
    present in every log, and scanning it globally SIGKILLed every healthy run.
    """
    if _NANINF.search(text):
        return True
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        try:
            float(tok[0])
        except ValueError:
            continue  # banner / header / non-data line
        if _OVERFLOW.search(line):
            return True
    return False


def diagnose(rows, wall_since_progress, stall_s, delt_min, min_cycle):
    """Return ``(kill, reason)``. Pure: all timing is passed in, nothing is read.

    ``wall_since_progress`` is the wall seconds since ``sim_time`` last advanced by
    the caller's ``min_progress`` threshold.
    """
    if not rows:
        return (False, "")
    last = rows[-1]
    if not (math.isfinite(last.delt) and math.isfinite(last.sim_time)):
        return (True, f"divergence: non-finite solver output (delt={last.delt})")
    if last.cycle >= min_cycle and last.delt < delt_min:
        return (True, f"divergence: delt collapsed to {last.delt:.2e} (< {delt_min:.0e}) "
                      f"at cycle {int(last.cycle)}")
    if wall_since_progress > stall_s:
        return (True, f"stall: sim_time stuck at {last.sim_time:.3e} s for "
                      f"{wall_since_progress:.0f}s (> {stall_s:.0f}s)")
    return (False, "")


SOLVE_MARK = "run_rectangular_case"  # marks a real FLOW-3D solve in until_pid's cmdline


def _cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _proc_starttime(pid):
    """Field 22 (starttime, jiffies since boot) of /proc/<pid>/stat. Unique to a
    process *instance*, so it changes when the PID is reused. None if unreadable."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
        return data[data.rindex(")") + 2:].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def _descendants(pid):
    """PIDs in the subtree rooted at ``pid`` (children first, ``pid`` last)."""
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                ppid = int(fh.read().split()[3])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    out, stack = [], [pid]
    while stack:
        for ch in children.get(stack.pop(), []):
            stack.append(ch)
            out.append(ch)
    out.append(pid)
    return out


def kill_solver(until_pid) -> None:
    """SIGKILL only the solve subtree rooted at ``until_pid`` -- never a global sweep.

    Gated on ``until_pid`` still being a real solve (cmdline carries ``SOLVE_MARK``):
    a stale monitor whose ``until_pid`` is dead or reused by a non-solve process kills
    nothing, so one case's monitor can never kill another case's solve (the 2026-07-15
    global-pkill churn that killed ~57% of healthy runs)."""
    if until_pid is None or SOLVE_MARK not in _cmdline(until_pid):
        return
    for pid in _descendants(until_pid):  # children before the root
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


SOLVE_DONE_MARK = "date of completion"  # written to hd3msg when the solver finishes


def solve_completed(text: str) -> bool:
    """True once the solver has written its completion block to hd3msg.

    The runner stays alive after the solve to export the NPZ, which can take many
    minutes on a large case, and no progress rows are written during it -- so the
    stall clock would otherwise run against a ``sim_time`` that is finished, not
    stuck.  That killed t64v_019 at frame 301/301 of its export on 2026-07-21.
    """
    return SOLVE_DONE_MARK in text


def watch(logpath, stall_s, delt_min, poll_s, min_progress, min_cycle,
          until_pid=None, clock=time.time, sleep=time.sleep, killer=kill_solver,
          max_solve_s=None):
    """Poll ``logpath``; kill the solve on stall/divergence.

    ``max_solve_s`` bounds the SOLVE, not the queue: the clock starts at the
    first solver output, which only exists once the seat lock is held — so a
    run queued hours behind another is never charged for the wait. This is what
    lets a per-case wall timeout be turned on at all; wrapping the whole runner
    in ``timeout`` charged the seat wait to the solve and had to stay off.

    Returns 0 if the solve ended on its own (``until_pid`` died), 1 if the monitor
    aborted it.
    """
    best_sim = -math.inf
    last_progress_wall = clock()
    solve_start_wall = None
    start_id = _proc_starttime(until_pid) if until_pid is not None else None
    while True:
        # Exit if the watched solve ended: PID gone, or reused by a different process
        # (starttime changed). Stops a stale monitor from firing on a later, unrelated
        # solve -- the root of the 2026-07-15 churn.
        if until_pid is not None and (
            not _alive(until_pid) or _proc_starttime(until_pid) != start_id
        ):
            return 0
        try:
            with open(logpath, errors="ignore") as fh:
                text = fh.read()
        except FileNotFoundError:
            text = ""
        # The solve is over; whatever the runner is still doing (NPZ export) is not
        # something this monitor can judge, and no further progress rows will appear.
        if solve_completed(text):
            return 0
        rows = parse_progress(text)
        now = clock()
        if text and solve_start_wall is None:
            solve_start_wall = now
        if (max_solve_s and solve_start_wall is not None
                and now - solve_start_wall > max_solve_s):
            print(f"[monitor] ABORT {logpath}: solve wall timeout: "
                  f"{now - solve_start_wall:.0f}s since first solver output "
                  f"(> {max_solve_s:.0f}s; seat wait excluded)", flush=True)
            killer(until_pid)
            return 1
        if rows and rows[-1].sim_time >= best_sim + min_progress:
            best_sim = rows[-1].sim_time
            last_progress_wall = now
        elif rows and rows[-1].sim_time < best_sim - min_progress:
            # sim_time went BACKWARDS: a solver never rewinds within one run, so the
            # file was recreated by a NEW run (requeued case overwriting the previous
            # attempt's hd3msg). Re-anchor instead of counting the fresh run as a
            # stall against the stale baseline (false ABORT of t64k_045, 2026-07-17).
            best_sim = rows[-1].sim_time
            last_progress_wall = now
        kill, reason = diagnose(rows, now - last_progress_wall, stall_s, delt_min, min_cycle)
        if not kill and has_naninf(text):
            kill, reason = True, f"divergence: NaN/Inf/overflow in {os.path.basename(logpath)}"
        if kill:
            print(f"[monitor] ABORT {logpath}: {reason}", flush=True)
            killer(until_pid)
            return 1
        sleep(poll_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Abort a stalled/diverging FLOW-3D solve.")
    ap.add_argument("logpath", help=(
        "Path to the solver's own hd3msg.<case> progress file (NOT runhyd.log, "
        "which is a block-buffered stdout capture that lags and triggers false stalls)"))
    ap.add_argument("--stall-s", type=float, default=900.0,
                    help="Abort if sim_time makes no progress for this many wall seconds")
    ap.add_argument("--delt-min", type=float, default=1e-9,
                    help="Abort if delt collapses below this (past --min-cycle)")
    ap.add_argument("--min-progress", type=float, default=1e-6,
                    help="Simulated seconds that count as real sim_time progress")
    ap.add_argument("--min-cycle", type=float, default=100,
                    help="Only apply the delt-collapse test past this cycle (skip startup ramp)")
    ap.add_argument("--poll-s", type=float, default=30.0, help="Poll interval (s)")
    ap.add_argument("--until-pid", type=int, default=0,
                    help="Exit when this PID (the solve) dies")
    ap.add_argument("--max-solve-s", type=float, default=0.0,
                    help="Abort the solve this many wall seconds after its FIRST "
                         "output (0 = off). Seat-lock wait is excluded by "
                         "construction: the log only exists once the solver lock is held.")
    a = ap.parse_args(argv)
    return watch(a.logpath, a.stall_s, a.delt_min, a.poll_s, a.min_progress,
                 a.min_cycle, a.until_pid or None, max_solve_s=a.max_solve_s or None)


if __name__ == "__main__":
    sys.exit(main())
