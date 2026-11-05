"""Classify GPU/host-RAM seat leases so a dead holder's booking can be reclaimed safely.

`gpu_spool` books a seat by writing `/var/tmp/pino_gpu_leases/gpu<N>/<pid>-<nonce>.lease`,
and the holder is expected to remove it on exit. A signalled seat wrapper never does, so the
booking outlives the process: measured 2026-09-05, a dead pid held 65536 MiB of host RAM and
16000 MiB of VRAM, the box reported 73728 MiB booked against 11 GB available, and five queued
evaluations could not be admitted to a machine that was actually 83 GB free.

That matters more since 2026-08-06 (`f90778e`), when the cgroup cap became opt-in and OFF:
admission now packs purely on DECLARED numbers, so a phantom booking shrinks the box for every
session and nothing reclaims it on its own.

Reaping is not `rm` on a failed `kill -0`, because pids are recycled. The discriminator is the
holder's start time against the lease's own `since`: a process that started AFTER the lease was
written cannot be the process that wrote it, so the original holder is gone even though the pid
resolves. Everything ambiguous is KEPT -- a phantom booking wastes capacity, while deleting a
live lease double-books a card and corrupts state shared with other processes.
"""
from __future__ import annotations

import datetime as _dt
import enum
from typing import Any, Mapping


class LeaseVerdict(enum.Enum):
    """What to do with one lease file. Only `reapable` verdicts may be deleted."""

    LIVE = "live"                      # holder is running and predates the lease
    STALE_DEAD = "stale_dead"          # holder pid does not exist
    STALE_REUSED = "stale_reused"      # pid exists but started after the lease was written
    UNPARSEABLE = "unparseable"        # cannot be reasoned about -> never delete

    @property
    def reapable(self) -> bool:
        return self in (LeaseVerdict.STALE_DEAD, LeaseVerdict.STALE_REUSED)


def _parse_time(value: Any) -> _dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def classify_lease(
    lease: Mapping[str, Any] | None,
    *,
    pid_alive: bool,
    pid_started: str | None,
    grace_s: float = 60.0,
) -> LeaseVerdict:
    """Decide the fate of one lease.

    `lease` is the parsed lease JSON. `pid_alive` is whether the pid currently resolves.
    `pid_started` is that pid's start time as an ISO string, or None when it could not be
    read -- which is treated as "cannot prove recycling" and therefore keeps the lease.
    """
    if not isinstance(lease, Mapping):
        return LeaseVerdict.UNPARSEABLE
    pid = lease.get("pid")
    if not isinstance(pid, int):
        return LeaseVerdict.UNPARSEABLE
    since = _parse_time(lease.get("since"))
    if since is None:
        # Without a timestamp a recycled pid is indistinguishable from the real holder,
        # so this lease can never be shown to be stale. Keep it.
        return LeaseVerdict.UNPARSEABLE
    if not pid_alive:
        return LeaseVerdict.STALE_DEAD
    started = _parse_time(pid_started)
    if started is None:
        return LeaseVerdict.LIVE
    if started.tzinfo is None and since.tzinfo is not None:
        started = started.replace(tzinfo=since.tzinfo)
    if since.tzinfo is None and started.tzinfo is not None:
        since = since.replace(tzinfo=started.tzinfo)
    # Start times come from `ps -o etimes=` at 1 s granularity and the holder writes
    # its lease a moment AFTER it starts, so a legitimate holder can read as a second
    # or two late. A recycled pid is minutes or hours late, never seconds, so a grace
    # window separates the two without weakening the discriminator.
    return (
        LeaseVerdict.STALE_REUSED
        if (started - since).total_seconds() > grace_s
        else LeaseVerdict.LIVE
    )
