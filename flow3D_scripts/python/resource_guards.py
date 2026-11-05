#!/usr/bin/env python3
"""One switch for every memory/disk guard: enforce or warn.

A guard calls ``resource_guard(message)`` at the point where it has decided to
refuse. Under ``enforce`` that raises (``RuntimeError`` unless the caller passes
another class); under ``warn`` it prints one line,
``[resource-guard] WARNING: <message> -- not enforced (F3D_RESOURCE_GUARDS=warn)``,
and returns, so the caller carries on. ``mode()`` is the same switch for a guard
that has to decide earlier -- the runner's floors stop WAITING under ``warn``,
because a 30-minute hold for disk that may never clear on a reader's laptop is the
complication being removed, not a safeguard.

The mode is ``F3D_RESOURCE_GUARDS`` (``enforce`` or ``warn``); unset means
``DEFAULT_MODE``. This copy ships ``"warn"``, since memory and disk limits are
hardware-dependent. On a shared machine, ``enforce`` makes the 30 GB disk floor and
the 25 GB MemAvailable gate binding: a full disk stalls the solver mid-write, and low
MemAvailable can lock out SSH logins. The default is a literal, set in the source,
because a daemon or spool worker restarted from a shell without the variable would
otherwise silently drop the enforcement.

Only refusals on a RAM or disk figure route through here. The seat lock, the pause
switch and the flsgrf cleanup are not resource guards and never warn.
"""

from __future__ import annotations

import os
import sys

DEFAULT_MODE = "warn"   # the public copy is assembled with "warn" here (README, "Disk hygiene")
ENV = "F3D_RESOURCE_GUARDS"
MODES = ("enforce", "warn")


def mode() -> str:
    value = os.environ.get(ENV, DEFAULT_MODE).strip().lower()
    if value not in MODES:
        raise SystemExit(f"{ENV} must be one of {', '.join(MODES)}; got {value!r}")
    return value


def resource_guard(message: str, exc: type[BaseException] = RuntimeError) -> None:
    """Refuse (raise ``exc``) under enforce; print a warning and return under warn."""
    if mode() == "enforce":
        raise exc(message)
    print(f"[resource-guard] WARNING: {message} -- not enforced ({ENV}=warn)",
          file=sys.stderr, flush=True)
