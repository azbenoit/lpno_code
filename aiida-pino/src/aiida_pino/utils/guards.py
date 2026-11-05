"""One switch for the memory and disk guards.

Every guard that acts on a memory or disk estimate -- the export pool clamp, the disk-fit
check before a runtime-cache build, the `host_ram_mb` floor on a training profile -- goes
through `resource_guard`. In `enforce` mode it refuses or clamps as it always did; in `warn`
mode it prints the same numbers as a warning and proceeds, because the thresholds are
measurements of one workstation (125 GB, 20 cores, a 1.8 TB root) and mean nothing on
another box.

The default is `"warn"`; set `RESOURCE_GUARDS_DEFAULT` below to `"enforce"` to make it refuse.
`AIIDA_PINO_RESOURCE_GUARDS=warn|enforce` overrides the default in any process (the daemon's copy comes from `daemon.env`).
"""

from __future__ import annotations

import os

RESOURCE_GUARDS_DEFAULT = "warn"

_MODES = ("enforce", "warn")


def resource_guard_mode() -> str:
    """`"enforce"` or `"warn"`; an unknown value falls back to the default, loudly."""
    raw = (os.environ.get("AIIDA_PINO_RESOURCE_GUARDS") or RESOURCE_GUARDS_DEFAULT).strip().lower()
    if raw not in _MODES:
        print(f"[resource-guard] AIIDA_PINO_RESOURCE_GUARDS={raw!r} is not one of {_MODES}; "
              f"using {RESOURCE_GUARDS_DEFAULT!r}", flush=True)
        return RESOURCE_GUARDS_DEFAULT
    return raw


def resource_guard(message: str, *, exc: type[Exception] = RuntimeError, log=None) -> bool:
    """Refuse (`enforce`) or warn and carry on (`warn`).

    Returns True when the caller may proceed, which is only ever in `warn` mode -- in
    `enforce` mode this raises `exc(message)`. `log` receives the warning line; default is
    stdout, which is where the adapters' and the daemon's other progress lines go.
    """
    if resource_guard_mode() == "enforce":
        raise exc(message)
    (log or (lambda text: print(text, flush=True)))(
        f"[resource-guard] WARNING (not enforced; AIIDA_PINO_RESOURCE_GUARDS=enforce would refuse): {message}"
    )
    return True
