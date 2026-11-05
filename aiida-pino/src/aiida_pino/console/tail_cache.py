"""A short-TTL cache in front of the remote log tail.

`process_tail` opens a transport per call, inside the web process. Locally that
is cheap; for a remote computer it is an SSH connect, and an unreachable box
turns every poll of an open Monitor tab into a stalled request — one per poll,
for as long as the tab is open.

There is no per-call connect timeout to reach for: the transport's
`exec_command_wait` takes none, and bounding it with a thread would put a
blocking call inside the request threadpool, which is the deadlock this project
already has an assumption about (A-THREADPOOL). So the bound here is
collapse-by-TTL: repeated polls of the same job share one attempt, and a
FAILURE is cached too — caching only successes would re-attempt a hung connect
on every poll, which is precisely the behaviour that makes the tab dangerous.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Hashable


class TailCache:
    """Tiny TTL cache. Not thread-safe by design — see the module docstring."""

    def __init__(self, *, ttl_seconds: float = 5.0, clock: Callable[[], float] = time.monotonic, max_entries: int = 256):
        self._ttl = float(ttl_seconds)
        self._clock = clock
        self._max_entries = int(max_entries)
        self._entries: dict[Hashable, tuple[float, Any]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def get_or_call(self, key: Hashable, produce: Callable[[], Any]) -> Any:
        """Cached value for `key`, or `produce()` stored under it."""
        now = self._clock()
        hit = self._entries.get(key)
        if hit is not None and (now - hit[0]) < self._ttl:
            return hit[1]
        value = produce()
        self._entries[key] = (now, value)
        self._evict(now)
        return value

    def _evict(self, now: float) -> None:
        for key, (stamped, _) in list(self._entries.items()):
            if (now - stamped) >= self._ttl:
                self._entries.pop(key, None)
        while len(self._entries) > self._max_entries:
            self._entries.pop(next(iter(self._entries)), None)
