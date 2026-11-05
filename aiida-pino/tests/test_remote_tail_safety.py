"""Polling a remote job's log must not be able to hang the console.

`process_tail` opens a transport per call, in the web process. For a LOCAL
computer that is cheap. For a remote one it is an SSH connect, and an
unreachable box turns every poll of an open Monitor tab into a stalled request.

The bound available here is a short TTL cache: repeated polls of the same job
collapse onto one connection attempt, so an unreachable box costs one slow
request per TTL rather than one per poll. There is NO per-call connect timeout
in the transport API (`exec_command_wait` takes none), so the hard bound has to
come from the computer's configured timeout — which is why it is reported.
"""

from __future__ import annotations

from aiida_pino.console.tail_cache import TailCache


def test_a_repeated_poll_is_served_from_cache():
    calls = []
    cache = TailCache(ttl_seconds=60, clock=lambda: 100.0)

    def produce():
        calls.append(1)
        return {"tail": "hello"}

    assert cache.get_or_call(("pk", 1), produce)["tail"] == "hello"
    assert cache.get_or_call(("pk", 1), produce)["tail"] == "hello"
    assert len(calls) == 1, "the second poll must not open another connection"


def test_a_different_job_is_not_served_another_jobs_tail():
    cache = TailCache(ttl_seconds=60, clock=lambda: 100.0)
    cache.get_or_call(("pk", 1), lambda: {"tail": "one"})
    assert cache.get_or_call(("pk", 2), lambda: {"tail": "two"})["tail"] == "two"


def test_the_cache_expires():
    now = [100.0]
    calls = []
    cache = TailCache(ttl_seconds=5, clock=lambda: now[0])
    cache.get_or_call(("pk", 1), lambda: calls.append(1) or {"tail": "a"})
    now[0] = 106.0
    cache.get_or_call(("pk", 1), lambda: calls.append(1) or {"tail": "b"})
    assert len(calls) == 2


def test_a_failure_is_cached_too():
    """An unreachable box must not be retried on every single poll.

    Caching only successes means a hung connect is re-attempted every poll,
    which is the exact behaviour that makes an open Monitor tab dangerous.
    """
    calls = []
    cache = TailCache(ttl_seconds=60, clock=lambda: 100.0)

    def failing():
        calls.append(1)
        return {"tail": "", "note": "tail unavailable: TimeoutError"}

    cache.get_or_call(("pk", 9), failing)
    cache.get_or_call(("pk", 9), failing)
    assert len(calls) == 1


def test_the_cache_does_not_grow_without_bound():
    cache = TailCache(ttl_seconds=60, clock=lambda: 100.0, max_entries=3)
    for pk in range(10):
        cache.get_or_call(("pk", pk), lambda: {"tail": "x"})
    assert len(cache) <= 3
