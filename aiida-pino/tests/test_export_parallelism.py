"""The runtime-cache export runs members in parallel, but bounded.

Measured 2026-08-05: exporting a 120-simulation dataset ran strictly one case at
a time in `do_upload`, pinning ONE core at 99% (84942 jiffies on the main thread,
7 on every other) while 19 cores and BOTH GPUs sat idle and two trainings waited
hours on it. The work is `zlib` decompression of NPZ members -- per-case
independent, each writing its own `member_root`.

Threads rather than processes: zlib releases the GIL during decompression, which
is the dominant cost, and threads avoid pickling AiiDA nodes or opening a DB
connection per worker.

The bound matters as much as the parallelism. This box is shared with FLOW-3D
solves and ML trainings, and the machine budget leaves at least 4 cores and
20 GB for the OS and interactive login -- an export that grabbed every core would
reproduce the 2026-07-09 lockout it exists to avoid.
"""

from __future__ import annotations


import pytest

from aiida_pino.utils.export_pool import export_worker_count


@pytest.fixture(autouse=True)
def no_job_memory_limit(monkeypatch):
    """Take the enclosing cgroup's memory limit out of every test in this file.

    `export_worker_count` also caps the count by `cgroup_memory_headroom()`, the live
    room left in the memory cgroup the test process runs in. A cluster login node or a
    batch job sets a per-user limit of a few GB, so with the cap left live
    `export_worker_count(cpu_count=20)` returns 1 there and the core and env rules
    below cannot be observed. The cgroup rule has its own tests in
    `test_export_job_limits.py`.
    """
    from aiida_pino.utils import export_pool

    monkeypatch.setattr(export_pool, "cgroup_memory_headroom", lambda *args, **kwargs: None)


@pytest.fixture
def plenty_of_memory(monkeypatch):
    """Pin the memory bound out of the way so the CORE rule can be tested alone.

    `export_worker_count` is bounded by cores AND by live
    `psutil.virtual_memory().available`, so a test that passes `cpu_count=20`
    still reads whatever the machine happens to have free. Measured 2026-08-15: a
    leaked orphan process held 86.6 GB, MemAvailable fell to 8.3 GB, the memory
    branch returned 1, and `test_leaves_headroom_on_a_big_box` failed with
    `assert 1 < 1` -- a green/red result decided by an unrelated process. The
    memory rule keeps its own test below; this fixture is what makes the two
    rules separable.
    """
    import sys
    import types

    fake = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=200e9)
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.delenv("AIIDA_PINO_EXPORT_WORKERS", raising=False)


def test_leaves_headroom_on_a_big_box(plenty_of_memory):
    """20 cores -> never more than cpu-4, and capped so it stays polite."""
    n = export_worker_count(cpu_count=20)
    assert 1 < n <= 16
    assert n <= 20 - 4, "must leave >=4 cores for the OS and interactive login"


def test_never_returns_less_than_one():
    for cpus in (1, 2, 3, 4, 5):
        assert export_worker_count(cpu_count=cpus) >= 1


def test_small_box_does_not_go_parallel_aggressively():
    """A 4-core box must not spawn 4 exporters beside a training."""
    assert export_worker_count(cpu_count=4) == 1


def test_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv("AIIDA_PINO_EXPORT_WORKERS", "3")
    assert export_worker_count(cpu_count=20) == 3


def test_env_override_is_clamped_not_trusted(monkeypatch):
    """A careless override must not be able to swamp the machine."""
    monkeypatch.setenv("AIIDA_PINO_EXPORT_WORKERS", "999")
    assert export_worker_count(cpu_count=20) <= 16
    monkeypatch.setenv("AIIDA_PINO_EXPORT_WORKERS", "0")
    assert export_worker_count(cpu_count=20) >= 1
    monkeypatch.setenv("AIIDA_PINO_EXPORT_WORKERS", "not-a-number")
    assert export_worker_count(cpu_count=20) >= 1


def test_serial_when_only_one_member():
    """One member must not pay thread-pool setup."""
    assert export_worker_count(cpu_count=20, n_members=1) == 1


def test_worker_count_is_bounded_by_memory_not_just_cores(monkeypatch):
    """A 3.3 GB-per-member job must not be sized by core count.

    Measured 2026-08-05: one member of a moving-frame dataset decompresses to 3.3 GB. The
    core-derived bound returned 16 on a 20-core machine, i.e. ~53 GB of inputs
    alone -- the shape of the 2026-07-09 swap-thrash lockout.
    """
    import types
    fake = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(available=40e9)  # 40 GB free
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
    monkeypatch.delenv("AIIDA_PINO_EXPORT_WORKERS", raising=False)
    n = export_worker_count(cpu_count=20, n_members=120)
    # (40 - 20 reserved) / 4 GB per member = 5, far below the 16 cores allow
    assert n <= 5, f"memory bound not applied (got {n})"
    assert n >= 1


def test_headroom_rule_is_not_decided_by_the_live_box(monkeypatch):
    """Regression guard for 2026-08-15: the core rule must survive a starved box.

    A leaked orphan held 86.6 GB, MemAvailable fell to 8.3 GB, and
    `test_leaves_headroom_on_a_big_box` failed with `assert 1 < 1` -- red because
    of an unrelated process. Both branches are asserted here explicitly, so the
    memory bound winning on a starved box is a TESTED behaviour rather than an
    invisible cause of flakiness.
    """
    import sys
    import types

    def _mem(available_bytes):
        return types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(available=available_bytes)
        )

    monkeypatch.delenv("AIIDA_PINO_EXPORT_WORKERS", raising=False)
    monkeypatch.setitem(sys.modules, "psutil", _mem(8.3e9))
    assert export_worker_count(cpu_count=20) == 1, "starved box must serialise"
    monkeypatch.setitem(sys.modules, "psutil", _mem(200e9))
    assert export_worker_count(cpu_count=20) > 1, "roomy box must parallelise"
