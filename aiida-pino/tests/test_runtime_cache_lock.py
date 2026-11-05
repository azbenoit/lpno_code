"""Concurrent submits must not each rebuild the same runtime cache.

Measured 2026-08-05: five `.build-*` directories existed simultaneously for ONE
dataset (97c7cb1d) from five different worker PIDs -- 342/431/90/342/268 MB, about
1.47 GB of duplicated work that all but one builder throws away at
`build.replace(final)`. Two trainings both needed the same moving-frame dataset, so the
same multi-hour export ran twice concurrently.

Each simulation in that dataset decompresses to 3.3 GB (a 49.8 MB NPZ expanding
66.8x), so duplicated builds are expensive in RAM and page cache, not just CPU.

The lock is `flock`-based on purpose: the kernel releases it when the holder
exits, so a killed or crashed builder cannot wedge every future submit.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from aiida_pino.calculations.base import runtime_cache_build_lock


def _hold(lock_root: str, name: str, seconds: float, started, done):
    with runtime_cache_build_lock(Path(lock_root), name):
        started.set()
        time.sleep(seconds)
    done.set()


def test_second_builder_waits_for_the_first(tmp_path):
    """Two processes, one cache name: the second must not build concurrently."""
    ctx = mp.get_context("fork")
    started, done = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold, args=(str(tmp_path), "ds-a", 1.5, started, done))
    holder.start()
    assert started.wait(timeout=10), "first builder never acquired"

    t0 = time.monotonic()
    with runtime_cache_build_lock(tmp_path, "ds-a"):
        waited = time.monotonic() - t0
    holder.join(timeout=10)
    assert waited > 0.5, f"second builder did not wait (waited {waited:.2f}s)"


def test_different_datasets_do_not_block_each_other(tmp_path):
    """The lock is per cache name, not global -- unrelated exports run in parallel."""
    ctx = mp.get_context("fork")
    started, done = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold, args=(str(tmp_path), "ds-a", 2.0, started, done))
    holder.start()
    assert started.wait(timeout=10)

    t0 = time.monotonic()
    with runtime_cache_build_lock(tmp_path, "ds-b"):
        waited = time.monotonic() - t0
    holder.join(timeout=10)
    assert waited < 0.5, f"unrelated dataset was blocked for {waited:.2f}s"


def test_lock_is_released_when_the_holder_dies(tmp_path):
    """A killed builder must not wedge every future submit.

    Workers were killed repeatedly during the 2026-08-05 incident; a lock that
    outlived them would have been worse than no lock at all.
    """
    ctx = mp.get_context("fork")
    started, done = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold, args=(str(tmp_path), "ds-c", 30.0, started, done))
    holder.start()
    assert started.wait(timeout=10)
    holder.kill()
    holder.join(timeout=10)

    t0 = time.monotonic()
    with runtime_cache_build_lock(tmp_path, "ds-c"):
        waited = time.monotonic() - t0
    assert waited < 1.0, "flock was not released on holder death"


def test_lock_file_lives_outside_the_cache_directory(tmp_path):
    """It must survive the cache dir being replaced or removed."""
    with runtime_cache_build_lock(tmp_path, "ds-d"):
        pass
    assert (tmp_path / "ds-d.lock").exists()
    assert not (tmp_path / "ds-d").exists(), "lock must not create the cache dir"
