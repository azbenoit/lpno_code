"""A failed derived export must leave no worker still writing, and no retry may inherit its files.

2026-09-10/11, three GAOT arms on one dataset (61 members, conduction_field baked, ~7 min a
case): the export's `ProcessPoolExecutor` went BROKEN (a worker died abruptly), and CPython
3.10's `terminate_broken` then aborted on `InvalidStateError` -- `pool.map`'s result iterator
had already cancelled the pending futures, `set_exception` on a cancelled future raises, and
the loop never reached `p.terminate()`. So the surviving workers kept building, each writing
`<case>.npz` into `.build-<cache>-<PARENT PID>`. AiiDA retried presubmit in the same parent
seven minutes later, reused that exact path, and its sequential pass found `mbc40_001.npz`
already there: `ValueError: Duplicate case name` -- a filesystem check reporting a zombie's
write as a membership defect. Every membership view checked was clean,
because the membership was clean. 14 h later the leftover dirs held 6.7 GB and were still
being written.

Two guarantees, each tested against the OUTCOME rather than the mechanism:
* after `_run_derived_pool` raises, no child process it started is alive;
* every `_dataset_runtime_cache` attempt gets a build dir nothing else can be writing to,
  and dead attempts of the same cache are swept before the new one starts.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from aiida_pino.calculations.base import _fresh_build_dir
from aiida_pino.utils.export_pool import _run_derived_pool


def _worker(task):
    """Task 0 dies abruptly (a broken pool); every other task would run for a long time."""
    if task["index"] == 0:
        os._exit(1)
    time.sleep(60)
    return task


def test_a_broken_pool_leaves_no_straggler_worker():
    before = {p.pid for p in multiprocessing.active_children()}
    tasks = [{"index": i} for i in range(6)]
    with pytest.raises(Exception):
        _run_derived_pool(tasks, workers=3, worker=_worker)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        stragglers = [p for p in multiprocessing.active_children() if p.pid not in before and p.is_alive()]
        if not stragglers:
            break
        time.sleep(0.2)
    assert not stragglers, f"workers still alive after the pool failed: {[p.pid for p in stragglers]}"


def test_a_healthy_pool_still_returns_every_result():
    tasks = [{"index": i, "v": i * i} for i in range(4)]
    assert _run_derived_pool(tasks, workers=2, worker=_square) == [0, 1, 4, 9]


def _square(task):
    return task["v"]


def test_each_attempt_gets_its_own_build_dir_and_dead_ones_are_swept():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        stale = root / "cafe-0000.build-999999-deadbeef"
        (stale / "members").mkdir(parents=True)
        (stale / "mbc40_001.npz").write_bytes(b"zombie wrote this")
        legacy = root / ".build-cafe-0000-999998"   # the retired pid-only spelling
        legacy.mkdir()
        other = root / ".build-OTHERCACHE-1-abcd"
        other.mkdir()
        first = _fresh_build_dir(root, "cafe-0000")
        second = _fresh_build_dir(root, "cafe-0000")
        assert first != second, "two attempts must never share a path"
        assert first.parent == root and second.parent == root
        assert not stale.exists(), "a dead attempt of the SAME cache is swept"
        assert not legacy.exists(), "so is one left by the pid-only naming"
        assert other.exists(), "another cache's build dir is not ours to touch"
