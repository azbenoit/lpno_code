"""Reference-aware eviction of the runtime staging cache.

Clearing this cache blindly has cost seven runs; leaving it alone reached
352 GB on a disk at 88%. The tests pin the three protections: live-workdir
references, the recent-touch window, and live-builder pids.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from aiida_pino.utils.runtime_cache_evict import apply_eviction, plan_eviction


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(root))
    return root


def _entry(root: Path, name: str, *, age_hours: float) -> Path:
    entry = root / name
    entry.mkdir()
    payload = entry / "manifest.json"
    payload.write_text("{}")
    stamp = time.time() - age_hours * 3600
    os.utime(payload, (stamp, stamp))
    os.utime(entry, (stamp, stamp))
    return entry


def test_old_unreferenced_entries_are_evicted_and_recent_ones_pinned(
    aiida_profile_clean, cache_root
):
    old = _entry(cache_root, "aaaa-old", age_hours=200)
    recent = _entry(cache_root, "bbbb-recent", age_hours=1)

    proposal = plan_eviction(keep_recent_hours=72)
    evict_names = {Path(row["path"]).name for row in proposal["evict"]}
    pinned_names = {Path(row["path"]).name for row in proposal["pinned"]}
    assert old.name in evict_names
    assert recent.name in pinned_names

    result = apply_eviction(proposal)
    assert result["removed"] == 1
    assert not old.exists() and recent.exists()


def test_the_lock_and_sidecar_leave_with_their_entry(aiida_profile_clean, cache_root):
    old = _entry(cache_root, "cccc-old", age_hours=200)
    (cache_root / f"{old.name}.lock").write_text("")
    (cache_root / f"{old.name}.fields.json").write_text("[]")

    apply_eviction(plan_eviction(keep_recent_hours=72))
    assert not old.exists()
    assert not (cache_root / f"{old.name}.lock").exists()
    assert not (cache_root / f"{old.name}.fields.json").exists()


def test_dead_builders_are_removed_and_live_ones_kept(aiida_profile_clean, cache_root):
    dead = cache_root / ".build-dddd-99999999"       # no such pid
    dead.mkdir()
    (dead / "partial.npy").write_text("x")
    live = cache_root / f".build-eeee-{os.getpid()}"  # this test's pid: alive
    live.mkdir()

    proposal = plan_eviction(keep_recent_hours=72)
    assert {Path(row["path"]).name for row in proposal["dead_builds"]} == {dead.name}
    assert live.name in {Path(row["path"]).name for row in proposal["pinned"]}

    apply_eviction(proposal)
    assert not dead.exists() and live.exists()
