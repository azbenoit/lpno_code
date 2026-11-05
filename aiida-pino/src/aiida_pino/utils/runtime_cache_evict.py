"""Evict the runtime staging cache without killing anything that reads it.

The cache (`~/.cache/aiida-pino/runtime/<uuid>[-<discriminator>]`) is keyed by
collection uuid and nothing has ever evicted it: measured 2026-08-05 at **352 GB
across 719 entries on a disk that is 88% full** — the disk-OOM the cache
docstring warns about, arrived. Entries became real bytes (not hardlinks) the
day the container was packed, and they accumulate per uuid forever, including
for superseded collections and one-shot evaluation subsets.

Deleting blindly has a body count: clearing the cache mid-flight cost seven runs
(2026-07-22), because uploaded jobs symlink their workdirs here and read from it
hours later. So eviction is REFERENCE-AWARE, and the reference scan is the
point of this module:

* every non-terminal process's calcjob workdirs are walked for symlinks into
  the cache root — those targets are pinned;
* entries touched recently (default 72 h) are pinned, covering jobs submitted
  but not yet uploaded, whose cache was built moments before their task
  reached the queue;
* `.build-*` dirs of live pids are pinned (a build in progress); dead-pid
  builds are deleted (crashed or killed builders, never published).

Everything else is rebuildable by construction — `repair_dataset_dir` already
rebuilds a vanished cache at job start, and the export it re-runs now takes
minutes, not hours. Dry run by default.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from aiida import orm


def runtime_cache_root() -> Path:
    return Path(os.environ.get("AIIDA_PINO_RUNTIME_CACHE") or (Path.home() / ".cache" / "aiida-pino" / "runtime"))


def workdir_is_readable_here(transport_type: str | None) -> bool:
    """True when a calcjob's workdir is a path on THIS filesystem.

    Reuses the transport predicate that already decides where pack-index
    payloads can resolve, rather than inventing a second list that would drift
    from it.
    """
    from aiida_pino.data.pino_dataset_collection import pack_index_allowed_for_transport

    return pack_index_allowed_for_transport(transport_type)


def _live_workdir_targets(root: Path) -> set[Path]:
    """Cache dirs symlinked from any non-terminal process's calcjob workdir."""
    from aiida.orm import CalcJobNode, ProcessNode, QueryBuilder

    pinned: set[Path] = set()
    query = QueryBuilder().append(ProcessNode, project=["*"])
    for (node,) in query.all():
        state = getattr(node.process_state, "value", None)
        if state in ("finished", "excepted", "killed"):
            continue
        family = [node] + list(getattr(node, "called_descendants", []) or [])
        for member in family:
            if not isinstance(member, CalcJobNode):
                continue
            # A remote job's workdir is a path on ITS machine. `is_dir()` and
            # `iterdir()` below are local calls, so for a remote job they either
            # see nothing -- or, on a box that shares /home/<user> with this one
            # (the first compute box does), see a directory belonging to
            # something else entirely and walk a stranger's tree. Skip those
            # jobs: a remote job's symlinks point into the REMOTE cache, so it
            # pins nothing here, and this must never narrow what a LOCAL job
            # pins (clearing the cache mid-flight has already cost seven runs).
            if not workdir_is_readable_here(getattr(getattr(member, "computer", None), "transport_type", None)):
                continue
            try:
                workdir = Path(member.outputs.remote_folder.get_remote_path())
            except Exception:  # noqa: BLE001 - not uploaded yet
                continue
            if not workdir.is_dir():
                continue
            for entry in workdir.iterdir():
                if entry.is_symlink():
                    try:
                        target = Path(os.path.realpath(entry))
                    except OSError:
                        continue
                    if target.parent == root:
                        pinned.add(target)
    return pinned


def plan_eviction(*, keep_recent_hours: float = 72.0) -> dict[str, Any]:
    """What would be deleted, what is pinned, and why. Reads only."""
    root = runtime_cache_root()
    if not root.is_dir():
        return {"root": str(root), "entries": 0, "evict": [], "pinned": [], "dead_builds": []}

    pinned_targets = _live_workdir_targets(root)
    now = time.time()
    horizon = now - keep_recent_hours * 3600

    evict: list[dict[str, Any]] = []
    pinned: list[dict[str, Any]] = []
    dead_builds: list[dict[str, Any]] = []

    def _entry_size(path: Path) -> int:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    for entry in sorted(root.iterdir()):
        if entry.name.endswith(".lock") or entry.name.endswith(".fields.json"):
            continue  # sidecars ride along with their entry
        if entry.name.startswith(".build-"):
            pid = entry.name.rsplit("-", 1)[-1]
            alive = pid.isdigit() and Path(f"/proc/{pid}").is_dir()
            row = {"path": str(entry), "size": _entry_size(entry) if entry.is_dir() else 0, "pid": pid}
            (pinned if alive else dead_builds).append({**row, "reason": "builder alive" if alive else "builder dead"})
            continue
        if not entry.is_dir():
            continue
        size = _entry_size(entry)
        mtimes = [f.stat().st_mtime for f in entry.rglob("*") if f.is_file()]
        newest = max(mtimes, default=entry.stat().st_mtime)
        row = {"path": str(entry), "name": entry.name, "size": size, "age_h": (now - newest) / 3600}
        if entry in pinned_targets:
            pinned.append({**row, "reason": "referenced by a live job workdir"})
        elif newest > horizon:
            pinned.append({**row, "reason": f"touched within {keep_recent_hours:.0f} h"})
        else:
            evict.append(row)

    return {
        "root": str(root),
        "entries": len(evict) + len(pinned),
        "evict": evict,
        "pinned": pinned,
        "dead_builds": dead_builds,
        "evict_bytes": sum(row["size"] for row in evict),
        "pinned_bytes": sum(row["size"] for row in pinned),
    }


def apply_eviction(proposal: dict[str, Any]) -> dict[str, Any]:
    """Delete what the plan marked. The lock and sidecar go with each entry."""
    freed = 0
    removed = 0
    for row in proposal["evict"] + proposal["dead_builds"]:
        path = Path(row["path"])
        if not path.exists():
            continue
        shutil.rmtree(path, ignore_errors=True)
        for suffix in (".lock", ".fields.json"):
            sidecar = path.parent / f"{path.name}{suffix}"
            sidecar.unlink(missing_ok=True)
        freed += row["size"]
        removed += 1
    return {"removed": removed, "freed_bytes": freed}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="delete; without it, print the plan")
    parser.add_argument("--keep-recent-hours", type=float, default=72.0)
    parser.add_argument("--profile", default=None, help="AiiDA profile (default: the configured default profile)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from aiida import load_profile

    load_profile(args.profile)
    proposal = plan_eviction(keep_recent_hours=args.keep_recent_hours)
    if args.json:
        print(json.dumps(proposal, indent=1))
    else:
        print(f"cache root {proposal['root']}: {proposal['entries']} entries")
        print(f"  evict : {len(proposal['evict']):4d} entries, {proposal['evict_bytes']/1e9:7.1f} GB")
        print(f"  pinned: {len(proposal['pinned']):4d} entries, {proposal['pinned_bytes']/1e9:7.1f} GB")
        for row in proposal["pinned"]:
            if row["reason"] == "referenced by a live job workdir":
                print(f"    live: {row['name'][:52]}  {row['size']/1e9:.1f} GB")
        print(f"  dead builds: {len(proposal['dead_builds'])}")
    if not args.apply:
        if not args.json:
            print("\ndry run: nothing was deleted. Re-run with --apply.")
        return 0
    result = apply_eviction(proposal)
    print(f"removed {result['removed']} entries, freed {result['freed_bytes']/1e9:.1f} GB")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
