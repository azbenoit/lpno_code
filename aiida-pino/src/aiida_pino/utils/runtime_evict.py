"""Delete runtime dataset caches that no in-flight job is holding.

`_dataset_runtime_cache` publishes one directory per node uuid under
``~/.cache/aiida-pino/runtime`` and nothing ever removes one, so the tree grows
by a full dataset every time a new collection is trained on. That was nearly
free while the disk-objectstore container held loose objects —
`_try_link_repository_object` hardlinked the payloads — but a packed object has
no standalone file to link to and the export falls back to a streamed copy. After
the 2026-07-24 pack the cache held 51 GB of genuine copies (247 npz, mean
219 MB) of data already in the packs.

Eviction is only safe for entries no job still reads. `stage_dataset_input`
symlinks a job's workdir here at UPLOAD time and the job can then sit hours in
the gpu_spool seat queue before reading it, so deleting a live entry kills the
run — that cost seven runs on 2026-07-22/23. Two guards, both required:

* an entry whose uuid is an input of a non-terminal process is never evicted;
* an entry touched within ``grace_hours`` is never evicted, covering the window
  between export and staging, where nothing references it yet.

Evicting is not destructive beyond the disk cost: the next submit re-exports,
and `runtime_repair` rebuilds an entry that vanished under an uploaded job.

Usage — dry-run first, it prints what it would remove::

    python -m aiida_pino.utils.runtime_evict
    python -m aiida_pino.utils.runtime_evict --apply
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from aiida_pino.utils.runtime_repair import _UUID_RE, runtime_cache_root

#: Transient dirs left by a crashed export; the uuid is the middle field.
_TRANSIENT_PREFIXES = (".build-", ".repair-")

#: Process states that mean a job may still read its staged cache.
ACTIVE_PROCESS_STATES = ("created", "running", "waiting")


def _entry_uuid(name: str) -> str | None:
    """The node uuid an entry belongs to, or None if the name is unrelated.

    Parses through `cache_identity`, which already understands
    `<uuid>[-<discriminator>]`, rather than re-deriving it with an anchored
    uuid regex. The re-derivation could not see a DISCRIMINATED name at all: a
    field-trimmed cache is `<uuid>-<hash12>`, and both the published entry and
    its `.build-`/`.repair-` residue returned None here, so this tool skipped
    them at any age. That is a leak in the documented reclaim path
    (`python -m aiida_pino.utils.runtime_evict`) on a box where disk is a hard
    resource. It fails closed -- nothing live is ever deleted -- which is why it
    went unnoticed.
    """
    from aiida_pino.utils.runtime_repair import cache_identity

    identity = cache_identity(name)
    if identity is not None:
        return identity[0]
    for prefix in _TRANSIENT_PREFIXES:
        if name.startswith(prefix):
            # ".build-<uuid>[-<disc>]-<pid>" -> <uuid>
            candidate = name[len(prefix) :].rsplit("-", 1)[0]
            identity = cache_identity(candidate)
            return identity[0] if identity is not None else None
    return None


def entry_size(path: Path) -> int:
    """Bytes an entry occupies, counting each inode once.

    Hard-linked payloads (a container with loose objects) cost no extra disk, so
    counting them per name would overstate what eviction actually frees.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                stat = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            if stat.st_nlink > 1:
                key = (stat.st_dev, stat.st_ino)
                if key in seen:
                    continue
                seen.add(key)
            total += stat.st_size
    return total


def stale_entries(
    root: Path,
    in_use: set[str],
    *,
    grace_hours: float = 6.0,
    now: float | None = None,
    evicted_sources: set[str] | None = None,
) -> list[Path]:
    """Cache entries safe to delete, sorted by path.

    Pure: takes the in-use uuid set rather than querying, so it is testable
    without a profile.

    `evicted_sources` is the one exception to this tool's premise. The premise --
    a cache entry is not data, it re-exports from the object store, so deleting
    one costs time rather than loss -- is why deleting 964 entries to free
    55.3 GB was safe on 2026-08-21. But `evict_dataset_data` deletes a dataset's
    array data while keeping its metadata and provenance, which is documented
    practice for reclaiming disk. **A cache built from a collection that has
    since been evicted cannot be re-exported: there is nothing left to export
    from.** Deleting it destroys the last usable copy of those bytes.

    Opt-in and defaulting to empty, so a caller that cannot query a profile
    keeps exactly the previous behaviour rather than silently changing it.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    cutoff = (time.time() if now is None else now) - grace_hours * 3600.0
    stale = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        uuid = _entry_uuid(child.name)
        if uuid is None or uuid in in_use:
            continue
        if evicted_sources and uuid in evicted_sources:
            continue
        try:
            if child.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        stale.append(child)
    return stale


def in_use_uuids() -> set[str]:
    """Uuids of everything a non-terminal process holds: its inputs, its outputs, and the
    inputs and outputs of every process it CALLED, whatever state those children are in.

    Direct inputs alone missed the one that mattered. `TrainModelWorkChain` resolves a
    derived collection, runs a short VRAM smoke CalcJob on it (whose upload builds the
    runtime cache), waits for it to FINISH, then creates the real training CalcJob on the
    same collection. In between, the collection is an input of a terminal child and a ctx
    value of the live parent -- and the cache dir is keyed by exactly that uuid. On
    2026-09-11 a disk guard running this tool every few minutes evicted that entry each
    cycle, re-deriving 4.8 GB for ~1 h per job (60.2 GB, three times, measured).
    A live parent holds what its children consumed and produced.
    """
    from aiida import load_profile, orm
    from aiida.orm import QueryBuilder

    load_profile()
    active = {"attributes.process_state": {"in": list(ACTIVE_PROCESS_STATES)}}
    held: set[str] = set()

    # Direct inputs and outputs of a live process.
    for direction in ("with_outgoing", "with_incoming"):
        builder = QueryBuilder()
        builder.append(orm.ProcessNode, filters=active, tag="proc")
        builder.append(orm.Node, **{direction: "proc"}, project="uuid")
        held.update(uuid for (uuid,) in builder.iterall())

    # Inputs and outputs of every process a live one CALLED, transitively, in any state.
    # Walked level by level over CALL links: `with_ancestors`/`with_descendants` follow
    # data-provenance links only (input/create/return) and never cross a CALL edge, so
    # a one-shot transitive query returned nothing here.
    from aiida.common.links import LinkType

    call_types = {"type": {"in": [LinkType.CALL_CALC.value, LinkType.CALL_WORK.value]}}
    builder = QueryBuilder()
    builder.append(orm.ProcessNode, filters=active, project="id")
    frontier = {pk for (pk,) in builder.iterall()}
    if not frontier:
        return held  # nothing live holds anything; `in []` is also a QueryBuilder error
    seen: set[int] = set(frontier)
    while frontier:
        builder = QueryBuilder()
        builder.append(orm.ProcessNode, filters={"id": {"in": list(frontier)}}, tag="caller")
        builder.append(orm.ProcessNode, with_incoming="caller", edge_filters=call_types, project="id")
        frontier = {pk for (pk,) in builder.iterall()} - seen
        seen.update(frontier)
    called = seen
    for direction in ("with_outgoing", "with_incoming"):
        builder = QueryBuilder()
        builder.append(orm.ProcessNode, filters={"id": {"in": list(called)}}, tag="sub")
        builder.append(orm.Node, **{direction: "sub"}, project="uuid")
        held.update(uuid for (uuid,) in builder.iterall())
    return held


def evicted_source_uuids() -> set[str]:
    """Uuids of collections whose array data has been evicted.

    A cache built from one of these cannot be re-exported -- there is nothing
    left to export from -- so it is the last usable copy of those bytes rather
    than a re-derivable convenience.
    """
    from aiida import load_profile
    from aiida.orm import QueryBuilder

    from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData

    load_profile()
    found: set[str] = set()
    for cls in (PinoDatasetCollectionData, PinoSimulationData):
        builder = QueryBuilder().append(
            cls, filters={"extras": {"has_key": "data_evicted"}}, project=["uuid"]
        )
        found.update(uuid for (uuid,) in builder.iterall())
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="delete (default: dry-run)")
    parser.add_argument(
        "--grace-hours",
        type=float,
        default=6.0,
        help="never evict an entry touched this recently (default: 6)",
    )
    args = parser.parse_args(argv)

    root = runtime_cache_root()
    in_use = in_use_uuids()
    evicted = evicted_source_uuids()
    stale = stale_entries(
        root, in_use, grace_hours=args.grace_hours, evicted_sources=evicted
    )

    print(f"cache root: {root}")
    print(f"held by non-terminal processes: {len(in_use)} uuids")
    print(f"source data evicted (cache is the last copy): {len(evicted)} uuids")
    print("MODE:", "APPLY (deleting)" if args.apply else "DRY-RUN (nothing deleted)")

    total = 0
    for entry in stale:
        size = entry_size(entry)
        total += size
        print(f"  {'delete' if args.apply else 'would delete'} {entry.name}  {size / 1e9:6.1f} GB")
        if args.apply:
            shutil.rmtree(entry)
    verb = "freed" if args.apply else "reclaimable (re-run with --apply)"
    print(f"\nTOTAL: {len(stale)} entries, {total / 1e9:.1f} GB {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
