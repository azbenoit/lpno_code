#!/usr/bin/env python3
"""Rebuild the datasets the published models trained and were scored on, from the data deposit.

The deposit ships the AiiDA provenance archive `lpno_provenance.aiida` with every simulation's
array payload evicted, and the payloads themselves as one uncompressed tar per simulation family
(`payloads_<family>.tar`, members `<uuid>.npz`, indexed by `payload_manifest.csv`). A training
does not read those payloads directly. It reads a runtime directory that aiida-pino's
`PinoDatasetCollectionData.export_runtime_arraydir` writes from a derived collection: for the
steady models, each case's last 30 laser-on frames cut to a beam-following window and averaged,
plus the conduction-field input channel, one `.npy` per field.

This script runs that same export code on the archive opened read-only (no `verdi archive
import`, no database), with the payload reads pointed at the tar members and checked against
their sha256. Each simulation is derived once and hard-linked into every collection that
contains it, so overlapping training sets cost one derivation per simulation; an interrupted
run resumes where it stopped.

    # every collection behind a published model: UUIDs (or unique prefixes) from the archive
    python reproducibility/rerun/export_deposit.py export --deposit DEPOSIT --out DATA 09e2a9c5 0b657a95 b28c2dc2
    # a subset of an exported collection, keeping the members of some families (fnmatch)
    python reproducibility/rerun/export_deposit.py subset DATA/<parent> DATA/<name> 'ti64_*' 'in718_loop_maximin_r01'
    # list the derived collections of the archive, with sizes
    python reproducibility/rerun/export_deposit.py list --deposit DEPOSIT [substring]

Needs the pino environment with aiida-core, aiida-pino and lpno installed (see
lpno/README.md). One member peaks at a few GB of memory, so members are derived one
at a time.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

# The fields the steady configs read (`custom.vtk_fields` of config_flow3D_*_pvlhs_steady);
# the coordinates, process scalars and conduction_field are added by the export itself.
DEFAULT_FIELDS = ["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"]
STORE_VERSION = 1


class Deposit:
    """The deposit directory: archive, payload index and per-family payload tars."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.archive = self.root / "lpno_provenance.aiida"
        index = self.root / "payload_manifest.csv"
        if not self.archive.is_file() or not index.is_file():
            sys.exit(f"{self.root} does not look like the deposit: need lpno_provenance.aiida and payload_manifest.csv")
        with open(index, newline="") as handle:
            self.rows = {row["uuid"]: row for row in csv.DictReader(handle)}
        self._tars: dict[Path, dict[str, tuple[int, int]]] = {}

    def locator(self, uuid: str) -> dict:
        row = self.rows.get(uuid)
        if row is None:
            raise KeyError(f"simulation {uuid} is not in payload_manifest.csv")
        tar = self.root / f"payloads_{row['family']}.tar"
        offset, size = self._index(tar)[f"{uuid}.npz"]
        if size != int(row["bytes"]):
            raise OSError(f"{tar.name}: {uuid}.npz is {size} bytes, payload_manifest.csv says {row['bytes']}")
        # The shape aiida-pino's runtime_export readers take; uuid and sha256 ride along.
        return {"kind": "pack", "path": str(tar), "offset": offset, "size": size,
                "uuid": uuid, "sha256": row["sha256"]}

    def _index(self, tar: Path) -> dict[str, tuple[int, int]]:
        if tar not in self._tars:
            with tarfile.open(tar, "r:") as handle:   # uncompressed: headers only, data skipped
                self._tars[tar] = {os.path.basename(m.name): (m.offset_data, m.size) for m in handle if m.isfile()}
        return self._tars[tar]


class _Slice(io.RawIOBase):
    """A read-only, seekable view of one byte range of a file (a tar member, read in place)."""

    def __init__(self, path: str, offset: int, size: int):
        super().__init__()
        self._handle = open(path, "rb")
        self._offset, self._size, self._pos = offset, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, pos, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + pos)
        return self._pos

    def readinto(self, buffer):
        n = max(0, min(len(buffer), self._size - self._pos))
        if n == 0:
            return 0
        self._handle.seek(self._offset + self._pos)
        data = self._handle.read(n)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self):
        self._handle.close()
        super().close()


def open_archive(deposit: Deposit, scratch: Path):
    """Load the archive as a read-only profile. AiiDA's config goes to scratch, never ~/.aiida."""
    os.environ["AIIDA_PATH"] = str(scratch / "aiida")
    from aiida import load_profile
    from aiida.storage.sqlite_zip.backend import SqliteZipBackend

    load_profile(SqliteZipBackend.create_profile(deposit.archive), allow_switch=True)


def redirect_payload_reads(deposit: Deposit, store: Path, conduction_cache: Path, *, window: bool = True) -> None:
    """Point aiida-pino's payload reads at the deposit tars, and derive each member once."""
    import numpy as np

    from aiida_pino.data.pino_simulation import PinoSimulationData
    from aiida_pino.utils import runtime_export as rx

    read_range = rx.read_locator_bytes
    derive = rx.derived_member_worker

    def payload_locator(simulation):
        return deposit.locator(str(simulation.uuid))

    def read_locator_bytes(locator):
        data = read_range(locator)
        if locator.get("sha256") and hashlib.sha256(data).hexdigest() != locator["sha256"]:
            raise OSError(f"sha256 mismatch for {locator.get('uuid')} in {locator['path']}")
        return data

    @contextlib.contextmanager
    def open_payload_archive(self):
        loc = deposit.locator(str(self.uuid))
        with _Slice(loc["path"], loc["offset"], loc["size"]) as raw:
            with np.load(io.BufferedReader(raw), allow_pickle=False) as archive:
                yield archive

    def get_time_values_s(self):
        # What `from_npz_file` stored as `_time_values_s` before the eviction.
        with self._open_payload_archive() as archive:
            return np.asarray(archive["time_s"], dtype=np.float64).astype(np.float32)

    def derived_member_worker(task):
        loc = task.get("locator") or {}
        uuid = loc.get("uuid")
        if not uuid:
            raise RuntimeError("export task without a deposit locator; nothing else is readable here")
        # `materialize` says how AiiDA stores a collection, not how members are derived; some
        # collections carry it and some do not, and keying on it derived the same member twice.
        profile_key = {k: v for k, v in task["profile"].items() if k != "materialize"}
        recipe = {"store": STORE_VERSION, "uuid": uuid, "profile": profile_key,
                  "fields": task.get("requested_fields"), "slices": task.get("moving_frame_slices"),
                  "stop": task.get("moving_frame_stop"), "times": task.get("time_values_s")}
        key = hashlib.sha1(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()[:12]
        cell = store / f"{uuid}__{key}"
        if not (cell / "result.json").is_file():
            partial = store / f".{cell.name}.partial"
            shutil.rmtree(partial, ignore_errors=True)
            started = time.time()
            # aiida-pino's worker reads only the frames a moving-frame derivation keeps
            # (`member_frame_window`); --full-frames turns that off. The bits are the same.
            result = derive({**task, "target_dir": str(partial), "frame_window": window})
            (partial / "result.json").write_text(json.dumps(result))
            shutil.rmtree(cell, ignore_errors=True)
            os.replace(partial, cell)
            for stale in conduction_cache.glob("conduction_*.npy"):   # ~0.5 GB per case, used once
                stale.unlink()
            print(f"  derived {uuid} ({deposit.rows[uuid]['case_name']}) in {time.time() - started:.0f} s", flush=True)
        target = Path(task["target_dir"])
        target.mkdir(parents=True, exist_ok=True)
        for item in cell.iterdir():
            if item.name != "result.json":
                link = target / item.name
                link.unlink(missing_ok=True)
                os.link(item, link)
        return {**json.loads((cell / "result.json").read_text()), "index": task["index"]}

    rx.payload_locator = payload_locator
    rx.read_locator_bytes = read_locator_bytes
    rx.derived_member_worker = derived_member_worker
    PinoSimulationData._open_payload_archive = open_payload_archive
    PinoSimulationData.get_time_values_s = get_time_values_s
    PinoSimulationData._require_data = lambda self: None   # every payload is served from the tars


def derived_collections(substring: str = ""):
    from aiida import orm

    from aiida_pino.data.pino_dataset_collection import PinoDatasetCollectionData

    query = orm.QueryBuilder().append(PinoDatasetCollectionData, project=["uuid", "label", "attributes.simulation_count",
                                                                          "attributes.dataset_role"])
    return sorted((r for r in query.all() if substring in (r[1] or "")), key=lambda r: r[1] or "")


def resolve(prefix: str):
    from aiida import orm

    from aiida_pino.data.pino_dataset_collection import PinoDatasetCollectionData

    hits = orm.QueryBuilder().append(PinoDatasetCollectionData, filters={"uuid": {"like": f"{prefix}%"}}).all(flat=True)
    if len(hits) != 1:
        sys.exit(f"collection {prefix!r}: {len(hits)} matches in the archive; give more of the UUID")
    return hits[0]


def dir_name(collection) -> str:
    head = re.sub(r"[^A-Za-z0-9_.-]+", "_", (collection.label or "collection").split(" | ")[0]).strip("_")
    return f"{head}__{str(collection.uuid)[:8]}"


def cmd_export(args) -> None:
    deposit = Deposit(args.deposit)
    out = Path(args.out).expanduser().resolve()
    if out == Path.home() or len(out.parts) < 3:
        sys.exit(f"refusing to write collections directly under {out}")
    store, scratch = out / ".members", out / ".scratch"
    for path in (store, scratch):
        path.mkdir(parents=True, exist_ok=True)
    conduction_cache = Path(args.cache_dir).expanduser() if args.cache_dir else scratch / "conduction_cache"
    conduction_cache.mkdir(parents=True, exist_ok=True)
    os.environ["PINO_CONDUCTION_CACHE"] = str(conduction_cache)
    os.environ["AIIDA_PINO_EXPORT_WORKERS"] = "1"   # in-process, so the redirection above applies
    open_archive(deposit, scratch)
    redirect_payload_reads(deposit, store, conduction_cache, window=not args.full_frames)
    collections = [resolve(p) for p in args.collections]
    for collection in collections:
        target = out / dir_name(collection)
        started = time.time()
        print(f"{collection.label} ({collection.uuid}): {len(collection.get_member_simulation_uuids())} members -> {target}",
              flush=True)
        collection.export_runtime_arraydir(target, runtime_fields=args.fields, allow_pack_index=False)
        print(f"  done in {time.time() - started:.0f} s", flush=True)


def cmd_subset(args) -> None:
    parent, child = Path(args.parent).resolve(), Path(args.child).resolve()
    manifest = json.loads((parent / "manifest.json").read_text())
    if child.exists():
        sys.exit(f"{child} exists; remove it first")
    kept = []
    for member in manifest["members"]:
        meta = json.loads((parent / member["manifest_file"]).read_text())["simulation_metadata"]
        family = meta.get("batch_keyword") or ""
        if any(fnmatch.fnmatch(family, pattern) for pattern in args.families):
            kept.append(member)
            src, dst = parent / member["relative_path"], child / member["relative_path"]
            dst.mkdir(parents=True)
            for item in src.iterdir():
                os.link(item, dst / item.name)
    if not kept:
        sys.exit("no member matched those families")
    uuids = {m["simulation_uuid"] for m in kept}
    metadata = dict(manifest.get("dataset_metadata") or {})
    order = list(metadata.get("member_simulation_uuids") or [])
    for key in ("member_simulation_uuids", "member_simulation_pks"):
        if key in metadata:
            metadata[key] = [value for uuid, value in zip(order, metadata[key]) if uuid in uuids]
    metadata["simulation_count"] = len(kept)
    total = len(manifest["members"])
    manifest.update(members=kept, simulation_count=len(kept), dataset_metadata=metadata,
                    subset_of={"path": parent.name, "families": list(args.families)})
    (child / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"{child.name}: {len(kept)} of {total} members")


def cmd_model(args) -> None:
    """Write a deposited model's files (weights, .hydra, training curve) to OUT/<model label>."""
    deposit = Deposit(args.deposit)
    out = Path(args.out).expanduser().resolve()
    scratch = out / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    open_archive(deposit, scratch)   # before any aiida import, or AiiDA writes ~/.aiida
    from aiida import orm

    from aiida_pino.data.pino_model import PinoModelData

    for name in args.models:
        query = orm.QueryBuilder().append(PinoModelData, filters={"or": [{"label": name},
                                                                          {"uuid": {"like": f"{name}%"}}]})
        hits = query.all(flat=True)
        if len(hits) != 1:
            sys.exit(f"model {name!r}: {len(hits)} matches in the archive")
        model = hits[0]
        target = out / model.label
        if target.exists():
            shutil.rmtree(target)
        model.base.repository.copy_tree(target)
        nested = target / "model"   # models registered after training keep their files under model/
        if nested.is_dir() and not (target / ".hydra").exists():
            for item in nested.iterdir():
                item.rename(target / item.name)
            nested.rmdir()
        (target / "deposit_model.json").write_text(json.dumps(
            {"uuid": model.uuid, "label": model.label, "extras": model.base.extras.all}, indent=2, default=str))
        print(f"{model.label} ({model.uuid}) -> {target}")


def cmd_list(args) -> None:
    deposit = Deposit(args.deposit)
    scratch = Path(args.scratch).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    open_archive(deposit, scratch)
    for uuid, label, count, role in derived_collections(args.substring):
        if role == "derived":
            print(f"{uuid[:8]}  n={count:<4} {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="export archived collections as runtime directories")
    p.add_argument("--deposit", required=True, help="the deposit directory")
    p.add_argument("--out", required=True, help="where the collection directories go")
    p.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS,
                   help="raw fields to keep (default: the steady configs' vtk_fields)")
    p.add_argument("--cache-dir", help="scratch for each case's lab-frame conduction field (up to ~0.5 GB, "
                                       "deleted after the case); a RAM disk such as /dev/shm/lpno saves those "
                                       "disk writes (default: OUT/.scratch/conduction_cache)")
    p.add_argument("--full-frames", action="store_true",
                   help="canonicalize every frame, as aiida-pino does, instead of only the 30 the steady "
                        "derivation keeps (same output, several times the memory; for checking)")
    p.add_argument("collections", nargs="+", help="collection UUIDs or unique prefixes")
    p.set_defaults(run=cmd_export)
    p = sub.add_parser("subset", help="a new collection directory from some families of an exported one")
    p.add_argument("parent")
    p.add_argument("child")
    p.add_argument("families", nargs="+", help="fnmatch patterns on the simulation family (batch_keyword)")
    p.set_defaults(run=cmd_subset)
    p = sub.add_parser("model", help="write deposited models (weights and .hydra) to a directory")
    p.add_argument("--deposit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("models", nargs="+", help="model labels (e.g. fno_maximin_r05_alloyflag_s42) or UUID prefixes")
    p.set_defaults(run=cmd_model)
    p = sub.add_parser("list", help="list the archive's derived collections")
    p.add_argument("--deposit", required=True)
    p.add_argument("--scratch", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "lpno_export_scratch"))
    p.add_argument("substring", nargs="?", default="")
    p.set_defaults(run=cmd_list)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
