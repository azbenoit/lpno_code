"""Precompute expensive conditioning fields per simulation and store them in AiiDA.

`conduction_field` costs 6.6-53 s/case/beam to compute (spectral stepping over every
frame) and was being recomputed from scratch by every training/eval that read a case
outside its LRU window — the GAOT multibeam arm spent ~12.9 s/it mostly here. The
datasource keeps a persistent local cache (`~/.pino/conduction_cache`, keyed by
case + grid + times + the beam source), but that cache is per-host and outside
provenance.

This driver closes that loop:

1. For each member simulation of a dataset, at each requested grid stride, it calls
   the DATASOURCE'S OWN channel builder — so the computation, the multibeam
   superposition and the cache keys are identical to what a training run does, by
   construction (nothing is reimplemented, so nothing can drift).
2. The builder populates the local cache as a side effect; the new cache files are
   then stored in AiiDA as `SinglefileData` via a calcfunction with the simulation
   as input, and indexed on the sim's mutable extras (`conduction_field_store`).
3. `seed_local_cache(sim)` does the reverse: on a host (or after a cache purge) it
   materialises the stored files back into `~/.pino/conduction_cache`, so runs load
   instead of recomputing.

Storage note: the cache files are float16 `.npy` (the datasource's own format);
they are stored as-is, one node per (case, grid, beam-source) key.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np

EXTRA_KEY = "conduction_field_store"


def _cache_dir() -> Path:
    return Path(os.environ.get("PINO_CONDUCTION_CACHE",
                               Path.home() / ".pino" / "conduction_cache"))


def compute_and_store(dataset_pk: int, stride=(1, 1, 1), limit: int | None = None,
                      fields=("conduction_field",)) -> dict:
    """Compute `fields` for every member of `dataset_pk` at `stride`; store new cache files."""
    from aiida import orm
    from aiida.engine import calcfunction

    from aiida_pino.data.pino_simulation import merge_multibeam_paths
    from lpno.custom_utils.npz_datasource import _build_path_channels

    @calcfunction
    def store_derived_field(sim, filename: orm.Str, meta: orm.Dict):
        # Compressed: a real full-res conduction file measured 827 MB -> 44 MB
        # (5.3%) -- the uncompressed variant is what caused the 2026-08-13
        # ENOSPC. The local cache stays raw .npy (the loader's format); only
        # the durable copy is compressed, and the seeder decompresses back.
        import io as _io

        raw = np.load(_cache_dir() / filename.value)
        buf = _io.BytesIO()
        np.savez_compressed(buf, field=raw)
        buf.seek(0)
        node = orm.SinglefileData(file=buf, filename=filename.value + ".npz")
        node.base.attributes.set("derived_field_meta",
                                 {**meta.get_dict(), "stored_format": "npz"})
        return node

    ds = orm.load_node(dataset_pk)
    uuids = ds.base.attributes.all["member_simulation_uuids"]
    if limit:
        uuids = uuids[:limit]
    sx, sy, sz = (int(v) for v in stride)
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    stored = skipped = 0
    import shutil as _shutil
    for u in uuids:
        # Free-disk floor (the flow3D lesson, relearned 2026-08-13 the hard way:
        # an uncompressed full-res store hit ENOSPC and took down every running
        # training). Stop cleanly rather than filling the disk.
        free_gb = _shutil.disk_usage("/").free / 1e9
        if free_gb < 60:
            print(f"[derive-fields] stopping: {free_gb:.0f} GB free < 60 GB floor", flush=True)
            break
        sim = orm.load_node(u)
        name = sim.base.attributes.all.get("source_folder_name", sim.uuid[:8])
        cm = sim.base.attributes.all["simulation_metadata"]["flow3d_case_manifest"]
        manifest = merge_multibeam_paths(cm, sim.base.extras.get("multibeam_paths", None), name)
        manifest.setdefault("case_name", name)
        buf = io.BytesIO(sim.base.repository.get_object_content("payload.npz", mode="rb"))
        with np.load(buf) as d:
            x = np.asarray(d["x"])[::sx]
            y = np.asarray(d["y"])[::sy]
            z = np.asarray(d["z"])[::sz]
            t = np.asarray(d["time_s"])
        index = list(sim.base.extras.get(EXTRA_KEY, []))
        before = {f.name for f in cache.glob("conduction_*.npy")}
        t0 = time.time()
        _build_path_channels(list(fields), manifest, x, y, z, t)
        new = sorted({f.name for f in cache.glob("conduction_*.npy")} - before)
        already = [e for e in index if e.get("grid") == [x.size, y.size, z.size]]
        if not new and already:
            skipped += 1
            continue
        for fname in new:
            meta = {"case": name, "grid": [int(x.size), int(y.size), int(z.size)],
                    "nt": int(t.size), "stride": [sx, sy, sz], "field": "conduction_field"}
            node = store_derived_field(sim, orm.Str(fname), orm.Dict(dict=meta))
            index.append({"file": fname, "uuid": node.uuid, "stored_format": "npz", **meta})
            stored += 1
        sim.base.extras.set(EXTRA_KEY, index)
        print(f"  {name}: {len(new)} field files in {time.time()-t0:.1f}s "
              f"(grid {x.size}x{y.size}x{z.size})", flush=True)
    summary = {"stored": stored, "cases_skipped": skipped, "cases": len(uuids)}
    # Dataset-level index so coverage is visible on the DATASET node, not only
    # spread across member extras (the findability lesson of 2026-08-11).
    idx = dict(ds.base.extras.get("derived_fields", {}) or {})
    idx[f"conduction_field@{sx}x{sy}x{sz}"] = {
        "cases": len(uuids), "stored_files": stored, "field": "conduction_field",
        "stride": [sx, sy, sz]}
    ds.base.extras.set("derived_fields", idx)
    return summary


def seed_local_cache_for_sims(sims) -> int:
    """Materialise every stored conduction file of `sims` into the local cache.

    Called automatically after runtime staging (pino_dataset_collection), so a
    training run on any host finds the precomputed fields without anyone having
    to remember this layer exists. Existing files are never rewritten.
    """
    from aiida import orm

    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    n = 0
    for sim in sims:
        for entry in sim.base.extras.get(EXTRA_KEY, []):
            target = cache / entry["file"]
            if target.exists():
                continue
            node = orm.load_node(entry["uuid"])
            payload = node.get_content(mode="rb")
            if entry.get("stored_format") == "npz" or node.filename.endswith(".npz"):
                import io as _io

                with np.load(_io.BytesIO(payload)) as z:
                    arr = z["field"]
                tmp = target.with_name(target.name + ".tmp-write")
                with open(tmp, "wb") as fh:   # handle, not path: np.save would
                    np.save(fh, arr)          # append its own .npy to the name
                os.replace(tmp, target)
            else:
                tmp = target.with_name(target.name + ".tmp-write")
                tmp.write_bytes(payload)
                os.replace(tmp, target)
            n += 1
    return n


def seed_local_cache(dataset_pk: int) -> int:
    """Materialise every stored conduction file of a dataset into the local cache."""
    from aiida import orm

    return seed_local_cache_for_sims(
        orm.load_node(u) for u in
        orm.load_node(dataset_pk).base.attributes.all["member_simulation_uuids"]
    )
