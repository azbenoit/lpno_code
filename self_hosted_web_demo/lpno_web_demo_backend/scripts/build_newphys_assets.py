#!/usr/bin/env python
"""Build the calibrated random-path ground truth (`rand150_newphys_test`).

The calibrated dataset is `ti64_rand150_newphys` (AiiDA collection pk 63771,
keywords `experimentally_anchored` / `regime_matched` / `recompute_of_1601`); its
test split is pk 64131, 22 cases. This is the GT the calibrated models must be
compared against -- serving them against the 1601 `kh2579_test` cases would
compare a calibrated model to LEGACY-physics ground truth, which reads as model
error and is not.

Differs from `build_kh2579_assets.py` in one important way: that script locates
each case through `sim.get_simulation_metadata()["source_uri"]`, and for this
dataset every one of those 22 URIs points into a git worktree that
no longer exists. Nothing is lost -- the node itself
carries both halves:

  * arrays   -> `sim.base.repository.open("payload.npz")`
  * manifest -> `sim.get_simulation_metadata()["flow3d_case_manifest"]` (a dict)

so this builder never touches the filesystem source at all.

Stride (2, 2, 1) for the same reason as kh2579: the models train on the strided
75x50x40 grid while the source cases are 150x100x40, and the demo compares GT to
prediction cell for cell. A grid mismatch does NOT raise -- RIGNO rebuilds its
graph from whatever coordinates it is handed -- so the stride must be applied
here or the comparison is silently misaligned.

    python scripts/build_newphys_assets.py [--limit N]
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

TEST_COLLECTION_PK = 64131  # rand150_newphys_test_split, 22 cases
FIELDS = ("Temperature", "Fraction Of Fluid")
SCALARS = ("power_w", "scan_speed_m_per_s", "case_name")
STRIDE = (2, 2, 1)

REPO = Path(__file__).resolve().parents[1]
GT = REPO / "data" / "random_path" / "ground_truth" / "rand150_newphys_test"


def build_ground_truth(limit: int = 0) -> None:
    from aiida import load_profile, orm

    load_profile()
    sx, sy, sz = STRIDE
    col = orm.load_node(TEST_COLLECTION_PK)
    sims = list(col.iter_member_simulations())
    if limit:
        sims = sims[:limit]
    GT.mkdir(parents=True, exist_ok=True)

    total = 0
    for index, sim in enumerate(sims, 1):
        md = sim.get_simulation_metadata()
        name = str(md.get("case_name") or sim.label.split("|")[0].strip())
        dest_npz = GT / f"{name}.npz"
        dest_manifest = GT / f"{name}.manifest.json"

        manifest = md.get("flow3d_case_manifest")
        if not isinstance(manifest, dict):
            print(f"  skip {name}: no flow3d_case_manifest on node {sim.pk}")
            continue

        if dest_npz.is_file() and dest_manifest.is_file():
            total += dest_npz.stat().st_size
            continue

        with sim.base.repository.open("payload.npz", "rb") as fh:
            buf = io.BytesIO(fh.read())
        size = write_case(buf, manifest, name, GT)
        if size:
            total += size
            print(f"  [{index}/{len(sims)}] {name}")

    print(f"ground truth: {len(sims)} cases, {total / 1073741824:.2f} GB in {GT}")


def write_case(npz_source, manifest: dict, name: str, dest: Path) -> int:
    """One served case, strided to the models' grid: `<name>.npz` and `<name>.manifest.json`
    in `dest`, from a case's full NPZ (a path or a file object) and its FLOW-3D manifest.
    Returns the bytes written, 0 if the payload lacks a field. Shared by the AiiDA route
    above and by `build_ground_truth_from_deposit.py`."""
    sx, sy, sz = STRIDE
    with np.load(npz_source) as data:
        missing = [k for k in FIELDS + SCALARS if k not in data.files]
        if missing:
            print(f"  skip {name}: payload lacks {missing}")
            return 0
        payload = {
            "x": np.asarray(data["x"], np.float32)[::sx],
            "y": np.asarray(data["y"], np.float32)[::sy],
            "z": np.asarray(data["z"], np.float32)[::sz],
            "time_s": np.asarray(data["time_s"], np.float32),
        }
        for key in SCALARS:
            payload[key] = data[key]
        for key in FIELDS:
            payload[key] = np.ascontiguousarray(np.asarray(data[key], np.float32)[:, ::sx, ::sy, ::sz])
    dest.mkdir(parents=True, exist_ok=True)
    dest_npz = dest / f"{name}.npz"
    np.savez_compressed(dest_npz, **payload)
    (dest / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"    {name}: T{payload['Temperature'].shape}")
    return dest_npz.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases")
    args = parser.parse_args()
    build_ground_truth(args.limit)


if __name__ == "__main__":
    main()
