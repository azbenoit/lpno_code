#!/usr/bin/env python
"""Build the CALIBRATED straight-line ground truth, in the demo's per-case format.

Same problem as the steady page, one page over: `data/straight_path_time/ground_truth`
is `resampled_straight_path_data`, the PRE-calibration dataset. Serving calibrated
straight models against it compares across physics eras, which renders as gross model
error and is not one.

Source is `ti64_straight_path_calibrated_with_keyholes` -- the same 123 cases the
calibrated STEADY page serves, which is what harmonizes the two pages: identical
P-V catalog, identical splits (96 train / 14 val / 13 test), identical physics era.

Writes to a SEPARATE directory (`ground_truth_calibrated`), leaving the legacy cache
untouched, so the switch is one setting (`LPFNO_STRAIGHT_DATA_DIR`) and reverting
costs nothing.

Which collection, and every derivation knob, is resolved from the MODEL's own
evaluation -- never assumed:

    model pk -> its PinoModelEvaluationData -> dataset_source_pk -> that
    collection's `derivation_profile`

Frames: the demo renders `N_FRAMES` (20) and takes its frame times FROM this file
(`adapter_straightpath.resolve` reads `time_s` and refuses a case that is not 20
long), so each case is subsampled to the 20 stored steps nearest an even sweep of
its own 0..laser_off_time_s. Stored times are the ACTUAL sampled times, not the
even targets, so the served prediction lands on the frames the arrays hold.

Also writes `<case>.manifest.json` beside each npz. The 16-channel calibrated models
read physics out of the case manifest (preheat_T0 <- initial_temperature_k,
beam_shape_uniform <- heat_distribution, conduction_field <- surface_z_m), and this
set runs at 273.15 K, not the 293.15 K the random-path builders pin. A wrong constant
there is silent: the channel builds, the shapes match, the physics is off.

    python scripts/build_calibrated_straight_gt.py [--splits test val]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

# Which MODEL the ground truth is built for. See build_calibrated_steady_gt.py for
# why the model, and not a hardcoded collection, is the entry point: the config's
# `geometry.resolution` is nominal and among these collections the z-crop takes
# three different values.
DEFAULT_MODEL_PK = 78585   # fno_lab_w3_32k_seed43, the served straight FNO

N_FRAMES = 20              # adapter_randompath.N_FRAMES; the adapter REFUSES any other count
FIELDS = ("Temperature", "Fraction Of Fluid")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "straight_path_time" / "ground_truth_calibrated"


def resolve_profile(model_pk: int) -> tuple[int, dict]:
    """(collection pk, derivation profile) for the collection this model was scored on."""
    from aiida import orm

    qb = orm.QueryBuilder()
    qb.append(
        orm.Node,
        filters={"and": [
            {"node_type": {"like": "%PinoModelEvaluationData%"}},
            {"attributes.evaluation_metadata.model_pk": int(model_pk)},
        ]},
        project=["attributes.evaluation_metadata"],
    )
    rows = [r[0] for r in qb.all()]
    if not rows:
        raise SystemExit(f"model {model_pk} has no evaluation node; cannot resolve its derivation")
    em = rows[0]
    ds_pk = int(em["dataset_source_pk"])
    profile = dict(em.get("derivation_profile") or {})
    if not profile:
        profile = dict(orm.load_node(ds_pk).base.attributes.all.get("derivation_profile") or {})
    return ds_pk, profile


def sibling_splits(ds_pk: int) -> dict[str, int]:
    """The train/val/test collections sharing this one's label stem AND profile."""
    from aiida import orm

    ref = orm.load_node(ds_pk)
    stem = ref.label.split("_split")[0].rsplit("_", 1)[0]
    want = dict(ref.base.attributes.all.get("derivation_profile") or {})
    qb = orm.QueryBuilder()
    qb.append(
        orm.Node,
        filters={"and": [
            {"node_type": {"like": "%PinoDatasetCollectionData%"}},
            {"label": {"like": f"%{stem}%"}},
        ]},
        project=["id", "label", "attributes.derivation_profile"],
    )
    out: dict[str, int] = {}
    for pk, label, prof in sorted(qb.all()):
        if dict(prof or {}) != want:
            continue
        for split in ("train", "val", "test"):
            if f"{split}_split" in label:
                out.setdefault(split, int(pk))
    return out


def _frame_indices(times: np.ndarray) -> np.ndarray:
    """The N_FRAMES stored steps nearest an even sweep of 0..t[-1].

    Nearest-stored rather than interpolated: the arrays are the reference, and
    interpolating a melt pool across ~5 us would invent a state the solver never
    produced. Matches the legacy cache, whose times are an even sweep snapped to
    stored frames.
    """
    targets = np.linspace(float(times[0]), float(times[-1]), N_FRAMES)
    idx = np.unique(np.abs(times.reshape(-1, 1) - targets.reshape(1, -1)).argmin(axis=0))
    if idx.size != N_FRAMES:
        raise SystemExit(
            f"only {idx.size} distinct frames for {N_FRAMES} targets "
            f"(series has {times.size} steps); the case is too short to serve"
        )
    return idx


def _apply_profile(arr: np.ndarray, profile: dict) -> np.ndarray:
    """Stride and gas-side z-crop from the PROFILE, applied like the lazy store.

    Upstream slices `[..., :zk:sz]` (npz_lazy_store.py:566), dropping the HIGH-index
    z planes, and high z is the gas side here. `crop_z_top` 0/None = full extent.
    Array is (nt, nx, ny, nz).
    """
    sx, sy, sz = (profile.get("stride") or [1, 1, 1])[:3]
    out = arr[:, ::int(sx or 1), ::int(sy or 1), ::int(sz or 1)]
    crop = int(profile.get("crop_z_top") or 0)
    if crop > 0:
        nz = out.shape[-1]
        if crop >= nz:
            raise SystemExit(f"crop_z_top={crop} removes all {nz} z-planes")
        out = out[..., : nz - crop]
    return np.ascontiguousarray(out)


def build_split(split: str, pk: int, profile: dict, *, limit: int = 0) -> None:
    from aiida import orm

    col = orm.load_node(pk)
    sims = list(col.iter_member_simulations())
    if limit:
        sims = sims[:limit]
    dest = OUT / split
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{split}] pk {pk} {col.label[:52]}: {len(sims)} cases -> {dest}")

    total = 0
    for i, sim in enumerate(sims, 1):
        md = sim.get_simulation_metadata()
        name = str(md.get("case_name") or sim.label.split("|")[0].strip())
        npz_path, man_path = dest / f"{name}.npz", dest / f"{name}.manifest.json"
        if npz_path.is_file() and man_path.is_file():
            total += npz_path.stat().st_size
            continue

        manifest = md.get("flow3d_case_manifest")
        if not isinstance(manifest, dict):
            print(f"  skip {name}: no flow3d_case_manifest on node {sim.pk}")
            continue

        # Spill the payload to disk before decoding: holding the compressed bytes
        # and two 231 MB decompressed fields at once is avoidable peak RSS on a
        # box that shares its RAM with trainings.
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            with sim.base.repository.open("payload.npz", "rb") as fh:
                while chunk := fh.read(8 << 20):
                    tmp.write(chunk)
            tmp.flush()
            size = write_case(Path(tmp.name), manifest, name, profile, dest)
        if size:
            total += size
            print(f"  [{i}/{len(sims)}] {name}")

    print(f"[{split}] {len(sims)} cases, {total / 1e9:.2f} GB")


def write_case(npz_file: Path, manifest: dict, name: str, profile: dict, dest: Path) -> int:
    """One served case: `<name>.npz` (20 frames, stride and z-crop from `profile`) and
    `<name>.manifest.json` in `dest`, from a case's full NPZ and its FLOW-3D manifest.
    Returns the bytes written, 0 if the payload lacks a field. Shared by the AiiDA route
    above and by `build_ground_truth_from_deposit.py`."""
    npz_path, man_path = dest / f"{name}.npz", dest / f"{name}.manifest.json"
    with np.load(npz_file) as data:
        missing = [k for k in FIELDS + ("x", "y", "z", "time_s") if k not in data.files]
        if missing:
            print(f"  skip {name}: payload lacks {missing}")
            return 0
        times = np.asarray(data["time_s"], dtype=np.float64)
        keep = _frame_indices(times)
        sx, sy, sz = (profile.get("stride") or [1, 1, 1])[:3]
        payload = {
            "x": np.asarray(data["x"], np.float32)[::int(sx or 1)],
            "y": np.asarray(data["y"], np.float32)[::int(sy or 1)],
            "z": np.asarray(data["z"], np.float32)[::int(sz or 1)],
            "time_s": times[keep].astype(np.float32),
            "power_w": np.float32(data["power_w"]),
            "scan_speed_m_per_s": np.float32(data["scan_speed_m_per_s"]),
            "case_name": np.asarray(name),
        }
        crop = int(profile.get("crop_z_top") or 0)
        if crop > 0:
            payload["z"] = payload["z"][: payload["z"].size - crop]
        for key in FIELDS:
            payload[key] = _apply_profile(np.asarray(data[key], np.float32)[keep], profile)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)
    man_path.write_text(json.dumps(manifest, indent=1))
    print(f"    {name}: T{payload['Temperature'].shape} "
          f"P={float(payload['power_w']):.1f} V={float(payload['scan_speed_m_per_s']):.3f}")
    return npz_path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-pk", type=int, default=DEFAULT_MODEL_PK,
                    help="build the GT this model was actually scored on")
    ap.add_argument("--splits", nargs="*", default=["test", "val", "train"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases per split")
    args = ap.parse_args()

    from aiida import load_profile

    load_profile()
    ds_pk, profile = resolve_profile(args.model_pk)
    print(f"model {args.model_pk} was scored on collection {ds_pk}")
    print(f"  derivation_profile: {profile}")
    splits = sibling_splits(ds_pk)
    print(f"  sibling splits sharing that profile: {splits}")
    for split in args.splits:
        pk = splits.get(split)
        if pk is None:
            print(f"[{split}] no collection with this profile; skipped")
            continue
        build_split(split, pk, profile, limit=args.limit)


if __name__ == "__main__":
    main()
