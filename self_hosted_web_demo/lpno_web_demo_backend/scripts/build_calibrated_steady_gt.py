#!/usr/bin/env python
"""Build the CALIBRATED steady ground truth, in the demo's consolidated format.

Why this exists: the steady GT under `data/steady-state/ground_truth` comes from
`original_straight_path_data` -- the PRE-calibration dataset (see its README).
Serving calibrated steady models against it compares across physics eras, which
renders as a gross model error and is not one: on 2026-09-02 that put a uniform
3123 K (`T_FILL_K`) band across the top of the GT-compare panel, because the
calibrated model correctly predicted gas where the legacy GT has metal.

Writes to a SEPARATE directory (`ground_truth_calibrated`), leaving the legacy
cache untouched, so the switch is one setting (`LPFNO_STEADY_GT_DATA_DIR`) and
reverting costs nothing.

The derivation is NOT reimplemented. It runs through the submodule's own
`npz_datasource.load_consolidated_npz` with the same knobs the models were
trained under (`moving_ref_frame`, `moving_frame_steps=30`, `avg_window_size=30`,
`super_res_factor` 1 and 2), because a hand-rolled moving frame or time average
that drifted from training would produce a plausible, wrong reference -- the same
class of silent error this script exists to fix.

Two things the eager path will not do for you, both handled here:

  * It cannot stack cases of differing length ("Cases have differing timestep
    counts [195, 402]"), and the calibrated cases genuinely differ. So each case
    is derived ALONE -- the steady derivation collapses it to a single frame --
    and the per-case results are stacked afterwards.
  * `PINO_LAZY_CROP_Z_TOP` is a read-time feature of the lazy store and is
    ignored here (it warns and proceeds on the FULL z extent). The gas-side crop
    is therefore applied explicitly below.

The crop is measured, not assumed: the calibrated cases are 40 z-planes of which
exactly the top 10 are fully gas (metal fraction 0 across x/y), and dropping them
leaves 30 -- matching `custom.geometry.resolution` [90, 40, 30] in every
calibrated steady config, with the last kept plane at z = -5.0e-6, the laser z0
(the free surface). The crop count is read from the collection's derivation
profile at build time (see `_crop_gas`), never assumed, and the script refuses to
write if it does not hold.

    python scripts/build_calibrated_steady_gt.py [--splits test]
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

# Which MODEL the ground truth is being built for. Everything else -- which
# collection, and every derivation knob including the z-crop -- is resolved from
# that model's own evaluation, never assumed:
#
#   model pk -> its PinoModelEvaluationData -> dataset_source_pk -> that
#   collection's `derivation_profile`
#
# This is the whole point. The knobs are NOT in the model's hydra config: it
# carries `geometry.resolution [90, 40, 30]`, which is nominal and is exactly
# what led me to crop 10 gas planes off a model trained on all 40 (metal-only MAE
# 353 K cropped vs 7.42 K uncropped, against its own recorded 6.03 K). And the
# crop is not a constant to flip the other way either -- among the calibrated
# steady collections it takes three different values: None, 0 and 10. The only
# authoritative source is the profile of the collection the model was scored on.
DEFAULT_MODEL_PK = 63732      # fno_spkh123_steady_32k_s44, the served steady FNO

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "steady-state" / "ground_truth_calibrated"


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


def _derive_one(case_dir: Path, super_res: int, profile: dict) -> dict[str, np.ndarray]:
    """Every knob from the profile; nothing assumed."""
    from lpno.custom_utils import npz_datasource as nds

    return nds.load_consolidated_npz(
        str(case_dir),
        super_res_factor=int(profile.get("super_res_factor") or super_res),
        time_axis=bool(profile.get("include_time", True)),
        avg_window_size=int(profile.get("time_avg_window") or 1),
        moving_frame_steps=int(profile.get("moving_frame_steps") or 1),
        moving_ref_frame=bool(profile.get("moving_ref_frame", True)),
    )


def _crop_gas(arrays: dict[str, np.ndarray], crop_z_top: int, super_res: int) -> dict[str, np.ndarray]:
    """Drop `crop_z_top * super_res` planes off the gas end of z.

    The count comes from the collection's derivation profile, never from a
    constant here and never from the model config's nominal
    `geometry.resolution`. Among the calibrated steady collections it is None, 0
    or 10 depending on the run, so either hardcoded answer is wrong for some
    model. `crop_z_top` of 0/None means serve the full extent.

    Upstream slices `[..., :zk:sz]` (npz_lazy_store.py:566), i.e. from the START,
    dropping the HIGH-index planes -- and high z is the gas side here (metal
    fraction is 1 at low index, 0 at high), so this matches.
    """
    n = int(crop_z_top or 0) * int(super_res)
    if n <= 0:
        return _align_frame(dict(arrays))
    nz = arrays["z"].shape[-1]
    if n >= nz:
        raise SystemExit(f"crop_z_top={crop_z_top} removes all {nz} z-planes")
    return _align_frame({k: np.ascontiguousarray(v[..., : nz - n]) for k, v in arrays.items()})


def _align_frame(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Identity. Kept as a named no-op because what it used to do was the bug.

    It re-centred each case's x window on 0, to match the legacy cache's
    convention. That is wrong, and measurably so. The moving-frame window is a
    61-cell slice of the ABSOLUTE domain that follows the beam, and its absolute
    position is NOT the same for every case: measured over the 13 test cases,
    raw x[0] runs -1.877e-4..-1.550e-4 and the beam sits at window index 35..44.
    Centring shifts every case by its OWN offset, which destroys the one thing
    the comparison needs -- that GT and prediction mean the same x.

    What it cost: the GT-compare panel reported 63-192 K where the model's own
    recorded evaluation (63732, eval 63850, ds 58056) is 7.51 K substrate. With
    the raw window kept and the prediction served on the training frame, the
    nine test cases whose window matches score 6.23 K; the four offset by 1-3
    cells scored 21.00 K until the serving side started shifting per case too
    (`x_shift_m`, service._predict_steady_cached).

    `load_steady_gt` already reads each case's own axes out of the stack, so
    keeping them raw is all that is needed here.
    """
    return dict(arrays)


def build_split(split: str, pk: int, profile: dict, super_res_factors=(1,)) -> None:
    from aiida import orm

    col = orm.load_node(pk)
    sims = list(col.iter_member_simulations())
    crop = profile.get("crop_z_top") or 0
    print(f"[{split}] pk {pk} {col.label[:52]}: {len(sims)} cases  "
          f"(steps={profile.get('moving_frame_steps')} avg={profile.get('time_avg_window')} "
          f"mref={profile.get('moving_ref_frame')} crop_z_top={crop})")

    stacks: dict[int, dict[str, list[np.ndarray]]] = {sr: {} for sr in super_res_factors}
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for i, sim in enumerate(sims, 1):
            md = sim.get_simulation_metadata()
            name = str(md.get("case_name") or sim.label.split("|")[0].strip())
            case_dir = work / name
            if case_dir.exists():
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True)
            with sim.base.repository.open("payload.npz", "rb") as fh:
                (case_dir / f"{name}.npz").write_bytes(fh.read())
            manifest = md.get("flow3d_case_manifest")
            if isinstance(manifest, dict):
                (case_dir / f"{name}.manifest.json").write_text(json.dumps(manifest))

            for sr in super_res_factors:
                arrays = _crop_gas(_derive_one(case_dir, sr, profile), profile.get("crop_z_top"), sr)
                for k, v in arrays.items():
                    stacks[sr].setdefault(k, []).append(v)
            shutil.rmtree(case_dir)
            print(f"  [{i}/{len(sims)}] {name}")

    dest = OUT / split
    dest.mkdir(parents=True, exist_ok=True)
    for sr in super_res_factors:
        payload = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in stacks[sr].items()}
        path = dest / f"var_{sr}_time_steps30_avg30.npz"
        np.savez_compressed(path, **payload)
        shape = payload["Temperature"].shape
        print(f"  -> {path.name}: Temperature{shape} ({path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-pk", type=int, default=DEFAULT_MODEL_PK,
                    help="build the GT this model was actually scored on")
    ap.add_argument("--splits", nargs="*", default=["test", "val"])
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
        build_split(split, pk, profile)


if __name__ == "__main__":
    main()
