"""Record, on every collection stamping a crop nothing applies, the extent it really serves.

`DEFAULT_DERIVATION_PROFILE["crop_z_top"] = 10` is a non-identity default (landed
2026-08-27). Every collection created since carries it, and `array_shapes` /
`spatial_shape` are computed at creation by RUNNING the derivation on member 0
(`pino_dataset_collection.py`, `derive_simulation_arrays` call) -- so the recorded
extent is cropped **regardless of role**. Only a `derived` collection has a read
path that applies the crop. For `base`, `subset` and `composed` nothing does: the
loader serves the members' full extent while the record claims 10 fewer z-layers.

Measured, not inferred. 90 CalcJobs consuming stamped non-derived collections
logged `[grid] serving ...` and **every one served the member extent**, never the
recorded one -- 6 base, 26 subset, 58 composed, zero counterexamples. Examples:
one collection records z=30 and served z=40 in seven separate jobs; another records
z=45 and served z=55.

The derivation profile is a sealed node ATTRIBUTE -- a direct write raises
`ModificationNotAllowed`, and provenance should stay immutable. What can be
recorded is what actually happens, as extras:

* `effective_crop_z_top` -- 0, the crop actually applied;
* `effective_spatial_shape` / `effective_array_shapes` -- the extent really served;
* `crop_correction` -- from/to/date/role/evidence, so the write is auditable and
  revertable.

`get_dataset_metadata()` overlays these when present, which is the single funnel
every reader already goes through. No hash or signature reads that accessor, so
the overlay cannot cause supersession churn, and `crop_z_top` is already in
`LAYOUT_COMPARISON_EXEMPT_KEYS` so the submission guard is unaffected.

`derived` collections are deliberately NOT touched: they are genuinely mixed
(one served its member extent, two others served a cropped one) and
correcting them needs the read path settled first.

Dry run by default.

    python -m aiida_pino.utils.crop_backfill
    python -m aiida_pino.utils.crop_backfill --apply
    python -m aiida_pino.utils.crop_backfill --revert --apply
"""

from __future__ import annotations

from typing import Any

from aiida import orm

#: Written into every correction so a later reader can date the rule.
CORRECTION_DATE = "2026-09-09"

#: The extras this module owns. `--revert` removes exactly these.
CORRECTION_KEYS = (
    "effective_crop_z_top",
    "effective_spatial_shape",
    "effective_array_shapes",
    "crop_correction",
)

REASON = (
    "derivation_profile.crop_z_top is recorded but never applied for this role: "
    "array_shapes/spatial_shape were computed by running the derivation at creation "
    "time, while the read path applies it only for role=derived. Measured against the "
    "'[grid] serving' line of 90 consuming CalcJobs, all of which served the member "
    "extent."
)


def apply_crop_correction(metadata: dict[str, Any], extras: Any) -> dict[str, Any]:
    """Overlay a recorded crop correction onto collection metadata.

    Shapes only, and deliberately so. `derivation_profile.crop_z_top` is left
    exactly as stored: it is a truthful record of what was REQUESTED, and a
    derived child built from this metadata must keep inheriting that intent --
    rewriting it here would silently change the child's DATA, not its record.
    What did not happen is recorded beside it in `effective_crop_z_top`.
    """
    if not extras.get("crop_correction", None):
        return metadata
    spatial = extras.get("effective_spatial_shape", None)
    if spatial:
        metadata["spatial_shape"] = list(spatial)
    shapes = extras.get("effective_array_shapes", None)
    if shapes:
        metadata["array_shapes"] = {name: list(shape) for name, shape in shapes.items()}
    return metadata


def _member_z(node) -> int | None:
    """The z-extent a member really holds. Prefer the recorded per-member shapes.

    `member_spatial_shapes` is read straight off each member's own metadata and is
    never passed through the derivation, so it is truthful where it exists. Falling
    back to loading member 0 covers collections created before that key.
    """
    metadata = node.base.attributes.get("dataset_metadata", None) or {}
    shapes = metadata.get("member_spatial_shapes") or []
    if shapes and shapes[0] and len(shapes[0]) >= 3:
        return int(shapes[0][-1])
    uuids = metadata.get("member_simulation_uuids") or node.base.attributes.get(
        "member_simulation_uuids", None
    ) or []
    if not uuids:
        return None
    try:
        member = orm.load_node(uuids[0])
        member_shape = (member.get_simulation_metadata() or {}).get("spatial_shape") or []
        return int(member_shape[-1]) if len(member_shape) >= 3 else None
    except Exception:  # noqa: BLE001 - an unreadable member is not our business
        return None


def _phantom_crop(node) -> dict[str, Any] | None:
    """The correction this collection needs, or None if it needs none.

    Returns None rather than guessing whenever the record does not carry the exact
    signature of an unapplied crop: recorded z == member z - crop, on the spatial
    shape AND on every array of rank >= 3. A record that disagrees with itself is
    left alone for a human.
    """
    attributes = node.base.attributes
    role = str(attributes.get("dataset_role", None) or "base")
    if role == "derived":
        return None
    profile = attributes.get("derivation_profile", None) or {}
    try:
        crop = int(profile.get("crop_z_top") or 0)
    except (TypeError, ValueError):
        return None
    if crop <= 0:
        return None

    metadata = attributes.get("dataset_metadata", None) or {}
    spatial = list(metadata.get("spatial_shape") or [])
    if len(spatial) < 3:
        return None
    member_z = _member_z(node)
    if member_z is None:
        return None

    expected = member_z - crop
    if expected <= 0 or int(spatial[-1]) != expected:
        # Either the crop was refused at creation (no gas layer on this grid, so the
        # record already tells the truth) or the record means something we did not
        # predict. Neither is ours to rewrite.
        return None

    shapes = dict(metadata.get("array_shapes") or {})
    corrected: dict[str, list[int]] = {}
    for name, shape in shapes.items():
        shape = list(shape)
        if len(shape) >= 3:
            if int(shape[-1]) != expected:
                return None  # the record disagrees with itself; leave it
            shape[-1] = member_z
        corrected[name] = shape

    return {
        "effective_crop_z_top": 0,
        "effective_spatial_shape": [*spatial[:-1], member_z],
        "effective_array_shapes": corrected,
        "crop_correction": {
            "from": crop,
            "to": 0,
            "date": CORRECTION_DATE,
            "role": role,
            "recorded_z": int(spatial[-1]),
            "actual_z": member_z,
            "reason": REASON,
        },
    }


def _stamped_collections() -> list[int]:
    from aiida_pino.data.pino_dataset_collection import PinoDatasetCollectionData

    query = orm.QueryBuilder().append(PinoDatasetCollectionData, project=["id"])
    return [pk for (pk,) in query.all()]


def backfill(*, apply: bool = False, revert: bool = False) -> dict[str, Any]:
    stats = {
        "collections": 0,
        "already": 0,
        "corrected": 0,
        "skipped_derived": 0,
        "skipped_no_crop": 0,
        "skipped_unclear": 0,
        "reverted": 0,
        "by_role": {},
        "z_moves": {},
    }
    for pk in _stamped_collections():
        node = orm.load_node(pk)
        stats["collections"] += 1

        if revert:
            if node.base.extras.get("crop_correction", None) is not None:
                stats["reverted"] += 1
                if apply:
                    for key in CORRECTION_KEYS:
                        try:
                            node.base.extras.delete(key)
                        except (AttributeError, KeyError):
                            pass
            continue

        if node.base.extras.get("crop_correction", None) is not None:
            stats["already"] += 1
            continue

        attributes = node.base.attributes
        role = str(attributes.get("dataset_role", None) or "base")
        profile = attributes.get("derivation_profile", None) or {}
        if not (profile.get("crop_z_top") or 0):
            stats["skipped_no_crop"] += 1
            continue
        if role == "derived":
            stats["skipped_derived"] += 1
            continue

        correction = _phantom_crop(node)
        if correction is None:
            stats["skipped_unclear"] += 1
            continue

        stats["corrected"] += 1
        stats["by_role"][role] = stats["by_role"].get(role, 0) + 1
        move = f"{correction['crop_correction']['recorded_z']}->{correction['crop_correction']['actual_z']}"
        stats["z_moves"][move] = stats["z_moves"].get(move, 0) + 1
        if apply:
            for key, value in correction.items():
                node.base.extras.set(key, value)
    return stats


def main(argv: list[str] | None = None) -> int:
    import argparse

    from aiida import load_profile

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the extras (default: dry run)")
    parser.add_argument("--revert", action="store_true", help="remove extras this module wrote")
    parser.add_argument("--profile", default=None, help="AiiDA profile (default: the configured default profile)")
    args = parser.parse_args(argv)

    load_profile(args.profile)
    stats = backfill(apply=args.apply, revert=args.revert)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"[crop-backfill] {mode}{' (revert)' if args.revert else ''}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
