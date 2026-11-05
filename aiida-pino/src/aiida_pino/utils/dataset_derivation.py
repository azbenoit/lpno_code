"""Helpers for canonical and derived PINO dataset handling."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any

import numpy as np
from aiida import orm

from aiida_pino.data import (
    PinoDatasetCollectionData,
    PinoDatasetData,
    PinoDatasetUnionData,
    build_family_identifier,
    derive_simulation_arrays,
    datasource_cache_basename_from_metadata,
    is_canonical_metadata,
)
from aiida_pino.utils.stride_npz import normalize_stride

_LOGGER = logging.getLogger(__name__)


#: The canonical (lossless base) dataset layout.
#:
#: `time_as_batch`, not `time_as_dim`: folding the trajectory into a grid axis
#: requires every case to hold the same number of frames, and real FLOW-3D
#: random-path data is ragged (one 300-case set spans 14 to 400 frames). A layout
#: that cannot represent the data cannot be its lossless base — six collections
#: were stamped `time_as_dim` with `uniform_timesteps=False`, an impossible
#: combination no derivation could satisfy, and it is what every FLOW-3D model
#: here trains on anyway.
#:
#: Safe to flip because the layout is a *reading* instruction, not a storage
#: format: members hold `(T, C, X, Y, Z)` on disk either way and
#: `derive_simulation_arrays` applies the layout at export time.
DEFAULT_DERIVATION_PROFILE = {
    "include_time": True,
    "time_as_batch": True,
    "time_as_dim": False,
    "moving_ref_frame": False,
    "moving_frame_steps": 1,
    "time_avg_window": 1,
    "super_res_factor": 1,
    # Spatial decimation baked into the *stored* arrays. `PINO_LAZY_STRIDE`
    # decimates after a full-resolution inflate, so it saves GPU memory and no
    # loader time; as a derivation the inflate itself shrinks (4.77 -> 1.19
    # GB/case on 1601). Carrying it here also makes the resolution part of the
    # dataset's identity, so a strided copy can never be double-strided by a
    # leftover env var.
    "stride": [1, 1, 1],
    # Keep N evenly-spaced frames, first and last always included; None keeps
    # every frame. A time subsample was previously unrepresentable, so a dataset
    # that held 20 of 60 frames carried a derivation profile byte-identical to
    # the full-resolution parent's and had to be marked unreconstructable.
    "time_frames": None,
    # Extra fields to BUILD during derivation, in the lab frame, before the
    # crop/average. Empty by default, so every existing submission produces
    # byte-identical arrays. See `derive_simulation_arrays` for why they cannot
    # be added to an already-staged steady member instead.
    "derived_fields": [],
    # Drop N topmost z layers from every field. The gas sits at high z and every current
    # NPZ export carries exactly 10 gas layers (100 um above z=0) in all four regimes, so
    # a model trained on a gas-cropped domain (e.g. z_range [-300, 0] um, 30 cells) cannot
    # otherwise be scored on current data: it is fed 10 layers of void it never saw, and
    # every metric is void. Expressed in LAYERS rather than a z_range because the arrays
    # are indexed and a layer count is exact; a physical range would need the axis and
    # could land between cells (2026-08-25).
    # TEN cells of gas. Every deck in this project puts the substrate surface at z = 0 with
    # 95 um of headroom above it at dz = 10 um, so the gas is exactly ten cells on every grid
    # in the store -- verified across nz = 40, 50, 60 and 80, which differ only in how deep
    # they go into the METAL (zmin -295, -395, -495, -695 um). Cropping it is the default
    # because a model has nothing to learn from the void above the surface; training on the
    # full extent is still available, it just has to be asked for with `crop_z_top: 0`.
    "crop_z_top": 10,
    #: Keep read-time-capable transforms IN the derivation so they are baked into staged
    #: arrays, instead of being demoted to loader env vars. Default False: demotion is right
    #: almost always (a stride alone once wrote 46 GB of `.npy` for a free transform). Set it
    #: when READ CPU dominates -- a demoted profile stages pack-index byte ranges into the
    #: object store, so the loader reads the stored DEFLATE `payload.npz` and inflates on
    #: every access: 13800 ms/iter with the GPU idle against 133 for a comparable run.
    #: Gated in series by `room_for_uncompressed_stage`, which decides whether the
    #: destination can afford the result.
    "materialize": False,
}
DERIVATION_PROFILE_KEYS = frozenset(DEFAULT_DERIVATION_PROFILE)

#: Fields `derived_fields` may request. An allowlist so a typo fails at submit
#: rather than after staging, which is minutes of work and a whole collection.
BUILDABLE_DERIVED_FIELDS = frozenset({"conduction_field"})


def buildable_derived_fields() -> frozenset[str]:
    """`BUILDABLE_DERIVED_FIELDS` plus every material channel the datasource can build.

    Resolved from `lpno` rather than restated here, because a hardcoded copy
    drifts the moment a channel is added there -- which is how this gap appeared. The
    static set is the fallback when the sibling package is not importable, so submit-time
    validation still works on a machine without it.

    Material channels are constant in space and time by construction, so the crop and the
    30-frame average are identity operations on them. That is why they need none of the
    lab-frame-then-transform ordering `conduction_field` needs: there is no
    build-after-the-transform hazard because the value does not depend on the frame.
    """
    try:
        from lpno.custom_utils.npz_datasource import ALL_MATERIAL_CHANNEL_NAMES
    except Exception:  # noqa: BLE001 - validation must not require the sibling package
        return BUILDABLE_DERIVED_FIELDS
    return BUILDABLE_DERIVED_FIELDS | frozenset(ALL_MATERIAL_CHANNEL_NAMES)


def canonical_base_layout_profile() -> dict[str, Any]:
    """Return the canonical lossless base-dataset layout."""
    return dict(DEFAULT_DERIVATION_PROFILE)


#: Derivation keys the LOADER can apply at read time, and the env var that asks it to.
#: Both are guarded in lpno against composing with an already-baked value
#: (`_variant_stride`, `_variant_crop_z_top`), which is what makes moving them safe.
READ_TIME_TWINS = {
    "stride": "PINO_LAZY_STRIDE",
    "crop_z_top": "PINO_LAZY_CROP_Z_TOP",
}

#: What each read-time key looks like when it transforms NOTHING. Not the package default:
#: since 2026-08-26 the default crop is 10, so "identity" and "default" are different
#: profiles and conflating them makes every dataset look transformed.
IDENTITY_DERIVATION_VALUES = {
    "stride": [1, 1, 1],
    "crop_z_top": 0,
}


def _twin_env_value(key: str, value) -> str | None:
    """The env string for a transform worth demoting, or None if it is a no-op."""
    if key == "stride":
        try:
            sx, sy, sz = (int(v) for v in value)
        except (TypeError, ValueError):
            return None
        if (sx, sy, sz) == (1, 1, 1):
            return None
        return f"{sx},{sy},{sz}"
    if key == "crop_z_top":
        try:
            crop = int(value)
        except (TypeError, ValueError):
            return None
        return str(crop) if crop > 0 else None
    return None


def demote_read_time_transforms(profile: dict | None, env: dict | None, *, read_time_supported: bool):  # noqa: E501
    """Move read-time-capable transforms out of the derivation profile into loader env vars.

    Returns `(profile, env, notes)`, always fresh dicts. `notes` is empty when nothing moved,
    which is also the signal that the caller should report nothing.

    The point is disk. `_export_members` copies a member whenever
    `derivation_changes_payload(profile)` is true, and a stride alone is enough to trigger
    that: `mb40_test_8case` wrote 46 GB of uncompressed `.npy` whose only cause was a
    `stride: [2,2,1]` the loader applies for free. Emptied of its read-time keys the profile
    is a no-op, so the members stage as `pack_index.json` byte ranges into the object store --
    a few KB.

    Three refusals, each of which would otherwise make things WORSE rather than better:

    - **Only when it achieves zero copy.** If anything else still rewrites the arrays the
      copy happens regardless, and removing the stride makes that copy four times bigger.
    - **Never when the target cannot apply the transform at read time.** A checkout that does
      not know `PINO_LAZY_CROP_Z_TOP` ignores it silently and trains on the full extent while
      the profile claims a crop -- a wrong number with no error, which is why this is a
      parameter and not an assumption.

    A remote target is NOT a reason to refuse, which is a correction to the first version of
    this helper. Pack-index byte ranges genuinely cannot travel, but the fallback for an
    untransformed `npz_file` member is a hardlink-or-copy of the stored `payload.npz`, still
    COMPRESSED. So the choice off-box is between copying 13.3 GB of archives and baking
    309 GB of `.npy` (measured on a 96-member collection) -- demoting wins by 23x there too.
    - **Never onto an env value the caller already set**, which would compose two reductions
      into one silently squared grid.
    """
    from aiida_pino.data.pino_dataset_collection import derivation_changes_payload

    profile = dict(profile or {})
    env = dict(env or {})
    if not read_time_supported or not derivation_changes_payload(profile):
        return profile, env, []
    if bool((profile or {}).get("materialize")):
        # The caller asked for these transforms BAKED. Demoting would empty the profile,
        # stage pack-index byte ranges into the object store, and leave the loader inflating
        # the stored DEFLATE payload on every read -- which is the cost `materialize` exists
        # to remove. Whether the destination can afford the result is decided separately by
        # `room_for_uncompressed_stage`; the two are in series and this one only says "ask".
        # NO notes. `notes` is a list of KEY NAMES that callers splice into "<keys> moved to
        # the loader, so the dataset stages pack-indexed instead of being copied" -- returning
        # prose here produced a report describing the OPPOSITE of what happened, and read as
        # evidence the flag was not taking its branch. Under `materialize` nothing moves to
        # the loader, and an empty list is already the documented signal to report nothing.
        return profile, env, []

    # Work from the EFFECTIVE profile, not the caller's literal keys. A stride-only profile
    # still carries the DEFAULT gas crop, and that crop transforms the arrays just as much
    # as one written out by hand -- so demoting only what the caller typed left the implicit
    # crop baked, the profile payload-changing, and the collection materialised.
    try:
        candidate = dict(validate_derivation_profile(dict(profile)))
    except Exception:  # noqa: BLE001 - an invalid profile is not this function's to reject
        candidate = dict(profile)
    moved: dict[str, str] = {}
    # Keys already expressed in `env` with the SAME value. They need no env update, but they
    # DO count as demoted -- otherwise `if not moved` below returns the profile with the key
    # still baked, which is the 2026-08-28 materialisation.
    already: dict[str, str] = {}
    for key, env_name in READ_TIME_TWINS.items():
        if key not in candidate:
            continue
        value = _twin_env_value(key, candidate[key])
        if value is None:
            candidate[key] = IDENTITY_DERIVATION_VALUES[key]  # already a no-op
            continue
        existing = str(env.get(env_name, "") or "").strip()
        if existing:
            # An env value the caller ALREADY set is only a composition when it DIFFERS.
            # When it is the same transform written twice, bailing out here left the key
            # baked in the profile, which made the derivation payload-changing (so the
            # collection MATERIALISED) and made the staged variant record a crop the env
            # var also asks for (so `_resolve_read_time_crop` then refused it as a
            # double-crop). Measured 2026-08-28: 117 GB written in 90 minutes, root to
            # 99 %, and every run that caused it failed anyway. Identical values are the
            # normal way a crop-equivalent rerun is expressed; treat them as already
            # demoted -- set the identity, do NOT re-announce the env var.
            if existing == value:
                candidate[key] = IDENTITY_DERIVATION_VALUES[key]
                already[env_name] = value
                continue
            return profile, env, []  # genuinely ambiguous; composing would square it
        # Set the IDENTITY value, do not pop the key. Popping worked only while every
        # default was already the identity: once `crop_z_top` defaulted to 10, deleting it
        # let validation put the crop straight back, the candidate stayed payload-changing,
        # and the demotion declined -- so a cropped dataset materialised instead of staging
        # pack-indexed, which is the opposite of the point.
        candidate[key] = IDENTITY_DERIVATION_VALUES[key]
        moved[env_name] = value

    if not (moved or already) or derivation_changes_payload(candidate):
        # Something else still rewrites the arrays: the copy is happening, and a
        # full-resolution copy is strictly worse than the strided one.
        return profile, env, []

    env.update(moved)
    # `already` entries are reported too: the caller uses `notes` to decide whether the
    # profile changed, and it did -- the key went to identity even though env was untouched.
    notes = [f"{name}={value}" for name, value in sorted({**moved, **already}.items())]
    return candidate, env, notes


def demote_for_code(profile, training_profile, code):
    """`demote_read_time_transforms` with `local` taken from the code's own computer.

    One entry point because the train and eval workchains both need it, and a rule
    implemented twice in this repo has diverged every single time.

    Reuses `pack_index_allowed_for_transport` -- the SAME test staging uses to decide
    whether a pack index can be written -- rather than a second opinion about what
    "local" means. If those two ever disagree, demotion would strip a stride from a
    collection that then gets copied at full resolution.
    """
    from aiida_pino.data.pino_dataset_collection import pack_index_allowed_for_transport

    computer = getattr(code, "computer", None)
    if computer is None:
        transport = "core.local"
    else:
        # `transport_type`, a PROPERTY. An earlier version called a `get_transport_type()`
        # that does not exist on `orm.Computer`; the AttributeError was swallowed by a broad
        # `except` and the demotion silently did nothing for EVERY run with a computer --
        # green tests included, because the fake computer in them defined the invented
        # method. Read it defensively but never silently: an unreadable transport falls back
        # to treating the target as remote-unknown, which bakes, which is the safe direction.
        transport = getattr(computer, "transport_type", None)
        if not transport:
            getter = getattr(computer, "get_transport_type", None)
            transport = getter() if callable(getter) else ""
        if not transport:
            return dict(profile or {}), dict((training_profile or {}).get("env") or {}), []

    # Local always supports it -- same interpreter, same checkout. A compute box supports it
    # only once its `lpno` carries the read-time crop (93255ff) and its guard
    # (e4ef66d); a box behind those ignores the env var and trains the full extent against a
    # profile claiming a crop. `AIIDA_PINO_REMOTE_READ_TIME` lists the computer labels
    # confirmed synced, so enabling a box is a deliberate act after the sync rather than an
    # assumption baked into the code.
    import os as _os

    if pack_index_allowed_for_transport(transport):
        supported = True
    else:
        allowed = {
            label.strip()
            for label in _os.environ.get("AIIDA_PINO_REMOTE_READ_TIME", "").split(",")
            if label.strip()
        }
        supported = bool(allowed) and str(getattr(computer, "label", "") or "") in allowed

    return demote_read_time_transforms(
        profile,
        (training_profile or {}).get("env") or {},
        read_time_supported=supported,
    )


def report_materialisation_cost(reporter, base, profile: dict) -> None:
    """Announce a materialising derivation's cost BEFORE it starts.

    The loader-side frame (`PINO_LAZY_MOVING_REF`) is the normal route, and materializing
    hundreds of GB for it is not a reasonable default. The
    materialising route stays reachable -- it is NOT the same experiment, since the lazy
    store refuses `moving_ref_frame=true` and so cannot produce the 30-frame beam-frame
    average at all -- but it must announce itself rather than committing silently and being
    caught only by a size gate, or not at all on a machine with room. That silence is how
    root reached 1.3 GB free.
    """
    from aiida_pino.utils.runtime_cache import estimated_derived_bytes

    try:
        members = list(base.iter_member_simulations())
    except Exception:  # noqa: BLE001 - never let the warning break the submission
        return
    if not members:
        return
    try:
        needed = estimated_derived_bytes(members, profile)
    except Exception:  # noqa: BLE001
        return

    detail = ""
    if profile.get("moving_ref_frame"):
        detail = (
            " This is the MATERIALISING beam-frame route; the loader-side alternative is "
            "`PINO_LAZY_MOVING_REF=\"behind,ahead\"` in the profile env, which stages nothing "
            "but cannot apply a time average."
        )
    reporter(
        f"Materialising a derived variant of {len(members)} members: about "
        f"{needed / 1e9:.1f} GB will be written to the local runtime cache.{detail}"
    )

def reconcile_crop_env(env: dict | None, dataset_metadata: dict | None):
    """Drop a read-time crop that the DATASET already bakes. Returns `(env, note)`.

    The crop can live in two places -- baked into a derived collection, or carried as
    `PINO_LAZY_CROP_Z_TOP` for the loader -- and the loader refuses when both are set,
    because applying them twice removes 20 cells where 10 were meant. That refusal is right,
    but WHICH place holds the crop is decided elsewhere: a plain training gets an uncropped
    collection and needs the env var, while an eval, or a training that sets
    `PINO_LAZY_STRIDE`, resolves to a collection that already bakes it and must NOT. Nothing
    states that pairing at submit time, so it arrives as a refusal one submission too late --
    it cost a ten-arm batch a day after the eval half of the same rule had been
    documented.

    When the two AGREE the intent is unambiguous, so the env var is dropped and the caller
    reports it. When they DISAGREE nothing is guessed: the env var is left alone and the
    loader's own guard refuses, naming both numbers.
    """
    env = dict(env or {})
    raw = str(env.get("PINO_LAZY_CROP_Z_TOP", "") or "").strip()
    if not raw:
        return env, ""
    try:
        wanted = int(raw)
    except ValueError:
        return env, ""
    # A derived collection keeps its crop in `derivation_profile`, not at the top level, so
    # reading only the top level made `baked` 0 and this returned the env untouched while the
    # loader refused the double crop at run time.
    #
    # But `derivation_profile.crop_z_top` alone does NOT mean the crop is materialised, and
    # acting on it where it is not turns a loud crash into a silent wrong number -- strictly
    # worse. Measured 2026-09-04 on a base collection (`crop_z_top: 10`): its 13
    # members serve RAW 40-cell arrays, read directly off the payload, and an FNO eval of it
    # measured `substrate_fraction` 0.75 = 30 metal cells of 40. The crop there is a REQUEST
    # to apply at read time, which `PINO_LAZY_CROP_Z_TOP` is the mechanism for -- dropping it
    # removes the only thing that would crop, and the eval then scores an uncropped domain
    # against baselines taken on a cropped one. 399 base collections carry such a record.
    #
    # `array_shapes` cannot arbitrate: on that collection it records the POST-crop 30, on two
    # derived ones the RAW 40/50. Two spellings of the same field, so comparing records only
    # compares records.
    #
    # So the crop counts as baked only where materialisation is established: `dataset_role
    # == "derived"`, which is what `materialize_derived_dataset` writes. Anything else keeps
    # the env var, which restores the loader's refusal -- loud, and the status quo before
    # this reconciliation existed. Being too narrow costs a clear error message; being too
    # broad costs a wrong number nobody sees.
    metadata = dataset_metadata or {}
    baked_raw = metadata.get("crop_z_top")
    if baked_raw in (None, ""):
        profile = metadata.get("derivation_profile")
        if isinstance(profile, dict) and str(metadata.get("dataset_role", "")) == "derived":
            baked_raw = profile.get("crop_z_top")
    try:
        baked = int(baked_raw or 0)
    except (TypeError, ValueError):
        return env, ""
    if baked <= 0:
        return env, ""
    if baked != wanted:
        # Ambiguous: two different crops asked for. Leave it to the loader's guard, which
        # names both numbers and both mechanisms.
        return env, ""
    env.pop("PINO_LAZY_CROP_Z_TOP", None)
    return env, (
        f"the resolved dataset already bakes crop_z_top={baked}, so PINO_LAZY_CROP_Z_TOP "
        "was dropped from the env; the geometry is unchanged (applying both would remove "
        f"{baked * 2} cells instead of {baked})"
    )


def validate_derivation_profile(values: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize and validate a derivation profile."""
    if values is None:
        return canonical_base_layout_profile()
    if not isinstance(values, dict):
        raise ValueError("derivation_profile must be a mapping.")

    profile = dict(DEFAULT_DERIVATION_PROFILE)
    unknown = sorted(set(values).difference(DERIVATION_PROFILE_KEYS))
    if unknown:
        raise ValueError(f"Unknown derivation_profile keys: {', '.join(unknown)}")
    profile.update(values)

    profile["include_time"] = bool(profile["include_time"])
    profile["time_as_batch"] = bool(profile["time_as_batch"])
    profile["time_as_dim"] = bool(profile["time_as_dim"])
    profile["moving_ref_frame"] = bool(profile["moving_ref_frame"])
    profile["moving_frame_steps"] = int(profile["moving_frame_steps"])
    profile["time_avg_window"] = int(profile["time_avg_window"])
    profile["super_res_factor"] = int(profile["super_res_factor"])
    if profile["time_frames"] is not None:
        profile["time_frames"] = int(profile["time_frames"])
        if profile["time_frames"] < 2:
            raise ValueError(
                "derivation_profile.time_frames must be >= 2 (first and last are both kept), "
                "or None to keep every frame."
            )
    # Normalized to a list so the stored profile round-trips through JSON and
    # two equal strides always hash alike (a tuple and a list would not).
    profile["stride"] = list(normalize_stride(profile["stride"]))

    # SORTED and de-duplicated, for the same reason `stride` is normalised
    # above: order and repetition must not change identity, or two profiles
    # asking for the same thing hash differently and stage the same data twice.
    crop = profile.get("crop_z_top", 0)
    try:
        crop_int = int(crop if crop is not None else 0)
    except (TypeError, ValueError):
        raise ValueError(f"derivation_profile.crop_z_top must be a non-negative integer; got {crop!r}.")
    if crop_int < 0:
        raise ValueError(f"derivation_profile.crop_z_top must be >= 0; got {crop_int}.")
    profile["crop_z_top"] = crop_int

    requested = profile.get("derived_fields") or []
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, (list, tuple, set)):
        raise ValueError("derivation_profile.derived_fields must be a list of field names.")
    names = sorted({str(item).strip() for item in requested if str(item).strip()})
    unknown = [name for name in names if name not in buildable_derived_fields()]
    if unknown:
        raise ValueError(
            f"derivation_profile.derived_fields contains unknown field(s): "
            f"{', '.join(unknown)}. Buildable: {', '.join(sorted(BUILDABLE_DERIVED_FIELDS))}."
        )
    profile["derived_fields"] = names
    if "materialize" in values:
        raw = values["materialize"]
        if not isinstance(raw, bool):
            # `bool("false")` is True; a flag that silently switches itself on would reverse
            # a measured 23x disk win by accident.
            raise ValueError("derivation_profile.materialize must be a boolean.")
        profile["materialize"] = raw

    if profile["super_res_factor"] <= 0:
        raise ValueError("derivation_profile.super_res_factor must be positive.")
    if profile["moving_frame_steps"] <= 0:
        raise ValueError("derivation_profile.moving_frame_steps must be positive.")
    if profile["time_avg_window"] <= 0:
        raise ValueError("derivation_profile.time_avg_window must be positive.")

    # The moving frame exists only as the beam-centred average of the steady layout, so a
    # request for it implies `time_as_batch: False`. Left to the base layout's default
    # (True) it used to be switched off without a word (changed 2026-10-30).
    if profile["moving_ref_frame"] and profile["include_time"] and not profile["time_as_dim"]:
        if "time_as_batch" not in values:
            profile["time_as_batch"] = False
            _LOGGER.warning(
                "derivation_profile: moving_ref_frame=True with time_as_batch unset; inferring "
                "time_as_batch=False (the beam-centred, time-averaged layout). State it to silence this."
            )
        elif profile["time_as_batch"]:
            raise ValueError(
                "derivation_profile: moving_ref_frame=True contradicts time_as_batch=True. The moving "
                "frame averages a beam-centred window over time into one sample, which a "
                "one-sample-per-frame layout cannot hold. Drop time_as_batch or set it to false."
            )

    if profile["time_as_dim"]:
        profile["include_time"] = True
        profile["time_as_batch"] = False

    if not profile["include_time"]:
        profile["time_as_batch"] = False
        profile["time_as_dim"] = False
        profile["moving_ref_frame"] = False
        profile["moving_frame_steps"] = 1
        profile["time_avg_window"] = 1

    if profile["time_as_batch"] or profile["time_as_dim"]:
        profile["moving_ref_frame"] = False
        profile["moving_frame_steps"] = 1
        profile["time_avg_window"] = 1

    if profile["time_avg_window"] > 1 and not profile["moving_ref_frame"]:
        # Name the OTHER half. Alone, this reads as "that combination is
        # invalid" and points at relaxing this check — which is where it pointed
        # a reader who then nearly asked for a correct guard to be weakened, and
        # it cost three submissions to find the real shape. A steady run states
        # steady in BOTH the training profile and an explicit derivation
        # profile; each validator here is individually right and neither knows
        # the other exists.
        raise ValueError(
            "derivation_profile.time_avg_window > 1 requires moving_ref_frame=True. "
            "For a steady (time-averaged, beam-centred) run, set BOTH consistently: "
            "`moving_ref_frame: true` with your `time_avg_window` in the derivation "
            "profile, AND the matching steady settings in the training profile — "
            "passing the derivation profile explicitly is what states that the "
            "substitution is intended."
        )

    return profile


#: Bump when the DERIVING CODE changes in a way that changes the bytes it
#: produces, so variants built by the previous version stop being reused.
#:
#: Deliberately NOT part of `derivation_hash`. The hash is the reuse KEY:
#: folding anything new into it re-hashes every profile, so no existing variant
#: matches any request and all of them re-materialise at full cost. That is not
#: hypothetical — `derivation_hash`'s own docstring records it happening when
#: `stride` was added unconditionally, making one test-split collection a
#: duplicate of another. The version is recorded BESIDE the hash and filtered on separately, so
#: identity never moves.
#:
#: Version 1 is "whatever produced the variants existing on 2026-08-20". It does
#: NOT retroactively invalidate them: they carry no recorded version and stay
#: reusable, because refusing them would be the mass re-derivation this design
#: avoids. Invalidating known-bad existing variants is a separate, deliberate
#: act (mark them `superseded`), not a side effect of introducing this.
DERIVATION_CODE_VERSION = 1


def reuse_filters_for(base_uuid: str, profile: dict[str, Any]) -> dict[str, Any]:
    """QueryBuilder filters selecting a reusable derived variant.

    Matches the base and the derivation hash as before, and additionally
    requires the recorded code version to be either the current one or ABSENT.
    Absent means "built before versions were recorded" — reusable, because
    excluding those would re-derive every existing variant.
    """
    return {
        "attributes.base_collection_uuid": {"==": str(base_uuid)},
        "attributes.derivation_hash": {"==": derivation_hash(profile)},
        "attributes.dataset_role": {"==": "derived"},
        "or": [
            {"attributes.derivation_code_version": {"==": DERIVATION_CODE_VERSION}},
            {"attributes": {"!has_key": "derivation_code_version"}},
        ],
    }


def derivation_hash(profile: dict[str, Any]) -> str:
    """Return a stable hash for a normalized derivation profile.

    A **default** stride is omitted from the payload so the hash matches what
    was stored before `stride` became a derivation key. Including it
    unconditionally silently re-hashed every unstrided profile, so
    `find_reusable_derived_dataset` stopped recognising existing variants and
    built duplicates: one test split was created as a second copy of another with
    the same 53 simulations and the same layout, differing only in the hash.
    Only a non-default stride changes identity -- which is the point, since a
    strided copy really is a different dataset.
    """
    encoded = json.dumps(
        identity_derivation_profile(profile), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def identity_derivation_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """The part of a derivation profile that is allowed to change identity.

    Every consumer that decides "is this the same dataset" must hash *this*,
    not the raw profile. `derivation_hash` alone was normalized first, while
    `collection_content_signature` kept hashing the raw dict — so a default
    stride still minted a twin through the content-addressed path even after
    the hash was fixed. Dataset 12968 (which holds every published evaluation
    of the 1601 test split) and 23659 hold the same 53 simulations and differ
    only by `stride: None` versus `stride: [1, 1, 1]`, which is the same thing.
    """
    normalized = dict(validate_derivation_profile(profile))
    if list(normalized.get("stride", [1, 1, 1])) == [1, 1, 1]:
        normalized.pop("stride", None)
    # Same rule as stride: "keep every frame" is what no subsample means, so a
    # present-but-default value must not fork a dataset.
    if normalized.get("time_frames") is None:
        normalized.pop("time_frames", None)
    # Same rule again: a profile that does not ask to materialise must hash exactly as it did
    # before the key existed, or every stored dataset forks a twin (23659 of 12968).
    # `materialize: true` DOES change identity, because a baked crop and a read-time crop
    # produce different stored arrays.
    if not normalized.get("materialize"):
        normalized.pop("materialize", None)
    return normalized


def dataset_layout_profile_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract the effective dataset layout profile from dataset metadata.

    `crop_z_top` is read from the metadata and defaults to ZERO, never to the package
    default. Every stored dataset was derived before the gas crop existed, so filling it
    from today's default makes each one read as "cropped 10" while an evaluation
    reconstructing its model asks for 0 -- the layouts then disagree and `_resolve_source`
    MATERIALISES a new variant instead of serving the dataset it was handed. Caught by
    `test_evaluate_model_integration_records_training_provenance`, which recorded
    `dataset_source_pk` 12 for a dataset passed in as 5.

    Same rule the stride branch below already applies for the same reason.
    """
    time_axis_mode = metadata.get("time_axis_mode", {})
    return validate_derivation_profile(
        {
            "crop_z_top": metadata.get("crop_z_top", 0),
            "include_time": bool(time_axis_mode.get("include_time", metadata.get("time_axis", False))),
            "time_as_batch": bool(time_axis_mode.get("time_as_batch", False)),
            "time_as_dim": bool(time_axis_mode.get("time_as_dim", False)),
            "moving_ref_frame": bool(metadata.get("moving_ref_frame", False)),
            "moving_frame_steps": int(metadata.get("moving_frame_steps", 1)),
            "time_avg_window": int(metadata.get("avg_window_size", 1)),
            "super_res_factor": int(metadata.get("super_res_factor", 1)),
            # Absent on every dataset stored before stride became a derivation,
            # so default to the no-op rather than treating them as unstrided-
            # but-unknown: they were built at full resolution.
            "stride": metadata.get("stride", [1, 1, 1]),
        }
    )


def _recorded_crop_z_top(training_profile: dict, unrecorded: int) -> int:
    """The gas crop a profile records, or `unrecorded` when it names none.

    The fallback is the caller's decision because the two callers want opposite things, and
    conflating them breaks one of them:

    - **Submitting** a new training that names no crop should get the package DEFAULT, which
      is the whole point of making the gas crop default.
    - **Reconstructing** what a stored model was trained on must fall back to ZERO. A profile
      that never mentions a crop belongs to a model trained before the crop existed, and
      scoring it on a cropped grid answers a different question than it was trained for. On
      the live profile that is 449 of 451 models.

    A first version of this took no argument and always answered "recorded or 0", which was
    right for evaluation and silently disabled the new default for every new training.
    """
    env = training_profile.get("env") or {}
    raw = env.get("PINO_LAZY_CROP_Z_TOP") if isinstance(env, dict) else None
    if raw is None:
        raw = training_profile.get("crop_z_top")
    if raw is None:
        return int(unrecorded)
    try:
        return max(int(str(raw).strip()), 0)
    except (TypeError, ValueError):
        return int(unrecorded)  # unparseable means unrecorded, not "invent a crop"


def domain_extent_mismatch(
    training_profile: dict,
    eval_derivation_profile: dict,
    evaluation_profile: dict | None = None,
    training_derivation_profile: dict | None = None,
) -> str | None:
    """Refusal message when an evaluation would score a model on a different DOMAIN.

    Returns None when the extents agree, so the caller can `if msg: refuse`.

    A RESOLUTION difference is not a mismatch and must never be refused here: trained
    strided / evaluated full-res is the norm for RIGNO, the graph
    rebuilds from whatever grid it is handed, and control 9716 re-scored unstrided gave
    16.93 K against the strided 16.96 K. `crop_z_top` is a different axis -- it decides how
    much of the domain EXISTS. A model trained with the gas cropped has never seen those
    cells, so scoring it on the uncropped grid charges it for every one of them.

    Why this is checked at all: `evaluate_model` reconstructs the model's own domain only
    when NO `derivation_profile` input is supplied. When one IS supplied it was trusted
    outright, and a `train_eval` passes the TRAINING derivation to both halves. On
    2026-08-27 that derivation carried `crop_z_top: 0` (the crop rode in the env instead, to
    clear the double-crop guard), so the eval served 40 cells to a model trained on 30:
    whole-domain MAE 816 K against 91 K for a comparable arm, while SUBSTRATE MAE was within
    2 %. Plausible numbers, no error, and it read as a broken architecture for half a day.
    """
    trained = _recorded_crop_z_top(training_profile, 0)
    # A crop BAKED into the training derivation is still the domain the model saw. A layout
    # that cannot lazy-load (`moving_ref_frame`) MUST bake it and drop the env var, so the
    # training profile is legitimately crop-free while the model trained 30-deep. Reading the
    # env alone made the guard refuse an eval whose domains agree -- caught 2026-08-29 on the
    # three steady crop arms, which would have trained 32000 steps each and then been refused.
    # Models record this: `get_model_metadata()['derivation_profile']`.
    if not trained and training_derivation_profile:
        try:
            trained = int((training_derivation_profile or {}).get("crop_z_top", 0) or 0)
        except (TypeError, ValueError):
            trained = 0
    # The crop can ride in EITHER place, and `crop_z_top: 0` in the derivation with the crop
    # in the env is the CORRECT spelling -- it is how `_resolve_read_time_crop`'s double-crop
    # refusal is avoided. Reading only the derivation field would refuse a configuration that
    # serves exactly the right domain. The two can never both be non-zero (that is the very
    # thing the double-crop guard refuses), so "whichever is set" is a max.
    try:
        served = int((eval_derivation_profile or {}).get("crop_z_top", 0) or 0)
    except (TypeError, ValueError):
        return None
    served = max(served, _recorded_crop_z_top(evaluation_profile or {}, 0))
    # A THIRD place the crop can come from, and the one that makes the common shape legal.
    # `merge_model_training_with_evaluation_profile` deep-copies the TRAINING profile, and an
    # evaluation profile REPLACES `env` wholesale rather than merging key-by-key. So an
    # evaluation profile that declares no `env` of its own inherits the training env -- crop
    # included -- and serves exactly the right domain.
    #
    # Reading only the two places above scores that as 0 and refuses it. Measured over the
    # last 120 TrainEvalWorkChains: 73 supply no `evaluation_profile` at all
    # (`ctx.evaluation_profile = {}`) and 0 supply one without an env, so this refused 73 of
    # 120 submissions the moment their model carried a crop -- and since the gas crop became
    # the default for all datasets, that is the direction every new training goes.
    #
    # It does NOT weaken the check that motivated it. Arm 72419's evaluation profile DID
    # declare an env (`PINO_LAZY_CACHE`, `PINO_LOADER_WORKERS`) with no crop in it, which is
    # precisely how the wholesale replacement dropped the training crop and served 40 cells
    # to a model trained on 30. An env that is present and silent about the crop still means
    # no crop, and is still refused.
    # PRESENCE of the key, not its truthiness. `validate_evaluation_profile({"env": {}})`
    # keeps the key, so an explicitly EMPTY env still replaces the training one wholesale and
    # still drops the crop -- that case must stay refused. `validate_evaluation_profile({})`
    # returns no `env` key at all, so the merge loop never touches it and the training env
    # survives intact. Verified against both functions rather than inferred.
    if "env" not in (evaluation_profile or {}):
        served = max(served, trained)
    if trained == served:
        return None
    return (
        f"[domain] the model was trained with crop_z_top={trained} but this evaluation "
        f"would serve crop_z_top={served}. That is a DOMAIN-EXTENT change, not a resolution "
        f"change: the model is scored on cells it never saw (or denied cells it did), which "
        f"returns a plausible number rather than an error -- whole-domain MAE moves ~9x while "
        f"substrate MAE barely moves. Drop the `derivation_profile` input so the evaluation "
        f"follows the model, or set its `crop_z_top` to {trained}. Pass "
        f"`allow_domain_change=Bool(True)` to score it anyway."
    )


def _override_list(value: str) -> list[str]:
    """`[a, b, 'c']` -> ['a', 'b', 'c']; Hydra's list-override spelling."""
    inner = value.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    return [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]


def requested_input_keys(training_profile: dict[str, Any], *, hydra_config: dict[str, Any] | None = None) -> frozenset[str]:
    """Every input channel the model declared, from its `custom.*_input_keys` lists.

    This is the set the datasource builds channels for (`"conduction_field" in
    requested_set`, npz_datasource.py:1247), so it is the one signal that says which derived
    fields the model actually consumed. An explicit `++custom.<name>_input_keys=[...]`
    override in the profile wins for THAT list (Hydra replaced it, so the archive's copy of
    the same list is what got overridden and must not resurrect a dropped key); lists the
    profile never overrode fall back to the resolved archived config. 207 of 590 stored
    models name `conduction_field` in an override; the packaged RIGNO_material and
    FNO_ck300_cond12 configs carry it by default, which is the fallback's job.
    """
    keys: set[str] = set()
    overridden: set[str] = set()
    for item in (training_profile or {}).get("hydra_overrides") or []:
        text = str(item).lstrip("+")
        if "=" not in text:
            continue
        name, _, value = text.partition("=")
        name = name.strip()
        if not (name.startswith("custom.") and name.endswith("_input_keys")):
            continue
        overridden.add(name[len("custom."):])
        keys.update(_override_list(value))
    custom = (hydra_config or {}).get("custom") if isinstance(hydra_config, dict) else None
    if isinstance(custom, dict):
        for name, value in custom.items():
            if name.endswith("_input_keys") and name not in overridden and isinstance(value, (list, tuple)):
                keys.update(str(v) for v in value)
    return frozenset(keys)


def with_requested_derived_fields(
    derivation_profile: dict[str, Any],
    training_profile: dict[str, Any],
    *,
    hydra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Union the buildable channels a model conditions on into ``derived_fields``.

    `build_derivation_profile_from_training_profile` already promotes these, but both
    training funnels call it only when NO derivation profile was passed; an explicit one is
    taken verbatim, and the documented submission workflow passes one. The channel
    then is not baked, so the loader rebuilds it per case into the shared per-host
    `~/.pino/conduction_cache`, which has no eviction (~40-80 GB every couple of days,
    hand-trimmed three times by 2026-09-10).

    UNION, never replace: crop, stride and frame are the submitter's to choose. Returns the
    profile unchanged when there is nothing to add, so it is free and idempotent.
    """
    # OPT-IN (2026-09-11). Applied unconditionally for one day, this was the accident it
    # exists beside: a submitter's identity profile was substituted to a derived collection
    # with the channel baked, which took 61 members off the pack-index path and onto a
    # full copy inside `do_upload` -- both local GPUs idle for two hours behind a
    # single-core export, so materialisation is made hard to do by accident. The loader builds these channels itself at read time; baking them
    # is a choice about disk and time, and `materialize: true` is the flag that already
    # says "bake it". Without it the promotion is a no-op and the read-time build stands.
    if not bool((derivation_profile or {}).get("materialize")):
        return derivation_profile
    requested = requested_input_keys(training_profile or {}, hydra_config=hydra_config)
    wanted = {name for name in buildable_derived_fields() if name in requested}
    present = set((derivation_profile or {}).get("derived_fields") or [])
    if not wanted or wanted <= present:
        return derivation_profile
    return validate_derivation_profile(
        {**derivation_profile, "derived_fields": sorted(present | wanted)}
    )


def build_derivation_profile_from_training_profile(training_profile: dict[str, Any], *, unrecorded_crop_z_top: int | None = None, hydra_config: dict[str, Any] | None = None, recorded_derived_fields: list[str] | None = None) -> dict[str, Any]:
    """Extract a derivation profile from a training runtime profile.

    ``hydra_config`` is the model's archived `.hydra/config.yaml` when the caller has it; it
    only matters for models whose input-key lists were never overridden in the profile.
    """
    # The conditioning channels the model consumed are part of "what the model saw" as
    # much as its layout is. Emitting none here sent every eval of a conduction-conditioned
    # model to a variant WITHOUT the channel baked, and the loader rebuilt it per case into
    # the shared on-disk cache -- six live evals refilling ~60 GB/day
    # (2026-09-04). Only buildable fields are promoted; trajectory channels are the loader's.
    # `derived_fields` follow the RECORD of what the model trained on, never its input
    # keys (changed 2026-09-11). Promoting from the input keys materialised an eval variant
    # for every conduction-conditioned model -- including the ~640 trained with the loader
    # building the channel at read time, which never saw a baked variant. A training bakes
    # only by opting in (`with_requested_derived_fields`, `materialize: true`), and that
    # decision is recorded on the model; an eval reproduces it. `hydra_config` and
    # `requested_input_keys` stay available to callers that ask the other question.
    derived_fields = sorted(str(name) for name in (recorded_derived_fields or []))
    env = training_profile.get("env") or {}
    lazy_stride = env.get("PINO_LAZY_STRIDE") if isinstance(env, dict) else None
    stride = [1, 1, 1]
    if lazy_stride:
        stride = list(normalize_stride([part.strip() for part in str(lazy_stride).split(",")]))
    return validate_derivation_profile(
        {
            "include_time": training_profile.get("include_time", True),
            "time_as_batch": training_profile.get("time_as_batch", False),
            "time_as_dim": training_profile.get("time_as_dim", False),
            # The crop a model ACTUALLY trained with, never the current default. This
            # function reconstructs what a stored model saw, and it did not mention
            # `crop_z_top` at all -- so when the default became 10, every one of the ~448
            # models trained on the full z extent would have been SCORED on 30 cells:
            # wrong numbers, no error. Resolution order is the recorded env var (where the
            # read-time crop rides), then a recorded field, then ZERO for anything that
            # predates the default. Same reasoning as the stride branch below.
            "crop_z_top": _recorded_crop_z_top(
                training_profile,
                DEFAULT_DERIVATION_PROFILE["crop_z_top"]
                if unrecorded_crop_z_top is None
                else unrecorded_crop_z_top,
            ),
            "moving_ref_frame": training_profile.get("moving_ref_frame", False),
            "moving_frame_steps": training_profile.get("moving_frame_steps", 1),
            "time_avg_window": training_profile.get("time_avg_window", 1),
            "super_res_factor": training_profile.get("super_res_factor", 1),
            "stride": stride,
            "derived_fields": derived_fields,
        }
    )


def apply_dataset_layout_to_training_profile(training_profile: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Overwrite layout-related runtime flags so they match the resolved dataset."""
    profile = copy.deepcopy(training_profile)
    layout = dataset_layout_profile_from_metadata(metadata)
    profile["include_time"] = layout["include_time"]
    profile["time_as_batch"] = layout["time_as_batch"]
    profile["time_as_dim"] = layout["time_as_dim"]
    profile["moving_ref_frame"] = layout["moving_ref_frame"]
    profile["moving_frame_steps"] = layout["moving_frame_steps"]
    profile["time_avg_window"] = layout["time_avg_window"]
    if layout["time_as_dim"]:
        profile["use_2d"] = True
    if layout["stride"] != [1, 1, 1]:
        env = dict(profile.get("env") or {})
        env.pop("PINO_LAZY_STRIDE", None)
        profile["env"] = env

    super_res_factor = int(metadata.get("super_res_factor", 1))
    overrides = list(profile.get("hydra_overrides", []))
    controlled_prefixes = (
        "custom.include_time=",
        "custom.time_as_batch=",
        "custom.time_as_dim=",
        "custom.moving_ref_frame=",
        "custom.moving_frame_steps=",
        "custom.time_avg_window=",
        "custom.super_res=",
    )
    if layout["time_as_dim"]:
        controlled_prefixes = controlled_prefixes + ("custom.use_2D=",)

    filtered_overrides = []
    for override in overrides:
        normalized = override.lstrip("+")
        if any(normalized.startswith(prefix) for prefix in controlled_prefixes):
            continue
        filtered_overrides.append(override)

    filtered_overrides.extend(
        [
            f"custom.include_time={'true' if layout['include_time'] else 'false'}",
            f"custom.time_as_batch={'true' if layout['time_as_batch'] else 'false'}",
            f"custom.time_as_dim={'true' if layout['time_as_dim'] else 'false'}",
            f"custom.moving_ref_frame={'true' if layout['moving_ref_frame'] else 'false'}",
            f"custom.moving_frame_steps={int(layout['moving_frame_steps'])}",
            f"custom.time_avg_window={int(layout['time_avg_window'])}",
            f"custom.super_res={super_res_factor}",
        ]
    )
    if layout["time_as_dim"]:
        filtered_overrides.append("custom.use_2D=true")
    profile["hydra_overrides"] = filtered_overrides
    return profile


def canonicalize_import_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Force VTK-import metadata into the canonical base-dataset layout."""
    normalized = dict(metadata)
    normalized["dataset_role"] = "base"
    normalized["canonical_layout"] = PinoDatasetData.CANONICAL_LAYOUT
    normalized["cache_basename"] = PinoDatasetData.DEFAULT_BASE_CACHE_BASENAME
    normalized["super_res_factor"] = 1
    normalized["time_axis"] = True
    normalized["avg_window_size"] = 1
    normalized["moving_frame_steps"] = 1
    normalized["moving_ref_frame"] = False
    # Derived from DEFAULT_DERIVATION_PROFILE, never written as a literal. The
    # 2026-07-22 canonical flip moved `canonical_layout` and the default profile
    # together but left three literal `time_as_dim: True` stamps behind, so
    # every dataset imported after it carried a time_axis_mode contradicting the
    # canonical_layout stamped two lines above (one base collection was
    # created five days after the flip). A literal here is a second source of
    # truth; there must not be one.
    normalized["time_axis_mode"] = {
        "include_time": bool(DEFAULT_DERIVATION_PROFILE["include_time"]),
        "time_as_batch": bool(DEFAULT_DERIVATION_PROFILE["time_as_batch"]),
        "time_as_dim": bool(DEFAULT_DERIVATION_PROFILE["time_as_dim"]),
    }
    normalized["family_identifier"] = build_family_identifier(
        normalized.get("source_folder_name", "dataset"),
        normalized.get("source_type", PinoDatasetData.DEFAULT_SOURCE_TYPE),
        grid_params=normalized.get("grid_params"),
        requested_fields=list(normalized.get("requested_fields", [])),
        source_uri=str(normalized.get("source_uri", "")),
    )
    normalized["derivation_profile"] = canonical_base_layout_profile()
    normalized["derivation_hash"] = ""
    normalized["reconstructable"] = False
    return normalized


def resolve_base_dataset(dataset: PinoDatasetCollectionData | PinoDatasetData) -> PinoDatasetCollectionData | PinoDatasetData:
    """Resolve the canonical base dataset for a dataset family."""
    metadata = dataset.get_dataset_metadata()
    role = metadata.get("dataset_role", "")
    if isinstance(dataset, PinoDatasetCollectionData):
        # composed/subset collections are self-contained member sets: they are
        # their own base for layout derivation. Without this, every
        # console-composed collection was born untrainable on non-canonical
        # layouts (train_model exited ERROR_INCOMPATIBLE_DATASET_VARIANT).
        if role in ("base", "composed", "subset"):
            return dataset
        if role == "derived":
            base_uuid = str(metadata.get("base_collection_uuid", "") or "")
            if base_uuid:
                node = orm.load_node(base_uuid)
                if isinstance(node, PinoDatasetCollectionData):
                    return node
        raise ValueError("Selected dataset collection is not a canonical base collection.")
    if role == "base":
        return dataset
    if role == "derived":
        base_uuid = str(metadata.get("base_dataset_uuid", "") or "")
        if base_uuid:
            node = orm.load_node(base_uuid)
            if isinstance(node, PinoDatasetData):
                return node
    if is_canonical_metadata(metadata):
        return dataset
    raise ValueError("Selected dataset is not a canonical base dataset and cannot be used to derive new variants.")


def find_reusable_derived_dataset(
    base_dataset: PinoDatasetCollectionData | PinoDatasetData,
    profile: dict[str, Any],
) -> PinoDatasetCollectionData | PinoDatasetData | None:
    """Return an existing derived dataset for the same base and derivation hash."""
    normalized = validate_derivation_profile(profile)
    # Reuse the base only when the base itself is stored in the requested layout.
    # Testing `normalized == canonical` assumed every base is stamped canonical,
    # which handed back a time_as_dim base for a time_as_batch request once the
    # canonical default moved to time_as_batch.
    resolved_base = resolve_base_dataset(base_dataset)
    if dataset_layout_profile_from_metadata(resolved_base.get_dataset_metadata()) == normalized:
        return resolved_base

    if isinstance(base_dataset, PinoDatasetCollectionData):
        base = resolve_base_dataset(base_dataset)
        if not isinstance(base, PinoDatasetCollectionData):
            raise ValueError("Expected a collection base dataset.")
        builder = orm.QueryBuilder()
        builder.append(
            PinoDatasetCollectionData,
            filters=reuse_filters_for(base.uuid, normalized),
            project=["*"],
        )
        builder.order_by({PinoDatasetCollectionData: {"ctime": "desc"}})
        results = [node for node in builder.limit(5).all(flat=True) if not bool(node.base.extras.get("superseded", False))]
        return results[0] if results else None

    builder = orm.QueryBuilder()
    builder.append(
        PinoDatasetData,
        filters={
            "extras.base_dataset_uuid": {"==": resolve_base_dataset(base_dataset).uuid},
            "extras.derivation_hash": {"==": derivation_hash(normalized)},
            "extras.dataset_role": {"==": "derived"},
        },
        project=["*"],
    )
    builder.order_by({PinoDatasetData: {"ctime": "desc"}})
    results = builder.limit(1).all(flat=True)
    return results[0] if results else None


def _strided_size(size: int, step: int) -> int:
    return len(range(0, int(size), int(step)))


def _derive_canonical_shape(shape: list[int] | tuple[int, ...], profile: dict[str, Any]) -> list[int]:
    """Return the canonical array shape after the derivation transforms."""
    normalized = validate_derivation_profile(profile)
    result = [int(dim) for dim in shape]
    if len(result) >= 3 and normalized["super_res_factor"] > 1:
        factor = int(normalized["super_res_factor"])
        result[-3:] = [int(dim) * factor for dim in result[-3:]]
    if normalized["moving_ref_frame"]:
        steps = int(normalized["moving_frame_steps"])
        factor = int(normalized["super_res_factor"])
        if len(result) == 5:
            _, channels, nx, ny, nz = result
            shifted_nx = nx - (steps - 1) * factor
            result = [steps, channels, shifted_nx, ny, nz]
            if normalized["time_avg_window"] > 1:
                result = [channels, shifted_nx, ny, nz]
        elif len(result) == 6:
            batch, _, channels, nx, ny, nz = result
            shifted_nx = nx - (steps - 1) * factor
            result = [batch, steps, channels, shifted_nx, ny, nz]
            if normalized["time_avg_window"] > 1:
                result = [batch, channels, shifted_nx, ny, nz]
    elif not normalized["include_time"]:
        if len(result) == 5:
            result = result[1:]
        elif len(result) == 6:
            result = [result[0], *result[2:]]
    if len(result) >= 4:
        sx, sy, sz = normalize_stride(normalized["stride"])
        result[-3:] = [
            _strided_size(result[-3], sx),
            _strided_size(result[-2], sy),
            _strided_size(result[-1], sz),
        ]
    return result


def _derive_time_count(base_count: int, profile: dict[str, Any], representative_shape: list[int]) -> int:
    normalized = validate_derivation_profile(profile)
    if normalized["moving_ref_frame"]:
        return 1 if normalized["time_avg_window"] > 1 else int(normalized["moving_frame_steps"])
    if not normalized["include_time"]:
        return 1
    if representative_shape and len(representative_shape) >= 5:
        return int(representative_shape[0])
    return int(base_count)


def build_derived_collection_metadata(
    base_collection: PinoDatasetCollectionData,
    profile: dict[str, Any],
    *,
    parent_collection: PinoDatasetCollectionData | None = None,
) -> dict[str, Any]:
    """Return metadata for a non-duplicating derived collection wrapper."""
    base = resolve_base_dataset(base_collection)
    if not isinstance(base, PinoDatasetCollectionData):
        raise ValueError("Expected a collection base dataset.")
    base_metadata = base.get_dataset_metadata()
    normalized = validate_derivation_profile(profile)
    metadata = copy.deepcopy(base_metadata)
    metadata["dataset_role"] = "derived"
    # A derived view SERVES the base's cases in another layout; it does not own
    # them. Deep-copying the base metadata carried `composition_kind="base"`
    # across, so every derived view became a phantom second owner without ever
    # passing `create_base_dataset`'s one-owner guard -- which is why
    # `base_dataset_of` returned a stride-2 view instead of the base for
    # 229 simulations. Both derived builders get this; they diverge
    # otherwise.
    if str(metadata.get("composition_kind") or "") == "base":
        metadata["composition_kind"] = "derived"
    metadata["base_collection_uuid"] = base.uuid
    metadata["base_collection_pk"] = base.pk
    metadata["parent_collection_uuid"] = parent_collection.uuid if parent_collection is not None else base_collection.uuid
    metadata["parent_collection_pk"] = parent_collection.pk if parent_collection is not None else base_collection.pk
    metadata["derivation_profile"] = normalized
    metadata["derivation_hash"] = derivation_hash(normalized)
    # Beside the hash, never inside it. See DERIVATION_CODE_VERSION.
    metadata["derivation_code_version"] = DERIVATION_CODE_VERSION
    metadata["reconstructable"] = True
    metadata["super_res_factor"] = normalized["super_res_factor"]
    metadata["time_axis"] = bool(normalized["include_time"])
    metadata["moving_ref_frame"] = normalized["moving_ref_frame"]
    metadata["moving_frame_steps"] = normalized["moving_frame_steps"]
    metadata["avg_window_size"] = normalized["time_avg_window"]
    # Written so `dataset_layout_profile_from_metadata` can read it back: without
    # the round-trip `find_reusable_derived_dataset` never matches an existing
    # strided variant and rebuilds it on every submit.
    metadata["stride"] = list(normalized["stride"])
    metadata["time_axis_mode"] = {
        "include_time": normalized["include_time"],
        "time_as_batch": normalized["time_as_batch"],
        "time_as_dim": normalized["time_as_dim"],
    }
    metadata["cache_basename"] = datasource_cache_basename_from_metadata(metadata)
    metadata["time_steps_min"] = base_metadata.get("time_steps_min", 0)
    metadata["time_steps_max"] = base_metadata.get("time_steps_max", 0)
    metadata["uniform_timesteps"] = bool(base_metadata.get("uniform_timesteps", False))
    members = base.iter_member_simulations()
    if members:
        member_shapes = members[0].get_array_shapes()
        metadata["array_shapes"] = {
            name: _derive_canonical_shape(shape, normalized) for name, shape in member_shapes.items()
        }
        first_shape = next(iter(metadata["array_shapes"].values()), [])
        metadata["spatial_shape"] = list(first_shape[-3:]) if len(first_shape) >= 3 else []
        metadata["grid_params"] = copy.deepcopy(base_metadata.get("grid_params", {}))
        if metadata["spatial_shape"]:
            metadata["grid_params"]["resolution"] = list(metadata["spatial_shape"])
        metadata["num_timesteps"] = _derive_time_count(members[0].get_num_timesteps(), normalized, first_shape)
    else:
        metadata["array_shapes"] = {}
        metadata["spatial_shape"] = []
        metadata["grid_params"] = copy.deepcopy(base_metadata.get("grid_params", {}))
        metadata["num_timesteps"] = 0
    metadata["dataset_identifier"] = (
        f"{metadata['family_identifier']}__{_variant_token(normalized)}__{metadata['derivation_hash'][:6]}"
    )
    return metadata


def native_source_incompatibility_reason(
    source: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData,
    profile: dict[str, Any],
) -> str | None:
    """Return a human-readable reason when a native source cannot satisfy a requested layout."""
    normalized = validate_derivation_profile(profile)

    if isinstance(source, PinoDatasetCollectionData):
        metadata = source.get_dataset_metadata()
        current_layout = dataset_layout_profile_from_metadata(metadata)
        if current_layout == normalized:
            if normalized["time_as_dim"] and not bool(metadata.get("uniform_timesteps", False)):
                return (
                    f"Dataset `{metadata.get('dataset_identifier', source.label or f'pk={source.pk}')}` mixes timestep counts "
                    "and cannot be used with time_as_dim."
                )
            return None
        try:
            base = resolve_base_dataset(source)
            if isinstance(base, PinoDatasetCollectionData):
                base_metadata = base.get_dataset_metadata()
                if normalized["time_as_dim"] and not bool(base_metadata.get("uniform_timesteps", False)):
                    return (
                        f"Dataset `{metadata.get('dataset_identifier', source.label or f'pk={source.pk}')}` mixes timestep counts "
                        "and cannot provide a time_as_dim variant."
                    )
                return None
        except Exception:
            pass
        identifier = str(metadata.get("dataset_identifier", "") or source.label or f"pk={source.pk}")
        role = str(metadata.get("dataset_role", "") or "legacy")
        return (
            f"Dataset `{identifier}` ({role}) cannot provide the requested native dataset variant from stored simulations. "
            "Choose a base collection or adjust the dataset variant."
        )

    if isinstance(source, PinoDatasetUnionData):
        for member in source.iter_member_datasets():
            reason = native_source_incompatibility_reason(member, normalized)
            if reason:
                return f"Union member pk={member.pk}: {reason}"
        return None

    metadata = source.get_dataset_metadata()
    try:
        current_layout = dataset_layout_profile_from_metadata(metadata)
    except Exception:
        current_layout = None

    if current_layout == normalized:
        return None

    try:
        resolve_base_dataset(source)
        return None
    except Exception:
        role = str(metadata.get("dataset_role", "") or "legacy")
        identifier = str(metadata.get("dataset_identifier", "") or source.label or f"pk={source.pk}")
        return (
            f"Dataset `{identifier}` ({role}) cannot provide the requested native dataset variant from stored arrays. "
            "Choose a base dataset or adjust the dataset variant."
        )


def _variant_token(profile: dict[str, Any]) -> str:
    """Build a human-readable variant token from a normalized derivation profile."""
    normalized = validate_derivation_profile(profile)
    spatial = f"var_{normalized['super_res_factor']}"
    if normalized["stride"] != [1, 1, 1]:
        sx, sy, sz = normalized["stride"]
        spatial = f"{spatial}_s{sx}x{sy}x{sz}"
    if not normalized["include_time"]:
        return spatial
    if normalized["time_as_dim"] or normalized["time_as_batch"] or not normalized["moving_ref_frame"]:
        return f"{spatial}_time_noref"
    return f"{spatial}_time_steps{normalized['moving_frame_steps']}_avg{normalized['time_avg_window']}"


def _grid_params_from_derived_arrays(base_grid_params: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    grid_params = copy.deepcopy(base_grid_params)
    if not arrays:
        return grid_params
    first_shape = next(iter(arrays.values())).shape
    if len(first_shape) >= 3:
        grid_params["resolution"] = [int(dim) for dim in first_shape[-3:]]
    axis_specs = {
        "x": ("x_range", (slice(None), 0, 0)),
        "y": ("y_range", (0, slice(None), 0)),
        "z": ("z_range", (0, 0, slice(None))),
    }
    for array_name, (range_name, spatial_selector) in axis_specs.items():
        array = arrays.get(array_name)
        if array is None or array.ndim < 3:
            continue
        prefix = (0,) * (array.ndim - 3)
        try:
            values = np.asarray(array[prefix + spatial_selector], dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if values.size:
            grid_params[range_name] = [float(values[0]), float(values[-1])]
    return grid_params


def _resample_axis_linear(array: np.ndarray, axis: int, new_size: int) -> np.ndarray:
    """Linearly interpolate along a single axis."""
    old_size = array.shape[axis]
    if old_size == new_size:
        return array.astype(np.float32, copy=False)

    old_coords = np.linspace(0.0, 1.0, old_size, dtype=np.float64)
    new_coords = np.linspace(0.0, 1.0, new_size, dtype=np.float64)

    moved = np.moveaxis(array, axis, -1)
    flat = moved.reshape(-1, old_size)
    out = np.empty((flat.shape[0], new_size), dtype=np.float32)
    for index, row in enumerate(flat):
        out[index] = np.interp(new_coords, old_coords, row.astype(np.float64)).astype(np.float32)
    out = out.reshape(moved.shape[:-1] + (new_size,))
    return np.moveaxis(out, -1, axis)


def _resample_spatial(array: np.ndarray, factor: int) -> np.ndarray:
    """Resample the last three spatial dimensions using trilinear interpolation."""
    if factor == 1:
        return array.astype(np.float32, copy=False)
    result = np.asarray(array, dtype=np.float32)
    spatial_axes = tuple(range(result.ndim - 3, result.ndim))
    for axis in spatial_axes:
        result = _resample_axis_linear(result, axis, result.shape[axis] * factor)
    return result


def _crop_shift_moving_frame(data: np.ndarray, steps: int, *, super_res_factor: int) -> np.ndarray:
    """Refuse: this is the unfixed twin of the beam-following crop.

    This was the (B, T, C, X, Y, Z) copy of the moving-frame crop, advancing the
    window exactly one cell per frame. That is a beam-following crop only where
    `V_scan * dt == dx`; measured on the spkh123 train split the beam advances
    0.051-0.482 cells per frame (median 0.200), so over a 30-frame window the
    beam drifts 4.2-5.4 beam radii out of the window meant to hold it still.

    The 2026-08-20 fix (`df90f7e`) taught the beam track to
    `data/pino_dataset_collection.py` and never reached here -- the copies had
    already been counted as "the third and fourth copy of the same window
    arithmetic" in that entry, and this is the fifth. Rather than carry a second
    implementation that silently returns smeared data, this one refuses: it has
    no internal callers, but `derive_dataset_arrays` is exported in
    `aiida_pino.utils.__all__`, so it is reachable.
    """
    raise NotImplementedError(
        "Moving-frame derivation through `derive_dataset_arrays` is withdrawn: this "
        "implementation advances the window one cell per frame, which follows the beam "
        "only when V_scan * dt == dx (measured 0.051-0.482 cells/frame on spkh123). Use "
        "the beam-track implementation in `data/pino_dataset_collection.py` "
        "(`_crop_shift_moving_frame_single`, via `_beam_following_slices_for`), which "
        "takes the laser path from the simulation's case manifest."
    )


def _moving_average_last_window(data: np.ndarray, window: int) -> np.ndarray:
    """Average over the time axis and keep the last averaged slice."""
    if data.ndim != 6:
        raise ValueError("Moving-average derivation requires 6D arrays shaped (B, T, C, X, Y, Z).")
    total_steps = data.shape[1]
    if not 1 <= window <= total_steps:
        raise ValueError(f"time_avg_window must be in [1, {total_steps}] for this dataset.")
    if window == 1:
        return data[:, -1]
    # Windowed average over time, then keep the final averaged slice to match current datasource behavior.
    windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=1)
    averaged = windows.mean(axis=-1, dtype=np.float64).astype(np.float32)
    return averaged[:, -1]


def _stride_canonical_spatial(array: np.ndarray, stride: Any) -> np.ndarray:
    sx, sy, sz = normalize_stride(stride)
    if (sx, sy, sz) == (1, 1, 1):
        return array.astype(np.float32, copy=False)
    if array.ndim < 4:
        return array.astype(np.float32, copy=False)
    return array[..., ::sx, ::sy, ::sz].astype(np.float32, copy=False)


def derive_dataset_arrays(base_dataset: PinoDatasetData, profile: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transform a canonical base dataset into a derived variant."""
    base = resolve_base_dataset(base_dataset)
    base_metadata = base.get_dataset_metadata()
    normalized = validate_derivation_profile(profile)

    arrays: dict[str, np.ndarray] = {}
    for name, array in base.iter_original_arrays():
        result = np.asarray(array, dtype=np.float32)
        if normalized["super_res_factor"] > 1:
            result = _resample_spatial(result, normalized["super_res_factor"])
        if normalized["moving_ref_frame"]:
            result = _crop_shift_moving_frame(
                result,
                normalized["moving_frame_steps"],
                super_res_factor=normalized["super_res_factor"],
            )
            if normalized["time_avg_window"] > 1:
                result = _moving_average_last_window(result, normalized["time_avg_window"])
        elif not normalized["include_time"] and result.ndim == 6:
            result = result[:, -1]
        result = _stride_canonical_spatial(result, normalized["stride"])
        arrays[name] = result.astype(np.float32, copy=False)

    metadata = copy.deepcopy(base_metadata)
    metadata["dataset_role"] = "derived"
    # A derived view SERVES the base's cases in another layout; it does not own
    # them. Deep-copying the base metadata carried `composition_kind="base"`
    # across, so every derived view became a phantom second owner without ever
    # passing `create_base_dataset`'s one-owner guard -- which is why
    # `base_dataset_of` returned a stride-2 view instead of the base for
    # 229 simulations. Both derived builders get this; they diverge
    # otherwise.
    if str(metadata.get("composition_kind") or "") == "base":
        metadata["composition_kind"] = "derived"
    metadata["canonical_layout"] = base_metadata.get("canonical_layout", PinoDatasetData.CANONICAL_LAYOUT)
    metadata["family_identifier"] = base_metadata.get("family_identifier") or build_family_identifier(
        base_metadata.get("source_folder_name", "dataset"),
        base_metadata.get("source_type", PinoDatasetData.DEFAULT_SOURCE_TYPE),
        grid_params=base_metadata.get("grid_params"),
        requested_fields=list(base_metadata.get("requested_fields", [])),
        source_uri=str(base_metadata.get("source_uri", "")),
    )
    metadata["base_dataset_uuid"] = base.uuid
    metadata["base_dataset_pk"] = base.pk
    metadata["parent_dataset_uuid"] = base_dataset.uuid
    metadata["parent_dataset_pk"] = base_dataset.pk
    metadata["derivation_profile"] = normalized
    metadata["derivation_hash"] = derivation_hash(normalized)
    # Beside the hash, never inside it. See DERIVATION_CODE_VERSION.
    metadata["derivation_code_version"] = DERIVATION_CODE_VERSION
    metadata["reconstructable"] = True
    metadata["super_res_factor"] = normalized["super_res_factor"]
    metadata["time_axis"] = bool(normalized["include_time"])
    metadata["moving_ref_frame"] = normalized["moving_ref_frame"]
    metadata["moving_frame_steps"] = normalized["moving_frame_steps"]
    metadata["avg_window_size"] = normalized["time_avg_window"]
    metadata["stride"] = list(normalized["stride"])
    metadata["time_axis_mode"] = {
        "include_time": normalized["include_time"],
        "time_as_batch": normalized["time_as_batch"],
        "time_as_dim": normalized["time_as_dim"],
    }
    metadata["cache_basename"] = datasource_cache_basename_from_metadata(metadata)
    grid_params = copy.deepcopy(base_metadata.get("grid_params", {}))
    resolution = list(grid_params.get("resolution", []))
    if len(resolution) == 3:
        resolution = [int(dim) * normalized["super_res_factor"] for dim in resolution]
        if normalized["moving_ref_frame"]:
            resolution[0] = resolution[0] - (normalized["moving_frame_steps"] - 1) * normalized["super_res_factor"]
        sx, sy, sz = normalize_stride(normalized["stride"])
        resolution = [
            len(range(0, int(resolution[0]), sx)),
            len(range(0, int(resolution[1]), sy)),
            len(range(0, int(resolution[2]), sz)),
        ]
        grid_params["resolution"] = resolution
        x_range = grid_params.get("x_range")
        if x_range and len(x_range) == 2 and resolution[0] > 1:
            x_min, x_max = float(x_range[0]), float(x_range[1])
            original_nx = int(base_metadata.get("grid_params", {}).get("resolution", [resolution[0]])[0])
            original_nx = max(original_nx, 2)
            dx = (x_max - x_min) / float(original_nx - 1)
            dx = dx / float(normalized["super_res_factor"])
            grid_params["x_range"] = [x_min, x_min + dx * float(resolution[0] - 1)]
    metadata["grid_params"] = grid_params
    metadata["array_shapes"] = {name: list(array.shape) for name, array in arrays.items()}
    metadata["grid_params"] = _grid_params_from_derived_arrays(metadata["grid_params"], arrays)
    metadata["dataset_identifier"] = (
        f"{metadata['family_identifier']}__{_variant_token(normalized)}__{metadata['derivation_hash'][:6]}"
    )
    return arrays, metadata


def build_derivation_summary(base_dataset: PinoDatasetData, derived_dataset: PinoDatasetData, profile: dict[str, Any], *, reused: bool) -> dict[str, Any]:
    """Return a queryable summary for dataset derivation."""
    normalized = validate_derivation_profile(profile)
    return {
        "base_dataset_pk": resolve_base_dataset(base_dataset).pk,
        "base_dataset_uuid": resolve_base_dataset(base_dataset).uuid,
        "derived_dataset_pk": derived_dataset.pk,
        "derived_dataset_uuid": derived_dataset.uuid,
        "family_identifier": derived_dataset.get_family_identifier(),
        "derivation_profile": normalized,
        "derivation_hash": derivation_hash(normalized),
        "reused_existing_dataset": reused,
    }
