"""The training-submission resolution sequence, in ONE place.

`TrainModelWorkChain.validate_inputs` + `resolve_sources` grew the only complete
statement of what happens between "here is a dataset and a profile" and "this is
the collection the job will actually read": profile validation, the split guard,
read-time demotion, layout compatibility, the undeclared-substitution refusal,
native-variant checks, and the as-is / base / reuse / materialize resolution.

A presubmit dry-run needs the same sequence without submitting, and reproducing
it in a second place is the duplicated-builder trap that hid the preheat bug for
a week (2026-08-10): the copies drift, and the copy you tested
is not the copy that runs. So the sequence lives here and BOTH callers delegate:

- the workchain calls with ``materialize=True`` and maps ``problems`` onto its
  exit codes;
- ``aiida-pino presubmit`` calls with ``materialize=False`` and prints the plan.

Every finding is a ``problem`` dict with a ``kind`` the workchain already has an
exit code for; ``notes`` carry the reports the workchain would have made.
"""

from __future__ import annotations

from typing import Any, Callable

from aiida_pino.utils.dataset_derivation import (
    build_derivation_profile_from_training_profile,
    with_requested_derived_fields,
    dataset_layout_profile_from_metadata,
    demote_for_code,
    find_reusable_derived_dataset,
    reconcile_crop_env,
    resolve_base_dataset,
    validate_derivation_profile,
)
from aiida_pino.utils.profiles import validate_training_profile

# Problem kinds, matched 1:1 by the workchain's exit-code map. A new kind here
# without a mapping there falls through to the workchain's generic failure,
# loudly -- never silently succeed.
INVALID_TRAINING_PROFILE = "invalid_training_profile"
INVALID_DERIVATION_PROFILE = "invalid_derivation_profile"
INVALID_DATASET_INPUT = "invalid_dataset_input"
SPLIT_VIOLATION = "split_violation"
INCOMPATIBLE_VARIANT = "incompatible_variant"
UNDECLARED_SUBSTITUTION = "undeclared_substitution"


def _problem(kind: str, message: str) -> dict[str, str]:
    return {"kind": kind, "message": message}


def _packaged_config(training_profile: dict[str, Any]) -> dict[str, Any] | None:
    """The named repo config, for models whose input lists were never overridden."""
    from aiida_pino.utils.profiles import _load_config_yaml

    name = (training_profile or {}).get("config_name")
    try:
        return _load_config_yaml(str(name)) if name else None
    except Exception:  # noqa: BLE001 - an unreadable config must not block a submission
        return None


def resolve_training_submission(
    train_dataset,
    val_dataset,
    training_profile: dict[str, Any],
    python_code,
    derivation_profile: dict[str, Any] | None = None,
    *,
    allow_split_violation: bool = False,
    materialize: bool = False,
    note: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the whole sequence; return profiles, problems, and a per-source plan.

    Stops at the FIRST problem, exactly as the workchain's step functions do --
    later stages consume earlier results, so continuing past a failure would
    report artifacts of the failure rather than further findings.

    With ``materialize=False`` nothing is written to the database: a source that
    would need a new derived collection comes back as a plan entry with
    ``action="materialize"`` and an estimated size instead of a node.
    """
    # Lazy: `layouts_compatible` lives in the workchain module (with its history), and
    # `normalize_dataset_collection_node` in data_nodes; importing them at module scope
    # would cycle through the workflow package.
    from aiida_pino.utils.data_nodes import normalize_dataset_collection_node
    from aiida_pino.workflows.train_model import layouts_compatible

    notes: list[str] = []

    def _note(text: str) -> None:
        notes.append(text)
        if note is not None:
            note(text)

    result: dict[str, Any] = {"problems": [], "notes": notes, "plans": {}}

    # --- datasets are datasets
    try:
        train = normalize_dataset_collection_node(train_dataset, label="train_dataset")
        val = normalize_dataset_collection_node(val_dataset, label="val_dataset")
    except ValueError as exc:
        result["problems"].append(_problem(INVALID_DATASET_INPUT, str(exc)))
        return result
    result["train_dataset"] = train
    result["val_dataset"] = val

    # --- the split guard
    from aiida_pino.workflows.train_model import (
        _member_uuids,
        _splits_by_uuid,
        split_violation_message,
    )

    train_uuids = _member_uuids(train)
    val_uuids = _member_uuids(val)
    reason = split_violation_message(
        train_uuids, val_uuids, _splits_by_uuid(set(train_uuids) | set(val_uuids))
    )
    if reason and allow_split_violation:
        _note(f"Split violation permitted by allow_split_violation: {reason}")
    elif reason:
        result["problems"].append(_problem(SPLIT_VIOLATION, f"Refusing to train: {reason}"))
        return result

    # --- profiles
    try:
        effective_training = validate_training_profile(
            dict(training_profile), expected_split_mode="standard"
        )
    except ValueError as exc:
        result["problems"].append(_problem(INVALID_TRAINING_PROFILE, str(exc)))
        return result
    if derivation_profile is not None:
        try:
            effective_derivation = validate_derivation_profile(dict(derivation_profile))
        except ValueError as exc:
            result["problems"].append(_problem(INVALID_DERIVATION_PROFILE, str(exc)))
            return result
    else:
        effective_derivation = build_derivation_profile_from_training_profile(effective_training)
    # Applies to BOTH branches: the built one already carries these, and the explicit one
    # would otherwise send a conditioned training to the loader's unevictable builder cache.
    effective_derivation = with_requested_derived_fields(
        effective_derivation, effective_training, hydra_config=_packaged_config(effective_training)
    )
    requested_layout = build_derivation_profile_from_training_profile(effective_training)

    # --- the gas crop is a property of the DATA, not of the mesh
    #
    # `crop_z_top` defaulted to a constant 10, read off the deck geometry (surface at z=0,
    # 95 um of headroom at dz=10 um). Material does not respect that line: the melt bulges
    # and the keyhole ejects into the headroom, and measured over `ti64_preheat60`'s
    # members only 5-9 of those 10 slices are actually metal-free. So the default has been
    # cutting metal on every case, silently, and the fix is to MEASURE rather than assume.
    #
    # An explicitly requested crop is honoured and warned about; a defaulted one is
    # replaced by the measured maximum. Warned, not refused: a refusing gate turns a
    # recoverable submission into dead work, and the message carries the count and the
    # recommendation so it can be acted on without re-deriving anything.
    from aiida_pino.utils.gas_crop import crop_advisory, survey_members

    def _requested_crop(training: dict, derivation: dict) -> tuple[int, bool]:
        """The crop this submission asks for, and whether a human actually asked for it."""
        env_value = ((training or {}).get("env") or {}).get("PINO_LAZY_CROP_Z_TOP")
        if env_value not in (None, ""):
            try:
                return int(env_value), True
            except (TypeError, ValueError):
                pass
        if derivation_profile is not None and "crop_z_top" in (derivation_profile or {}):
            return int((derivation or {}).get("crop_z_top") or 0), True
        return int((derivation or {}).get("crop_z_top") or 0), False

    crop, explicit = _requested_crop(effective_training, effective_derivation)
    if crop:
        try:
            survey = survey_members(train.iter_member_simulations())
        except Exception:  # noqa: BLE001 - an advisory must never block a submission
            survey = {"per_case": {}, "unmeasured": [], "safe_crop": None, "measured": 0}
        result["crop_survey"] = survey
        advisory = crop_advisory(
            survey, crop, baked=bool((effective_derivation or {}).get("materialize"))
        )
        if advisory:
            _note(advisory)
        safe = survey.get("safe_crop")
        if safe is not None and not explicit and crop > int(safe):
            # Nobody asked for this number; it came from the package default. Use the
            # measured one instead of quietly discarding material.
            effective_training = dict(effective_training)
            env = dict(effective_training.get("env") or {})
            env["PINO_LAZY_CROP_Z_TOP"] = str(int(safe))
            effective_training["env"] = env
            if "crop_z_top" in effective_derivation:
                effective_derivation = {**effective_derivation, "crop_z_top": int(safe)}
            result["training_profile"] = effective_training
            result["derivation_profile"] = effective_derivation
            _note(f"defaulted crop_z_top lowered {crop} -> {int(safe)}, the deepest crop "
                  f"that touches no metal in this dataset")

    # --- read-time demotion, so the collection stages pack-indexed not copied
    _profile, _env, _demote_notes = demote_for_code(
        effective_derivation, effective_training, python_code
    )
    if _demote_notes:
        effective_derivation = _profile
        effective_training = dict(effective_training)
        effective_training["env"] = _env
        _note(
            f"staging zero-copy: {', '.join(_demote_notes)} moved to the loader, "
            "so the dataset stages pack-indexed instead of being copied"
        )
    result["training_profile"] = effective_training
    result["derivation_profile"] = effective_derivation

    env_stride = str(
        ((effective_training or {}).get("env") or {}).get("PINO_LAZY_STRIDE", "") or ""
    ).strip()
    if not layouts_compatible(requested_layout, effective_derivation, env_stride=env_stride):
        result["problems"].append(_problem(
            INCOMPATIBLE_VARIANT,
            "Training profile layout does not match the requested dataset variant. "
            "Refresh the UI and choose a compatible base dataset or variant.",
        ))
        return result

    # --- no silent layout substitution when the layout came from the packaged config
    if derivation_profile is None:
        selected = dataset_layout_profile_from_metadata(train.get_dataset_metadata())
        differing = sorted(
            key for key in set(requested_layout) | set(selected)
            # `stride` from `env.PINO_LAZY_STRIDE` and the read-time crop are loader
            # settings applied on top of the dataset, not demands for a transformed one.
            if key != "stride" or not env_stride
            if key != "crop_z_top"
            if requested_layout.get(key) != selected.get(key)
        )
        if differing:
            detail = ", ".join(
                f"{key}: config={requested_layout.get(key)!r} dataset={selected.get(key)!r}"
                for key in differing
            )
            result["problems"].append(_problem(
                UNDECLARED_SUBSTITUTION,
                f"Refusing an undeclared layout substitution. The training config "
                f"`{effective_training.get('config_name', '?')}` implies a layout that "
                f"disagrees with the selected dataset {train.pk} ({train.label!r}) on: "
                f"{detail}. Training would have been run on a DIFFERENT dataset built to "
                f"match the config. Either select a dataset whose layout matches, or pass "
                f"`derivation_profile` explicitly to state that the substitution is intended.",
            ))
            return result

    # --- the stored form must be able to serve the variant at all
    from aiida_pino.utils.dataset_derivation import native_source_incompatibility_reason

    for label, source in (("training", train), ("validation", val)):
        why = native_source_incompatibility_reason(source, effective_derivation)
        if why:
            result["problems"].append(_problem(
                INCOMPATIBLE_VARIANT, f"{label.capitalize()} dataset incompatibility: {why}"
            ))
            return result

    # --- resolution: as-is / base / reuse / materialize
    def _plan(source) -> dict[str, Any]:
        metadata = source.get_dataset_metadata()
        if dataset_layout_profile_from_metadata(metadata) == effective_derivation:
            return {"action": "as-is", "resolved": source}
        base = resolve_base_dataset(source)
        if dataset_layout_profile_from_metadata(base.get_dataset_metadata()) == effective_derivation:
            return {"action": "base", "resolved": base, "asked": source}
        existing = find_reusable_derived_dataset(base, effective_derivation)
        if existing is not None:
            return {"action": "reuse-derived", "resolved": existing, "asked": source}
        # Nothing stored serves this layout: a new derived collection is needed.
        from aiida_pino.utils.runtime_cache import estimated_derived_bytes

        try:
            members = list(base.iter_member_simulations())
            estimated = estimated_derived_bytes(members, effective_derivation)
        except Exception:  # noqa: BLE001 - the estimate is advisory
            estimated = None
        if not materialize:
            return {"action": "materialize", "resolved": None, "asked": source,
                    "base": base, "estimated_bytes": estimated}
        from aiida import orm

        from aiida_pino.utils.dataset_derivation import report_materialisation_cost
        from aiida_pino.workflows.derive_dataset import materialize_derived_dataset

        report_materialisation_cost(_note, base, effective_derivation)
        return {"action": "materialize", "asked": source, "estimated_bytes": estimated,
                "resolved": materialize_derived_dataset(base, orm.Dict(dict=effective_derivation))}

    try:
        train_plan = _plan(train)
        val_plan = _plan(val)
    except ValueError as exc:
        result["problems"].append(_problem(INCOMPATIBLE_VARIANT, str(exc)))
        return result
    for name, plan in (("train", train_plan), ("val", val_plan)):
        result["plans"][name] = plan
        asked = plan.get("asked")
        if asked is not None and plan.get("resolved") is not None:
            _note(
                f"Dataset substitution ({name}): asked for {asked.pk} ({asked.label!r}), "
                f"serving {plan['resolved'].pk} ({plan['resolved'].label!r}) to match the "
                f"requested layout."
            )

    # --- the crop lives in ONE place, decided by what the dataset resolved to
    resolved_train = train_plan.get("resolved")
    if resolved_train is not None:
        _env2, _crop_note = reconcile_crop_env(
            (effective_training or {}).get("env"), resolved_train.get_dataset_metadata()
        )
        if _crop_note:
            effective_training = dict(effective_training)
            effective_training["env"] = _env2
            result["training_profile"] = effective_training
            _note(_crop_note)

    return result


def local_build_report(resolved, *, runtime_fields=None) -> dict[str, Any]:
    """Whether a LOCAL job on `resolved` will export it inside `do_upload` first.

    "local computer: jobs read the runtime cache directly" was true only when the cache
    existed. When it does not, the export -- one member at a time on the submitting machine,
    minutes a member for a derived variant -- runs inside the upload task, and the job
    reads "Waiting for transport task: upload" with both GPUs idle until it is done. On
    2026-09-11 three fine-tunes sat behind a 61-member export for two hours and nothing at
    submit time had said so.
    """
    from aiida_pino.data.pino_dataset_collection import derivation_changes_payload, encoding_changes
    from aiida_pino.utils.runtime_cache import (
        cache_discriminator_for,
        estimated_derived_bytes,
        runtime_cache_name,
        runtime_cache_root,
        staged_cache_is_current,
    )

    metadata = resolved.get_dataset_metadata() or {}
    profile = dict(metadata.get("derivation_profile") or {})
    copies = metadata.get("dataset_role") == "derived" and (
        derivation_changes_payload(profile) or encoding_changes(profile)
    )
    if not copies:
        return {"needed": False, "detail": "zero-copy: served through a pack index, nothing is exported"}
    discriminator = cache_discriminator_for(resolved, runtime_fields=runtime_fields, transport_type="core.local")
    cache_dir = runtime_cache_root() / runtime_cache_name(resolved.uuid, discriminator or None)
    if staged_cache_is_current(cache_dir)[0]:
        return {"needed": False, "detail": f"runtime cache for ds{resolved.pk} is built and current"}
    from aiida_pino.utils.export_pool import per_member_export_peak_bytes

    try:
        members = list(resolved.iter_member_simulations())
        estimated = estimated_derived_bytes(members, profile)
    except Exception:  # noqa: BLE001 - the estimate is advisory
        members, estimated = [], None
    try:
        peak = per_member_export_peak_bytes(members, profile)
    except Exception:  # noqa: BLE001
        peak = 0
    size = f"~{estimated / 1e9:.1f} GB" if estimated else "size unknown"
    peak_note = f", expected peak RAM ~{peak / 1e9:.0f} GB per export worker" if peak else ""
    return {
        "needed": True,
        "staged": False,
        "members": len(members),
        "estimated_bytes": estimated,
        "peak_ram_bytes": peak,
        "detail": (
            f"runtime cache for ds{resolved.pk} is NOT built: submitting will export "
            f"{len(members)} members ({size}{peak_note}) on this machine INSIDE the job's upload step, "
            f"and the job cannot reach the GPU until that finishes. Baked derived_fields "
            f"{profile.get('derived_fields') or []} are the usual cause; the loader builds "
            f"path channels at read time, so drop them from the derivation unless you meant "
            f"`materialize: true`, or `aiida-pino warm {resolved.pk}` first."
        ),
    }


def staging_report(plan: dict[str, Any], python_code) -> dict[str, Any]:
    """Whether the RESOLVED collection is already staged on the code's computer.

    Presubmit-only. A remote job does not fetch: the collection it reads must have
    been staged beforehand (`aiida-pino stage-remote`), and submitting without it
    fails after the queue wait instead of before the submit. Reports, never
    guesses: with the collection unresolved (a dry-run that would materialize),
    staging cannot be checked and says so instead of answering.
    """
    computer = getattr(python_code, "computer", None)
    transport_type = getattr(computer, "transport_type", "core.local") if computer else "core.local"
    if transport_type == "core.local":
        resolved = plan.get("resolved")
        if resolved is None:
            return {"needed": True, "staged": None,
                    "detail": "collection not resolved yet (would be materialized), and then "
                              "exported inside this job's upload step before it can reach the GPU"}
        return local_build_report(plan.get("resolved"))
    resolved = plan.get("resolved")
    if resolved is None:
        return {"needed": True, "staged": None,
                "detail": "collection not resolved yet (would be materialized); stage it after"}
    from aiida_pino.utils.stage_remote import staged_caches_on

    rows = staged_caches_on(computer)
    matching = [r for r in rows if str(r.get("cache_name", "")).startswith(str(resolved.uuid))]
    return {
        "needed": True,
        "staged": bool(matching),
        "matches": [r.get("cache_name") for r in matching],
        "detail": (
            f"{len(matching)} staged cache(s) on {computer.label} for uuid {resolved.uuid}"
            if matching else
            f"NOT staged on {computer.label}: run `aiida-pino stage-remote {resolved.pk} "
            f"--computer {computer.label}` before submitting"
        ),
    }
