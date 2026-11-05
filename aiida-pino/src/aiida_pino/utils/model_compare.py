"""Compare two models the way an ablation referee would.

"A beat B" is only evidence about the thing that was CHANGED if nothing else
differs. This repo has already paid for each of the confounds this module
checks: seed swings peak-T by ~1700 K across replicates; 31 of 158 models
trained fewer steps than their record claims (worst 4000 of a declared 48000);
18 models trained at 2x2x1 while their record said full resolution, one of them
holding a best-T-MAE badge; and 4 evaluations scored their own training data.
Every one of those, found late, invalidates an ablation conclusion silently.

`compare_models(a, b)` returns every difference between two models, classified:

* ``confound``  -- changes the learning problem or the measurement (data, seed,
                   steps actually trained, resolution, lr, batch, amp, arch and
                   objective overrides, scheduler/config).
* ``caution``   -- can bias a comparison indirectly (validation subset size,
                   inference batching, unrecognised keys -- unknown is not safe).
* ``benign``    -- bookkeeping that cannot reach the math (names, GPU booking,
                   wandb, cuda device index, timestamps).

The verdict is deliberately blunt: an ablation is CLEAN only when the models
share their training data member-for-member and every non-benign difference is
the one under study. Unknown never reads as same: a fact absent on either side
is reported as a difference of kind "unknown", not silently skipped.
"""

from __future__ import annotations

from typing import Any

from aiida import orm

#: Profile/override keys that cannot reach the math. Everything NOT listed here
#: or in _CAUTION is treated as a confound -- unknown keys must fail loud, not
#: slip through as benign.
_BENIGN = {
    "run_name", "model_name", "cuda_device", "wandb_offline", "config_path",
    "gpu_vram_mb", "host_ram_mb", "custom.name", "custom.wandb_offline",
    "custom.val_data_dir", "custom.data_dir", "network_dir", "seat_priority",
    # The real profile key (adversarial pass: "seat_priority" alone made GPU
    # queue booking read as a confounded factor in every cross-era comparison).
    "gpu_seat_priority",
}
_CAUTION = {
    "batch_size_validation", "inference_batch_size", "custom.val_max_cases",
    "custom.validator_dumps", "custom.inference_batch_size",
}


def _parse_overrides(overrides: list | None) -> dict[str, str]:
    """Hydra overrides as {key: last value} -- last wins, prefixes stripped.

    A bare `~key` (hydra's deletion syntax, no `=`) is recorded as
    `<deleted>`: dropping it made a deleted config key read as SAME, which is
    the one direction this module promises never to fail (adversarial pass).
    """
    out: dict[str, str] = {}
    for item in overrides or []:
        text = str(item)
        key, sep, value = text.partition("=")
        if sep:
            out[key.strip().lstrip("+~")] = value.strip()
        elif text.strip().startswith("~"):
            out[text.strip().lstrip("~")] = "<deleted>"
    return out


def _member_uuids(summary: dict | None) -> set[str]:
    out = set()
    for member in (summary or {}).get("members") or []:
        if isinstance(member, dict) and member.get("uuid"):
            out.add(member["uuid"])
        elif isinstance(member, str):
            out.add(member)
    return out


def _flat_config(config: dict | None, prefix: str = "") -> dict[str, Any]:
    """A resolved hydra config as dotted scalar keys, for content-level diffing."""
    out: dict[str, Any] = {}
    for key, value in (config or {}).items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat_config(value, f"{dotted}."))
        else:
            out[dotted] = value
    return out


def _facts(model) -> dict[str, Any]:
    """Everything about one model that a comparison can act on."""
    from aiida_pino.utils.profiles import effective_profile, override_conflicts

    metadata = model.get_model_metadata()
    stored_profile = dict(metadata.get("training_profile") or {})
    # Compare what Hydra ACTUALLY applied: a structured field its own override
    # overrules is dead but still stored, so comparing the stored value
    # compares a number that never trained anything (7 models record
    # weight_decay 0.0 while training with 0.01-0.1).
    profile = effective_profile(stored_profile)
    shadowed = override_conflicts(stored_profile)
    extras = model.base.extras.all
    train_summary = metadata.get("training_source_summary") or {}
    val_summary = metadata.get("validation_source_summary") or {}
    return {
        "pk": model.pk,
        "uuid": model.uuid,
        "name": extras.get("model_name") or model.label,
        "ctime": model.ctime.isoformat() if model.ctime else None,
        "config_name": metadata.get("config_name") or profile.get("config_name"),
        "keywords": sorted(extras.get("keywords") or []),
        "profile": profile,
        "overrides": _parse_overrides(profile.get("hydra_overrides")),
        "env": dict(profile.get("env") or {}),
        "has_calcjob": bool(extras.get("calcjob_pk")),
        "seed": extras.get("training_seed"),
        "seed_source": extras.get("training_seed_source"),
        "stride": extras.get("training_stride"),
        "max_steps": extras.get("max_steps"),
        "steps_trained": extras.get("steps_trained"),
        "wall_seconds": extras.get("wall_seconds"),
        "gpu": extras.get("env_gpu_name"),
        "code_version": extras.get("code_version_inferred"),
        "train_uuid": metadata.get("training_source_uuid"),
        "train_label": metadata.get("training_source_label") or train_summary.get("label"),
        "train_members": _member_uuids(train_summary),
        "val_uuid": val_summary.get("uuid") or extras.get("val_dataset_uuid"),
        "backfilled": set(extras.get("backfilled_keys") or []),
        "shadowed_fields": sorted(shadowed),
        "config_flat": _archived_config_flat(model),
    }


def _archived_config_flat(model) -> dict[str, Any] | None:
    """The RESOLVED config the run archived, flattened. None when unarchived.

    `config_name` is a pointer, not content: the same name can resolve to
    different yaml on different days (commit c0cc754 flipped
    config_flow3D_RIGNO_material's amp.enabled with the name unchanged), and two
    different names can hide what actually differs. The archived
    `.hydra/config.yaml` is what the run really saw -- 205 of 223 models carry
    one -- so the comparison diffs content and treats the name as a label.
    (Friction report 2026-08-13: in-config differences collapsed into one
    `config_name` confound, and same-name drift was invisible.)
    """
    try:
        from aiida_pino.console.model_metrics import model_hydra_config

        config = model_hydra_config(model)
    except Exception:  # noqa: BLE001 - repo unreadable: unarchived, not equal
        return None
    if not isinstance(config, dict) or not config:
        return None
    return _flat_config(config)


#: One experimental FACTOR often surfaces as several fields: the seed appears as
#: the stamped extra AND as `override.seed`; training length as `max_steps`,
#: `profile.max_steps` and `steps_trained`. Counting fields made two seed
#: replicates read "CONFOUNDED: 2 confounds" for one changed knob -- measured on
#: the first live trial (alsi_fno_r01_m1 vs m2). Verdicts count factors.
_FACTOR_OF = {
    "seed": "seed", "override.seed": "seed", "override.custom.seed": "seed",
    "override.training.seed": "seed", "profile.seed": "seed",
    "max_steps": "training length", "profile.max_steps": "training length",
    "steps_trained": "training length", "override.training.max_steps": "training length",
    "stride": "resolution", "env.PINO_LAZY_STRIDE": "resolution",
    "profile.stride": "resolution", "override.custom.stride": "resolution",
    "config_name": "config", "profile.config_name": "config",
    "config_content": "config content",
    "config.custom.seed": "seed", "config.seed": "seed",
    "config.training.max_steps": "training length",
    "code_version": "training code",
    "training_data": "training data", "validation_data": "validation data",
}


def _factor(field: str) -> str:
    """The experimental factor a field belongs to.

    Beyond the explicit map, a leading `override.`/`config.`/`profile.` is stripped so the
    SAME knob reported from two sources counts once. `override.custom.include_time` and
    `config.custom.include_time` are one decision seen at its input and at its output, and
    counting them separately doubled the factor count of every pair that stated a knob
    explicitly on one side and inherited it on the other.
    """
    if field in _FACTOR_OF:
        return _FACTOR_OF[field]
    for prefix in ("override.", "config.", "profile."):
        if field.startswith(prefix):
            return field[len(prefix):]
    return field


def _severity(key: str) -> str:
    if key in _BENIGN:
        return "benign"
    if key in _CAUTION:
        return "caution"
    return "confound"


def _list_delta(va, vb) -> dict[str, list[str]] | None:
    """Token-level added/removed for list-shaped strings like [a,b,c].

    A 22-item channel list differing from a 12-item one by a single entry had
    to be eyeball-diffed in every renderer (adversarial pass); the delta makes
    the one changed token the thing displayed.
    """
    def tokens(value):
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [t.strip() for t in text.split(",") if t.strip()]
        return parts if len(parts) > 1 else None

    ta, tb = tokens(va), tokens(vb)
    if ta is None or tb is None:
        return None
    added = [t for t in tb if t not in ta]
    removed = [t for t in ta if t not in tb]
    if not added and not removed:
        return None
    return {"added": added, "removed": removed}


def _env_unrecorded(facts: dict[str, Any]) -> bool:
    """True when a model never RECORDED an env, as distinct from having an empty one.

    `import-model` used to store whatever profile the caller typed, and a caller who
    passed none got `env: {}`. Absent is not empty: it is no evidence about the run's
    environment, so diffing it against a native model's env charged every key as a
    confound -- 41 false confounds, found 2026-08-27.

    The discriminator is provenance, not the value. A model with a calcjob recorded its
    own env, so an empty one there is a real measurement and a real difference; a model
    with no calcjob came in through the import route, where the field was optional.
    (`submit_import_model` now reads `train_summary.json`, so models imported from here
    on record the effective env and never take this path.)
    """
    return not facts["env"] and not facts["has_calcjob"]


#: Sentinel for "this key is not in the resolved config", distinct from a stored None.
_ABSENT = object()


def _diff_maps(kind: str, a: dict, b: dict, skip: set[str],
               severity: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(a) | set(b)):
        if key in skip:
            continue
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if va != vb:
            row = {"field": f"{kind}.{key}", "a": va, "b": vb,
                   "severity": severity or _severity(key)}
            delta = _list_delta(va, vb)
            if delta:
                row["delta"] = delta
            rows.append(row)
    return rows


def _eval_rows(model_uuid: str) -> dict[str, dict[str, Any]]:
    """This model's evaluations, keyed by scored-dataset uuid."""
    rows = orm.QueryBuilder().append(
        orm.Data,
        filters={"node_type": {"like": "%pino_model_evaluation%"},
                 "extras.model_uuid": model_uuid},
        project=["id", "extras.dataset_source_uuid", "extras.held_out",
                 "extras.eval_stride",
                 "attributes.evaluation_metadata.metrics.T.abs_err_mean",
                 "attributes.evaluation_metadata.dataset_source_label"],
    ).all()
    out: dict[str, dict[str, Any]] = {}
    for pk, ds_uuid, held, stride, mae, label in rows:
        if not ds_uuid:
            continue
        entry = {"eval_pk": pk, "held_out": bool(held) if held is not None else None,
                 "eval_stride": stride, "t_abs_err_mean": mae, "dataset_label": label}
        # keep the best-documented eval per dataset: held-out beats unknown
        current = out.get(ds_uuid)
        if current is None or (entry["held_out"] and not current["held_out"]):
            out[ds_uuid] = entry
    return out


def compare_models(a, b) -> dict[str, Any]:
    """The full referee's report for models `a` and `b` (nodes or pks)."""
    node_a = orm.load_node(a) if isinstance(a, int) else a
    node_b = orm.load_node(b) if isinstance(b, int) else b
    fa, fb = _facts(node_a), _facts(node_b)

    differences: list[dict[str, Any]] = []

    # --- scalar facts, each with its own severity story
    scalar_fields = (
        ("config_name", "confound"), ("seed", "confound"), ("stride", "confound"),
        ("max_steps", "confound"), ("steps_trained", "confound"),
        ("code_version", "confound"),
        ("gpu", "benign"), ("name", "benign"), ("ctime", "benign"),
    )
    # Fields where absence is an unknown, not an equality. The adversarial pass
    # found the blocker here: when a fact was missing on BOTH sides, va == vb
    # short-circuited the row entirely, so two models with no stride stamp read
    # "valid ablation" while resolution was unknowable on either -- the exact
    # contract ("unknown never reads as same") the docstring promises. 127 of
    # 243 models lack training_stride and 24 lack steps_trained, so
    # double-absence is the COMMON case, not a corner.
    unknown_eligible = {"seed", "stride", "steps_trained", "code_version"}
    for field, severity in scalar_fields:
        va, vb = fa[field], fb[field]
        if va is None and vb is None and field in unknown_eligible:
            differences.append({"field": field, "a": "<unrecorded>", "b": "<unrecorded>",
                                "severity": "unknown"})
            continue
        if va != vb:
            kind = severity
            if field in unknown_eligible and (va is None or vb is None):
                kind = "unknown"
            differences.append({"field": field, "a": va, "b": vb, "severity": kind})

    # --- structured profile, overrides and env
    skip = {"hydra_overrides", "env", "run_name", "config_name"}
    differences += _diff_maps("profile", fa["profile"], fb["profile"], skip)
    # Overrides are the INPUT to Hydra; `.hydra/config.yaml` is what it materialized. When
    # both sides archived one, the materialized config decides, and an override difference
    # that resolves to the same value is a spelling difference, not an experimental one.
    #
    # This is why exact repeats read CONFOUNDED (2026-08-27). Stating a knob
    # explicitly on one side and inheriting it from the config on the other produces
    # different override LISTS and an identical model. Measured over 14 same-config
    # same-dataset pairs: 39 override rows scored `confound`, and the resolved configs
    # AGREE on 25 of them.
    both_archived = fa["config_flat"] is not None and fb["config_flat"] is not None
    override_rows = _diff_maps("override", fa["overrides"], fb["overrides"], set())
    if both_archived:
        spelled_differently = []
        for row in override_rows:
            key = row["field"][len("override."):]
            va = fa["config_flat"].get(key, _ABSENT)
            vb = fb["config_flat"].get(key, _ABSENT)
            if va is _ABSENT or vb is _ABSENT:
                # The archived config does not record this key, so it cannot testify that
                # the two resolved the same. Absence is unknown, never equality -- the same
                # contract the scalar fields keep for double-absence. Demoting here would
                # HIDE a real difference: `custom.keys=[x,y,z]` against
                # `custom.keys=[x,y,z,beam_s]` is a changed channel list, and a config that
                # never mentions `custom.keys` says nothing about it either way.
                differences.append(row)
                continue
            if va == vb:
                spelled_differently.append(key)
            # A row whose resolved values DIFFER is dropped rather than kept: the config
            # diff below no longer skips override keys, so it reports that difference
            # against the value Hydra actually used instead of against the flag text.
        if spelled_differently:
            differences.append({
                "field": "override_spelling",
                "a": ", ".join(sorted(spelled_differently)) or "-",
                "b": "same resolved value",
                "severity": "benign",
            })
    else:
        differences += override_rows
    # One side that never recorded an env cannot testify that the envs differed; the
    # module's own "unknown" severity says exactly that, and the verdict already reads
    # it as "cannot be verified equal" rather than as a measured difference.
    env_unknown = _env_unrecorded(fa) or _env_unrecorded(fb)
    differences += _diff_maps("env", fa["env"], fb["env"], set(),
                              severity="unknown" if env_unknown else None)

    # --- the RESOLVED config content. Catches same-name yaml drift and shows
    # what actually differs between different names, instead of collapsing
    # everything into one `config_name` row. Keys already reported as override
    # rows are skipped: the resolved config contains applied overrides, and the
    # override row already tells that story.
    if both_archived:
        # No longer skipping override keys. The whole point is to compare what Hydra
        # materialized, and the archived config already contains the applied overrides --
        # skipping them left the flag text as the only witness for exactly the keys the run
        # cared about most.
        differences += _diff_maps("config", fa["config_flat"], fb["config_flat"], set())
    else:
        missing = [side for side, f in (("a", fa), ("b", fb)) if f["config_flat"] is None]
        differences.append({
            "field": "config_content",
            "a": "<unarchived>" if fa["config_flat"] is None else "archived",
            "b": "<unarchived>" if fb["config_flat"] is None else "archived",
            "severity": "unknown",
        })
        del missing

    # --- the data itself, member-for-member and TWIN-COLLAPSED. This database
    # stores payload twins (same simulation re-imported under a new uuid, same
    # content_hash), and the eval path already collapses them when marking
    # held_out. Raw uuids made compare_models(195, 10237) report "DIFFERENT
    # DATA: 0 shared of 10 / 10" for two byte-identical training sets -- 1232
    # model pairs disagree between raw and collapsed overlap (adversarial pass).
    from aiida_pino.utils.model_registry import payload_keys

    if fa["train_uuid"] == fb["train_uuid"] and fa["train_uuid"]:
        data_relation = "same dataset node"
    elif fa["train_members"] and fb["train_members"]:
        keys_a, keys_b = payload_keys(fa["train_members"]), payload_keys(fb["train_members"])
        if keys_a == keys_b:
            data_relation = ("different nodes, identical members"
                             if fa["train_members"] == fb["train_members"]
                             else "different nodes, identical PAYLOADS (twin-collapsed)")
        else:
            shared = len(keys_a & keys_b)
            raw_shared = len(fa["train_members"] & fb["train_members"])
            data_relation = (f"DIFFERENT DATA: {shared} shared of "
                             f"{len(keys_a)} (a) / {len(keys_b)} (b)")
            if shared != raw_shared:
                data_relation += f" [twin-collapsed; raw-uuid overlap would say {raw_shared}]"
            differences.append({"field": "training_data", "a": fa["train_label"],
                                "b": fb["train_label"], "severity": "confound"})
    else:
        data_relation = "unknown: a member list is missing"
        differences.append({"field": "training_data", "a": fa["train_label"],
                            "b": fb["train_label"], "severity": "unknown"})
    if fa["val_uuid"] != fb["val_uuid"]:
        differences.append({"field": "validation_data", "a": fa["val_uuid"],
                            "b": fb["val_uuid"], "severity": "caution"})

    # --- shared evaluations: where a comparison is actually legitimate
    evals_a, evals_b = _eval_rows(fa["uuid"]), _eval_rows(fb["uuid"])
    shared_evals = []
    for ds_uuid in sorted(set(evals_a) & set(evals_b)):
        ea, eb = evals_a[ds_uuid], evals_b[ds_uuid]
        problems = []
        if ea["held_out"] is False or eb["held_out"] is False:
            problems.append("scored on training data")
        if ea["held_out"] is None or eb["held_out"] is None:
            problems.append("held-out status unknown")
        if ea["eval_stride"] != eb["eval_stride"]:
            problems.append(f"different eval resolution ({ea['eval_stride']} vs {eb['eval_stride']})")
        shared_evals.append({
            "dataset_uuid": ds_uuid,
            "dataset_label": ea["dataset_label"] or eb["dataset_label"],
            "a": ea, "b": eb,
            "comparable": not problems,
            "problems": problems,
        })

    for d in differences:
        d["factor"] = _factor(d["field"])
    confounds = [d for d in differences if d["severity"] == "confound"]
    unknowns = [d for d in differences if d["severity"] == "unknown"]
    cautions = [d for d in differences if d["severity"] == "caution"]
    confound_factors = sorted({d["factor"] for d in confounds})
    unknown_factors = sorted({d["factor"] for d in unknowns} - set(confound_factors))

    if not confound_factors and not unknown_factors:
        verdict = ("CLEAN: no substantive differences besides bookkeeping"
                   if not cautions else f"CLEAN with {len(cautions)} caution(s)")
    elif len(confound_factors) == 1 and not unknown_factors:
        verdict = (f"SINGLE-FACTOR: only '{confound_factors[0]}' differs -- "
                   "a valid ablation of that factor")
    elif len(confound_factors) == 1:
        # One measured factor plus unknowns: the ablation is plausible but the
        # unknowns must be resolved before the comparison is citable.
        verdict = (f"SINGLE-FACTOR with unknowns: '{confound_factors[0]}' differs, "
                   f"but {', '.join(unknown_factors)} cannot be verified equal")
    elif not confound_factors:
        # Zero measured differences and some unrecorded facts. This fell into the CONFOUNDED
        # branch below and printed "0 confounded factor(s)" -- the strongest word in this
        # vocabulary attached to the weakest evidence there is. Measured 2026-08-27: of six
        # pairs whose materialized Hydra configs are BYTE-IDENTICAL, three read CONFOUNDED
        # with an empty confound list, which is what the report that exact repeats read
        # confounded but give the same model actually was.
        verdict = ("NO MEASURED DIFFERENCE: nothing this comparison can measure differs, but "
                   f"{', '.join(unknown_factors)} cannot be verified equal")
    else:
        parts = [f"{len(confound_factors)} confounded factor(s): "
                 + ", ".join(confound_factors)]
        if unknown_factors:
            parts.append(f"unknown: {', '.join(unknown_factors)}")
        verdict = "CONFOUNDED: " + "; ".join(parts)

    return {
        "a": {k: fa[k] for k in ("pk", "uuid", "name", "config_name", "ctime")},
        "b": {k: fb[k] for k in ("pk", "uuid", "name", "config_name", "ctime")},
        "training_data": data_relation,
        "differences": differences,
        "confounds": confounds,
        "confound_factors": confound_factors,
        "cautions": cautions,
        "unknowns": unknowns,
        "unknown_factors": unknown_factors,
        "shared_evaluations": shared_evals,
        "backfilled_facts": sorted((fa["backfilled"] | fb["backfilled"])
                                   & {"training_seed", "training_stride", "steps_trained",
                                      "material_name", "held_out"}),
        "verdict": verdict,
    }
