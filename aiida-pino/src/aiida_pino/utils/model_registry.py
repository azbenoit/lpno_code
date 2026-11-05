"""Helpers for dataset unions, model registry nodes, and evaluation records."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import tempfile
from typing import Any, Iterable

from aiida import orm

from aiida_pino.data import (
    PinoDatasetCollectionData,
    PinoDatasetData,
    PinoDatasetUnionData,
    PinoModelData,
    PinoModelEvaluationData,
    PinoSimulationData,
    collection_content_signature,
    common_field_set,
    safe_attribute_get,
    validate_collection_members,
    build_family_identifier,
    datasource_cache_basename_from_metadata,
)
from aiida_pino.data.pino_model import pk_for_uuid
from aiida_pino.utils.dataset_derivation import (
    apply_dataset_layout_to_training_profile,
    canonical_base_layout_profile,
    validate_derivation_profile,
)
from aiida_pino.utils.profiles import validate_training_profile


PUBLIC_DATASET_INPUT_TYPES = (PinoDatasetCollectionData,)
MODEL_INPUT_TYPES = (PinoModelData, orm.FolderData, orm.RemoteData)
EVALUATION_OVERRIDE_KEYS = frozenset(
    {
        "run_name",
        "cuda_device",
        "wandb_offline",
        "inference_batch_size",
        "infer_velocities",
        "full_frames",
        "use_packaged_config",
        "hydra_overrides",
        "autoregressive",
        # Measured peaks so gpu_spool can pack this eval beside other runs;
        # omitted means the whole card, as before.
        "gpu_vram_mb",
        "host_ram_mb",
        # Priority buys a queue-jump but costs a deadline: gpu_spool waits
        # forever for a plain seat and gives up after 3600s with --priority.
        # false = wait indefinitely instead of failing on a saturated box.
        "gpu_seat_priority",
        # A refusal that reflects a JUDGEMENT must be overridable. The
        # VRAM guard refuses a grid-bound eval booked below what its model needed, which is
        # right by default and wrong when a human means it: one model declared 14000 by
        # hand to skip the whole-card smoke, was refused at 8000, and was forced to a figure
        # nothing on the box could free -- unevaluable, with no way to say so.
        "allow_vram_below_training",
        # Same switches training takes (PINO_LAZY_STRIDE, PINO_LOADER_WORKERS,
        # PYTORCH_CUDA_ALLOC_CONF, ...). Without this an eval could not reproduce
        # the grid its model trained on: for RIGNO the graph is rebuilt from the
        # coordinates, so a missing stride silently scores a different resolution
        # instead of raising.
        "env",
    }
)
EVALUATION_AUTOREGRESSIVE_OVERRIDE_KEYS = frozenset(
    {
        "eval_horizon_k",
        "eval_use_gt_init",
        "require_time",
        "state_prev_suffix",
    }
)


def public_dataset_type_tuple() -> tuple[type[Any], ...]:
    """Return the native public dataset input tuple."""
    return PUBLIC_DATASET_INPUT_TYPES


def model_input_type_tuple() -> tuple[type[Any], ...]:
    """Return model input types accepted by low-level staging helpers."""
    return MODEL_INPUT_TYPES


def dataset_source_kind(node: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData) -> str:
    """Return the public dataset-source kind name."""
    if isinstance(node, PinoDatasetCollectionData):
        return "dataset_collection"
    return "dataset_union" if isinstance(node, PinoDatasetUnionData) else "dataset"


def dataset_sample_count(node: PinoDatasetCollectionData | PinoDatasetData) -> int:
    """Return the number of samples in a dataset."""
    if isinstance(node, PinoDatasetCollectionData):
        return node.get_simulation_count()
    metadata = node.get_dataset_metadata()
    sample_metadata = list(metadata.get("sample_metadata", []))
    if sample_metadata:
        return len(sample_metadata)
    shapes = node.get_array_shapes()
    if not shapes:
        return 0
    first_shape = next(iter(shapes.values()), [])
    return int(first_shape[0]) if first_shape else 0


def dataset_compatibility_signature(node: PinoDatasetCollectionData | PinoDatasetData) -> str:
    """Return a stable compatibility signature for dataset unions."""
    if isinstance(node, PinoDatasetCollectionData):
        metadata = node.get_dataset_metadata()
        payload = {
            "compatibility_signature": node.get_compatibility_signature(),
            "dataset_role": metadata.get("dataset_role", ""),
            "derivation_profile": metadata.get("derivation_profile", {}),
            "source_type": metadata.get("source_type", ""),
            "family_identifier": metadata.get("family_identifier", ""),
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
    metadata = node.get_dataset_metadata()
    shapes = node.get_array_shapes()
    non_sample_shapes = {name: list(shape[1:]) for name, shape in shapes.items()}
    payload = {
        "canonical_layout": metadata.get("canonical_layout", ""),
        "dataset_role": metadata.get("dataset_role", ""),
        "derivation_profile": metadata.get("derivation_profile", {}),
        "array_names": sorted(node.list_original_array_names()),
        "non_sample_shapes": non_sample_shapes,
        "grid_params": metadata.get("grid_params", {}),
        "requested_fields": sorted(metadata.get("requested_fields", [])),
        "source_type": metadata.get("source_type", ""),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _flatten_dataset_members(
    members: Iterable[PinoDatasetData | PinoDatasetUnionData],
) -> list[PinoDatasetData]:
    flattened: list[PinoDatasetData] = []
    seen: set[str] = set()
    for member in members:
        if isinstance(member, PinoDatasetUnionData):
            nested = member.iter_member_datasets()
        elif isinstance(member, PinoDatasetData):
            nested = [member]
        else:
            raise ValueError("Dataset unions can only contain native PINO datasets or dataset unions.")
        for dataset in nested:
            if dataset.uuid in seen:
                continue
            flattened.append(dataset)
            seen.add(dataset.uuid)
    return flattened


def validate_union_members(
    members: Iterable[PinoDatasetData | PinoDatasetUnionData],
) -> tuple[list[PinoDatasetData], str, int]:
    """Flatten, de-duplicate, and validate union members."""
    flattened = _flatten_dataset_members(members)
    if not flattened:
        raise ValueError("Select at least one native dataset to create a union.")

    signature = dataset_compatibility_signature(flattened[0])
    total_samples = dataset_sample_count(flattened[0])
    for dataset in flattened[1:]:
        candidate = dataset_compatibility_signature(dataset)
        if candidate != signature:
            raise ValueError("Selected datasets are not compatible for union concatenation.")
        total_samples += dataset_sample_count(dataset)
    return flattened, signature, total_samples


def _union_name_fallback(members: list[PinoDatasetData]) -> str:
    if len(members) == 1:
        return f"{members[0].label or members[0].pk}_union"
    return f"{members[0].label or members[0].pk}_plus_{len(members) - 1}_more"


def find_reusable_dataset_union(
    members: Iterable[PinoDatasetData | PinoDatasetUnionData],
    *,
    union_name: str,
) -> PinoDatasetUnionData | None:
    """Return an existing stored union for the same ordered member list and name."""
    flattened, signature, _ = validate_union_members(members)
    member_uuids = [member.uuid for member in flattened]
    builder = orm.QueryBuilder()
    builder.append(
        PinoDatasetUnionData,
        filters={
            "attributes.member_dataset_uuids": {"==": member_uuids},
            "attributes.compatibility_signature": {"==": signature},
            "attributes.union_name": {"==": union_name},
        },
        project=["*"],
    )
    builder.order_by({PinoDatasetUnionData: {"ctime": "desc"}})
    return builder.first(flat=True)


def get_or_create_dataset_union(
    members: Iterable[PinoDatasetData | PinoDatasetUnionData],
    *,
    union_name: str | None = None,
) -> tuple[PinoDatasetUnionData, bool]:
    """Return a stored metadata-only union node for compatible native datasets."""
    flattened, signature, total_samples = validate_union_members(members)
    resolved_name = (union_name or "").strip() or _union_name_fallback(flattened)
    existing = find_reusable_dataset_union(flattened, union_name=resolved_name)
    if existing is not None:
        return existing, False
    node = PinoDatasetUnionData.from_members(
        flattened,
        union_name=resolved_name,
        compatibility_signature=signature,
        total_samples=total_samples,
    ).store()
    return node, True


def recent_collection_nodes(limit: int = 50) -> list[PinoDatasetCollectionData]:
    """Return recent active dataset collections.

    Superseded collections are excluded in the query, not after fetching: the
    old fetch-2x-then-filter approach let supersession churn (e.g. an AL loop
    creating hundreds of superseded chain nodes) fill the recency window and
    silently push every older live dataset out of all listings.
    """
    builder = orm.QueryBuilder()
    builder.append(
        PinoDatasetCollectionData,
        filters={"or": [{"extras": {"!has_key": "superseded"}}, {"extras.superseded": {"==": False}}]},
        project=["*"],
    )
    builder.order_by({PinoDatasetCollectionData: {"ctime": "desc"}})
    return list(builder.limit(limit).all(flat=True))


def _simulation_fingerprint_sequence(simulations: Iterable[PinoSimulationData]) -> list[str]:
    """Return the ordered simulation-fingerprint sequence for a simulation set."""
    return [member.get_simulation_fingerprint() for member in simulations]


def find_reusable_dataset_collection(
    simulations: Iterable[PinoSimulationData],
    *,
    metadata: dict[str, Any],
) -> PinoDatasetCollectionData | None:
    """Return an existing collection for the same ordered member list and derivation."""
    members, signature = validate_collection_members(
        simulations, field_policy=str(metadata.get("field_policy") or "strict")
    )
    normalized = dict(metadata)
    # Content identity, not name identity: two collections holding the same
    # members under the same derivation serve identical training data, whatever
    # they are called and whatever role they carry. Matching on
    # family/identifier/role let the same data re-enter under a new name and
    # split every downstream model comparison (kh2579 vs the 1601 splits: the
    # same 195/52/53 sims listed twice, retired by hand 2026-07-27).
    content = collection_content_signature(
        (member.get_simulation_fingerprint() for member in members),
        derivation_profile=dict(normalized.get("derivation_profile", {}) or {}),
        compatibility_signature=signature,
    )
    builder = orm.QueryBuilder()
    builder.append(
        PinoDatasetCollectionData,
        filters={"attributes.content_signature": {"==": content}},
        project=["*"],
    )
    builder.order_by({PinoDatasetCollectionData: {"ctime": "asc"}})
    for candidate in builder.all(flat=True):
        if not bool(candidate.base.extras.get("superseded", False)):
            return candidate
    return None


def get_or_create_dataset_collection(
    simulations: Iterable[PinoSimulationData],
    *,
    metadata: dict[str, Any],
) -> tuple[PinoDatasetCollectionData, bool]:
    """Return a stored dataset collection for compatible simulations."""
    members, _signature = validate_collection_members(
        simulations, field_policy=str(metadata.get("field_policy") or "strict")
    )
    existing = find_reusable_dataset_collection(members, metadata=metadata)
    if existing is not None:
        return existing, False
    node = PinoDatasetCollectionData.from_simulations(members, metadata=metadata).store()
    supersede_equivalent_dataset_collections(node)
    return node, True


def mark_superseded(old, new) -> None:
    """Retire `old` in favour of `new`, carrying the user's annotations across.

    Supersession replaces the node the user curated, so the title, notes and
    keywords have to move with it; left behind they vanish from every listing
    (superseded nodes are filtered out) and the live successor looks untagged.
    Existing annotations on `new` win — it is the more recent intent. The label
    is deliberately not carried: collections auto-generate one, so a differing
    old label is as likely a deliberate rename as a user title.
    """
    old.base.extras.set("superseded", True)
    old.base.extras.set("replaced_by_uuid", new.uuid)
    old.base.extras.set("replaced_by_pk", new.pk)
    if not new.description and old.description:
        new.description = old.description
    inherited = [k for k in (old.base.extras.get("keywords", None) or []) if k not in (new.base.extras.get("keywords", None) or [])]
    if inherited:
        new.base.extras.set("keywords", list(new.base.extras.get("keywords", None) or []) + inherited)


def supersede_equivalent_dataset_collections(
    node: PinoDatasetCollectionData,
) -> list[PinoDatasetCollectionData]:
    """Mark older equivalent collections as superseded so selectors stay uncluttered.

    Equivalence is content identity (`content_signature`): same members, same
    derivation — role- and name-agnostic, so a rename or re-role cannot leave
    two live nodes serving identical data. `core` reference sets are never
    superseded: if new content collides with one, the core set stays live and
    callers should be handed the core node by the reuse path instead.
    """
    content = safe_attribute_get(node, "content_signature", "")
    if not content:
        return []
    builder = orm.QueryBuilder()
    builder.append(
        PinoDatasetCollectionData,
        filters={"attributes.content_signature": {"==": content}},
        project=["*"],
    )
    builder.order_by({PinoDatasetCollectionData: {"ctime": "asc"}})
    superseded: list[PinoDatasetCollectionData] = []
    for candidate in builder.all(flat=True):
        if candidate.uuid == node.uuid:
            continue
        if "core" in (candidate.base.extras.get("keywords", None) or []):
            continue
        mark_superseded(candidate, node)
        superseded.append(candidate)
    node.base.extras.set("superseded", False)
    for key in ("replaced_by_uuid", "replaced_by_pk"):
        try:
            node.base.extras.delete(key)
        except AttributeError:
            pass
    return superseded


def query_simulation_nodes(
    *,
    source_type: str | None = None,
    source_folder_contains: str | None = None,
    required_fields: Iterable[str] | None = None,
    split: str | None = None,
    keywords: Iterable[str] | None = None,
    include_uuids: Iterable[str] | None = None,
    exclude_uuids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[PinoSimulationData]:
    """Return simulations matching simple composition filters.

    ``split`` matches the ``split`` extra ("unassigned" also matches nodes
    without the extra); ``keywords`` requires every given keyword to be present
    in the ``keywords`` extra (AND semantics via JSON containment).
    """
    builder = orm.QueryBuilder()
    filters: dict[str, Any] = {}
    if source_type:
        filters["extras.source_type"] = {"==": source_type}
    if split:
        if split == "unassigned":
            filters["or"] = [
                {"extras.split": {"==": "unassigned"}},
                {"extras": {"!has_key": "split"}},
            ]
        else:
            filters["extras.split"] = {"==": split}
    keyword_list = [str(keyword) for keyword in (keywords or []) if str(keyword)]
    if keyword_list:
        filters["extras.keywords"] = {"contains": keyword_list}
    builder.append(PinoSimulationData, filters=filters, project=["*"])
    builder.order_by({PinoSimulationData: {"ctime": "desc"}})
    nodes = builder.all(flat=True)
    if source_folder_contains:
        needle = source_folder_contains.strip().lower()
        nodes = [node for node in nodes if needle in str(node.base.extras.get("source_folder_name", "")).lower()]
    fields = [field for field in (required_fields or []) if field]
    if fields:
        nodes = [
            node
            for node in nodes
            if all(field in set(node.get_simulation_metadata().get("requested_fields", [])) for field in fields)
        ]
    include = {value for value in (include_uuids or []) if value}
    exclude = {value for value in (exclude_uuids or []) if value}
    if include:
        nodes = [node for node in nodes if node.uuid in include]
    if exclude:
        nodes = [node for node in nodes if node.uuid not in exclude]
    if limit is not None:
        nodes = nodes[: int(limit)]
    return nodes


def build_collection_metadata_from_simulations(
    simulations: Iterable[PinoSimulationData],
    *,
    collection_name: str,
    dataset_role: str = "base",
    composition_kind: str = "subset",
    derivation_profile: dict[str, Any] | None = None,
    base_collection: PinoDatasetCollectionData | None = None,
    parent_collection: PinoDatasetCollectionData | None = None,
    source_uri: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build collection metadata for an ordered simulation set."""
    field_policy = str((extra_metadata or {}).get("field_policy") or "strict")
    members, signature = validate_collection_members(simulations, field_policy=field_policy)
    first_metadata = members[0].get_simulation_metadata()
    if derivation_profile is None:
        time_counts = {int(m.get_simulation_metadata().get("num_timesteps") or 0) for m in members}
        if len(time_counts) > 1:
            # Ragged member set: the canonical time_as_dim layout can never be
            # served for it (native_source_incompatibility_reason rejects
            # time_as_dim without uniform timesteps), so record the
            # time-as-batch variant instead of a layout the training path
            # cannot use.
            derivation_profile = {**canonical_base_layout_profile(), "time_as_dim": False, "time_as_batch": True}
    profile = validate_derivation_profile(derivation_profile or canonical_base_layout_profile())
    source_folder_name = collection_name.strip() or first_metadata.get("source_folder_name", "dataset_collection")
    source_type = first_metadata.get("source_type", "flow3d")
    metadata: dict[str, Any] = {
        "source_type": source_type,
        "source_uri": source_uri if source_uri is not None else first_metadata.get("source_uri", ""),
        "source_folder_name": source_folder_name,
        "source_origin_kind": first_metadata.get("source_origin_kind", "import"),
        "requested_fields": (
            common_field_set(members)
            if field_policy == "intersect"
            else list(first_metadata.get("requested_fields", []))
        ),
        "grid_params": dict(first_metadata.get("grid_params", {})),
        "dataset_role": dataset_role,
        "composition_kind": composition_kind,
        "derivation_profile": profile,
        "derivation_hash": "" if dataset_role == "base" else hashlib.sha1(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12],
        "reconstructable": dataset_role == "derived",
        "time_axis": bool(profile.get("include_time", True)),
        "moving_ref_frame": bool(profile.get("moving_ref_frame", False)),
        "moving_frame_steps": int(profile.get("moving_frame_steps", 1)),
        "avg_window_size": int(profile.get("time_avg_window", 1)),
        "super_res_factor": int(profile.get("super_res_factor", 1)),
        "time_axis_mode": {
            "include_time": bool(profile.get("include_time", True)),
            "time_as_batch": bool(profile.get("time_as_batch", False)),
            "time_as_dim": bool(profile.get("time_as_dim", True)),
        },
        "family_identifier": build_family_identifier(
            source_folder_name,
            source_type,
            grid_params=first_metadata.get("grid_params", {}),
            requested_fields=list(first_metadata.get("requested_fields", [])),
            source_uri=source_uri if source_uri is not None else first_metadata.get("source_uri", ""),
        ),
        "compatibility_signature": signature,
        "base_collection_uuid": base_collection.uuid if base_collection is not None else "",
        "base_collection_pk": base_collection.pk if base_collection is not None else None,
        "parent_collection_uuid": parent_collection.uuid if parent_collection is not None else "",
        "parent_collection_pk": parent_collection.pk if parent_collection is not None else None,
        "sample_metadata": [],
    }
    metadata["cache_basename"] = datasource_cache_basename_from_metadata(metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def export_dataset_union(node: PinoDatasetUnionData, target_dir: str | Path) -> Path:
    """Stage a metadata-only runtime union without concatenating member payloads."""
    members = node.iter_member_datasets()
    if not members:
        raise ValueError("Dataset union has no member datasets.")
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    manifest_members: list[dict[str, Any]] = []
    total_samples = 0

    for index, member in enumerate(members):
        relative_path = Path("members") / f"member_{index:03d}"
        member_manifest = member.export_runtime_arraydir(target_path / relative_path)
        member_samples = dataset_sample_count(member)
        manifest_members.append(
            {
                "pk": member.pk,
                "uuid": member.uuid,
                "label": member.label or "",
                "dataset_identifier": member.get_dataset_metadata().get("dataset_identifier", ""),
                "relative_path": str(relative_path),
                "manifest_file": str(relative_path / member_manifest.name),
                "sample_offset": total_samples,
                "sample_count": member_samples,
            }
        )
        total_samples += member_samples

    manifest = {
        "format": PinoDatasetData.RUNTIME_MANIFEST_VERSION,
        "source_kind": "dataset_union",
        "union_identifier": node.get_union_identifier(),
        "union_name": node.get_union_name(),
        "compatibility_signature": node.get_compatibility_signature(),
        "members": manifest_members,
        "total_samples": total_samples,
    }
    (target_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return target_path / "manifest.json"


def recent_union_nodes(limit: int = 50) -> list[PinoDatasetUnionData]:
    """Return recent stored dataset-union nodes."""
    builder = orm.QueryBuilder()
    builder.append(PinoDatasetUnionData, project=["*"])
    builder.order_by({PinoDatasetUnionData: {"ctime": "desc"}})
    return builder.limit(limit).all(flat=True)


def recent_model_nodes(limit: int = 50, offset: int = 0) -> list[PinoModelData]:
    """Return recent stored models. `limit=0` means *no cap*, not "none".

    `QueryBuilder.limit(0)` returns an empty result, so the console's "show
    everything" contract (0 = no cap) silently answered zero models until this
    was guarded.

    `offset` exists so the console can build ONE PAGE of rows instead of all of them.
    Building a row is the expensive part -- 447 rows cost 22.7 s -- so slicing after the
    fact saved nothing; the page has to be narrowed here, at the fetch.
    """
    builder = orm.QueryBuilder()
    builder.append(PinoModelData, project=["*"])
    builder.order_by({PinoModelData: {"ctime": "desc"}})
    if offset:
        builder.offset(int(offset))
    return builder.limit(limit or None).all(flat=True)


def model_evaluations_for_model(model: PinoModelData) -> list[PinoModelEvaluationData]:
    """Return linked evaluation records for a stored model."""
    builder = orm.QueryBuilder()
    builder.append(
        PinoModelEvaluationData,
        filters={"extras.model_uuid": {"==": model.uuid}},
        project=["*"],
    )
    builder.order_by({PinoModelEvaluationData: {"ctime": "desc"}})
    return builder.all(flat=True)


def summarize_model_evaluation_record(record: PinoModelEvaluationData) -> dict[str, Any]:
    """Return structured metadata for a model evaluation record."""
    metadata = record.get_evaluation_metadata()
    return {
        "pk": record.pk,
        "label": record.label or metadata.get("evaluation_name") or "evaluation",
        "ctime": record.ctime.isoformat(timespec="seconds"),
        **metadata,
    }


def describe_model_node(node: PinoModelData) -> str:
    """Create a compact model label for selectors."""
    metadata = node.get_model_metadata()
    model_name = metadata.get("model_name") or metadata.get("run_name") or node.label or "-"
    source_kind = metadata.get("training_source_kind", "-")
    source_label = metadata.get("training_source_label", "-")
    config_name = metadata.get("config_name", "-")
    return f"{node.pk} | PinoModelData | {model_name} | {config_name} | {source_kind} | {source_label}"


def summarize_model_node(
    node: PinoModelData,
    *,
    include_evaluations: bool = True,
    include_files: bool = True,
) -> dict[str, Any]:
    """Return a metadata-first summary for model browser views.

    The two `include_*` flags exist for LISTINGS, which need neither and paid dearly for
    both. `model_evaluations_for_model` is one DB query per model and `safe_files` is one
    repository listing per model, so a 100-model page ran 200 round trips for an evaluation
    count it could take from the index it had already built. Measured 2.76 s of a 10.9 s
    response.

    Both default to True, so the detail view -- which does use the records and the file
    list -- is unchanged.
    """
    metadata = node.get_model_metadata()
    evaluations = (
        [summarize_model_evaluation_record(record) for record in model_evaluations_for_model(node)]
        if include_evaluations
        else []
    )
    return {
        "pk": node.pk,
        "label": node.label or "-",
        "model_name": node.get_model_name(),
        "config_name": metadata.get("config_name", ""),
        "training_profile": metadata.get("training_profile", {}),
        "derivation_profile": metadata.get("derivation_profile", {}),
        "training_source_kind": metadata.get("training_source_kind", ""),
        "training_source_pk": metadata.get("training_source_pk"),
        "training_source_uuid": metadata.get("training_source_uuid", ""),
        "training_source_label": metadata.get("training_source_label", ""),
        "primary_checkpoint": metadata.get("primary_checkpoint", ""),
        "training_summary": metadata.get("training_summary", {}),
        "copied_files": list(safe_files(node)) if include_files else [],
        "evaluations": evaluations,
        # None, not 0: a listing supplies its own count from the evaluation index, and a
        # hard 0 here would silently overwrite it with "never evaluated".
        "evaluation_count": len(evaluations) if include_evaluations else None,
    }


def safe_files(node: orm.FolderData) -> list[str]:
    """Return repository file paths for a folder-like node."""
    try:
        return list(node.base.repository.list_object_names())
    except Exception:
        return []


def validate_evaluation_profile(values: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the limited set of eval-time overrides."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("evaluation_profile must be a mapping.")
    unknown = sorted(set(values).difference(EVALUATION_OVERRIDE_KEYS))
    if unknown:
        raise ValueError(f"Unknown evaluation_profile keys: {', '.join(unknown)}")

    profile: dict[str, Any] = {}
    if "run_name" in values:
        profile["run_name"] = str(values["run_name"])
    if "cuda_device" in values:
        profile["cuda_device"] = "" if values["cuda_device"] is None else str(values["cuda_device"])
    if "wandb_offline" in values:
        profile["wandb_offline"] = bool(values["wandb_offline"])
    if "inference_batch_size" in values:
        profile["inference_batch_size"] = int(values["inference_batch_size"])
        if profile["inference_batch_size"] <= 0:
            raise ValueError("evaluation_profile.inference_batch_size must be positive.")
    if "infer_velocities" in values:
        profile["infer_velocities"] = bool(values["infer_velocities"])
    if "full_frames" in values:
        profile["full_frames"] = bool(values["full_frames"])
    # Escape hatch for the archived-config default (2026-08-14): an eval reads
    # the model's OWN `.hydra/config.yaml` so an old checkpoint stays loadable
    # as the packaged schema moves. Set this to score a model under today's
    # config deliberately -- e.g. to reproduce the schema drift itself.
    if "use_packaged_config" in values:
        profile["use_packaged_config"] = bool(values["use_packaged_config"])
    for key in ("gpu_vram_mb", "host_ram_mb"):
        if key in values:
            value = values[key]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"evaluation_profile.{key} must be a positive integer of MiB.")
            profile[key] = value
    if "allow_vram_below_training" in values:
        raw = values["allow_vram_below_training"]
        if not isinstance(raw, bool):
            # `bool("false")` is True, and a bypass that silently switches itself on is worse
            # than no bypass -- the same reasoning as `gpu_seat_priority` below.
            raise ValueError("evaluation_profile.allow_vram_below_training must be a boolean.")
        profile["allow_vram_below_training"] = raw
    if "gpu_seat_priority" in values:
        # CLI values arrive as strings, so accept the usual spellings rather
        # than letting bool("false") silently mean True.
        raw = values["gpu_seat_priority"]
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError("evaluation_profile.gpu_seat_priority must be a boolean.")
            profile["gpu_seat_priority"] = lowered in {"true", "1", "yes"}
        else:
            profile["gpu_seat_priority"] = bool(raw)
    if "env" in values:
        env = values["env"]
        if not isinstance(env, dict):
            raise ValueError("evaluation_profile.env must be a mapping of environment variables.")
        # REPLACES the model's training env, it does not merge into it, so omitting an
        # inherited key is how you unset it (`merge_model_training_with_evaluation_profile`
        # assigns this dict). Stated because the opposite was reported as a defect on
        # 2026-09-02 after two evals kept an inherited `PINO_LAZY_CROP_Z_TOP` despite an
        # explicit env that omitted it -- the variable was re-added afterwards from
        # `ctx.demoted_env`, which describes how the dataset was STAGED and legitimately
        # wins over an inherited value. So a key can reappear after this replacement, and
        # `reconcile_crop_env` is what resolves the crop case.
        profile["env"] = {str(key): str(value) for key, value in env.items()}
    if "hydra_overrides" in values:
        if not isinstance(values["hydra_overrides"], list) or not all(
            isinstance(item, str) for item in values["hydra_overrides"]
        ):
            raise ValueError("evaluation_profile.hydra_overrides must be a list of strings.")
        profile["hydra_overrides"] = list(values["hydra_overrides"])
    if "autoregressive" in values:
        autoregressive = dict(values["autoregressive"] or {})
        autoregressive.pop("enabled", None)
        unknown_autoregressive = sorted(
            set(autoregressive).difference(EVALUATION_AUTOREGRESSIVE_OVERRIDE_KEYS)
        )
        if unknown_autoregressive:
            raise ValueError(
                "Unknown evaluation_profile.autoregressive keys: "
                + ", ".join(unknown_autoregressive)
            )
        profile["autoregressive"] = autoregressive
    return profile


def merge_model_training_with_evaluation_profile(
    model: PinoModelData,
    evaluation_profile: dict[str, Any] | None,
    *,
    dataset_metadata: dict | None = None,
) -> dict[str, Any]:
    """Merge a model's frozen training profile with eval-time overrides only."""
    frozen_profile = model.get_training_profile()
    # for_launch=False: this profile's training already ran. The resource check is a
    # pre-launch question, and the eval's own profile governs eval resources.
    base_profile = validate_training_profile(
        frozen_profile, expected_split_mode="standard", for_launch=False
    )
    # A model frozen before the power_weighted_dose stamp existed trained on
    # power-blind dose channels; pin its evals to False so neither the new
    # default-ON stamp (which `validate_training_profile` just applied to the
    # normalized copy) nor an edited yaml (config_flow3D_GAOT_pvlhs now sets
    # true) silently feeds it rescaled inputs. Pre-stamp-ness is judged on the
    # RAW frozen profile; stamped models carry their explicit value through.
    if frozen_profile.get("power_weighted_dose") is None:
        base_profile["power_weighted_dose"] = False
    # A training's GPU choice has no bearing on where its EVAL should run, and inheriting it is
    # actively harmful: the eval runs later, under different occupancy, and a pinned job can only
    # wait for its own card while every other card frees around it. Measured 2026-08-21: three
    # models all carry cuda_device="0", so every eval of the trio was pinned to GPU 0
    # -- one sat 24 minutes behind an 18000 MiB reservation there while GPU 1 was reachable, and
    # another (from a model with cuda_device="1") sat 71 minutes on the mirror of it. Default to
    # auto; an evaluation profile that names a device explicitly still wins, via `overrides` below.
    base_profile["cuda_device"] = "auto"
    overrides = validate_evaluation_profile(evaluation_profile)
    merged = copy.deepcopy(base_profile)
    for key, value in overrides.items():
        if key == "autoregressive":
            merged["autoregressive"] = dict(merged.get("autoregressive", {}))
            merged["autoregressive"].update(value)
        elif key == "hydra_overrides":
            # APPENDED, not replaced. These strings include the ones that decide
            # the ARCHITECTURE -- `++custom.FNO_input_keys=[...]` sets the lift
            # layer's channel count -- so replacing the list rebuilds the model
            # to the packaged default and the checkpoint no longer fits it.
            #
            # 2026-08-21: six evals died at `_assert_eval_checkpoint_loadable`
            # with [32,20,...] and [32,15,...] checkpoints against a [32,19,...]
            # model. The evals had supplied ONE unrelated override
            # (`++custom.T_boil_k=3520.0`), which discarded all eleven of the
            # training's. Appending keeps hydra's last-wins semantics, so an
            # eval that restates a key still overrides the training's value.
            inherited = list(merged.get("hydra_overrides") or [])
            merged["hydra_overrides"] = inherited + [
                item for item in value if item not in inherited
            ]
        else:
            merged[key] = value
    if dataset_metadata:
        merged = apply_dataset_layout_to_training_profile(merged, dataset_metadata)
    # The measured peak is the real anchor; the frozen booking may never have been one.
    # Read defensively: a model whose metadata cannot be reached must fall back to the
    # declared booking (the documented fallback), not raise. Hard-requiring the accessor
    # broke six tests whose stub models implement only `get_training_profile`, and would
    # have done the same to any caller holding a lighter model object.
    try:
        _measured_peak = (
            (getattr(model, "get_model_metadata", dict)() or {}).get("training_summary") or {}
        ).get("gpu_vram_peak_mb")
    except Exception:  # noqa: BLE001 - a missing measurement must never block a merge
        _measured_peak = None
    _check_graph_eval_vram_against_training(frozen_profile, merged, _measured_peak)
    return validate_training_profile(merged, expected_split_mode="standard")


# Architectures whose peak allocation is set by the GRID, not the batch: RIGNO builds an
# edge MLP over the full edge set and GAOT builds an encoder graph over every node. Cutting
# the batch does not shrink either, so an eval of one needs about what its training needed.
_GRID_BOUND_ARCHITECTURES = ("RIGNO", "GAOT")


def _check_graph_eval_vram_against_training(
    frozen_profile: dict[str, Any], merged: dict[str, Any], measured_peak_mb: Any = None
) -> None:
    """Refuse an eval VRAM booking a grid-bound model cannot honour.

    Under-declaring is the DAMAGING direction, and it is not symmetric with over-declaring.
    An over-declared job waits too long in the queue and then works. An under-declared one is
    ADMITTED on the strength of its own booking, is placed on a card alongside someone else's
    9.5 GiB, runs the model, and dies at the first big allocation -- losing the seat time and
    the result, which is the expensive half.

    That is exactly how `varspeed_rescore_m47850` died on 2026-08-21 (calc 61877, exit 500):
    booked 4000 MiB, needed 3.86 GiB in ONE allocation inside RIGNO's edge MLP against 3.16 GiB
    free, having been admitted onto a card another job held 9.5 GiB of. 4000 was picked by
    generalising from a GAOT model that was running fine at that figure -- but memory here is
    set by the edge set, so the number did not transfer across architectures.

    FNO is deliberately not covered: its footprint does scale with the batch, and steady FNO
    evals genuinely peak at ~164 MiB against far larger training bookings. Refusing those
    would be the over-strict mistake in the other direction.
    """
    config_name = str(frozen_profile.get("config_name") or "").upper()
    if not any(arch in config_name for arch in _GRID_BOUND_ARCHITECTURES):
        return
    # Time-averaged treatments are exempt, and this exemption is load-bearing rather than
    # cautious. A steady eval collapses `time_avg_window` frames into one sample, so its working
    # set is ~30x smaller than the per-frame evals this check exists for. Measured: one
    # RIGNO steady model that TRAINED at 12000 MiB and evaluated cleanly declared at 3000
    # (calc 61787, peak 164 MiB). Without this branch the check would refuse that -- an eval that
    # demonstrably works -- which is precisely the over-strict failure this file committed once
    # already today. The two evals that actually OOMed are both
    # time_avg_window=1, moving_ref_frame=False: every frame processed, nothing collapsed.
    try:
        if int(merged.get("time_avg_window") or 1) > 1:
            return
    except (TypeError, ValueError):
        pass
    # Anchor on what was MEASURED, not on what was booked. A declared `gpu_vram_mb` is
    # whatever a human or a clone put there, and hand-setting it high to skip the
    # whole-card auto-size smoke is a documented practice -- so this guard was enforcing
    # numbers that were never measurements. Measured 2026-08-24 on two models identical in
    # architecture, config, dataset, channels and steps, with the SAME 5004 MiB training
    # peak: 66267 declared 6000 and its eval ran at 8000 (actual peak 1514 MiB), while
    # 66067 declared 14000 -- hand-set, its own label says "to skip the whole-card smoke"
    # -- and its eval was refused at 8000 and forced to a figure nothing on the box could
    # free, so it could not start at all.
    #
    # The docstring above already argues for this: the quantity that OOMed calc 61877 is
    # the edge-set/encoder graph footprint, and 4000 was wrong precisely because it was
    # generalised rather than measured. `gpu_vram_peak_mb` is that quantity; the booking
    # is not. Protection for the case that motivated the guard is unchanged -- that
    # model's measured peak also exceeded the 4000 it declared.
    try:
        measured = int(measured_peak_mb) if measured_peak_mb else 0
    except (TypeError, ValueError):
        measured = 0
    booked = frozen_profile.get("gpu_vram_mb")
    trained = measured or booked
    source = "measured training peak" if measured else "training booking (no measured peak recorded)"
    declared = merged.get("gpu_vram_mb")
    if not trained or not declared or int(declared) >= int(trained):
        return
    if measured:
        # Correction first: the bypass opts out of a REFUSAL, not out of a raise to a figure
        # that was actually measured. Raising to a measurement only makes the job likelier to
        # run, so suppressing it would serve nobody.
        # CORRECT it rather than refuse. The refusal already quotes the right number, so a
        # whole submission round was being spent to deliver a value we hold: 74 of 505 failed
        # evals since 2026-08-15 died here, and 52 more OOMed under-declared, a quarter of all
        # eval failures between them (2026-09-06). The workchain's auto-sizer
        # cannot help, because it only fires when the profile supplies NOTHING.
        #
        # Only when the figure is MEASURED. Raising to the training BOOKING would repeat the
        # model-66067 failure of forcing a number nothing on the box could free, so the
        # no-measurement case still refuses and keeps the protection calc 61877 needed.
        merged["gpu_vram_mb"] = int(measured)
        print(
            f"[eval-vram] raising evaluation_profile.gpu_vram_mb {declared} -> {int(measured)} "
            f"MiB: {config_name} is grid-bound and that is the model's measured training peak. "
            f"A smaller eval batch does not shrink the edge set / encoder graph.",
            flush=True,
        )
        return
    if bool(merged.get("allow_vram_below_training")):
        print(
            f"[eval-vram] allow_vram_below_training set: keeping gpu_vram_mb={declared} MiB "
            f"against the {int(trained)} MiB this model needed to train ({source}). "
            f"Under-declaring is the damaging direction -- the job is admitted on its own "
            f"booking and can die at the first big allocation -- so this is deliberate.",
            flush=True,
        )
        return
    raise ValueError(
        f"evaluation_profile.gpu_vram_mb={declared} is below the {int(trained)} MiB this model "
        f"needed to train ({source}), and {config_name} is grid-bound: its peak allocation comes "
        f"from the edge set / encoder graph, which a smaller eval BATCH does not shrink (only "
        f"time-averaging does, and this eval is per-frame). Booking less does not "
        f"make it fit -- it makes the job admissible onto a busy card, where it OOMs after doing "
        f"the work (calc 61877, 2026-08-21). Declare >= {int(trained)}."
    )


def lineage_member_uuids(summary) -> set:
    """Every simulation uuid a model has EVER trained on, warm-starts included.

    `training_source_summary` records only the last training stage, so a model
    fine-tuned from an ancestor carried none of the ancestor's cases — and the
    eval-side overlap guard (`evaluate_model.py:227-243`) is only as good as the
    set it is handed. 8 of 13 warm-start chains here are the campaign's
    established pattern, so this is the one contamination path that the
    label-independent guard did not already close.

    `init_lineage_uuids` is written at model creation as the transitive union of
    the init model's own lineage, so this stays a flat read rather than a walk.
    """
    if not isinstance(summary, dict):
        return set()
    out = {
        str(member.get("uuid"))
        for member in (summary.get("members") or [])
        if isinstance(member, dict) and member.get("uuid")
    }
    out |= {str(u) for u in (summary.get("init_lineage_uuids") or []) if u}
    return out


def build_training_source_summary(
    source: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData,
) -> dict[str, Any]:
    """Return a compact display summary for a training/evaluation source."""
    if isinstance(source, PinoDatasetCollectionData):
        metadata = source.get_dataset_metadata()
        members = source.iter_member_simulations()
        return {
            "kind": "dataset_collection",
            "pk": source.pk,
            "uuid": source.uuid,
            "label": source.label or metadata.get("dataset_identifier", ""),
            "dataset_identifier": metadata.get("dataset_identifier", ""),
            "family_identifier": metadata.get("family_identifier", ""),
            "dataset_role": metadata.get("dataset_role", ""),
            "composition_kind": metadata.get("composition_kind", ""),
            "simulation_count": source.get_simulation_count(),
            "members": [
                {
                    "pk": member.pk,
                    "uuid": member.uuid,
                    "label": member.label or "",
                    "simulation_identifier": member.get_simulation_identifier(),
                }
                for member in members
            ],
        }
    if isinstance(source, PinoDatasetUnionData):
        return {
            "kind": "dataset_union",
            "pk": source.pk,
            "uuid": source.uuid,
            "label": source.label or source.get_union_name(),
            "union_identifier": source.get_union_identifier(),
            "total_samples": source.get_total_samples(),
            "members": [
                {
                    "pk": member.pk,
                    "uuid": member.uuid,
                    "label": member.label or "",
                    "dataset_identifier": member.get_dataset_metadata().get("dataset_identifier", ""),
                }
                for member in source.iter_member_datasets()
            ],
        }
    metadata = source.get_dataset_metadata()
    return {
        "kind": "dataset",
        "pk": source.pk,
        "uuid": source.uuid,
        "label": source.label or metadata.get("dataset_identifier", ""),
        "dataset_identifier": metadata.get("dataset_identifier", ""),
        "family_identifier": metadata.get("family_identifier", ""),
        "dataset_role": metadata.get("dataset_role", ""),
        "canonical_layout": metadata.get("canonical_layout", ""),
    }


def build_model_metadata(
    training_artifacts: orm.FolderData,
    train_summary: orm.Dict,
    training_profile: orm.Dict,
    training_source: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData,
    derivation_profile: dict[str, Any],
    *,
    model_name: str | None = None,
    validation_source=None,
    init_model=None,
) -> dict[str, Any]:
    """Construct compact model metadata for a trained PINO model."""
    # for_import=True here means PRESERVE, not "this is an import": freeze-time
    # validation must never apply submission-time launch defaults. For a trained
    # run the submitted profile already carries its resolved values, so nothing
    # changes; for an import, re-applying the None->True power_weighted_dose
    # default here stamped every imported model as power-weighted -- fabricated
    # provenance the import workchain had explicitly avoided one call earlier,
    # and its evals then fed the checkpoint a dose channel it never trained on
    # (measured 2026-08-21: three demo imports all froze True).
    frozen_profile = validate_training_profile(
        training_profile.get_dict(), expected_split_mode="standard", for_import=True
    )
    summary = train_summary.get_dict()
    source_summary = build_training_source_summary(training_source)
    # Warm start: carry the ancestor's whole lineage forward, so the eval-side
    # overlap guard protects a fine-tuned model against what its ANCESTOR saw.
    # Transitive by construction -- the ancestor already unioned its own.
    if init_model is not None:
        try:
            ancestry = lineage_member_uuids(
                init_model.get_model_metadata().get("training_source_summary")
            )
        except Exception:  # noqa: BLE001 - a FolderData/RemoteData init has no lineage
            ancestry = set()
            source_summary["init_lineage_unknown"] = True
        if ancestry:
            source_summary["init_lineage_uuids"] = sorted(ancestry)
    primary_checkpoint = "model.ckpt"
    checkpoint_candidates: list[str] = []
    try:
        candidates: list[str] = []
        for root, _directories, files in training_artifacts.base.repository.walk():
            root_path = PurePosixPath(root)
            for filename in files:
                relative = root_path / filename if str(root_path) != "." else PurePosixPath(filename)
                if relative.suffix.lower() in {".ckpt", ".pt", ".pth"}:
                    candidates.append(str(relative))
        if candidates:
            # ONE sort key, shared with `resolve_primary_checkpoint`. It was
            # duplicated here, and the two must agree: a fix to checkpoint
            # preference that landed in only one of them would change what a
            # model registers as while leaving what it resolves to untouched.
            from aiida_pino.data.pino_model import _checkpoint_sort_key

            candidates.sort(key=_checkpoint_sort_key)
            primary_checkpoint = candidates[0]
            checkpoint_candidates = list(candidates)
    except Exception:
        pass
    # Which checkpoints exist, and a note when the one chosen is NOT the best.
    # `u_network.best.pth` is written to protect against late divergence and is
    # never read: it ties with the final checkpoint on every ranked field, so
    # the sort falls through to the raw path and '0' beats 'b'. Recorded rather
    # than acted on -- preferring best would change what every existing model
    # resolves to, including published numbers.
    from aiida_pino.data.pino_model import checkpoint_choice_note

    shadowed = checkpoint_choice_note(primary_checkpoint, checkpoint_candidates) if primary_checkpoint else None
    return {
        "checkpoint_candidates": checkpoint_candidates,
        "checkpoint_choice_note": shadowed,
        "model_name": model_name or frozen_profile.get("run_name", "model"),
        "run_name": frozen_profile.get("run_name", "model"),
        "config_name": frozen_profile.get("config_name", ""),
        "training_profile": frozen_profile,
        "derivation_profile": validate_derivation_profile(derivation_profile),
        "training_source_kind": source_summary["kind"],
        "training_source_pk": source_summary["pk"],
        "training_source_uuid": source_summary["uuid"],
        "training_source_label": source_summary["label"],
        "training_source_summary": source_summary,
        "primary_checkpoint": primary_checkpoint,
        "training_summary": summary,
        # The split that chose the kept checkpoint. It was on no model node in
        # any form — only in the workchain inputs and in an archived
        # `custom.val_data_dir=.../runtime/<uuid>` override, so a model's
        # validation set could not be named without walking the provenance graph.
        **(
            {"validation_source_summary": build_training_source_summary(validation_source)}
            if validation_source is not None
            else {}
        ),
    }


def eval_metrics(node) -> dict:
    """The metrics payload of ANY eval-shaped node, one entry point.

    The friction was real but misdiagnosed as version drift: every vintage in
    the database uses the same port names (measured 2026-08-09: 205 calcjobs
    all expose `metrics`; all 340 eval workchains expose `metrics`,
    `eval_summary` AND `evaluation`). The probing pain is that there are THREE
    similarly-named outputs and nothing said which holds the numbers:
    `metrics` is the flat payload, `eval_summary` is the adapter's run summary,
    `evaluation` is the record node whose metadata embeds a copy.

    Accepts a PinoModelEvaluationData, an eval CalcJob or WorkChain, or a pk,
    and returns the flat metrics dict ({} when the eval produced none).
    """
    from aiida_pino.data.pino_model_evaluation import PinoModelEvaluationData

    if isinstance(node, int):
        node = orm.load_node(node)
    if isinstance(node, PinoModelEvaluationData):
        return dict(node.get_evaluation_metadata().get("metrics") or {})
    for label in ("metrics",):
        try:
            return dict(getattr(node.outputs, label).get_dict())
        except Exception:  # noqa: BLE001 - port absent; fall through
            continue
    try:
        return dict(node.outputs.evaluation.get_evaluation_metadata().get("metrics") or {})
    except Exception:  # noqa: BLE001
        return {}


def resolution_mismatch_warning(training_stride, eval_stride) -> str | None:
    """A warning when an eval scores a model at a resolution it did not train at.

    Warns for EVERY architecture and refuses for none. Measured on three, same weights each,
    scored at the model's own stride and at full resolution (2026-09-05..07,
    read from each eval's recorded `kelvin` block):

        RIGNO   7.767 -> 8.607 K substrate  (x1.11)   IoU 0.926 -> 0.923
        GAOT    5.039 -> 11.379 K           (x2.26)   IoU 0.931 -> 0.806
        FNO     5.361 -> 55.437 K           (x10.3)   IoU 0.935 -> 0.541

    A spectrum, not a spectral/graph split: GAOT is a GRAPH model and still degrades by 2.3x,
    so the tempting rule "refuse for spectral, warn for the rest" would have silently passed
    it. A refusal needs a threshold somewhere in that spread, and picking one is a project
    decision -- the domain-extent guard's FNO exemption is what a guessed threshold costs.

    Unknown on either side stays silent. Most stored models carry no `training_stride`, and a
    warning on every one of them is a warning nobody reads.
    """

    def _as_triple(value):
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                return [int(v) for v in value]
            except (TypeError, ValueError):
                return None
        return None

    trained, scored = _as_triple(training_stride), _as_triple(eval_stride)
    if trained is None or scored is None or trained == scored:
        return None
    return (
        f"RESOLUTION MISMATCH: this model trained at stride {trained} and is being scored at "
        f"{scored}. Nothing will stop the run -- it exits 0 and returns numbers that look "
        f"real. How much they move is architecture-dependent and was measured across a wide "
        f"spread on the same weights: RIGNO x1.11 (within seed noise), GAOT x2.3 with IoU "
        f"-0.13, FNO x10.3 with IoU -0.39, and the FNO case INVERTED the ranking of the two "
        f"arms being compared. Score at {trained} to compare against this model's own "
        f"history, or state the resolution beside any number you publish from this run."
    )


def served_grid_facts(metrics: dict | None, eval_summary: dict | None = None) -> dict:
    """What the run ACTUALLY served, captured from the loader rather than recomputed.

    `derivation_profile` on an eval node is the derivation REQUESTED for the dataset, and the
    crop that applies can come from somewhere else entirely: the eval inherits the model's
    training env, and `ctx.demoted_env` is overlaid ON TOP after the merge, so a crop the
    submitter explicitly blanked is put back. On nine laboratory-frame evals of one test set every node
    recorded `crop_z_top = 0` while three served 90x40x30, and the only field that
    distinguished them -- `PINO_LAZY_CROP_Z_TOP = ''` on exactly those three -- is how the
    loader spells NO crop. The one discriminating value asserted the opposite of the truth
    (2026-09-09).

    lpno `4722344` made the store report what it served; this consumes it. A
    capture beats a computation, which is the rule the eval_stride labels were settled by four
    days earlier -- a computed label disagreed with counted cells and the count won.

    Returns None for every field when the capture is absent: evals predating `4722344` have
    only their log, and an unknown crop must never read as "no crop".
    """
    raw = None
    for source in (metrics or {}, eval_summary or {}):
        candidate = source.get("served_grid") if isinstance(source, dict) else None
        if isinstance(candidate, dict) and candidate:
            raw = candidate
            break
    facts: dict = {"served_grid": None, "eval_crop_z_top": None, "eval_stride": None}
    if raw is None:
        return facts
    try:
        crop = int(raw["crop_z_top"])
        stride = [int(v) for v in raw["stride"]]
        nz = int(raw["nz"])
    except (KeyError, TypeError, ValueError):
        return facts
    if len(stride) != 3 or nz <= 0:
        return facts
    facts["served_grid"] = dict(raw)
    facts["eval_crop_z_top"] = crop
    facts["eval_stride"] = stride
    return facts


def _model_training_stride(model) -> list[int] | None:
    """The stride the model TRAINED at, from its own recorded extra.

    Read rather than re-derived: `pino_model` already resolves it (including the
    `training_stride_source` it came from), and deriving it again from the training env would
    miss the bake path, where the stride lives in the resolved collection and the env is empty.
    """
    try:
        value = model.base.extras.all.get("training_stride")
    except Exception:  # noqa: BLE001 - a model without the extra is unknown, not [1,1,1]
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError):
            return None
    return None


def _eval_effective_stride(
    evaluation_profile, model, derivation_profile, dataset_source=None, effective_env=None
) -> list[int] | None:
    """The stride the eval loader actually served: BAKED x env, COMPOSED.

    The env travels either in the evaluation profile's own overrides or,
    absent that, in the model's frozen training profile -- the eval inherits
    the training env unless overridden. The env stride then applies ON TOP of
    whatever decimation is baked into the served dataset, so the two multiply;
    an earlier version let env "win over" derivation, which reports [2,2,1]
    for the double-stride misconfiguration ([2,2,1] data + [2,2,1] loader =
    [4,4,1] served) instead of exposing it. `effective_stride` in
    utils.profiles is the single implementation of that composition.

    The first factor is what the SERVED dataset bakes, never the REQUESTED
    `derivation_profile`. Absent an explicit profile, the eval builds one from the
    model's own training env -- so the same stride stands on both sides and multiplies
    with itself. That mislabelled 20 evals since 2026-08-01 as [4,4,1] from a model
    whose env is `2,2,1`: `beambox_n` counted 88552 for those,
    identical to the [2,2,1] rows on the same split against 397546 at [1,1,1], where a
    real 16x decimation would read ~22k. `eval_stride` is computed and `beambox_n` is
    counted; the count decides. Serving was never wrong, only the label -- and the label
    is what league tables and "scored at two resolutions?" read.

    `dataset_source` defaults to None for callers that cannot supply it, and that yields
    the unknown None rather than a guess, for the same reason as the except branch.
    """
    from aiida_pino.utils.profiles import effective_stride

    # PRESENT-but-blank is an explicit "no stride", not "unspecified". Testing the value's
    # truthiness treated `PINO_LAZY_STRIDE: ""` as absent and inherited the model's stride,
    # which mislabelled one eval [2,2,1] on a run that counted 397546 cells (full-res on
    # that split; strided is 88552) and `true_T_max_K` 4118 against 3853 strided.
    # `effective_env` is what the RUN actually had, after the workchain's full-res opt-in
    # dropped an inherited loader stride. Passed rather than inferred: the tempting inference
    # ("derivation [1,1,1] against a striding model must mean the opt-in fired") is right for
    # new evals and WRONG for every eval predating the opt-in -- one eval counted 88552 with
    # exactly that shape -- so it would corrupt any recompute over history.
    if effective_env is not None:
        env_profile = {"env": dict(effective_env)}
    else:
        env_profile = evaluation_profile or {}
        if "PINO_LAZY_STRIDE" not in (env_profile.get("env") or {}):
            env_profile = model.get_model_metadata().get("training_profile") or {}
    if dataset_source is None:
        return None
    try:
        served = (dataset_source.get_dataset_metadata() or {}).get("derivation_profile") or {}
        baked = [int(v) for v in (served.get("stride") or [1, 1, 1])]
        if baked != [1, 1, 1]:
            # The served variant already carries the decimation, and
            # `apply_dataset_layout_to_training_profile` popped the loader stride precisely so
            # it is not applied twice -- so on this path the two do NOT compose. Proven by
            # counting rather than reasoning: one eval read a baked [2,2,1] variant with
            # `PINO_LAZY_STRIDE=2,2,1` still in its profile and scored `beambox_n` 88552,
            # identical to an eval that read the FULL-RES parent with the same env. A
            # genuine [4,4,1] would be ~22k. Composing them here is what mislabelled 20 evals.
            return baked
        return list(effective_stride({"derivation_profile": served}, dict(env_profile)))
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError):
        # Unknown is None, never a guessed [1,1,1] -- sizing and league labels
        # must not treat "could not tell" as "full resolution".
        return None


def _member_uuids(summary: dict | None) -> set:
    """The uuid set of a source summary's members.

    uuid, never pk: every pk changed in the 2026-07-21 PostgreSQL migration, so a
    pk-keyed comparison silently matches unrelated nodes rather than failing.
    """
    out = set()
    for m in ((summary or {}).get("members") or []):
        if isinstance(m, dict) and m.get("uuid"):
            out.add(m["uuid"])
        elif isinstance(m, str):
            out.add(m)
    return out


def payload_keys(uuids) -> set:
    """Identity keys for a set of simulation uuids, twin-collapsed.

    Two nodes with the same `content_hash` hold the same simulation, and
    datasets reference whichever twin existed when they were built (68 of 70
    duplicated payloads have their twins split across different datasets). A
    raw-uuid comparison therefore misses a case that is present through its
    twin. Keyed on content_hash when the node has one, on the uuid otherwise.
    """
    uuids = {u for u in uuids if u}
    if not uuids:
        return set()
    rows = orm.QueryBuilder().append(
        orm.Data,
        filters={"uuid": {"in": list(uuids)}},
        project=["uuid", "extras.content_hash"],
    ).all()
    keys = {h or u for u, h in rows}
    # uuids that resolve to no node (deleted sims) still count as themselves
    keys |= uuids - {u for u, _ in rows}
    return keys


def train_eval_overlap(model_metadata: dict, dataset_summary: dict) -> int | None:
    """How many scored cases the model was trained on. None when unknowable.

    A score computed on the model's own training data is not a held-out result,
    and nothing distinguished the two: QueryBuilder has no array-intersection
    predicate, so "show me only the held-out scores" was not expressible at all.
    Without this check, an evaluation whose cases overlap the model's training
    set sits beside the held-out scores and looks exactly like one of them.

    Returns None -- not 0 -- when either member list is empty, because "no
    overlap" and "no evidence" must not read the same. 2 of 208 are in that
    state: their training source node was deleted AND no frozen member list
    survives. A dangling training_source_uuid alone is not enough to be unknown;
    two evals point at deleted sources but still carry the frozen list
    that answers the question.
    """
    train = _member_uuids(model_metadata.get("training_source_summary"))
    scored = _member_uuids(dataset_summary)
    if not train or not scored:
        return None
    return len(train & scored)


def build_evaluation_metadata(
    model: PinoModelData,
    dataset_source: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData,
    evaluation_profile: dict[str, Any],
    metrics: orm.Dict,
    eval_summary: orm.Dict,
    derivation_profile: dict[str, Any],
    effective_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct structured evaluation-record metadata.

    ``effective_env`` is the env the run actually used; without it the stride label is
    reconstructed from the profiles and cannot see a stride the workchain dropped.
    """
    from aiida_pino.console.model_metrics import build_metric_units_with_geometry

    _served = served_grid_facts(metrics.get_dict() if hasattr(metrics, "get_dict") else metrics,
                                eval_summary.get_dict() if hasattr(eval_summary, "get_dict") else eval_summary)
    source_summary = build_training_source_summary(dataset_source)
    model_metadata = model.get_model_metadata()
    evaluation_name = (
        evaluation_profile.get("run_name")
        or f"{model.get_model_name()}_on_{source_summary['label']}"
    )
    return {
        "evaluation_name": evaluation_name,
        "model_pk": model.pk,
        "model_uuid": model.uuid,
        "model_name": model.get_model_name(),
        "dataset_source_kind": source_summary["kind"],
        "dataset_source_pk": source_summary["pk"],
        "dataset_source_uuid": source_summary["uuid"],
        "dataset_source_label": source_summary["label"],
        "dataset_source_summary": source_summary,
        # Held-out or not, as one queryable number. Both member lists are already
        # in local scope here; SQL cannot intersect them, so do it once at write.
        "train_overlap_n": train_eval_overlap(model_metadata, source_summary),
        "scored_n": len(_member_uuids(source_summary)) or None,
        "config_name": model_metadata.get("config_name", ""),
        "training_source_kind": model_metadata.get("training_source_kind", ""),
        # Copied from the model's stored metadata, which for anything created
        # before the 2026-07-21 migration holds a pk that now names a different
        # node. Resolve it from the uuid sitting beside it instead.
        "training_source_pk": pk_for_uuid(
            model_metadata.get("training_source_uuid"), model_metadata.get("training_source_pk")
        ),
        "training_source_uuid": model_metadata.get("training_source_uuid", ""),
        "evaluation_profile": validate_evaluation_profile(evaluation_profile),
        "derivation_profile": validate_derivation_profile(derivation_profile),
        # The grid this eval actually scored, as one field. It used to be spread
        # across `training_profile.env.PINO_LAZY_STRIDE` (loader stride, applied
        # on top) and the dataset's `derivation.stride`, with nothing on the
        # eval stating the result -- so "has any checkpoint been evaluated at
        # two resolutions?" meant loading every eval CalcJob and reconstructing
        # pairs by hand (2026-08-09). Same disease as the model-side F2.
        # MEASURED first: `served_grid` is what the loader reported serving, so it overrides
        # the computed stride below wherever it exists. Absent (pre-4722344 evals) the
        # computed value stands, and the crop stays None rather than reading as zero.
        **{k: v for k, v in _served.items() if k in ("served_grid", "eval_crop_z_top")},
        "eval_stride": _served["eval_stride"] or _eval_effective_stride(
            evaluation_profile, model, derivation_profile, dataset_source, effective_env
        ),
        # Beside it, so the pair can be compared without loading the model node.
        "training_stride": _model_training_stride(model),
        "metrics": metrics.get_dict(),
        # The unit travels with the number: metrics are normalised, and until
        # this was recorded the Kelvin factor lived only in the *model's*
        # archived Hydra config (absent for 19 of 122 models).
        "metric_units": build_metric_units_with_geometry(
            model, metrics.get_dict(), derivation_profile
        ),
        "eval_summary": eval_summary.get_dict(),
    }


def folder_data_from_remote(remote: orm.RemoteData) -> orm.FolderData:
    """Materialize a `RemoteData` tree into a stored local `FolderData` node."""
    with tempfile.TemporaryDirectory() as tmp_path:
        tmpdir = Path(tmp_path)
        with remote.get_authinfo().get_transport() as transport:
            transport.gettree(remote.get_remote_path(), str(tmpdir))
        return orm.FolderData(tree=str(tmpdir)).store()
