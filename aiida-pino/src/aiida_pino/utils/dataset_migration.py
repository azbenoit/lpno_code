"""Helpers for migrating legacy batched datasets into simulation-wise collections."""

from __future__ import annotations

from typing import Any

import numpy as np
from aiida import orm

from aiida_pino.data import PinoDatasetCollectionData, PinoDatasetData, PinoDatasetUnionData, PinoSimulationData
from .dataset_derivation import build_derived_collection_metadata, resolve_base_dataset
from .model_registry import build_collection_metadata_from_simulations, get_or_create_dataset_collection


def _legacy_group_entries(metadata: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    grouped = list(metadata.get("grouped_files", []))
    if grouped and len(grouped) == batch_size:
        return grouped
    sample_metadata = list(metadata.get("sample_metadata", []))
    if sample_metadata and len(sample_metadata) == batch_size:
        return [
            {
                "group": entry.get("group", f"simulation_{index:04d}"),
                "files": [entry],
            }
            for index, entry in enumerate(sample_metadata)
        ]
    return [{"group": f"simulation_{index:04d}", "files": []} for index in range(batch_size)]


def _time_values_from_group_entry(group_entry: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for file_entry in group_entry.get("files", []):
        params = file_entry.get("params", {}) or {}
        value = params.get("time_s")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _find_simulation_by_fingerprint(fingerprint: str) -> PinoSimulationData | None:
    builder = orm.QueryBuilder()
    builder.append(PinoSimulationData, filters={"extras.simulation_fingerprint": {"==": fingerprint}}, project=["*"])
    builder.order_by({PinoSimulationData: {"ctime": "desc"}})
    return builder.first(flat=True)


def _mark_legacy_node_migrated(node: orm.Node, collection: PinoDatasetCollectionData) -> None:
    node.base.extras.set("legacy_format", True)
    node.base.extras.set("superseded", True)
    node.base.extras.set("migrated_to_uuid", collection.uuid)
    node.base.extras.set("migrated_to_pk", collection.pk)
    node.base.extras.set("hidden_from_selectors", True)


def migrate_legacy_dataset_node(node: PinoDatasetData) -> PinoDatasetCollectionData:
    """Migrate one legacy native dataset into a simulation-wise collection."""
    migrated_uuid = node.base.extras.get("migrated_to_uuid", None)
    if migrated_uuid:
        migrated = orm.load_node(migrated_uuid)
        if isinstance(migrated, PinoDatasetCollectionData):
            return migrated

    metadata = node.get_dataset_metadata()
    role = str(metadata.get("dataset_role", "") or "base")
    if role == "derived":
        base_dataset = resolve_base_dataset(node)
        if not isinstance(base_dataset, PinoDatasetData):
            raise ValueError(f"Could not resolve base dataset for legacy derived dataset pk={node.pk}.")
        migrated_base = migrate_legacy_dataset_node(base_dataset)
        derived_metadata = build_derived_collection_metadata(
            migrated_base,
            metadata.get("derivation_profile", {}),
            parent_collection=migrated_base,
        )
        derived_metadata["composition_kind"] = "migration"
        derived_metadata["source_folder_pk"] = metadata.get("source_folder_pk")
        derived_metadata["source_folder_uuid"] = metadata.get("source_folder_uuid")
        collection, _ = get_or_create_dataset_collection(migrated_base.iter_member_simulations(), metadata=derived_metadata)
        _mark_legacy_node_migrated(node, collection)
        return collection

    array_payloads = {name: array for name, array in node.iter_original_arrays()}
    first_shape = next(iter(array_payloads.values())).shape if array_payloads else ()
    if not first_shape:
        raise ValueError(f"Legacy dataset pk={node.pk} has no arrays to migrate.")
    batch_size = int(first_shape[0])
    groups = _legacy_group_entries(metadata, batch_size)
    if len(groups) != batch_size:
        raise ValueError(f"Legacy dataset pk={node.pk} cannot be split reliably into simulations.")

    source = dict(node.source) if getattr(node, "source", None) else None
    simulations: list[PinoSimulationData] = []
    for index, group in enumerate(groups):
        sim_arrays = {name: np.asarray(array[index], dtype=np.float32) for name, array in array_payloads.items()}
        time_values = _time_values_from_group_entry(group)
        member_shape = next(iter(sim_arrays.values())).shape
        if len(member_shape) >= 5 and len(time_values) != int(member_shape[0]):
            # Legacy group metadata can carry one file entry per simulation
            # rather than one per timestep; a mismatched time list makes
            # PinoSimulationData.from_arrays raise, so drop it and let the
            # timestep count be inferred from the array shape.
            time_values = []
        simulation_metadata = {
            "source_type": metadata.get("source_type", PinoSimulationData.DEFAULT_SOURCE_TYPE),
            "source_origin_kind": "migrated_legacy_dataset",
            "source_folder_name": metadata.get("source_folder_name", node.label or "dataset"),
            "source_uri": metadata.get("source_uri", ""),
            "grid_params": metadata.get("grid_params", {}),
            "requested_fields": metadata.get("requested_fields", list(sim_arrays)),
            "source_group": group.get("group", f"simulation_{index:04d}"),
            "group_name": group.get("group", f"simulation_{index:04d}"),
            "relative_paths": [entry.get("relative_path", "") for entry in group.get("files", [])],
            "sample_metadata": list(group.get("files", [])),
            "time_values_s": time_values,
        }
        candidate = PinoSimulationData.from_arrays(sim_arrays, metadata=simulation_metadata, source=source)
        existing = _find_simulation_by_fingerprint(candidate.get_simulation_fingerprint())
        if existing is not None:
            simulations.append(existing)
        else:
            simulations.append(candidate.store())

    collection_metadata = build_collection_metadata_from_simulations(
        simulations,
        collection_name=metadata.get("source_folder_name", node.label or "dataset"),
        composition_kind="migration",
        source_uri=metadata.get("source_uri", ""),
        extra_metadata={
            "source_folder_pk": metadata.get("source_folder_pk"),
            "source_folder_uuid": metadata.get("source_folder_uuid"),
            "legacy_dataset_uuid": node.uuid,
            "legacy_dataset_pk": node.pk,
            "legacy_dataset_identifier": metadata.get("dataset_identifier", node.label or ""),
            "sample_metadata": list(metadata.get("sample_metadata", [])),
        },
    )
    collection, _ = get_or_create_dataset_collection(simulations, metadata=collection_metadata)
    _mark_legacy_node_migrated(node, collection)
    return collection


def migrate_legacy_union_node(node: PinoDatasetUnionData) -> PinoDatasetCollectionData:
    """Migrate one legacy union into a single composed dataset collection."""
    migrated_uuid = node.base.extras.get("migrated_to_uuid", None)
    if migrated_uuid:
        migrated = orm.load_node(migrated_uuid)
        if isinstance(migrated, PinoDatasetCollectionData):
            return migrated

    simulations: list[PinoSimulationData] = []
    for member in node.iter_member_datasets():
        migrated_collection = migrate_legacy_dataset_node(member)
        simulations.extend(migrated_collection.iter_member_simulations())
    collection_metadata = build_collection_metadata_from_simulations(
        simulations,
        collection_name=node.get_union_name(),
        composition_kind="union_migrated",
        extra_metadata={
            "legacy_union_uuid": node.uuid,
            "legacy_union_pk": node.pk,
            "legacy_union_identifier": node.get_union_identifier(),
        },
    )
    collection, _ = get_or_create_dataset_collection(simulations, metadata=collection_metadata)
    _mark_legacy_node_migrated(node, collection)
    return collection


def migrate_all_legacy_datasets(*, limit: int | None = None) -> dict[str, Any]:
    """Migrate all legacy native datasets and unions reachable in the current profile."""
    builder = orm.QueryBuilder()
    builder.append(PinoDatasetData, project=["*"])
    builder.order_by({PinoDatasetData: {"ctime": "asc"}})
    legacy_datasets = builder.all(flat=True)
    if limit is not None:
        legacy_datasets = legacy_datasets[: int(limit)]
    base_nodes = []
    derived_nodes = []
    for node in legacy_datasets:
        role = str(node.get_dataset_metadata().get("dataset_role", "") or "base")
        (derived_nodes if role == "derived" else base_nodes).append(node)

    migrated_collections: list[PinoDatasetCollectionData] = []
    for node in [*base_nodes, *derived_nodes]:
        migrated_collections.append(migrate_legacy_dataset_node(node))

    union_builder = orm.QueryBuilder()
    union_builder.append(PinoDatasetUnionData, project=["*"])
    union_builder.order_by({PinoDatasetUnionData: {"ctime": "asc"}})
    legacy_unions = union_builder.all(flat=True)
    if limit is not None:
        legacy_unions = legacy_unions[: int(limit)]
    migrated_union_collections = [migrate_legacy_union_node(node) for node in legacy_unions]

    # Two legacy nodes can migrate onto one collection — collections are
    # content-addressed, so a union of a single dataset, or a "derived" node
    # with no derivation, is the same content as its base. Dedupe by uuid and
    # keep first-seen order, so the count and the pk list cannot disagree.
    seen: set[str] = set()
    migrated: list = []
    for node in [*migrated_collections, *migrated_union_collections]:
        if node.uuid not in seen:
            seen.add(node.uuid)
            migrated.append(node)
    return {
        "legacy_dataset_count": len(legacy_datasets),
        "legacy_union_count": len(legacy_unions),
        "migrated_collection_count": len(migrated),
        "migrated_collection_pks": [node.pk for node in migrated],
    }
