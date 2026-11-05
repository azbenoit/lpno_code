"""Helpers for dataset and folder-like AiiDA data nodes."""

from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from aiida import orm

from aiida_pino.data import (
    PinoDatasetCollectionData,
    PinoDatasetData,
    PinoDatasetUnionData,
    PinoSimulationData,
    is_canonical_metadata,
    safe_attribute_get,
    validate_source_type,
)
from .model_registry import recent_collection_nodes

FOLDERLIKE_DATA = (orm.FolderData, orm.RemoteData)
DATASET_INPUT_DATA = (PinoDatasetCollectionData,)


def validate_remote_data_for_code(node: orm.FolderData | orm.RemoteData, code: orm.AbstractCode) -> None:
    """Validate that a `RemoteData` input can be staged by the selected code."""
    if not isinstance(node, orm.RemoteData):
        return

    computer = getattr(code, "computer", None)
    if computer is None:
        raise ValueError("RemoteData inputs require a Code with an associated Computer.")
    if node.computer is None:
        raise ValueError("RemoteData input is missing a Computer.")
    if node.computer.uuid != computer.uuid:
        raise ValueError(
            f"RemoteData input computer `{node.computer.label}` does not match code computer `{computer.label}`."
        )


def normalize_dataset_collection_node(node: orm.Data, *, label: str = "dataset") -> PinoDatasetCollectionData:
    """Reload and validate a public native dataset collection input.

    AiiDA can occasionally deserialize custom data nodes as generic ``Data`` at
    nested process boundaries. Ports therefore accept ``orm.Data`` and this
    helper performs the concrete type check after reloading by PK.
    """
    if isinstance(node, PinoDatasetCollectionData):
        return node

    pk = getattr(node, "pk", None)
    if pk is not None:
        loaded = orm.load_node(pk)
        if isinstance(loaded, PinoDatasetCollectionData):
            return loaded

    raise ValueError(
        f"`{label}` must be a native PinoDatasetCollectionData node; "
        f"got `{node.__class__.__name__}` (pk={pk})."
    )


def validate_dataset_input_port(node: orm.Data, _port=None) -> str | None:
    """Port validator: accept a `FolderData` or a native dataset collection.

    The train/eval/kfold ports are typed `orm.Data` so that a collection which
    AiiDA deserialized as generic `Data` at a nested boundary still passes. This
    validator runs at submission (where the top-level type is concrete) and
    rejects a genuinely wrong type with a defined message, instead of letting
    ``normalize_dataset_collection_node`` raise inside ``prepare_for_submission``
    and leave the CalcJob EXCEPTED.
    """
    if isinstance(node, orm.FolderData):
        return None
    try:
        normalize_dataset_collection_node(node)
    except ValueError as exc:
        return str(exc)
    return None


def validate_source_type_port(node: orm.Str, _port=None) -> str | None:
    """Port validator: reject a `source_type` outside the supported set at submission."""
    try:
        validate_source_type(str(node.value))
    except ValueError as exc:
        return str(exc)
    return None


def find_remote_data(computer: orm.Computer, remote_path: str | Path) -> orm.RemoteData | None:
    """Return an existing `RemoteData` node for the same computer/path when present."""
    resolved_path = str(Path(remote_path).expanduser().resolve())
    builder = orm.QueryBuilder()
    builder.append(orm.RemoteData, filters={"attributes.remote_path": resolved_path}, project=["*"])
    for node in builder.all(flat=True):
        if node.computer and node.computer.uuid == computer.uuid:
            return node
    return None


def get_or_create_remote_data(
    remote_path: str | Path,
    computer: orm.Computer,
    *,
    kind: str,
    description: str,
) -> tuple[orm.RemoteData, bool]:
    """Create or reuse a stored `RemoteData` node for an external folder."""
    resolved_path = Path(remote_path).expanduser().resolve()
    existing = find_remote_data(computer, resolved_path)
    if existing is not None:
        return existing, False

    node = orm.RemoteData(remote_path=str(resolved_path), computer=computer)
    node.label = f"{kind}:{resolved_path.name or resolved_path.anchor}"
    node.source = {
        "uri": f"file://{resolved_path}",
        "description": description,
        "extras": {"aiida_pino_kind": kind},
    }
    node.store()
    node.base.extras.set("aiida_pino_kind", kind)
    node.base.extras.set("aiida_pino_source_path", str(resolved_path))
    return node, True


#: Extras marker for a runtime cache copied onto a compute box by `stage-remote`.
RUNTIME_CACHE_KIND = "runtime_cache"


def find_remote_runtime_cache(computer: orm.Computer, cache_name: str) -> orm.RemoteData | None:
    """The pre-staged runtime cache called `cache_name` on `computer`, if any.

    Looked up by NAME rather than by path, deliberately. The remote cache root
    belongs to the compute box and is not knowable here, and guessing it is how
    you get a confident wrong answer on a box that shares `/home/<user>` with
    the primary — which the first one does. `find_remote_data` cannot be reused
    for this: it matches on a path that it first resolves against the LOCAL
    filesystem (`data_nodes.py` `get_or_create_remote_data`), which is
    meaningless for a directory that only exists somewhere else.

    Never opens a transport, so this is safe at `prepare_for_submission` and
    testable without a reachable box.
    """
    builder = orm.QueryBuilder()
    builder.append(
        orm.RemoteData,
        filters={
            "extras.aiida_pino_kind": RUNTIME_CACHE_KIND,
            "extras.aiida_pino_cache_name": str(cache_name),
            "extras.superseded": {"!==": True},
        },
        project=["*"],
    )
    builder.order_by({orm.RemoteData: {"ctime": "desc"}})
    for node in builder.all(flat=True):
        if node.computer and node.computer.uuid == computer.uuid:
            return node
    return None


def register_remote_runtime_cache(
    computer: orm.Computer,
    *,
    cache_name: str,
    remote_path: str,
    verified: dict | None = None,
) -> orm.RemoteData:
    """Record that `cache_name` is staged at `remote_path` on `computer`.

    `remote_path` is stored VERBATIM. It is the compute box's path, so
    `expanduser()` would expand this machine's home and `resolve()` would follow
    this machine's symlinks — both silently plausible on a box with the same
    home layout, and both wrong.

    Re-registering the same name at a new path supersedes the old record rather
    than leaving two answers to one question.
    """
    remote_path = str(remote_path)
    existing = find_remote_runtime_cache(computer, cache_name)
    if existing is not None:
        if existing.get_remote_path() == remote_path:
            if verified is not None:
                existing.base.extras.set("aiida_pino_cache_verified", verified)
            return existing
        existing.base.extras.set("superseded", True)

    node = orm.RemoteData(remote_path=remote_path, computer=computer)
    node.label = f"{RUNTIME_CACHE_KIND}:{cache_name}"
    node.store()
    node.base.extras.set("aiida_pino_kind", RUNTIME_CACHE_KIND)
    node.base.extras.set("aiida_pino_cache_name", str(cache_name))
    if verified is not None:
        node.base.extras.set("aiida_pino_cache_verified", verified)
    return node


def describe_folderlike_node(node: orm.FolderData | orm.RemoteData) -> str:
    """Create a compact label for folder-like nodes in the UI."""
    label = node.label or "-"
    kind = node.base.extras.get("aiida_pino_kind", "")
    suffix = f" [{kind}]" if kind else ""
    if isinstance(node, orm.RemoteData):
        location = node.get_remote_path()
        return f"{node.pk} | RemoteData | {label}{suffix} | {location}"
    source = node.source.get("uri") if getattr(node, "source", None) else None
    if source:
        return f"{node.pk} | FolderData | {label}{suffix} | {source}"
    return f"{node.pk} | FolderData | {label}{suffix}"


def recent_folderlike_nodes(limit: int = 50) -> list[orm.FolderData | orm.RemoteData]:
    """Return recent `FolderData` and `RemoteData` nodes for UI selection."""
    nodes: list[orm.FolderData | orm.RemoteData] = []
    for cls in FOLDERLIKE_DATA:
        builder = orm.QueryBuilder()
        builder.append(cls, project=["*"])
        builder.order_by({cls: {"ctime": "desc"}})
        nodes.extend(builder.limit(limit).all(flat=True))
    nodes.sort(key=lambda node: node.ctime, reverse=True)
    return nodes[:limit]


def parse_pk_from_option(value: str) -> int:
    """Parse a leading PK from a UI selection option."""
    try:
        return int(value.split("|", 1)[0].strip())
    except (IndexError, ValueError) as exc:
        raise ValueError("Select a valid AiiDA node entry.") from exc


def folderlike_type_tuple() -> tuple[type[Any], ...]:
    """Return the tuple form used by AiiDA input specs."""
    return FOLDERLIKE_DATA


def format_bytes(num_bytes: int | float | None) -> str:
    """Format a byte count for human-readable display."""
    if not num_bytes:
        return "0 B"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{int(num_bytes)} B"


def repository_disk_usage_bytes(node: orm.Node) -> tuple[int, int]:
    """Return the stored repository size and file count for a node."""
    repository = getattr(node.base, "repository", None)
    if repository is None:
        return 0, 0

    file_paths: list[str] = []
    try:
        for root, _, files in repository.walk():
            root_path = PurePosixPath(root)
            for filename in files:
                relative = root_path / filename if str(root_path) != "." else PurePosixPath(filename)
                file_paths.append(str(relative))
    except Exception:
        try:
            file_paths = list(repository.list_object_names())
        except Exception:
            return 0, 0

    total_bytes = 0
    counted_files = 0
    for relative_path in file_paths:
        try:
            total_bytes += int(repository.get_object_size(relative_path))
            counted_files += 1
        except Exception:
            continue
    return total_bytes, counted_files


def dataset_collection_disk_usage_bytes(node: PinoDatasetCollectionData) -> tuple[int, int]:
    """Return aggregate repository usage for a simulation-backed collection.

    Members are referenced by UUID only (no provenance link), so a member can be
    deleted or an imported archive can lack its members, leaving dangling UUIDs.
    Skip anything that fails to load or is not a simulation instead of raising,
    so one bad member does not blank the whole dataset browser.
    """
    total_bytes = 0
    total_files = 0
    for identifier in node.get_member_simulation_uuids():
        try:
            simulation = orm.load_node(identifier)
        except Exception:
            continue  # deleted/dangling member UUID
        if not isinstance(simulation, PinoSimulationData):
            continue  # imported/mistyped member
        simulation_bytes, simulation_files = repository_disk_usage_bytes(simulation)
        total_bytes += simulation_bytes
        total_files += simulation_files
    return total_bytes, total_files


def describe_dataset_node(node: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData | orm.FolderData) -> str:
    """Create a compact label for dataset selectors in the UI."""
    if isinstance(node, PinoDatasetCollectionData):
        metadata = node.get_dataset_metadata()
        identifier = metadata.get("dataset_identifier", node.label or "-")
        role = metadata.get("dataset_role", "base")
        source_type = metadata.get("source_type", "flow3d")
        folder_name = metadata.get("source_folder_name", "-")
        return f"{node.pk} | PinoDatasetCollectionData | {identifier} | {role} | {source_type} | {folder_name}"
    if isinstance(node, PinoDatasetUnionData):
        return (
            f"{node.pk} | PinoDatasetUnionData | {node.get_union_identifier()} | "
            f"union | {node.get_total_samples()} samples"
        )
    if isinstance(node, PinoDatasetData):
        source_type = safe_attribute_get(node, "source_type", PinoDatasetData.DEFAULT_SOURCE_TYPE)
        identifier = safe_attribute_get(node, "dataset_identifier", node.label or "-")
        folder_name = safe_attribute_get(node, "source_folder_name", "-")
        role = safe_attribute_get(node, "dataset_role", "legacy")
        return f"{node.pk} | PinoDatasetData | {identifier} | {role} | {source_type} | {folder_name}"

    label = node.label or "-"
    kind = node.base.extras.get("aiida_pino_kind", "")
    suffix = f" [{kind}]" if kind else ""
    return f"{node.pk} | FolderData | {label}{suffix}"


def active_dataset_nodes(include_superseded: bool = False) -> list[PinoDatasetCollectionData]:
    """Every non-superseded dataset collection, newest first — no window.

    Browsing is not tailing. A recency window silently hid whole datasets once
    the collection count outgrew it: on 2026-07-30, with 1064 collections, all
    four `core` reference sets were invisible to every listing, search and
    folder view because they were among the oldest. The active population is
    small (hundreds) and each row is a cheap attribute read, so the caller
    filters first and truncates last — never the reverse.

    `include_superseded` brings back the retired ones. Supersession marks about
    half the collections, and with no way to see them a dataset that was
    replaced is indistinguishable from one that never existed.
    """
    query = orm.QueryBuilder().append(
        PinoDatasetCollectionData,
        filters=None if include_superseded else {"extras.superseded": {"!==": True}},
        project=["*"],
        tag="collection",
    )
    query.order_by({"collection": {"ctime": "desc"}})
    return list(query.all(flat=True))


def active_dataset_records(
    include_superseded: bool = False,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """Every listable collection as a plain dict, from **one** projected query.

    `active_dataset_nodes()` loads full ORM nodes and then reads
    `dataset_metadata`, `description`, extras and the member list off each one.
    At 734 collections that is 78-92 s; the same population projected in a
    single query is sub-second, which is what `dataset_options()` has always
    done for the pickers. The browser needs more fields, not a different method.

    Returns dicts with the node's own values; no lazy loading, so a caller
    cannot accidentally reintroduce per-row queries.
    """
    keys = (
        "pk", "uuid", "label", "description", "ctime", "extras", "metadata", "member_uuids",
    )
    query = orm.QueryBuilder().append(
        PinoDatasetCollectionData,
        filters=None if include_superseded else {"extras.superseded": {"!==": True}},
        project=[
            "id", "uuid", "label", "description", "ctime", "extras",
            "attributes.dataset_metadata", "attributes.member_simulation_uuids",
        ],
        tag="collection",
    )
    query.order_by({"collection": {"ctime": "desc"}})
    # A WINDOW, when the caller only needs one page. `attributes.dataset_metadata` and
    # `attributes.member_simulation_uuids` are large per row -- a member list runs to
    # 1601 uuids -- so the whole population costs 15.7 s at 2180 collections while the
    # first 60 rows cost 0.13 s. The comment above still describes the right FIX (project,
    # never load nodes); its "0.1-0.5 s" simply predates a population that has since
    # tripled.
    if offset:
        query.offset(int(offset))
    if limit:
        query.limit(int(limit))
    out = []
    for row in query.all():
        record = dict(zip(keys, row))
        record["extras"] = record["extras"] or {}
        record["metadata"] = record["metadata"] or {}
        record["member_uuids"] = [str(u) for u in (record["member_uuids"] or [])]
        out.append(record)
    return out


def describe_dataset_record(pk: int, metadata: dict, label: str | None) -> str:
    """`describe_dataset_node` for a projected row (no node object)."""
    identifier = metadata.get("dataset_identifier", label or "-")
    role = metadata.get("dataset_role", "base")
    source_type = metadata.get("source_type", "flow3d")
    folder_name = metadata.get("source_folder_name", "-")
    return f"{pk} | PinoDatasetCollectionData | {identifier} | {role} | {source_type} | {folder_name}"


def recent_dataset_nodes(limit: int = 50) -> list[PinoDatasetCollectionData]:
    """Deprecated window over :func:`active_dataset_nodes`.

    Kept for callers that genuinely want "the newest N" (the picker previews).
    Listings must use ``active_dataset_nodes`` and truncate after filtering.
    """
    return active_dataset_nodes()[:limit]


def dataset_family_nodes(node: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData | orm.FolderData) -> list[PinoDatasetCollectionData]:
    """Return all active dataset collections that belong to the same family as the selected node."""
    if not isinstance(node, PinoDatasetCollectionData):
        return []
    family_identifier = safe_attribute_get(node, "family_identifier", "")
    if not family_identifier:
        return [node]
    builder = orm.QueryBuilder()
    builder.append(PinoDatasetCollectionData, filters={"extras.family_identifier": {"==": family_identifier}}, project=["*"])
    builder.order_by({PinoDatasetCollectionData: {"ctime": "asc"}})
    return [member for member in builder.all(flat=True) if not bool(member.base.extras.get("superseded", False))]


def source_folder_node_for_dataset(node: PinoDatasetCollectionData | PinoDatasetData) -> orm.FolderData | None:
    """Return the linked source `FolderData` node when present."""
    metadata = node.get_dataset_metadata()
    source_folder_uuid = metadata.get("source_folder_uuid")
    source_folder_pk = metadata.get("source_folder_pk")
    try:
        if source_folder_uuid:
            loaded = orm.load_node(str(source_folder_uuid))
            if isinstance(loaded, orm.FolderData):
                return loaded
        if source_folder_pk is not None:
            loaded = orm.load_node(int(source_folder_pk))
            if isinstance(loaded, orm.FolderData):
                return loaded
    except Exception:
        return None
    return None


def summarize_dataset_node(node: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData | orm.FolderData) -> dict[str, Any]:
    """Return a metadata-first summary for dataset browser views."""
    if isinstance(node, PinoDatasetCollectionData):
        metadata = node.get_dataset_metadata()
        # Derived collections are metadata-only wrappers that share their base's
        # member simulations, so they occupy no new disk. Only base collections
        # own their members' bytes; counting derived ones multiplies real usage
        # across a family view.
        if metadata.get("dataset_role", "base") == "base":
            stored_bytes, stored_files = dataset_collection_disk_usage_bytes(node)
        else:
            stored_bytes, stored_files = 0, 0
        return {
            "pk": node.pk,
            "type": node.__class__.__name__,
            "label": node.label or "-",
            "dataset_identifier": metadata.get("dataset_identifier", node.label or "-"),
            "dataset_role": metadata.get("dataset_role", "base"),
            "canonical_layout": metadata.get("canonical_layout", ""),
            "family_identifier": metadata.get("family_identifier", ""),
            "base_dataset_pk": metadata.get("base_collection_pk"),
            "base_dataset_uuid": metadata.get("base_collection_uuid"),
            "parent_dataset_pk": metadata.get("parent_collection_pk"),
            "parent_dataset_uuid": metadata.get("parent_collection_uuid"),
            "derivation_profile": metadata.get("derivation_profile", {}),
            "derivation_hash": metadata.get("derivation_hash", ""),
            "reconstructable": bool(metadata.get("reconstructable", False)),
            "source_folder_pk": metadata.get("source_folder_pk"),
            "source_folder_uuid": metadata.get("source_folder_uuid"),
            "source_type": metadata.get("source_type", ""),
            "source_folder_name": metadata.get("source_folder_name", ""),
            "source_uri": metadata.get("source_uri", ""),
            "requested_fields": metadata.get("requested_fields", []),
            "array_names": metadata.get("requested_fields", []),
            "array_shapes": metadata.get("array_shapes", {}),
            "num_arrays": len(metadata.get("requested_fields", [])),
            "num_samples": node.get_simulation_count(),
            "sample_metadata": metadata.get("sample_metadata", []),
            "stored_bytes": stored_bytes,
            "stored_size": format_bytes(stored_bytes),
            "stored_files": stored_files,
            "metadata": metadata,
        }
    if isinstance(node, PinoDatasetUnionData):
        return {
            "pk": node.pk,
            "type": node.__class__.__name__,
            "label": node.label or node.get_union_name(),
            "dataset_identifier": node.get_union_identifier(),
            "dataset_role": "union",
            "canonical_layout": "",
            "family_identifier": "",
            "base_dataset_pk": None,
            "base_dataset_uuid": "",
            "parent_dataset_pk": None,
            "parent_dataset_uuid": "",
            "derivation_profile": {},
            "derivation_hash": "",
            "reconstructable": True,
            "source_folder_pk": None,
            "source_folder_uuid": "",
            "source_type": "union",
            "source_folder_name": node.get_union_name(),
            "source_uri": "",
            "requested_fields": [],
            "array_names": [],
            "array_shapes": {},
            "num_arrays": 0,
            "num_samples": node.get_total_samples(),
            "sample_metadata": [],
            "stored_bytes": 0,
            "stored_size": format_bytes(0),
            "stored_files": 0,
            "metadata": {
                "union_name": node.get_union_name(),
                "union_identifier": node.get_union_identifier(),
                "member_dataset_pks": node.get_member_dataset_pks(),
                "member_dataset_uuids": node.get_member_dataset_uuids(),
                "compatibility_signature": node.get_compatibility_signature(),
                "total_samples": node.get_total_samples(),
            },
        }
    if isinstance(node, PinoDatasetData):
        metadata = node.get_dataset_metadata()
        sample_metadata = metadata.get("sample_metadata", [])
        role = metadata.get("dataset_role") or ("base" if is_canonical_metadata(metadata) else "legacy")
        stored_bytes, stored_files = repository_disk_usage_bytes(node)
        return {
            "pk": node.pk,
            "type": node.__class__.__name__,
            "label": node.label or "-",
            "dataset_identifier": safe_attribute_get(node, "dataset_identifier", node.label or "-"),
            "dataset_role": role,
            "canonical_layout": metadata.get("canonical_layout", ""),
            "family_identifier": metadata.get("family_identifier", ""),
            "base_dataset_pk": metadata.get("base_dataset_pk"),
            "base_dataset_uuid": metadata.get("base_dataset_uuid"),
            "parent_dataset_pk": metadata.get("parent_dataset_pk"),
            "parent_dataset_uuid": metadata.get("parent_dataset_uuid"),
            "derivation_profile": metadata.get("derivation_profile", {}),
            "derivation_hash": metadata.get("derivation_hash", ""),
            "reconstructable": bool(metadata.get("reconstructable", False)),
            "source_folder_pk": metadata.get("source_folder_pk"),
            "source_folder_uuid": metadata.get("source_folder_uuid"),
            "source_type": safe_attribute_get(node, "source_type", metadata.get("source_type", "")),
            "source_folder_name": safe_attribute_get(node, "source_folder_name", metadata.get("source_folder_name", "")),
            "source_uri": safe_attribute_get(node, "source_uri", metadata.get("source_uri", "")),
            "requested_fields": metadata.get("requested_fields", []),
            "array_names": node.list_original_array_names(),
            "array_shapes": node.get_array_shapes(),
            "num_arrays": len(node.list_original_array_names()),
            "num_samples": len(sample_metadata),
            "sample_metadata": sample_metadata,
            "stored_bytes": stored_bytes,
            "stored_size": format_bytes(stored_bytes),
            "stored_files": stored_files,
            "metadata": metadata,
        }

    stored_bytes, stored_files = repository_disk_usage_bytes(node)
    return {
        "pk": node.pk,
        "type": node.__class__.__name__,
        "label": node.label or "-",
        "dataset_identifier": node.label or "-",
        "dataset_role": "legacy-folder",
        "canonical_layout": "",
        "family_identifier": "",
        "base_dataset_pk": None,
        "base_dataset_uuid": "",
        "parent_dataset_pk": None,
        "parent_dataset_uuid": "",
        "derivation_profile": {},
        "derivation_hash": "",
        "reconstructable": False,
        "source_folder_pk": None,
        "source_folder_uuid": "",
        "source_type": node.base.extras.get("source_type", "legacy-folder"),
        "source_folder_name": node.label or "-",
        "source_uri": node.source.get("uri") if getattr(node, "source", None) else None,
        "requested_fields": [],
        "array_names": [],
        "array_shapes": {},
        "num_arrays": 0,
        "num_samples": 0,
        "sample_metadata": [],
        "stored_bytes": stored_bytes,
        "stored_size": format_bytes(stored_bytes),
        "stored_files": stored_files,
        "metadata": {"legacy_folderdata": True},
    }


def dataset_type_tuple() -> tuple[type[Any], ...]:
    """Return the tuple form used by AiiDA dataset input specs."""
    return DATASET_INPUT_DATA


def payload_twin_uuids(uuid: str) -> list[str]:
    """Every simulation node holding the SAME payload as `uuid`, including itself.

    The same case can exist as more than one node -- re-imports, a rebuild, a
    reorganised batch -- and each node has its own uuid. Datasets reference
    whichever twin existed when they were built, so a reverse lookup keyed on a
    single uuid answers about that ONE node, not about the case.

    Measured 2026-08-09: 70 content hashes cover 140 nodes, and for 68 of those
    70 the twins disagree -- one twin reports 0 datasets, the other 3. The failure
    is silent: a shorter list, no error.
    """
    from aiida import orm

    node = orm.load_node(uuid)
    digest = node.base.extras.get("content_hash", None)
    if not digest:
        return [node.uuid]
    twins = orm.QueryBuilder().append(
        orm.Data,
        filters={"node_type": {"like": "%pino_simulation%"}, "extras.content_hash": digest},
        project=["uuid"],
    ).all(flat=True)
    return sorted(set(twins) | {node.uuid})


def datasets_containing(uuid: str) -> list[int]:
    """pks of every dataset holding this simulation, by payload rather than node.

    Use this instead of a bare
    `{"attributes.member_simulation_uuids": {"contains": [uuid]}}`: that filter
    is correct only for the twin you happen to be holding (see
    :func:`payload_twin_uuids`).
    """
    from aiida import orm

    pks = set()
    for twin in payload_twin_uuids(uuid):
        pks.update(orm.QueryBuilder().append(
            orm.Data,
            filters={
                "node_type": {"like": "%pino_dataset_collection%"},
                "attributes.member_simulation_uuids": {"contains": [twin]},
            },
            project=["id"],
        ).all(flat=True))
    return sorted(pks)
