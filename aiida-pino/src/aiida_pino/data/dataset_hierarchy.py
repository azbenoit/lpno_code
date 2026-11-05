"""Base/composed dataset hierarchy over simulations.

Three kinds of dataset, distinguished by `composition_kind`:

* **base** -- built from *orphaned* simulations, and owning them exclusively. A
  base dataset normally corresponds to the batch that produced its members.
* **composed** -- built from base/composed/custom datasets taken whole, never
  from raw simulations. Composition shares members by reference and does not
  transfer ownership: the base dataset keeps it.
* **custom** -- like composed, but it may break its sources up: only the
  members matching a selection (keyword annotations and/or explicit uuids) are
  kept, e.g. just the "keyhole" simulations of two base datasets.

A simulation belongs to at most one base dataset. One in none is **orphaned**
and shows up in `list_orphaned_simulations` until it is attached.

This is deliberately distinct from composition. The FLOW-3D batch path emits one
provisional single-simulation collection per case (`composition_kind="import"`);
those are ingest records, not base datasets, so their members stay orphaned and
remain available to `create_base_dataset`.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from aiida import orm

from .pino_dataset_collection import PinoDatasetCollectionData
from .pino_simulation import PinoSimulationData

BASE_COMPOSITION_KIND = "base"
COMPOSED_COMPOSITION_KIND = "composed"
CUSTOM_COMPOSITION_KIND = "custom"

#: Kinds a composed or custom dataset may be built from.
COMPOSABLE_KINDS = (BASE_COMPOSITION_KIND, COMPOSED_COMPOSITION_KIND, CUSTOM_COMPOSITION_KIND)


class DatasetHierarchyError(Exception):
    """Base class for hierarchy-rule violations."""


class SimulationAlreadyAttachedError(DatasetHierarchyError):
    """A simulation is already owned by another base dataset."""


class InvalidCompositionSourceError(DatasetHierarchyError):
    """A composed dataset was given something other than a base/composed dataset."""


def is_base_dataset(node: PinoDatasetCollectionData) -> bool:
    """Does this collection *own* its members?

    `composition_kind` decides, and it is the only thing that does.
    `create_base_dataset` is the sole writer of `composition_kind="base"`, so
    the attribute records a deliberate act of ownership.

    `dataset_role` looks like it should answer this and cannot: it defaults to
    `"base"` for every collection ever built (`DEFAULT_DATASET_ROLE`), so 741
    live import records claim the role without having been granted anything.
    Reading ownership off it would make 3273 simulations abruptly owned, 546 of
    them by two datasets at once, and would declare `train_20`, `train_40` and
    `train_60` separate owners of simulations that `train` also holds — 53
    overlapping pairs among the multi-member import collections alone. The
    definition lives here, in one function, because keeping it in two attributes
    is what hid it.
    """
    return str(node.base.attributes.get("composition_kind", None) or "") == BASE_COMPOSITION_KIND


def iter_base_datasets() -> Iterator[PinoDatasetCollectionData]:
    """Every live base dataset. Superseded ones have released their members."""
    builder = orm.QueryBuilder().append(
        PinoDatasetCollectionData,
        filters={"attributes.composition_kind": BASE_COMPOSITION_KIND},
    )
    for (node,) in builder.all():
        if node.base.extras.get("superseded", False):
            continue
        yield node


def iter_holding_collections(
    simulation: PinoSimulationData, *, min_members: int = 2
) -> Iterator[PinoDatasetCollectionData]:
    """Live collections that *hold* `simulation` without owning it.

    An import collection is not a base dataset and its members stay orphaned by
    design — but a **multi-member** one is a batch someone assembled, and
    minting a base dataset over its members duplicates it. That is not
    hypothetical: a 150-case batch reported 150 orphans, and only
    content-signature dedup stopped a second base dataset being created for it.
    Dedup catches the exact-match case and nothing else, so a partial re-file
    would have gone through silently.

    Singletons are excluded (`min_members=2`): one collection per case is what
    the ingest is supposed to emit, and reporting each as a holder would flag
    every simulation in the store.
    """
    for node, _overlap in holding_collections_for([simulation], min_members=min_members).values():
        yield node


def holding_collections_for(
    simulations: Iterable[PinoSimulationData], *, min_members: int = 2
) -> dict[int, tuple[PinoDatasetCollectionData, int]]:
    """Live non-owning collections holding any of `simulations`, with overlap counts.

    Keyed by pk, valued `(node, how many of `simulations` it holds)`.

    One projected pass over the collection table for the whole member list, not
    one pass per member. The per-member form this replaced loaded all 2410
    collection nodes and faulted their attributes lazily -- 21 s per call -- so
    `create_base_dataset`'s reassembly guard spent ~105 min on a 300-case batch,
    inside a synchronous workchain step. A workchain step that does not yield
    starves the daemon worker's event loop, and a starved worker answers no RPC
    at all: on 2026-08-19 `verdi process kill` reported every process
    unreachable, on a daemon whose broker and task state were entirely
    consistent. Projecting the three fields needed keeps this one SQL round-trip
    and loads a node only for the rare hit.
    """
    wanted = {str(simulation.uuid) for simulation in simulations}
    held: dict[int, tuple[PinoDatasetCollectionData, int]] = {}
    if not wanted:
        return held
    builder = orm.QueryBuilder().append(
        PinoDatasetCollectionData,
        project=[
            "id",
            "attributes.member_simulation_uuids",
            "attributes.composition_kind",
            "extras.superseded",
        ],
    )
    for pk, member_uuids, composition_kind, superseded in builder.iterall():
        if superseded or str(composition_kind or "") == BASE_COMPOSITION_KIND:
            continue
        members = {str(uuid) for uuid in (member_uuids or [])}
        if len(members) < min_members:
            continue
        overlap = len(members & wanted)
        if overlap:
            held[pk] = (orm.load_node(pk), overlap)
    return held


def _member_uuids(node: PinoDatasetCollectionData) -> set[str]:
    return {str(uuid) for uuid in (node.base.attributes.get("member_simulation_uuids", None) or [])}


def base_dataset_of(simulation: PinoSimulationData) -> PinoDatasetCollectionData | None:
    """The base dataset owning `simulation`, or ``None`` when it is orphaned."""
    # A JSONB containment filter, not a scan. The scan this replaced was written
    # for "tens of nodes"; there are 95 base datasets holding 10762 member uuids,
    # and it cost 1.48 s per call. `Flow3DBatchWorkChain.finalize` calls it once
    # per case, so a 300-case batch spent ~7 min inside one synchronous workchain
    # step -- long enough to starve the daemon worker's event loop, which then
    # answers no RPC at all: `verdi process kill` reports every process
    # unreachable and the queue shows all tasks held but unacknowledged
    # (2026-08-19 wedge). Containment is ~0.14 s and indexed.
    uuid = str(simulation.uuid)
    builder = orm.QueryBuilder().append(
        PinoDatasetCollectionData,
        filters={
            "attributes.composition_kind": BASE_COMPOSITION_KIND,
            "attributes.member_simulation_uuids": {"contains": [uuid]},
        },
    )
    for (node,) in builder.all():
        if node.base.extras.get("superseded", False):
            continue
        return node
    return None


def is_orphaned(simulation: PinoSimulationData) -> bool:
    """True when no base dataset owns `simulation`.

    Orphaned is not the same as unfiled: a simulation can be orphaned by this
    rule and still sit in a multi-member import collection that represents its
    batch. Use `iter_holding_collections` before acting on an orphan count —
    filing an already-assembled batch is how a duplicate gets made.
    """
    return base_dataset_of(simulation) is None


def list_orphaned_simulations(*, source_type: str | None = None) -> list[PinoSimulationData]:
    """Every simulation not attached to a base dataset."""
    claimed: set[str] = set()
    for node in iter_base_datasets():
        claimed |= _member_uuids(node)
    filters = {"attributes.source_type": source_type} if source_type else None
    builder = orm.QueryBuilder().append(PinoSimulationData, filters=filters)
    return [node for (node,) in builder.all() if str(node.uuid) not in claimed]


def create_base_dataset(
    simulations: Iterable[PinoSimulationData],
    *,
    name: str,
    extra_metadata: dict[str, Any] | None = None,
    allow_reassembly: bool = False,
) -> tuple[PinoDatasetCollectionData, bool]:
    """Attach orphaned simulations to a new base dataset.

    Raises `SimulationAlreadyAttachedError` if any member already belongs to a
    base dataset, naming the owner so the caller can act on it. The check runs
    over the whole member list before anything is stored, so a rejected call
    leaves every member orphaned.

    It also refuses when a multi-member import collection already holds these
    members, because that is a batch someone has already assembled and this call
    would duplicate it. `allow_reassembly=True` says a second grouping of the
    same cases is deliberate.
    """
    members = list(simulations)
    if not members:
        raise ValueError("A base dataset needs at least one simulation.")

    conflicts = [(sim, owner) for sim in members if (owner := base_dataset_of(sim)) is not None]
    if conflicts:
        detail = ", ".join(f"{sim.uuid} is in base dataset pk {owner.pk}" for sim, owner in conflicts)
        raise SimulationAlreadyAttachedError(
            f"A simulation belongs to one base dataset only: {detail}. "
            "Detach it there, or compose the two base datasets instead."
        )
    # The guard is wider than the definition on purpose. These members are
    # orphaned by the ownership rule, but if a multi-member import collection
    # already holds them then this batch has been assembled once already, and
    # this call is about to assemble it again under a second pk.
    held: dict[int, tuple[PinoDatasetCollectionData, int]] = {}
    if not allow_reassembly:
        held = holding_collections_for(members)
    if held:
        detail = ", ".join(
            f"pk {pk} ({node.label!r}, {count} of these {len(members)} members)"
            for pk, (node, count) in sorted(held.items())
        )
        raise SimulationAlreadyAttachedError(
            f"These simulations are already assembled into: {detail}. "
            "That collection is an import record, not a base dataset, so its members read as "
            "orphaned — creating a base dataset here would duplicate the batch under a second pk. "
            "Promote the existing collection instead, or pass allow_reassembly=True if a second "
            "grouping of the same cases is genuinely what you want."
        )

    from aiida_pino.utils import (  # deferred: aiida_pino.utils imports this package
        build_collection_metadata_from_simulations,
    )
    from aiida_pino.utils.compose import compose_collection, member_inputs

    metadata = build_collection_metadata_from_simulations(
        members,
        collection_name=name,
        dataset_role="base",
        composition_kind=BASE_COMPOSITION_KIND,
        extra_metadata=extra_metadata,
    )
    # Through the calcfunction, with the members as inputs: membership is then a
    # path in the graph rather than only a JSON attribute, so `verdi archive
    # create -N <dataset>` exports the data and `verdi node delete` on a member
    # names the datasets holding it (review criticism A).
    node, created = compose_collection(members, metadata, **member_inputs(members))
    if not is_base_dataset(node):
        # Collections dedupe on content signature, so an identical *import*
        # collection is handed back instead of a base dataset being created —
        # and the caller would get `created=False` and a node that owns nothing,
        # reported as success. This is the mechanism that hid the duplicate near
        # miss: dedup looked like protection while establishing no ownership.
        raise SimulationAlreadyAttachedError(
            f"An identical collection already exists as pk {node.pk} ({node.label!r}), and it is "
            f"an import record rather than a base dataset (composition_kind="
            f"{node.base.attributes.get('composition_kind', None)!r}). Nothing was created and "
            "nothing owns these simulations. Promote that collection instead of creating a "
            "second grouping of the same members."
        )
    return node, created


def extend_base_dataset(
    base: PinoDatasetCollectionData,
    additional_simulations: Iterable[PinoSimulationData],
    *,
    name: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> PinoDatasetCollectionData:
    """Grow a batch's base dataset: new node with old + new members, old superseded.

    Collections are immutable, so an in-flight batch cannot mutate its base
    dataset as cases land. Extension creates the replacement first and only then
    marks `base` superseded (releasing ownership), so a failed call changes
    nothing. The additions must be orphaned; members owned by *other* base
    datasets are rejected exactly as in `create_base_dataset`.
    """
    additions = list(additional_simulations)
    existing = list(base.iter_member_simulations())
    existing_uuids = {str(sim.uuid) for sim in existing}

    conflicts = []
    for sim in additions:
        owner = base_dataset_of(sim)
        if owner is not None and owner.uuid != base.uuid:
            conflicts.append((sim, owner))
    if conflicts:
        detail = ", ".join(f"{sim.uuid} is in base dataset pk {owner.pk}" for sim, owner in conflicts)
        raise SimulationAlreadyAttachedError(
            f"A simulation belongs to one base dataset only: {detail}. "
            "Detach it there, or compose the two base datasets instead."
        )

    members = existing + [sim for sim in additions if str(sim.uuid) not in existing_uuids]

    from aiida_pino.utils import (  # deferred: see create_base_dataset
        build_collection_metadata_from_simulations,
    )
    from aiida_pino.utils.compose import compose_collection, member_inputs
    from aiida_pino.utils.model_registry import mark_superseded

    resolved_name = name or str(base.get_dataset_metadata().get("source_folder_name", "")) or "base_dataset"
    metadata = build_collection_metadata_from_simulations(
        members,
        collection_name=resolved_name,
        dataset_role="base",
        composition_kind=BASE_COMPOSITION_KIND,
        extra_metadata=extra_metadata,
    )
    # `base` is an input too: an extension is derived from the dataset it grows,
    # and recording that keeps the supersession chain walkable in the graph.
    node, _created = compose_collection(
        members, metadata, previous=base, **member_inputs(members)
    )
    if node.uuid != base.uuid:
        mark_superseded(base, node)
    return node


def _validated_member_pool(datasets: Iterable[PinoDatasetCollectionData]) -> dict[str, PinoSimulationData]:
    """Validate composition sources and pool their members, deduped by uuid."""
    sources = list(datasets)
    if not sources:
        raise ValueError("At least one source dataset is required.")
    for source in sources:
        if not isinstance(source, PinoDatasetCollectionData):
            raise InvalidCompositionSourceError(
                f"Sources must be base, composed or custom datasets, got "
                f"{type(source).__name__}. Attach loose simulations to a base dataset first."
            )
        kind = str(source.get_dataset_metadata().get("composition_kind", ""))
        if kind not in COMPOSABLE_KINDS:
            raise InvalidCompositionSourceError(
                f"Dataset pk {source.pk} has composition_kind={kind!r}; only "
                f"{', '.join(COMPOSABLE_KINDS)} datasets can be composed."
            )
    pool: dict[str, PinoSimulationData] = {}
    for source in sources:
        for simulation in source.iter_member_simulations():
            pool.setdefault(str(simulation.uuid), simulation)
    return pool


def create_custom_dataset(
    datasets: Iterable[PinoDatasetCollectionData],
    *,
    name: str,
    keywords: Iterable[str] | None = None,
    member_uuids: Iterable[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[PinoDatasetCollectionData, bool]:
    """A composed-like dataset that can break up its sources.

    Where `create_composed_dataset` takes sources whole, a custom dataset keeps
    only the members matching the selection: simulations carrying **all** of
    `keywords` (the `extras["keywords"]` annotations), and/or the explicit
    `member_uuids`. Both given means both applied (intersection). Ownership is
    untouched -- the base dataset still owns every selected member.
    """
    sources = list(datasets)
    wanted_keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    wanted_uuids = {str(u) for u in (member_uuids or [])}
    if not wanted_keywords and not wanted_uuids:
        raise ValueError(
            "A custom dataset needs a selection (keywords and/or member_uuids); "
            "to take sources whole, use create_composed_dataset."
        )

    pool = _validated_member_pool(sources)
    members = []
    for uuid, simulation in pool.items():
        if wanted_uuids and uuid not in wanted_uuids:
            continue
        if wanted_keywords:
            tags = {str(t).strip() for t in (simulation.base.extras.get("keywords", None) or [])}
            if not tags.issuperset(wanted_keywords):
                continue
        members.append(simulation)
    if not members:
        raise ValueError("The selection matched no simulation in the source datasets.")

    from aiida_pino.utils import (  # deferred: see create_base_dataset
        build_collection_metadata_from_simulations,
        get_or_create_dataset_collection,
    )

    metadata = build_collection_metadata_from_simulations(
        members,
        collection_name=name,
        dataset_role="composed",
        composition_kind=CUSTOM_COMPOSITION_KIND,
        extra_metadata=extra_metadata,
    )
    node, created = get_or_create_dataset_collection(members, metadata=metadata)
    if not node.base.extras.get("composed_from", None):
        node.base.extras.set("composed_from", [{"pk": s.pk, "uuid": s.uuid} for s in sources])
    if wanted_keywords and not node.base.extras.get("selection_keywords", None):
        node.base.extras.set("selection_keywords", wanted_keywords)
    return node, created


def create_composed_dataset(
    datasets: Iterable[PinoDatasetCollectionData],
    *,
    name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[PinoDatasetCollectionData, bool]:
    """Compose base/composed/custom datasets, taken whole, into one dataset.

    Members are shared by reference and ownership is unchanged -- the base
    dataset still owns them. Duplicate members across sources collapse to one.
    To keep only a subset of the sources' members, use `create_custom_dataset`.
    """
    sources = list(datasets)
    members = list(_validated_member_pool(sources).values())

    from aiida_pino.utils import (  # deferred: see create_base_dataset
        build_collection_metadata_from_simulations,
        get_or_create_dataset_collection,
    )

    metadata = build_collection_metadata_from_simulations(
        members,
        collection_name=name,
        dataset_role="composed",
        composition_kind=COMPOSED_COMPOSITION_KIND,
        extra_metadata=extra_metadata,
    )
    node, created = get_or_create_dataset_collection(members, metadata=metadata)
    if not node.base.extras.get("composed_from", None):
        # UI-layer mirror of the source links, matching the console's compose path.
        node.base.extras.set("composed_from", [{"pk": s.pk, "uuid": s.uuid} for s in sources])
    return node, created
