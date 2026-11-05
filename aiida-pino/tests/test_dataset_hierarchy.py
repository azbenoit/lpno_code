"""Tests for the base/composed dataset hierarchy and simulation orphan tracking.

The model these tests pin down:

* a simulation belongs to **at most one** base dataset;
* a simulation in no base dataset is **orphaned**;
* a base dataset is built **only** from orphaned simulations, and usually
  corresponds to the batch that produced them;
* a composed dataset is built from base and composed datasets, never from raw
  simulations.
"""

from __future__ import annotations

import pytest

from aiida_pino.data import PinoDatasetCollectionData
from aiida_pino.data.dataset_hierarchy import (
    BASE_COMPOSITION_KIND,
    COMPOSED_COMPOSITION_KIND,
    CUSTOM_COMPOSITION_KIND,
    InvalidCompositionSourceError,
    SimulationAlreadyAttachedError,
    base_dataset_of,
    create_base_dataset,
    is_base_dataset,
    iter_holding_collections,
    create_composed_dataset,
    create_custom_dataset,
    extend_base_dataset,
    is_orphaned,
    list_orphaned_simulations,
)


def _sim(factory, name):
    return factory("batch", group_name=name)


def test_a_fresh_simulation_is_orphaned(aiida_profile_clean, pino_simulation_factory) -> None:
    sim = _sim(pino_simulation_factory, "case_000")
    assert is_orphaned(sim) is True
    assert base_dataset_of(sim) is None


def test_create_base_dataset_claims_its_simulations(aiida_profile_clean, pino_simulation_factory) -> None:
    sims = [_sim(pino_simulation_factory, f"case_{i:03d}") for i in range(3)]
    node, created = create_base_dataset(sims, name="mck_batch")

    assert created is True
    assert isinstance(node, PinoDatasetCollectionData)
    assert node.get_dataset_metadata()["composition_kind"] == BASE_COMPOSITION_KIND
    assert node.get_dataset_role() == "base"
    for sim in sims:
        assert is_orphaned(sim) is False
        assert base_dataset_of(sim).pk == node.pk


def test_a_simulation_cannot_join_two_base_datasets(aiida_profile_clean, pino_simulation_factory) -> None:
    shared = _sim(pino_simulation_factory, "case_000")
    other = _sim(pino_simulation_factory, "case_001")
    first, _ = create_base_dataset([shared], name="first_batch")

    with pytest.raises(SimulationAlreadyAttachedError) as excinfo:
        create_base_dataset([shared, other], name="second_batch")

    # The error has to name the owner, otherwise the user cannot act on it.
    assert str(first.pk) in str(excinfo.value)
    # The rejected call must not have attached the innocent member either.
    assert is_orphaned(other) is True


def test_list_orphaned_simulations_excludes_attached_ones(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    attached = [_sim(pino_simulation_factory, f"case_{i:03d}") for i in range(2)]
    loose = _sim(pino_simulation_factory, "case_099")
    create_base_dataset(attached, name="mck_batch")

    orphan_uuids = {sim.uuid for sim in list_orphaned_simulations()}
    assert loose.uuid in orphan_uuids
    assert orphan_uuids.isdisjoint({sim.uuid for sim in attached})


def test_legacy_singleton_import_collection_does_not_claim_its_simulation(
    aiida_profile_clean, pino_simulation_factory, pino_dataset_factory
) -> None:
    """The per-case collections the FLOW-3D ingest emits are provisional records.

    143 of them exist in the live profile (`role=base, kind=import, 1 sim`). They
    must not count as base datasets, or every batch simulation would already be
    attached and no real base dataset could ever be formed.
    """
    legacy = pino_dataset_factory("legacy_case", role="base")
    member = next(iter(legacy.iter_member_simulations()))
    assert legacy.get_dataset_metadata()["composition_kind"] != BASE_COMPOSITION_KIND
    assert is_orphaned(member) is True

    node, _ = create_base_dataset([member], name="adopting_batch")
    assert base_dataset_of(member).pk == node.pk


def test_composed_dataset_is_built_from_datasets(aiida_profile_clean, pino_simulation_factory) -> None:
    left = [_sim(pino_simulation_factory, f"l_{i:03d}") for i in range(2)]
    right = [_sim(pino_simulation_factory, f"r_{i:03d}") for i in range(2)]
    base_left, _ = create_base_dataset(left, name="batch_left")
    base_right, _ = create_base_dataset(right, name="batch_right")

    composed, created = create_composed_dataset([base_left, base_right], name="union")

    assert created is True
    assert composed.get_dataset_role() == "composed"
    assert composed.get_dataset_metadata()["composition_kind"] == COMPOSED_COMPOSITION_KIND
    assert composed.base.attributes.get("simulation_count") == 4
    # Composition must not steal ownership: the base datasets still own them.
    for sim in left:
        assert base_dataset_of(sim).pk == base_left.pk


def test_composed_dataset_can_compose_composed_datasets(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    a, _ = create_base_dataset([_sim(pino_simulation_factory, "a_000")], name="batch_a")
    b, _ = create_base_dataset([_sim(pino_simulation_factory, "b_000")], name="batch_b")
    c, _ = create_base_dataset([_sim(pino_simulation_factory, "c_000")], name="batch_c")
    inner, _ = create_composed_dataset([a, b], name="inner")

    outer, _ = create_composed_dataset([inner, c], name="outer")

    assert outer.base.attributes.get("simulation_count") == 3


def test_composed_dataset_rejects_raw_simulations(aiida_profile_clean, pino_simulation_factory) -> None:
    sim = _sim(pino_simulation_factory, "case_000")
    base, _ = create_base_dataset([_sim(pino_simulation_factory, "case_001")], name="batch")

    with pytest.raises(InvalidCompositionSourceError):
        create_composed_dataset([base, sim], name="bad_union")


def test_base_dataset_rejects_an_empty_member_list(aiida_profile_clean) -> None:
    with pytest.raises(ValueError):
        create_base_dataset([], name="empty")


def _tag(sim, *keywords) -> None:
    sim.base.extras.set("keywords", list(keywords))


def test_custom_dataset_selects_by_keyword_across_base_datasets(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    """The motivating case: only the "keyhole" simulations of two base datasets."""
    left = [_sim(pino_simulation_factory, f"l_{i:03d}") for i in range(3)]
    right = [_sim(pino_simulation_factory, f"r_{i:03d}") for i in range(2)]
    _tag(left[0], "keyhole")
    _tag(left[1], "conduction")
    _tag(right[0], "keyhole", "random path")
    base_left, _ = create_base_dataset(left, name="batch_left")
    base_right, _ = create_base_dataset(right, name="batch_right")

    custom, created = create_custom_dataset(
        [base_left, base_right], name="keyhole_only", keywords=["keyhole"]
    )

    assert created is True
    assert custom.get_dataset_metadata()["composition_kind"] == CUSTOM_COMPOSITION_KIND
    member_uuids = set(custom.base.attributes.get("member_simulation_uuids"))
    assert member_uuids == {str(left[0].uuid), str(right[0].uuid)}
    # Breaking up a base dataset must not break it: ownership is untouched.
    assert base_dataset_of(left[0]).pk == base_left.pk
    assert base_dataset_of(left[1]).pk == base_left.pk


def test_custom_dataset_selects_explicit_members(aiida_profile_clean, pino_simulation_factory) -> None:
    sims = [_sim(pino_simulation_factory, f"c_{i:03d}") for i in range(3)]
    base, _ = create_base_dataset(sims, name="batch")

    custom, _ = create_custom_dataset([base], name="picked", member_uuids=[sims[2].uuid, sims[0].uuid])

    member_uuids = set(custom.base.attributes.get("member_simulation_uuids"))
    assert member_uuids == {str(sims[0].uuid), str(sims[2].uuid)}


def test_custom_dataset_requires_a_selection(aiida_profile_clean, pino_simulation_factory) -> None:
    base, _ = create_base_dataset([_sim(pino_simulation_factory, "s_000")], name="batch")
    # No keywords and no explicit members would just be a composed dataset.
    with pytest.raises(ValueError):
        create_custom_dataset([base], name="everything")


def test_custom_dataset_rejects_raw_simulations_and_empty_selection_result(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    sim = _sim(pino_simulation_factory, "s_000")
    base, _ = create_base_dataset([_sim(pino_simulation_factory, "s_001")], name="batch")

    with pytest.raises(InvalidCompositionSourceError):
        create_custom_dataset([base, sim], name="bad", keywords=["keyhole"])
    # A selection matching nothing is an error, not an empty dataset.
    with pytest.raises(ValueError):
        create_custom_dataset([base], name="none_match", keywords=["keyhole"])


def test_composed_dataset_accepts_a_custom_dataset_as_source(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    sims = [_sim(pino_simulation_factory, f"k_{i:03d}") for i in range(2)]
    _tag(sims[0], "keyhole")
    base, _ = create_base_dataset(sims, name="batch")
    custom, _ = create_custom_dataset([base], name="keyhole_only", keywords=["keyhole"])
    other, _ = create_base_dataset([_sim(pino_simulation_factory, "o_000")], name="other")

    composed, _ = create_composed_dataset([custom, other], name="union")

    assert composed.base.attributes.get("simulation_count") == 2


def test_extend_base_dataset_grows_a_batch_and_supersedes_the_old_node(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    """A batch's base dataset grows as its sweep lands: extending creates a new
    node owning old + new members and releases the old node's ownership."""
    first = [_sim(pino_simulation_factory, f"case_{i:03d}") for i in range(2)]
    late = [_sim(pino_simulation_factory, f"case_{i:03d}") for i in range(2, 4)]
    old, _ = create_base_dataset(first, name="mck_batch")

    new = extend_base_dataset(old, late)

    assert new.base.attributes.get("simulation_count") == 4
    assert old.base.extras.get("superseded") is True
    assert old.base.extras.get("replaced_by_pk") == new.pk
    # Ownership moved wholesale: every member resolves to the new node.
    for sim in first + late:
        assert base_dataset_of(sim).pk == new.pk
    # And the superseded node no longer blocks anything: its members are owned
    # exactly once.
    assert is_orphaned(first[0]) is False


def test_extend_base_dataset_still_rejects_members_owned_elsewhere(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    mine, _ = create_base_dataset([_sim(pino_simulation_factory, "m_000")], name="mine")
    theirs, _ = create_base_dataset([_sim(pino_simulation_factory, "t_000")], name="theirs")
    stolen = next(iter(theirs.iter_member_simulations()))

    with pytest.raises(SimulationAlreadyAttachedError):
        extend_base_dataset(mine, [stolen])
    # The failed extension must not have superseded the extended node.
    assert mine.base.extras.get("superseded", False) is False


def test_ownership_is_decided_by_composition_kind_not_dataset_role(
    aiida_profile_clean, pino_simulation_factory, pino_dataset_factory
) -> None:
    """`dataset_role` cannot answer "does this own its members?".

    It defaults to "base" for every collection ever built
    (`PinoDatasetCollectionData.DEFAULT_DATASET_ROLE`), so on the live profile
    741 import records carry the role without having been granted anything.
    Reading ownership off it would make 3273 simulations abruptly owned, 546 of
    them by two datasets at once, and would make `train_20`, `train_40` and
    `train_60` separate owners of simulations `train` also holds. One predicate,
    one attribute.
    """
    imported = pino_dataset_factory("imported_batch", role="base")
    assert imported.get_dataset_metadata()["dataset_role"] == "base"
    assert is_base_dataset(imported) is False

    member = next(iter(imported.iter_member_simulations()))
    owned, _ = create_base_dataset([member], name="real_base")
    assert is_base_dataset(owned) is True


def test_create_base_dataset_refuses_to_reassemble_an_assembled_batch(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    """The near-miss, made explicit.

    A 150-case batch held by a multi-member import collection reported 150
    orphans, and the natural response — file them — would have created a second
    base dataset for a batch that already had one. Only content-signature dedup
    stopped it, and dedup catches the exact-match case only: a partial re-file,
    or a re-ingest with one extra case, would have gone through silently.
    """
    from aiida_pino.utils import build_collection_metadata_from_simulations
    from aiida_pino.utils.compose import compose_collection, member_inputs

    sims = [_sim(pino_simulation_factory, f"case_{i:03d}") for i in range(3)]
    metadata = build_collection_metadata_from_simulations(
        sims, collection_name="assembled_batch", dataset_role="base", composition_kind="import"
    )
    batch = compose_collection(sims, metadata, **member_inputs(sims))[0]

    # Unchanged: an import collection does not own its members.
    assert is_base_dataset(batch) is False
    assert all(is_orphaned(sim) for sim in sims)
    # But it is findable, which is what stops the duplicate.
    assert [holder.pk for holder in iter_holding_collections(sims[0])] == [batch.pk]

    with pytest.raises(SimulationAlreadyAttachedError, match="already assembled into"):
        create_base_dataset(sims, name="duplicate_of_the_same_batch")
    # A partial re-file is the case dedup cannot catch, so it must be refused too.
    with pytest.raises(SimulationAlreadyAttachedError, match="already assembled into"):
        create_base_dataset(sims[:2], name="partial_duplicate")

    # Even with the override, an identical member set dedupes to the existing
    # import collection, so no ownership is established. That must be loud: it
    # is the mechanism that made dedup look like protection.
    with pytest.raises(SimulationAlreadyAttachedError, match="import record rather than a base dataset"):
        create_base_dataset(sims, name="deliberate_regroup", allow_reassembly=True)
    assert all(is_orphaned(sim) for sim in sims)

    # A different member set is a genuinely different collection, so it creates.
    extra = _sim(pino_simulation_factory, "case_003")
    node, _ = create_base_dataset(sims + [extra], name="regrouped_with_more",
                                  allow_reassembly=True)
    assert is_base_dataset(node) is True
    assert base_dataset_of(sims[0]).pk == node.pk


def test_a_singleton_import_collection_is_not_treated_as_an_assembled_batch(
    aiida_profile_clean, pino_simulation_factory, pino_dataset_factory
) -> None:
    """One collection per case is what the ingest is supposed to emit.

    710 of them are live. Reporting each as a holder would refuse every filing
    call in the repo, so the reassembly guard starts at two members.
    """
    legacy = pino_dataset_factory("single_case", role="base")
    member = next(iter(legacy.iter_member_simulations()))
    assert list(iter_holding_collections(member)) == []

    node, _ = create_base_dataset([member], name="adopting_batch")
    assert base_dataset_of(member).pk == node.pk


def test_base_dataset_of_matches_a_naive_scan(aiida_profile_clean, pino_simulation_factory) -> None:
    """`base_dataset_of` is a JSONB containment query; pin it to the scan it replaced.

    The scan was O(base datasets) per call and cost 1.48 s against the real store,
    which starved the daemon worker's event loop during a 300-case batch finalize
    (2026-08-19). The rewrite is only safe while it answers identically -- and its
    failure mode is silent: a filter that matches nothing reports every simulation
    orphaned, which is exactly the state that mints a duplicate base dataset.
    """
    from aiida_pino.data.dataset_hierarchy import _member_uuids, iter_base_datasets

    def scan(simulation):
        uuid = str(simulation.uuid)
        for node in iter_base_datasets():
            if uuid in _member_uuids(node):
                return node
        return None

    owned = [_sim(pino_simulation_factory, f"case_{i:03d}") for i in range(3)]
    node, _ = create_base_dataset(owned, name="scan_parity")
    loose = _sim(pino_simulation_factory, "case_099")

    for sim in owned:
        assert base_dataset_of(sim).pk == node.pk == scan(sim).pk
    assert base_dataset_of(loose) is scan(loose) is None


def test_a_superseded_base_dataset_does_not_own_its_members(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    """Superseding releases members, so the containment hit must be skipped.

    The uuids stay in the superseded node's attributes, so a containment query
    still matches it -- only the `superseded` extra says it no longer owns them.
    """
    sims = [_sim(pino_simulation_factory, f"case_{i:03d}") for i in range(2)]
    node, _ = create_base_dataset(sims, name="retired_batch")
    node.base.extras.set("superseded", True)

    for sim in sims:
        assert base_dataset_of(sim) is None
        assert is_orphaned(sim) is True


def test_holding_collections_for_counts_overlap_in_one_pass(
    aiida_profile_clean, pino_simulation_factory
) -> None:
    """The reassembly guard's scan, hoisted out of the per-member loop.

    It used to run once per member: 21 s a call against 2410 collections, so a
    300-case batch spent ~105 min here inside one workchain step and starved the
    daemon worker (2026-08-19). The count per holder must survive the hoist --
    `create_base_dataset` puts it in the error message ("N of these M members").
    """
    from aiida_pino.data.dataset_hierarchy import holding_collections_for

    sims = [_sim(pino_simulation_factory, f"case_{i:03d}") for i in range(3)]
    base, _ = create_base_dataset(sims, name="owning_batch")
    # A non-base collection holding two of the three: this is the shape the guard
    # exists to catch -- a batch already assembled under another pk.
    holder, _ = create_custom_dataset(
        [base], name="import_record", member_uuids=[sims[0].uuid, sims[1].uuid]
    )
    assert holder.get_dataset_metadata()["composition_kind"] == CUSTOM_COMPOSITION_KIND

    held = holding_collections_for(sims)
    assert list(held) == [holder.pk]
    assert held[holder.pk][1] == 2, "holds 2 of the 3 simulations"

    # Same answer as the single-simulation generator it now backs.
    assert [n.pk for n in iter_holding_collections(sims[0])] == [holder.pk]
    assert holding_collections_for([sims[2]]) == {}
