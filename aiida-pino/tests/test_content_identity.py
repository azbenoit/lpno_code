"""Dataset identity is the data served, not the name it was given.

Duplicated collections were the standing failure: the same members composed
under a new name or role minted a fresh node, so models trained on identical
data pointed at different datasets (kh2579 vs the 1601 splits — nine nodes
retired by hand on 2026-07-27). `content_signature` (sorted member
fingerprints + derivation + layout) is stamped at creation; reuse and
supersession match on it, role- and name-agnostic.
"""

from __future__ import annotations

from aiida_pino.data import collection_content_signature


def test_signature_ignores_order_and_names() -> None:
    profile = {"time_as_batch": True}
    a = collection_content_signature(["f1", "f2", "f3"], derivation_profile=profile, compatibility_signature="c")
    b = collection_content_signature(["f3", "f1", "f2"], derivation_profile=profile, compatibility_signature="c")
    assert a == b, "reordered members are not new data"
    assert a != collection_content_signature(["f1", "f2"], derivation_profile=profile, compatibility_signature="c")
    assert a != collection_content_signature(
        ["f1", "f2", "f3"], derivation_profile={"time_as_batch": False}, compatibility_signature="c"
    ), "a different derivation is different served data"


def _collection(sims, name, role="composed"):
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    metadata = build_collection_metadata_from_simulations(sims, collection_name=name, dataset_role=role)
    return get_or_create_dataset_collection(sims, metadata=metadata)


def test_same_members_different_name_reuse_one_node(aiida_profile_clean, pino_simulation_factory) -> None:
    sims = [pino_simulation_factory(f"ci_{i}", group_name=f"g{i}") for i in range(3)]
    first, created_first = _collection(sims, "friendly_name")
    second, created_second = _collection(sims, "totally_different_name")
    assert created_first and not created_second
    assert first.pk == second.pk, "same data under a new name must resolve to the same node"


def test_same_members_different_role_reuse_one_node(aiida_profile_clean, pino_simulation_factory) -> None:
    sims = [pino_simulation_factory(f"cr_{i}", group_name=f"h{i}") for i in range(3)]
    base, _ = _collection(sims, "as_composed", role="composed")
    subset, created = _collection(sims, "as_subset", role="subset")
    assert not created and subset.pk == base.pk, "a role is a view, not new data"


def test_stamped_signature_is_queryable(aiida_profile_clean, pino_simulation_factory) -> None:
    from aiida import orm

    from aiida_pino.data import PinoDatasetCollectionData, safe_attribute_get

    sims = [pino_simulation_factory(f"cq_{i}", group_name=f"q{i}") for i in range(2)]
    node, _ = _collection(sims, "queryable")
    content = safe_attribute_get(node, "content_signature", "")
    assert content
    hits = (orm.QueryBuilder()
            .append(PinoDatasetCollectionData, filters={"attributes.content_signature": content})
            .count())
    assert hits == 1


def test_supersession_collapses_renamed_duplicates_but_never_core(aiida_profile_clean, pino_simulation_factory) -> None:
    from aiida_pino.data import PinoDatasetCollectionData
    from aiida_pino.utils import build_collection_metadata_from_simulations
    from aiida_pino.utils.annotations import update_node_annotations
    from aiida_pino.utils.model_registry import supersede_equivalent_dataset_collections

    sims = [pino_simulation_factory(f"cs_{i}", group_name=f"s{i}") for i in range(2)]
    # store two nodes with identical content directly (bypassing reuse), as the
    # historical duplicates did
    meta_a = build_collection_metadata_from_simulations(sims, collection_name="old_name", dataset_role="base")
    old = PinoDatasetCollectionData.from_simulations(sims, metadata=meta_a).store()
    meta_b = build_collection_metadata_from_simulations(sims, collection_name="new_name", dataset_role="subset")
    new = PinoDatasetCollectionData.from_simulations(sims, metadata=meta_b).store()

    losers = supersede_equivalent_dataset_collections(new)
    assert [n.pk for n in losers] == [old.pk], "renamed/re-roled duplicate must collapse"
    assert old.base.extras.get("superseded") is True

    # core sets survive: give the OLD node core and re-run from a third dupe
    update_node_annotations(old, keywords_add=["core"])
    old.base.extras.set("superseded", False)
    meta_c = build_collection_metadata_from_simulations(sims, collection_name="third", dataset_role="composed")
    third = PinoDatasetCollectionData.from_simulations(sims, metadata=meta_c).store()
    losers = supersede_equivalent_dataset_collections(third)
    assert old.pk not in [n.pk for n in losers], "a core reference set is never superseded"


def test_a_default_stride_does_not_fork_a_dataset():
    """A key that is present-but-default must not change identity.

    When `stride` was added to the derivation profile, `stride: [1, 1, 1]` —
    "every point", i.e. no striding — hashed differently from no stride at all.
    That minted a live twin of the random-path test split, the dataset every
    published evaluation of that split points at (same 53 simulations, both
    live). The first repair normalised `derivation_hash` only, while the
    minting path went through `collection_content_signature`, so the fork kept
    happening: 208 stored signatures had drifted and 10 live duplicate groups
    existed by the time it was caught.
    """
    from aiida_pino.data.pino_dataset_collection import collection_content_signature
    from aiida_pino.utils.dataset_derivation import canonical_base_layout_profile, derivation_hash

    members = ["fp-a", "fp-b", "fp-c"]
    plain = canonical_base_layout_profile()
    unit_stride = {**plain, "stride": [1, 1, 1]}
    real_stride = {**plain, "stride": [2, 1, 1]}

    same = collection_content_signature(members, derivation_profile=plain)
    noop = collection_content_signature(members, derivation_profile=unit_stride)
    strided = collection_content_signature(members, derivation_profile=real_stride)

    assert same == noop, "a no-op stride forked the dataset"
    assert strided != same, "a real stride must still be a different dataset"
    # and the two identity functions must agree about what counts
    assert derivation_hash(plain) == derivation_hash(unit_stride)
    assert derivation_hash(real_stride) != derivation_hash(plain)
