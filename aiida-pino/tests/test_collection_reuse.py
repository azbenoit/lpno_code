"""Regression: composing the same collection twice must reuse, not duplicate.

Hit live 2026-07-23. ``find_reusable_dataset_collection`` filtered on
``attributes.dataset_identifier`` equal to ``metadata["dataset_identifier"]``,
but the metadata builder never emits that key (the identifier is derived at node
creation). The filter therefore compared against "" and never matched, so every
re-composition minted a fresh node and superseded the previous one. Visible
damage: ~350 superseded chain nodes, duplicate ``*_train_split``/``*_val_split``
datasets on every training submission, and user keywords stranded on the
superseded predecessor while the unannotated clone was the one listed.
"""

from __future__ import annotations

from aiida_pino.data.dataset_hierarchy import create_base_dataset, extend_base_dataset
from aiida_pino.utils.annotations import update_node_annotations
from aiida_pino.utils.model_registry import (
    build_collection_metadata_from_simulations,
    get_or_create_dataset_collection,
)


def _compose(sims, name):
    metadata = build_collection_metadata_from_simulations(
        sims, collection_name=name, dataset_role="composed", composition_kind="composed"
    )
    return get_or_create_dataset_collection(sims, metadata=metadata)


def test_identical_composition_reuses_the_existing_collection(aiida_profile_clean, pino_simulation_factory) -> None:
    sims = [pino_simulation_factory(f"case_{i}") for i in range(3)]
    first, created_first = _compose(sims, "train split")
    second, created_second = _compose(sims, "train split")

    assert created_first is True
    assert created_second is False, "identical re-composition minted a duplicate collection"
    assert second.pk == first.pk
    assert not first.base.extras.get("superseded", False), "the reused collection was superseded by its own clone"


def test_extension_carries_user_annotations_forward(aiida_profile_clean, pino_simulation_factory) -> None:
    """AL rounds extend a base dataset every round; the curated keywords
    and notes must land on the live successor instead of being stranded on the
    superseded predecessor."""
    base, _ = create_base_dataset([pino_simulation_factory(f"case_{i}") for i in range(3)], name="alsi_round")
    update_node_annotations(base, keywords_add=["core", "steady"], description="the curated one")

    grown = extend_base_dataset(base, [pino_simulation_factory("case_extra")])

    assert grown.pk != base.pk
    assert base.base.extras.get("superseded", False) is True
    assert set(grown.base.extras.get("keywords", [])) >= {"core", "steady"}, "keywords lost on supersession"
    assert grown.description == "the curated one"
