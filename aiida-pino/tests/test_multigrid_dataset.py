"""One dataset can hold members with different spatial grids, honestly.

Batch `ti64_geom40` varies the in-plane extent at fixed 10 µm
resolution: 40 cases, 40 distinct spatial shapes, meant to land in ONE dataset.
Before the 2026-08-08 change this failed in three places: the raw
import adapter refused mixed grids outright (exit 410), a collection records exactly one grid
taken from the first member — so a batch-built dataset would silently describe
39 of 40 members with the wrong shape — and, found while writing these tests,
`validate_collection_members` also refuses, because the compatibility signature
hashes `grid_params` and `spatial_shape` per member.

The fix mirrors `uniform_timesteps`, which solved the identical problem for the
time axis: record per-member values, flag uniformity, keep the legacy
single-value keys meaning what they always meant, and gate consumers on the
flag instead of refusing at import.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.data.pino_dataset_collection import (
    PinoDatasetCollectionData,
    collection_content_signature,
    validate_collection_members,
)


def _sim(pino_simulation_factory, name: str, shape: tuple[int, int, int]):
    frames = 2
    arrays = {
        "Temperature": np.random.rand(frames, *shape).astype(np.float32),
        "alpha": np.ones((frames, *shape), dtype=np.float32),
    }
    return pino_simulation_factory(name, arrays=arrays)


@pytest.fixture
def mixed_pair(pino_simulation_factory):
    """The extreme pair from the real batch: geom40_000 and geom40_039 shapes."""
    small = _sim(pino_simulation_factory, "geom40_000", (90, 41, 40))
    large = _sim(pino_simulation_factory, "geom40_039", (187, 125, 40))
    return small, large


def test_mixed_grids_form_one_collection_and_say_so(aiida_profile_clean, mixed_pair):
    """Handoff tests 1 and 2: honest per-member record, unchanged legacy keys."""
    small, large = mixed_pair
    node = PinoDatasetCollectionData.from_simulations(
        [small, large], metadata={"collection_name": "geom40_pair"}
    ).store()

    assert node.has_uniform_grid is False
    metadata = node.get_dataset_metadata()
    shapes = metadata["member_spatial_shapes"]
    assert shapes == [[90, 41, 40], [187, 125, 40]], "member order must be preserved"
    assert len(metadata["member_grid_params"]) == 2

    # Backward compatibility: the single-value keys keep their exact meaning —
    # populated from the FIRST member — so every existing reader is unchanged.
    first_meta = small.get_simulation_metadata()
    assert metadata["grid_params"] == first_meta.get("grid_params", {})
    assert metadata["spatial_shape"] == [90, 41, 40]


def test_a_single_grid_collection_reports_uniform(aiida_profile_clean, pino_simulation_factory):
    """Handoff test 3."""
    members = [_sim(pino_simulation_factory, f"uni_{i}", (30, 30, 50)) for i in range(2)]
    node = PinoDatasetCollectionData.from_simulations(
        members, metadata={"collection_name": "uniform_pair"}
    ).store()
    assert node.has_uniform_grid is True
    assert node.get_dataset_metadata()["member_spatial_shapes"] == [[30, 30, 50]] * 2


def test_a_pre_change_collection_reads_as_uniform(aiida_profile_clean, pino_simulation_factory):
    """Handoff test 4: an absent key must NOT read as heterogeneous.

    The 1366 existing base datasets are single-grid by construction; making them
    suddenly report False would gate every legacy dataset out of consumers that
    require a uniform grid.
    """
    node = PinoDatasetCollectionData.from_simulations(
        [_sim(pino_simulation_factory, "legacy", (30, 30, 50))],
        metadata={"collection_name": "legacy_style"},
    ).store()
    # Simulate a node stored before the change by removing the new keys.
    node.backend_entity.set_attribute("uniform_grid", None)
    metadata = dict(node.base.attributes.get("dataset_metadata"))
    metadata.pop("member_spatial_shapes", None)
    metadata.pop("uniform_grid", None)
    node.backend_entity.set_attribute("dataset_metadata", metadata)

    assert node.has_uniform_grid is True


def test_validation_permits_mixed_grids_but_still_refuses_mixed_fields(
    aiida_profile_clean, mixed_pair, pino_simulation_factory
):
    """Found while writing these tests, beyond the handoff's three sites.

    `validate_collection_members` hashes `grid_params` and `spatial_shape` into
    the per-member compatibility signature, so mixed grids were refused with
    "not compatible for one dataset collection" before `_accumulate` ever ran —
    it is what refused `e6_mesh5_90w` from the plume dataset. Grid divergence is
    now recorded rather than refused; a genuinely different FIELD SET must still
    be refused, because a collection whose members disagree about what arrays
    exist serves no coherent training data.
    """
    small, large = mixed_pair
    members, signature = validate_collection_members([small, large])
    assert len(members) == 2 and signature

    odd = pino_simulation_factory(
        "odd_fields",
        arrays={"Temperature": np.random.rand(2, 30, 30, 50).astype(np.float32)},
    )
    with pytest.raises(ValueError, match="not compatible"):
        validate_collection_members([small, odd])


def test_content_signature_of_single_grid_collections_is_unchanged(
    aiida_profile_clean, pino_simulation_factory
):
    """Handoff test 6: signature drift mints duplicate collections.

    Precedent: one filter change created 433 superseded clones. The signature is
    member fingerprints + derivation, so the new attributes must not enter it.
    """
    sim = _sim(pino_simulation_factory, "sig_stability", (30, 30, 50))
    node = PinoDatasetCollectionData.from_simulations(
        [sim], metadata={"collection_name": "sig_check"}
    ).store()
    metadata = node.get_dataset_metadata()
    # Recompute the signature exactly as `from_simulations` does: fingerprints
    # plus derivation plus the (first-member) compatibility signature. If the
    # multigrid keys ever leaked into any of those inputs, this stops matching
    # and every existing collection would dedup to a new node.
    recomputed = collection_content_signature(
        [sim.get_simulation_fingerprint()],
        derivation_profile=dict(metadata.get("derivation_profile", {}) or {}),
        compatibility_signature=str(metadata.get("compatibility_signature", "")),
    )
    assert node.base.attributes.get("content_signature") == recomputed
    for key in ("member_spatial_shapes", "member_grid_params", "uniform_grid"):
        assert key not in str(
            metadata.get("compatibility_signature", "")
        ), "the new keys must never enter a signature input"
