"""Mixed-field view collections: strict refusal must say why, intersect must work.

The live case (2026-07-17): `material_lhs_300_plus_original_steady_split` could
not be represented because `Temperature Gradient At Tgrdout` exists only in the
straight-steady members — while the identical folder on disk trains fine (the
datasource intersects fields at load). `field_policy: "intersect"` makes the
collection layer say what the consumer already does. Identity is safe by
construction: a mixed member set cannot exist under "strict", so no strict
collection can collide with an intersect one.
"""

from __future__ import annotations

import numpy as np
import pytest


def _members(pino_simulation_factory):
    a = pino_simulation_factory(
        "steady_a",
        arrays={
            "Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32),
            "alpha": np.ones((2, 1, 4, 4, 4), dtype=np.float32),
            "Temperature Gradient At Tgrdout": np.ones((2, 1, 4, 4, 4), dtype=np.float32),
        },
    )
    b = pino_simulation_factory(
        "random_b",
        arrays={
            "Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32),
            "alpha": np.ones((2, 1, 4, 4, 4), dtype=np.float32),
        },
    )
    return a, b


def test_strict_refusal_names_the_odd_fields_and_the_opt_in(aiida_profile_clean, pino_simulation_factory):
    from aiida_pino.data import validate_collection_members

    a, b = _members(pino_simulation_factory)
    with pytest.raises(ValueError) as err:
        validate_collection_members([a, b])
    message = str(err.value)
    assert "Temperature Gradient At Tgrdout" in message, "the refusal must name what differs"
    assert "intersect" in message, "the refusal must name the opt-in"


def test_intersect_creates_the_view_on_shared_fields(aiida_profile_clean, pino_simulation_factory):
    from aiida_pino.data import PinoDatasetCollectionData

    a, b = _members(pino_simulation_factory)
    node = PinoDatasetCollectionData.from_simulations(
        [a, b], metadata={"field_policy": "intersect"}
    ).store()
    attrs = node.base.attributes.all
    assert attrs["field_policy"] == "intersect", "must be a queryable top-level attribute"
    assert attrs["dataset_metadata"]["requested_fields"] == ["Temperature", "alpha"], \
        "the recorded fields must be what EVERY member carries"


def test_intersect_does_not_bridge_source_types(aiida_profile_clean, pino_simulation_factory):
    from aiida_pino.data import validate_collection_members

    a = pino_simulation_factory("f3d", source_type="flow3d")
    b = pino_simulation_factory("of", source_type="openfoam")
    with pytest.raises(ValueError, match="source types"):
        validate_collection_members([a, b], field_policy="intersect")


def test_an_empty_intersection_is_refused(aiida_profile_clean, pino_simulation_factory):
    from aiida_pino.data import validate_collection_members

    a = pino_simulation_factory(
        "only_t", arrays={"Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32)}
    )
    b = pino_simulation_factory(
        "only_alpha", arrays={"alpha": np.ones((2, 1, 4, 4, 4), dtype=np.float32)}
    )
    with pytest.raises(ValueError, match="train on nothing"):
        validate_collection_members([a, b], field_policy="intersect")


def test_strict_collections_are_unchanged(aiida_profile_clean, pino_simulation_factory):
    """Same-field members keep the exact pre-change metadata shape."""
    from aiida_pino.data import PinoDatasetCollectionData

    a = pino_simulation_factory("same_1")
    b = pino_simulation_factory("same_2")
    node = PinoDatasetCollectionData.from_simulations([a, b], metadata={}).store()
    attrs = node.base.attributes.all
    assert attrs["field_policy"] == "strict"
    assert sorted(attrs["dataset_metadata"]["requested_fields"]) == ["Temperature", "alpha"]
