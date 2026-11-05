"""Regression: supersession churn must not push live datasets out of listings.

Hit live 2026-07-22: the AL loop had created ~350 superseded chain collections,
recent_collection_nodes fetched the 2x-limit newest rows and only then dropped
superseded ones, so the console listed 54 datasets and none of the core sets
(original_straight_path_steady, the 300-case variable random path set) appeared
— they looked deleted while being fully intact.
"""

from __future__ import annotations

from aiida_pino.data.dataset_hierarchy import create_base_dataset
from aiida_pino.utils.model_registry import recent_collection_nodes


def _base(factory, name):
    node, _ = create_base_dataset([factory(name)], name=name)
    return node


def test_old_live_dataset_survives_newer_superseded_churn(aiida_profile_clean, pino_simulation_factory) -> None:
    keeper = _base(pino_simulation_factory, "core_batch")
    for i in range(5):  # newer churn, all dead — more than 2x the limit below
        chaff = _base(pino_simulation_factory, f"chaff_{i}")
        chaff.base.extras.set("superseded", True)
    listed = recent_collection_nodes(limit=2)
    assert keeper.pk in [n.pk for n in listed], "live dataset fell out of the recency window"
    assert all(not n.base.extras.get("superseded", False) for n in listed)
