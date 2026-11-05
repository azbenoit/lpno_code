"""Metadata-only eviction: drop a simulation's array data, keep everything else.

The point is to reclaim the object store without losing the record: after
eviction the node keeps its pk, uuid, every attribute (all its metadata) and
every provenance link, so a model trained on the dataset still resolves its
training source. Only the heavy `payload.npz` is gone, and any attempt to read
the arrays fails with a clear message instead of a low-level file error.
"""

from __future__ import annotations

import pytest

from aiida_pino.data.pino_simulation import DataEvictedError, PinoSimulationData


def test_eviction_keeps_metadata_and_identity(aiida_profile_clean, pino_simulation_factory) -> None:
    sim = pino_simulation_factory("case_a")
    pk, uuid = sim.pk, sim.uuid
    fingerprint = sim.get_simulation_fingerprint()
    assert sim.base.repository.list_object_names(), "fixture should have data to evict"

    freed = sim.evict_data(reason="disk pressure 2026-07-24")

    assert freed > 0
    reloaded = PinoSimulationData.collection.get(uuid=uuid)
    assert reloaded.pk == pk and reloaded.uuid == uuid
    assert reloaded.base.repository.list_object_names() == [], "data survived eviction"
    # Metadata lives in attributes, so it reads back unchanged.
    assert reloaded.get_simulation_fingerprint() == fingerprint
    assert reloaded.get_array_shapes()  # shapes are attributes, not payload
    assert reloaded.is_data_evicted() is True
    assert reloaded.base.extras.get("data_evicted")["reason"] == "disk pressure 2026-07-24"


def test_reading_evicted_arrays_fails_clearly(aiida_profile_clean, pino_simulation_factory) -> None:
    sim = pino_simulation_factory("case_a")
    name = sim.list_original_array_names()[0]
    sim.evict_data()
    with pytest.raises(DataEvictedError):
        sim.get_array(name)


def test_eviction_is_idempotent(aiida_profile_clean, pino_simulation_factory) -> None:
    sim = pino_simulation_factory("case_a")
    assert sim.evict_data() > 0
    assert sim.evict_data() == 0, "second eviction should be a no-op, not an error"


def test_provenance_link_survives_eviction(aiida_profile_clean, pino_simulation_factory) -> None:
    """A dataset that references an evicted sim, and anything downstream of it,
    keeps working — eviction touches no links."""
    from aiida_pino.data.dataset_hierarchy import create_base_dataset

    sims = [pino_simulation_factory(f"c_{i}") for i in range(3)]
    base, _ = create_base_dataset(sims, name="evict_me")

    for sim in sims:
        sim.evict_data()

    # The collection still lists its members and their metadata by uuid.
    members = list(base.iter_member_simulations())
    assert len(members) == 3
    assert all(m.is_data_evicted() for m in members)
    assert members[0].get_simulation_fingerprint()  # attribute, intact


def test_eviction_does_not_corrupt_a_sim_sharing_the_same_bytes(aiida_profile_clean, pino_simulation_factory) -> None:
    """The object store is content-addressed: two identical simulations share
    one copy of the bytes. Evicting one must not delete data the other needs."""
    import numpy as np

    arrays = {"Temperature": np.arange(64, dtype=np.float32).reshape(1, 1, 4, 4, 4)}
    a = pino_simulation_factory("twin_a", arrays={k: v.copy() for k, v in arrays.items()})
    b = pino_simulation_factory("twin_b", arrays={k: v.copy() for k, v in arrays.items()})

    a.evict_data()

    assert a.is_data_evicted() and not b.is_data_evicted()
    got = b.get_array(b.list_original_array_names()[0])
    assert np.asarray(got).size == 64, "the twin's shared bytes were deleted with its sibling"


def _svc():
    from web.app.aiida_service import AiidaConsoleService
    from web.app.config import Settings

    return AiidaConsoleService(Settings())


def test_service_dry_run_reports_without_evicting(aiida_profile_clean, pino_simulation_factory) -> None:
    from aiida_pino.data.dataset_hierarchy import create_base_dataset

    sims = [pino_simulation_factory(f"c_{i}") for i in range(3)]
    base, _ = create_base_dataset(sims, name="scratch_set")
    svc = _svc()

    report = svc.evict_dataset_data(base.pk, dry_run=True)
    assert report["would_evict"] == 3 and report["reclaimable_bytes"] > 0
    assert all(not s.is_data_evicted() for s in sims), "dry run evicted data"

    done = svc.evict_dataset_data(base.pk, dry_run=False, reason="test")
    assert done["evicted"] == 3 and done["bytes_freed"] > 0
    assert all(PinoSimulationData.collection.get(uuid=s.uuid).is_data_evicted() for s in sims)


def test_service_refuses_core_datasets(aiida_profile_clean, pino_simulation_factory) -> None:
    from web.app.errors import InvalidRequestError

    from aiida_pino.data.dataset_hierarchy import create_base_dataset
    from aiida_pino.utils.annotations import update_node_annotations

    sims = [pino_simulation_factory(f"c_{i}") for i in range(2)]
    base, _ = create_base_dataset(sims, name="reference_set")
    update_node_annotations(base, keywords_add=["core"])

    with pytest.raises(InvalidRequestError):
        _svc().evict_dataset_data(base.pk, dry_run=False)
    assert all(not s.is_data_evicted() for s in sims)
