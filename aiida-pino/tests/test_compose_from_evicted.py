"""A collection can be composed from metadata-only (evicted) simulations.

2026-10-08, registering retrains against the published deposit: every deposited simulation is
metadata-only, and `from_simulations` read its first member's arrays to record the collection's
shapes, so `aiida-pino collection create` failed with DataEvictedError. For a collection that
serves its members unchanged (every role but `derived`), the shapes are the member's recorded
canonical shapes, which need no array.
"""
from __future__ import annotations

from pathlib import Path

from aiida_pino.data.pino_dataset_collection import PinoDatasetCollectionData, derive_simulation_arrays
from test_deposit_restore import _deposited_sim


def test_recorded_shapes_are_what_the_arrays_give(aiida_profile_clean, tmp_path):
    from aiida_pino.utils.dataset_derivation import DEFAULT_DERIVATION_PROFILE, IDENTITY_DERIVATION_VALUES

    sim, _ = _deposited_sim(Path(tmp_path))
    arrays, times = derive_simulation_arrays(sim, dict(DEFAULT_DERIVATION_PROFILE) | dict(IDENTITY_DERIVATION_VALUES))
    derived = {name: list(value.shape) for name, value in arrays.items()}
    recorded = PinoDatasetCollectionData._recorded_member_shapes(sim)
    assert recorded["array_shapes"] == derived
    assert recorded["spatial_shape"] == list(next(iter(arrays.values())).shape[-3:])
    assert recorded["num_timesteps"] == int(times.shape[0])


def test_composing_from_evicted_members_records_the_same_collection(aiida_profile_clean, tmp_path):
    a, _ = _deposited_sim(Path(tmp_path), case_name="case_a")
    b, _ = _deposited_sim(Path(tmp_path), case_name="case_b")
    live = PinoDatasetCollectionData.from_simulations([a, b], metadata={"source_folder_name": "live"})
    a.evict_data(reason="deposit")
    b.evict_data(reason="deposit")
    evicted = PinoDatasetCollectionData.from_simulations([a, b], metadata={"source_folder_name": "live"})
    for key in ("array_shapes", "spatial_shape", "num_timesteps", "content_signature"):
        assert evicted.base.attributes.get("dataset_metadata")[key] == live.base.attributes.get("dataset_metadata")[key], key
