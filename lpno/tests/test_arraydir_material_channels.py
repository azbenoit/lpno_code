"""ArrayDir runtime stores must synthesize material channels like NpzDirStore.

AiiDA-staged collections (runtime arraydir layout) carry the FLOW-3D case
manifest in each member's simulation_metadata, but read_case only returned the
stored arrays — a material-conditioned config (mat_rho ... mat_A input keys)
then died with KeyError 'mat_rho' at input assembly (alsi round-0 trainings,
2026-07-20). The stores now build the constant material fields from the
member manifest, mirroring npz_datasource._build_material_channels.
"""

import json

import numpy as np
import pytest

from lpno.models.pipeline.lazy_runtime import (
    RUNTIME_FORMAT_V2,
    ArrayDirCollectionStore,
    ArrayDirSimulationStore,
)

MATERIAL = {
    "properties": {
        "density_kg_m3": 4420.0,
        "specific_heat_j_kg_k": 546.0,
        "thermal_conductivity_w_m_k": 7.0,
        "latent_heat_j_kg": 345000.0,
        "liquidus_temperature_k": 1923.15,
    },
    "optical": {"normal_absorptivity": 0.35},
}


def _write_member(root, with_material=True):
    root.mkdir(parents=True, exist_ok=True)
    field = np.random.rand(1, 4, 3, 2).astype(np.float32)
    np.save(root / "Temperature.npy", field)
    metadata = {"case_name": "case_a"}
    if with_material:
        metadata["flow3d_case_manifest"] = {"input": {}, "material": MATERIAL}
    manifest = {
        "format": RUNTIME_FORMAT_V2,
        "source_kind": "simulation",
        "simulation_metadata": metadata,
        "array_entries": [
            {"original_name": "Temperature", "filename": "Temperature.npy", "shape": list(field.shape)}
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return field


def test_simulation_store_builds_material_channels(tmp_path):
    _write_member(tmp_path)
    store = ArrayDirSimulationStore(tmp_path)
    payload = store.read_case(0)
    assert "mat_rho" in payload and "mat_A" in payload
    assert payload["mat_rho"].shape == payload["Temperature"].shape
    assert np.allclose(payload["mat_rho"], 4420.0)
    assert np.allclose(payload["mat_A"], 0.35)


def test_material_channels_survive_explicit_keys(tmp_path):
    _write_member(tmp_path)
    store = ArrayDirSimulationStore(tmp_path)
    payload = store.read_case(0, keys=["Temperature", "mat_rho"])
    assert set(payload) >= {"Temperature", "mat_rho"}


def test_no_manifest_means_no_material_channels(tmp_path):
    _write_member(tmp_path, with_material=False)
    store = ArrayDirSimulationStore(tmp_path)
    payload = store.read_case(0)
    assert "mat_rho" not in payload
    with pytest.raises(KeyError):
        store.read_case(0, keys=["mat_rho"])


def test_collection_store_propagates_material_channels(tmp_path):
    member = tmp_path / "members" / "simulation_0000"
    _write_member(member)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": RUNTIME_FORMAT_V2,
        "source_kind": "dataset_collection",
        "dataset_metadata": {},
        "members": [{"relative_path": "members/simulation_0000"}],
    }))
    store = ArrayDirCollectionStore(tmp_path)
    payload = store.read_case(0)
    assert np.allclose(payload["mat_cp"], 546.0)
