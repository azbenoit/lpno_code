from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal env fallback.
    torch = None

from lpno.models.pipeline.lazy_runtime import (
    ArrayDirCollectionStore,
    ArrayDirDatasetStore,
    ArrayDirSimulationStore,
    ArrayDirUnionStore,
    MaskedTemporalPatchDataset,
    apply_random_mask,
)


pytestmark = pytest.mark.skipif(torch is None, reason="torch is required for lazy runtime tests")


def _write_runtime_dataset(root: Path, *, offset: float = 0.0, sample_count: int = 2) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x": np.linspace(0.0, 1.0, sample_count * 3 * 1 * 4 * 4 * 4, dtype=np.float32).reshape(
            sample_count, 3, 1, 4, 4, 4
        )
        + offset,
        "Temperature": np.full((sample_count, 3, 1, 4, 4, 4), 10.0 + offset, dtype=np.float32),
        "alpha": np.full((sample_count, 3, 1, 4, 4, 4), 0.25 + offset, dtype=np.float32),
        "P": np.full((sample_count, 3, 1, 4, 4, 4), 2.0 + offset, dtype=np.float32),
        "V_scan": np.full((sample_count, 3, 1, 4, 4, 4), 3.0 + offset, dtype=np.float32),
    }
    entries = []
    aliases = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(root / filename, array)
        entries.append(
            {
                "original_name": name,
                "stored_name": name,
                "filename": filename,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
        aliases[name] = name
    manifest = {
        "format": "aiida_arraydir_v1",
        "source_kind": "dataset",
        "dataset_uuid": f"uuid-{offset}",
        "dataset_pk": int(offset) + 1,
        "dataset_identifier": f"dataset-{offset}",
        "family_identifier": "family",
        "dataset_role": "base",
        "canonical_layout": "time_dim_noavg_noref",
        "source_type": "flow3d",
        "array_aliases": aliases,
        "array_entries": entries,
        "dataset_metadata": {
            "time_axis": True,
            "time_axis_mode": {"include_time": True, "time_as_batch": False, "time_as_dim": True},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return root


def _write_runtime_simulation(root: Path, *, offset: float = 0.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "P": np.full((3, 1, 4, 4, 4), 2.0 + offset, dtype=np.float32),
        "V_scan": np.full((3, 1, 4, 4, 4), 3.0 + offset, dtype=np.float32),
        "Temperature": np.full((3, 1, 4, 4, 4), 10.0 + offset, dtype=np.float32),
    }
    entries = []
    aliases = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(root / filename, array)
        entries.append(
            {
                "original_name": name,
                "stored_name": name,
                "filename": filename,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
        aliases[name] = name
    np.save(root / "time_values_s.npy", np.asarray([0.0, 1.0, 2.0], dtype=np.float32))
    manifest = {
        "format": "aiida_arraydir_v2",
        "source_kind": "simulation",
        "simulation_uuid": f"sim-{offset}",
        "simulation_pk": int(offset) + 1,
        "simulation_identifier": f"simulation-{offset}",
        "source_type": "flow3d",
        "array_aliases": aliases,
        "array_entries": entries,
        "time_values_filename": "time_values_s.npy",
        "simulation_metadata": {"source_type": "flow3d"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return root


def test_arraydir_dataset_store_reads_temporal_case_slices(tmp_path: Path) -> None:
    store = ArrayDirDatasetStore(_write_runtime_dataset(tmp_path / "dataset"))

    assert store.sample_count() == 2
    assert store.time_steps() == 3
    assert store.spatial_shape() == (4, 4, 4)

    sample = store.read_case(1, time_idx=2, spatial_slices=(slice(1, 3), slice(0, 2), slice(2, 4)), keys=["x", "Temperature"])

    assert sample["x"].shape == (1, 1, 2, 2, 2)
    assert sample["Temperature"].shape == (1, 1, 2, 2, 2)
    assert np.isclose(float(sample["Temperature"].mean()), 10.0)


def test_arraydir_union_store_flattens_members_without_concatenating(tmp_path: Path) -> None:
    first = _write_runtime_dataset(tmp_path / "members" / "member_000", offset=0.0, sample_count=1)
    second = _write_runtime_dataset(tmp_path / "members" / "member_001", offset=5.0, sample_count=2)
    manifest = {
        "format": "aiida_arraydir_v1",
        "source_kind": "dataset_union",
        "union_identifier": "union",
        "union_name": "union",
        "compatibility_signature": "sig",
        "total_samples": 3,
        "members": [
            {
                "relative_path": str(first.relative_to(tmp_path)),
                "manifest_file": str(first.relative_to(tmp_path) / "manifest.json"),
                "sample_offset": 0,
                "sample_count": 1,
            },
            {
                "relative_path": str(second.relative_to(tmp_path)),
                "manifest_file": str(second.relative_to(tmp_path) / "manifest.json"),
                "sample_offset": 1,
                "sample_count": 2,
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    store = ArrayDirUnionStore(tmp_path)
    assert store.sample_count() == 3
    sample = store.read_case(2, time_idx=0, keys=["Temperature"])
    assert sample["Temperature"].shape == (1, 1, 4, 4, 4)
    assert np.isclose(float(sample["Temperature"].mean()), 15.0)
    assert store.array_shape("Temperature") == (3, 3, 1, 4, 4, 4)


def test_arraydir_collection_store_reports_shapes_and_scalars(tmp_path: Path) -> None:
    first = _write_runtime_simulation(tmp_path / "members" / "simulation_0000", offset=0.0)
    second = _write_runtime_simulation(tmp_path / "members" / "simulation_0001", offset=5.0)
    manifest = {
        "format": "aiida_arraydir_v2",
        "source_kind": "dataset_collection",
        "collection_uuid": "collection",
        "collection_pk": 1,
        "dataset_identifier": "collection",
        "source_type": "flow3d",
        "simulation_count": 2,
        "members": [
            {
                "relative_path": str(first.relative_to(tmp_path)),
                "manifest_file": str(first.relative_to(tmp_path) / "manifest.json"),
                "case_index": 0,
                "simulation_pk": 1,
                "simulation_uuid": "sim-0",
            },
            {
                "relative_path": str(second.relative_to(tmp_path)),
                "manifest_file": str(second.relative_to(tmp_path) / "manifest.json"),
                "case_index": 1,
                "simulation_pk": 2,
                "simulation_uuid": "sim-1",
            },
        ],
        "dataset_metadata": {"source_type": "flow3d"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    simulation = ArrayDirSimulationStore(first)
    assert np.isclose(simulation.scalar_value("P", 0, time_idx=1), 2.0)

    collection = ArrayDirCollectionStore(tmp_path)
    assert collection.array_shape("Temperature") == (2, 3, 1, 4, 4, 4)
    assert np.isclose(collection.scalar_value("P", 1, time_idx=1), 7.0)


def test_masked_temporal_patch_dataset_returns_mask_channels() -> None:
    def _build_patch_sample(*, rng, case_indices):
        del rng, case_indices
        invar = {
            "T_prev": np.ones((1, 1, 4, 4, 4), dtype=np.float32),
            "T_prev_mask": np.zeros((1, 1, 4, 4, 4), dtype=np.float32),
        }
        outvar = {"T": np.full((1, 1, 4, 4, 4), 2.0, dtype=np.float32)}
        return invar, outvar

    dataset = MaskedTemporalPatchDataset(
        store=None,  # type: ignore[arg-type]
        case_indices=[0],
        build_patch_sample=_build_patch_sample,
        length=2,
        seed=7,
        invar_keys=["T_prev", "T_prev_mask"],
        outvar_keys=["T"],
    )

    invar, outvar, lambda_weighting = dataset[0]

    assert set(invar) == {"T_prev", "T_prev_mask"}
    assert invar["T_prev"].shape == torch.Size([1, 4, 4, 4])
    assert outvar["T"].shape == torch.Size([1, 4, 4, 4])
    assert torch.allclose(lambda_weighting["T"], torch.ones_like(outvar["T"]))


def test_apply_random_mask_block_masks_subset() -> None:
    rng = np.random.default_rng(5)
    array = np.ones((1, 8, 8, 8), dtype=np.float32)
    masked, mask = apply_random_mask(array, rng=rng, mask_mode="block", mask_ratio=0.25)

    assert masked.shape == array.shape
    assert mask.shape == array.shape
    assert float(mask.sum()) > 0.0
    assert np.all(masked[mask.astype(bool)] == 0.0)
