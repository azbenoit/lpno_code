"""Tests for the native PINO dataset data node."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aiida_pino.data import (
    PinoDatasetCollectionData,
    PinoDatasetData,
    PinoSimulationData,
    build_dataset_identifier,
    build_dataset_label,
    datasource_cache_basename_from_metadata,
)
from aiida_pino.utils import apply_dataset_layout_to_training_profile


def test_build_dataset_label_and_identifier() -> None:
    label = build_dataset_label("cases", "var_1_time_steps2_avg2.npz", "flow3d")
    identifier = build_dataset_identifier("cases", "var_1_time_steps2_avg2.npz", "flow3d")
    assert label == "cases | var_1_time_steps2_avg2 | flow3d"
    assert "cases" in identifier
    assert "flow3d" in identifier


def test_dataset_defaults_source_type_to_flow3d() -> None:
    dataset = PinoDatasetData.from_arrays(
        {"Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float32)},
        metadata={"source_folder_name": "cases", "cache_basename": "var_1.npz"},
    )
    assert dataset.base.attributes.get("source_type") == "flow3d"


def test_simulation_empty_time_values_falls_back_to_array_time_dim() -> None:
    simulation = PinoSimulationData.from_arrays(
        {"Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32)},
        metadata={"source_folder_name": "cases", "time_values_s": []},
    )
    assert simulation.get_num_timesteps() == 2
    assert simulation.get_time_values_s() is None


def test_simulation_time_values_length_mismatch_raises() -> None:
    arrays = {"Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match=r"3 entries.*2 time steps"):
        PinoSimulationData.from_arrays(dict(arrays), metadata={"time_values_s": [0.0, 1.0, 2.0]})
    simulation = PinoSimulationData.from_arrays(dict(arrays), metadata={"time_values_s": [0.0, 1.0]})
    assert simulation.get_num_timesteps() == 2


def test_simulation_annotations_default_and_promote_to_attributes_and_extras() -> None:
    arrays = {"Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32)}
    default_sim = PinoSimulationData.from_arrays(dict(arrays), metadata={"time_values_s": [0.0, 1.0]})
    assert default_sim.base.attributes.get("description") == ""
    assert default_sim.base.attributes.get("batch_keyword") == ""
    assert default_sim.base.attributes.get("usable_for_training") is True
    assert default_sim.base.extras.get("usable_for_training") is True

    sim = PinoSimulationData.from_arrays(
        dict(arrays),
        metadata={
            "time_values_s": [0.0, 1.0],
            "description": "melt sweep, LHS point 12",
            "batch_keyword": "lhs_melt_2026",
            "usable_for_training": False,
        },
    )
    assert sim.base.attributes.get("description") == "melt sweep, LHS point 12"
    assert sim.base.attributes.get("batch_keyword") == "lhs_melt_2026"
    assert sim.base.attributes.get("usable_for_training") is False
    assert sim.base.extras.get("batch_keyword") == "lhs_melt_2026"
    meta = sim.get_simulation_metadata()
    assert meta["description"] == "melt sweep, LHS point 12"
    assert meta["batch_keyword"] == "lhs_melt_2026"
    assert meta["usable_for_training"] is False
    # Annotations are not part of identity: same fingerprint as the un-annotated node.
    assert sim.get_simulation_fingerprint() == default_sim.get_simulation_fingerprint()


def test_usable_for_training_extra_is_mutable_after_promotion() -> None:
    arrays = {"Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32)}
    sim = PinoSimulationData.from_arrays(dict(arrays), metadata={"time_values_s": [0.0, 1.0]})
    assert sim.get_simulation_metadata()["usable_for_training"] is True
    sim.base.extras.set("usable_for_training", False)
    assert sim.get_simulation_metadata()["usable_for_training"] is False


def test_derived_collection_num_timesteps_is_not_channel_count(pino_simulation_factory, tmp_path) -> None:
    # Member without stored time values, 3 channels: dropping time yields a
    # (C, X, Y, Z) layout whose leading axis is channels, not time.
    simulation = pino_simulation_factory(
        "no-time-values",
        arrays={"Temperature": np.ones((2, 3, 4, 4, 4), dtype=np.float32)},
        metadata={"time_values_s": None},
    )
    profile = {
        "include_time": False,
        "time_as_batch": False,
        "time_as_dim": False,
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "time_avg_window": 1,
        "super_res_factor": 1,
    }
    collection = PinoDatasetCollectionData.from_simulations(
        [simulation],
        metadata={
            "dataset_role": "derived",
            "time_axis": False,
            "time_axis_mode": {"include_time": False, "time_as_batch": False, "time_as_dim": False},
            "derivation_profile": profile,
        },
    )
    assert collection.get_dataset_metadata()["num_timesteps"] == 1

    collection.export_runtime_arraydir(tmp_path / "runtime")
    member_manifest = json.loads(
        (tmp_path / "runtime" / "members" / "simulation_0000" / "manifest.json").read_text(encoding="utf-8")
    )
    assert member_manifest["simulation_metadata"]["num_timesteps"] == 1


def test_build_derived_collection_metadata_num_timesteps_is_not_channel_count(pino_simulation_factory) -> None:
    # Same BUG-19 rule for the third site: build_derived_collection_metadata's
    # representative fallback must not report the channel axis of a 4D layout.
    from aiida_pino.utils.dataset_derivation import build_derived_collection_metadata

    simulation = pino_simulation_factory(
        "no-time-values-derive",
        arrays={"Temperature": np.ones((2, 3, 4, 4, 4), dtype=np.float32)},
        metadata={"time_values_s": None},
    )
    base = PinoDatasetCollectionData.from_simulations(
        [simulation], metadata={"dataset_role": "base"}
    ).store()
    profile = {
        "include_time": False,
        "time_as_batch": False,
        "time_as_dim": False,
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "time_avg_window": 1,
        "super_res_factor": 1,
    }
    metadata = build_derived_collection_metadata(base, profile)
    assert metadata["num_timesteps"] == 1


def test_unknown_source_type_is_rejected_and_openfoam_raw_accepted(pino_simulation_factory) -> None:
    arrays = {"Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match="Unsupported source_type"):
        PinoDatasetData.from_arrays(dict(arrays), metadata={"source_type": "flowd3"})
    with pytest.raises(ValueError, match="Unsupported source_type"):
        PinoSimulationData.from_arrays(dict(arrays), metadata={"source_type": "flowd3"})

    simulation = pino_simulation_factory("raw-of", source_type="openfoam_raw")
    assert simulation.base.attributes.get("source_type") == "openfoam_raw"
    with pytest.raises(ValueError, match="Unsupported source_type"):
        PinoDatasetCollectionData.from_simulations([simulation], metadata={"source_type": "flowd3"})
    collection = PinoDatasetCollectionData.from_simulations([simulation], metadata={})
    assert collection.base.attributes.get("source_type") == "openfoam_raw"


def test_dataset_preserves_explicit_source_type_and_exports_npz(tmp_path) -> None:
    dataset = PinoDatasetData.from_arrays(
        {
            "Temperature": np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2),
            "Marangoni Vector": np.ones((1, 3, 2, 2, 2), dtype=np.float32),
        },
        metadata={
            "source_type": "openfoam",
            "source_folder_name": "laser_cases",
            "cache_basename": "var_1_time_steps2_avg1.npz",
            "requested_fields": ["Temperature", "Marangoni Vector"],
            "input_dir": "/tmp/laser_cases",
            "source_uri": "file:///tmp/laser_cases",
        },
        source={"uri": "file:///tmp/laser_cases", "description": "test"},
    )
    aliases = dataset.get_array_aliases()
    assert aliases["Marangoni Vector"] != "Marangoni Vector"
    assert dataset.base.attributes.get("source_type") == "openfoam"

    npz_path = dataset.export_npz_dataset(tmp_path)
    assert npz_path.name == "var_1_time_steps2_avg1.npz"
    with np.load(npz_path) as archive:
        assert "Temperature" in archive.files
        assert "Marangoni Vector" in archive.files

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_type"] == "openfoam"
    assert "dataset_metadata" in manifest


def test_dataset_can_be_created_directly_from_npz_file(tmp_path) -> None:
    npz_path = tmp_path / "cache.npz"
    np.savez_compressed(
        npz_path,
        Temperature=np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2),
        alpha=np.ones((1, 1, 2, 2, 2), dtype=np.float32),
    )

    dataset = PinoDatasetData.from_npz_file(
        npz_path,
        metadata={
            "source_type": "flow3d",
            "source_folder_name": "cases",
            "cache_basename": "var_1_time_dim_base.npz",
            "requested_fields": ["Temperature", "alpha"],
            "input_dir": "/tmp/cases",
            "source_uri": "file:///tmp/cases",
        },
        source={"uri": "file:///tmp/cases", "description": "test"},
    )

    assert dataset.get_array_shapes()["Temperature"] == [1, 1, 2, 2, 2]
    assert dataset.get_array_by_original_name("alpha").shape == (1, 1, 2, 2, 2)


def test_dataset_export_writes_runtime_cache_alias_for_legacy_base_basename(tmp_path) -> None:
    dataset = PinoDatasetData.from_arrays(
        {"Temperature": np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2)},
        metadata={
            "source_type": "flow3d",
            "source_folder_name": "cases",
            "cache_basename": "var_1_time_dim_base.npz",
            "requested_fields": ["Temperature"],
            "input_dir": "/tmp/cases",
            "source_uri": "file:///tmp/cases",
            "time_axis": True,
            "avg_window_size": 1,
            "moving_frame_steps": 1,
            "moving_ref_frame": False,
            "super_res_factor": 1,
            "time_axis_mode": {
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": True,
            },
        },
    )

    npz_path = dataset.export_npz_dataset(tmp_path)

    assert npz_path.name == "var_1_time_dim_base.npz"
    assert (tmp_path / "var_1_time_noref.npz").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_cache_file"] == "var_1_time_noref.npz"
    assert "var_1_time_noref.npz" in manifest["cache_files"]


def test_dataset_export_runtime_arraydir_writes_manifest_and_shards(tmp_path) -> None:
    dataset = PinoDatasetData.from_arrays(
        {
            "Temperature": np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2),
            "alpha": np.ones((1, 1, 2, 2, 2), dtype=np.float32),
        },
        metadata={
            "source_type": "openfoam",
            "source_folder_name": "laser_cases",
            "cache_basename": "var_1_time_noref.npz",
            "requested_fields": ["Temperature", "alpha"],
            "input_dir": "/tmp/laser_cases",
            "source_uri": "file:///tmp/laser_cases",
            "time_axis": True,
            "avg_window_size": 1,
            "moving_frame_steps": 1,
            "moving_ref_frame": False,
            "time_axis_mode": {
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": True,
            },
        },
    ).store()

    manifest_path = dataset.export_runtime_arraydir(tmp_path)

    assert manifest_path.name == "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == PinoDatasetData.RUNTIME_MANIFEST_VERSION
    assert manifest["source_kind"] == "dataset"
    assert manifest["source_type"] == "openfoam"
    assert sorted(manifest["array_aliases"]) == ["Temperature", "alpha"]
    filenames = {entry["filename"] for entry in manifest["array_entries"]}
    assert filenames == {"Temperature.npy", "alpha.npy"}
    for filename in filenames:
        assert (tmp_path / filename).is_file()
    loaded = np.load(tmp_path / "Temperature.npy", mmap_mode="r", allow_pickle=False)
    assert loaded.shape == (1, 1, 2, 2, 2)


def test_datasource_cache_basename_matches_moving_frame_profile() -> None:
    basename = datasource_cache_basename_from_metadata(
        {
            "super_res_factor": 1,
            "moving_ref_frame": True,
            "moving_frame_steps": 30,
            "avg_window_size": 30,
            "time_axis": True,
            "time_axis_mode": {
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": False,
            },
        }
    )

    assert basename == "var_1_time_steps30_avg30.npz"


def test_apply_dataset_layout_to_training_profile_overrides_conflicting_hydra_settings() -> None:
    profile = {
        "include_time": False,
        "time_as_batch": False,
        "time_as_dim": False,
        "moving_ref_frame": True,
        "moving_frame_steps": 12,
        "time_avg_window": 12,
        "use_2d": False,
        "hydra_overrides": [
            "custom.super_res=99",
            "++custom.time_as_batch=false",
            "custom.moving_ref_frame=true",
            "trainer.max_epochs=10",
        ],
    }
    metadata = {
        "time_axis": True,
        "time_axis_mode": {
            "include_time": True,
            "time_as_batch": True,
            "time_as_dim": False,
        },
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "avg_window_size": 1,
        "super_res_factor": 3,
    }

    resolved = apply_dataset_layout_to_training_profile(profile, metadata)

    assert resolved["include_time"] is True
    assert resolved["time_as_batch"] is True
    assert resolved["time_as_dim"] is False
    assert resolved["moving_ref_frame"] is False
    assert resolved["time_avg_window"] == 1
    assert "trainer.max_epochs=10" in resolved["hydra_overrides"]
    assert "custom.super_res=99" not in resolved["hydra_overrides"]
    assert "++custom.time_as_batch=false" not in resolved["hydra_overrides"]
    assert resolved["hydra_overrides"][-1] == "custom.super_res=3"
