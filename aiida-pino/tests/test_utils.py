"""Tests for utility helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aiida import orm

from aiida_pino.utils import (
    dataset_collection_disk_usage_bytes,
    get_or_create_remote_data,
    get_or_create_dataset_union,
    export_dataset_union,
    merge_model_training_with_evaluation_profile,
    merge_training_profile,
    parse_model_errors,
    parse_stdout_metric_dicts,
    summarize_dataset_node,
    summarize_model_node,
    validate_dataset_input_port,
    validate_evaluation_profile,
    validate_profile_keys,
    validate_training_profile,
)


def test_validate_profile_keys_accepts_required_mapping() -> None:
    validate_profile_keys("profile", {"a": 1, "b": 2}, {"a", "b"})


def test_validate_profile_keys_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        validate_profile_keys("profile", {"a": 1}, {"a", "b"})


def test_parse_model_errors_reads_nested_metric_file(tmp_path: Path) -> None:
    path = tmp_path / "model_errors.txt"
    path.write_text("[T]\nrmse: 0.1\n\n[alpha]\niou_mean: 0.9\n", encoding="utf-8")

    assert parse_model_errors(path) == {"T": {"rmse": 0.1}, "alpha": {"iou_mean": 0.9}}


def test_parse_stdout_metric_dicts_reads_stdout_style_metrics() -> None:
    text = """
    some unrelated line
    T: {'rmse': 0.1, 'rel_rmse': 0.2}
    alpha: {'iou_mean': 0.9}
    ignored: {'not_numeric': 'x'}
    """

    assert parse_stdout_metric_dicts(text) == {
        "T": {"rmse": 0.1, "rel_rmse": 0.2},
        "alpha": {"iou_mean": 0.9},
    }


def test_merge_training_profile_loads_packaged_defaults() -> None:
    merged = merge_training_profile(
        {
            "config_name": "config_NS_energy_PINO_flow3D",
            "run_name": "run",
            "hydra_overrides": [],
            "cuda_device": "0",
            "wandb_offline": True,
        }
    )
    assert merged["config_name"] == "config_NS_energy_PINO_flow3D"
    assert merged["inference_batch_size"] > 0
    assert "autoregressive" in merged


def test_validate_training_profile_rejects_autoregressive_with_pde_loss() -> None:
    with pytest.raises(ValueError, match="data-only training"):
        validate_training_profile(
            {
                "config_name": "config_NS_energy_PINO_flow3D",
                "run_name": "run",
                "hydra_overrides": [],
                "cuda_device": "0",
                "wandb_offline": True,
                "time_as_batch": True,
                "pde_loss": True,
                "autoregressive": {"enabled": True},
            }
        )


def _ar_profile(mode: str) -> dict:
    return {
        "config_name": "config_flow3D_RIGNO_randompath",
        "run_name": "run",
        "hydra_overrides": [],
        "cuda_device": "0",
        "wandb_offline": True,
        "time_as_batch": True,
        "autoregressive": {"enabled": True, "mode": mode},
    }


def test_validate_training_profile_accepts_autoregressive_rollout_mode() -> None:
    """`rollout` is the mode the lpno trainer supports for
    fno/rigno/deeponet (BUG-20). Rejecting it here made every AR rollout
    training unreachable through AiiDA, which is the default run path."""
    profile = validate_training_profile(_ar_profile("rollout"))

    assert profile["autoregressive"]["mode"] == "rollout"


def test_validate_training_profile_still_accepts_masked_patch() -> None:
    profile = validate_training_profile(_ar_profile("masked_patch"))

    assert profile["autoregressive"]["mode"] == "masked_patch"


def test_validate_training_profile_rejects_unknown_autoregressive_mode() -> None:
    with pytest.raises(ValueError, match="rollout"):
        validate_training_profile(_ar_profile("bogus_mode"))


def test_get_or_create_remote_data_reuses_existing(aiida_localhost, tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    node, created = get_or_create_remote_data(
        root,
        aiida_localhost,
        kind="dataset",
        description="test dataset",
    )
    reused, created_again = get_or_create_remote_data(
        root,
        aiida_localhost,
        kind="dataset",
        description="test dataset",
    )
    assert created is True
    assert created_again is False
    assert reused.pk == node.pk


def test_summarize_dataset_node_native_dataset(pino_dataset_factory) -> None:
    dataset = pino_dataset_factory("summary-dataset")
    summary = summarize_dataset_node(dataset)
    assert summary["source_type"] == "flow3d"
    assert summary["dataset_identifier"]
    assert "Temperature" in summary["array_names"]
    assert summary["stored_bytes"] > 0
    assert summary["stored_files"] > 0
    assert summary["stored_size"] != "0 B"


def test_dataset_collection_disk_usage_tolerates_dangling_member(pino_dataset_factory) -> None:
    """A deleted member leaves a dangling UUID; the browser must not crash on it."""
    from aiida.tools import delete_nodes

    dataset = pino_dataset_factory("dangling-member")
    member_pk = orm.load_node(dataset.get_member_simulation_uuids()[0]).pk
    delete_nodes([member_pk], dry_run=False)

    stored_bytes, stored_files = dataset_collection_disk_usage_bytes(dataset)
    assert stored_bytes == 0
    assert stored_files == 0
    # summarize_dataset_node feeds the (unguarded) dataset browser loops.
    summary = summarize_dataset_node(dataset)
    assert summary["stored_size"] == "0 B"


def test_summarize_derived_collection_reports_zero_stored_bytes(pino_dataset_factory) -> None:
    """Derived collections share their base's simulations, so they own no disk."""
    derived = pino_dataset_factory("derived-summary", role="derived")
    summary = summarize_dataset_node(derived)
    assert summary["dataset_role"] == "derived"
    assert summary["stored_bytes"] == 0
    assert summary["stored_files"] == 0
    assert summary["stored_size"] == "0 B"


def test_validate_dataset_input_port_accepts_collection_and_folder(pino_dataset_factory) -> None:
    assert validate_dataset_input_port(pino_dataset_factory("port-ok")) is None
    assert validate_dataset_input_port(orm.FolderData().store()) is None


def test_validate_dataset_input_port_rejects_wrong_type() -> None:
    message = validate_dataset_input_port(orm.Dict(dict={"not": "a dataset"}).store())
    assert message is not None
    assert "PinoDatasetCollectionData" in message


def test_get_or_create_dataset_union_reuses_existing_members(
    legacy_pino_dataset_factory,
    pino_dataset_union_factory,
) -> None:
    first = legacy_pino_dataset_factory("union-summary-a")
    second = legacy_pino_dataset_factory(
        "union-summary-b",
        metadata={"family_identifier": first.get_family_identifier()},
    )
    union = pino_dataset_union_factory(first, second, union_name="train_plus_val")
    reused, created = get_or_create_dataset_union([first, second], union_name="train_plus_val")

    assert created is False
    assert reused.pk == union.pk
    summary = summarize_dataset_node(union)
    assert summary["dataset_role"] == "union"
    assert summary["num_samples"] == 2


def test_export_dataset_union_concatenates_member_samples(
    tmp_path: Path,
    legacy_pino_dataset_factory,
    pino_dataset_union_factory,
) -> None:
    first = legacy_pino_dataset_factory(
        "export-a",
        arrays={"Temperature": np.zeros((1, 2, 1, 4, 4, 4), dtype=np.float32), "alpha": np.ones((1, 2, 1, 4, 4, 4), dtype=np.float32)},
    )
    second = legacy_pino_dataset_factory(
        "export-b",
        arrays={"Temperature": np.full((2, 2, 1, 4, 4, 4), 3.0, dtype=np.float32), "alpha": np.zeros((2, 2, 1, 4, 4, 4), dtype=np.float32)},
        metadata={
            "family_identifier": first.get_family_identifier(),
            "sample_metadata": [
                {
                    "group": "export-b",
                    "name": "results_0.vtk",
                    "relative_path": "results_0.vtk",
                    "stem": "results_0",
                    "params": {"time_s": 0.0},
                },
                {
                    "group": "export-b",
                    "name": "results_1.vtk",
                    "relative_path": "results_1.vtk",
                    "stem": "results_1",
                    "params": {"time_s": 1.0},
                },
            ],
        },
    )
    union = pino_dataset_union_factory(first, second, union_name="export_union")

    manifest_path = export_dataset_union(union, tmp_path)

    assert manifest_path.name == "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "aiida_arraydir_v1"
    assert manifest["source_kind"] == "dataset_union"
    assert manifest["total_samples"] == 3
    assert len(manifest["members"]) == 2
    first_member = tmp_path / manifest["members"][0]["relative_path"]
    second_member = tmp_path / manifest["members"][1]["relative_path"]
    assert (first_member / "manifest.json").is_file()
    assert (second_member / "manifest.json").is_file()
    first_temperature = np.load(first_member / "Temperature.npy", mmap_mode="r", allow_pickle=False)
    second_temperature = np.load(second_member / "Temperature.npy", mmap_mode="r", allow_pickle=False)
    assert first_temperature.shape[0] == 1
    assert second_temperature.shape[0] == 2
    assert float(second_temperature[1, 0, 0, 0, 0, 0]) == 3.0


def test_merge_model_training_with_evaluation_profile_preserves_training_defaults(
    pino_dataset_factory,
    pino_model_factory,
) -> None:
    dataset = pino_dataset_factory("model-utils")
    model = pino_model_factory("model-utils", training_source=dataset)

    merged = merge_model_training_with_evaluation_profile(
        model,
        {"run_name": "fresh-eval", "inference_batch_size": 8, "autoregressive": {"eval_horizon_k": 4}},
        dataset_metadata=dataset.get_dataset_metadata(),
    )

    assert merged["run_name"] == "fresh-eval"
    assert merged["config_name"] == "config_NS_energy_PINO_flow3D"
    assert merged["inference_batch_size"] == 8
    assert merged["time_as_dim"] is True
    assert merged["autoregressive"]["eval_horizon_k"] == 4


def test_validate_evaluation_profile_ignores_autoregressive_enabled_key() -> None:
    profile = validate_evaluation_profile(
        {
            "run_name": "eval",
            "autoregressive": {
                "enabled": True,
                "eval_horizon_k": 6,
                "eval_use_gt_init": False,
                "require_time": True,
                "state_prev_suffix": "_prev",
            },
        }
    )

    assert "enabled" not in profile["autoregressive"]
    assert profile["autoregressive"]["eval_horizon_k"] == 6


def test_summarize_model_node_includes_evaluation_history(
    pino_dataset_factory,
    pino_model_factory,
    pino_model_evaluation_factory,
) -> None:
    dataset = pino_dataset_factory("model-summary")
    model = pino_model_factory("model-summary", training_source=dataset)
    pino_model_evaluation_factory(model=model, dataset_source=dataset, evaluation_name="summary-eval")

    summary = summarize_model_node(model)

    assert summary["model_name"] == "model-summary"
    assert summary["evaluation_count"] == 1
    assert summary["evaluations"][0]["evaluation_name"] == "summary-eval"


def test_resolve_vtk_geometry_finds_vtk_files_in_subdirectories(monkeypatch, tmp_path: Path) -> None:
    """BUG-08: multi-timestep FLOW-3D conversion writes per-case subdirs; inference must find them."""
    from aiida_pino.utils import vtk_geometry as openfoam_import

    (tmp_path / "case_a").mkdir()
    (tmp_path / "case_a" / "frame_0.vtk").write_text("vtk", encoding="utf-8")

    class FakeDatasource:
        @staticmethod
        def read_file(path):
            xs, ys, zs = np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], indexing="ij")
            return {"coordinates": np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)}

    monkeypatch.setattr(openfoam_import, "_load_vtk_datasource_class", lambda: FakeDatasource)

    geometry = openfoam_import.resolve_vtk_geometry({}, vtk_dir=tmp_path, flip_axes=False)

    assert geometry["resolution"] == [2, 2, 2]
    assert geometry["x_range"] == [0.0, 1.0]


def test_resolve_flip_axes_defaults_by_source_type() -> None:
    """BUG-11: flip_axes absent from the profile resolves by source, not by KeyError/False."""
    from aiida_pino.utils.vtk_geometry import resolve_flip_axes

    assert resolve_flip_axes({}, "flow3d") is False
    assert resolve_flip_axes({}, "openfoam") is True
    assert resolve_flip_axes({}, "openfoam_raw") is True
    assert resolve_flip_axes({"flip_axes": False}, "openfoam") is False
    assert resolve_flip_axes({"flip_axes": True}, "flow3d") is True


def test_dataset_profile_keys_allow_missing_flip_axes() -> None:
    """BUG-11: flip_axes is optional so the source-aware default is reachable."""
    from aiida_pino.utils import DATASET_PROFILE_KEYS

    validate_profile_keys("dataset_profile", {"geometry": {}, "vtk_fields": []}, DATASET_PROFILE_KEYS)
