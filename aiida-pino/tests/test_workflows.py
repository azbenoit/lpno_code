"""Tests for workflow validation and integration."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from aiida import orm
from aiida.engine import run_get_node

from aiida_pino.data import (
    PinoDatasetCollectionData,
    PinoKfoldRunData,
    PinoModelData,
    PinoModelEvaluationData,
    PinoSimulationData,
)
from aiida_pino.calculations.flow3d_run import Flow3DRunCalculation
from aiida_pino.calculations.vtk_to_dataset import VtkToDatasetCalculation
from aiida_pino.utils import load_training_profile_defaults
from aiida_pino.workflows.derive_dataset import DeriveDatasetWorkChain
from aiida_pino.workflows.evaluate_model import EvaluateModelWorkChain
from aiida_pino.workflows.eval_only import EvalOnlyWorkChain
from aiida_pino.workflows.flow3d_batch import Flow3DBatchWorkChain
from aiida_pino.workflows.flow3d_generate import Flow3DGenerateWorkChain
from aiida_pino.workflows.flow3d_ingest import Flow3DIngestWorkChain
from aiida_pino.workflows.full_flow3d_train_eval import FullFlow3DTrainEvalWorkChain
from aiida_pino.workflows.import_model import ImportModelWorkChain
from aiida_pino.workflows.kfold_train_eval import KfoldTrainEvalWorkChain
from aiida_pino.workflows.raw_flow3d_import import RawFlow3DImportWorkChain
from aiida_pino.workflows.train_eval import TrainEvalWorkChain
from aiida_pino.workflows.train_model import TrainModelWorkChain


def _dataset_profile():
    return orm.Dict(
        dict={
            "geometry": {"x_range": [0, 1], "y_range": [0, 1], "z_range": [0, 1], "resolution": [4, 4, 4]},
            "vtk_fields": ["Temperature"],
            "flip_axes": False,
        }
    )


def _derivation_profile():
    return orm.Dict(
        dict={
            "include_time": True,
            "time_as_batch": True,
            "time_as_dim": False,
            "moving_ref_frame": False,
            "moving_frame_steps": 1,
            "time_avg_window": 1,
            "super_res_factor": 1,
        }
    )


def _conversion_profile():
    return orm.Dict(dict={"case_regex": ".*", "num_timesteps": 2, "output_basename_mode": "legacy"})


def _training_profile(mode: str = "standard"):
    profile = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    # Manual reservation: these tests exercise that path; the automatic VRAM path is covered by tests/test_vram_auto.py.
    profile["gpu_vram_mb"] = 8000
    profile["run_name"] = "run"
    profile["hydra_overrides"] = []
    profile["split_mode"] = mode
    profile["include_time"] = True
    profile["time_as_batch"] = False
    profile["time_as_dim"] = True
    profile["moving_ref_frame"] = False
    profile["moving_frame_steps"] = 1
    profile["time_avg_window"] = 1
    profile["autoregressive"]["enabled"] = False
    return orm.Dict(dict=profile)


def test_flow3d_ingest_rejects_empty_source_path(local_code_factory) -> None:
    pvpython = local_code_factory("/bin/bash", label="pv-empty")
    python = local_code_factory("/bin/bash", label="py-empty")
    _, node = run_get_node(
        Flow3DIngestWorkChain,
        source_path=orm.Str(""),
        conversion_profile=_conversion_profile(),
        dataset_profile=_dataset_profile(),
        pvpython_code=pvpython,
        python_code=python,
    )
    assert node.exit_status == Flow3DIngestWorkChain.exit_codes.ERROR_EMPTY_SOURCE_PATH.status


def test_flow3d_ingest_rejects_invalid_profile(local_code_factory) -> None:
    pvpython = local_code_factory("/bin/bash", label="pv-invalid-profile")
    python = local_code_factory("/bin/bash", label="py-invalid-profile")
    _, node = run_get_node(
        Flow3DIngestWorkChain,
        source_path=orm.Str("/tmp/raw"),
        conversion_profile=orm.Dict(dict={"case_regex": ".*"}),
        dataset_profile=_dataset_profile(),
        pvpython_code=pvpython,
        python_code=python,
    )
    assert node.exit_status == Flow3DIngestWorkChain.exit_codes.ERROR_INVALID_CONVERSION_PROFILE.status


def test_train_model_rejects_invalid_training_profile(local_code_factory, pino_dataset_factory) -> None:
    python = local_code_factory("/bin/bash", label="py-train-invalid")
    dataset = pino_dataset_factory("train-invalid")
    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict={"config_name": "x"}),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.exit_status == TrainModelWorkChain.exit_codes.ERROR_INVALID_TRAINING_PROFILE.status


def test_import_model_registers_remote_training_artifacts(
    pino_dataset_factory,
    remote_data_factory,
    tmp_path,
) -> None:
    dataset = pino_dataset_factory("import-remote")
    training_dir = tmp_path / "training-remote"
    training_dir.mkdir()
    (training_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")
    hydra_dir = training_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text("hydra: true", encoding="utf-8")
    training = remote_data_factory(training_dir, kind="training_artifacts")

    results, node = run_get_node(
        ImportModelWorkChain,
        training_artifacts=training,
        training_profile=_training_profile(),
        training_source=dataset,
        model_name=orm.Str("imported-model"),
    )
    assert node.is_finished_ok
    assert isinstance(results["model"], PinoModelData)
    metadata = results["model"].get_model_metadata()
    assert metadata["training_source_kind"] == "dataset_collection"
    assert metadata["training_source_pk"] == dataset.pk
    assert results["model"].label == "imported-model"


def test_full_workflow_requires_codes() -> None:
    with pytest.raises((TypeError, ValueError)):
        run_get_node(
            FullFlow3DTrainEvalWorkChain,
            source_path=orm.Str("/tmp/raw"),
            conversion_profile=_conversion_profile(),
            dataset_profile=_dataset_profile(),
            training_profile=_training_profile(),
        )


def test_flow3d_ingest_integration(fake_pvpython_executable, fake_python_executable, local_code_factory) -> None:
    pvpython = local_code_factory(str(fake_pvpython_executable), label="fake-pvpython")
    python = local_code_factory(str(fake_python_executable), label="fake-python-ingest")
    results, node = run_get_node(
        Flow3DIngestWorkChain,
        source_path=orm.Str("/tmp/raw"),
        conversion_profile=_conversion_profile(),
        dataset_profile=_dataset_profile(),
        pvpython_code=pvpython,
        python_code=python,
    )
    assert node.is_finished_ok
    assert "dataset" in results
    assert isinstance(results["dataset"], PinoDatasetCollectionData)
    assert results["ingest_summary"].get_dict()["conversion"]["converted_cases"][0]["case_name"] == "case_a"
    assert results["dataset"].base.attributes.get("source_type") == "flow3d"
    assert results["dataset"].base.attributes.get("dataset_role") == "base"


def test_flow3d_ingest_passes_case_manifests_as_source_metadata(
    fake_pvpython_executable, fake_python_executable, local_code_factory, tmp_path
) -> None:
    """BUG-02: the ingest must auto-read `case_manifest.json` per run directory."""
    source = tmp_path / "runs"
    for name, power in (("case_a", 100.0), ("case_b", 200.0)):
        case_dir = source / name
        case_dir.mkdir(parents=True)
        (case_dir / "case_manifest.json").write_text(
            json.dumps(
                {
                    "case_name": name,
                    "input": {
                        "laser": {"power_w": power, "scan_speed_m_per_s": 0.5, "beam_radius_m": 5e-5},
                        "simulation": {"finish_time_s": 1e-3},
                    },
                }
            ),
            encoding="utf-8",
        )

    pvpython = local_code_factory(str(fake_pvpython_executable), label="fake-pvpython-manifests")
    python = local_code_factory(str(fake_python_executable), label="fake-python-manifests")
    _, node = run_get_node(
        Flow3DIngestWorkChain,
        source_path=orm.Str(str(source)),
        conversion_profile=_conversion_profile(),
        dataset_profile=_dataset_profile(),
        pvpython_code=pvpython,
        python_code=python,
    )
    assert node.is_finished_ok

    dataset_calc = next(
        child for child in node.called_descendants if child.process_class is VtkToDatasetCalculation
    )
    manifests = dataset_calc.inputs.source_metadata.get_dict()["flow3d"]["case_manifests"]
    assert set(manifests) == {"case_a", "case_b"}
    assert manifests["case_a"]["laser_power"] == 100.0
    assert manifests["case_a"]["V_scan"] == 0.5
    assert manifests["case_b"]["laser_power"] == 200.0
    assert manifests["case_b"]["flow3d_case_manifest"]["input"]["laser"]["power_w"] == 200.0


def test_flow3d_ingest_user_source_metadata_merges_over_auto(
    fake_pvpython_executable, fake_python_executable, local_code_factory, tmp_path
) -> None:
    source = tmp_path / "runs"
    case_dir = source / "case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "case_manifest.json").write_text(
        json.dumps({"case_name": "case_a", "input": {"laser": {"power_w": 100.0, "scan_speed_m_per_s": 0.5}}}),
        encoding="utf-8",
    )

    pvpython = local_code_factory(str(fake_pvpython_executable), label="fake-pvpython-user-meta")
    python = local_code_factory(str(fake_python_executable), label="fake-python-user-meta")
    _, node = run_get_node(
        Flow3DIngestWorkChain,
        source_path=orm.Str(str(source)),
        conversion_profile=_conversion_profile(),
        dataset_profile=_dataset_profile(),
        source_metadata=orm.Dict(dict={"operator_note": "melt sweep", "flow3d": {"case_manifests": {}}}),
        pvpython_code=pvpython,
        python_code=python,
    )
    assert node.is_finished_ok

    dataset_calc = next(
        child for child in node.called_descendants if child.process_class is VtkToDatasetCalculation
    )
    payload = dataset_calc.inputs.source_metadata.get_dict()
    assert payload["operator_note"] == "melt sweep"
    # The user-supplied dict wins over the auto-scanned manifests.
    assert payload["flow3d"] == {"case_manifests": {}}


def test_derive_dataset_reuses_existing_variant(pino_dataset_factory) -> None:
    dataset = pino_dataset_factory("derive-base")
    results, node = run_get_node(
        DeriveDatasetWorkChain,
        dataset=dataset,
        derivation_profile=_derivation_profile(),
    )
    assert node.is_finished_ok
    first_dataset = results["dataset"]
    second_results, second_node = run_get_node(
        DeriveDatasetWorkChain,
        dataset=dataset,
        derivation_profile=_derivation_profile(),
    )
    assert second_node.is_finished_ok
    assert second_results["dataset"].pk == first_dataset.pk
    assert second_results["dataset_summary"].get_dict()["reused_existing_dataset"] is True


def test_train_eval_integration_native_dataset(fake_python_executable, local_code_factory, pino_dataset_factory) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-train-eval")
    dataset = pino_dataset_factory("native-train-eval")

    results, node = run_get_node(
        TrainEvalWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=_training_profile(),
        allow_training_data=orm.Bool(True),  # fixture evals on its own training set
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.is_finished_ok
    assert isinstance(results["model"], PinoModelData)
    assert isinstance(results["evaluation"], PinoModelEvaluationData)
    assert results["metrics"].get_dict()["T"]["rmse"] == 0.1
    assert results["train_summary"].get_dict()["contains_hydra"] is True
    assert results["training_artifacts"].list_object_names(".hydra") == ["config.yaml"]


def test_eval_on_training_data_refuses_without_the_flag(
    fake_python_executable, local_code_factory, pino_dataset_factory
) -> None:
    """The 2026-08-09 guard: same fixture as above, minus allow_training_data.

    Four historical evals (one 78% contaminated) got through because nothing
    refused; the refusal, not the bypass, is the contract worth pinning.
    """
    python = local_code_factory(str(fake_python_executable), label="fake-python-guard")
    dataset = pino_dataset_factory("native-guard-refusal")

    _, node = run_get_node(
        TrainEvalWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=_training_profile(),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert not node.is_finished_ok
    evaluate = [c for c in node.called_descendants if c.process_label == "EvaluateModelWorkChain"]
    assert evaluate and evaluate[-1].exit_status == 305, "expected ERROR_TRAINING_DATA_OVERLAP"


def test_train_eval_integration_with_derivation(
    fake_python_executable,
    local_code_factory,
    pino_dataset_factory,
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-train-eval-derived")
    dataset = pino_dataset_factory("native-train-eval-derived")
    profile = _training_profile().get_dict()
    profile["include_time"] = True
    profile["time_as_batch"] = True
    profile["time_as_dim"] = False
    profile["moving_ref_frame"] = False
    profile["moving_frame_steps"] = 1
    profile["time_avg_window"] = 1

    results, node = run_get_node(
        TrainEvalWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=profile),
        derivation_profile=_derivation_profile(),
        # This fixture evals on its own training set (train==val), which the
        # 2026-08-09 guard refuses without an explicit bypass -- the flag is the
        # new contract, not a test convenience.
        allow_training_data=orm.Bool(True),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.is_finished_ok
    train_job = node.called_descendants[-2]
    assert train_job.inputs.training_profile.get_dict()["time_as_batch"] is True
    assert train_job.inputs.training_profile.get_dict()["time_as_dim"] is False


def test_train_model_integration_dataset_collection(
    fake_python_executable,
    local_code_factory,
    pino_dataset_factory,
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-train-collection")
    dataset = pino_dataset_factory(
        "native-train-collection",
        arrays={
            "Temperature": np.full((2, 2, 1, 4, 4, 4), 3.0, dtype=np.float32),
            "alpha": np.zeros((2, 2, 1, 4, 4, 4), dtype=np.float32),
        },
    )
    results, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=_training_profile(),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.is_finished_ok
    assert isinstance(results["model"], PinoModelData)
    metadata = results["model"].get_model_metadata()
    assert metadata["training_source_kind"] == "dataset_collection"
    assert metadata["training_source_pk"] == dataset.pk
    assert metadata["training_source_label"] == (
        dataset.label or dataset.get_dataset_metadata().get("dataset_identifier", "")
    )


def test_train_model_fails_cleanly_for_incompatible_native_variant(
    fake_python_executable,
    local_code_factory,
    pino_dataset_factory,
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-train-incompatible")
    dataset = pino_dataset_factory(
        "legacy-derived-train",
        role="derived",
        metadata={
            "base_dataset_uuid": "",
            "base_dataset_pk": None,
            "parent_dataset_uuid": "",
            "parent_dataset_pk": None,
            "time_axis_mode": {"include_time": True, "time_as_batch": True, "time_as_dim": False},
            "derivation_profile": {
                "include_time": True,
                "time_as_batch": True,
                "time_as_dim": False,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
            },
        },
    )
    results, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=_training_profile(),
        derivation_profile=orm.Dict(
            dict={
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": True,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
            }
        ),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert results == {}
    assert node.exit_status == TrainModelWorkChain.exit_codes.ERROR_INCOMPATIBLE_DATASET_VARIANT.status


def test_train_model_rejects_layout_mismatch_between_profile_and_derivation(
    fake_python_executable,
    local_code_factory,
    pino_dataset_factory,
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-train-layout-mismatch")
    dataset = pino_dataset_factory("layout-mismatch")
    profile = _training_profile().get_dict()
    profile["time_as_batch"] = True
    profile["time_as_dim"] = False

    results, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=profile),
        derivation_profile=orm.Dict(
            dict={
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": True,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
            }
        ),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert results == {}
    assert node.exit_status == TrainModelWorkChain.exit_codes.ERROR_INCOMPATIBLE_DATASET_VARIANT.status


def test_eval_only_integration_with_registered_model(
    fake_python_executable,
    local_code_factory,
    pino_dataset_factory,
    pino_model_factory,
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-eval-only")
    dataset = pino_dataset_factory("native-eval")
    model = pino_model_factory("native-eval-model", training_source=dataset)

    results, node = run_get_node(
        EvalOnlyWorkChain,
        dataset=dataset,
        model=model,
        evaluation_profile=orm.Dict(dict={"run_name": "eval-only"}),
        allow_training_data=orm.Bool(True),  # fixture evals on its own training set
        python_code=python,
    )
    assert node.is_finished_ok
    assert isinstance(results["evaluation"], PinoModelEvaluationData)
    assert results["eval_summary"].get_dict()["metrics_file_present"] is True
    assert results["metrics"].get_dict()["T"]["rmse"] == 0.1


def test_evaluate_model_integration_records_training_provenance(
    fake_python_executable,
    local_code_factory,
    pino_dataset_factory,
    pino_model_factory,
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-evaluate-model")
    train_dataset = pino_dataset_factory("model-train-source")
    eval_dataset = pino_dataset_factory("model-eval-source")
    model = pino_model_factory("registered-model", training_source=train_dataset)

    results, node = run_get_node(
        EvaluateModelWorkChain,
        dataset=eval_dataset,
        model=model,
        evaluation_profile=orm.Dict(dict={"run_name": "registered-model-eval"}),
        python_code=python,
    )
    assert node.is_finished_ok
    evaluation = results["evaluation"]
    assert isinstance(evaluation, PinoModelEvaluationData)
    metadata = evaluation.get_evaluation_metadata()
    assert metadata["model_pk"] == model.pk
    assert metadata["training_source_pk"] == train_dataset.pk
    assert metadata["dataset_source_pk"] == eval_dataset.pk
    assert metadata["dataset_source_kind"] == "dataset_collection"
    assert metadata["metrics"]["T"]["rmse"] == 0.1


def test_evaluate_model_fails_cleanly_for_incompatible_native_variant(
    fake_python_executable,
    local_code_factory,
    pino_dataset_factory,
    pino_model_factory,
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-eval-incompatible")
    train_dataset = pino_dataset_factory("eval-train-source")
    eval_dataset = pino_dataset_factory(
        "legacy-derived-eval",
        role="derived",
        metadata={
            "base_dataset_uuid": "",
            "base_dataset_pk": None,
            "parent_dataset_uuid": "",
            "parent_dataset_pk": None,
            "time_axis_mode": {"include_time": True, "time_as_batch": True, "time_as_dim": False},
            "derivation_profile": {
                "include_time": True,
                "time_as_batch": True,
                "time_as_dim": False,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
            },
        },
    )
    model = pino_model_factory("eval-incompatible-model", training_source=train_dataset)

    results, node = run_get_node(
        EvaluateModelWorkChain,
        dataset=eval_dataset,
        model=model,
        evaluation_profile=orm.Dict(dict={"run_name": "incompatible-eval"}),
        derivation_profile=orm.Dict(
            dict={
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": True,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
            }
        ),
        python_code=python,
    )
    assert results == {}
    assert node.exit_status == EvaluateModelWorkChain.exit_codes.ERROR_INCOMPATIBLE_DATASET_VARIANT.status


def test_kfold_integration_with_derivation(fake_python_executable, local_code_factory, pino_dataset_factory) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-kfold-derived")
    dataset = pino_dataset_factory("native-kfold-derived")
    profile = _training_profile(mode="kfold").get_dict()
    profile["include_time"] = True
    profile["time_as_batch"] = True
    profile["time_as_dim"] = False
    profile["moving_ref_frame"] = False
    profile["moving_frame_steps"] = 1
    profile["time_avg_window"] = 1

    results, node = run_get_node(
        KfoldTrainEvalWorkChain,
        dataset=dataset,
        training_profile=orm.Dict(dict=profile),
        derivation_profile=_derivation_profile(),
        python_code=python,
    )
    assert node.is_finished_ok
    assert isinstance(results["kfold_run"], PinoKfoldRunData)
    fold_train_jobs = [child for child in node.called_descendants if child.process_label == "PinoTrainCalculation"]
    fold_eval_jobs = [child for child in node.called_descendants if child.process_label == "PinoEvalCalculation"]
    assert len(fold_train_jobs) == 8
    assert len(fold_eval_jobs) == 8
    assert all(job.inputs.training_profile.get_dict()["time_as_batch"] is True for job in fold_train_jobs)
    assert all(job.inputs.training_profile.get_dict()["split_mode"] == "kfold" for job in fold_train_jobs)
    assert results["metrics"].get_dict()["aggregated"]["T"]["rmse"] == pytest.approx(0.1)


def test_kfold_integration(fake_python_executable, local_code_factory, pino_dataset_factory) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-kfold")
    dataset = pino_dataset_factory("native-kfold")

    results, node = run_get_node(
        KfoldTrainEvalWorkChain,
        dataset=dataset,
        training_profile=_training_profile(mode="kfold"),
        python_code=python,
    )
    assert node.is_finished_ok
    assert isinstance(results["kfold_run"], PinoKfoldRunData)
    assert results["kfold_summary"].get_dict()["aggregated_errors_present"] is True
    assert results["metrics"].get_dict()["aggregated"]["T"]["rmse"] == pytest.approx(0.1)
    metadata = results["kfold_run"].get_kfold_metadata()
    assert metadata["fold_count"] == 8
    assert metadata["dataset_source_pk"] == dataset.pk
    assert [fold["fold_index"] for fold in metadata["folds"]] == list(range(8))
    assert all("train_process_pk" in fold and "eval_process_pk" in fold for fold in metadata["folds"])


def test_vtk_to_dataset_failure_is_reported_cleanly(tmp_path, local_code_factory, folder_data_factory) -> None:
    failing_executable = tmp_path / "fake_vtk_failure.sh"
    failing_executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cat > dataset_summary.json <<'JSON'
{
  "status": "error",
  "source_type": "openfoam",
  "source_uri": "file:///tmp/failure",
  "source_folder_name": "failure",
  "profile": {"include_time": true},
  "error_type": "ValueError",
  "error_message": "[VTK] moving_frame_steps must be in [1, 2]; got 30."
}
JSON
exit 1
""",
        encoding="utf-8",
    )
    failing_executable.chmod(0o755)
    code = local_code_factory(str(failing_executable), label="fake-vtk-failure")
    vtk_dir = tmp_path / "vtk"
    vtk_dir.mkdir()
    (vtk_dir / "sample.vtk").write_text("vtk", encoding="utf-8")
    vtk_folder = folder_data_factory(vtk_dir)

    results, node = run_get_node(
        VtkToDatasetCalculation,
        code=code,
        vtk_folder=vtk_folder,
        dataset_profile=_dataset_profile(),
        source_type=orm.Str("openfoam"),
        source_folder_name=orm.Str("failure"),
        source_uri=orm.Str("file:///tmp/failure"),
    )

    assert results["dataset_summary"].get_dict()["status"] == "error"
    assert results["dataset_summary"].get_dict()["error_type"] == "ValueError"
    assert node.exit_status == VtkToDatasetCalculation.exit_codes.ERROR_CONVERSION_FAILED.status
    assert node.is_failed


def test_vtk_to_dataset_reuses_existing_collection(fake_python_executable, local_code_factory, tmp_path) -> None:
    code = local_code_factory(str(fake_python_executable), label="fake-python-reuse")
    vtk_dir = tmp_path / "vtk"
    vtk_dir.mkdir()
    (vtk_dir / "sample.vtk").write_text("vtk", encoding="utf-8")

    inputs = dict(
        code=code,
        vtk_source_path=orm.Str(str(vtk_dir)),
        dataset_profile=_dataset_profile(),
        source_type=orm.Str("flow3d"),
        source_folder_name=orm.Str("raw_cases"),
        source_uri=orm.Str("file:///tmp/raw_cases"),
    )

    simulation_count_before = orm.QueryBuilder().append(PinoSimulationData, project=["id"]).count()

    first_results, first_node = run_get_node(VtkToDatasetCalculation, **inputs)
    simulation_count_after_first = orm.QueryBuilder().append(PinoSimulationData, project=["id"]).count()
    second_results, second_node = run_get_node(VtkToDatasetCalculation, **inputs)
    simulation_count_after_second = orm.QueryBuilder().append(PinoSimulationData, project=["id"]).count()

    assert first_node.is_finished_ok
    assert second_node.is_finished_ok
    assert first_results["dataset"].pk != second_results["dataset"].pk
    assert simulation_count_after_first >= simulation_count_before
    assert simulation_count_after_second == simulation_count_after_first
    assert first_results["dataset"].base.extras.get("superseded", False) is True
    assert second_results["dataset"].base.extras.get("superseded", False) is False


def _build_raw_flow3d_tree(root: Path, cases: dict[str, tuple[float, float]]) -> Path:
    """Write synthetic `flow3D_scripts` run dirs: case_manifest.json + npz/<case>.npz."""
    num_frames, (nx, ny, nz) = 6, (6, 5, 4)
    rng = np.random.default_rng(11)
    for case_name, (power_w, scan_speed) in cases.items():
        case_dir = root / case_name
        (case_dir / "npz").mkdir(parents=True)
        (case_dir / "case_manifest.json").write_text(
            json.dumps(
                {
                    "case_name": case_name,
                    "schema_version": 3,
                    "units": "SI",
                    "input": {
                        "laser": {"power_w": power_w, "scan_speed_m_per_s": scan_speed, "beam_radius_m": 6.5e-5},
                        "simulation": {"finish_time_s": 1e-3},
                    },
                    "material": {"name": "ti_imi318_mills", "flow3d_label": "Ti IMI318 (Mills)"},
                }
            ),
            encoding="utf-8",
        )
        np.savez(
            case_dir / "npz" / f"{case_name}.npz",
            x=np.linspace(0.0, 2e-3, nx).astype(np.float32),
            y=np.linspace(-5e-4, 5e-4, ny).astype(np.float32),
            z=np.linspace(-5e-4, 2e-4, nz).astype(np.float32),
            time_s=np.linspace(0.0, 1e-3, num_frames).astype(np.float64),
            power_w=np.float64("nan"),
            scan_speed_m_per_s=np.float64("nan"),
            case_name=np.str_(case_name),
            **{
                "Temperature": (300.0 + rng.random((num_frames, nx, ny, nz))).astype(np.float32),
                "Melt Region": rng.random((num_frames, nx, ny, nz)).astype(np.float32),
                "Velocity": rng.standard_normal((num_frames, nx, ny, nz, 3)).astype(np.float32),
            },
        )
    return root


def test_raw_flow3d_import_integration(local_code_factory, tmp_path: Path) -> None:
    """True end-to-end run of the numpy-only FLOW-3D NPZ adapter (no pvpython)."""
    cases = {"case_a": (150.0, 0.8), "case_b": (250.0, 1.2)}
    source_root = _build_raw_flow3d_tree(tmp_path / "flow3d-runs", cases)
    python = local_code_factory(sys.executable, label="real-python-flow3d-npz")

    results, node = run_get_node(
        RawFlow3DImportWorkChain,
        source_path=orm.Str(str(source_root)),
        dataset_profile=orm.Dict(dict={"time_target_frames": 4}),
        python_code=python,
    )

    assert node.is_finished_ok
    assert isinstance(results["dataset"], PinoDatasetCollectionData)
    assert results["dataset"].base.attributes.get("source_type") == "flow3d"
    assert results["dataset"].get_simulation_count() == 2
    summary = results["dataset_summary"].get_dict()
    assert sorted(summary["imported_cases"]) == ["case_a", "case_b"]
    assert summary["grouping_mode"] == "raw_flow3d_npz"

    seen_groups = set()
    for simulation in results["dataset"].iter_member_simulations():
        metadata = simulation.get_simulation_metadata()
        group = metadata["source_group"]
        seen_groups.add(group)
        power_w, scan_speed = cases[group]
        # P/V_scan channels are constant, non-zero grids at the manifest values.
        for channel, expected in (("P", power_w), ("V_scan", scan_speed)):
            values = np.unique(simulation.get_array_by_original_name(channel))
            assert values.tolist() == [pytest.approx(expected)]
        assert metadata["requested_fields"] == ["Temperature", "Melt Region", "Velocity"]
        assert metadata["source_origin_kind"] == "flow3d_npz_import"
        assert len(simulation.get_time_values_s()) == 4
        assert simulation.get_array_by_original_name("Velocity").shape == (4, 3, 6, 5, 4)
        assert metadata["flow3d_case_manifest"]["input"]["laser"]["power_w"] == power_w
        assert metadata["flow3d_material"]["name"] == "ti_imi318_mills"
        # Laser extras are queryable on the node.
        assert simulation.base.extras.get("laser_power") == pytest.approx(power_w)
        assert simulation.base.extras.get("V_scan") == pytest.approx(scan_speed)
        assert simulation.base.extras.get("beam_radius") == pytest.approx(6.5e-5)
    assert seen_groups == {"case_a", "case_b"}


def test_raw_flow3d_import_rejects_unknown_profile_key(local_code_factory) -> None:
    python = local_code_factory("/bin/bash", label="py-flow3d-invalid-profile")
    _, node = run_get_node(
        RawFlow3DImportWorkChain,
        source_path=orm.Str("/tmp/flow3d-runs"),
        dataset_profile=orm.Dict(dict={"vtk_fields": ["Temperature"]}),
        python_code=python,
    )
    assert node.exit_status == RawFlow3DImportWorkChain.exit_codes.ERROR_INVALID_DATASET_PROFILE.status


def _flow3d_run_config(power_w: float = 250.0, scan_speed: float = 1.2) -> dict:
    """A minimal flow3D_scripts-style config; passed to the runner verbatim."""
    return {
        "schema_version": 3,
        "laser": {"power_w": power_w, "scan_speed_m_per_s": scan_speed, "beam_radius_m": 6.5e-5},
        "simulation": {"finish_time_s": 1e-3},
        "geometry": {"length_m": 2e-3, "width_m": 1e-3, "thickness_m": 5e-4, "gas_clearance_m": 2e-4},
        "mesh": {"cell_size_m": 1e-5},
    }


def test_flow3d_run_calculation_integration(fake_flow3d_scripts_root, local_code_factory, tmp_path: Path) -> None:
    """End-to-end Flow3DRunCalculation against the stub runner (no license, no solver)."""
    python = local_code_factory(sys.executable, label="real-python-flow3d-run")
    out_root = tmp_path / "flow3d-runs"

    results, node = run_get_node(
        Flow3DRunCalculation,
        code=python,
        run_config=orm.Dict(dict=_flow3d_run_config()),
        execution_profile=orm.Dict(
            dict={
                "flow3d_scripts_root": str(fake_flow3d_scripts_root),
                "out_root": str(out_root),
                "case_name": "case_run_calc",
                "export": "npz",
            }
        ),
    )

    assert node.is_finished_ok
    summary = results["run_summary"].get_dict()
    assert summary["status"] == "ok"
    assert summary["case_dir"] == str(out_root / "case_run_calc")
    assert summary["npz_file"] == str(out_root / "case_run_calc" / "npz" / "case_run_calc.npz")
    assert summary["manifest"]["input"]["laser"]["power_w"] == 250.0
    artifact_names = results["run_artifacts"].base.repository.list_object_names()
    assert "case_manifest.json" in artifact_names
    assert "summary.md" in artifact_names
    assert "runhyd_tail.log" in artifact_names
    assert not any(name.startswith("flsgrf") or name.endswith(".npz") for name in artifact_names)
    # Big artifacts stay on disk under out_root.
    assert (out_root / "case_run_calc" / "flsgrf1.case_run_calc").is_file()
    assert (out_root / "case_run_calc" / "npz" / "case_run_calc.npz").is_file()


def test_flow3d_generate_integration(fake_flow3d_scripts_root, local_code_factory, tmp_path: Path) -> None:
    """Stub runner + real NPZ ingest adapter: run -> dataset with manifest P/V channels."""
    python = local_code_factory(sys.executable, label="real-python-flow3d-generate")
    out_root = tmp_path / "flow3d-runs"

    results, node = run_get_node(
        Flow3DGenerateWorkChain,
        run_config=orm.Dict(dict=_flow3d_run_config(power_w=180.0, scan_speed=0.9)),
        execution_profile=orm.Dict(
            dict={
                "flow3d_scripts_root": str(fake_flow3d_scripts_root),
                "out_root": str(out_root),
                "case_name": "case_generate",
            }
        ),
        dataset_profile=orm.Dict(dict={}),
        python_code=python,
    )

    assert node.is_finished_ok
    assert results["run_summary"].get_dict()["status"] == "ok"
    dataset = results["dataset"]
    assert isinstance(dataset, PinoDatasetCollectionData)
    assert dataset.get_simulation_count() == 1
    assert results["dataset_summary"].get_dict()["imported_cases"] == ["case_generate"]
    (simulation,) = list(dataset.iter_member_simulations())
    for channel, expected in (("P", 180.0), ("V_scan", 0.9)):
        values = np.unique(simulation.get_array_by_original_name(channel))
        assert values.tolist() == [pytest.approx(expected)]
    metadata = simulation.get_simulation_metadata()
    assert metadata["flow3d_case_manifest"]["input"]["laser"]["power_w"] == 180.0
    assert metadata["flow3d_material"]["name"] == "ti_imi318_mills"


def test_flow3d_generate_prepare_only_skips_ingest(
    fake_flow3d_scripts_root, local_code_factory, tmp_path: Path
) -> None:
    python = local_code_factory(sys.executable, label="real-python-flow3d-prepare")

    results, node = run_get_node(
        Flow3DGenerateWorkChain,
        run_config=orm.Dict(dict=_flow3d_run_config()),
        execution_profile=orm.Dict(
            dict={
                "flow3d_scripts_root": str(fake_flow3d_scripts_root),
                "out_root": str(tmp_path / "flow3d-runs"),
                "case_name": "case_prepare",
                "prepare_only": True,
            }
        ),
        dataset_profile=orm.Dict(dict={}),
        python_code=python,
    )

    assert node.is_finished_ok
    assert results["run_summary"].get_dict()["prepare_only"] is True
    assert "dataset" not in results
    from aiida_pino.calculations.raw_flow3d_to_dataset import RawFlow3DToDatasetCalculation

    assert not [
        child for child in node.called_descendants if child.process_class is RawFlow3DToDatasetCalculation
    ]


def test_flow3d_generate_rejects_license_in_execution_profile(local_code_factory) -> None:
    python = local_code_factory("/bin/bash", label="py-flow3d-generate-license")
    _, node = run_get_node(
        Flow3DGenerateWorkChain,
        run_config=orm.Dict(dict=_flow3d_run_config()),
        execution_profile=orm.Dict(dict={"case_name": "x", "F3D_LICENSE": "PORT@HOST"}),
        dataset_profile=orm.Dict(dict={}),
        python_code=python,
    )
    assert node.exit_status == Flow3DGenerateWorkChain.exit_codes.ERROR_INVALID_EXECUTION_PROFILE.status


def test_flow3d_batch_integration(fake_flow3d_scripts_root, local_code_factory, tmp_path: Path) -> None:
    python = local_code_factory(sys.executable, label="real-python-flow3d-batch")
    out_root = tmp_path / "flow3d-batch-runs"

    results, node = run_get_node(
        Flow3DBatchWorkChain,
        run_configs=orm.List(list=[_flow3d_run_config(power_w=150.0), _flow3d_run_config(power_w=250.0)]),
        execution_profile=orm.Dict(
            dict={
                "flow3d_scripts_root": str(fake_flow3d_scripts_root),
                "out_root": str(out_root),
                "case_name": "batch_case",
            }
        ),
        dataset_profile=orm.Dict(dict={}),
        python_code=python,
    )

    assert node.is_finished_ok
    batch_summary = results["batch_summary"].get_dict()
    assert batch_summary["total_runs"] == 2
    assert batch_summary["max_concurrent_runs"] == 1
    children = batch_summary["children"]
    assert [child["case_name"] for child in children] == ["batch_case_0000", "batch_case_0001"]
    assert all(child["exit_status"] == 0 for child in children)
    assert len({child["dataset_pk"] for child in children}) == 2
    # Each case landed in its own run dir under the shared out_root.
    assert (out_root / "batch_case_0000" / "case_manifest.json").is_file()
    assert (out_root / "batch_case_0001" / "case_manifest.json").is_file()


def test_flow3d_batch_keeps_case_names_the_configs_carry(
    fake_flow3d_scripts_root, local_code_factory, tmp_path: Path
) -> None:
    """A named sweep (mck_000, mck_001, ...) must survive the batch unchanged.

    Positional `<base>_<index>` names break any sweep whose design table, material
    folders or downstream analysis key off the case name, which is every real
    flow3D_scripts batch. Configs without a name still get the positional fallback.
    """
    python = local_code_factory(sys.executable, label="real-python-flow3d-batch-named")
    out_root = tmp_path / "flow3d-named-runs"
    named = [
        {**_flow3d_run_config(power_w=150.0), "case_name": "mck_000"},
        {**_flow3d_run_config(power_w=250.0), "case_name": "mck_001"},
    ]

    results, node = run_get_node(
        Flow3DBatchWorkChain,
        run_configs=orm.List(list=named),
        execution_profile=orm.Dict(
            dict={
                "flow3d_scripts_root": str(fake_flow3d_scripts_root),
                "out_root": str(out_root),
                "case_name": "ignored_base",
            }
        ),
        dataset_profile=orm.Dict(dict={}),
        python_code=python,
    )

    assert node.is_finished_ok
    children = results["batch_summary"].get_dict()["children"]
    assert [child["case_name"] for child in children] == ["mck_000", "mck_001"]
    assert (out_root / "mck_000" / "case_manifest.json").is_file()
    assert (out_root / "mck_001" / "case_manifest.json").is_file()


def test_flow3d_batch_records_progress_before_it_finishes(
    fake_flow3d_scripts_root, local_code_factory, tmp_path: Path
) -> None:
    """`batch_summary` only exists at finalize; a killed sweep must still show progress."""
    python = local_code_factory(sys.executable, label="real-python-flow3d-batch-progress")

    _, node = run_get_node(
        Flow3DBatchWorkChain,
        run_configs=orm.List(list=[{**_flow3d_run_config(), "case_name": "prog_000"}]),
        execution_profile=orm.Dict(
            dict={
                "flow3d_scripts_root": str(fake_flow3d_scripts_root),
                "out_root": str(tmp_path / "progress"),
            }
        ),
        dataset_profile=orm.Dict(dict={}),
        python_code=python,
    )

    recorded = node.base.extras.get("child_summaries")
    assert [entry["case_name"] for entry in recorded] == ["prog_000"]


def test_flow3d_batch_rejects_duplicate_case_names(
    fake_flow3d_scripts_root, local_code_factory, tmp_path: Path
) -> None:
    """Two configs naming the same case would overwrite one run dir with the other."""
    python = local_code_factory(sys.executable, label="real-python-flow3d-batch-dup")
    clashing = [
        {**_flow3d_run_config(power_w=150.0), "case_name": "same"},
        {**_flow3d_run_config(power_w=250.0), "case_name": "same"},
    ]

    _, node = run_get_node(
        Flow3DBatchWorkChain,
        run_configs=orm.List(list=clashing),
        execution_profile=orm.Dict(
            dict={"flow3d_scripts_root": str(fake_flow3d_scripts_root), "out_root": str(tmp_path / "dup")}
        ),
        dataset_profile=orm.Dict(dict={}),
        python_code=python,
    )

    assert not node.is_finished_ok
    assert node.exit_status == Flow3DBatchWorkChain.exit_codes.ERROR_INVALID_RUN_CONFIGS.status


