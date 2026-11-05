"""Pytest fixtures for aiida-pino."""

from __future__ import annotations

import contextlib
import os
import sys
import textwrap
from pathlib import Path
from uuid import uuid4

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def local_code_factory(aiida_localhost):
    """Create stored `InstalledCode` nodes pointing at local executables."""
    from aiida.orm import InstalledCode

    def _factory(executable: str, label: str | None = None):
        code = InstalledCode(
            computer=aiida_localhost,
            filepath_executable=executable,
            default_calc_job_plugin="core.arithmetic.add",
        )
        code.label = label or f"code-{Path(executable).name}-{uuid4().hex[:8]}"
        return code.store()

    return _factory


@pytest.fixture
def folder_data_factory():
    """Create `FolderData` nodes from temporary directory trees."""
    from aiida.orm import FolderData

    def _factory(root: Path):
        return FolderData(tree=str(root)).store()

    return _factory


@pytest.fixture
def legacy_pino_dataset_factory():
    """Create stored legacy native `PinoDatasetData` nodes."""
    import numpy as np

    from aiida_pino.data import PinoDatasetData, build_family_identifier

    def _factory(
        folder_name: str = "dataset",
        *,
        source_type: str = "flow3d",
        role: str = "base",
        arrays: dict | None = None,
        metadata: dict | None = None,
    ):
        payload = arrays or {
            "Temperature": np.arange(128, dtype=np.float32).reshape(1, 2, 1, 4, 4, 4),
            "alpha": np.ones((1, 2, 1, 4, 4, 4), dtype=np.float32),
        }
        family_identifier = build_family_identifier(
            folder_name,
            source_type,
            grid_params={
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "z_range": [0.0, 1.0],
                "resolution": [4, 4, 4],
            },
            requested_fields=["Temperature", "alpha"],
            source_uri=f"file:///tmp/{folder_name}",
        )
        base_metadata = {
            "source_type": source_type,
            "source_folder_name": folder_name,
            "cache_basename": "var_1_time_dim_base.npz" if role == "base" else "var_1_time_batch.npz",
            "dataset_role": role,
            "canonical_layout": "time_dim_noavg_noref",
            "family_identifier": family_identifier,
            "requested_fields": ["Temperature", "alpha"],
            "input_dir": f"/tmp/{folder_name}",
            "source_uri": f"file:///tmp/{folder_name}",
            "grid_params": {
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "z_range": [0.0, 1.0],
                "resolution": [4, 4, 4],
            },
            "load_all_fields": False,
            "super_res_factor": 1,
            "debug_params": False,
            "force_recompute": False,
            "flip_axes": True,
            "time_axis": True,
            "avg_window_size": 1,
            "moving_frame_steps": 1,
            "moving_ref_frame": False,
            "time_axis_mode": {
                "include_time": True,
                "time_as_batch": False if role == "base" else True,
                "time_as_dim": True if role == "base" else False,
            },
            "base_dataset_uuid": "",
            "base_dataset_pk": None,
            "parent_dataset_uuid": "",
            "parent_dataset_pk": None,
            "derivation_profile": {
                "include_time": True,
                "time_as_batch": False if role == "base" else True,
                "time_as_dim": True if role == "base" else False,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
            },
            "derivation_hash": "" if role == "base" else "derived123",
            "reconstructable": role == "derived",
            "sample_metadata": [
                {
                    "group": folder_name,
                    "name": "results_V_scan_0p01_laser_power_5_395.vtk",
                    "relative_path": "results_V_scan_0p01_laser_power_5_395.vtk",
                    "stem": "results_V_scan_0p01_laser_power_5_395",
                    "params": {
                        "V_scan": 0.01,
                        "laser_power": 5.0,
                        "time_s": 3.95e-4,
                    },
                }
            ],
            "grouped_files": [],
        }
        if metadata:
            base_metadata.update(metadata)
        node = PinoDatasetData.from_arrays(
            payload,
            metadata=base_metadata,
            source={"uri": base_metadata["source_uri"], "description": "Test native dataset"},
        )
        return node.store()

    return _factory


@pytest.fixture
def pino_simulation_factory():
    """Create stored native `PinoSimulationData` nodes."""
    import numpy as np

    from aiida_pino.data import PinoSimulationData

    def _factory(
        folder_name: str = "dataset",
        *,
        source_type: str = "flow3d",
        group_name: str | None = None,
        arrays: dict | None = None,
        metadata: dict | None = None,
    ):
        payload = arrays or {
            "Temperature": np.arange(128, dtype=np.float32).reshape(2, 1, 4, 4, 4),
            "alpha": np.ones((2, 1, 4, 4, 4), dtype=np.float32),
        }
        first_array = next(iter(payload.values()))
        num_timesteps = int(first_array.shape[0]) if len(first_array.shape) >= 5 else 1
        resolved_group = group_name or folder_name
        base_metadata = {
            "source_type": source_type,
            "source_origin_kind": "import",
            "source_folder_name": folder_name,
            "source_group": resolved_group,
            "source_uri": f"file:///tmp/{folder_name}",
            "requested_fields": list(payload),
            "grid_params": {
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "z_range": [0.0, 1.0],
                "resolution": [4, 4, 4],
            },
            "relative_paths": [f"{resolved_group}/results_{index}.vtk" for index in range(num_timesteps)],
            "time_values_s": np.arange(num_timesteps, dtype=np.float32),
            "sample_metadata": [
                {
                    "group": resolved_group,
                    "name": f"results_{index}.vtk",
                    "relative_path": f"{resolved_group}/results_{index}.vtk",
                    "stem": f"results_{index}",
                    "params": {
                        "V_scan": 0.01,
                        "laser_power": 5.0,
                        "time_s": float(index),
                    },
                }
                for index in range(num_timesteps)
            ],
        }
        if metadata:
            base_metadata.update(metadata)
        node = PinoSimulationData.from_arrays(
            payload,
            metadata=base_metadata,
            source={"uri": base_metadata["source_uri"], "description": "Test native simulation"},
        )
        return node.store()

    return _factory


@pytest.fixture(autouse=True)
def _isolate_flow3d_spool(tmp_path_factory):
    """Tests must never queue on the production FLOW-3D seat — nor on each other's.

    The flow3d adapter takes `<spool>/.seat.lock` before the solve timeout
    starts; without this override a stub-runner test blocks for hours behind a
    real solve holding `~/.flow3d/spool/.seat.lock`.

    **Per test, not per session.** `flow3d_run_adapter._SEAT` is a module-global
    ExitStack that deliberately holds the seat for the whole process — right for
    a one-shot adapter, fatal under pytest, where one process runs every test:
    the first test to call `_main()` kept the lock and the next one blocked on
    `flock(LOCK_EX)` against its own process, with no timeout, forever. That is
    why the full suite never finished; it was being read as "~2 h wall" when it
    was in fact deadlocked, and only a per-test timeout made it look finite.

    Giving each test its own spool directory removes the shared resource
    instead of relying on every caller to release it.
    """
    os.environ["F3D_SPOOL_DIR"] = str(tmp_path_factory.mktemp("f3d_spool"))
    yield
    os.environ.pop("F3D_SPOOL_DIR", None)


@pytest.fixture(autouse=True, scope="session")
def _allow_legacy_arrays_storage():
    """Tests build legacy arrays-mode sims deliberately (fixtures, dedup and
    migration suites); production paths are blocked by the store()-level guard.
    The guard's own test deletes this env var to assert the refusal."""
    os.environ["AIIDA_PINO_ALLOW_ARRAYS_STORAGE"] = "1"
    yield
    os.environ.pop("AIIDA_PINO_ALLOW_ARRAYS_STORAGE", None)


@pytest.fixture
def pino_dataset_factory(pino_simulation_factory):
    """Create stored native `PinoDatasetCollectionData` nodes."""
    import numpy as np

    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    def _factory(
        folder_name: str = "dataset",
        *,
        source_type: str = "flow3d",
        role: str = "base",
        arrays: dict | None = None,
        metadata: dict | None = None,
    ):
        payload = arrays or {
            "Temperature": np.arange(128, dtype=np.float32).reshape(2, 1, 4, 4, 4),
            "alpha": np.ones((2, 1, 4, 4, 4), dtype=np.float32),
        }
        first_array = next(iter(payload.values()))
        if len(first_array.shape) >= 6:
            simulation_count = int(first_array.shape[0])
        else:
            simulation_count = 1

        simulations = []
        for index in range(simulation_count):
            simulation_arrays = {
                name: (np.asarray(array[index], dtype=np.float32) if len(np.asarray(array).shape) >= 6 else np.asarray(array, dtype=np.float32))
                for name, array in payload.items()
            }
            simulations.append(
                pino_simulation_factory(
                    folder_name,
                    source_type=source_type,
                    group_name=f"{folder_name}_{index:03d}",
                    arrays=simulation_arrays,
                )
            )

        overrides = dict(metadata or {})
        if "base_dataset_uuid" in overrides and "base_collection_uuid" not in overrides:
            overrides["base_collection_uuid"] = overrides.pop("base_dataset_uuid")
        if "base_dataset_pk" in overrides and "base_collection_pk" not in overrides:
            overrides["base_collection_pk"] = overrides.pop("base_dataset_pk")
        if "parent_dataset_uuid" in overrides and "parent_collection_uuid" not in overrides:
            overrides["parent_collection_uuid"] = overrides.pop("parent_dataset_uuid")
        if "parent_dataset_pk" in overrides and "parent_collection_pk" not in overrides:
            overrides["parent_collection_pk"] = overrides.pop("parent_dataset_pk")

        derivation_profile = overrides.pop(
            "derivation_profile",
            {
                "include_time": True,
                "time_as_batch": False if role == "base" else True,
                "time_as_dim": True if role == "base" else False,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
            },
        )
        sample_metadata = [
            {
                "group": simulation.get_simulation_metadata().get("source_group", f"{folder_name}_{index:03d}"),
                "name": f"simulation_{index:03d}",
                "relative_path": simulation.get_simulation_metadata().get("relative_paths", [""])[0],
                "stem": f"simulation_{index:03d}",
                "params": {"V_scan": 0.01 + 0.01 * index, "laser_power": 5.0 + index},
            }
            for index, simulation in enumerate(simulations)
        ]
        collection_metadata = build_collection_metadata_from_simulations(
            simulations,
            collection_name=folder_name,
            dataset_role=role,
            composition_kind="import" if role == "base" else "derived",
            derivation_profile=derivation_profile,
            extra_metadata={"sample_metadata": sample_metadata, **overrides},
        )
        node, _ = get_or_create_dataset_collection(simulations, metadata=collection_metadata)
        return node

    return _factory


@pytest.fixture
def pino_dataset_union_factory():
    """Create stored metadata-only `PinoDatasetUnionData` nodes."""
    from aiida_pino.utils import get_or_create_dataset_union

    def _factory(*members, union_name: str | None = None):
        node, _ = get_or_create_dataset_union(list(members), union_name=union_name)
        return node

    return _factory


@pytest.fixture
def pino_model_factory(folder_data_factory, pino_dataset_factory, tmp_path: Path):
    """Create stored `PinoModelData` nodes for browser and workflow tests."""
    from aiida_pino.data import PinoModelData

    def _factory(
        model_name: str = "model",
        *,
        training_source=None,
        training_profile: dict | None = None,
        derivation_profile: dict | None = None,
    ):
        source = training_source or pino_dataset_factory(f"{model_name}-dataset")
        artifacts_dir = tmp_path / f"{model_name}-artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")
        hydra_dir = artifacts_dir / ".hydra"
        hydra_dir.mkdir(exist_ok=True)
        (hydra_dir / "config.yaml").write_text("hydra: true", encoding="utf-8")
        training_artifacts = folder_data_factory(artifacts_dir)
        if source.__class__.__name__ == "PinoDatasetUnionData":
            training_source_kind = "dataset_union"
        elif source.__class__.__name__ == "PinoDatasetCollectionData":
            training_source_kind = "dataset_collection"
        else:
            training_source_kind = "dataset"
        metadata = {
            "model_name": model_name,
            "run_name": model_name,
            "config_name": "config_NS_energy_PINO_flow3D",
            "training_profile": training_profile
            or {
                "config_name": "config_NS_energy_PINO_flow3D",
                "run_name": model_name,
                "split_mode": "standard",
                "hydra_overrides": [],
                "cuda_device": "0",
                "wandb_offline": True,
                "time_as_batch": False,
                "time_as_dim": True,
                "include_time": True,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
                "autoregressive": {
                    "enabled": False,
                    "eval_horizon_k": 1,
                    "eval_use_gt_init": True,
                    "require_time": False,
                    "state_prev_suffix": "_prev",
                },
            },
            "derivation_profile": derivation_profile
            or {
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": True,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
            },
            "training_source_kind": training_source_kind,
            "training_source_pk": source.pk,
            "training_source_uuid": source.uuid,
            "training_source_label": source.label
            or getattr(source, "get_union_name", lambda: "")()
            or getattr(source, "get_dataset_metadata", lambda: {})().get("dataset_identifier", ""),
            "training_source_summary": {},
            "primary_checkpoint": "model.ckpt",
            "training_summary": {"contains_hydra": True},
        }
        return PinoModelData.from_training_artifacts(training_artifacts, metadata=metadata).store()

    return _factory


@pytest.fixture
def pino_model_evaluation_factory(pino_dataset_factory, pino_model_factory):
    """Create stored `PinoModelEvaluationData` nodes attached to a model."""
    from aiida_pino.data import PinoModelEvaluationData

    def _factory(model=None, dataset_source=None, evaluation_name: str = "evaluation"):
        resolved_dataset = dataset_source or pino_dataset_factory("eval-dataset")
        resolved_model = model or pino_model_factory("eval-model", training_source=resolved_dataset)
        dataset_source_kind = (
            "dataset_collection" if resolved_dataset.__class__.__name__ == "PinoDatasetCollectionData" else "dataset"
        )
        node = PinoModelEvaluationData.from_payload(
            metadata={
                "evaluation_name": evaluation_name,
                "model_pk": resolved_model.pk,
                "model_uuid": resolved_model.uuid,
                "model_name": resolved_model.get_model_name(),
                "dataset_source_kind": dataset_source_kind,
                "dataset_source_pk": resolved_dataset.pk,
                "dataset_source_uuid": resolved_dataset.uuid,
                "dataset_source_label": resolved_dataset.label or resolved_dataset.get_dataset_metadata().get(
                    "dataset_identifier", ""
                ),
                "evaluation_profile": {"run_name": evaluation_name},
                "derivation_profile": {
                    "include_time": True,
                    "time_as_batch": False,
                    "time_as_dim": True,
                    "moving_ref_frame": False,
                    "moving_frame_steps": 1,
                    "time_avg_window": 1,
                    "super_res_factor": 1,
                },
                "metrics": {"T": {"rmse": 0.1}},
                "eval_summary": {"metrics_file_present": True},
            }
        )
        return node.store()

    return _factory


@pytest.fixture
def pino_kfold_run_factory(pino_dataset_factory):
    """Create stored `PinoKfoldRunData` nodes for grouped k-fold browser tests."""
    from aiida_pino.data import PinoKfoldRunData

    def _factory(dataset_source=None, run_name: str = "kfold-run"):
        source = dataset_source or pino_dataset_factory("kfold-browser-source")
        dataset_source_kind = (
            "dataset_collection" if source.__class__.__name__ == "PinoDatasetCollectionData" else "dataset"
        )
        return PinoKfoldRunData.from_payload(
            metadata={
                "run_name": run_name,
                "config_name": "config_NS_energy_PINO_flow3D",
                "dataset_source_kind": dataset_source_kind,
                "dataset_source_pk": source.pk,
                "dataset_source_uuid": source.uuid,
                "dataset_source_label": source.label or source.get_dataset_metadata().get("dataset_identifier", ""),
                "dataset_source_summary": {"kind": dataset_source_kind, "pk": source.pk},
                "training_profile": {
                    "config_name": "config_NS_energy_PINO_flow3D",
                    "run_name": run_name,
                    "split_mode": "kfold",
                    "k_folds": 2,
                    "fold_index": -1,
                },
                "derivation_profile": {
                    "include_time": True,
                    "time_as_batch": False,
                    "time_as_dim": True,
                    "moving_ref_frame": False,
                    "moving_frame_steps": 1,
                    "time_avg_window": 1,
                    "super_res_factor": 1,
                },
                "kfold_summary": {
                    "fold_indices": [0, 1],
                    "fold_count": 2,
                    "aggregated_errors_present": True,
                },
                "metrics": {
                    "aggregated": {"T": {"rmse": 0.11}},
                    "folds": {
                        "fold_0": {"T": {"rmse": 0.12}},
                        "fold_1": {"T": {"rmse": 0.10}},
                    },
                },
                "folds": [
                    {"fold_index": 0, "train_process_pk": 101, "eval_process_pk": 102, "metrics": {"T": {"rmse": 0.12}}},
                    {"fold_index": 1, "train_process_pk": 103, "eval_process_pk": 104, "metrics": {"T": {"rmse": 0.10}}},
                ],
                "fold_count": 2,
            }
        ).store()

    return _factory


@pytest.fixture
def remote_data_factory(aiida_localhost):
    """Create stored `RemoteData` nodes for local filesystem directories."""
    from aiida.orm import RemoteData

    def _factory(root: Path, computer=None, kind: str = "dataset"):
        node = RemoteData(remote_path=str(root.resolve()), computer=computer or aiida_localhost)
        node.label = f"{kind}:{root.name}"
        node.source = {"uri": f"file://{root.resolve()}", "description": f"Test {kind} import"}
        return node.store()

    return _factory


@pytest.fixture
def other_local_computer():
    """Create a second local computer for mismatch validation tests."""
    from aiida.orm import Computer

    computer = Computer(
        label=f"other-local-{uuid4().hex[:8]}",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.direct",
        workdir="/tmp/aiida-{username}",
    )
    return computer.store()


@pytest.fixture
def fake_pvpython_executable(tmp_path: Path):
    """Create a fake pvpython executable that synthesizes Flow3D outputs."""
    script = tmp_path / "fake_pvpython.sh"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p vtk_output
            echo "vtk" > vtk_output/sample_0.vtk
            cat > flow3d_summary.json <<'JSON'
            {"source_path": "/tmp/raw", "output_dir": "vtk_output", "profile": {"num_timesteps": 2}, "converted_cases": [{"case_name": "case_a"}], "skipped_cases": []}
            JSON
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def fake_python_executable(tmp_path: Path):
    """Create a fake Python executable that synthesizes dataset, train, and eval outputs."""
    script = tmp_path / "fake_python.sh"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            SOURCE_TYPE="flow3d"
            SOURCE_METADATA_FILE=""
            while [[ $# -gt 0 ]]; do
              case "$1" in
                --source-type)
                  SOURCE_TYPE="$2"
                  shift 2
                  ;;
                --source-metadata-file)
                  SOURCE_METADATA_FILE="$2"
                  shift 2
                  ;;
                *)
                  shift
                  ;;
              esac
            done
            export SOURCE_TYPE
            export SOURCE_METADATA_FILE
            python3 - <<'PY'
            import json
            import os
            import numpy as np

            os.makedirs("dataset_output/payloads", exist_ok=True)
            array = np.arange(128, dtype=np.float32).reshape(2, 1, 4, 4, 4)
            np.savez_compressed("dataset_output/payloads/simulation_0000.npz", Temperature=array)
            simulation_metadata = {
                "format_version": 1,
                "source_type": os.environ.get("SOURCE_TYPE", "flow3d"),
                "source_origin_kind": "vtk_import",
                "source_folder_name": "raw_cases",
                "source_group": "raw_cases",
                "source_uri": "file:///tmp/raw_cases",
                "requested_fields": ["Temperature"],
                "grid_params": {
                    "x_range": [0.0, 1.0],
                    "y_range": [0.0, 1.0],
                    "z_range": [0.0, 1.0],
                    "resolution": [4, 4, 4],
                },
                "relative_paths": [
                    "results_V_scan_0p01_laser_power_5_395.vtk",
                    "results_V_scan_0p01_laser_power_5_396.vtk",
                ],
                "time_values_s": [3.95e-4, 3.96e-4],
                "sample_metadata": [
                    {
                        "group": "raw_cases",
                        "name": "results_V_scan_0p01_laser_power_5_395.vtk",
                        "relative_path": "results_V_scan_0p01_laser_power_5_395.vtk",
                        "stem": "results_V_scan_0p01_laser_power_5_395",
                        "params": {"V_scan": 0.01, "laser_power": 5.0, "time_s": 3.95e-4},
                    },
                    {
                        "group": "raw_cases",
                        "name": "results_V_scan_0p01_laser_power_5_396.vtk",
                        "relative_path": "results_V_scan_0p01_laser_power_5_396.vtk",
                        "stem": "results_V_scan_0p01_laser_power_5_396",
                        "params": {"V_scan": 0.01, "laser_power": 5.0, "time_s": 3.96e-4},
                    },
                ],
                "simulation_fingerprint": "testfingerprint-sim0",
                "simulation_identifier": "raw_cases::raw_cases::sim0000",
            }
            metadata = {
                "format_version": 1,
                "source_type": os.environ.get("SOURCE_TYPE", "flow3d"),
                "source_folder_name": "raw_cases",
                "source_uri": "file:///tmp/raw_cases",
                "source_description": "Synthetic test dataset",
                "input_dir": "/tmp/raw_cases",
                "dataset_role": "base",
                "canonical_layout": "time_dim_noavg_noref",
                "family_identifier": "raw_cases__flow3d__testfam",
                "grid_params": {
                    "x_range": [0.0, 1.0],
                    "y_range": [0.0, 1.0],
                    "z_range": [0.0, 1.0],
                    "resolution": [4, 4, 4],
                },
                "requested_fields": ["Temperature"],
                "load_all_fields": False,
                "super_res_factor": 1,
                "debug_params": False,
                "force_recompute": False,
                "flip_axes": False,
                "time_axis": True,
                "avg_window_size": 1,
                "moving_frame_steps": 1,
                "moving_ref_frame": False,
                "time_axis_mode": {
                    "include_time": True,
                    "time_as_batch": False,
                    "time_as_dim": True,
                },
                "source_origin_kind": "vtk_import",
                "base_dataset_uuid": "",
                "base_dataset_pk": None,
                "parent_dataset_uuid": "",
                "parent_dataset_pk": None,
                "derivation_profile": {
                    "include_time": True,
                    "time_as_batch": False,
                    "time_as_dim": True,
                    "moving_ref_frame": False,
                    "moving_frame_steps": 1,
                    "time_avg_window": 1,
                    "super_res_factor": 1,
                },
                "derivation_hash": "",
                "reconstructable": False,
                "grouping_mode": "flat_files",
                "grouped_files": [
                    {
                        "group": "raw_cases",
                        "num_files": 1,
                        "files": [
                            {
                                "group": "raw_cases",
                                "name": "results_V_scan_0p01_laser_power_5_395.vtk",
                                "relative_path": "results_V_scan_0p01_laser_power_5_395.vtk",
                                "stem": "results_V_scan_0p01_laser_power_5_395",
                                "params": {"V_scan": 0.01, "laser_power": 5.0, "time_s": 3.95e-4},
                            }
                        ],
                    }
                ],
                "sample_metadata": [
                    {
                        "group": "raw_cases",
                        "name": "results_V_scan_0p01_laser_power_5_395.vtk",
                        "relative_path": "results_V_scan_0p01_laser_power_5_395.vtk",
                        "stem": "results_V_scan_0p01_laser_power_5_395",
                        "params": {"V_scan": 0.01, "laser_power": 5.0, "time_s": 3.95e-4},
                    }
                ],
                "array_names": ["Temperature"],
                "array_shapes": {"Temperature": [2, 1, 4, 4, 4]},
                "cache_basename": "var_1_time_dim_base.npz",
                "cache_mode": "sandbox_cache",
                "cache_root": "/tmp/raw_cases",
                "cache_fingerprint": "testfingerprint",
                "simulation_count": 1,
                "sample_metadata": [
                    {
                        "group": "raw_cases",
                        "name": "simulation_0000",
                        "relative_path": "results_V_scan_0p01_laser_power_5_395.vtk",
                        "stem": "simulation_0000",
                        "params": {"V_scan": 0.01, "laser_power": 5.0},
                    }
                ],
                "label": f"raw_cases | var_1_time_dim_base | {os.environ.get('SOURCE_TYPE', 'flow3d')}",
            }
            source_metadata_file = os.environ.get("SOURCE_METADATA_FILE", "")
            if source_metadata_file:
                with open(source_metadata_file, "r", encoding="utf-8") as handle:
                    metadata.update(json.load(handle))
            with open("dataset_output/metadata.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
            with open("dataset_output/manifest.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "format": "aiida_arraydir_v2",
                        "source_kind": "dataset_collection",
                        "source_type": metadata["source_type"],
                        "source_uri": metadata["source_uri"],
                        "source_folder_name": metadata["source_folder_name"],
                        "members": [
                            {
                                "group": "raw_cases",
                                "payload_file": "payloads/simulation_0000.npz",
                                "simulation_metadata": simulation_metadata,
                            }
                        ],
                        "cache_mode": "sandbox_cache",
                        "cache_root": "/tmp/raw_cases",
                        "cache_fingerprint": "testfingerprint",
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            with open("dataset_summary.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset_identifier_hint": f"raw_cases | var_1_time_dim_base | {os.environ.get('SOURCE_TYPE', 'flow3d')}",
                        "source_type": os.environ.get("SOURCE_TYPE", "flow3d"),
                        "source_uri": "file:///tmp/raw_cases",
                        "source_folder_name": "raw_cases",
                        "cache_files": ["var_1_time_dim_base.npz"],
                        "requested_fields": ["Temperature"],
                        "array_names": ["Temperature"],
                        "array_shapes": {"Temperature": [2, 1, 4, 4, 4]},
                        "grouping_mode": "flat_files",
                        "num_groups": 1,
                        "num_simulations": 1,
                        "num_files": 1,
                        "profile": {"flip_axes": False},
                        "canonical_layout": "time_dim_noavg_noref",
                        "cache_mode": "sandbox_cache",
                        "cache_root": "/tmp/raw_cases",
                        "cache_fingerprint": "testfingerprint",
                        "source_metadata": json.load(open(source_metadata_file, "r", encoding="utf-8")) if source_metadata_file else {},
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            PY

            mkdir -p training_output/.hydra
            echo "hydra: true" > training_output/.hydra/config.yaml
            echo "checkpoint" > training_output/model.ckpt
            cat > train_summary.json <<'JSON'
            {"output_dir": "training_output", "contains_hydra": true, "gpu_vram_peak_mb": 4200}
            JSON

            mkdir -p eval_output/vti
            cat > eval_output/model_errors.txt <<'JSON'
            [T]
            rmse: 0.1
            rel_rmse: 0.2
            
            [alpha]
            iou_mean: 0.9
            JSON
            echo "prediction" > eval_output/predictions.npz
            cat > eval_summary.json <<'JSON'
            {"output_dir": "eval_output", "metrics_file_present": true, "gpu_vram_peak_mb": 4200}
            JSON

            mkdir -p kfold_output/fno_eval
            mkdir -p kfold_output/fold_0
            cat > kfold_output/fno_eval/aggregated_errors.txt <<'JSON'
            [T]
            rmse: 0.11
            JSON
            cat > kfold_output/fold_0/model_errors.txt <<'JSON'
            [T]
            rmse: 0.12
            JSON
            cat > kfold_output/fno_eval/val_errors_by_sample.csv <<'CSV'
            sample,rmse
            0,0.12
            CSV
            cat > kfold_summary.json <<'JSON'
            {"output_dir": "kfold_output", "aggregated_errors_present": true, "fold_directories": ["fold_0"]}
            JSON

            cat > metrics.json <<'JSON'
            {
              "T": {"rmse": 0.1, "rel_rmse": 0.2},
              "alpha": {"iou_mean": 0.9},
              "aggregated": {"T": {"rmse": 0.11}},
              "folds": {"fold_0": {"T": {"rmse": 0.12}}}
            }
            JSON
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def fake_flow3d_scripts_root(tmp_path: Path):
    """Create a fake flow3D_scripts checkout whose runner writes a realistic run dir.

    The stub `python/run_rectangular_case.py` parses the real runner's CLI,
    reads the config JSON, and writes `case_manifest.json` (current schema),
    a tiny synthetic `npz/<case>.npz`, `summary.md`, `prepin.<case>`, and
    `runhyd.log` into `<out_root>/<case_name>/`. `--prepare-only` skips the
    NPZ and solver log, like the real runner.
    """
    root = tmp_path / "fake_flow3d_scripts"
    (root / "python").mkdir(parents=True)
    (root / "python" / "run_rectangular_case.py").write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import json
            from pathlib import Path

            import numpy as np

            parser = argparse.ArgumentParser()
            parser.add_argument("config", type=Path)
            parser.add_argument("--case-name", required=True)
            parser.add_argument("--out-root", type=Path, required=True)
            parser.add_argument("--export", default="npz")
            parser.add_argument("--prepare-only", action="store_true")
            parser.add_argument("--overwrite", action="store_true")
            parser.add_argument("--cores", type=int)
            parser.add_argument("--threads-per-rank", type=int)
            parser.add_argument("--flow3d-env", type=Path)
            parser.add_argument("--post-home", type=Path)
            args = parser.parse_args()

            config = json.loads(args.config.read_text(encoding="utf-8"))
            case_dir = args.out_root / args.case_name
            case_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "case_name": args.case_name,
                "schema_version": 3,
                "source_config": str(args.config),
                "units": "SI",
                "input": config,
                "derived": {"scan_time_s": 1.25e-3},
                "material": {
                    "name": "ti_imi318_mills",
                    "file": "materials/ti_imi318_mills/ti_imi318_mills.json",
                    "flow3d_label": "Ti IMI318 (Mills)",
                    "properties": {"rho_s": 4420.0},
                    "optical": {"absorptivity": 0.35},
                    "table_rows": 42,
                },
            }
            (case_dir / "case_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (case_dir / "summary.md").write_text("# fake run summary\\n", encoding="utf-8")
            (case_dir / f"prepin.{args.case_name}").write_text("fake prepin\\n", encoding="utf-8")

            if not args.prepare_only:
                (case_dir / "runhyd.log").write_text(
                    "\\n".join(f"line {i}" for i in range(300))
                    + "\\n normal completion: finish time exceeded\\n",
                    encoding="utf-8",
                )
                (case_dir / f"flsgrf1.{args.case_name}").write_text("BIG SOLVER OUTPUT", encoding="utf-8")
                if args.export in ("npz", "both"):
                    nx, ny, nz, frames = 6, 5, 4, 4
                    rng = np.random.default_rng(3)
                    (case_dir / "npz").mkdir(exist_ok=True)
                    np.savez(
                        case_dir / "npz" / f"{args.case_name}.npz",
                        x=np.linspace(0.0, 2e-3, nx).astype(np.float32),
                        y=np.linspace(-5e-4, 5e-4, ny).astype(np.float32),
                        z=np.linspace(-5e-4, 2e-4, nz).astype(np.float32),
                        time_s=np.linspace(0.0, 1e-3, frames).astype(np.float64),
                        power_w=np.float64("nan"),
                        scan_speed_m_per_s=np.float64("nan"),
                        case_name=np.str_(args.case_name),
                        **{
                            "Temperature": (300.0 + rng.random((frames, nx, ny, nz))).astype(np.float32),
                            "Melt Region": rng.random((frames, nx, ny, nz)).astype(np.float32),
                            "Velocity": rng.standard_normal((frames, nx, ny, nz, 3)).astype(np.float32),
                        },
                    )
            raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    return root


_PROFILE_FIXTURES = {
    "aiida_profile_clean",
    "aiida_localhost",
    "aiida_computer",
    "local_code_factory",
    "pino_simulation_factory",
    "pino_dataset_factory",
    "pino_model_factory",
    "pino_dataset_union_factory",
    "legacy_pino_dataset_factory",
}


_NO_PINO_SIMULATIONS = (
    "needs the sibling lpno checkout installed in this environment "
    "(README: Companion repositories)"
)


def _lpno_importable() -> bool:
    try:
        import lpno  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    have_pino = _lpno_importable()
    for item in items:
        if _PROFILE_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.slow)
        if not have_pino and item.get_closest_marker("requires_lpno"):
            item.add_marker(pytest.mark.skip(reason=_NO_PINO_SIMULATIONS))


@pytest.fixture(autouse=True)
def _guards_enforce_under_test(monkeypatch):
    """The suite tests the ENFORCING behaviour of the memory and disk guards whichever copy it
    runs in -- the public export defaults to `warn` (utils/guards.py). A test of warn mode sets
    the variable itself, after this fixture."""
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "enforce")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """A test that reaches into lpno on a box without it skips, with the
    install hint, instead of failing 50 times with the bare ModuleNotFoundError.
    Only that one module is translated; any other missing import stays a failure."""
    outcome = yield
    exc = outcome.excinfo
    if exc is None or not issubclass(exc[0], ModuleNotFoundError):
        return
    name = getattr(exc[1], "name", "") or ""
    if name == "lpno" or name.startswith("lpno."):
        outcome.force_exception(pytest.skip.Exception(_NO_PINO_SIMULATIONS))


@pytest.fixture(autouse=True)
def _release_flow3d_seat():
    """Hand the FLOW-3D seat back between tests.

    `flow3d_run_adapter._SEAT` is a module-global ExitStack holding the seat for
    the *process* lifetime. That is right for the adapter — it is a one-shot
    process and must not release the seat before the solve ends — and fatal
    under pytest, where every test shares one process: the first test that calls
    `_main()` takes the lock and never gives it back, so the next one blocks on
    `flock(LOCK_EX)` against its own process, forever.

    This is why the full suite never finished. It was read as "~2 h wall"; it is
    not slow, it deadlocks, and only a per-test timeout made it look finite.
    Measured after this fixture: the 11 `test_flow3d_run_adapter.py` tests go
    from 45 s each (the timeout, i.e. unbounded) to under a second.
    """
    yield
    module = sys.modules.get("aiida_pino.scripts.flow3d_run_adapter")
    if module is None:
        return
    module._SEAT.close()
    module._SEAT = contextlib.ExitStack()
    os.environ.pop("F3D_SEAT_HELD", None)
