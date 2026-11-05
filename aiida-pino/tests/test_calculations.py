"""Tests for calcjob preparation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aiida import orm
from aiida.common.folders import SandboxFolder
from aiida.engine.utils import instantiate_process
from aiida.manage import get_manager

from aiida_pino.calculations.eval import PinoEvalCalculation
from aiida_pino.calculations.flow3d_run import Flow3DRunCalculation
from aiida_pino.calculations.flow3d_to_vtk import Flow3DToVtkCalculation
from aiida_pino.calculations.kfold import PinoKfoldCalculation
from aiida_pino.calculations.raw_flow3d_to_dataset import RawFlow3DToDatasetCalculation
from aiida_pino.calculations.train import PinoTrainCalculation
from aiida_pino.calculations.vtk_to_dataset import VtkToDatasetCalculation
from aiida_pino.data import PinoModelData
from aiida_pino.utils import load_training_profile_defaults


def instantiate_process_for(calc_cls, **inputs):
    runner = get_manager().get_runner()
    return instantiate_process(runner, calc_cls, **inputs)


def _training_profile(mode: str = "standard") -> dict:
    profile = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    profile["run_name"] = "run"
    profile["hydra_overrides"] = []
    profile["split_mode"] = mode
    return profile


def test_flow3d_to_vtk_prepare_for_submission(local_code_factory) -> None:
    code = local_code_factory("/bin/bash")
    process = instantiate_process_for(
        Flow3DToVtkCalculation,
        code=code,
        source_path=orm.Str("/tmp/raw"),
        conversion_profile=orm.Dict(
            dict={"case_regex": ".*", "num_timesteps": 3, "output_basename_mode": "legacy"}
        ),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert calcinfo.retrieve_list == ["aiida.out", "flow3d_summary.json", "vtk_output"]
        assert "flow3d_to_vtk_adapter.py" in folder.get_content_list()
        assert "live_tail.py" in folder.get_content_list()
        assert "--pino-repo-path" not in calcinfo.codes_info[0].cmdline_params


def test_vtk_to_dataset_prepare_for_submission_from_folder(local_code_factory, folder_data_factory, tmp_path) -> None:
    code = local_code_factory("/bin/bash")
    vtk_dir = tmp_path / "vtk"
    vtk_dir.mkdir()
    (vtk_dir / "a.vtk").write_text("vtk", encoding="utf-8")
    vtk_folder = folder_data_factory(vtk_dir)

    process = instantiate_process_for(
        VtkToDatasetCalculation,
        code=code,
        vtk_folder=vtk_folder,
        dataset_profile=orm.Dict(
            dict={
                "geometry": {"x_range": [0, 1], "y_range": [0, 1], "z_range": [0, 1], "resolution": [4, 4, 4]},
                "vtk_fields": ["Temperature"],
                "super_res_factor": 1,
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": False,
                "moving_ref_frame": True,
                "moving_frame_steps": 2,
                "time_avg_window": 2,
                "flip_axes": False,
            }
        ),
        source_folder_name=orm.Str("laser_cases"),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert "dataset_summary.json" in calcinfo.retrieve_list
        assert calcinfo.retrieve_temporary_list == ["dataset_output"]
        assert calcinfo.local_copy_list == [(vtk_folder.uuid, ".", "vtk_input")]
        assert "--source-folder-name" in calcinfo.codes_info[0].cmdline_params


def test_vtk_to_dataset_prepare_for_submission_from_path(local_code_factory) -> None:
    code = local_code_factory("/bin/bash")
    process = instantiate_process_for(
        VtkToDatasetCalculation,
        code=code,
        vtk_source_path=orm.Str("/tmp/vtk"),
        dataset_profile=orm.Dict(
            dict={
                "geometry": {"x_range": [0, 1], "y_range": [0, 1], "z_range": [0, 1], "resolution": [4, 4, 4]},
                "vtk_fields": ["Temperature"],
                "super_res_factor": 1,
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": False,
                "moving_ref_frame": True,
                "moving_frame_steps": 2,
                "time_avg_window": 2,
                "flip_axes": False,
            }
        ),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert calcinfo.local_copy_list == []
        assert "--vtk-source-path" in calcinfo.codes_info[0].cmdline_params
        assert calcinfo.retrieve_temporary_list == ["dataset_output"]


def test_vtk_to_dataset_prepare_for_submission_with_source_metadata(local_code_factory) -> None:
    code = local_code_factory("/bin/bash")
    process = instantiate_process_for(
        VtkToDatasetCalculation,
        code=code,
        vtk_source_path=orm.Str("/tmp/vtk"),
        dataset_profile=orm.Dict(
            dict={
                "geometry": {"x_range": [0, 1], "y_range": [0, 1], "z_range": [0, 1], "resolution": [4, 4, 4]},
                "vtk_fields": ["Temperature"],
                "super_res_factor": 1,
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": False,
                "moving_ref_frame": True,
                "moving_frame_steps": 2,
                "time_avg_window": 2,
                "flip_axes": False,
            }
        ),
        source_metadata=orm.Dict(dict={"laserbeamfoam": {"run_summary": {"results_dir": "results_case"}}}),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert Path(folder.get_abs_path("source_metadata.json")).exists()
        assert "--source-metadata-file" in calcinfo.codes_info[0].cmdline_params


def test_vtk_to_dataset_prepare_for_submission_with_force_recompute(local_code_factory) -> None:
    code = local_code_factory("/bin/bash")
    process = instantiate_process_for(
        VtkToDatasetCalculation,
        code=code,
        vtk_source_path=orm.Str("/tmp/vtk"),
        dataset_profile=orm.Dict(
            dict={
                "geometry": {"x_range": [0, 1], "y_range": [0, 1], "z_range": [0, 1], "resolution": [4, 4, 4]},
                "vtk_fields": ["Temperature"],
                "flip_axes": False,
                "force_recompute": True,
            }
        ),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert "--force-recompute" in calcinfo.codes_info[0].cmdline_params


def test_dataset_calcjobs_reject_unknown_source_type(local_code_factory) -> None:
    code = local_code_factory("/bin/bash")
    profile = orm.Dict(
        dict={
            "geometry": {"x_range": [0, 1], "y_range": [0, 1], "z_range": [0, 1], "resolution": [4, 4, 4]},
            "vtk_fields": ["Temperature"],
            "flip_axes": False,
        }
    )
    with pytest.raises(ValueError, match="Unsupported source_type"):
        instantiate_process_for(
            VtkToDatasetCalculation,
            code=code,
            vtk_source_path=orm.Str("/tmp/vtk"),
            dataset_profile=profile,
            source_type=orm.Str("flowd3"),
        )
    # `openfoam_raw` stays a supported VALUE (nodes imported before the OpenFOAM path was
    # removed carry it) and must pass port validation on the kept calcjob.
    instantiate_process_for(
        VtkToDatasetCalculation,
        code=code,
        vtk_source_path=orm.Str("/tmp/vtk"),
        dataset_profile=profile,
        source_type=orm.Str("openfoam_raw"),
    )


def test_raw_flow3d_to_dataset_prepare_for_submission(local_code_factory) -> None:
    code = local_code_factory("/bin/bash")
    process = instantiate_process_for(
        RawFlow3DToDatasetCalculation,
        code=code,
        raw_source_path=orm.Str("/tmp/flow3d-runs"),
        dataset_profile=orm.Dict(dict={"npz_fields": ["Temperature"], "time_target_frames": 40}),
        materials_root=orm.Str("/tmp/flow3D_scripts"),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert "dataset_summary.json" in calcinfo.retrieve_list
        assert calcinfo.retrieve_temporary_list == ["dataset_output"]
        assert "raw_flow3d_to_dataset_adapter.py" in folder.get_content_list()
        assert "live_tail.py" in folder.get_content_list()
        cmdline = calcinfo.codes_info[0].cmdline_params
        assert "--raw-source-path" in cmdline
        assert cmdline[cmdline.index("--source-type") + 1] == "flow3d"
        assert cmdline[cmdline.index("--materials-root") + 1] == "/tmp/flow3D_scripts"
        profile_json = json.loads(Path(folder.get_abs_path("dataset_profile.json")).read_text(encoding="utf-8"))
        assert profile_json == {"npz_fields": ["Temperature"], "time_target_frames": 40}


def test_flow3d_run_prepare_for_submission(local_code_factory) -> None:
    code = local_code_factory("/bin/bash")
    process = instantiate_process_for(
        Flow3DRunCalculation,
        code=code,
        run_config=orm.Dict(dict={"laser": {"power_w": 250.0, "scan_speed_m_per_s": 1.2}}),
        execution_profile=orm.Dict(dict={"case_name": "case_a", "export": "npz", "cores": 2}),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert calcinfo.retrieve_list == ["aiida.out", "flow3d_run_summary.json", "run_artifacts"]
        assert "flow3d_run_adapter.py" in folder.get_content_list()
        assert "live_tail.py" in folder.get_content_list()
        cmdline = calcinfo.codes_info[0].cmdline_params
        assert cmdline[cmdline.index("--config-file") + 1] == "run_config.json"
        assert cmdline[cmdline.index("--profile-file") + 1] == "execution_profile.json"
        staged_config = json.loads(Path(folder.get_abs_path("run_config.json")).read_text(encoding="utf-8"))
        assert staged_config["laser"]["power_w"] == 250.0
        staged_profile = json.loads(Path(folder.get_abs_path("execution_profile.json")).read_text(encoding="utf-8"))
        assert staged_profile == {"case_name": "case_a", "export": "npz", "cores": 2}


def test_flow3d_run_rejects_license_and_unknown_profile_keys(local_code_factory) -> None:
    code = local_code_factory("/bin/bash")
    with pytest.raises(ValueError, match="license"):
        instantiate_process_for(
            Flow3DRunCalculation,
            code=code,
            run_config=orm.Dict(dict={}),
            execution_profile=orm.Dict(dict={"license": "PORT@HOST"}),
        )
    with pytest.raises(ValueError, match="unknown keys"):
        instantiate_process_for(
            Flow3DRunCalculation,
            code=code,
            run_config=orm.Dict(dict={}),
            execution_profile=orm.Dict(dict={"casename": "typo"}),
        )


def test_train_prepare_for_submission_native_dataset(local_code_factory, pino_dataset_factory) -> None:
    code = local_code_factory("/bin/bash")
    dataset = pino_dataset_factory("native-train")

    process = instantiate_process_for(
        PinoTrainCalculation,
        code=code,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=_training_profile()),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        profile_json = json.loads(Path(folder.get_abs_path("training_profile.json")).read_text(encoding="utf-8"))
        assert profile_json["run_name"] == "run"
        # native datasets stage as a symlink to the persistent runtime cache
        # instead of a full copy through the sandbox
        assert calcinfo.codes_info[0].cmdline_params[calcinfo.codes_info[0].cmdline_params.index("--val-dir") + 1] == "train_dataset"
        assert calcinfo.local_copy_list == []
        assert len(calcinfo.remote_symlink_list) == 1
        _computer, cache_path, target = calcinfo.remote_symlink_list[0]
        assert target == "train_dataset"
        assert (Path(cache_path) / "manifest.json").exists()
        assert (Path(cache_path) / "members/simulation_0000/manifest.json").exists()


def test_train_prepare_for_submission_legacy_folder_data(local_code_factory, folder_data_factory, tmp_path) -> None:
    code = local_code_factory("/bin/bash")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "var_1.npz").write_text("dataset", encoding="utf-8")
    dataset_folder = folder_data_factory(dataset_dir)

    process = instantiate_process_for(
        PinoTrainCalculation,
        code=code,
        train_dataset=dataset_folder,
        val_dataset=dataset_folder,
        training_profile=orm.Dict(dict=_training_profile()),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert len(calcinfo.local_copy_list) == 1
        assert calcinfo.codes_info[0].cmdline_params[calcinfo.codes_info[0].cmdline_params.index("--val-dir") + 1] == "train_dataset"
        assert calcinfo.remote_symlink_list == []


def test_eval_prepare_for_submission_native_dataset_and_remote_training(
    local_code_factory,
    pino_dataset_factory,
    remote_data_factory,
    tmp_path,
) -> None:
    code = local_code_factory("/bin/bash")
    dataset = pino_dataset_factory("native-eval")
    training_dir = tmp_path / "training-remote"
    training_dir.mkdir()
    training = remote_data_factory(training_dir, kind="training_artifacts")

    process = instantiate_process_for(
        PinoEvalCalculation,
        code=code,
        dataset=dataset,
        training_artifacts=training,
        training_profile=orm.Dict(dict=_training_profile()),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert calcinfo.local_copy_list == []
        # dataset cache symlink + remote training artifacts symlink
        assert len(calcinfo.remote_symlink_list) == 2
        by_target = {entry[2]: entry[1] for entry in calcinfo.remote_symlink_list}
        assert (Path(by_target["dataset_input"]) / "manifest.json").exists()


def test_eval_prepare_for_submission_uses_primary_checkpoint_parent_for_model_bundle(
    local_code_factory,
    pino_dataset_factory,
    folder_data_factory,
    tmp_path,
) -> None:
    code = local_code_factory("/bin/bash")
    dataset = pino_dataset_factory("native-eval-model")
    artifacts_dir = tmp_path / "imported-model"
    checkpoint_dir = artifacts_dir / "run" / "outputs" / "run"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "u_network.0.pth").write_text("checkpoint", encoding="utf-8")
    hydra_dir = artifacts_dir / "run" / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text("hydra: true", encoding="utf-8")
    training_artifacts = folder_data_factory(artifacts_dir)
    model = PinoModelData.from_training_artifacts(
        training_artifacts,
        metadata={
            "model_name": "imported",
            "run_name": "imported",
            "config_name": "config_NS_energy_PINO_flow3D",
            "training_profile": _training_profile(),
            "derivation_profile": {
                "include_time": True,
                "time_as_batch": False,
                "time_as_dim": False,
                "moving_ref_frame": True,
                "moving_frame_steps": 30,
                "time_avg_window": 30,
                "super_res_factor": 1,
            },
            "training_source_kind": "dataset",
            "training_source_pk": dataset.pk,
            "training_source_uuid": dataset.uuid,
            "training_source_label": dataset.label,
            "training_source_summary": {},
            "primary_checkpoint": "model.ckpt",
            "training_summary": {"imported": True},
        },
    ).store()

    process = instantiate_process_for(
        PinoEvalCalculation,
        code=code,
        dataset=dataset,
        training_artifacts=model,
        training_profile=orm.Dict(dict=_training_profile()),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        assert "training_input/run/outputs/run" in calcinfo.codes_info[0].cmdline_params
        assert calcinfo.local_copy_list == []
        # CHANGED 2026-08-24; was `assert not ... .exists()`. That line carried no
        # docstring, comment or rationale -- it codified staging behaviour from
        # 2026-04-07 (b7b17ca), four months before the decision that supersedes it.
        # On 2026-08-14 (18b90a0) evals were changed to score a checkpoint under the
        # config it was TRAINED with, and `eval_adapter.archived_config_dir` looks for
        # exactly this path. Deleting it made that fix inert from the day it landed:
        # the adapter fell back to today's packaged config with only a warning, and an
        # evaluation failed loading its model with `Missing key(s) in state_dict:
        # "proc_steps.8.edge_mlp..."` because the arch default had drifted since training.
        assert Path(folder.get_abs_path("training_input/run/.hydra/config.yaml")).is_file(), (
            "the model's archived config must reach the eval, or it is scored under "
            "today's packaged config and degrades as the schema moves"
        )


def test_eval_prepare_rejects_mismatched_remote_training_artifacts(
    local_code_factory,
    pino_dataset_factory,
    remote_data_factory,
    other_local_computer,
    tmp_path,
) -> None:
    code = local_code_factory("/bin/bash")
    dataset = pino_dataset_factory("native-eval-mismatch")
    training_dir = tmp_path / "training-mismatch"
    training_dir.mkdir()
    training = remote_data_factory(training_dir, computer=other_local_computer, kind="training_artifacts")

    process = instantiate_process_for(
        PinoEvalCalculation,
        code=code,
        dataset=dataset,
        training_artifacts=training,
        training_profile=orm.Dict(dict=_training_profile()),
    )
    with SandboxFolder() as folder:
        with pytest.raises(ValueError, match="does not match code computer"):
            process.prepare_for_submission(folder)


def test_kfold_prepare_for_submission_native_dataset(local_code_factory, pino_dataset_factory) -> None:
    code = local_code_factory("/bin/bash")
    dataset = pino_dataset_factory("native-kfold")

    process = instantiate_process_for(
        PinoKfoldCalculation,
        code=code,
        dataset=dataset,
        training_profile=orm.Dict(dict=_training_profile(mode="kfold")),
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        profile_json = json.loads(Path(folder.get_abs_path("training_profile.json")).read_text(encoding="utf-8"))
        assert profile_json["split_mode"] == "kfold"
        assert len(calcinfo.remote_symlink_list) == 1
        _computer, cache_path, target = calcinfo.remote_symlink_list[0]
        assert target == "dataset_input"
        assert (Path(cache_path) / "manifest.json").exists()
        assert (Path(cache_path) / "members/simulation_0000/manifest.json").exists()


@pytest.mark.parametrize(
    ("script_name", "help_flag"),
    [
        ("eval_adapter.py", "--dataset-dir"),
        ("train_adapter.py", "--train-dir"),
        ("kfold_adapter.py", "--dataset-dir"),
        ("flow3d_to_vtk_adapter.py", "--source-path"),
    ],
)
def test_staged_adapter_can_run_standalone_with_staged_live_tail(script_name: str, help_flag: str, tmp_path: Path) -> None:
    # Resolve from THIS file, not the maintainer's home: an absolute path
    # validates whatever happens to sit at that location, which off the maintainer's
    # machine is nothing and on it is a second checkout.
    scripts_root = Path(__file__).resolve().parents[1] / "src" / "aiida_pino" / "scripts"
    script_path = scripts_root / script_name
    staged_script = tmp_path / script_name
    staged_helper = tmp_path / "live_tail.py"
    staged_script.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")
    staged_helper.write_text((scripts_root / "live_tail.py").read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(staged_script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert help_flag in result.stdout


def test_train_prepare_for_submission_with_init_artifacts(
    local_code_factory, pino_dataset_factory, folder_data_factory, tmp_path
) -> None:
    """Warm start: init_artifacts is staged and --init-dir passed to the adapter."""
    code = local_code_factory("/bin/bash")
    dataset = pino_dataset_factory("warm-train")
    init_dir = tmp_path / "prev_model"
    init_dir.mkdir()
    (init_dir / "model.pth").write_bytes(b"weights")
    init_folder = folder_data_factory(init_dir)

    process = instantiate_process_for(
        PinoTrainCalculation,
        code=code,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=_training_profile()),
        init_artifacts=init_folder,
    )
    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)
        params = calcinfo.codes_info[0].cmdline_params
        assert params[params.index("--init-dir") + 1] == "init_model"
        staged = {src for _uuid, src, _dst in calcinfo.local_copy_list}
        assert staged or Path(folder.get_abs_path("init_model/model.pth")).exists()
