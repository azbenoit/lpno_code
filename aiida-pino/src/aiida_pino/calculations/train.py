"""CalcJob for training the current PINO workflow."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from aiida import orm
from aiida.common.folders import Folder

from aiida_pino.data import PinoModelData
from aiida_pino.utils import normalize_dataset_collection_node, validate_dataset_input_port

from .base import ScriptCalculation, runtime_fields_from_training_profile


class PinoTrainCalculation(ScriptCalculation):
    """Run the existing standard PINO training entrypoint."""

    script_name = "train_adapter.py"
    parser_entry_point = "pino.train"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "train_dataset",
            valid_type=orm.Data,
            validator=validate_dataset_input_port,
            help="Prepared training dataset folder.",
        )
        spec.input(
            "val_dataset",
            valid_type=orm.Data,
            validator=validate_dataset_input_port,
            help="Prepared validation dataset folder.",
        )
        spec.input("training_profile", valid_type=orm.Dict, help="Training profile.")
        spec.input(
            "init_artifacts",
            valid_type=(PinoModelData, orm.FolderData, orm.RemoteData),
            required=False,
            help="Previously trained model bundle whose checkpoint files seed "
            "network_dir before training starts (warm start via the trainer's "
            "own checkpoint resume).",
        )
        spec.output("training_artifacts", valid_type=orm.FolderData)
        spec.output("train_summary", valid_type=orm.Dict)
        spec.exit_code(500, "ERROR_MISSING_SUMMARY", message="Training did not produce `train_summary.json`.")
        spec.exit_code(501, "ERROR_MISSING_ARTIFACTS", message="Training did not produce a `training_output` artifacts folder.")

    def prepare_for_submission(self, folder: Folder):
        script_name = self.stage_script(folder)
        profile_name = "training_profile.json"
        with folder.open(profile_name, "w", encoding="utf-8") as handle:
            profile = self.inputs.training_profile.get_dict()
            json.dump(profile, handle, indent=2, sort_keys=True)
        runtime_fields = runtime_fields_from_training_profile(profile)

        train_dataset = self.inputs.train_dataset
        val_dataset = self.inputs.val_dataset
        if not isinstance(train_dataset, orm.FolderData):
            train_dataset = normalize_dataset_collection_node(train_dataset, label="train_dataset")
        if not isinstance(val_dataset, orm.FolderData):
            val_dataset = normalize_dataset_collection_node(val_dataset, label="val_dataset")

        val_dir = "val_dataset"
        shared_dataset = train_dataset.uuid == val_dataset.uuid
        if shared_dataset:
            val_dir = "train_dataset"

        args = [
            script_name,
            "--train-dir",
            "train_dataset",
            "--val-dir",
            val_dir,
            "--output-dir",
            "training_output",
            "--profile-file",
            profile_name,
        ]
        local_copy_list: list[tuple[str, str, str | None]] = []
        remote_symlink_list: list[tuple[str, str, str | None]] = []
        if "init_artifacts" in self.inputs:
            init_dir = "init_model"
            init = self.inputs.init_artifacts
            if isinstance(init, PinoModelData):
                # Same staging as the eval calc: model repo tree, checkpoint
                # subdir resolved so --init-dir points at the .pth files.
                init_root = folder.get_abs_path(init_dir)
                init.base.repository.copy_tree(init_root)
                checkpoint_parent = PurePosixPath(init.resolve_primary_checkpoint()).parent
                if str(checkpoint_parent) not in ("", "."):
                    init_dir = str(PurePosixPath(init_dir) / checkpoint_parent)
            else:
                local_entries, remote_entries = self.stage_folderlike_input(init, init_dir)
                local_copy_list.extend(local_entries)
                remote_symlink_list.extend(remote_entries)
            args.extend(["--init-dir", init_dir])
        codeinfo = self.build_codeinfo(args)
        dataset_targets = [(train_dataset, "train_dataset")]
        if not shared_dataset:
            dataset_targets.append((val_dataset, "val_dataset"))
        for node, target in dataset_targets:
            local_entries, remote_entries = self.stage_dataset_input(folder, node, target, runtime_fields=runtime_fields)
            local_copy_list.extend(local_entries)
            remote_symlink_list.extend(remote_entries)
        return self.build_calcinfo(
            codeinfo,
            [self.options.output_filename, "train_summary.json", "training_output"],
            local_copy_list=local_copy_list,
            remote_symlink_list=remote_symlink_list,
        )
