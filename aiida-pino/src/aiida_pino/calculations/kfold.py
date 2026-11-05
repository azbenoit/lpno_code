"""CalcJob for packaged PINO k-fold training and evaluation."""

from __future__ import annotations

import json

from aiida import orm
from aiida.common.folders import Folder

from aiida_pino.utils import normalize_dataset_collection_node, validate_dataset_input_port

from .base import ScriptCalculation, runtime_fields_from_training_profile


class PinoKfoldCalculation(ScriptCalculation):
    """Run the packaged k-fold PINO entrypoint."""

    script_name = "kfold_adapter.py"
    parser_entry_point = "pino.kfold"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "dataset",
            valid_type=orm.Data,
            validator=validate_dataset_input_port,
            help="Prepared dataset folder for k-fold training/evaluation.",
        )
        spec.input("training_profile", valid_type=orm.Dict, help="K-fold runtime profile.")
        spec.output("kfold_artifacts", valid_type=orm.FolderData)
        spec.output("metrics", valid_type=orm.Dict)
        spec.output("kfold_summary", valid_type=orm.Dict)
        spec.exit_code(500, "ERROR_MISSING_SUMMARY", message="K-fold run did not produce `kfold_summary.json`.")
        spec.exit_code(501, "ERROR_MISSING_ARTIFACTS", message="K-fold run did not produce a `kfold_output` artifacts folder.")
        spec.exit_code(502, "ERROR_MISSING_METRICS", message="K-fold run produced no `metrics.json` and no stdout metrics.")

    def prepare_for_submission(self, folder: Folder):
        script_name = self.stage_script(folder)
        profile_name = "training_profile.json"
        with folder.open(profile_name, "w", encoding="utf-8") as handle:
            profile = self.inputs.training_profile.get_dict()
            json.dump(profile, handle, indent=2, sort_keys=True)
        runtime_fields = runtime_fields_from_training_profile(profile)

        codeinfo = self.build_codeinfo(
            [
                script_name,
                "--dataset-dir",
                "dataset_input",
                "--output-dir",
                "kfold_output",
                "--profile-file",
                profile_name,
            ]
        )
        dataset = self.inputs.dataset
        if not isinstance(dataset, orm.FolderData):
            dataset = normalize_dataset_collection_node(dataset, label="dataset")
        local_copy_list, remote_symlink_list = self.stage_dataset_input(
            folder,
            dataset,
            "dataset_input",
            runtime_fields=runtime_fields,
        )
        return self.build_calcinfo(
            codeinfo,
            [self.options.output_filename, "kfold_summary.json", "metrics.json", "kfold_output"],
            local_copy_list=local_copy_list,
            remote_symlink_list=remote_symlink_list,
        )
