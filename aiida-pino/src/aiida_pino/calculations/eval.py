"""CalcJob for evaluating a trained PINO run."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from aiida import orm
from aiida.common.folders import Folder

from aiida_pino.data import PinoModelData
from aiida_pino.utils import normalize_dataset_collection_node, validate_dataset_input_port

from .base import ScriptCalculation, runtime_fields_from_training_profile


class PinoEvalCalculation(ScriptCalculation):
    """Run the existing standard PINO evaluation entrypoint."""

    script_name = "eval_adapter.py"
    parser_entry_point = "pino.eval"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "dataset",
            valid_type=orm.Data,
            validator=validate_dataset_input_port,
            help="Prepared dataset folder for evaluation.",
        )
        spec.input(
            "training_artifacts",
            valid_type=(PinoModelData, orm.FolderData, orm.RemoteData),
            help="Registered model bundle or legacy training artifact folder.",
        )
        spec.input("training_profile", valid_type=orm.Dict, help="Training profile reused for evaluation.")
        spec.output("eval_artifacts", valid_type=orm.FolderData)
        spec.output("metrics", valid_type=orm.Dict)
        spec.output("eval_summary", valid_type=orm.Dict)
        spec.exit_code(500, "ERROR_MISSING_SUMMARY", message="Evaluation did not produce `eval_summary.json`.")
        spec.exit_code(501, "ERROR_MISSING_ARTIFACTS", message="Evaluation did not produce an `eval_output` artifacts folder.")
        spec.exit_code(502, "ERROR_MISSING_METRICS", message="Evaluation produced no `metrics.json` and no stdout metrics.")

    def _supply_h_norm_record(self, target: Path) -> None:
        """Stage the H_norm denominator the model TRAINED with beside its weights.

        A model trained before 2026-10-30 carries no `h_norm_denom.json`, and its
        evaluation used to refit the denominator on the scored split, scaling every H_norm
        input by train max / scored max (`aiida_pino.utils.h_norm`). The record is rebuilt
        here from the model's training-source simulations. A failure only warns: the
        evaluation then recomputes the value from the training data if it can read it, and
        otherwise refits on the scored split under a banner and flags it in its metrics.
        """
        try:
            from aiida_pino.utils.h_norm import backfill_h_norm_record

            path, why = backfill_h_norm_record(self.inputs.training_artifacts, target)
        except Exception as exc:  # noqa: BLE001 - an eval must not die on bookkeeping
            self.logger.warning(
                f"H_norm: could not supply the model's training denominator ({exc}). The "
                "evaluation falls back as described in lpno h_norm_record.")
            return
        if path is not None:
            self.report(f"H_norm: supplied the training denominator, {why}, as {path.name}")

    def prepare_for_submission(self, folder: Folder):
        script_name = self.stage_script(folder)
        profile_name = "training_profile.json"
        with folder.open(profile_name, "w", encoding="utf-8") as handle:
            profile = self.inputs.training_profile.get_dict()
            json.dump(profile, handle, indent=2, sort_keys=True)
        runtime_fields = runtime_fields_from_training_profile(profile)

        training_dir = "training_input"
        local_copy_list: list[tuple[str, str, str | None]] = []
        remote_symlink_list: list[tuple[str, str, str | None]] = []
        dataset = self.inputs.dataset
        if not isinstance(dataset, orm.FolderData):
            dataset = normalize_dataset_collection_node(dataset, label="dataset")

        if isinstance(self.inputs.training_artifacts, PinoModelData):
            training_root = folder.get_abs_path(training_dir)
            # `.hydra` is STAGED, deliberately. It used to be deleted here (from
            # 2026-04-07), which silently defeated the archived-config evaluation added on
            # 2026-08-14: `eval_adapter.archived_config_dir`
            # looks for `<training_dir>/.hydra/config.yaml`, staging removed it, and the
            # adapter fell back to TODAY's packaged config with only a warning in the log
            # -- so that fix was inert from the day it landed. The consequence is the one
            # its docstring says a provenance store must not have: a stored model becomes
            # unloadable as the packaged schema moves. Measured 2026-08-24 on an eval
            # of a model that the same machine had trained and scored cleanly days
            # earlier: `Missing key(s) in state_dict: "proc_steps.8.edge_mlp..."`, because
            # the eval skeleton was rebuilt from a config whose arch defaults had drifted
            # away from the archived `processor_steps: 8`.
            self.inputs.training_artifacts.base.repository.copy_tree(training_root)
            primary_checkpoint = self.inputs.training_artifacts.resolve_primary_checkpoint()
            checkpoint_parent = PurePosixPath(primary_checkpoint).parent
            if str(checkpoint_parent) not in ("", "."):
                training_dir = str(PurePosixPath(training_dir) / checkpoint_parent)
            self._supply_h_norm_record(Path(folder.get_abs_path(training_dir)))
        else:
            local_entries, remote_entries = self.stage_folderlike_input(self.inputs.training_artifacts, "training_input")
            local_copy_list.extend(local_entries)
            remote_symlink_list.extend(remote_entries)

        codeinfo = self.build_codeinfo(
            [
                script_name,
                "--dataset-dir",
                "dataset_input",
                "--training-dir",
                training_dir,
                "--output-dir",
                "eval_output",
                "--profile-file",
                profile_name,
            ]
        )
        local_entries, remote_entries = self.stage_dataset_input(
            folder,
            dataset,
            "dataset_input",
            runtime_fields=runtime_fields,
        )
        local_copy_list.extend(local_entries)
        remote_symlink_list.extend(remote_entries)
        return self.build_calcinfo(
            codeinfo,
            [self.options.output_filename, "eval_summary.json", "metrics.json", "eval_output"],
            local_copy_list=local_copy_list,
            remote_symlink_list=remote_symlink_list,
        )
