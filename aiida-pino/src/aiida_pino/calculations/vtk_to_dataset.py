"""CalcJob for preparing cached dataset folders from VTK data."""

from __future__ import annotations

import json

from aiida import orm
from aiida.common.folders import Folder

from .base import ScriptCalculation
from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData
from aiida_pino.utils import validate_source_type_port


class VtkToDatasetCalculation(ScriptCalculation):
    """Run the current datasource code against a staged VTK folder."""

    script_name = "vtk_to_dataset_adapter.py"
    parser_entry_point = "pino.vtk_to_dataset"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("vtk_folder", valid_type=orm.FolderData, required=False, help="Staged VTK folder to import.")
        spec.input("vtk_source_path", valid_type=orm.Str, required=False, help="External VTK folder path to import.")
        spec.input("dataset_profile", valid_type=orm.Dict, help="Canonical dataset import profile.")
        spec.input(
            "source_uri",
            valid_type=orm.Str,
            required=False,
            help="Optional provenance URI recorded on the resulting dataset node.",
        )
        spec.input(
            "source_folder_name",
            valid_type=orm.Str,
            required=False,
            help="Optional human-readable folder name used in dataset labels.",
        )
        spec.input(
            "source_type",
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str("flow3d"),
            validator=validate_source_type_port,
            help="Dataset source flag stored in metadata.",
        )
        spec.input(
            "source_metadata",
            valid_type=orm.Dict,
            required=False,
            help="Optional metadata overlay merged into the stored native dataset metadata.",
        )
        spec.output("dataset", valid_type=PinoDatasetCollectionData)
        # The simulations this import created. Dynamic because the count is
        # whatever the source directory held; a re-imported simulation is NOT
        # emitted here -- it keeps the creator it already has.
        spec.output_namespace(
            "simulations",
            valid_type=PinoSimulationData,
            required=False,
            dynamic=True,
            help="Simulations created by this import, one per new case.",
        )
        spec.output("dataset_summary", valid_type=orm.Dict)
        spec.exit_code(300, "ERROR_MISSING_SUMMARY", message="VTK conversion did not produce `dataset_summary.json`.")
        spec.exit_code(301, "ERROR_CONVERSION_FAILED", message="VTK conversion adapter reported a failure.")
        spec.exit_code(302, "ERROR_MISSING_DATASET_PAYLOAD", message="VTK conversion did not produce dataset payload files.")
        spec.exit_code(303, "ERROR_INVALID_DATASET_PAYLOAD", message="VTK conversion payload files are invalid.")

    def prepare_for_submission(self, folder: Folder):
        vtk_folder = getattr(self.inputs, "vtk_folder", None)
        vtk_source_path = getattr(self.inputs, "vtk_source_path", None)
        source_uri = getattr(self.inputs, "source_uri", None)
        source_folder_name = getattr(self.inputs, "source_folder_name", None)

        has_folder = vtk_folder is not None
        has_source_path = vtk_source_path is not None
        if has_folder == has_source_path:
            raise ValueError("Exactly one of `vtk_folder` or `vtk_source_path` must be provided.")

        script_name = self.stage_script(folder)
        profile_name = "dataset_profile.json"
        with folder.open(profile_name, "w", encoding="utf-8") as handle:
            json.dump(self.inputs.dataset_profile.get_dict(), handle, indent=2, sort_keys=True)
        if "source_metadata" in self.inputs:
            source_metadata_name = "source_metadata.json"
            with folder.open(source_metadata_name, "w", encoding="utf-8") as handle:
                json.dump(self.inputs.source_metadata.get_dict(), handle, indent=2, sort_keys=True)
        else:
            source_metadata_name = None

        cmdline_params = [
            script_name,
            "--output-dir",
            "dataset_output",
            "--profile-file",
            profile_name,
            "--source-type",
            self.inputs.source_type.value,
        ]
        if source_uri is not None:
            cmdline_params.extend(["--source-uri", source_uri.value])
        if source_folder_name is not None:
            cmdline_params.extend(["--source-folder-name", source_folder_name.value])
        if source_metadata_name is not None:
            cmdline_params.extend(["--source-metadata-file", source_metadata_name])
        if bool(self.inputs.dataset_profile.get_dict().get("force_recompute", False)):
            cmdline_params.append("--force-recompute")
        local_copy_list: list[tuple[str, str, str | None]] = []
        if has_folder:
            cmdline_params.extend(["--vtk-input-dir", "vtk_input"])
            local_copy_list.append((vtk_folder.uuid, ".", "vtk_input"))
        else:
            cmdline_params.extend(["--vtk-source-path", vtk_source_path.value])

        codeinfo = self.build_codeinfo(cmdline_params)
        return self.build_calcinfo(
            codeinfo,
            [self.options.output_filename, "dataset_summary.json"],
            retrieve_temporary_list=["dataset_output"],
            local_copy_list=local_copy_list,
        )
