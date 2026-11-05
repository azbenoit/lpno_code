"""CalcJob for converting Flow3D flasks into VTK files."""

from __future__ import annotations

import json

from aiida import orm
from aiida.common.folders import Folder

from .base import ScriptCalculation


class Flow3DToVtkCalculation(ScriptCalculation):
    """Run a ParaView-backed adapter that converts Flow3D folders into VTK outputs."""

    script_name = "flow3d_to_vtk_adapter.py"
    parser_entry_point = "pino.flow3d_to_vtk"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("source_path", valid_type=orm.Str, help="Absolute path to the raw Flow3D directory.")
        spec.input("conversion_profile", valid_type=orm.Dict, help="Conversion profile for Flow3D import.")
        spec.output("vtk_folder", valid_type=orm.FolderData)
        spec.output("conversion_summary", valid_type=orm.Dict)
        spec.exit_code(310, "ERROR_MISSING_SUMMARY", message="Flow3D conversion did not produce `flow3d_summary.json`.")
        spec.exit_code(311, "ERROR_CONVERSION_FAILED", message="Flow3D conversion adapter reported a failure.")
        spec.exit_code(312, "ERROR_NO_CASES_CONVERTED", message="Flow3D conversion converted zero cases.")

    def prepare_for_submission(self, folder: Folder):
        script_name = self.stage_script(folder)
        profile_name = "conversion_profile.json"
        with folder.open(profile_name, "w", encoding="utf-8") as handle:
            json.dump(self.inputs.conversion_profile.get_dict(), handle, indent=2, sort_keys=True)

        codeinfo = self.build_codeinfo(
            [
                script_name,
                "--source-path",
                self.inputs.source_path.value,
                "--output-dir",
                "vtk_output",
                "--profile-file",
                profile_name,
            ]
        )
        return self.build_calcinfo(codeinfo, [self.options.output_filename, "flow3d_summary.json", "vtk_output"])
