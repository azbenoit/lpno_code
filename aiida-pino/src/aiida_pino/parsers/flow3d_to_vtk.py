"""Parser for Flow3D to VTK conversion jobs."""

from __future__ import annotations

from aiida import orm
from aiida.common import exceptions

from .common import FolderResultParser


class Flow3DToVtkParser(FolderResultParser):
    """Parse VTK conversion outputs."""

    folder_label = "vtk_folder"
    folder_subpath = "vtk_output"
    summary_label = "conversion_summary"
    summary_filename = "flow3d_summary.json"

    def parse(self, **kwargs):
        try:
            summary_data = self.read_json(self.summary_filename)
        except exceptions.ParsingError:
            return self.exit_codes.ERROR_MISSING_SUMMARY

        summary = orm.Dict(dict=summary_data)
        summary.label = self.summary_label
        self.out(self.summary_label, summary)

        if summary_data.get("status") == "error":
            if not summary_data.get("converted_cases"):
                return self.exit_codes.ERROR_NO_CASES_CONVERTED
            return self.exit_codes.ERROR_CONVERSION_FAILED

        folder = self.build_folder_output(self.folder_subpath)
        folder.label = self.folder_label
        folder.base.extras.set("aiida_pino_kind", self.folder_label)
        self.out(self.folder_label, folder)
