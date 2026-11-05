"""Workflow for importing FLOW-3D compact-NPZ run trees into native datasets."""

from __future__ import annotations

import json
from pathlib import Path

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_pino.calculations.raw_flow3d_to_dataset import RawFlow3DToDatasetCalculation
from aiida_pino.data import PinoDatasetCollectionData
from aiida_pino.utils import validate_flow3d_npz_profile
from aiida_pino.workflows.labels import InheritsCallerLabel


class RawFlow3DImportWorkChain(InheritsCallerLabel, WorkChain):
    """Import `flow3D_scripts` run directories into one native dataset collection."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("source_path", valid_type=orm.Str)
        spec.input("dataset_profile", valid_type=orm.Dict)
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.input("materials_root", valid_type=orm.Str, required=False)
        spec.outline(
            cls.validate_inputs,
            cls.run_import,
            cls.finalize,
        )
        spec.output("dataset", valid_type=PinoDatasetCollectionData)
        spec.output("dataset_summary", valid_type=orm.Dict)
        spec.output("import_summary", valid_type=orm.Dict)
        spec.exit_code(400, "ERROR_EMPTY_SOURCE_PATH", message="The raw FLOW-3D source path is empty.")
        spec.exit_code(401, "ERROR_INVALID_DATASET_PROFILE", message="Invalid `dataset_profile`.")
        spec.exit_code(410, "ERROR_IMPORT_FAILED", message="Raw FLOW-3D dataset import failed.")

    def validate_inputs(self):
        source_path = self.inputs.source_path.value.strip()
        if not source_path:
            return self.exit_codes.ERROR_EMPTY_SOURCE_PATH
        try:
            validate_flow3d_npz_profile(self.inputs.dataset_profile.get_dict())
        except ValueError as exc:
            self.report(f"Invalid dataset_profile: {exc}")
            return self.exit_codes.ERROR_INVALID_DATASET_PROFILE
        self.ctx.source_path = source_path

    def run_import(self):
        inputs = {
            "code": self.inputs.python_code,
            "raw_source_path": orm.Str(self.ctx.source_path),
            "dataset_profile": self.inputs.dataset_profile,
            "source_type": orm.Str("flow3d"),
            "source_uri": orm.Str(f"file://{self.ctx.source_path}"),
            "source_folder_name": orm.Str(Path(self.ctx.source_path).name or "flow3d_runs"),
        }
        if "materials_root" in self.inputs:
            inputs["materials_root"] = self.inputs.materials_root
        future = self.submit(RawFlow3DToDatasetCalculation, **inputs)
        self.report(f"Submitted raw FLOW-3D dataset import: {future.pk}")
        return ToContext(import_job=future)

    def finalize(self):
        if not self.ctx.import_job.is_finished_ok:
            summary = getattr(self.ctx.import_job.outputs, "dataset_summary", None)
            if summary is not None:
                payload = summary.get_dict()
                message = payload.get("error_message") or json.dumps(payload, sort_keys=True)
                self.report(f"Raw FLOW-3D import failed: {message}")
            return self.exit_codes.ERROR_IMPORT_FAILED
        self.out("dataset", self.ctx.import_job.outputs.dataset)
        self.out("dataset_summary", self.ctx.import_job.outputs.dataset_summary)
        self.out(
            "import_summary",
            orm.Dict(
                dict={
                    "source_path": self.ctx.source_path,
                    "dataset_summary": self.ctx.import_job.outputs.dataset_summary.get_dict(),
                }
            ).store(),
        )
