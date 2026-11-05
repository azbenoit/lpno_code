"""WorkChain for Flow3D ingest and dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path

from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction

from aiida_pino.calculations.flow3d_to_vtk import Flow3DToVtkCalculation
from aiida_pino.calculations.vtk_to_dataset import VtkToDatasetCalculation
from aiida_pino.data import PinoDatasetCollectionData
from aiida_pino.utils import (
    CONVERSION_PROFILE_KEYS,
    DATASET_PROFILE_KEYS,
    discover_flow3d_cases,
    read_flow3d_case_metadata,
    validate_profile_keys,
)
from aiida_pino.workflows.labels import InheritsCallerLabel


@calcfunction
def merge_ingest_summaries(source_path: orm.Str, conversion_summary: orm.Dict, dataset_summary: orm.Dict) -> orm.Dict:
    """Merge the conversion and dataset summaries into one queryable dict."""
    merged = {
        "source_path": source_path.value,
        "conversion": conversion_summary.get_dict(),
        "dataset": dataset_summary.get_dict(),
    }
    return orm.Dict(dict=merged)


class Flow3DIngestWorkChain(InheritsCallerLabel, WorkChain):
    """Convert a Flow3D folder into VTKs and a native PINO dataset node."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("source_path", valid_type=orm.Str)
        spec.input("conversion_profile", valid_type=orm.Dict)
        spec.input("dataset_profile", valid_type=orm.Dict)
        spec.input("pvpython_code", valid_type=orm.AbstractCode)
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.input(
            "source_metadata",
            valid_type=orm.Dict,
            required=False,
            help="Optional metadata overlay merged over the auto-scanned FLOW-3D case manifests.",
        )
        spec.outline(
            cls.validate_inputs,
            cls.run_conversion,
            cls.run_dataset_preparation,
            cls.finalize,
        )
        spec.output("vtk_folder", valid_type=orm.FolderData)
        spec.output("dataset", valid_type=PinoDatasetCollectionData)
        spec.output("ingest_summary", valid_type=orm.Dict)
        spec.exit_code(200, "ERROR_EMPTY_SOURCE_PATH", message="`source_path` must not be empty.")
        spec.exit_code(201, "ERROR_INVALID_CONVERSION_PROFILE", message="Invalid `conversion_profile`.")
        spec.exit_code(202, "ERROR_INVALID_DATASET_PROFILE", message="Invalid `dataset_profile`.")

    def validate_inputs(self):
        if not self.inputs.source_path.value.strip():
            return self.exit_codes.ERROR_EMPTY_SOURCE_PATH
        try:
            validate_profile_keys(
                "conversion_profile", self.inputs.conversion_profile.get_dict(), CONVERSION_PROFILE_KEYS
            )
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_CONVERSION_PROFILE
        try:
            validate_profile_keys("dataset_profile", self.inputs.dataset_profile.get_dict(), DATASET_PROFILE_KEYS)
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_DATASET_PROFILE

    def run_conversion(self):
        future = self.submit(
            Flow3DToVtkCalculation,
            code=self.inputs.pvpython_code,
            source_path=self.inputs.source_path,
            conversion_profile=self.inputs.conversion_profile,
        )
        self.report(f"Submitted Flow3D conversion job: {future.pk}")
        return ToContext(conversion=future)

    def _collect_case_manifests(self) -> dict:
        """Auto-scan the source path for flow3D_scripts run dirs with `case_manifest.json` (BUG-02)."""
        try:
            cases = discover_flow3d_cases(self.inputs.source_path.value)
        except ValueError:
            return {}
        manifests = {}
        for entry in cases:
            case_dir = Path(entry["case_dir"])
            try:
                manifests[case_dir.name] = read_flow3d_case_metadata(case_dir)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.report(f"Skipping unreadable FLOW-3D case manifest in {case_dir}: {exc}")
        return manifests

    def run_dataset_preparation(self):
        case_manifests = self._collect_case_manifests()
        source_metadata = {"flow3d": {"case_manifests": case_manifests}} if case_manifests else {}
        if "source_metadata" in self.inputs:
            source_metadata.update(self.inputs.source_metadata.get_dict())

        inputs = {
            "code": self.inputs.python_code,
            "vtk_folder": self.ctx.conversion.outputs.vtk_folder,
            "dataset_profile": self.inputs.dataset_profile,
            "source_type": orm.Str("flow3d"),
            "source_uri": orm.Str(f"file://{self.inputs.source_path.value}"),
            "source_folder_name": orm.Str(
                Path(self.inputs.source_path.value).name or Path(self.inputs.source_path.value).anchor
            ),
        }
        if source_metadata:
            inputs["source_metadata"] = orm.Dict(dict=source_metadata)
        future = self.submit(VtkToDatasetCalculation, **inputs)
        self.report(f"Submitted dataset preparation job: {future.pk}")
        return ToContext(dataset=future)

    def finalize(self):
        self.out("vtk_folder", self.ctx.conversion.outputs.vtk_folder)
        self.out("dataset", self.ctx.dataset.outputs.dataset)
        self.out(
            "ingest_summary",
            merge_ingest_summaries(
                self.inputs.source_path,
                self.ctx.conversion.outputs.conversion_summary,
                self.ctx.dataset.outputs.dataset_summary,
            ),
        )
