"""Workflow that runs one FLOW-3D case through flow3D_scripts and ingests its NPZ export."""

from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_pino.calculations.flow3d_run import Flow3DRunCalculation
from aiida_pino.calculations.raw_flow3d_to_dataset import RawFlow3DToDatasetCalculation
from aiida_pino.data import PinoDatasetCollectionData
from copy import deepcopy

from aiida_pino.utils import dataset_profile_metadata_gaps, validate_flow3d_npz_profile, validate_flow3d_run_execution_profile
from aiida_pino.workflows.labels import InheritsCallerLabel


def should_file_own_dataset(*, caller) -> bool:
    """True for a standalone run, False for a Flow3DBatchWorkChain child.

    The batch workchain files all of its children together into one named
    dataset. If each child filed itself first, the batch would become N
    one-member datasets and its own filing step would find nothing to attach.
    """
    return caller is None


class Flow3DGenerateWorkChain(InheritsCallerLabel, WorkChain):
    """Generate a FLOW-3D run and convert its compact NPZ export into a native dataset.

    The dataset ingest is skipped for prepare-only runs and for runs whose
    export mode produced no ``npz/<case>.npz`` file; the workchain still
    finishes ok in that case, with only the run outputs.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("run_config", valid_type=orm.Dict)
        spec.input("execution_profile", valid_type=orm.Dict)
        spec.input("dataset_profile", valid_type=orm.Dict)
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.input("materials_root", valid_type=orm.Str, required=False)
        spec.input(
            "target_dataset",
            valid_type=PinoDatasetCollectionData,
            required=False,
            help="File this run's simulation into this existing base dataset instead of "
                 "resolving one by name from dataset_profile.batch_keyword.",
        )
        spec.outline(
            cls.validate_inputs,
            cls.run_simulation,
            cls.run_ingest,
            cls.finalize,
        )
        spec.output("run_summary", valid_type=orm.Dict)
        spec.output("run_artifacts", valid_type=orm.FolderData, required=False)
        spec.output("dataset", valid_type=PinoDatasetCollectionData, required=False)
        spec.output("dataset_summary", valid_type=orm.Dict, required=False)
        spec.exit_code(400, "ERROR_INVALID_EXECUTION_PROFILE", message="Invalid `execution_profile`.")
        spec.exit_code(401, "ERROR_INVALID_DATASET_PROFILE", message="Invalid `dataset_profile`.")
        spec.exit_code(410, "ERROR_SIMULATION_RUN_FAILED", message="FLOW-3D run failed.")
        spec.exit_code(411, "ERROR_DATASET_INGEST_FAILED", message="FLOW-3D NPZ dataset ingest failed.")

    def validate_inputs(self):
        try:
            validate_flow3d_run_execution_profile(self.inputs.execution_profile.get_dict())
        except ValueError as exc:
            self.report(f"Invalid execution_profile: {exc}")
            return self.exit_codes.ERROR_INVALID_EXECUTION_PROFILE
        try:
            validate_flow3d_npz_profile(self.inputs.dataset_profile.get_dict())
        except ValueError as exc:
            self.report(f"Invalid dataset_profile: {exc}")
            return self.exit_codes.ERROR_INVALID_DATASET_PROFILE
        gaps = dataset_profile_metadata_gaps(self.inputs.dataset_profile.get_dict())
        if gaps:
            self.report(gaps)

    def run_simulation(self):
        # Honour a `case_name` typed into the run config. The batch path does
        # this (flow3d_batch.py: `profile["case_name"] = run_config.case_name or
        # <base>_<index>`); the single-case path did not, so a name a user chose
        # was discarded and the run got the adapter's timestamp fallback --
        # 3 of 3 console submits in the database. `case_name` is the run
        # directory, `prepin.<case>`, and the exported NPZ's name, so losing it
        # makes every artifact of that run unidentifiable.
        execution = deepcopy(self.inputs.execution_profile.get_dict())
        if not str(execution.get("case_name") or "").strip():
            typed = str(self.inputs.run_config.get_dict().get("case_name") or "").strip()
            if typed:
                execution["case_name"] = typed
        future = self.submit(
            Flow3DRunCalculation,
            code=self.inputs.python_code,
            run_config=self.inputs.run_config,
            execution_profile=orm.Dict(dict=execution),
        )
        self.report(f"Submitted FLOW-3D run: {future.pk}")
        return ToContext(run_job=future)

    def run_ingest(self):
        if not self.ctx.run_job.is_finished_ok:
            return None
        run_summary = self.ctx.run_job.outputs.run_summary.get_dict()
        if run_summary.get("prepare_only"):
            self.report("Prepare-only run: skipping dataset ingest.")
            return None
        if not run_summary.get("npz_file"):
            self.report(f"No NPZ export for export mode `{run_summary.get('export')}`: skipping dataset ingest.")
            return None

        case_dir = str(run_summary["case_dir"])
        inputs = {
            "code": self.inputs.python_code,
            "raw_source_path": orm.Str(case_dir),
            "dataset_profile": self.inputs.dataset_profile,
            "source_type": orm.Str("flow3d"),
            "source_uri": orm.Str(f"file://{case_dir}"),
            "source_folder_name": orm.Str(Path(case_dir).name),
        }
        if "materials_root" in self.inputs:
            inputs["materials_root"] = self.inputs.materials_root
        future = self.submit(RawFlow3DToDatasetCalculation, **inputs)
        self.report(f"Submitted FLOW-3D NPZ dataset ingest: {future.pk}")
        return ToContext(ingest_job=future)

    def finalize(self):
        run_job = self.ctx.run_job
        if "run_summary" in run_job.outputs:
            self.out("run_summary", run_job.outputs.run_summary)
        if "run_artifacts" in run_job.outputs:
            self.out("run_artifacts", run_job.outputs.run_artifacts)
        if not run_job.is_finished_ok:
            return self.exit_codes.ERROR_SIMULATION_RUN_FAILED
        if "ingest_job" not in self.ctx:
            return None
        if not self.ctx.ingest_job.is_finished_ok:
            return self.exit_codes.ERROR_DATASET_INGEST_FAILED
        self.out("dataset", self.ctx.ingest_job.outputs.dataset)
        # Emitted for every successful ingest, including batch children. It used
        # to be the last line of the filing method, so a child of a batch -- or
        # any run with nothing left to file -- finished without it.
        self.out("dataset_summary", self.ctx.ingest_job.outputs.dataset_summary)
        if should_file_own_dataset(caller=self.node.caller) or "target_dataset" in self.inputs:
            # An explicit target overrides the batch-child rule: if the caller
            # named the dataset, that is where the case goes.
            self._file_into_base_dataset()

    def _file_into_base_dataset(self):
        """File a standalone run into its batch's named base dataset.

        Requeues, one-off validation cases and the web console's single-run
        endpoint all submit this workchain directly, bypassing
        Flow3DBatchWorkChain's filing step; without this they stay ingested but
        unattached. Joins the batch's existing dataset when there is one (a
        requeued case belongs with its sweep, not in a folder of its own).
        Failures are reported and swallowed -- the solve is the expensive part.
        """
        from .dataset_filing import dataset_metadata_from_profile, file_simulations
        from .flow3d_batch import member_simulations

        profile = self.inputs.dataset_profile.get_dict()
        target = self.inputs.get("target_dataset", None)
        name = (
            str(target.label or "").strip() if target is not None else ""
        ) or str(profile.get("batch_keyword") or "").strip() or "flow3d_runs"
        sims = member_simulations(self.ctx.ingest_job.outputs.dataset)
        try:
            dataset, action = file_simulations(
                sims,
                name=name,
                target_dataset=target,
                extra_metadata=dataset_metadata_from_profile(profile),
            )
            if dataset is None:
                self.report("Every simulation already belongs to a base dataset; nothing to file.")
                return
            where = f" (explicit target {target.pk})" if target is not None else ""
            self.report(f"Filed {len(sims)} simulation(s) into base dataset '{name}' "
                        f"(pk {dataset.pk}, {action}){where}.")
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a run
            self.report(f"Could not file simulation into a base dataset: {exc}")
