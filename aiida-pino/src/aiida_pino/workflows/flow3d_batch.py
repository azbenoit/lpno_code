"""Batch workflow submitting a bounded series of FLOW-3D generate workchains."""

from __future__ import annotations

from copy import deepcopy

from aiida import orm
from aiida.engine import ToContext, WorkChain, while_

from aiida_pino.data import PinoDatasetCollectionData
from aiida_pino.utils import dataset_profile_metadata_gaps, validate_flow3d_npz_profile, validate_flow3d_run_execution_profile
from aiida_pino.workflows.labels import InheritsCallerLabel

from .flow3d_generate import Flow3DGenerateWorkChain


def base_dataset_name(dataset_profile: dict, case_base: str | None) -> str:
    """Human-readable folder name for a batch's base dataset.

    Prefers the `batch_keyword` the sweep already labels its simulations with, so
    the dataset folder and the simulation attribute agree; falls back to the
    positional case base, then to a fixed literal so the name is never empty.
    """
    keyword = str(dataset_profile.get("batch_keyword") or "").strip()
    if keyword:
        return keyword
    base = str(case_base or "").strip()
    return base or "flow3d_batch"


def ingested_dataset_pks(child_summaries: list[dict]) -> list[int]:
    """Per-child single-simulation dataset pks, skipping children that never ingested."""
    return [int(s["dataset_pk"]) for s in child_summaries if s.get("dataset_pk") is not None]


def dataset_display_name(dataset_node) -> str:
    """Human name of a dataset collection.

    `create_base_dataset(name=...)` folds the name into the label
    (`"<name> | <cache> | <source>"`) and the identifier attributes; there is
    no `collection_name` attribute on the stored node, and asking for one
    raises (AiiDA attribute access) -- which silently broke both the batch
    summary and the standalone extend-by-name lookup until 2026-07-29.
    """
    return str(getattr(dataset_node, "label", "") or "").split("|")[0].strip()


def should_extend_existing(*, existing) -> bool:
    """True when a dataset of this name already exists, so filing must extend it.

    `create_base_dataset` never merges: filing a second batch under the same
    `batch_keyword` mints a twin with an identical display name and makes
    selection-by-name ambiguous (observed 2026-07-29: two `bundle_gate`
    datasets from two batches of one campaign).
    """
    return existing is not None


def member_simulations(dataset_node) -> list:
    """Member simulations of an ingest collection, resolved by **UUID**.

    Membership is an attribute, not a graph link -- the npz_file ingest stores
    its simulations as standalone nodes with NO links to the collection, so
    link-based discovery (`get_incoming(node_class=PinoSimulationData)`) matches
    nothing: on 2026-07-29 every batch of the calibration campaign reported "No
    unattached simulations to file" and 23 simulations sat orphaned until filed
    by hand.

    Resolution is by uuid and never by the sibling `member_simulation_pks`
    attribute. Primary keys are not stable across an archive round-trip: the
    2026-07-21 SQLite->PostgreSQL migration renumbered every node, leaving 162
    collections whose stored pks load *different* simulations -- or Dicts, Strs
    and CalcJobNodes -- with no error raised. UUIDs survived it intact in all
    1,064 collections.
    """
    getter = getattr(dataset_node, "iter_member_simulations", None)
    if getter is not None:
        return list(getter())
    uuids = dataset_node.base.attributes.get("member_simulation_uuids", None) or []
    return [orm.load_node(str(uuid)) for uuid in uuids]


class Flow3DBatchWorkChain(InheritsCallerLabel, WorkChain):
    """Run a list of full flow3D_scripts configs through Flow3DGenerateWorkChain.

    ``max_concurrent_runs`` defaults to 1 because the FLOW-3D solves are
    machine-wide serialized anyway: a second concurrent run just blocks on the
    seat lock in ``~/.flow3d/spool/.seat.lock`` until the running solve
    finishes.

    A config that carries its own ``case_name`` keeps it; the rest get a unique
    positional ``<base>_<index>`` derived from the shared execution profile (or
    from this workchain's UUID when none is set). Real flow3D_scripts sweeps are
    named -- ``mck_000``, ``t64v_017`` -- and their design tables, per-case
    material folders and downstream analysis all key off that name, so renaming
    them positionally silently detaches a run from its own design row. Either way
    no two children share a run directory: duplicate names are rejected up front.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("run_configs", valid_type=orm.List, help="List of full flow3D_scripts config dicts.")
        spec.input("execution_profile", valid_type=orm.Dict)
        spec.input("dataset_profile", valid_type=orm.Dict)
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.input("materials_root", valid_type=orm.Str, required=False)
        spec.input("max_concurrent_runs", valid_type=orm.Int, required=False, default=lambda: orm.Int(1))
        spec.input(
            "target_dataset",
            valid_type=PinoDatasetCollectionData,
            required=False,
            help="File every case of this sweep into this existing base dataset instead of "
                 "resolving one by name from dataset_profile.batch_keyword.",
        )
        spec.outline(
            cls.validate_inputs,
            cls.initialize_batches,
            while_(cls.has_next_batch)(
                cls.submit_next_batch,
                cls.collect_current_batch,
            ),
            cls.finalize,
        )
        spec.output("batch_summary", valid_type=orm.Dict)
        spec.exit_code(400, "ERROR_INVALID_RUN_CONFIGS", message="`run_configs` must be a non-empty list of config dicts.")
        spec.exit_code(401, "ERROR_INVALID_EXECUTION_PROFILE", message="Invalid `execution_profile`.")
        spec.exit_code(402, "ERROR_INVALID_DATASET_PROFILE", message="Invalid `dataset_profile`.")

    def validate_inputs(self):
        run_configs = self.inputs.run_configs.get_list()
        if not run_configs or not all(isinstance(entry, dict) for entry in run_configs):
            return self.exit_codes.ERROR_INVALID_RUN_CONFIGS
        # Names the configs carry are kept verbatim, so a collision would point two
        # children at one run directory and the second would overwrite the first.
        named = [str(entry.get("case_name") or "").strip() for entry in run_configs]
        duplicates = sorted({name for name in named if name and named.count(name) > 1})
        if duplicates:
            self.report(f"Duplicate case_name in run_configs: {', '.join(duplicates)}")
            return self.exit_codes.ERROR_INVALID_RUN_CONFIGS
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
        self.ctx.run_configs = run_configs
        # Batch scope as a queryable property. With `max_concurrent_runs=1`
        # `called_descendants` returns one child, so "4 arms, 1 done" could only
        # be answered by reading the `run_configs` input list by hand -- the
        # planned work was invisible on the node.
        self.node.base.extras.set("planned_runs", len(run_configs))
        self.node.base.extras.set("planned_case_names", named)

    def _record_progress(self) -> None:
        """`done / planned`, refreshed as the cursor advances."""
        done = int(getattr(self.ctx, "cursor", 0))
        planned = int(self.node.base.extras.get("planned_runs", None) or 0)
        self.node.base.extras.set("completed_runs", done)
        if planned:
            self.node.set_process_status(f"run {min(done + 1, planned)} of {planned}")

    def initialize_batches(self):
        self.ctx.cursor = 0
        self.ctx.child_summaries = []
        self.ctx.current_batch_keys = []
        profile = self.inputs.execution_profile.get_dict()
        self.ctx.case_base = str(profile.get("case_name") or f"flow3d_batch_{self.node.uuid[:8]}")

    def has_next_batch(self):
        return self.ctx.cursor < len(self.ctx.run_configs)

    def submit_next_batch(self):
        max_concurrent = max(1, int(self.inputs.max_concurrent_runs))
        batch = self.ctx.run_configs[self.ctx.cursor : self.ctx.cursor + max_concurrent]
        self.ctx.current_batch_keys = []
        updates = {}
        for offset, run_config in enumerate(batch):
            index = self.ctx.cursor + offset
            key = f"batch_job_{index}"
            self.ctx.current_batch_keys.append(key)
            profile = deepcopy(self.inputs.execution_profile.get_dict())
            profile["case_name"] = str(run_config.get("case_name") or "").strip() or f"{self.ctx.case_base}_{index:04d}"
            inputs = {
                "run_config": orm.Dict(dict=run_config),
                "execution_profile": orm.Dict(dict=profile),
                "dataset_profile": self.inputs.dataset_profile,
                "python_code": self.inputs.python_code,
            }
            if "materials_root" in self.inputs:
                inputs["materials_root"] = self.inputs.materials_root
            future = self.submit(Flow3DGenerateWorkChain, **inputs)
            self.report(f"Submitted FLOW-3D batch child {future.pk} as `{profile['case_name']}`")
            updates[key] = future
        self.ctx.cursor += len(batch)
        self._record_progress()
        return ToContext(**updates)

    def collect_current_batch(self):
        for key in self.ctx.current_batch_keys:
            node = self.ctx[key]
            summary = {
                "pk": node.pk,
                "process_label": node.process_label,
                "state": getattr(node.process_state, "value", "n/a"),
                "exit_status": getattr(node, "exit_status", None),
            }
            if "run_summary" in node.outputs:
                run_summary = node.outputs["run_summary"].get_dict()
                summary["case_name"] = run_summary.get("case_name")
                summary["case_dir"] = run_summary.get("case_dir")
                summary["run_status"] = run_summary.get("status")
            if "dataset" in node.outputs:
                summary["dataset_pk"] = node.outputs["dataset"].pk
            self.ctx.child_summaries.append(summary)
        # `batch_summary` is only emitted at finalize, so a 299-case sweep that is
        # killed or dies midway would leave no record of what completed. Mirroring the
        # summaries into an extra as they land makes progress readable at any point.
        self.node.base.extras.set("child_summaries", self.ctx.child_summaries)

    def finalize(self):
        batch_summary = {
            "total_runs": len(self.ctx.run_configs),
            "max_concurrent_runs": max(1, int(self.inputs.max_concurrent_runs)),
            "case_base": self.ctx.case_base,
            "children": self.ctx.child_summaries,
        }
        base = self._file_into_base_dataset()
        if base is not None:
            batch_summary["base_dataset_pk"] = base.pk
            batch_summary["base_dataset_name"] = dataset_display_name(base)
        self.out("batch_summary", orm.Dict(dict=batch_summary).store())

    def _file_into_base_dataset(self):
        """Group this batch's simulations into one named, human-readable dataset.

        Filing used to be a separate manual step, so a finished sweep left its
        simulations ingested but unattached -- invisible in the dataset views and
        indistinguishable from training data (2026-07-28: 74 sims across four
        batches, including a 60-case evaluation-only set). Failures here are
        reported and swallowed: the solves are the expensive artifact and must
        not be lost to a bookkeeping error.

        Known limitation: a batch whose simulations span more than one
        compatibility signature (different domains/grids -- calibration
        batches, not production sweeps) cannot form one collection;
        create_base_dataset raises and the report says why. File such batches
        by hand, split by signature (2026-07-29: bundle_gate_gan_domain /
        bundle_gate_nist_domain).
        """
        from aiida_pino.data.dataset_hierarchy import (
            base_dataset_of,
            create_base_dataset,
            extend_base_dataset,
            iter_base_datasets,
        )

        profile = self.inputs.dataset_profile.get_dict()
        name = base_dataset_name(profile, self.ctx.case_base)

        sims, seen = [], set()
        for pk in ingested_dataset_pks(self.ctx.child_summaries):
            for sim in member_simulations(orm.load_node(pk)):
                if sim.uuid in seen:
                    continue
                seen.add(sim.uuid)
                if base_dataset_of(sim) is None:      # never steal from an existing owner
                    sims.append(sim)
        if not sims:
            self.report("No unattached simulations to file into a base dataset.")
            return None
        target = self.inputs.get("target_dataset", None)
        try:
            from .dataset_filing import dataset_metadata_from_profile, resolve_filing_target

            existing = resolve_filing_target(target_dataset=target, name=name)
            if target is not None or should_extend_existing(existing=existing):
                dataset, created = extend_base_dataset(existing, sims, name=name), False
            else:
                dataset, created = create_base_dataset(
                    sims,
                    name=name,
                    extra_metadata=dataset_metadata_from_profile(
                        profile, batch_workchain_pk=self.node.pk
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a sweep
            self.report(f"Could not file simulations into a base dataset: {exc}")
            return None
        self.report(
            f"Filed {len(sims)} simulations into base dataset "
            f"'{name}' (pk {dataset.pk}, {'created' if created else 'reused'})."
        )
        return dataset
