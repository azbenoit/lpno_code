"""Workflow for grouped native PINO k-fold training/evaluation."""

from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction, while_

from aiida_pino.calculations.eval import PinoEvalCalculation
from aiida_pino.calculations.train import PinoTrainCalculation
from aiida_pino.data import PinoDatasetCollectionData, PinoKfoldRunData
from aiida_pino.utils import (
    aggregate_metric_payloads,
    build_kfold_run_metadata,
    build_derivation_profile_from_training_profile,
    with_requested_derived_fields,
    canonical_base_layout_profile,
    dataset_layout_profile_from_metadata,
    find_reusable_derived_dataset,
    normalize_dataset_collection_node,
    resolve_base_dataset,
    validate_derivation_profile,
    validate_training_profile,
)
from aiida_pino.workflows.derive_dataset import materialize_derived_dataset
from aiida_pino.workflows.labels import InheritsCallerLabel


@calcfunction
def materialize_kfold_run_node(
    dataset_source,
    training_profile: orm.Dict,
    derivation_profile: orm.Dict,
    metrics: orm.Dict,
    kfold_summary: orm.Dict,
    fold_payload: orm.Dict,
) -> PinoKfoldRunData:
    """Create a grouped native record for a k-fold run."""
    metadata = build_kfold_run_metadata(
        dataset_source,
        training_profile.get_dict(),
        derivation_profile.get_dict(),
        metrics.get_dict(),
        kfold_summary.get_dict(),
        fold_payload.get_dict(),
    )
    return PinoKfoldRunData.from_payload(metadata=metadata)


class KfoldTrainEvalWorkChain(InheritsCallerLabel, WorkChain):
    """Run grouped PINO k-fold training/evaluation over native dataset sources."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("dataset", valid_type=orm.Data)
        spec.input("training_profile", valid_type=orm.Dict)
        spec.input("derivation_profile", valid_type=orm.Dict, required=False)
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.outline(
            cls.validate_inputs,
            cls.prepare_dataset,
            cls.initialize_folds,
            while_(cls.should_run_next_fold)(
                cls.run_fold_training,
                cls.run_fold_evaluation,
                cls.collect_fold_outputs,
            ),
            cls.build_outputs,
            cls.finalize,
        )
        spec.output("kfold_run", valid_type=PinoKfoldRunData)
        spec.output("metrics", valid_type=orm.Dict)
        spec.output("kfold_summary", valid_type=orm.Dict)
        spec.exit_code(300, "ERROR_INVALID_TRAINING_PROFILE", message="Invalid `training_profile`.")
        spec.exit_code(301, "ERROR_INVALID_DERIVATION_PROFILE", message="Invalid `derivation_profile`.")
        spec.exit_code(302, "ERROR_DERIVATION_FAILED", message="Dataset derivation failed.")
        spec.exit_code(303, "ERROR_FOLD_TRAINING_FAILED", message="A fold training job failed.")
        spec.exit_code(304, "ERROR_FOLD_EVALUATION_FAILED", message="A fold evaluation job failed.")
        spec.exit_code(305, "ERROR_INVALID_DATASET_INPUT", message="Invalid dataset input type.")
        spec.exit_code(
            306,
            "ERROR_EVALUATION_ONLY_DATASET",
            message="Dataset is reserved for evaluation and cannot be trained on.",
        )

    def validate_inputs(self):
        try:
            self.ctx.dataset = normalize_dataset_collection_node(self.inputs.dataset, label="dataset")
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_DATASET_INPUT
        # kfold had NO split guard, so the only gate was the console's
        # evaluation-only predicate -- which fired on `== {"test"}` and so missed
        # `eval`. `submit_kfold(48923)` therefore trained k folds on the
        # galiegue fusion-ladder validation set, with no exit code and no report
        # line. The guard belongs here, where every kfold submission passes.
        from aiida_pino.console.service import is_evaluation_only_split_set

        member_splits = {
            str(sim.base.extras.get("split", "") or "unassigned")
            for sim in self.ctx.dataset.iter_member_simulations()
        }
        if is_evaluation_only_split_set(member_splits):
            self.report(
                f"REFUSED: dataset {self.ctx.dataset.pk} is reserved for evaluation "
                f"(member splits {sorted(member_splits)}); k-fold would train on it."
            )
            return self.exit_codes.ERROR_EVALUATION_ONLY_DATASET
        try:
            normalized = validate_training_profile(
                self.inputs.training_profile.get_dict(),
                expected_split_mode="kfold",
            )
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_TRAINING_PROFILE
        requested_fold = int(normalized["fold_index"])
        k_folds = int(normalized["k_folds"])
        if requested_fold != -1 and not 0 <= requested_fold < k_folds:
            return self.exit_codes.ERROR_INVALID_TRAINING_PROFILE
        self.ctx.training_profile = normalized
        if "derivation_profile" in self.inputs:
            try:
                self.ctx.derivation_profile = validate_derivation_profile(self.inputs.derivation_profile.get_dict())
            except ValueError as exc:
                self.report(str(exc))
                return self.exit_codes.ERROR_INVALID_DERIVATION_PROFILE
        else:
            self.ctx.derivation_profile = build_derivation_profile_from_training_profile(self.ctx.training_profile)
        # Both branches: an explicit profile skips the builder's promotion, and a training
        # that conditions on a buildable channel must read it baked rather than rebuilding
        # it per case into the shared, unevictable `~/.pino/conduction_cache`.
        self.ctx.derivation_profile = with_requested_derived_fields(
            self.ctx.derivation_profile, self.ctx.training_profile
        )

    def _resolve_source(self, source: PinoDatasetCollectionData):
        profile = self.ctx.derivation_profile
        metadata = source.get_dataset_metadata()
        if dataset_layout_profile_from_metadata(metadata) == profile:
            return source
        base = resolve_base_dataset(source)
        # Only serve the base directly when the base's OWN stamped layout is what
        # was asked for. Testing `profile == canonical` instead assumed every base
        # is stamped canonical, which silently served a time_as_dim base for a
        # time_as_batch request once the canonical default moved.
        if dataset_layout_profile_from_metadata(base.get_dataset_metadata()) == profile:
            return base
        existing = find_reusable_derived_dataset(base, profile)
        if existing is not None:
            return existing
        return materialize_derived_dataset(base, orm.Dict(dict=profile))

    def prepare_dataset(self):
        self.ctx.dataset_source = self._resolve_source(self.ctx.dataset)

    def initialize_folds(self):
        requested_fold = int(self.ctx.training_profile["fold_index"])
        k_folds = int(self.ctx.training_profile["k_folds"])
        self.ctx.fold_indices = [requested_fold] if requested_fold != -1 else list(range(k_folds))
        self.ctx.fold_cursor = 0
        self.ctx.fold_results = []

    def should_run_next_fold(self):
        return self.ctx.fold_cursor < len(self.ctx.fold_indices)

    def _current_fold(self) -> int:
        return int(self.ctx.fold_indices[self.ctx.fold_cursor])

    def _fold_profile(self, fold_index: int) -> dict:
        profile = dict(self.ctx.training_profile)
        profile.update(dict(self.ctx.derivation_profile))
        profile["split_mode"] = "kfold"
        profile["fold_index"] = int(fold_index)
        profile["run_name"] = f"{self.ctx.training_profile['run_name']}_fold{fold_index}"
        return profile

    def run_fold_training(self):
        fold_index = self._current_fold()
        profile = self._fold_profile(fold_index)
        self.ctx.current_fold_profile = profile
        future = self.submit(
            PinoTrainCalculation,
            code=self.inputs.python_code,
            train_dataset=self.ctx.dataset_source,
            val_dataset=self.ctx.dataset_source,
            training_profile=orm.Dict(dict=profile),
        )
        self.report(f"Submitted fold {fold_index} training job: {future.pk}")
        return ToContext(current_train_job=future)

    def run_fold_evaluation(self):
        fold_index = self._current_fold()
        if not self.ctx.current_train_job.is_finished_ok:
            self.report(f"Fold {fold_index} training failed with exit status {self.ctx.current_train_job.exit_status}.")
            return self.exit_codes.ERROR_FOLD_TRAINING_FAILED
        future = self.submit(
            PinoEvalCalculation,
            code=self.inputs.python_code,
            dataset=self.ctx.dataset_source,
            training_artifacts=self.ctx.current_train_job.outputs.training_artifacts,
            training_profile=orm.Dict(dict=self.ctx.current_fold_profile),
        )
        self.report(f"Submitted fold {fold_index} evaluation job: {future.pk}")
        return ToContext(current_eval_job=future)

    def collect_fold_outputs(self):
        fold_index = self._current_fold()
        if not self.ctx.current_eval_job.is_finished_ok:
            self.report(f"Fold {fold_index} evaluation failed with exit status {self.ctx.current_eval_job.exit_status}.")
            return self.exit_codes.ERROR_FOLD_EVALUATION_FAILED

        self.ctx.fold_results.append(
            {
                "fold_index": fold_index,
                "run_name": self.ctx.current_fold_profile["run_name"],
                "train_process_pk": self.ctx.current_train_job.pk,
                "eval_process_pk": self.ctx.current_eval_job.pk,
                "training_artifacts_pk": self.ctx.current_train_job.outputs.training_artifacts.pk,
                "train_summary": self.ctx.current_train_job.outputs.train_summary.get_dict(),
                "eval_summary": self.ctx.current_eval_job.outputs.eval_summary.get_dict(),
                "metrics": self.ctx.current_eval_job.outputs.metrics.get_dict(),
            }
        )
        self.ctx.fold_cursor += 1

    def build_outputs(self):
        folds = list(self.ctx.fold_results)
        aggregated_metrics = aggregate_metric_payloads([fold["metrics"] for fold in folds])
        fold_metrics = {f"fold_{fold['fold_index']}": fold["metrics"] for fold in folds}
        metrics_payload = {
            "aggregated": aggregated_metrics,
            "folds": fold_metrics,
        }
        summary_payload = {
            "run_name": self.ctx.training_profile["run_name"],
            "config_name": self.ctx.training_profile["config_name"],
            "dataset_source_kind": "dataset_collection",
            "dataset_source_pk": self.ctx.dataset_source.pk,
            "dataset_source_uuid": self.ctx.dataset_source.uuid,
            "dataset_source_label": getattr(self.ctx.dataset_source, "label", "")
            or self.ctx.dataset_source.get_dataset_metadata().get("dataset_identifier", ""),
            "k_folds": int(self.ctx.training_profile["k_folds"]),
            "requested_fold_index": int(self.ctx.training_profile["fold_index"]),
            "fold_indices": [fold["fold_index"] for fold in folds],
            "fold_count": len(folds),
            "aggregated_errors_present": bool(aggregated_metrics),
            "derivation_profile": dict(self.ctx.derivation_profile),
        }
        self.ctx.metrics = orm.Dict(dict=metrics_payload)
        self.ctx.kfold_summary = orm.Dict(dict=summary_payload)
        self.ctx.kfold_run = materialize_kfold_run_node(
            self.ctx.dataset_source,
            orm.Dict(dict=self.ctx.training_profile),
            orm.Dict(dict=self.ctx.derivation_profile),
            self.ctx.metrics,
            self.ctx.kfold_summary,
            orm.Dict(dict={"folds": folds}),
        )

    def finalize(self):
        self.out("kfold_run", self.ctx.kfold_run)
        self.out("metrics", self.ctx.metrics)
        self.out("kfold_summary", self.ctx.kfold_summary)
