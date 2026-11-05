"""Compatibility wrapper around the first-class model-evaluation workflow."""

from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_pino.data import PinoModelData, PinoModelEvaluationData
from aiida_pino.utils import normalize_dataset_collection_node
from aiida_pino.workflows.evaluate_model import EvaluateModelWorkChain
from aiida_pino.workflows.labels import InheritsCallerLabel


class EvalOnlyWorkChain(InheritsCallerLabel, WorkChain):
    """Evaluate a native model against a dataset or dataset union."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("dataset", valid_type=orm.Data)
        spec.input("model", valid_type=PinoModelData)
        spec.input("evaluation_profile", valid_type=orm.Dict, required=False)
        spec.input("derivation_profile", valid_type=orm.Dict, required=False)
        spec.input(
            "allow_training_data",
            valid_type=orm.Bool,
            required=False,
            help="Forwarded to EvaluateModelWorkChain: explicitly permit scoring "
            "cases the model was trained on (the record is stamped held_out=False).",
        )
        spec.input(
            "allow_domain_change",
            valid_type=orm.Bool,
            required=False,
            help="Forwarded to EvaluateModelWorkChain: explicitly permit scoring the model "
            "on a different DOMAIN EXTENT than it trained on. Declared here because it was "
            "not, so the remedy the 306 refusal names had no route through this workchain -- "
            "and a dataset whose derivation BAKES a crop records `training_crop_z_top=0`, "
            "which is true of the read-time request and misleading about the extent, so the "
            "guard refuses evals whose geometry actually matches (a crop-baked collection at "
            "[90,40,30] against its uncropped parent plus crop 10, also [90,40,30]).",
        )
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.outline(cls.run_evaluation, cls.check_evaluation, cls.finalize)
        spec.output("evaluation", valid_type=PinoModelEvaluationData)
        spec.output("eval_artifacts", valid_type=orm.FolderData)
        spec.output("metrics", valid_type=orm.Dict)
        spec.output("eval_summary", valid_type=orm.Dict)
        spec.exit_code(300, "ERROR_EVALUATE_MODEL_FAILED", message="The evaluate-model step failed.")
        spec.exit_code(301, "ERROR_INVALID_DATASET_INPUT", message="Invalid dataset input type.")

    def run_evaluation(self):
        try:
            dataset = normalize_dataset_collection_node(self.inputs.dataset, label="dataset")
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_DATASET_INPUT
        inputs = {
            "dataset": dataset,
            "model": self.inputs.model,
            "python_code": self.inputs.python_code,
        }
        if "evaluation_profile" in self.inputs:
            inputs["evaluation_profile"] = self.inputs.evaluation_profile
        if "derivation_profile" in self.inputs:
            inputs["derivation_profile"] = self.inputs.derivation_profile
        if "allow_training_data" in self.inputs:
            inputs["allow_training_data"] = self.inputs.allow_training_data
        if "allow_domain_change" in self.inputs:
            inputs["allow_domain_change"] = self.inputs.allow_domain_change
        future = self.submit(EvaluateModelWorkChain, **inputs)
        self.report(f"Submitted evaluate-model workflow: {future.pk}")
        return ToContext(evaluate_model=future)

    def check_evaluation(self):
        if not self.ctx.evaluate_model.is_finished_ok:
            return self.exit_codes.ERROR_EVALUATE_MODEL_FAILED

    def finalize(self):
        self.out("evaluation", self.ctx.evaluate_model.outputs.evaluation)
        self.out("eval_artifacts", self.ctx.evaluate_model.outputs.eval_artifacts)
        self.out("metrics", self.ctx.evaluate_model.outputs.metrics)
        self.out("eval_summary", self.ctx.evaluate_model.outputs.eval_summary)
