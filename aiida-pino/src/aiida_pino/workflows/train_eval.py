"""Compatibility wrapper chaining first-class PINO training and evaluation workflows."""

from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_pino.data import PinoModelData, PinoModelEvaluationData
from aiida_pino.utils import normalize_dataset_collection_node
from aiida_pino.workflows.evaluate_model import EvaluateModelWorkChain
from aiida_pino.workflows.train_model import TrainModelWorkChain
from aiida_pino.workflows.labels import InheritsCallerLabel


class TrainEvalWorkChain(InheritsCallerLabel, WorkChain):
    """Train a model and immediately evaluate it on the validation source."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("train_dataset", valid_type=orm.Data)
        spec.input("val_dataset", valid_type=orm.Data)
        spec.input(
            "evaluate_on_test",
            valid_type=orm.Bool,
            required=False,
            help="Evaluate on eval_dataset (the test split) instead of the default "
                 "val_dataset. Off by default: the automatic post-training eval is a "
                 "health readout, and test-split scoring should be an explicit act.",
        )
        spec.input(
            "eval_dataset",
            valid_type=orm.Data,
            required=False,
            help="Dataset to evaluate the trained model on. Defaults to val_dataset when absent; "
            "the console passes the dataset's test split here so eval scores held-out data.",
        )
        spec.input("training_profile", valid_type=orm.Dict)
        spec.input("derivation_profile", valid_type=orm.Dict, required=False)
        spec.input("evaluation_profile", valid_type=orm.Dict, required=False)
        spec.input(
            "init_model",
            # Same widening as TrainModelWorkChain (2026-08-09): the calcjob port
            # always took all three, and this wrapper was the remaining bottleneck
            # forcing the import-model round trip.
            valid_type=(PinoModelData, orm.FolderData, orm.RemoteData),
            required=False,
            help="Warm start: a model node, or a FolderData/RemoteData holding "
            "checkpoint files, seeds the trainer's network_dir.",
        )
        spec.input(
            "allow_training_data",
            valid_type=orm.Bool,
            required=False,
            help="Forwarded to the post-training evaluation: explicitly permit "
            "scoring cases the model was trained on. Needed when val_dataset "
            "overlaps the training set (the record is stamped held_out=False).",
        )
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.input(
            "allow_split_violation",
            valid_type=orm.Bool,
            required=False,
            help="Forwarded to TrainModelWorkChain: permit test/excluded members "
            "in the training set or train/val overlap (refused by default).",
        )
        spec.outline(cls.run_training, cls.run_evaluation, cls.check_evaluation, cls.finalize)
        spec.output("model", valid_type=PinoModelData)
        spec.output("evaluation", valid_type=PinoModelEvaluationData)
        spec.output("training_artifacts", valid_type=orm.FolderData)
        spec.output("train_summary", valid_type=orm.Dict)
        spec.output("eval_artifacts", valid_type=orm.FolderData)
        spec.output("metrics", valid_type=orm.Dict)
        spec.output("eval_summary", valid_type=orm.Dict)
        spec.exit_code(300, "ERROR_TRAIN_MODEL_FAILED", message="The train-model step failed.")
        spec.exit_code(301, "ERROR_EVALUATE_MODEL_FAILED", message="The evaluate-model step failed.")
        spec.exit_code(302, "ERROR_INVALID_DATASET_INPUT", message="Invalid dataset input type.")

    def run_training(self):
        try:
            train_dataset = normalize_dataset_collection_node(self.inputs.train_dataset, label="train_dataset")
            val_dataset = normalize_dataset_collection_node(self.inputs.val_dataset, label="val_dataset")
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_DATASET_INPUT
        inputs = {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "training_profile": self.inputs.training_profile,
            "python_code": self.inputs.python_code,
        }
        if "derivation_profile" in self.inputs:
            inputs["derivation_profile"] = self.inputs.derivation_profile
        if "init_model" in self.inputs:
            inputs["init_model"] = self.inputs.init_model
        if "allow_split_violation" in self.inputs:
            inputs["allow_split_violation"] = self.inputs.allow_split_violation
        future = self.submit(TrainModelWorkChain, **inputs)
        self.report(f"Submitted train-model workflow: {future.pk}")
        return ToContext(train_model=future)

    def run_evaluation(self):
        if not self.ctx.train_model.is_finished_ok:
            return self.exit_codes.ERROR_TRAIN_MODEL_FAILED
        # Evaluate on the held-out test set when one is given (the console passes
        # the dataset's test split); fall back to val_dataset otherwise.
        #
        # DEFAULT FLIPPED to val_dataset (2026-08-06): the automatic
        # post-training eval exists as a health readout, not the paper number, and on
        # full-resolution sets the test split costs hours (varspeed: 22 cases x ~400
        # frames ~ 2.9 h measured) while the val split (8 cases) is proportionally
        # smaller. A held-out TEST evaluation is an explicit act: submit an
        # EvalOnlyWorkChain against the test split, or pass eval_dataset with
        # evaluate_on_test=True.
        use_test = bool(self.inputs.evaluate_on_test) if "evaluate_on_test" in self.inputs else False
        if "eval_dataset" in self.inputs and use_test:
            eval_input = self.inputs.eval_dataset
        else:
            if "eval_dataset" in self.inputs and not use_test:
                self.report(
                    "eval_dataset was provided but evaluate_on_test is not set; "
                    "evaluating on val_dataset (the default health readout). Submit an "
                    "EvalOnlyWorkChain for a test-split score."
                )
            eval_input = self.inputs.val_dataset
        try:
            eval_dataset = normalize_dataset_collection_node(eval_input, label="eval_dataset")
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_DATASET_INPUT
        inputs = {
            "dataset": eval_dataset,
            "model": self.ctx.train_model.outputs.model,
            "python_code": self.inputs.python_code,
        }
        if "derivation_profile" in self.inputs:
            inputs["derivation_profile"] = self.inputs.derivation_profile
        if "evaluation_profile" in self.inputs:
            inputs["evaluation_profile"] = self.inputs.evaluation_profile
        if "allow_training_data" in self.inputs:
            inputs["allow_training_data"] = self.inputs.allow_training_data
        future = self.submit(EvaluateModelWorkChain, **inputs)
        self.report(f"Submitted evaluate-model workflow: {future.pk}")
        return ToContext(evaluate_model=future)

    def check_evaluation(self):
        if not self.ctx.evaluate_model.is_finished_ok:
            return self.exit_codes.ERROR_EVALUATE_MODEL_FAILED

    def finalize(self):
        self.out("model", self.ctx.train_model.outputs.model)
        self.out("training_artifacts", self.ctx.train_model.outputs.training_artifacts)
        self.out("train_summary", self.ctx.train_model.outputs.train_summary)
        self.out("evaluation", self.ctx.evaluate_model.outputs.evaluation)
        self.out("eval_artifacts", self.ctx.evaluate_model.outputs.eval_artifacts)
        self.out("metrics", self.ctx.evaluate_model.outputs.metrics)
        self.out("eval_summary", self.ctx.evaluate_model.outputs.eval_summary)
