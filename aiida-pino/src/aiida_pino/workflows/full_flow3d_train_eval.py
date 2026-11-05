"""Top-level workflow chaining Flow3D ingest to train/eval."""

from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_pino.data import PinoDatasetCollectionData, PinoModelData, PinoModelEvaluationData
from aiida_pino.workflows.flow3d_ingest import Flow3DIngestWorkChain
from aiida_pino.workflows.train_eval import TrainEvalWorkChain
from aiida_pino.workflows.labels import InheritsCallerLabel


class FullFlow3DTrainEvalWorkChain(InheritsCallerLabel, WorkChain):
    """Run the full V1 Flow3D to standard PINO train/eval pipeline."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("source_path", valid_type=orm.Str)
        spec.input("conversion_profile", valid_type=orm.Dict)
        spec.input("dataset_profile", valid_type=orm.Dict)
        spec.input("training_profile", valid_type=orm.Dict)
        spec.input("derivation_profile", valid_type=orm.Dict, required=False)
        spec.input("pvpython_code", valid_type=orm.AbstractCode)
        spec.input("python_code", valid_type=orm.AbstractCode)
        spec.outline(cls.run_ingest, cls.run_train_eval, cls.finalize)
        spec.output("dataset", valid_type=PinoDatasetCollectionData)
        spec.output("model", valid_type=PinoModelData)
        spec.output("evaluation", valid_type=PinoModelEvaluationData)
        spec.output("training_artifacts", valid_type=orm.FolderData)
        spec.output("train_summary", valid_type=orm.Dict)
        spec.output("eval_artifacts", valid_type=orm.FolderData)
        spec.output("metrics", valid_type=orm.Dict)
        spec.output("eval_summary", valid_type=orm.Dict)

    def run_ingest(self):
        future = self.submit(
            Flow3DIngestWorkChain,
            source_path=self.inputs.source_path,
            conversion_profile=self.inputs.conversion_profile,
            dataset_profile=self.inputs.dataset_profile,
            pvpython_code=self.inputs.pvpython_code,
            python_code=self.inputs.python_code,
        )
        self.report(f"Submitted Flow3D ingest workflow: {future.pk}")
        return ToContext(ingest=future)

    def run_train_eval(self):
        future = self.submit(
            TrainEvalWorkChain,
            train_dataset=self.ctx.ingest.outputs.dataset,
            val_dataset=self.ctx.ingest.outputs.dataset,
            training_profile=self.inputs.training_profile,
            derivation_profile=getattr(self.inputs, "derivation_profile", None),
            python_code=self.inputs.python_code,
        )
        self.report(f"Submitted train/eval workflow: {future.pk}")
        return ToContext(train_eval=future)

    def finalize(self):
        self.out("dataset", self.ctx.ingest.outputs.dataset)
        self.out("model", self.ctx.train_eval.outputs.model)
        self.out("evaluation", self.ctx.train_eval.outputs.evaluation)
        self.out("training_artifacts", self.ctx.train_eval.outputs.training_artifacts)
        self.out("train_summary", self.ctx.train_eval.outputs.train_summary)
        self.out("eval_artifacts", self.ctx.train_eval.outputs.eval_artifacts)
        self.out("metrics", self.ctx.train_eval.outputs.metrics)
        self.out("eval_summary", self.ctx.train_eval.outputs.eval_summary)
