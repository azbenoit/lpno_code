"""Workflow for registering an existing trained run as a native PINO model node."""

from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain

from aiida_pino.data import PinoModelData
from aiida_pino.utils import (
    canonical_base_layout_profile,
    folder_data_from_remote,
    normalize_dataset_collection_node,
    validate_derivation_profile,
    validate_training_profile,
)
from .train_model import materialize_model_node


class ImportModelWorkChain(WorkChain):
    """Register an existing training-artifact folder or remote run as a model node."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("training_artifacts", valid_type=(orm.FolderData, orm.RemoteData))
        spec.input("training_source", valid_type=orm.Data)
        spec.input(
            "validation_source",
            valid_type=orm.Data,
            required=False,
            help="The val split this run was validated on. Without it 'what was "
            "this validated on' is unanswerable from the node -- these runs had "
            "separate train/ and val/ directories (2026-08-10 friction, item 7).",
        )
        spec.input("training_profile", valid_type=orm.Dict)
        spec.input("derivation_profile", valid_type=orm.Dict, required=False)
        spec.input("model_name", valid_type=orm.Str, required=False)
        spec.outline(cls.validate_inputs, cls.normalize_artifacts, cls.build_model, cls.finalize)
        spec.output("model", valid_type=PinoModelData)
        spec.output("import_summary", valid_type=orm.Dict)
        spec.exit_code(300, "ERROR_INVALID_TRAINING_PROFILE", message="Invalid `training_profile`.")
        spec.exit_code(301, "ERROR_INVALID_DERIVATION_PROFILE", message="Invalid `derivation_profile`.")
        spec.exit_code(302, "ERROR_INVALID_DATASET_INPUT", message="Invalid training-source dataset input type.")

    def validate_inputs(self):
        try:
            self.ctx.training_source = normalize_dataset_collection_node(
                self.inputs.training_source,
                label="training_source",
            )
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_DATASET_INPUT
        try:
            self.ctx.training_profile = validate_training_profile(
                self.inputs.training_profile.get_dict(),
                expected_split_mode="standard",
                # A finished run cannot be asked which GPU to use or whether to
                # log to wandb: those configure a launch that already happened.
                for_import=True,
            )
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_TRAINING_PROFILE
        self.ctx.derivation_defaulted = "derivation_profile" not in self.inputs
        if "derivation_profile" in self.inputs:
            try:
                self.ctx.derivation_profile = validate_derivation_profile(self.inputs.derivation_profile.get_dict())
            except ValueError as exc:
                self.report(str(exc))
                return self.exit_codes.ERROR_INVALID_DERIVATION_PROFILE
        else:
            # Recorded, but flagged: the canonical default says stride [1,1,1],
            # and stamping that as the model's training stride states a
            # measurement nobody made (2026-08-10 friction, item 1).
            self.ctx.derivation_profile = canonical_base_layout_profile()
            self.report(
                "No derivation_profile given: the layout is defaulted, so this model "
                "records NO training stride rather than an unverified [1, 1, 1]. "
                "Pass derivation_profile.stride if the run read a pre-strided copy."
            )

    def normalize_artifacts(self):
        artifacts = self.inputs.training_artifacts
        if isinstance(artifacts, orm.RemoteData):
            self.ctx.training_artifacts = folder_data_from_remote(artifacts)
        else:
            self.ctx.training_artifacts = artifacts

    def build_model(self):
        # Metrics the run already wrote, lifted out of the opaque artifact files
        # so "which seed had the best beam-box error" is a query rather than a
        # tarball dig (2026-08-10 friction, item 6).
        from aiida_pino.utils.profiles import metrics_from_artifacts

        harvested = metrics_from_artifacts(self.ctx.training_artifacts)
        summary = orm.Dict(
            dict={
                "imported": True,
                "source_kind": self.ctx.training_artifacts.__class__.__name__,
                "source_pk": self.inputs.training_artifacts.pk,
                "derivation_profile_defaulted": bool(self.ctx.derivation_defaulted),
                **({"imported_metrics": harvested} if harvested else {}),
            }
        )
        profile = orm.Dict(dict=self.ctx.training_profile)
        model = materialize_model_node(
            self.ctx.training_artifacts,
            summary,
            profile,
            self.ctx.training_source,
            orm.Dict(dict=self.ctx.derivation_profile),
            self.inputs.model_name if "model_name" in self.inputs else None,
            self.inputs.validation_source if "validation_source" in self.inputs else None,
        )
        self.ctx.model = model
        self.ctx.import_summary = orm.Dict(
            dict={
                "model_pk": model.pk,
                "model_uuid": model.uuid,
                "model_name": model.label,
                "training_source_pk": self.ctx.training_source.pk,
                "training_artifacts_pk": self.inputs.training_artifacts.pk,
            }
        ).store()

    def finalize(self):
        self.out("model", self.ctx.model)
        self.out("import_summary", self.ctx.import_summary)
