"""Workflow for explicit derivation of dataset variants from a canonical base dataset."""

from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, workfunction

from aiida_pino.data import PinoDatasetCollectionData
from aiida_pino.utils import (
    build_derived_collection_metadata,
    build_derivation_summary,
    canonical_base_layout_profile,
    find_reusable_derived_dataset,
    get_or_create_dataset_collection,
    resolve_base_dataset,
    validate_derivation_profile,
)


@workfunction
def materialize_derived_dataset(dataset: PinoDatasetCollectionData, derivation_profile: orm.Dict) -> PinoDatasetCollectionData:
    """Create a stored derived dataset collection from a canonical base collection."""
    base_dataset = resolve_base_dataset(dataset)
    if not isinstance(base_dataset, PinoDatasetCollectionData):
        raise ValueError("Expected a collection base dataset.")
    metadata = build_derived_collection_metadata(base_dataset, derivation_profile.get_dict(), parent_collection=dataset)
    collection, _ = get_or_create_dataset_collection(base_dataset.iter_member_simulations(), metadata=metadata)
    return collection


class DeriveDatasetWorkChain(WorkChain):
    """Resolve or create a derived dataset variant from a canonical base dataset."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("dataset", valid_type=PinoDatasetCollectionData)
        spec.input("derivation_profile", valid_type=orm.Dict)
        spec.outline(
            cls.validate_inputs,
            cls.resolve_or_create_dataset,
            cls.finalize,
        )
        spec.output("dataset", valid_type=PinoDatasetCollectionData)
        spec.output("dataset_summary", valid_type=orm.Dict)
        spec.exit_code(300, "ERROR_INVALID_DERIVATION_PROFILE", message="Invalid `derivation_profile`.")
        spec.exit_code(301, "ERROR_NONCANONICAL_SOURCE_DATASET", message="Selected dataset cannot be used as a derivation base.")

    def validate_inputs(self):
        try:
            self.ctx.derivation_profile = validate_derivation_profile(self.inputs.derivation_profile.get_dict())
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_DERIVATION_PROFILE

        try:
            self.ctx.base_dataset = resolve_base_dataset(self.inputs.dataset)
        except ValueError as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_NONCANONICAL_SOURCE_DATASET

    def resolve_or_create_dataset(self):
        if self.ctx.derivation_profile == canonical_base_layout_profile():
            self.ctx.dataset = self.ctx.base_dataset
            self.ctx.reused = True
            return

        existing = find_reusable_derived_dataset(self.ctx.base_dataset, self.ctx.derivation_profile)
        if existing is not None:
            self.ctx.dataset = existing
            self.ctx.reused = True
            self.report(f"Reused derived dataset pk={existing.pk}")
            return

        self.ctx.dataset = materialize_derived_dataset(
            self.ctx.base_dataset,
            orm.Dict(dict=self.ctx.derivation_profile),
        )
        self.ctx.reused = False
        self.report(f"Created derived dataset pk={self.ctx.dataset.pk}")

    def finalize(self):
        dataset = self.ctx.dataset
        summary = orm.Dict(
            dict=build_derivation_summary(
                self.ctx.base_dataset,
                dataset,
                self.ctx.derivation_profile,
                reused=bool(self.ctx.reused),
            )
        ).store()
        self.out("dataset", dataset)
        self.out("dataset_summary", summary)
