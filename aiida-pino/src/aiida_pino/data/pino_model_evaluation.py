"""Structured AiiDA data node for model evaluation records."""

from __future__ import annotations

from aiida import orm

from .pino_dataset import safe_attribute_get


class PinoModelEvaluationData(orm.Data):
    """Immutable evaluation record linked to a model and dataset source."""

    FORMAT_VERSION = 1

    @classmethod
    def from_payload(cls, *, metadata: dict) -> "PinoModelEvaluationData":
        """Create an evaluation record from metadata and metrics payload."""
        node = cls()
        node.base.attributes.set("format_version", cls.FORMAT_VERSION)
        node.base.attributes.set("evaluation_metadata", dict(metadata))
        node.label = str(metadata.get("evaluation_name") or metadata.get("model_name") or "evaluation")
        node.base.extras.set("aiida_pino_kind", "model_evaluation")
        if metadata.get("model_uuid"):
            node.base.extras.set("model_uuid", metadata["model_uuid"])
        if metadata.get("dataset_source_uuid"):
            node.base.extras.set("dataset_source_uuid", metadata["dataset_source_uuid"])
        # The model's training set, mirrored here: querying training_source_uuid
        # on an evaluation used to return None that looked like a real answer
        # (2026-08-10).
        if metadata.get("training_source_uuid"):
            node.base.extras.set("training_source_uuid", metadata["training_source_uuid"])
        # The grid this eval scored, as ONE queryable field. It used to be
        # reconstructable only by reading `env.PINO_LAZY_STRIDE` together with
        # the dataset's `derivation.stride`, per eval, by hand — "has any
        # checkpoint been evaluated at two resolutions?" could not be expressed
        # in QueryBuilder at all (2026-08-09).
        stride = metadata.get("eval_stride")
        if isinstance(stride, (list, tuple)) and len(stride) == 3:
            node.base.extras.set("eval_stride", [int(v) for v in stride])
            node.base.extras.set("eval_full_resolution", list(stride) == [1, 1, 1])
        # The stride the model TRAINED at, recorded beside the one it was scored at, and a
        # flag for the comparison. Scoring a model at a resolution it did not train on exits
        # 0 and returns numbers that look real: an FNO trained at [2,2,1] and scored with the
        # stride blanked read substrate 43-55 K / beam box 145-167 K against a true 11.90 /
        # 45.17, a 4x fake regression that also INVERTED the ranking of the two arms being
        # compared (2026-09-06). It is architecture-dependent -- RIGNO and
        # GAOT are graph models and absorb the change -- so in a mixed batch two of three
        # architectures reproduce and the third looks broken, which is what hides it.
        #
        # Recorded, not refused: both numbers were already on their own nodes and nothing
        # compared them, so this makes the mismatch queryable without changing what runs.
        # Refusing such an evaluation for grid-bound (spectral) architectures would be the
        # stricter option; it is not implemented.
        # What the run actually served, queryable. `derivation_profile.crop_z_top` answers a
        # different question (what was REQUESTED of the dataset) and read 0 on three evals
        # that served a 30-deep domain, so an audit of crop from provenance returned a clean
        # uncropped set with nothing to indicate otherwise (2026-09-09).
        served = metadata.get("served_grid")
        if isinstance(served, dict) and served:
            node.base.extras.set("served_grid", served)
        crop = metadata.get("eval_crop_z_top")
        if isinstance(crop, int):
            node.base.extras.set("eval_crop_z_top", crop)
        trained = metadata.get("training_stride")
        if isinstance(trained, (list, tuple)) and len(trained) == 3:
            node.base.extras.set("training_stride", [int(v) for v in trained])
            if isinstance(stride, (list, tuple)) and len(stride) == 3:
                node.base.extras.set(
                    "eval_resolution_matches_training",
                    [int(v) for v in trained] == [int(v) for v in stride],
                )
        # Whether this score is held out, as one filterable field. `held_out`
        # stays unset when the overlap is unknowable, so "no overlap" and "no
        # evidence" never read the same -- the same unknown-stays-unknown rule
        # as eval_stride above.
        overlap = metadata.get("train_overlap_n")
        if isinstance(overlap, int):
            node.base.extras.set("train_overlap_n", overlap)
            node.base.extras.set("held_out", overlap == 0)
        return node

    def get_evaluation_metadata(self) -> dict:
        """Return evaluation metadata."""
        return dict(safe_attribute_get(self, "evaluation_metadata", {}))
