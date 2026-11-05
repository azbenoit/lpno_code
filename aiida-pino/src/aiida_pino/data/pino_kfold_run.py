"""Structured AiiDA data node for grouped k-fold training/evaluation runs."""

from __future__ import annotations

from aiida import orm

from .pino_dataset import safe_attribute_get


class PinoKfoldRunData(orm.Data):
    """Compact grouped record for a k-fold run and its child fold jobs."""

    FORMAT_VERSION = 1

    @classmethod
    def from_payload(cls, *, metadata: dict) -> "PinoKfoldRunData":
        """Create a k-fold run record from metadata."""
        node = cls()
        node.base.attributes.set("format_version", cls.FORMAT_VERSION)
        node.base.attributes.set("kfold_metadata", dict(metadata))
        node.label = str(metadata.get("run_name") or metadata.get("label") or "kfold_run")
        node.base.extras.set("aiida_pino_kind", "kfold_run")
        if metadata.get("dataset_source_uuid"):
            node.base.extras.set("dataset_source_uuid", metadata["dataset_source_uuid"])
        if metadata.get("dataset_source_kind"):
            node.base.extras.set("dataset_source_kind", metadata["dataset_source_kind"])
        return node

    def get_kfold_metadata(self) -> dict:
        """Return the stored grouped k-fold metadata."""
        return dict(safe_attribute_get(self, "kfold_metadata", {}))
