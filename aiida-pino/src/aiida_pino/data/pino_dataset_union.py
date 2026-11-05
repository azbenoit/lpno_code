"""Metadata-only AiiDA node for reusable unions of compatible PINO datasets."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

from aiida import orm

from .pino_dataset import PinoDatasetData, safe_attribute_get


def _sanitize_union_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "dataset_union"


def build_union_identifier(member_uuids: Iterable[str], *, union_name: str = "") -> str:
    """Build a stable identifier for a dataset union."""
    payload = {
        "member_uuids": list(member_uuids),
        "union_name": union_name,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    name = _sanitize_union_name(union_name) if union_name else "dataset_union"
    return f"{name}__{digest}"


class PinoDatasetUnionData(orm.Data):
    """Metadata-only reusable union of ordered native dataset members."""

    FORMAT_VERSION = 1

    @classmethod
    def from_members(
        cls,
        members: list[PinoDatasetData],
        *,
        union_name: str,
        compatibility_signature: str,
        total_samples: int,
    ) -> "PinoDatasetUnionData":
        """Create a dataset-union node from a validated member list."""
        if not members:
            raise ValueError("Dataset unions require at least one member dataset.")
        member_uuids = [member.uuid for member in members]
        member_pks = [member.pk for member in members]
        union_identifier = build_union_identifier(member_uuids, union_name=union_name)

        node = cls()
        node.base.attributes.set("format_version", cls.FORMAT_VERSION)
        node.base.attributes.set("union_name", union_name)
        node.base.attributes.set("union_identifier", union_identifier)
        node.base.attributes.set("member_dataset_uuids", member_uuids)
        node.base.attributes.set("member_dataset_pks", member_pks)
        node.base.attributes.set("compatibility_signature", compatibility_signature)
        node.base.attributes.set("total_samples", int(total_samples))
        node.label = f"{union_name} | {len(members)} dataset(s)"
        node.base.extras.set("aiida_pino_kind", "dataset_union")
        node.base.extras.set("union_identifier", union_identifier)
        node.base.extras.set("member_dataset_uuids", member_uuids)
        node.base.extras.set("compatibility_signature", compatibility_signature)
        return node

    def get_union_name(self) -> str:
        """Return the user-facing union name."""
        return str(safe_attribute_get(self, "union_name", self.label or "dataset_union"))

    def get_union_identifier(self) -> str:
        """Return the stable union identifier."""
        return str(safe_attribute_get(self, "union_identifier", ""))

    def get_member_dataset_uuids(self) -> list[str]:
        """Return member dataset UUIDs in staging order."""
        return list(safe_attribute_get(self, "member_dataset_uuids", []))

    def get_member_dataset_pks(self) -> list[int]:
        """Return member dataset PKs in staging order."""
        return list(safe_attribute_get(self, "member_dataset_pks", []))

    def iter_member_datasets(self) -> list[PinoDatasetData]:
        """Load and return the member datasets."""
        members: list[PinoDatasetData] = []
        for identifier in self.get_member_dataset_uuids():
            node = orm.load_node(identifier)
            if not isinstance(node, PinoDatasetData):
                raise ValueError(f"Union member `{identifier}` is not a native PINO dataset.")
            members.append(node)
        return members

    def get_compatibility_signature(self) -> str:
        """Return the compatibility signature shared by all members."""
        return str(safe_attribute_get(self, "compatibility_signature", ""))

    def get_total_samples(self) -> int:
        """Return the aggregated member sample count."""
        return int(safe_attribute_get(self, "total_samples", 0))
