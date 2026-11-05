"""Shared annotation vocabulary and the extras-based annotation writer.

Simulations carry ``split`` (one of :data:`SPLIT_VALUES`, default
``"unassigned"``) and ``keywords`` (list of strings) in their extras; every
other annotatable node kind carries only ``keywords``. The user-editable
description lives on the native AiiDA ``node.description`` column and, for
simulations, is mirrored into the ``description`` extra so
``get_simulation_metadata`` keeps returning it.
"""

from __future__ import annotations

from typing import Any, Iterable

# Re-exported from a leaf module so the CLI can name the legal values in its
# help without importing AiiDA. Import them from either place.
from aiida_pino.split_values import (  # noqa: F401
    EXCLUDED_SPLIT_VALUES,
    HELD_OUT_SPLIT_VALUES,
    SPLIT_VALUES,
)


def audit_actor() -> str:
    """Best-effort identity for a split-change record: OS user @ host."""
    import getpass
    import socket

    try:
        return f"{getpass.getuser()}@{socket.gethostname()}"
    except Exception:  # noqa: BLE001
        return "unknown"


def append_split_audit(log, *, before, after, reason, actor, method):
    """Append one split-change record to `log` and return it.

    The user's requirement was history, not immutability: "resplitting must
    remain possible, but only with a clear history of how and when it was done".
    38427 is what its absence looks like — a stored recipe claiming a random
    0.8/0.05/0.15 split while the labels are family-inverted, with nothing
    saying who changed them. Append-only by convention: callers add, never
    rewrite. Digests rather than full maps keep the entry small on a 600-member
    dataset while still proving which assignment was in force.
    """
    import datetime as _dt
    import hashlib as _hashlib
    import json as _json

    def _digest(mapping):
        payload = _json.dumps(dict(sorted((mapping or {}).items())), separators=(",", ":"))
        return _hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    before = dict(before or {})
    after = dict(after or {})
    changed = sum(1 for key in set(before) | set(after) if before.get(key) != after.get(key))
    entry = {
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "actor": str(actor),
        "reason": str(reason),
        "method": str(method),
        "changed": int(changed),
        "member_count": len(after) or len(before),
        "before_digest": _digest(before),
        "after_digest": _digest(after),
    }
    log.append(entry)
    return entry

SUGGESTED_KEYWORDS = (
    "keyhole",
    "conduction",
    "random path",
    "variable material",
    "variable path length",
    "straight path",
    "even timesteps",
    "steady",
    "downsampled",
    "smoke",
)

_KIND_BY_CLASS = {
    "PinoSimulationData": "simulation",
    "PinoDatasetData": "dataset",
    "PinoDatasetCollectionData": "dataset_collection",
    "PinoDatasetUnionData": "dataset_union",
    "PinoModelData": "model",
    "PinoModelEvaluationData": "model_evaluation",
    "PinoKfoldRunData": "kfold_run",
}


def normalize_keywords(values: Iterable[Any] | None) -> list[str]:
    """Strip whitespace, drop empties, dedupe preserving first-seen order."""
    out: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def node_kind(node) -> str:
    """Short kind name for an annotatable node (falls back to the class name)."""
    return _KIND_BY_CLASS.get(type(node).__name__, type(node).__name__)


def _is_simulation(node) -> bool:
    from aiida_pino.data import PinoSimulationData

    return isinstance(node, PinoSimulationData)


def update_node_annotations(
    node,
    *,
    title: str | None = None,
    description: str | None = None,
    split: str | None = None,
    split_reason: str | None = None,
    split_method: str = "manual",
    keywords_set: Iterable[str] | None = None,
    keywords_add: Iterable[str] | None = None,
    keywords_remove: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Apply annotation edits to a loaded node and return its annotation state.

    Raises ``ValueError`` for an invalid split value or a split on a
    non-simulation node (the console service maps that to
    ``InvalidRequestError``). With no edit arguments this is a read.

    A split change appends to the node's own ``split_audit``. It is recorded
    here rather than in each caller because this is where every path converges:
    `aiida-pino split` went through it and wrote no history at all, so ten
    changed simulations had no record of their previous values and the only way
    back was a chat transcript (2026-08-19). That defeats the requirement the
    audit exists for -- "resplitting must remain possible, but only with a clear
    history of how and when it was done".
    """
    is_sim = _is_simulation(node)
    if split is not None:
        if not is_sim:
            raise ValueError(f"split only applies to simulations, not {node_kind(node)}")
        if split not in SPLIT_VALUES:
            raise ValueError(f"invalid split {split!r}; expected one of {SPLIT_VALUES}")
        previous = str(node.base.extras.get("split", "") or "unassigned")
        node.base.extras.set("split", split)
        if split != previous:
            uuid = str(node.uuid)
            audit = list(node.base.extras.get("split_audit", None) or [])
            append_split_audit(
                audit,
                before={uuid: previous},
                after={uuid: split},
                reason=str(split_reason or "single-node split change"),
                actor=audit_actor(),
                method=str(split_method),
            )
            node.base.extras.set("split_audit", audit)
    if title is not None:
        # A short human name on the native label column (process nodes keep
        # their class name in process_label, so this stays free for the user).
        node.label = str(title)
    if description is not None:
        node.description = str(description)
        if is_sim:
            node.base.extras.set("description", str(description))
    if keywords_set is not None or keywords_add or keywords_remove:
        current = keywords_set if keywords_set is not None else node.base.extras.get("keywords", [])
        keywords = normalize_keywords(current)
        for keyword in normalize_keywords(keywords_add):
            if keyword not in keywords:
                keywords.append(keyword)
        removals = set(normalize_keywords(keywords_remove))
        keywords = [k for k in keywords if k not in removals]
        node.base.extras.set("keywords", keywords)
    result: dict[str, Any] = {
        "pk": node.pk,
        "kind": node_kind(node),
        "title": node.label or None,
        "label": node.label or str(node.base.extras.get("simulation_identifier", "") or "") or node.uuid,
        "description": node.description,
        "keywords": normalize_keywords(node.base.extras.get("keywords", [])),
    }
    if is_sim:
        result["split"] = str(node.base.extras.get("split", "") or "unassigned")
    return result
