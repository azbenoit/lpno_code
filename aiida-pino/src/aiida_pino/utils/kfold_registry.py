"""Helpers for grouped native k-fold run records and metric aggregation."""

from __future__ import annotations

from datetime import datetime
from numbers import Number
from typing import Any

from aiida import orm

from aiida_pino.data import PinoDatasetCollectionData, PinoKfoldRunData
from aiida_pino.utils.model_registry import build_training_source_summary


def _average_numeric(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, Number)]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def aggregate_metric_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively average nested numeric metric payloads."""
    if not payloads:
        return {}

    aggregated: dict[str, Any] = {}
    keys = sorted({key for payload in payloads for key in payload})
    for key in keys:
        values = [payload[key] for payload in payloads if key in payload]
        if values and all(isinstance(value, dict) for value in values):
            nested = aggregate_metric_payloads([value for value in values if isinstance(value, dict)])
            if nested:
                aggregated[key] = nested
            continue
        averaged = _average_numeric(values)
        if averaged is not None:
            aggregated[key] = averaged
    return aggregated


def build_kfold_run_metadata(
    dataset_source: PinoDatasetCollectionData,
    training_profile: dict[str, Any],
    derivation_profile: dict[str, Any],
    metrics: dict[str, Any],
    kfold_summary: dict[str, Any],
    fold_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build compact metadata for a grouped native k-fold run record."""
    source_summary = build_training_source_summary(dataset_source)
    folds = list(fold_payload.get("folds", []))
    return {
        "run_name": training_profile.get("run_name", "kfold_run"),
        "config_name": training_profile.get("config_name", ""),
        "dataset_source_kind": source_summary["kind"],
        "dataset_source_pk": source_summary["pk"],
        "dataset_source_uuid": source_summary["uuid"],
        "dataset_source_label": source_summary["label"],
        "dataset_source_summary": source_summary,
        "training_profile": dict(training_profile),
        "derivation_profile": dict(derivation_profile),
        "kfold_summary": dict(kfold_summary),
        "metrics": dict(metrics),
        "folds": folds,
        "fold_count": len(folds),
    }


def recent_kfold_run_nodes(limit: int = 50) -> list[PinoKfoldRunData]:
    """Return recent stored grouped k-fold run nodes."""
    builder = orm.QueryBuilder()
    builder.append(PinoKfoldRunData, project=["*"])
    builder.order_by({PinoKfoldRunData: {"ctime": "desc"}})
    return builder.limit(limit).all(flat=True)


def describe_kfold_run_node(node: PinoKfoldRunData) -> str:
    """Create a compact selector label for a grouped k-fold run."""
    metadata = node.get_kfold_metadata()
    run_name = metadata.get("run_name") or node.label or "kfold_run"
    config_name = metadata.get("config_name") or "-"
    dataset_label = metadata.get("dataset_source_label") or "-"
    fold_count = metadata.get("fold_count", len(metadata.get("folds", [])))
    return f"{node.pk} | PinoKfoldRunData | {run_name} | {config_name} | folds={fold_count} | {dataset_label}"


def summarize_kfold_run_node(node: PinoKfoldRunData) -> dict[str, Any]:
    """Return a metadata-first summary for grouped k-fold runs."""
    metadata = node.get_kfold_metadata()
    return {
        "pk": node.pk,
        "label": node.label or "-",
        "run_name": metadata.get("run_name", node.label or "kfold_run"),
        "config_name": metadata.get("config_name", ""),
        "dataset_source_label": metadata.get("dataset_source_label", ""),
        "dataset_source_kind": metadata.get("dataset_source_kind", ""),
        "training_profile": metadata.get("training_profile", {}),
        "derivation_profile": metadata.get("derivation_profile", {}),
        "kfold_summary": metadata.get("kfold_summary", {}),
        "metrics": metadata.get("metrics", {}),
        "folds": metadata.get("folds", []),
        "fold_count": metadata.get("fold_count", len(metadata.get("folds", []))),
        "ctime": (
            node.ctime.isoformat(timespec="seconds")
            if isinstance(node.ctime, datetime)
            else str(node.ctime)
        ),
        "raw_metadata": metadata,
    }
