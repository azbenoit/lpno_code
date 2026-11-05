"""Folder-based model discovery.

Each page's models live under ``data/<page>/models/<id>/`` with a ``meta.json``
(``{"label": ..., "type": ..., "order": N}``). The per-page dropdown is built by
listing those folders, so dropping in a new model folder makes it selectable
with no code change (see ``data/README.md``).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .model_physics import ModelPhysics, physics_from_meta

# discover_models() is called per request (model-id validation on every route,
# /models), so its result is cached per models_dir and invalidated by mtime:
# the directory's own (folders added/removed) and each folder's meta.json
# (edited in place). A stat per folder is far cheaper than re-reading and
# re-parsing every meta.json on every request.
_CACHE: dict[str, tuple[tuple, list[dict[str, Any]]]] = {}


def _stamp(root: Path) -> tuple:
    def mtime(p: Path) -> int:
        try:
            return p.stat().st_mtime_ns
        except OSError:
            return -1

    # The folder's own mtime covers a checkpoint copied in after the first
    # discovery (a meta-less folder is admitted on u_network.0.pth alone).
    return (mtime(root),) + tuple(
        (c.name, mtime(c), mtime(c / "meta.json")) for c in sorted(root.iterdir()) if c.is_dir()
    )


def evict_to_cap(cache: "OrderedDict[str, Any]", cap: int) -> list[str]:
    """Drop least-recently-used entries until at most ``cap`` remain; return the
    evicted keys.

    Shared by both model caches so they cannot drift apart. A loaded model pins
    its weights AND the neighbour graphs RIGNO/GAOT build lazily per grid, which
    dwarf the weights (a 150x80x60 grid is 720k nodes at k=8). An unbounded dict
    therefore grows to the sum of every model a viewer has ever selected, and
    ``gpu.empty_cache`` cannot help: those are live objects, not allocator slack.
    Keeps one entry minimum -- the caller has just inserted the model it is about
    to use, and evicting that would defeat the load it just paid for.
    """
    evicted: list[str] = []
    while len(cache) > max(1, int(cap)):
        key, _ = cache.popitem(last=False)
        evicted.append(key)
    return evicted


def discover_models(models_dir: str | Path) -> list[dict[str, Any]]:
    """Return ``[{id, label, type, order, default, physics, physics_parsed}]``
    for the model folders under ``models_dir``, sorted by ``order`` then ``id``.
    A folder without a ``meta.json`` still appears (label defaults to the folder
    id) as long as it holds a checkpoint, so a model is never silently hidden.
    ``physics`` is the raw meta.json block (or None) for the /models
    passthrough; ``physics_parsed`` is the ``ModelPhysics`` the adapters use.
    Cached per models_dir; see ``_stamp``."""
    root = Path(models_dir)
    if not root.is_dir():
        return []
    key = str(root.resolve())
    stamp = _stamp(root)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    models: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        meta: dict[str, Any] = {}
        meta_path = child / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                meta = {}
        elif not (child / "u_network.0.pth").is_file():
            continue  # neither metadata nor a checkpoint -> not a model folder
        models.append(
            {
                "id": child.name,
                "label": str(meta.get("label") or child.name),
                "type": str(meta.get("type") or ""),
                "order": int(meta["order"]) if str(meta.get("order", "")).lstrip("-").isdigit() else 1_000,
                "default": bool(meta.get("default", False)),
                # Optional variable-physics capability block (dict passthrough);
                # absent/invalid -> None = fixed-physics model.
                "physics": meta.get("physics") if isinstance(meta.get("physics"), dict) else None,
                "physics_parsed": physics_from_meta(meta),
            }
        )
    models.sort(key=lambda m: (m["order"], m["id"]))
    _CACHE[key] = (stamp, models)
    return models


def model_entry(models_dir: str | Path, model_id: str) -> dict[str, Any] | None:
    """One discovered model's entry, or None when the id is not a model folder."""
    return next((m for m in discover_models(models_dir) if m["id"] == model_id), None)


def physics_for(models_dir: str | Path, model_id: str) -> ModelPhysics:
    """The model's ``ModelPhysics``; an unknown id yields the fixed defaults so
    the caller's own id validation produces the 422, not a 500 here."""
    entry = model_entry(models_dir, model_id)
    return entry["physics_parsed"] if entry is not None else ModelPhysics()


def resolve_model_dir(models_dir: str | Path, model_id: str) -> Path:
    """Path of one model's folder, validated. Rejects path-traversal ids."""
    if not model_id or "/" in model_id or "\\" in model_id or model_id in (".", ".."):
        raise FileNotFoundError(f"Invalid model id: {model_id!r}")
    d = Path(models_dir) / model_id
    if not d.is_dir():
        raise FileNotFoundError(f"Model '{model_id}' not found under {models_dir}")
    return d
