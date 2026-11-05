"""Helpers for FLOW-3D run-directory discovery and case-manifest metadata.

The sibling ``flow3D_scripts`` repo writes every run into ``runs/<case>/`` with
a ``case_manifest.json`` carrying the verbatim input config (``input.laser.*``,
``input.simulation.*``, ...), derived quantities, and on newer runs a
``material`` block ``{name, file, flow3d_label, properties, optical,
table_rows}``. Old v1 manifests (the ``rectangular_moving_laser_melt_v5`` era)
lack the ``material``/``parallel`` blocks; absence is tolerated everywhere.

Curated archives use a second, flat layout: a folder of per-case compact NPZs
(``<stem>.npz``) with optional ``<stem>.manifest.json`` sidecars holding the
same full case manifest (this is the manual ``lpno/data/flow3D``
format). ``discover_flow3d_cases`` handles both layouts.

This mirrors ``read_openfoam_case_metadata`` for the FLOW-3D side (BUG-02).
FLOW-3D has no ``beam_blocks`` concept, so the reader always reports
``beam_blocks = None``.
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
from typing import Any

CASE_MANIFEST_NAME = "case_manifest.json"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_flow3d_case_manifest(
    manifest: dict[str, Any],
    *,
    case_name_fallback: str = "",
    materials_root: str | Path | None = None,
) -> dict[str, Any]:
    """Extract normalized laser/material metadata from one FLOW-3D case manifest.

    When ``materials_root`` is given and the manifest's ``material.file``
    (repo-relative, e.g. ``materials/<name>/<name>.json``) resolves to a file
    under it, the parsed material JSON is attached as ``flow3d_material_source``.
    """
    config = manifest.get("input") or {}
    laser = config.get("laser") or {}
    simulation = config.get("simulation") or {}
    power_w = _float_or_none(laser.get("power_w"))
    v_scan = _float_or_none(laser.get("scan_speed_m_per_s"))
    beams = laser.get("beams")
    if power_w is None and isinstance(beams, list) and beams:
        # Independent beams (2-4, each with its own power, path and turn-off)
        # carry no top-level scalar, so both fields read None and the
        # positive-value guard rejects the case. Summarise them the same way the
        # exporter already did: total delivered power and mean speed, which is
        # exactly what `power_w` and `scan_speed_m_per_s` hold inside the NPZ
        # (flow3D_scripts `export_provenance_scalars`), so the two agree. One
        # scalar cannot describe several beams; the per-beam detail stays in
        # `flow3d_case_manifest`, and `beam_count` records that this is a
        # summary rather than a single condition.
        fallback = v_scan or 0.1
        speeds = [_float_or_none(b.get("scan_speed_m_per_s")) or fallback for b in beams]
        power_w = sum(_float_or_none(b.get("power_w")) or 0.0 for b in beams)
        v_scan = sum(speeds) / len(speeds)

    metadata: dict[str, Any] = {
        "case_name": str(manifest.get("case_name") or case_name_fallback),
        "V_scan": v_scan,
        "laser_power": power_w,
        "beam_radius": _float_or_none(laser.get("beam_radius_m")),
        "beam_blocks": None,
        "effective_end_time_s": _float_or_none(simulation.get("finish_time_s")),
        "flow3d_case_manifest": manifest,
    }
    if isinstance(beams, list) and beams:
        metadata["beam_count"] = len(beams)

    material = manifest.get("material")
    if isinstance(material, dict):
        metadata["flow3d_material"] = material
        material_file = material.get("file")
        if materials_root is not None and material_file:
            root = Path(materials_root).expanduser().resolve()
            stem = Path(material_file).stem
            # Older manifests record the flat pre-2026-06-30 layout
            # (materials/<name>.json); fall back to the current per-material
            # directory layout (materials/<name>/<name>.json).
            candidates = [
                (root / material_file).resolve(),
                (root / "materials" / stem / f"{stem}.json").resolve(),
            ]
            for candidate in candidates:
                if candidate.is_relative_to(root) and candidate.is_file():
                    metadata["flow3d_material_source"] = json.loads(candidate.read_text(encoding="utf-8"))
                    break
    return metadata


def read_flow3d_case_metadata(case_dir: str | Path, materials_root: str | Path | None = None) -> dict[str, Any]:
    """Read one ``flow3D_scripts`` run directory into normalized metadata."""
    case_path = Path(case_dir).expanduser().resolve()
    if not case_path.is_dir():
        raise ValueError(f"FLOW-3D case does not exist or is not a directory: {case_path}")
    manifest_path = case_path / CASE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"FLOW-3D case has no {CASE_MANIFEST_NAME}: {case_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metadata: dict[str, Any] = {
        "source_origin_kind": "flow3d_run_import",
        "source_folder_name": case_path.parent.name,
        "source_group": case_path.name,
        "source_uri": f"file://{case_path}",
    }
    metadata.update(
        parse_flow3d_case_manifest(
            manifest, case_name_fallback=case_path.name, materials_root=materials_root
        )
    )
    return metadata


def discover_flow3d_cases(
    root: str | Path, *, exclude_globs: list[str] | None = None
) -> list[dict[str, Any]]:
    """Return FLOW-3D cases under ``root``, in either supported layout.

    Run-directory layout: ``root`` itself when it holds a
    ``case_manifest.json``, otherwise its immediate subdirectories that do.
    Each entry records ``case_dir``, ``case_name`` (the directory name), and
    ``has_npz`` (whether the compact ``npz/<case>.npz`` export exists).

    Flat layout: any ``*.npz`` found recursively (skipping run directories,
    ``var_*.npz`` datasource caches, dot-directories such as
    ``.aiida_pino_cache``, and ``npz_tools/``) is one case. Flat entries carry
    ``layout="flat_npz"``, ``npz_path``, ``manifest_path`` (the
    ``<stem>.manifest.json`` sidecar, or None), and ``source_group`` (the
    subdirectory relative to ``root``, ``""`` for the root itself).
    ``exclude_globs`` are relative glob patterns (e.g. ``smoke/*`` or
    ``*_twin.npz``) matched against each NPZ's relative path.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"FLOW-3D source path does not exist or is not a directory: {root_path}")

    if (root_path / CASE_MANIFEST_NAME).is_file():
        case_dirs = [root_path]
    else:
        case_dirs = sorted(
            child for child in root_path.iterdir() if child.is_dir() and (child / CASE_MANIFEST_NAME).is_file()
        )
    entries: list[dict[str, Any]] = [
        {
            "case_dir": str(case),
            "case_name": case.name,
            "has_npz": (case / "npz" / f"{case.name}.npz").is_file(),
        }
        for case in case_dirs
    ]
    if case_dirs == [root_path]:
        return entries

    patterns = [str(pattern) for pattern in exclude_globs or []]
    flat_entries: list[dict[str, Any]] = []
    # Follow directory symlinks so linked split views are discoverable; the
    # visited set breaks symlink cycles and keeps re-linked dirs from
    # producing duplicate entries.
    visited: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=True):
        real_dir = os.path.realpath(dirpath)
        if real_dir in visited:
            dirnames[:] = []
            continue
        visited.add(real_dir)
        current = Path(dirpath)
        # Prune caches, tooling, and run directories (those are covered by the
        # run-directory layout above; their npz/<case>.npz must not double up).
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not name.startswith(".")
            and name != "npz_tools"
            and not (current / name / CASE_MANIFEST_NAME).is_file()
        )
        for filename in sorted(filenames):
            if not filename.endswith(".npz") or filename.startswith("var_"):
                continue
            npz_path = current / filename
            relative = npz_path.relative_to(root_path).as_posix()
            if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
                continue
            sidecar = npz_path.with_name(f"{npz_path.stem}.manifest.json")
            parent = Path(relative).parent.as_posix()
            flat_entries.append(
                {
                    "case_dir": str(current),
                    "case_name": npz_path.stem,
                    "has_npz": True,
                    "layout": "flat_npz",
                    "npz_path": str(npz_path),
                    "manifest_path": str(sidecar) if sidecar.is_file() else None,
                    "source_group": "" if parent == "." else parent,
                }
            )
    entries.extend(sorted(flat_entries, key=lambda entry: entry["npz_path"]))
    return entries
