"""VTK geometry helpers for the legacy VTK ingest (`vtk_to_dataset_adapter`).

Kept from the OpenFOAM import path when that path was removed (2026-09-15): these three
helpers are about VTK series, not OpenFOAM -- the legacy FLOW-3D `flow3d_to_vtk` pvpython
ingest resolves its grid and axis convention through them.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

def _load_vtk_datasource_class():
    module = importlib.import_module("lpno.custom_utils.openFoam_datasource")
    return module.VTKOpenFOAMDataSource


def resolve_flip_axes(profile: dict[str, Any] | None, source_type: str) -> bool:
    """Resolve the axis-flip flag source-aware: OpenFOAM flips axes, FLOW-3D does not (BUG-11).

    An explicit ``flip_axes`` in the profile always wins; when absent the default
    follows the source type instead of silently applying the OpenFOAM convention.
    """
    values = profile or {}
    if "flip_axes" in values:
        return bool(values["flip_axes"])
    return str(source_type) != "flow3d"


def axis_convention(flip_axes: bool) -> str:
    """Name the coordinate convention applied by ``flip_axes`` for stored metadata."""
    return "openfoam_y_negate_yz_swap" if flip_axes else "identity"


def resolve_vtk_geometry(
    geometry: dict[str, Any] | None,
    *,
    vtk_dir: str | Path,
    flip_axes: bool,
) -> dict[str, Any]:
    """Return explicit geometry, inferring a full-grid geometry when resolution is omitted."""
    normalized = dict(geometry or {})
    resolution = normalized.get("resolution")
    if resolution:
        return {
            "x_range": [float(normalized["x_range"][0]), float(normalized["x_range"][1])],
            "y_range": [float(normalized["y_range"][0]), float(normalized["y_range"][1])],
            "z_range": [float(normalized["z_range"][0]), float(normalized["z_range"][1])],
            "resolution": [int(resolution[0]), int(resolution[1]), int(resolution[2])],
        }

    vtk_root = Path(vtk_dir).expanduser().resolve()
    vtk_files = sorted(list(vtk_root.glob("*.vtu")) + list(vtk_root.glob("*.vtk")))
    if not vtk_files:
        # Multi-timestep FLOW-3D conversion writes per-case subdirectories.
        vtk_files = sorted(list(vtk_root.rglob("*.vtu")) + list(vtk_root.rglob("*.vtk")))
    if not vtk_files:
        raise ValueError(f"No VTK/VTU files were found under `{vtk_root}` (searched recursively).")

    datasource = _load_vtk_datasource_class()()
    data = datasource.read_file(vtk_files[0])
    coords = np.asarray(data["coordinates"], dtype=np.float64).copy()
    if flip_axes:
        coords[:, 1] = -coords[:, 1]
        coords = coords[:, [0, 2, 1]]

    rounded = np.round(coords, decimals=10)
    x_coords = np.unique(rounded[:, 0])
    y_coords = np.unique(rounded[:, 1])
    z_coords = np.unique(rounded[:, 2])
    if int(x_coords.size * y_coords.size * z_coords.size) != int(coords.shape[0]):
        raise ValueError(
            "Resolution inference requires a regular grid VTK dataset, but the detected cell centers do not "
            "form a tensor-product grid."
        )

    return {
        "x_range": [float(x_coords[0]), float(x_coords[-1])],
        "y_range": [float(y_coords[0]), float(y_coords[-1])],
        "z_range": [float(z_coords[0]), float(z_coords[-1])],
        "resolution": [int(x_coords.size), int(y_coords.size), int(z_coords.size)],
    }
