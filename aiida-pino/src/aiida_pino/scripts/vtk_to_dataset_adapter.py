"""Standalone adapter for VTK dataset preparation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from aiida_pino.data import PinoDatasetCollectionData, PinoDatasetData, PinoSimulationData
from aiida_pino.data.pino_simulation import check_laser_scalars, is_stationary_case
from aiida_pino.utils.dataset_derivation import DEFAULT_DERIVATION_PROFILE
from aiida_pino.scripts.live_tail import run_with_live_tail
from aiida_pino.utils import canonicalize_import_metadata
from aiida_pino.utils.vtk_geometry import axis_convention, resolve_flip_axes, resolve_vtk_geometry

PACKAGE_INSTALL_HINT = (
    "`lpno` is not installed in this Code environment. "
    "Install the sibling checkout with `pip install -e /path/to/lpno`."
)
SOURCE_CACHE_ROOTNAME = ".aiida_pino_cache"
SOURCE_CACHE_NAMESPACE = "vtk_to_dataset"
TEMP_PAYLOAD_DIRNAME = "payloads"


def lpno_version() -> str:
    """Return the installed `lpno` version when available."""
    try:
        return importlib.metadata.version("lpno")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def load_vtk_datasource_class():
    """Import the packaged datasource with a clear error if the package is missing."""
    try:
        module = importlib.import_module("lpno.custom_utils.openFoam_datasource")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("lpno"):
            raise SystemExit(PACKAGE_INSTALL_HINT) from exc
        raise
    return module.VTKOpenFOAMDataSource


def cache_fingerprint(source_identifier: str, profile: dict, *, source_uri: str, flip_axes: bool) -> str:
    """Build a stable fingerprint for cache reuse across repeated imports."""
    requested_fields = [str(field) for field in profile.get("vtk_fields", []) if str(field).strip()]
    load_all_fields = len(requested_fields) == 0
    payload = {
        "source_identifier": source_identifier,
        "source_uri": source_uri,
        "geometry": profile["geometry"],
        "vtk_fields": requested_fields,
        "load_all_fields": load_all_fields,
        "flip_axes": bool(flip_axes),
        "canonical_layout": PinoDatasetData.CANONICAL_LAYOUT,
        # # Derived, never a literal -- see test_no_literal_time_as_dim_stamp_survives.
        "canonical_import": {
            "include_time": bool(DEFAULT_DERIVATION_PROFILE["include_time"]),
            "time_as_dim": bool(DEFAULT_DERIVATION_PROFILE["time_as_dim"]),
            "time_as_batch": bool(DEFAULT_DERIVATION_PROFILE["time_as_batch"]),
            "moving_ref_frame": False,
            "moving_frame_steps": 1,
            "time_avg_window": 1,
            "super_res_factor": 1,
        },
        "lpno_version": lpno_version(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def collect_source_metadata(datasource, input_dir: Path, *, time_axis: bool) -> tuple[str, list[dict], list[dict]]:
    """Collect filename/group metadata from the VTK input directory."""
    flat_paths = sorted(list(input_dir.glob("*.vtu")) + list(input_dir.glob("*.vtk")))
    groups: dict[str, list[Path]]
    grouping_mode: str

    if time_axis:
        if flat_paths:
            groups = datasource._group_files_by_base(paths=flat_paths)
            grouping_mode = "flat_files"
        else:
            groups = datasource._group_files_by_subdir()
            grouping_mode = "subdirectories"
    else:
        selected = [datasource._resolve_file_path(stem) for stem in datasource.get_file_list()]
        groups = {datasource._base_without_time(path.stem): [path] for path in selected}
        grouping_mode = "latest_per_base"

    sample_metadata: list[dict] = []
    grouped_metadata: list[dict] = []
    for group_name, paths in sorted(groups.items()):
        file_entries = []
        for path in paths:
            params = datasource._parse_params(path.stem)
            entry = {
                "group": group_name,
                "name": path.name,
                "relative_path": str(path.relative_to(input_dir)),
                "stem": path.stem,
                "params": params,
            }
            file_entries.append(entry)
            sample_metadata.append(entry)
        grouped_metadata.append(
            {
                "group": group_name,
                "num_files": len(file_entries),
                "files": file_entries,
            }
        )
    return grouping_mode, grouped_metadata, sample_metadata


def build_symlink_mirror(source_root: Path, mirror_root: Path) -> None:
    """Expose a source tree through lightweight top-level symlinks."""
    mirror_root.mkdir(parents=True, exist_ok=True)
    excluded_cache_root = source_root / SOURCE_CACHE_ROOTNAME
    for child in source_root.iterdir():
        if child == excluded_cache_root:
            continue
        if child.match("var_*.npz"):
            # BUG-05: a pre-existing datasource cache would be reused verbatim despite
            # a different geometry/flip; the datasource regenerates it in the workspace.
            continue
        target = mirror_root / child.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(child.resolve())


def link_or_copy_file(source: Path, destination: Path) -> None:
    """Create a hardlink when possible, otherwise copy the file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_external_cache_workspace(
    source_root: Path,
    fingerprint: str,
    *,
    force_recompute: bool,
) -> tuple[Path, str, Path]:
    """Prepare a source-side or sandbox-local symlink workspace for an external VTK path."""
    source_cache_root = source_root / SOURCE_CACHE_ROOTNAME / SOURCE_CACHE_NAMESPACE / fingerprint
    fallback_cache_root = Path("vtk_cache") / fingerprint

    try:
        source_cache_root.parent.mkdir(parents=True, exist_ok=True)
        if force_recompute and source_cache_root.exists():
            shutil.rmtree(source_cache_root)
        source_cache_root.mkdir(parents=True, exist_ok=True)
        build_symlink_mirror(source_root, source_cache_root)
        return source_cache_root, "source_cache", source_cache_root
    except OSError:
        if force_recompute and fallback_cache_root.exists():
            shutil.rmtree(fallback_cache_root)
        fallback_cache_root.mkdir(parents=True, exist_ok=True)
        build_symlink_mirror(source_root, fallback_cache_root)
        return fallback_cache_root, "sandbox_cache", fallback_cache_root


def locate_generated_cache(cache_dir: Path) -> Path:
    """Resolve the datasource-generated compressed cache file within a workspace."""
    canonical_candidate = cache_dir / PinoDatasetData.DEFAULT_BASE_CACHE_BASENAME
    if canonical_candidate.exists():
        return canonical_candidate

    matches = sorted(cache_dir.glob("var_*.npz"))
    if not matches:
        raise FileNotFoundError("Datasource did not generate a `var_*.npz` cache file.")
    return matches[0]


def _group_time_values(group_entry: dict) -> list[float]:
    values: list[float] = []
    for file_entry in group_entry.get("files", []):
        params = file_entry.get("params", {}) or {}
        value = params.get("time_s")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return []
    return values


def _run_summary_member_overrides(base_collection_metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract simulation metadata overrides from structured source metadata when available."""
    laserbeamfoam = base_collection_metadata.get("laserbeamfoam", {}) or {}
    run_summary = laserbeamfoam.get("run_summary", {}) or {}
    simulation_profile = run_summary.get("simulation_profile", {}) or {}
    derived_parameters = run_summary.get("derived_parameters", {}) or {}

    overrides: dict[str, Any] = {}
    if simulation_profile.get("V_scan") is not None:
        overrides["V_scan"] = float(simulation_profile["V_scan"])
    if simulation_profile.get("laser_power") is not None:
        overrides["laser_power"] = float(simulation_profile["laser_power"])
    if simulation_profile.get("beam_radius") is not None:
        overrides["beam_radius"] = float(simulation_profile["beam_radius"])
    if simulation_profile.get("beam_blocks") is not None:
        overrides["beam_blocks"] = int(simulation_profile["beam_blocks"])
    if derived_parameters.get("effective_end_time") is not None:
        overrides["effective_end_time_s"] = float(derived_parameters["effective_end_time"])
    if derived_parameters.get("deltaT") is not None:
        overrides["delta_t_s"] = float(derived_parameters["deltaT"])
    return overrides


_FLOW3D_OVERRIDE_KEYS = (
    "case_name",
    "V_scan",
    "laser_power",
    "beam_radius",
    "beam_blocks",
    "effective_end_time_s",
    "flow3d_case_manifest",
    "flow3d_material",
    "flow3d_material_source",
)


def _flow3d_member_overrides(base_metadata: dict[str, Any], group_name: str) -> dict[str, Any]:
    """Per-group metadata from flow3D_scripts case manifests (BUG-02).

    Converted VTK groups are named after the case directory, but legacy
    basename mode or time grouping can rename them, so an exact group-name
    match is tried first and a unique prefix relation second.
    """
    case_manifests = (base_metadata.get("flow3d") or {}).get("case_manifests") or {}
    entry = case_manifests.get(group_name)
    if entry is None:
        matches = [key for key in case_manifests if group_name.startswith(key) or key.startswith(group_name)]
        if len(matches) == 1:
            entry = case_manifests[matches[0]]
    if not isinstance(entry, dict):
        return {}
    overrides = {key: entry[key] for key in _FLOW3D_OVERRIDE_KEYS if key in entry}
    for scalar in ("V_scan", "laser_power", "beam_radius", "effective_end_time_s"):
        if overrides.get(scalar) is not None:
            overrides[scalar] = float(overrides[scalar])
    return overrides


_FLOW3D_LASER_CHANNELS = (("P", "laser_power"), ("V_scan", "V_scan"))


def _apply_flow3d_laser_channels(
    simulation_arrays: dict[str, np.ndarray],
    simulation_metadata: dict[str, Any],
    group_name: str,
    *,
    has_manifest_override: bool,
) -> None:
    """BUG-01: FLOW-3D `P`/`V_scan` training channels come from manifest metadata.

    The filename-derived grids are no longer trusted: manifest values overwrite
    (or create) constant grids, and a member whose laser power or scan speed
    still resolves to zero/None fails loudly instead of training on silent zeros.
    """
    if has_manifest_override:
        template = next(iter(simulation_arrays.values()))
        for channel, metadata_key in _FLOW3D_LASER_CHANNELS:
            value = simulation_metadata.get(metadata_key)
            if value is None:
                continue
            reference = simulation_arrays.get(channel, template)
            simulation_arrays[channel] = np.full(reference.shape, float(value), dtype=np.float32)

    # An all-zero GRID still means a dropped value -- except for a dwell weld's
    # V_scan, where zero is the measurement. This array check has no equivalent
    # in the scalar guard below, which is why it stays here rather than being
    # delegated with the rest.
    stationary = is_stationary_case(simulation_metadata)
    for channel, metadata_key in _FLOW3D_LASER_CHANNELS:
        if stationary and metadata_key == "V_scan":
            continue
        array = simulation_arrays.get(channel)
        if array is not None and array.size > 0 and not np.any(array):
            raise ValueError(
                f"FLOW-3D group `{group_name}` resolves laser channel `{channel}` "
                f"({metadata_key}) to zero/None. Point the ingest source path at "
                "flow3D_scripts run directories that contain `case_manifest.json` "
                "(runs/<case>/) so the laser power and scan speed can be read from "
                "the manifest instead of the filename."
            )

    # The SCALAR rule is shared with the NPZ adapter and the node constructor, so
    # it is called rather than typed a third time. It took two separate fixes a
    # week apart (dfa07b8, 755b61d) to teach the other two copies that a spot
    # weld may have V_scan = 0; this site never got either. Note it now runs even
    # WITH a manifest override -- the previous `metadata_missing` term was False
    # whenever a manifest existed, so a NaN manifest value was written straight
    # into the conditioning channel.
    check_laser_scalars(simulation_metadata, where=f"FLOW-3D group `{group_name}`")


def split_collection_payload(
    arrays: dict[str, np.ndarray],
    *,
    grouped_metadata: list[dict],
    base_collection_metadata: dict,
) -> tuple[list[dict], dict]:
    """Split canonical batched arrays into one payload per simulation group."""
    num_groups = len(grouped_metadata)
    if num_groups <= 0:
        raise ValueError("No grouped simulations were detected for import.")

    member_overrides = _run_summary_member_overrides(base_collection_metadata)
    run_summary_overrides_applied = True
    if num_groups > 1 and member_overrides:
        # BUG-18: a single laserbeamfoam run_summary cannot describe several
        # groups, so its shared laser scalars are only applied to single-group
        # imports; per-group flow3d manifest overrides are unaffected.
        member_overrides = {}
        run_summary_overrides_applied = False
    has_case_manifests = bool((base_collection_metadata.get("flow3d") or {}).get("case_manifests"))
    matched_groups: list[str] = []
    missed_groups: list[str] = []
    members: list[dict] = []
    for group_index, group_entry in enumerate(grouped_metadata):
        simulation_arrays: dict[str, np.ndarray] = {}
        for name, array in arrays.items():
            if int(array.shape[0]) != num_groups:
                raise ValueError(f"Array `{name}` batch axis does not match detected group count {num_groups}.")
            simulation_arrays[name] = np.asarray(array[group_index], dtype=np.float32)

        source_group = group_entry.get("group", f"group_{group_index:04d}")
        simulation_metadata = {
            "format_version": PinoSimulationData.FORMAT_VERSION,
            "source_type": base_collection_metadata["source_type"],
            "source_origin_kind": "vtk_import",
            "source_folder_name": base_collection_metadata["source_folder_name"],
            "source_uri": base_collection_metadata["source_uri"],
            "grid_params": base_collection_metadata["grid_params"],
            "requested_fields": list(base_collection_metadata["requested_fields"]),
            "source_group": source_group,
            "group_name": source_group,
            "relative_paths": [entry.get("relative_path", "") for entry in group_entry.get("files", [])],
            "sample_metadata": list(group_entry.get("files", [])),
            "time_values_s": _group_time_values(group_entry),
        }
        flow3d_overrides = _flow3d_member_overrides(base_collection_metadata, source_group) if has_case_manifests else {}
        if has_case_manifests:
            (matched_groups if flow3d_overrides else missed_groups).append(source_group)
        simulation_metadata.update(member_overrides)
        simulation_metadata.update(flow3d_overrides)
        if str(base_collection_metadata["source_type"]) == "flow3d":
            _apply_flow3d_laser_channels(
                simulation_arrays,
                simulation_metadata,
                source_group,
                has_manifest_override=bool(flow3d_overrides),
            )
        members.append(
            {
                "group": source_group,
                "payload_file": str(Path(TEMP_PAYLOAD_DIRNAME) / f"simulation_{group_index:04d}.npz"),
                "simulation_metadata": simulation_metadata,
                "payload_arrays": simulation_arrays,
            }
        )

    collection_metadata = dict(base_collection_metadata)
    if not run_summary_overrides_applied:
        collection_metadata["run_summary_overrides_applied"] = False
    if has_case_manifests:
        collection_metadata["flow3d_manifest_groups"] = {"matched": matched_groups, "missed": missed_groups}
    collection_metadata["simulation_count"] = len(members)
    collection_metadata["composition_kind"] = "import"
    collection_metadata["source_origin_kind"] = "vtk_import"
    return members, collection_metadata


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtk-input-dir")
    parser.add_argument("--vtk-source-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-file", required=True)
    parser.add_argument("--source-type", default="flow3d")
    parser.add_argument("--source-uri", default="")
    parser.add_argument("--source-folder-name", default="")
    parser.add_argument("--source-metadata-file", default="")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    has_input_dir = bool(args.vtk_input_dir)
    has_source_path = bool(args.vtk_source_path)
    if has_input_dir == has_source_path:
        raise SystemExit("Exactly one of --vtk-input-dir or --vtk-source-path must be provided.")

    source_root = Path(args.vtk_input_dir or args.vtk_source_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = json.loads(Path(args.profile_file).read_text(encoding="utf-8"))
    source_metadata = {}
    if args.source_metadata_file:
        source_metadata = json.loads(Path(args.source_metadata_file).read_text(encoding="utf-8"))
    VTKOpenFOAMDataSource = load_vtk_datasource_class()

    datasource = VTKOpenFOAMDataSource()
    source_uri = args.source_uri.strip() or f"file://{source_root}"
    source_folder_name = args.source_folder_name.strip() or (source_root.name or source_root.anchor)
    force_recompute = bool(args.force_recompute or profile.get("force_recompute", False))
    flip_axes = resolve_flip_axes(profile, args.source_type)
    requested_fields = [str(field) for field in profile.get("vtk_fields", []) if str(field).strip()]
    load_all_fields = len(requested_fields) == 0
    fingerprint = cache_fingerprint(str(source_root), profile, source_uri=source_uri, flip_axes=flip_axes)

    try:
        if has_input_dir:
            working_input_dir = source_root
            cache_mode = "sandbox_cache"
            cache_root = working_input_dir
        else:
            working_input_dir, cache_mode, cache_root = prepare_external_cache_workspace(
                source_root,
                fingerprint,
                force_recompute=force_recompute,
            )

        datasource.input_dir = working_input_dir
        grouping_mode, grouped_metadata, sample_metadata = collect_source_metadata(
            datasource,
            working_input_dir,
            time_axis=True,
        )

        geometry = resolve_vtk_geometry(
            profile.get("geometry", {}),
            vtk_dir=working_input_dir,
            flip_axes=flip_axes,
        )

        arrays = datasource.parse_directory(
            grid_params=geometry,
            input_dir=str(working_input_dir),
            requested_fields=None if load_all_fields else requested_fields,
            load_all_fields=load_all_fields,
            super_res_factor=1,
            debug_params=False,
            force_recompute=force_recompute,
            time_axis=True,
            avg_window_size=1,
            moving_frame_steps=1,
            moving_ref_frame=False,
            flip_axes=flip_axes,
        )

        imported_fields = list(datasource.extra_fields) if load_all_fields else requested_fields

        generated_cache = locate_generated_cache(working_input_dir)

        metadata = canonicalize_import_metadata(
            {
                "format_version": 1,
                "source_type": args.source_type,
                "source_folder_name": source_folder_name,
                "source_uri": source_uri,
                "source_description": f"VTK dataset imported via aiida-pino ({args.source_type}).",
                "input_dir": str(source_root),
                "grid_params": geometry,
                "requested_fields": imported_fields,
                "load_all_fields": load_all_fields,
                "super_res_factor": 1,
                "debug_params": False,
                "force_recompute": force_recompute,
                "flip_axes": flip_axes,
                "axis_convention": axis_convention(flip_axes),
                "time_axis": True,
                "avg_window_size": 1,
                "moving_frame_steps": 1,
                "moving_ref_frame": False,
                # # Derived, never a literal -- see test_no_literal_time_as_dim_stamp_survives.
                "time_axis_mode": {
                    "include_time": bool(DEFAULT_DERIVATION_PROFILE["include_time"]),
                    "time_as_batch": bool(DEFAULT_DERIVATION_PROFILE["time_as_batch"]),
                    "time_as_dim": bool(DEFAULT_DERIVATION_PROFILE["time_as_dim"]),
                },
                "grouping_mode": grouping_mode,
                "grouped_files": grouped_metadata,
                "sample_metadata": sample_metadata,
                "array_names": list(arrays),
                "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                "cache_basename": PinoDatasetData.DEFAULT_BASE_CACHE_BASENAME,
                "cache_mode": cache_mode,
                "cache_root": str(cache_root),
                "cache_fingerprint": fingerprint,
                "legacy_payload_file": generated_cache.name,
            }
        )
        if source_metadata:
            metadata.update(source_metadata)

        members, collection_metadata = split_collection_payload(
            arrays,
            grouped_metadata=grouped_metadata,
            base_collection_metadata=metadata,
        )
        for entry in members:
            payload_file = output_dir / entry["payload_file"]
            payload_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(payload_file, **entry.pop("payload_arrays"))

        manifest = {
            "format": PinoDatasetCollectionData.RUNTIME_MANIFEST_VERSION,
            "source_kind": "dataset_collection",
            "members": members,
            "grouping_mode": grouping_mode,
            "source_type": args.source_type,
            "source_uri": source_uri,
            "source_folder_name": source_folder_name,
            "cache_mode": cache_mode,
            "cache_root": str(cache_root),
            "cache_fingerprint": fingerprint,
            "dataset_metadata": collection_metadata,
        }
        summary = {
            "status": "ok",
            "dataset_identifier_hint": f"{source_folder_name} | {Path(collection_metadata['cache_basename']).stem} | {args.source_type}",
            "source_type": args.source_type,
            "source_uri": source_uri,
            "source_folder_name": source_folder_name,
            "requested_fields": metadata["requested_fields"],
            "array_names": metadata["array_names"],
            "array_shapes": metadata["array_shapes"],
            "grouping_mode": grouping_mode,
            "num_groups": len(grouped_metadata),
            "num_simulations": len(members),
            "num_files": len(sample_metadata),
            "profile": profile,
            "resolved_geometry": geometry,
            "canonical_layout": PinoDatasetData.CANONICAL_LAYOUT,
            "cache_mode": cache_mode,
            "cache_root": str(cache_root),
            "cache_fingerprint": fingerprint,
            "force_recompute": force_recompute,
        }
        if source_metadata:
            summary["source_metadata"] = source_metadata

        (output_dir / "metadata.json").write_text(
            json.dumps(collection_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        Path("dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        failure_summary = {
            "status": "error",
            "source_type": args.source_type,
            "source_uri": source_uri,
            "source_folder_name": source_folder_name,
            "profile": profile,
            "cache_fingerprint": fingerprint,
            "force_recompute": force_recompute,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        if source_metadata:
            failure_summary["source_metadata"] = source_metadata
        Path("dataset_summary.json").write_text(json.dumps(failure_summary, indent=2, sort_keys=True), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(run_with_live_tail(_main))
