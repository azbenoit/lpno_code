"""Standalone adapter importing FLOW-3D compact-NPZ sources into native datasets.

This is the recommended FLOW-3D ingestion path: it needs neither ParaView
(``pvpython``) nor ``lpno`` — only numpy and the standard library.
Two source layouts are supported:

- run directories: every ``flow3D_scripts`` run dir carries a
  ``case_manifest.json`` plus the compact ``npz/<case>.npz`` export. The
  compact NPZ is stored verbatim (``storage_mode "npz_file"``): the adapter
  validates fields streaming (finiteness, shapes) and emits the file's
  ``os.path.realpath``; with ``time_target_frames`` it writes a frame-sliced
  compact copy through the workdir instead. Broadcast canonical arrays are
  never materialized for storage — that was the pre-2026-07-27 behaviour and
  it stored every simulation uncompressed at ~20x the size.
- flat per-case NPZ folders (the manual ``lpno/data/flow3D``
  format): ``<stem>.npz`` files with optional ``<stem>.manifest.json``
  sidecars, discovered recursively. No array bytes move through the workdir —
  the adapter validates each case from headers plus small axis reads and emits
  its ``os.path.realpath`` in the manifest; the parser stores the original file
  verbatim with ``PinoSimulationData.from_npz_file`` (``storage_mode
  "npz_file"``). Frame subsampling (``time_target_frames``) is not supported
  for this layout: those simulations always keep every stored frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from aiida_pino.data import PinoDatasetCollectionData, PinoDatasetData, build_simulation_fingerprint
from aiida_pino.scripts.live_tail import run_with_live_tail
from aiida_pino.scripts.vtk_to_dataset_adapter import TEMP_PAYLOAD_DIRNAME
from aiida_pino.utils import canonicalize_import_metadata, validate_flow3d_npz_profile
from aiida_pino.utils.dataset_derivation import DEFAULT_DERIVATION_PROFILE
from aiida_pino.utils.flow3d_import import (
    discover_flow3d_cases,
    parse_flow3d_case_manifest,
    read_flow3d_case_metadata,
)
from aiida_pino.utils.flow3d_npz import (
    canonical_shapes_from_headers,
    grid_params_from_axes,
    read_npz_headers,
    require_monotonic,
    select_frame_indices as select_frame_indices,  # re-export: public adapter API
)

GROUPING_MODE = "raw_flow3d_npz"
# NPZ provenance scalar -> manifest metadata key (manifest wins on mismatch).
PROVENANCE_SCALARS = (("power_w", "laser_power"), ("scan_speed_m_per_s", "V_scan"))


def npz_cache_fingerprint(source_identifier: str, profile: dict, *, source_uri: str) -> str:
    """Build a stable fingerprint for repeated imports of the same source."""
    payload = {
        "source_identifier": source_identifier,
        "source_uri": source_uri,
        "npz_fields": [str(field) for field in profile.get("npz_fields", []) or []],
        "time_target_frames": profile.get("time_target_frames"),
        "grouping_mode": GROUPING_MODE,
        "canonical_layout": PinoDatasetData.CANONICAL_LAYOUT,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _is_stationary(case_dir) -> bool:
    """True when the case manifest declares a stationary (dwell) laser.

    A dwell run has scan speed 0 by design; only then may the V_scan guard accept
    a zero. Absent or unreadable manifest -> False, so the strict rule applies.
    """
    try:
        target = Path(case_dir)
        # run-dir layout: <case_dir>/case_manifest.json;
        # flat layout: the sidecar path itself (<case>.manifest.json)
        path = target if target.is_file() else target / "case_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing/broken manifest keeps the strict rule
        return False
    derived = manifest.get("derived") or {}
    return bool(derived.get("laser_stationary"))


def _require_positive_laser_metadata(
    case_metadata: dict[str, Any],
    case_name: str,
    origin: str,
    *,
    stationary: bool = False,
) -> None:
    """Reject silent-zero/NaN P and V_scan channels.

    A stationary (dwell) case genuinely scans at 0 m/s, so for those the zero is
    the measurement, not a missing value -- but only when the manifest actually
    declares the run stationary (`derived.laser_stationary`). Everything else
    still has to be positive, which is what keeps a dropped speed from silently
    becoming a zero conditioning channel.
    """
    for metadata_key in ("laser_power", "V_scan"):
        value = case_metadata.get(metadata_key)
        allow_zero = stationary and metadata_key == "V_scan"
        if allow_zero and value is None:
            # A dwell config omits `laser.scan_speed_m_per_s` entirely, so the
            # manifest carries no value at all. For a declared-stationary run the
            # physical truth is 0 m/s; normalize it here so the conditioning
            # channel is the measurement rather than a hole.
            case_metadata[metadata_key] = value = 0.0
        if value is None or not math.isfinite(float(value)):
            ok = False
        elif allow_zero:
            ok = float(value) >= 0.0
        else:
            ok = float(value) > 0.0
        if not ok:
            detail = (
                "The laser power must be positive and the scan speed non-negative "
                "(this case is stationary); "
                if allow_zero
                else "The laser power and scan speed must be present, finite, and positive; "
            )
            raise ValueError(
                f"FLOW-3D case `{case_name}` resolves `{metadata_key}` to {value!r} from {origin}. "
                f"{detail}this import never produces silent-zero/NaN P/V_scan channels."
            )


def _provenance_warnings(archive, case_metadata: dict[str, Any], case_name: str) -> list[str]:
    """Compare NPZ provenance scalars against the manifest values (manifest wins)."""
    warnings: list[str] = []
    for npz_key, metadata_key in PROVENANCE_SCALARS:
        if npz_key not in archive.files:
            continue
        npz_value = float(np.asarray(archive[npz_key]))
        if np.isnan(npz_value):
            continue
        manifest_value = float(case_metadata[metadata_key])
        if abs(npz_value - manifest_value) > 1e-9 * abs(manifest_value):
            warnings.append(
                f"Case `{case_name}`: NPZ provenance scalar `{npz_key}` ({npz_value!r}) disagrees with "
                f"manifest `{metadata_key}` ({manifest_value!r}); the manifest value was used."
            )
    return warnings


def _case_member_payload(
    case_entry: dict[str, Any],
    *,
    profile: dict[str, Any],
    materials_root: str | None,
    payload_index: int,
    output_dir: Path,
    source_type: str,
    dataset_source_name: str,
) -> tuple[dict[str, Any], list[str], dict[str, list[int]], dict[str, Any], dict[str, Any], list[str]]:
    case_dir = Path(case_entry["case_dir"])
    case_name = str(case_entry["case_name"])
    case_metadata = read_flow3d_case_metadata(case_dir, materials_root=materials_root)
    # Recorded as a first-class field, not left inside the manifest blob: a spot
    # weld is physically a different experiment (V_scan = 0 is its measurement,
    # not a hole), consumers need to filter on it, and the node-level guard that
    # accepts the zero reads this.
    case_metadata["laser_stationary"] = _is_stationary(case_dir)
    _require_positive_laser_metadata(
        case_metadata, case_name, "its case_manifest.json",
        stationary=case_metadata["laser_stationary"],
    )

    relative_npz = str(Path("npz") / f"{case_name}.npz")
    real_path = Path(os.path.realpath(case_dir / relative_npz))
    requested = [str(field) for field in profile.get("npz_fields", []) or []] or None
    headers = read_npz_headers(str(real_path))
    array_shapes, selected_fields = canonical_shapes_from_headers(
        headers, case_name=case_name, selected_fields=requested
    )

    payload_file: Path | None = None
    with np.load(real_path, allow_pickle=False) as archive:
        warnings = _provenance_warnings(archive, case_metadata, case_name)
        x = np.asarray(archive["x"], dtype=np.float32)
        y = np.asarray(archive["y"], dtype=np.float32)
        z = np.asarray(archive["z"], dtype=np.float32)
        times = np.asarray(archive["time_s"], dtype=np.float64)
        for axis, name in ((x, "x"), (y, "y"), (z, "z"), (times, "time_s")):
            require_monotonic(axis, name, case_name)
        frame_indices = select_frame_indices(int(times.shape[0]), profile.get("time_target_frames"))
        subsampled = len(frame_indices) != int(times.shape[0])
        if subsampled:
            # Frame subsampling still needs a rewritten payload — but a *compact*
            # one (sliced fields + 1-D axes), never materialized broadcast
            # arrays. The parser stores it verbatim via `from_npz_file`.
            payload_file = Path(TEMP_PAYLOAD_DIRNAME) / f"simulation_{payload_index:04d}.npz"
            _write_compact_subset(
                archive,
                output_dir / payload_file,
                case_name=case_name,
                fields=selected_fields,
                frame_indices=frame_indices,
                x=x, y=y, z=z, times=times,
            )
        elif not bool(profile.get("trust_npz_payload", False)):
            # Verbatim store: validate the fields without keeping them — one at
            # a time, so a big case never holds more than one array in memory.
            for field_name in selected_fields:
                data = np.asarray(archive[field_name])
                if not np.isfinite(data).all():
                    raise ValueError(
                        f"FLOW-3D case `{case_name}` field `{field_name}` contains non-finite values."
                    )
                del data
    selected_times = times[frame_indices]
    if subsampled:
        array_shapes = {
            name: [len(frame_indices), *shape[1:]] for name, shape in array_shapes.items()
        }

    grid_params = grid_params_from_axes(x, y, z)
    samples = [
        {"relative_path": relative_npz, "frame_index": int(index), "params": {"time_s": float(times[index])}}
        for index in frame_indices
    ]
    deltas = np.diff(selected_times)
    delta_t_s = float(deltas[0]) if deltas.size and np.allclose(deltas, deltas[0]) else None

    simulation_metadata = {
        "format_version": 1,
        "source_type": source_type,
        "source_origin_kind": "flow3d_npz_import",
        "source_folder_name": dataset_source_name,
        "source_group": case_name,
        "group_name": case_name,
        "source_uri": case_metadata["source_uri"],
        "requested_fields": list(selected_fields),
        "grid_params": grid_params,
        # One entry per selected frame so the simulation fingerprint tracks the
        # frame subsampling: distinct `time_target_frames` of the same NPZ must
        # not collide onto one dedup identity (would reuse stale-frame members
        # and wrongly supersede the honest full-frame collection).
        "relative_paths": [f"{relative_npz}#frame={index}" for index in frame_indices],
        "sample_metadata": samples,
        "time_values_s": [float(value) for value in selected_times],
        "V_scan": float(case_metadata["V_scan"]),
        "laser_power": float(case_metadata["laser_power"]),
        # Forward the flag this adapter already resolved for its own guard.
        # Omitting it stored `laser_stationary=False` next to `V_scan=0.0` on
        # every spot weld -- the exact combination the guard rejects.
        "laser_stationary": bool(case_metadata.get("laser_stationary", False)),
        "beam_radius": case_metadata.get("beam_radius"),
        "beam_blocks": case_metadata.get("beam_blocks"),
        "effective_end_time_s": float(selected_times[-1]),
        "delta_t_s": delta_t_s,
        "flip_axes": False,
        "axis_convention": "identity",
        # Optional human annotations, applied to every member of this import
        # (the whole run directory is one batch). `usable_for_training` defaults
        # True and can be flipped per node later via its extra.
        "description": profile.get("description"),
        "batch_keyword": profile.get("batch_keyword"),
        "usable_for_training": profile.get("usable_for_training", True),
        "flow3d_case_manifest": case_metadata["flow3d_case_manifest"],
    }
    for optional_key in ("flow3d_material", "flow3d_material_source"):
        if optional_key in case_metadata:
            simulation_metadata[optional_key] = case_metadata[optional_key]

    # Pin the dedup identity to the legacy `from_arrays` recipe: these members
    # historically fingerprinted over the run metadata (manifest source_uri,
    # folder/group names, per-frame relative_paths), while `from_npz_file`
    # defaults to a realpath-based recipe. Without this pin, every re-import of
    # an already-ingested run dir would mint a new identity and duplicate the
    # member instead of deduplicating onto it.
    simulation_metadata["simulation_fingerprint"] = build_simulation_fingerprint(simulation_metadata)

    member = {
        "group": case_name,
        "storage_mode": "npz_file",
        "simulation_metadata": simulation_metadata,
    }
    if payload_file is not None:
        member["payload_file"] = str(payload_file)
    else:
        member["source_npz_realpath"] = str(real_path)
    group_entry = {"group": case_name, "num_files": len(samples), "files": samples}
    return member, list(selected_fields), array_shapes, grid_params, group_entry, warnings


def _write_compact_subset(
    archive,
    destination: Path,
    *,
    case_name: str,
    fields: list[str],
    frame_indices: list[int],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    times: np.ndarray,
) -> None:
    """Write a frame-sliced copy of a compact per-case NPZ.

    Same schema as the source (1-D axes, provenance scalars, `(T, nx, ny, nz[, C])`
    fields at their original dtype), just fewer frames. Streams one field at a
    time; bookkeeping arrays other than the provenance scalars are dropped.
    """
    import zipfile

    from numpy.lib import format as npformat

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:

        def _write(name: str, array: np.ndarray) -> None:
            with bundle.open(f"{name}.npy", "w", force_zip64=True) as handle:
                npformat.write_array(handle, np.ascontiguousarray(array), allow_pickle=False)

        _write("x", x)
        _write("y", y)
        _write("z", z)
        _write("time_s", times[frame_indices])
        for scalar_key in ("power_w", "scan_speed_m_per_s", "case_name"):
            if scalar_key in archive.files:
                _write(scalar_key, np.asarray(archive[scalar_key]))
        for field_name in fields:
            data = np.asarray(archive[field_name])
            if not np.isfinite(data).all():
                raise ValueError(
                    f"FLOW-3D case `{case_name}` field `{field_name}` contains non-finite values."
                )
            _write(field_name, data[frame_indices])
            del data


def _flat_case_member(
    case_entry: dict[str, Any],
    *,
    profile: dict[str, Any],
    materials_root: str | None,
    source_root: Path,
    source_type: str,
    dataset_source_name: str,
) -> tuple[dict[str, Any], list[str], dict[str, list[int]], dict[str, Any], dict[str, Any], list[str]]:
    """Validate one flat per-case NPZ and emit an ``npz_file``-mode member.

    Header-only shape/dtype validation plus small axis reads; the field payloads
    are deliberately never decompressed or copied here (a full batch would be a
    multi-10-GB transient in the CalcJob workdir). The parser stores the file
    verbatim from ``source_npz_realpath`` via ``from_npz_file``.
    """
    case_name = str(case_entry["case_name"])
    npz_path = Path(case_entry["npz_path"])
    real_path = Path(os.path.realpath(npz_path))
    relative_path = npz_path.relative_to(source_root).as_posix()

    case_metadata: dict[str, Any] = {}
    if case_entry.get("manifest_path"):
        manifest = json.loads(Path(case_entry["manifest_path"]).read_text(encoding="utf-8"))
        case_metadata = parse_flow3d_case_manifest(
            manifest, case_name_fallback=case_name, materials_root=materials_root
        )

    headers = read_npz_headers(str(real_path))
    requested = [str(field) for field in profile.get("npz_fields", []) or []] or None
    array_shapes, selected_fields = canonical_shapes_from_headers(
        headers, case_name=case_name, selected_fields=requested
    )

    warnings: list[str] = []
    with np.load(real_path, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float32)
        y = np.asarray(archive["y"], dtype=np.float32)
        z = np.asarray(archive["z"], dtype=np.float32)
        times = np.asarray(archive["time_s"], dtype=np.float64)
        for axis, name in ((x, "x"), (y, "y"), (z, "z"), (times, "time_s")):
            require_monotonic(axis, name, case_name)
        if case_metadata:
            warnings = _provenance_warnings(archive, case_metadata, case_name)
            laser_origin = "its manifest sidecar"
        else:
            # No sidecar manifest: fall back to the NPZ provenance scalars.
            for scalar_key, metadata_key in PROVENANCE_SCALARS:
                value = float(np.asarray(archive[scalar_key])) if scalar_key in archive.files else None
                case_metadata[metadata_key] = value
            case_metadata.setdefault("beam_radius", None)
            case_metadata.setdefault("beam_blocks", None)
            laser_origin = "its NPZ provenance scalars (no manifest sidecar found)"
    # BUG: `case_dir` is a run-directory-path local; in the flat-case path it is
    # undefined, so every flat NPZ ingest died with NameError (hit 2026-07-29
    # ingesting a flat steady-frame pool). The stationary check belongs to the
    # sidecar manifest here, which sits beside the NPZ.
    case_metadata["laser_stationary"] = (
        _is_stationary(case_entry["manifest_path"]) if case_entry.get("manifest_path") else False
    )
    _require_positive_laser_metadata(
        case_metadata, case_name, laser_origin,
        stationary=case_metadata["laser_stationary"],
    )

    target = profile.get("time_target_frames")
    if target is not None and int(times.shape[0]) > int(target):
        raise ValueError(
            f"dataset_profile.time_target_frames={target} is not supported for flat per-case NPZ "
            f"ingest: case `{case_name}` has {int(times.shape[0])} stored frames and npz_file-mode "
            "simulations always keep every frame. Drop time_target_frames (or import the "
            "run-directory layout, which supports frame subsampling)."
        )

    grid_params = grid_params_from_axes(x, y, z)
    deltas = np.diff(times)
    delta_t_s = float(deltas[0]) if deltas.size and np.allclose(deltas, deltas[0]) else None

    simulation_metadata = {
        "format_version": 1,
        "source_type": source_type,
        "source_origin_kind": "flow3d_npz_file_import",
        "source_folder_name": dataset_source_name,
        "source_group": str(case_entry.get("source_group", "")),
        "case_name": case_name,
        "source_uri": f"file://{real_path}",
        "requested_fields": list(selected_fields),
        "grid_params": grid_params,
        "relative_paths": [relative_path],
        # One sample entry per case (not per frame) keeps flat-batch manifests
        # small; the frame times live on the node via time_values_s.
        "sample_metadata": [
            {
                "group": case_name,
                "relative_path": relative_path,
                "params": {"num_frames": int(times.shape[0]), "time_s_end": float(times[-1])},
            }
        ],
        "V_scan": float(case_metadata["V_scan"]),
        "laser_power": float(case_metadata["laser_power"]),
        # Forward the flag this adapter already resolved for its own guard.
        # Omitting it stored `laser_stationary=False` next to `V_scan=0.0` on
        # every spot weld -- the exact combination the guard rejects.
        "laser_stationary": bool(case_metadata.get("laser_stationary", False)),
        "beam_radius": case_metadata.get("beam_radius"),
        "beam_blocks": case_metadata.get("beam_blocks"),
        "delta_t_s": delta_t_s,
        "flip_axes": False,
        "axis_convention": "identity",
        "description": profile.get("description"),
        "batch_keyword": profile.get("batch_keyword"),
        "usable_for_training": profile.get("usable_for_training", True),
    }
    for optional_key in ("flow3d_case_manifest", "flow3d_material", "flow3d_material_source"):
        if optional_key in case_metadata:
            simulation_metadata[optional_key] = case_metadata[optional_key]

    member = {
        "group": case_name,
        "storage_mode": "npz_file",
        "source_npz_realpath": str(real_path),
        "simulation_metadata": simulation_metadata,
    }
    group_entry = {
        "group": case_name,
        "num_files": 1,
        "files": list(simulation_metadata["sample_metadata"]),
    }
    return member, list(selected_fields), array_shapes, grid_params, group_entry, warnings


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-source-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-file", required=True)
    parser.add_argument("--source-type", default="flow3d")
    parser.add_argument("--source-uri", default="")
    parser.add_argument("--source-folder-name", default="")
    parser.add_argument("--source-metadata-file", default="")
    parser.add_argument("--materials-root", default="")
    args = parser.parse_args()

    source_root = Path(args.raw_source_path).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TEMP_PAYLOAD_DIRNAME).mkdir(parents=True, exist_ok=True)
    profile = json.loads(Path(args.profile_file).read_text(encoding="utf-8"))
    source_metadata = {}
    if args.source_metadata_file:
        source_metadata = json.loads(Path(args.source_metadata_file).read_text(encoding="utf-8"))
    materials_root = args.materials_root.strip() or None

    source_uri = args.source_uri.strip() or f"file://{source_root}"
    source_folder_name = args.source_folder_name.strip() or (source_root.name or source_root.anchor)
    fingerprint = npz_cache_fingerprint(str(source_root), profile, source_uri=source_uri)

    skipped_case_names: list[str] = []
    geometry: dict[str, Any] | None = None
    manifest_members: list[dict[str, Any]] = []
    grouped_files: list[dict[str, Any]] = []
    sample_metadata: list[dict[str, Any]] = []
    imported_case_names: list[str] = []
    flat_case_names: list[str] = []
    imported_fields: list[str] = []
    representative_shapes: dict[str, list[int]] = {}
    provenance_warnings: list[str] = []

    try:
        validate_flow3d_npz_profile(profile)
        cases = discover_flow3d_cases(source_root, exclude_globs=profile.get("exclude_globs"))
        run_cases = [case for case in cases if case.get("layout") != "flat_npz"]
        flat_cases = [case for case in cases if case.get("layout") == "flat_npz"]
        ingestable = [case for case in run_cases if case["has_npz"]]
        skipped_case_names.extend(sorted(case["case_name"] for case in run_cases if not case["has_npz"]))
        if not ingestable and not flat_cases:
            raise ValueError(
                f"No FLOW-3D cases were found under `{source_root}`: neither run directories with "
                "a compact `npz/<case>.npz` export nor flat per-case NPZ files."
            )

        def _accumulate(result: tuple) -> None:
            nonlocal geometry, representative_shapes, imported_fields
            member, member_fields, array_shapes, case_geometry, group_entry, warnings = result
            if geometry is None:
                geometry = case_geometry
            elif geometry != case_geometry:
                # A record, not a refusal. This raise was exit 410 for any
                # multi-case import whose cases differ in domain extent — which
                # is a real dataset shape (`ti64_geom40`: 40 cases, 40 extents,
                # one resolution). The collection layer now carries
                # `uniform_grid: False` with per-member shapes, and consumers
                # that need one shape gate on the flag — the same move the time
                # axis made with `uniform_timesteps`. `geometry` keeps the FIRST
                # case's value, matching what `grid_params` means everywhere.
                provenance_warnings.append(
                    f"mixed grids: first case uses {geometry}, `{member['group']}` "
                    f"uses {case_geometry}; the dataset will report uniform_grid=False"
                )
            manifest_members.append(member)
            grouped_files.append(group_entry)
            sample_metadata.extend(group_entry["files"])
            representative_shapes = array_shapes
            imported_fields = member_fields
            imported_case_names.append(str(member["group"]))
            provenance_warnings.extend(warnings)

        for payload_index, case_entry in enumerate(ingestable):
            _accumulate(
                _case_member_payload(
                    case_entry,
                    profile=profile,
                    materials_root=materials_root,
                    payload_index=payload_index,
                    output_dir=output_dir,
                    source_type=args.source_type,
                    dataset_source_name=source_folder_name,
                )
            )
        for case_entry in flat_cases:
            _accumulate(
                _flat_case_member(
                    case_entry,
                    profile=profile,
                    materials_root=materials_root,
                    source_root=source_root,
                    source_type=args.source_type,
                    dataset_source_name=source_folder_name,
                )
            )
            flat_case_names.append(str(case_entry["case_name"]))

        metadata = canonicalize_import_metadata(
            {
                "format_version": 1,
                "source_type": args.source_type,
                "source_folder_name": source_folder_name,
                "source_uri": source_uri,
                "source_description": "FLOW-3D compact NPZ runs imported through aiida-pino.",
                "input_dir": str(source_root),
                "grid_params": geometry,
                "requested_fields": imported_fields,
                "load_all_fields": not profile.get("npz_fields"),
                "super_res_factor": 1,
                "debug_params": False,
                "force_recompute": False,
                "flip_axes": False,
                "axis_convention": "identity",
                "time_axis": True,
                "avg_window_size": 1,
                "moving_frame_steps": 1,
                "moving_ref_frame": False,
                # From DEFAULT_DERIVATION_PROFILE, not a literal: the
                # 2026-07-22 canonical flip left three copies of this stamp
                # behind, each contradicting the canonical_layout written
                # beside it.
                "time_axis_mode": {
                    "include_time": bool(DEFAULT_DERIVATION_PROFILE["include_time"]),
                    "time_as_batch": bool(DEFAULT_DERIVATION_PROFILE["time_as_batch"]),
                    "time_as_dim": bool(DEFAULT_DERIVATION_PROFILE["time_as_dim"]),
                },
                "grouping_mode": GROUPING_MODE,
                "grouped_files": grouped_files,
                "sample_metadata": sample_metadata,
                "array_names": imported_fields,
                "array_shapes": representative_shapes,
                "cache_basename": PinoDatasetData.DEFAULT_BASE_CACHE_BASENAME,
                "cache_mode": "none",
                "cache_root": "",
                "cache_fingerprint": fingerprint,
                "simulation_count": len(manifest_members),
                "imported_cases": imported_case_names,
                "skipped_cases": skipped_case_names,
            }
        )
        if source_metadata:
            metadata.update(source_metadata)

        manifest = {
            "format": PinoDatasetCollectionData.RUNTIME_MANIFEST_VERSION,
            "source_kind": "dataset_collection",
            "members": manifest_members,
            "grouping_mode": GROUPING_MODE,
            "source_type": args.source_type,
            "source_uri": source_uri,
            "source_folder_name": source_folder_name,
            "cache_mode": "none",
            "cache_root": "",
            "cache_fingerprint": fingerprint,
            "dataset_metadata": metadata,
        }
        summary = {
            "status": "ok",
            "dataset_identifier_hint": f"{source_folder_name} | {Path(metadata['cache_basename']).stem} | {args.source_type}",
            "source_type": args.source_type,
            "source_uri": source_uri,
            "source_folder_name": source_folder_name,
            "requested_fields": imported_fields,
            "array_names": imported_fields,
            "array_shapes": representative_shapes,
            "grouping_mode": GROUPING_MODE,
            "num_groups": len(grouped_files),
            "num_simulations": len(manifest_members),
            "num_files": len(sample_metadata),
            "profile": profile,
            "resolved_geometry": geometry,
            "canonical_layout": PinoDatasetData.CANONICAL_LAYOUT,
            "cache_mode": "none",
            "cache_root": "",
            "cache_fingerprint": fingerprint,
            "imported_cases": imported_case_names,
            "flat_npz_cases": flat_case_names,
            "skipped_cases": skipped_case_names,
            "provenance_warnings": provenance_warnings,
        }
        if source_metadata:
            summary["source_metadata"] = source_metadata

        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
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
            "skipped_cases": skipped_case_names,
            "provenance_warnings": provenance_warnings,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        if source_metadata:
            failure_summary["source_metadata"] = source_metadata
        Path("dataset_summary.json").write_text(json.dumps(failure_summary, indent=2, sort_keys=True), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(run_with_live_tail(_main))
