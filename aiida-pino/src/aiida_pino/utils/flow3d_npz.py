"""Shared canonicalization for FLOW-3D compact per-case NPZ files.

A ``flow3D_scripts`` compact export stores ``x(nx,) y(ny,) z(nz,) time_s(nt,)``,
the provenance scalars ``power_w``/``scan_speed_m_per_s``/``case_name``, and
fields shaped ``(nt, nx, ny, nz)`` (scalar) or ``(nt, nx, ny, nz, 3)`` (vector,
e.g. ``Velocity``). The native simulation layout is ``(T, C, nx, ny, nz)`` with
broadcast axis channels and constant ``P``/``V_scan`` channels.

This module is the single owner of that raw->canonical mapping. The
``raw_flow3d_to_dataset_adapter`` uses it at import time (``from_arrays`` path)
and ``PinoSimulationData`` uses it at read time for ``storage_mode="npz_file"``
nodes, so both storage modes produce byte-identical arrays.

Deliberately numpy + stdlib only: ``aiida_pino.data`` imports this lazily and
must not pull the ``aiida_pino.utils`` package init (circular import).
"""

from __future__ import annotations

import zipfile
from typing import IO, Any, Iterable, NamedTuple

import numpy as np
from numpy.lib import format as npformat

# NPZ keys that are axes/provenance, not physical fields. frame_index and
# skipped_fields are bookkeeping arrays the VTK->NPZ conversion tooling writes
# (kept verbatim inside the stored payload, just never served as fields).
# frame_stride records time subsampling (flow3D_scripts export_all_timesteps_to_npz
# --frame-stride) and is scalar, like power_w.
NON_FIELD_KEYS = frozenset(
    {
        "x", "y", "z", "time_s", "power_w", "scan_speed_m_per_s", "case_name",
        "frame_index", "skipped_fields", "frame_stride",
    }
)


def member_shape(archive, name: str) -> tuple[int, ...]:
    """Shape of one archive member WITHOUT decompressing it.

    Indexing an `NpzFile` decompresses that member in full, so classifying
    members with `np.shape(archive[name])` materialised the entire payload just
    to read shapes — on every call, even a single-field read. `get_array` calls
    the canonicalizer once per field and one melt-metrics pass needs ~8 arrays,
    so each case decompressed its whole payload about eight times. Measured
    consequence: 3193 s/case on the core-dataset backfill against ~4 min for the
    same work through the console.

    Reads the `.npy` header from the zip stream instead, which touches only the
    first bytes. Falls back to the honest (expensive) shape for a plain mapping
    or anything unexpected — correctness first, speed where it is available.
    """
    zip_file = getattr(archive, "zip", None)
    if zip_file is not None:
        import numpy.lib.format as _fmt

        try:
            with zip_file.open(f"{name}.npy") as handle:
                version = _fmt.read_magic(handle)
                shape, _fortran, _dtype = _fmt._read_array_header(handle, version)
            return tuple(shape)
        except Exception:  # noqa: BLE001 - fall back rather than fail a read
            pass
    import numpy as _np

    return tuple(_np.shape(archive[name]))


def is_metadata_array(name: str, shape: tuple[int, ...]) -> bool:
    """True for anything that is not a per-cell physical field.

    Named keys above, plus **any 0-d array**. The name list alone is a standing
    trap: on 2026-08-02 the exporter gained a new scalar (`frame_stride`) and four
    live AiiDA batches failed ingest with "frame_stride has shape [], expected
    (T, nx, ny, nz[, C])" purely because the name was not listed yet. Treating
    rank-0 arrays as metadata makes the next such scalar additive instead of
    breaking."""
    return name in NON_FIELD_KEYS or len(shape) == 0
REQUIRED_AXIS_KEYS = ("x", "y", "z", "time_s")
# Canonical channel names prepended before the physical fields.
CHANNEL_NAMES = ("x", "y", "z", "P", "V_scan", "time_s")


class CanonicalCase(NamedTuple):
    """Canonicalized arrays plus the raw axes they were derived from."""

    arrays: dict[str, np.ndarray]
    fields: list[str]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    times: np.ndarray  # full time axis, float64
    frame_indices: list[int]
    selected_times: np.ndarray  # float64


def canonical_array_names(fields: Iterable[str]) -> list[str]:
    """Return the canonical original-array-name order: channels then fields."""
    return [*CHANNEL_NAMES, *fields]


def select_frame_indices(num_frames: int, target: int | None) -> list[int]:
    """Evenly subsample frame indices, always keeping the first and last frame."""
    if target is None or num_frames <= int(target):
        return list(range(num_frames))
    picks = np.round(np.linspace(0, num_frames - 1, int(target))).astype(int)
    return sorted(set(picks.tolist()))


def require_monotonic(values: np.ndarray, name: str, case_name: str) -> None:
    """Require a non-empty strictly increasing 1-D axis array."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"FLOW-3D case `{case_name}` NPZ array `{name}` must be a non-empty 1-D array.")
    if values.size > 1 and not np.all(np.diff(values.astype(np.float64)) > 0):
        raise ValueError(f"FLOW-3D case `{case_name}` NPZ array `{name}` is not strictly increasing.")


def resolve_selected_fields(
    available_fields: list[str], requested_fields: list[str] | None, case_name: str
) -> list[str]:
    """Resolve the requested field subset against the fields present in the NPZ."""
    selected = [str(field) for field in requested_fields or []] or list(available_fields)
    missing = sorted(set(selected) - set(available_fields))
    if missing:
        raise ValueError(
            f"FLOW-3D case `{case_name}` NPZ has no field(s) {', '.join(missing)}. "
            f"Available fields: {', '.join(available_fields)}."
        )
    return selected


def _validate_field_shape(
    name: str, shape: tuple[int, ...], num_frames: int, nx: int, ny: int, nz: int, case_name: str
) -> None:
    if len(shape) not in (4, 5) or tuple(shape[:4]) != (num_frames, nx, ny, nz):
        raise ValueError(
            f"FLOW-3D case `{case_name}` field `{name}` has shape {list(shape)}, "
            f"expected (T, nx, ny, nz[, C]) = ({num_frames}, {nx}, {ny}, {nz}[, C])."
        )


def grid_params_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    """Build the standard ``grid_params`` block from the raw axis arrays."""
    return {
        "x_range": [float(x[0]), float(x[-1])],
        "y_range": [float(y[0]), float(y[-1])],
        "z_range": [float(z[0]), float(z[-1])],
        "resolution": [int(x.size), int(y.size), int(z.size)],
    }


def canonicalize_flow3d_case_arrays(
    archive,
    *,
    case_name: str,
    laser_power: float,
    v_scan: float,
    selected_fields: list[str] | None = None,
    time_target_frames: int | None = None,
    only: Iterable[str] | None = None,
) -> CanonicalCase:
    """Canonicalize one open compact-NPZ archive into ``(T, C, nx, ny, nz)`` arrays.

    ``archive`` is an open ``np.load`` NpzFile (or any mapping with ``.files``).
    ``only`` restricts which output arrays are materialized (a single-field read
    still validates the axes but skips decompressing every other field).
    """
    for axis_name in REQUIRED_AXIS_KEYS:
        if axis_name not in archive.files:
            raise ValueError(f"FLOW-3D case `{case_name}` NPZ is missing required array `{axis_name}`.")
    x = np.asarray(archive["x"], dtype=np.float32)
    y = np.asarray(archive["y"], dtype=np.float32)
    z = np.asarray(archive["z"], dtype=np.float32)
    times = np.asarray(archive["time_s"], dtype=np.float64)
    for axis, name in ((x, "x"), (y, "y"), (z, "z"), (times, "time_s")):
        require_monotonic(axis, name, case_name)

    # `member_shape` reads the npy header instead of decompressing: indexing an
    # NpzFile materialises the member, so this loop used to decompress the whole
    # payload on EVERY call, including a single-field read.
    available_fields = [
        name for name in archive.files
        if not is_metadata_array(name, member_shape(archive, name))
    ]
    selected = resolve_selected_fields(available_fields, selected_fields, case_name)

    frame_indices = select_frame_indices(times.shape[0], time_target_frames)
    selected_times = times[frame_indices]
    num_frames = len(frame_indices)
    nx, ny, nz = int(x.size), int(y.size), int(z.size)
    grid_shape = (num_frames, 1, nx, ny, nz)
    wanted = None if only is None else set(only)

    arrays: dict[str, np.ndarray] = {}

    def _want(name: str) -> bool:
        return wanted is None or name in wanted

    if _want("x"):
        arrays["x"] = np.broadcast_to(x[None, None, :, None, None], grid_shape)
    if _want("y"):
        arrays["y"] = np.broadcast_to(y[None, None, None, :, None], grid_shape)
    if _want("z"):
        arrays["z"] = np.broadcast_to(z[None, None, None, None, :], grid_shape)
    if _want("P"):
        arrays["P"] = np.full(grid_shape, float(laser_power), dtype=np.float32)
    if _want("V_scan"):
        arrays["V_scan"] = np.full(grid_shape, float(v_scan), dtype=np.float32)
    if _want("time_s"):
        arrays["time_s"] = np.broadcast_to(
            selected_times.astype(np.float32)[:, None, None, None, None], grid_shape
        )
    for field_name in selected:
        if not _want(field_name):
            continue
        data = np.asarray(archive[field_name])
        _validate_field_shape(field_name, data.shape, times.shape[0], nx, ny, nz, case_name)
        if not np.isfinite(data).all():
            raise ValueError(
                f"FLOW-3D case `{case_name}` field `{field_name}` contains non-finite values."
            )
        if data.ndim == 5:  # vector field, e.g. Velocity (T, nx, ny, nz, 3)
            arrays[field_name] = np.moveaxis(data[frame_indices], -1, 1).astype(np.float32)
        else:
            arrays[field_name] = data[frame_indices][:, None].astype(np.float32)

    return CanonicalCase(
        arrays=arrays,
        fields=selected,
        x=x,
        y=y,
        z=z,
        times=times,
        frame_indices=frame_indices,
        selected_times=selected_times,
    )


def read_npz_headers(source: str | IO[bytes]) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Read ``member -> (shape, dtype)`` from an NPZ without decompressing payloads."""
    headers: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as member:
                version = npformat.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = npformat.read_array_header_1_0(member)
                else:
                    shape, _, dtype = npformat.read_array_header_2_0(member)
            headers[name[: -len(".npy")]] = (tuple(int(dim) for dim in shape), np.dtype(dtype))
    return headers


def canonical_shapes_from_headers(
    headers: dict[str, tuple[tuple[int, ...], np.dtype]],
    *,
    case_name: str,
    selected_fields: list[str] | None = None,
) -> tuple[dict[str, list[int]], list[str]]:
    """Compute canonical array shapes from NPZ headers alone (no array loads).

    Returns ``(shapes, fields)`` where ``shapes`` maps every canonical original
    array name (channels + fields) to its ``(T, C, nx, ny, nz)`` shape.
    """
    for axis_name in REQUIRED_AXIS_KEYS:
        if axis_name not in headers:
            raise ValueError(f"FLOW-3D case `{case_name}` NPZ is missing required array `{axis_name}`.")
    nx = int(headers["x"][0][0]) if headers["x"][0] else 0
    ny = int(headers["y"][0][0]) if headers["y"][0] else 0
    nz = int(headers["z"][0][0]) if headers["z"][0] else 0
    num_frames = int(headers["time_s"][0][0]) if headers["time_s"][0] else 0
    for axis_name in REQUIRED_AXIS_KEYS:
        if len(headers[axis_name][0]) != 1 or headers[axis_name][0][0] == 0:
            raise ValueError(
                f"FLOW-3D case `{case_name}` NPZ array `{axis_name}` must be a non-empty 1-D array."
            )

    available_fields = [
        name for name in headers
        if not is_metadata_array(name, tuple(headers[name][0]))
    ]
    fields = resolve_selected_fields(available_fields, selected_fields, case_name)

    channel_shape = [num_frames, 1, nx, ny, nz]
    shapes: dict[str, list[int]] = {name: list(channel_shape) for name in CHANNEL_NAMES}
    for field_name in fields:
        shape = headers[field_name][0]
        _validate_field_shape(field_name, shape, num_frames, nx, ny, nz, case_name)
        channels = int(shape[4]) if len(shape) == 5 else 1
        shapes[field_name] = [num_frames, channels, nx, ny, nz]
    return shapes, fields
