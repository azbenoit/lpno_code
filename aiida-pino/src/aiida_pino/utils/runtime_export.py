"""Fast, parallel runtime staging: payload locators, pack-slice IO, pure workers.

Why this module exists, measured on the development machine (2026-08-05):

* Staging a derived view of the 120-member varspeed split pegged one daemon
  worker at 99% CPU for ~6 hours, wedged the daemon (RPC shares the event
  loop), and re-wedged it within a minute of every restart, because the killed
  task is redelivered and resumes the same export. The stuck `.build-*` dirs in
  the runtime cache carry the pids of both wedged workers.
* Two multipliers stacked: `derive_simulation_arrays` read arrays through
  `iter_original_arrays`, which re-opens and re-canonicalizes the *entire*
  payload once per field — twelve fields, twelve full decompress passes of a
  3.3 GB-decompressed member — and the whole thing ran serially in the daemon.
* A prior attempt to parallelise with threads was reverted the same day:
  AiiDA's SQLAlchemy session is not thread-safe. The design here is the one
  that comment asks for — workers that never touch the session. The parent
  (session-bound) collects a *locator* per member; the workers are pure
  numpy/zipfile subprocesses.

The locator is possible because of how this profile actually stores payloads:
`payload.npz` objects live **uncompressed and contiguous** inside
disk-objectstore packs (`get_object_meta`: `size == pack_length`, compression
never applied — the NPZ is already deflated internally). A payload is therefore
a byte range of a pack file, readable by any process with `pread`, and copyable
kernel-side with `sendfile` at 2.7 GB/s (measured; the Python streamed copy
manages 1.0 GB/s and the raw range read 0.5–1.0 GB/s cold).
"""

from __future__ import annotations

import io
import json
import itertools
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


# ------------------------------------------------------------------ locators
def payload_locator(simulation) -> dict[str, Any] | None:
    """Where `payload.npz` physically lives. Parent-side (touches the session).

    Returns `{"kind": "pack"|"loose", "path": str, "offset": int, "size": int}`,
    or None when the object is not directly readable (compressed in the pack,
    remote backend, anything unexpected) — callers then fall back to
    `repository.open`, which is always correct.
    """
    try:
        obj = simulation.base.repository.get_object(simulation.PAYLOAD_FILENAME)
        backend = simulation.base.repository._repository.backend
        container = getattr(backend, "_container", None)
        if container is None or not hasattr(container, "get_object_meta"):
            return None
        meta = container.get_object_meta(obj.key)
        kind = getattr(getattr(meta, "type", None), "value", None)
        if kind == "loose":
            loose = Path(container._get_loose_path_from_hashkey(obj.key))
            if loose.is_file():
                return {"kind": "loose", "path": str(loose), "offset": 0, "size": int(meta.size)}
            return None
        if kind == "packed":
            # Only when stored verbatim: a compressed pack entry is not a slice
            # of the original bytes. `size == pack_length` is that guarantee.
            if getattr(meta, "pack_length", None) != getattr(meta, "size", None):
                return None
            pack = Path(container._folder) / "packs" / str(meta.pack_id)
            if not pack.is_file():
                return None
            return {"kind": "pack", "path": str(pack), "offset": int(meta.pack_offset), "size": int(meta.size)}
    except Exception:  # noqa: BLE001 - locator is an optimisation, never a failure
        return None
    return None


def read_locator_bytes(locator: dict[str, Any]) -> bytes:
    """Read the payload bytes named by a locator. Safe in any process."""
    with open(locator["path"], "rb") as handle:
        handle.seek(locator["offset"])
        remaining = int(locator["size"])
        chunks = []
        while remaining:
            chunk = handle.read(min(1 << 24, remaining))
            if not chunk:
                raise OSError(
                    f"short read staging payload from {locator['path']} @ {locator['offset']}: "
                    f"{remaining} bytes missing — the container may have been repacked underneath us"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
    return b"".join(chunks)


def sendfile_locator(locator: dict[str, Any], destination: Path) -> None:
    """Kernel-side copy of a locator range into `destination` (2.7 GB/s measured)."""
    src = os.open(locator["path"], os.O_RDONLY)
    try:
        dst = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            offset = int(locator["offset"])
            remaining = int(locator["size"])
            while remaining:
                sent = os.sendfile(dst, src, offset, remaining)
                if sent == 0:
                    raise OSError(f"sendfile stalled staging {destination.name}")
                offset += sent
                remaining -= sent
        finally:
            os.close(dst)
    finally:
        os.close(src)


# ------------------------------------------------- the pure transform worker
def apply_derivation(
    array: np.ndarray,
    profile: dict[str, Any],
    moving_frame_slices: list[tuple[int, int]] | None = None,
    *,
    stop: int | None = None,
) -> np.ndarray:
    """One canonical array -> its derived variant. Mirrors derive_simulation_arrays.

    Import here is deliberate: the transform helpers live beside the collection
    class, are pure numpy, and must stay byte-identical between the serial path
    and the workers — one implementation, imported by both.

    ``moving_frame_slices`` is the per-frame beam-following x-window as (start, stop)
    pairs, computed once per simulation by the parent (the worker has no AiiDA session
    and so cannot read the case manifest itself). Required whenever
    ``moving_ref_frame`` is set: the window must follow the beam's real position, not
    the frame index. See `lpno.models.pipeline.frame_transfer`.

    ``stop`` travels with them and for the same reason -- it comes from the manifest,
    which the worker cannot read. It ends the tail where the beam leaves its declared
    path, so a `turn_off_at_end: false` overrun does not pin the window against the +x
    wall. Both are per-simulation, computed once by the parent.
    """
    from aiida_pino.data.pino_dataset_collection import (
        _crop_shift_moving_frame_single,
        _moving_average_last_window_single,
        _resample_spatial,
        _stride_canonical_spatial,
    )

    result = np.asarray(array, dtype=np.float32)
    crop_top = int(profile.get("crop_z_top", 0) or 0)
    if crop_top:
        # FIRST, before super-res/frame/stride: this restricts the DOMAIN, and everything
        # after it should operate on the restricted one. z is the last axis in canonical
        # `(nt, 1, nx, ny, nz)` shape, and the gas sits at high z, so the top layers are
        # the tail. Applied to every array uniformly -- including the broadcast x/y/z
        # coordinates and any derived field -- so nothing can disagree about the extent.
        # A crop that would take the WHOLE extent means there is no gas here to crop,
        # not that the caller asked for something impossible. That distinction became
        # load-bearing when the gas crop became the DEFAULT: a 10-cell crop is exactly
        # the gas on every real deck (surface at z=0, 95 um of headroom at dz=10 um,
        # verified for nz = 40/50/60/80), but it is grid-blind, and applying it to a
        # small array raised on 201 tests whose fixtures are 4 cells deep. Refusing
        # would make the default unusable on any coarse grid; cropping everything would
        # be worse. So it crops NOTHING and says so once.
        if result.shape[-1] <= crop_top:
            print(
                f"[derivation] crop_z_top={crop_top} exceeds the {result.shape[-1]}-cell "
                "z extent; nothing cropped (no gas layer on this grid).",
                flush=True,
            )
            crop_top = 0
        if crop_top:
            result = result[..., : result.shape[-1] - crop_top]
    if int(profile.get("super_res_factor", 1)) > 1:
        result = _resample_spatial(result, int(profile["super_res_factor"]))
    if bool(profile.get("moving_ref_frame", False)):
        result = _crop_shift_moving_frame_single(
            result,
            int(profile.get("moving_frame_steps", 1)),
            super_res_factor=int(profile.get("super_res_factor", 1)),
            slices=(
                [slice(int(a), int(b)) for a, b in moving_frame_slices]
                if moving_frame_slices is not None
                else None
            ),
            stop=stop,
        )
        if int(profile.get("time_avg_window", 1)) > 1:
            result = _moving_average_last_window_single(result, int(profile["time_avg_window"]))
    elif not bool(profile.get("include_time", True)) and result.ndim == 5:
        result = result[-1]
    result = _stride_canonical_spatial(result, profile.get("stride", [1, 1, 1]))
    # `time_frames` LAST, matching `_write_runtime_strided_npz_manifest`'s order (stride,
    # then subsample). This transform was implemented only in that OTHER writer -- the
    # duplicated-builders trap, again: a `time_frames: 60` derivation through this path
    # recorded the intent in the profile and wrote every one of the 627 native frames
    # (2026-08-28). A "successful" training on that payload
    # answers a different question than the collection claims -- the preheat lesson
    # verbatim. Same `frame_indices` helper, so the kept frames are identical across
    # writers. Gated off the moving-frame branch (which already reshapes time) and off
    # the collapsed no-time branch (whose leading axis is channels, not time).
    target_frames = int(profile.get("time_frames") or 0)
    if (
        target_frames
        and bool(profile.get("include_time", True))
        and not bool(profile.get("moving_ref_frame", False))
        and result.ndim >= 4
    ):
        from aiida_pino.data.pino_dataset_collection import frame_indices

        result = result[frame_indices(result.shape[0], target_frames)]
    return result.astype(np.float32, copy=False)


def derive_time_values(
    time_values: np.ndarray | None,
    profile: dict[str, Any],
    *,
    stop: int | None = None,
) -> np.ndarray | None:
    """The time half of derive_simulation_arrays, shared by parent and worker."""
    if time_values is None:
        return None
    time_values = np.asarray(time_values, dtype=np.float32)
    if bool(profile.get("moving_ref_frame", False)):
        steps = int(profile.get("moving_frame_steps", 1))
        # Same cut the arrays get, so `time_s` names the frames actually kept.
        if stop is not None:
            time_values = time_values[: int(stop)]
        if steps > 0:
            time_values = time_values[-steps:]
        if int(profile.get("time_avg_window", 1)) > 1 and time_values.size > 0:
            time_values = np.asarray([time_values[-1]], dtype=np.float32)
    elif not bool(profile.get("include_time", True)):
        time_values = np.asarray([time_values[-1]], dtype=np.float32) if time_values.size else None
    elif int(profile.get("time_frames") or 0):
        # The SAME indices the arrays keep, or `time_s` names instants the frames are not.
        from aiida_pino.data.pino_dataset_collection import frame_indices

        time_values = time_values[
            frame_indices(time_values.shape[0], int(profile["time_frames"]))
        ]
    return time_values


def write_derived_case_npz(
    target_dir: Path,
    *,
    case_name: str,
    derived_arrays: dict,
    time_values,
    raw_axes: dict,
    laser_power: float,
    v_scan: float,
    profile: dict,
    case_manifest: dict | None = None,
) -> dict:
    """Write ONE compressed `<case>.npz` to the flat root; return its manifest fields.

    ONE implementation called by BOTH the pooled worker and the serial path. They were
    briefly two, and `test_worker_matches_the_serial_derivation` caught it within a single
    suite run -- the pooled path emitting a compressed npz while the serial path still wrote
    loose `.npy`. That test exists because this exact divergence has shipped before; this
    time it did not get out.

    COMPRESSED, because `np.save` per array is uncompressed object storage -- measured 58x
    on real members, and material-conditioning channels are `np.full` broadcasts that
    compress to almost nothing.

    FLAT ROOT, because that is what decides WHICH READER runs, and getting it wrong is
    silent: a dir of `<case>.npz` satisfies `is_consolidated_npz` so `NpzDirStore` serves it
    as archives, while under `members/` it stays an arraydir that `ArrayDir*Store` mmaps --
    and `mmap_mode` is IGNORED on a zip, so compressing in place would quietly load whole
    arrays into RAM instead of failing.
    """
    import numpy as _np

    flat_root = target_dir.parent.parent if target_dir.parent.name == "members" else target_dir
    flat_root.mkdir(parents=True, exist_ok=True)
    npz_destination = flat_root / f"{case_name}.npz"
    if npz_destination.exists() and flat_root != target_dir:
        raise ValueError(
            f"Duplicate case name `{case_name}` while staging a derived collection: "
            f"`{npz_destination}` already exists. Either two members share this case name, "
            "or a worker from an earlier, failed attempt is still writing into this build "
            "directory (see `_run_derived_pool` / `_fresh_build_dir`)."
        )

    # Axes are taken FROM the derived shape rather than re-deriving the transform on 1-D
    # arrays. `apply_derivation` works on canonical `(nt, 1, nx, ny, nz)`, the consolidated
    # format wants 1-D axes, and a second implementation of crop-then-stride is exactly the
    # duplicated-builder trap. Interpolating the raw axis onto the derived length cannot
    # disagree with the arrays it describes.
    # 5-D is canonical `(nt, 1, nx, ny, nz)`, but a 4-D `(C, X, Y, Z)` member is legal here
    # too -- `from_simulations` treats a 4-D leading axis as channels. Assuming 5-D skipped
    # the axis block entirely for those, leaving 5-D/4-D coordinate broadcasts in the staged
    # npz where the loader expects 1-D, so `raw_nz` would read the whole grid.
    axes_out: dict = {}
    reference = next((v for v in derived_arrays.values() if _np.asarray(v).ndim == 5), None)
    axis_positions = (("x", 2), ("y", 3), ("z", 4))
    if reference is None:
        reference = next((v for v in derived_arrays.values() if _np.asarray(v).ndim == 4), None)
        axis_positions = (("x", 1), ("y", 2), ("z", 3))
    if reference is not None:
        for axis_name, axis_index in axis_positions:
            source = raw_axes.get(axis_name)
            length = int(_np.asarray(reference).shape[axis_index])
            raw_axis = (
                _np.asarray(source, dtype=_np.float32).ravel() if source is not None else None
            )
            if raw_axis is not None and raw_axis.size == length:
                # Untransformed on this axis: the raw axis is exact.
                axes_out[axis_name] = raw_axis
                continue
            # Any length change means a transform moved this axis, and the coordinate
            # BROADCAST went through the SAME `apply_derivation` as the data -- it cannot
            # disagree with the arrays it describes. RECONSTRUCTING the axis from the raw
            # span can: the moving-frame x-crop keeps a beam-following window with the
            # original spacing, and the old full-span linspace here produced an absolute
            # [-4.45e-4, +4.45e-4] axis at dx 14.8 um where the true derived axis is
            # beam-relative [-1.73e-4, +4.27e-4] at dx 10.0 um (one preheat case, arrays
            # verified bit-exact -- only x was wrong). The same preference
            # is also exactly right for a stride (the broadcast was strided with the data,
            # where a linspace re-spans the endpoints) and for the z crop.
            broadcast = derived_arrays.get(axis_name)
            if broadcast is not None:
                broadcast = _np.asarray(broadcast)
                if broadcast.ndim in (4, 5):
                    picker = [0] * broadcast.ndim
                    picker[axis_index] = slice(None)
                    axes_out[axis_name] = _np.asarray(broadcast[tuple(picker)], dtype=_np.float32)
                    continue
                if raw_axis is None:
                    raw_axis = _np.asarray(broadcast, dtype=_np.float32).ravel()
            if raw_axis is None:
                continue
            if raw_axis.size == length:
                axes_out[axis_name] = raw_axis
            elif raw_axis.size >= 2 and length >= 1:
                # No transformed broadcast shipped: reconstruct. `crop_z_top` removes the
                # TOP of z, so the kept span is the head of the raw axis, not a rescaling
                # of its full range.
                crop = int((profile or {}).get("crop_z_top", 0) or 0)
                if axis_name == "z" and crop:
                    span_hi = raw_axis[max(raw_axis.size - crop - 1, 0)]
                else:
                    span_hi = raw_axis[-1]
                axes_out[axis_name] = _np.linspace(raw_axis[0], span_hi, length, dtype=_np.float32)
            else:
                axes_out[axis_name] = raw_axis[:length]

    # COORDINATES ARE OVERWRITTEN, EVERYTHING ELSE IS NOT, and the asymmetry is a format
    # requirement rather than a preference.
    #
    # Canonicalization carries `x`, `y`, `z` and `time_s` as 5-D BROADCASTS -- verified on
    # real staged output, where each came back (247, 1, 90, 40, 30). The consolidated NPZ
    # format wants them as 1-D axes: the loader takes `raw_nz` from `z.size`, which on a
    # broadcast would read 26.6 million cells instead of 30. Nothing is lost, since a
    # broadcast coordinate is recoverable from its axis.
    #
    # Every OTHER name is left alone. Assignment there silently replaced a (1, 4, 5, 3)
    # `power_w` field with a (1,) scalar; a collection may legitimately carry a field by
    # that name, and clobbering corrupts the data while leaving a file that loads clean.
    npz_payload = {k: _np.asarray(v) for k, v in derived_arrays.items()}
    npz_payload.update(axes_out)
    if time_values is not None:
        npz_payload["time_s"] = _np.asarray(time_values, dtype=_np.float32)
    # The three `_SIGNATURE_KEYS`; without them the staged dir is not recognised as
    # consolidated NPZ and the loader falls back to the arraydir reader.
    #
    # `setdefault`, NOT assignment: a derived collection can legitimately carry a FIELD
    # named `power_w`, and overwriting it with the scalar silently replaced a
    # (1, 4, 5, 3) array with a (1,) one -- caught by the worker-vs-serial identity test,
    # which compares every array rather than just the shapes it expects. The signature
    # check is presence-only (`_SIGNATURE_KEYS.issubset(keys)`), so the derived array
    # satisfies it just as well and the real data must win.
    for name in list(derived_arrays):
        value = _np.asarray(npz_payload[name])
        if value.ndim == 5 and value.shape[1] == 1:
            # Canonical (nt, 1, nx, ny, nz) -> native flat-npz (nt, nx, ny, nz). The
            # channel axis is an internal convention of the transform pipeline; the
            # stored format never had it, and the reader indexes positionally.
            npz_payload[name] = value[:, 0]
    npz_payload.setdefault("power_w", _np.asarray(float(laser_power), dtype=_np.float32))
    npz_payload.setdefault("scan_speed_m_per_s", _np.asarray(float(v_scan), dtype=_np.float32))
    npz_payload.setdefault("case_name", _np.asarray(str(case_name)))
    _np.savez_compressed(npz_destination, **npz_payload)

    sidecar_filename = None
    if isinstance(case_manifest, dict) and case_manifest:
        sidecar_filename = f"{case_name}.manifest.json"
        (flat_root / sidecar_filename).write_text(
            json.dumps(case_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    return {
        "npz_file": os.path.relpath(npz_destination, target_dir),
        "case_manifest_file": (
            os.path.relpath(flat_root / sidecar_filename, target_dir) if sidecar_filename else None
        ),
        "case_name": str(case_name),
        # Shapes of what was ACTUALLY written, not of what was handed in. The coordinates
        # are stored as 1-D axes, so recording the pre-write broadcast shape would make the
        # manifest describe a file that does not exist -- caught by the manifest-shape test,
        # which reads every recorded shape back out of the staged npz.
        "array_shapes": {k: list(_np.asarray(npz_payload[k]).shape) for k in derived_arrays},
    }


def member_frame_window(task: dict[str, Any], profile: dict[str, Any]) -> tuple[int, int] | None:
    """Frames [start, stop) a moving-frame derivation keeps, when only those need reading.

    `apply_derivation` keeps frames [stop - moving_frame_steps, stop) of every array of a
    moving-frame derivation, the steady window among them. Canonicalizing every frame first
    inflated each field whole, three float32 copies at a time: about 8.5 GB for the Velocity
    of the 798-frame case, which OOM-killed a public reproduction's evaluation in a 32 GB job
    (2026-10-08). None (read every frame) when the task opts out (`frame_window: False`),
    `AIIDA_PINO_EXPORT_FULL_FRAMES=1` is set, or the window is not known.
    """
    if task.get("frame_window") is False or os.environ.get("AIIDA_PINO_EXPORT_FULL_FRAMES", "").strip() in ("1", "true", "yes"):
        return None
    stop = task.get("moving_frame_stop")
    steps = int(profile.get("moving_frame_steps", 1) or 1)
    if not profile.get("moving_ref_frame") or stop is None or int(stop) < steps:
        return None
    return int(stop) - steps, int(stop)


def _read_member_frames(archive, name: str, start: int, stop: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Frames [start, stop) of one NPZ member, decompressing past the others without keeping them."""
    import numpy.lib.format as npformat

    with archive.zip.open(f"{name}.npy") as member:
        version = npformat.read_magic(member)
        shape, fortran, dtype = npformat._read_array_header(member, version)
        if fortran:
            raise ValueError(f"{name}.npy is Fortran-ordered; its frames are not contiguous")
        frame = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        member.seek(member.tell() + start * frame)  # forward seek: decompress and discard
        data = member.read((stop - start) * frame)
    return np.frombuffer(data, dtype=dtype).reshape((stop - start, *shape[1:])), tuple(shape)


def canonicalize_frame_window(archive, start: int, stop: int, *, case_name: str, laser_power: float,
                              v_scan: float, selected_fields=None):
    """`canonicalize_flow3d_case_arrays` for frames [start, stop) only: the same arrays, cut.

    Handing these to `apply_derivation` with `stop=None` gives the bits the full read gives
    with `stop` (tests/test_frame_window_export.py). Non-finite values are checked in the kept
    frames only.
    """
    from aiida_pino.utils import flow3d_npz as fz

    for axis_name in fz.REQUIRED_AXIS_KEYS:
        if axis_name not in archive.files:
            raise ValueError(f"FLOW-3D case `{case_name}` NPZ is missing required array `{axis_name}`.")
    x = np.asarray(archive["x"], dtype=np.float32)
    y = np.asarray(archive["y"], dtype=np.float32)
    z = np.asarray(archive["z"], dtype=np.float32)
    times = np.asarray(archive["time_s"], dtype=np.float64)
    for axis, axis_name in ((x, "x"), (y, "y"), (z, "z"), (times, "time_s")):
        fz.require_monotonic(axis, axis_name, case_name)
    available = [n for n in archive.files if not fz.is_metadata_array(n, fz.member_shape(archive, n))]
    selected = fz.resolve_selected_fields(available, selected_fields, case_name)
    kept = list(range(start, stop))
    kept_times = times[kept]
    nx, ny, nz = int(x.size), int(y.size), int(z.size)
    grid = (len(kept), 1, nx, ny, nz)
    arrays: dict[str, np.ndarray] = {
        "x": np.broadcast_to(x[None, None, :, None, None], grid),
        "y": np.broadcast_to(y[None, None, None, :, None], grid),
        "z": np.broadcast_to(z[None, None, None, None, :], grid),
        "P": np.full(grid, float(laser_power), dtype=np.float32),
        "V_scan": np.full(grid, float(v_scan), dtype=np.float32),
        "time_s": np.broadcast_to(kept_times.astype(np.float32)[:, None, None, None, None], grid),
    }
    for field in selected:
        data, full_shape = _read_member_frames(archive, field, start, stop)
        fz._validate_field_shape(field, full_shape, times.shape[0], nx, ny, nz, case_name)
        if not np.isfinite(data).all():
            raise ValueError(f"FLOW-3D case `{case_name}` field `{field}` contains non-finite values.")
        if data.ndim == 5:  # vector field (T, nx, ny, nz, 3)
            arrays[field] = np.moveaxis(data, -1, 1).astype(np.float32)
        else:
            arrays[field] = data[:, None].astype(np.float32)
    return fz.CanonicalCase(arrays=arrays, fields=selected, x=x, y=y, z=z, times=times,
                            frame_indices=kept, selected_times=kept_times)


def derived_member_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Decompress, canonicalize ONCE, transform, write. No AiiDA, no session.

    `task` carries everything the session knew: the payload locator (or a
    pre-staged file path), the canonicalization scalars, and the derivation
    profile. Runs in a subprocess; must stay importable at module level.
    """
    from aiida_pino.utils.flow3d_npz import canonicalize_flow3d_case_arrays

    target_dir = Path(task["target_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    if task.get("staged_path"):
        payload = Path(task["staged_path"]).read_bytes()
    else:
        payload = read_locator_bytes(task["locator"])

    profile = dict(task["profile"])
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        # ONE canonicalization pass for every field. The serial path went
        # through get_array per field, re-reading and re-canonicalizing the
        # whole payload each time: twelve fields, twelve passes.
        #
        # Only the requested fields that are archive MEMBERS are inflated: a training's
        # list also names V_scan and P, which canonicalize synthesises (they are in the
        # canonical name list but in no archive), and `selected_fields` raises on a name
        # the archive lacks. Filtering here, where the archive is open, is what the
        # strided path does; filtering upstream against the canonical names let V_scan/P
        # through and failed every member of a 61-member warm (self-improve, 2026-09-14).
        # An empty intersection means None: every field, the pre-trim behaviour.
        requested = [f for f in (task.get("requested_fields") or []) if f in archive.files] or None
        window = member_frame_window(task, profile)
        if window is None:
            case = canonicalize_flow3d_case_arrays(
                archive,
                case_name=task["case_name"],
                laser_power=float(task["laser_power"]),
                v_scan=float(task["v_scan"]),
                selected_fields=requested,
            )
        else:
            case = canonicalize_frame_window(
                archive, *window,
                case_name=task["case_name"],
                laser_power=float(task["laser_power"]),
                v_scan=float(task["v_scan"]),
                selected_fields=requested,
            )
        # Kept while the archive is open; the axes the derived arrays are described by.
        raw_axes = {
            name: np.asarray(archive[name]) for name in ("x", "y", "z") if name in archive.files
        }
        # Built in the LAB frame from the RAW 1-D axes, and built HERE -- before
        # the crop and the average that `apply_derivation` applies below. That
        # order is the physics argument, and it must not invert in this second
        # caller: rebuilt after the transform you would get an instantaneous
        # lab-frame field against a window-averaged ground truth, which is a
        # wrong-NUMBER bug rather than a crash. Same function as the serial path.
        extra_fields: dict[str, np.ndarray] = {}
        extra_names = list(profile.get("derived_fields") or [])
        if extra_names:
            from aiida_pino.data.pino_dataset_collection import extra_fields_from_axes

            if not task.get("case_manifest"):
                raise ValueError(
                    f"derived_fields {extra_names} requested for {task['case_name']!r} but the "
                    f"parent shipped no case_manifest; the worker cannot read it (no session)."
                )
            extra_fields = extra_fields_from_axes(
                extra_names,
                dict(task["case_manifest"]),
                np.asarray(archive["x"]),
                np.asarray(archive["y"]),
                np.asarray(archive["z"]),
                np.asarray(archive["time_s"]),
                label=str(task["case_name"]),
            )
    del payload

    shapes: dict[str, list[int]] = {}
    derived_arrays: dict[str, np.ndarray] = {}
    for name, array in itertools.chain(case.arrays.items(), extra_fields.items()):
        # Windowed canonical arrays are already cut to [stop - steps, stop); the derived fields
        # are built over every frame (lab frame, before the cut) and are cut here as before.
        already_cut = window is not None and name in case.arrays
        derived = apply_derivation(
            array, profile, task.get("moving_frame_slices"),
            stop=None if already_cut else task.get("moving_frame_stop"),
        )
        derived_arrays[name] = derived
        shapes[name] = list(derived.shape)
    time_values = derive_time_values(
        np.asarray(task["time_values_s"], dtype=np.float32) if task.get("time_values_s") is not None else None,
        profile,
        stop=task.get("moving_frame_stop"),
    )

    if bool(profile.get("moving_ref_frame", False)):
        # ARRAYDIR layout for moving-frame derivations, exactly the pre-b616eaa writer.
        # The flat consolidated npz routes to pino's NpzDirStore, whose lazy path refuses
        # `moving_ref_frame=true` BY DESIGN -- it cannot know the derivation is already
        # baked. Every steady eval that ever worked (61779 -> 7.4 K, 68119) was served by
        # `ArrayDirCollectionStore` from this layout; switching steady members to the flat
        # npz regressed both routes into a steady eval to the refusing store (four
        # steady evals failed this way). Steady members are small (the time average collapses the frame axis), so
        # the compressed-npz disk motive does not apply here.
        target = Path(task["target_dir"])
        array_entries = []
        shapes: dict[str, list[int]] = {}
        for name, value in derived_arrays.items():
            filename = f"{name}.npy"
            np.save(target / filename, value)
            shapes[name] = list(np.asarray(value).shape)
            array_entries.append({
                "original_name": name,
                "stored_name": name,
                "filename": filename,
                "shape": shapes[name],
                "dtype": str(np.asarray(value).dtype),
            })
        time_values_filename = None
        if time_values is not None:
            time_values_filename = "time_values_s.npy"
            np.save(target / time_values_filename, time_values)
        return {
            "index": task["index"],
            "layout": "arraydir",
            "array_entries": array_entries,
            "array_shapes": shapes,
            "time_values_filename": time_values_filename,
            "time_values": time_values.tolist() if time_values is not None else None,
            "num_timesteps": int(time_values.shape[0]) if time_values is not None else None,
            "case_manifest_file": None,
        }

    written = write_derived_case_npz(
        target_dir,
        case_name=str(task["case_name"]),
        derived_arrays=derived_arrays,
        time_values=time_values,
        raw_axes=raw_axes,
        laser_power=float(task["laser_power"]),
        v_scan=float(task["v_scan"]),
        profile=profile,
        case_manifest=task.get("case_manifest"),
    )

    return {
        "index": task["index"],
        "layout": "npz_file",
        "array_entries": [
            {
                "original_name": name,
                "stored_name": name,
                "filename": f"{name}.npy",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in derived_arrays.items()
        ],
        # From the WRITER, not the pre-write loop: the coordinates are stored as 1-D axes,
        # so `shapes` describes arrays as handed in rather than as written.
        "array_shapes": written["array_shapes"],
        "time_values_filename": None,
        "time_values": time_values.tolist() if time_values is not None else None,
        "npz_file": written["npz_file"],
        "case_manifest_file": written["case_manifest_file"],
        "case_name": written["case_name"],
        "num_timesteps": int(time_values.shape[0]) if time_values is not None else None,
    }


def payload_stage_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Copy one untransformed payload into place, kernel-side where possible."""
    destination = Path(task["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    sendfile_locator(task["locator"], destination)
    return {"index": task["index"], "destination": str(destination)}
