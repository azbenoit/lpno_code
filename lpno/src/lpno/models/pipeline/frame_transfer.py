"""One beam-following window, and transfers between the three frame treatments.

THE THREE TREATMENTS. A laser-melt run can be presented to a model in three ways, and
this project trains an arm in each so the architectures can be compared on all of them:

- ``lab``        the raw domain, one sample per stored frame, beam moving through it;
- ``moving_ref`` an x-window that follows the beam, one sample per frame;
- ``steady``     the same window, time-averaged over ``avg_window`` frames -> one sample.

Their metrics are NOT comparable as they stand: the domains differ in size and the
steady field has had its peaks averaged away (measured on nine shared cases, true T max
2333 K steady against 3685 K per-frame). The comparison that IS meaningful is to take a
model's predictions in its own frame and push them into a common one -- lab predictions
into the moving frame, moving-frame predictions into the steady frame -- and rescore
there. That is what `transfer` is for.

WHY THIS MODULE OWNS THE WINDOW. The window arithmetic existed twice, and the two copies
disagreed:

- ``NpzDirStore._moving_window`` (the lazy reader, ``PINO_LAZY_MOVING_REF``) crops around
  the beam's ACTUAL per-frame cell, read from the case manifest. Correct.
- ``npz_datasource._crop_shift_moving_frame`` (the steady/time-averaged derivation) shifts
  frame ``t`` by exactly ``super_res_factor * t`` cells, which follows the beam only when
  ``V_scan * dt == dx``.

Measured on the spkh123 train split (96 cases, 2026-08-20): the beam advances
**0.051 to 0.482 cells per frame, median 0.200**, against the assumed 1.000 -- 78 % mean
deviation, 95 % worst. Over a 30-frame window the index shift slides 29 cells while the
beam moves 1.5 to 14, so the beam drifts 21-27 cells (4.2-5.4 beam radii) inside the
window meant to hold it still, and the average smears the melt pool across that drift.
The assumption was probably true of the original straight-path data, whose export cadence
was one cell per frame, which is why it survived this long.

So there is one implementation here and every path calls it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "FrameSpec",
    "beam_centres_from_manifest",
    "beam_following_x_slice",
    "crop_batch_to_moving_ref",
    "legacy_window_split",
    "to_moving_ref",
    "to_steady",
    "transfer",
]

# `PINO_LAZY_MOVING_REF="32,16"`: two thirds of the window trails the beam, because the
# melt pool and its cooling tail sit behind it while only the preheat sits ahead.
_BEHIND_FRACTION = 2.0 / 3.0


#: (frame count, stop) pairs already announced, so a 150-case load prints once per shape.
_STOP_ANNOUNCED: set = set()


def laser_on_frame_stop(manifest, t_seconds) -> int:
    """Exclusive index of the last frame on which the beam is still on its DECLARED path.

    Every steady/moving-frame consumer takes "the last N frames"; this says where that
    tail ends. It is ``T`` (the export end) unless the manifest records a laser that keeps
    MOVING past its declared path end -- `turn_off_at_end: false` with a scan that finishes
    before the run does (`laser_on_past_declared_time`, or the pre-2026-08-15 form
    `laser_turn_off_at_end: false` with `scan_time_s` < the last frame). Then it is the
    number of frames with ``t <= scan_time_s``.

    Why the tail is wrong there: on the anchorbox anchor replicates the overrun carries the
    beam to ix 86-90 of a 90-cell domain, `beam_following_x_slice` clamps the 61-wide
    window at start 29, and the beam sits at window index 57-61 instead of the nominal 41
    -- a placement no training case has (their paths end at ix 72, window start 31, never
    clamped). Model 83645 scored substrate MAE 96.5 K / fl IoU 0.370 there against 18.4 K /
    0.886 on the 30 laser-on frames before the path end (Materials variations, 2026-09-01):
    a ~10x penalty that is entirely which frames the window lands on. Those frames are also
    where the pool meets the +x wall, an end-wall artefact the solver batch already records.

    A stationary dwell keeps ``T``: the laser stays on past its declared time but does not
    move, so its late frames ARE the steady state and nothing approaches a wall. A time
    axis already cut at the path end also keeps ``T`` (no overrun frame present), which is
    what makes a truncated derived dataset and this selector agree rather than compound.
    """
    t = np.asarray(t_seconds, dtype=np.float64).reshape(-1)
    T = int(t.size)
    derived = (manifest or {}).get("derived") if isinstance(manifest, dict) else None
    if not isinstance(derived, dict) or T == 0:
        return T
    keeps_going = bool(derived.get("laser_on_past_declared_time")) or (
        derived.get("laser_turn_off_at_end") is False
    )
    if not keeps_going:
        return T
    scan_time = derived.get("scan_time_s")
    if scan_time is None:
        return T
    length = derived.get("scan_path_length_m")
    if length is None:
        xy = derived.get("laser_path_xy_m")
        try:
            xy = np.asarray(xy, dtype=np.float64)
            length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if xy.ndim == 2 and xy.shape[0] >= 2 else 0.0
        except (TypeError, ValueError):
            length = 0.0
    if not float(length) > 0.0:
        return T   # dwell: on past its time, not moving
    # A frame is past the path end only if it is more than HALF a frame interval beyond
    # scan_time. Frame times are quantised and the export's last frame sits at
    # finish_time, which for a `finish_time == scan_time` case exceeds scan_time by a
    # rounding error: a relative tolerance called that an overrun and would have shifted
    # the core straight-path steady target (1389 family, 69 + 52 trained models on its
    # train/val splits) by one frame for 82 of 83 cases -- and announced a fake overrun on
    # every load. Measured 2026-09-02: 161 of 168 (T, stop) shapes fleet-wide were exactly
    # one frame; the real overruns are 28-77 frames. Half a frame keeps the former on-path
    # and cannot touch the latter.
    half_frame = 0.5 * float(np.median(np.diff(t))) if T >= 2 else 0.0
    stop = int(np.count_nonzero(t <= float(scan_time) + half_frame))
    if stop >= T:
        return T   # no overrun frame in this axis
    if stop == 0:
        raise ValueError(
            f"laser_on_frame_stop: no frame at or before scan_time_s={scan_time!r} "
            f"(first frame t={t[0]!r}); the time axis and the manifest disagree."
        )
    key = (T, stop)
    if key not in _STOP_ANNOUNCED:
        _STOP_ANNOUNCED.add(key)
        print(
            f"[steady-window] laser runs {T - stop} of {T} frames past its declared path "
            f"end (scan_time_s={float(scan_time):.6g}); the steady/moving-frame tail ends "
            f"at frame {stop}, not {T}, so the window is not pinned against the +x wall.",
            flush=True,
        )
    return stop


def _tail_end(T: int, stop) -> int:
    """Where "the last N frames" end: ``T`` or a `laser_on_frame_stop` result."""
    end = int(T) if stop is None else int(stop)
    if not 1 <= end <= int(T):
        raise ValueError(f"frame stop must be in [1, {T}], got {stop!r}.")
    return end


#: Window geometries (nx, behind, ahead, edge sign) whose clamp has been announced.
_CLAMP_ANNOUNCED: set = set()


def beam_following_x_slice(
    *, beam_cx: float, x0: float, dx: float, nx: int, behind: int, ahead: int
) -> slice:
    """The x-window of constant width ``behind + ahead`` that follows the beam.

    ``beam_cx`` is the beam centre in the same units as ``x0``/``dx``. The window starts
    ``behind`` cells before the beam's cell and is clamped inside ``[0, nx]``, so every
    frame yields the same shape (which is what batching needs) even when the beam is near
    an edge. Clamping trades beam-centring for a constant shape at the two ends of the
    track; the alternative is a ragged batch.
    """
    width = int(behind) + int(ahead)
    if width < 2:
        raise ValueError(f"window width must be >= 2 cells, got {width}.")
    if width > int(nx):
        raise ValueError(f"window width {width} exceeds the domain's {nx} cells.")
    if float(dx) == 0.0:
        # Almost always a broadcast x FIELD flattened instead of the x line: every
        # element of the first row is x[0], so the spacing reads zero.
        raise ValueError(
            "x spacing is zero. Pass the x coordinate LINE, not a flattened broadcast "
            "field (take the axis -3 slice of an (..., X, Y, Z) array)."
        )
    ix = int(np.clip(np.round((float(beam_cx) - float(x0)) / float(dx)), 0, int(nx) - 1))
    ix0 = int(np.clip(ix - int(behind), 0, int(nx) - width))
    displaced = ix0 - (ix - int(behind))
    if displaced != 0:
        # Say so. A clamped window still has the right SHAPE, but the beam is no longer
        # at index `behind` -- it sits `displaced` cells further along -- and nothing
        # downstream can tell unless it reads x/beam_x. On the 15 anchorbox anchor
        # replicates (2026-09-01) the laser overran to ix 86-90 of a 90-cell domain, so
        # a 41+20 window pinned at start 29 put the beam at index 57-61 instead of 41;
        # the derived steady target read as "cavity 5-12 cells AHEAD of the beam", was
        # diagnosed as a frozen beam track, and survived that fix byte-identical because
        # the clamp -- not the track -- decided the window. Once per window geometry
        # per process, so a 345-frame export does not print 345 lines.
        key = (int(nx), int(behind), int(ahead), int(np.sign(displaced)))
        if key not in _CLAMP_ANNOUNCED:
            _CLAMP_ANNOUNCED.add(key)
            edge = "start" if displaced > 0 else "end"
            print(
                f"[moving-frame] beam-following window clamped at the domain {edge}: beam at "
                f"ix {ix} of {nx}, window {behind}+{ahead} would start at {ix - int(behind)}, "
                f"pinned to {ix0}. The beam sits at window index {ix - ix0}, not {behind}; "
                f"locate it from x/beam_x, not from the nominal split.",
                flush=True,
            )
    return slice(ix0, ix0 + width)


def legacy_window_split(width: int) -> tuple[int, int]:
    """Split ``width`` into (behind, ahead) on the 2:1 convention the lazy path uses.

    Lets a caller that only knows a legacy window WIDTH -- e.g. the
    ``nx - (Nsteps - 1) * super_res_factor`` the old index-shift crop produced -- keep
    that exact grid shape while gaining beam-following behaviour, so existing configs and
    latent lattices sized against the old grid still fit.
    """
    width = int(width)
    behind = int(round(width * _BEHIND_FRACTION))
    behind = max(1, min(width - 1, behind))
    return behind, width - behind


@dataclass(frozen=True)
class FrameSpec:
    """Which of the three treatments an array is in.

    ``behind``/``ahead`` describe the window for ``moving_ref`` and ``steady``;
    ``avg_window`` is the number of frames ``steady`` averages.
    """

    kind: str
    behind: int = 0
    ahead: int = 0
    avg_window: int = 1

    def __post_init__(self) -> None:
        if self.kind not in ("lab", "moving_ref", "steady"):
            raise ValueError(
                f"FrameSpec.kind must be 'lab', 'moving_ref' or 'steady', got {self.kind!r}."
            )
        if self.kind != "lab" and self.width < 2:
            raise ValueError(f"{self.kind} needs a window of >= 2 cells, got {self.width}.")
        if self.kind == "steady" and self.avg_window < 1:
            raise ValueError(f"steady needs avg_window >= 1, got {self.avg_window}.")

    @property
    def width(self) -> int:
        return int(self.behind) + int(self.ahead)


def _check_6d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 6:
        raise ValueError(f"expected a 6D (B, T, C, X, Y, Z) array, got shape {a.shape}.")
    return a


def to_moving_ref(
    arr: np.ndarray,
    *,
    beam_cx: Sequence[float] | np.ndarray,
    x0: float,
    dx: float,
    behind: int,
    ahead: int,
) -> np.ndarray:
    """Crop ``(B, T, C, X, Y, Z)`` to the per-frame beam-following window.

    ``beam_cx`` is the beam centre per frame: either ``(T,)`` shared by every case, or
    ``(B, T)`` when the batch holds different cases (which it does whenever the scan speed
    varies, so the shared form is only for a single case or a synthetic test).
    """
    a = _check_6d(arr)
    B, T, C, nx = a.shape[0], a.shape[1], a.shape[2], a.shape[3]
    cx = np.asarray(beam_cx, dtype=np.float64)
    if cx.ndim == 1:
        cx = np.broadcast_to(cx.reshape(1, -1), (B, T))
    if cx.shape != (B, T):
        raise ValueError(f"beam_cx must be (T,) or (B, T) = {(B, T)}, got {cx.shape}.")
    out = np.empty((B, T, C, int(behind) + int(ahead), a.shape[4], a.shape[5]), dtype=a.dtype)
    for b in range(B):
        for t in range(T):
            sl = beam_following_x_slice(
                beam_cx=float(cx[b, t]), x0=x0, dx=dx, nx=nx, behind=behind, ahead=ahead
            )
            out[b, t] = a[b, t, :, sl]
    return out


def to_steady(arr: np.ndarray, avg_window: int, stop: int | None = None) -> np.ndarray:
    """Average the LAST ``avg_window`` frames into a single frame, keeping the time axis.

    Returns ``(B, 1, C, X, Y, Z)``. Averaging the tail rather than the head mirrors the
    derivation this replaces, which took the last entry of a sliding mean: the quasi-steady
    state is what the END of the track has settled into, not the transient at its start.
    ``stop`` (from `laser_on_frame_stop`) moves that end off an overrun tail.
    """
    a = _check_6d(arr)
    w = int(avg_window)
    end = _tail_end(a.shape[1], stop)
    if not 1 <= w <= end:
        raise ValueError(f"avg_window must be in [1, {end}], got {w}.")
    return a[:, end - w:end].mean(axis=1, dtype=np.float64, keepdims=True).astype(a.dtype)


def transfer(
    arr: np.ndarray,
    source: FrameSpec,
    target: FrameSpec,
    *,
    beam_cx: Optional[Sequence[float] | np.ndarray] = None,
    x0: Optional[float] = None,
    dx: Optional[float] = None,
) -> np.ndarray:
    """Move ``arr`` from ``source`` to ``target``, or refuse when information is missing.

    Allowed: lab -> {lab, moving_ref, steady}, moving_ref -> {moving_ref, steady},
    steady -> steady. The reverse directions are refused rather than approximated: a crop
    discards the domain outside the window and an average discards the frames, and neither
    is recoverable. That refusal is the point -- it is what stops a comparison table from
    quietly containing a number that was invented rather than measured.
    """
    if source.kind == target.kind and source == target:
        return np.asarray(arr)

    if source.kind == "steady" and target.kind != "steady":
        raise ValueError(
            f"a steady field cannot be un-averaged back to {target.kind!r}: the individual "
            "frames are gone. Score in the steady frame instead."
        )
    if source.kind == "moving_ref" and target.kind == "lab":
        raise ValueError(
            "a moving-frame field cannot be un-cropped back to the lab frame: everything "
            "outside the window was discarded. Push the lab predictions into the moving "
            "frame instead, which is the direction that loses nothing."
        )

    if source.kind == "lab":
        if beam_cx is None or x0 is None or dx is None:
            raise ValueError(
                "lab -> moving frame needs the beam track: pass beam_cx, x0 and dx."
            )
        arr = to_moving_ref(
            arr, beam_cx=beam_cx, x0=x0, dx=dx, behind=target.behind, ahead=target.ahead
        )
    elif source.kind == "moving_ref":
        if (source.behind, source.ahead) != (target.behind, target.ahead):
            raise ValueError(
                f"moving-frame window mismatch: source is {source.behind}+{source.ahead} "
                f"cells, target is {target.behind}+{target.ahead}. Re-crop from the lab "
                "frame rather than resampling one window into another."
            )

    if target.kind == "steady":
        arr = to_steady(arr, target.avg_window)
    return arr


def crop_batch_to_moving_ref(
    arrays: dict,
    invar: dict,
    *,
    x_key: str = "x",
    beam_key: str = "beam_x",
    behind: int,
    ahead: int,
) -> tuple[dict, dict]:
    """Crop one eval batch of ``(B, C, X, Y, Z)`` arrays into the beam-following window.

    Evaluation runs with time as the batch axis, so ``B`` is a set of frames each with its
    own beam position, and the window is chosen per frame.

    Both dicts are cropped -- predictions/truth AND the input channels -- because every
    downstream metric reads the inputs: the beam box comes from ``beam_dist``, the
    substrate/surface masks from ``z``, the melt geometry from ``x``/``y``. Cropping them
    together is what makes the whole existing metric bundle report in the target frame
    without a second implementation of any of it.

    ``x`` is rewritten beam-relative, matching what the lazy reader does for a natively
    moving-frame dataset, so a transferred lab prediction and a native moving-frame one are
    scored on the same coordinates rather than merely the same shape.

    Returns ``(arrays, invar)``, both new dicts; the inputs are not modified.
    """
    if beam_key not in invar:
        raise ValueError(
            f"cropping to the moving frame needs the beam position channel {beam_key!r} in "
            f"the eval inputs; got {sorted(invar)}. Add it to the model's input keys, or "
            "score this model in its native frame."
        )
    if x_key not in invar:
        raise ValueError(
            f"cropping to the moving frame needs the coordinate channel {x_key!r} in the "
            f"eval inputs; got {sorted(invar)}."
        )

    x_field = np.asarray(invar[x_key])
    beam_field = np.asarray(invar[beam_key])
    nb, nx = x_field.shape[0], x_field.shape[-3]
    x_line = x_field[0, 0, :, 0, 0].astype(np.float64)
    if x_line.size < 2:
        raise ValueError(f"{x_key!r} must span at least 2 cells to give a spacing.")
    x0 = float(x_line[0])
    dx = float(x_line[1] - x_line[0])

    slices = []
    for b in range(nb):
        cx = float(np.asarray(beam_field[b]).reshape(-1)[0])
        slices.append(
            beam_following_x_slice(
                beam_cx=cx, x0=x0, dx=dx, nx=nx, behind=behind, ahead=ahead
            )
        )

    def _crop(d: dict) -> dict:
        out = {}
        for name, arr in d.items():
            a = np.asarray(arr)
            if a.ndim < 3 or a.shape[-3] != nx:
                out[name] = a          # not on the cropped axis; pass through untouched
                continue
            out[name] = np.stack([a[b][..., slices[b], :, :] for b in range(nb)], axis=0)
        return out

    out_arrays = _crop(arrays)
    out_invar = _crop(invar)
    if x_key in out_invar:
        xr = out_invar[x_key].astype(np.float64)
        for b in range(nb):
            xr[b] -= float(np.asarray(beam_field[b]).reshape(-1)[0])
        out_invar[x_key] = xr.astype(np.asarray(invar[x_key]).dtype)
    return out_arrays, out_invar


def beam_centres_from_manifest(manifest: dict, t_seconds) -> np.ndarray:
    """Per-frame beam centre ``(T, 2)`` in metres, from a FLOW-3D case manifest.

    The public door onto the beam track, so consumers outside this package -- aiida-pino's
    dataset derivation in particular -- follow the beam through the same code the loaders
    use instead of keeping a fourth copy of the window arithmetic. There were three.
    """
    from ...custom_utils.npz_datasource import _beam_path_over_time

    centres, _directions, _speeds = _beam_path_over_time(
        manifest, np.asarray(t_seconds, dtype=np.float64)
    )
    return np.asarray(centres, dtype=np.float64)


def beam_following_slices(
    centres, x_coords, *, steps: int, super_res_factor: int = 1, stop: int | None = None
) -> list[slice]:
    """The per-frame x-slices for the LAST ``steps`` frames, at the legacy window width.

    Width is ``nx - (steps - 1) * super_res_factor``, the grid the index-shift crop used to
    produce, split 2:1 behind:ahead. Same shape as before, correct position. ``stop``
    (from `laser_on_frame_stop`) ends the tail before an overrun; the frames it selects
    are the ones the caller must crop, so pass the same value to the array cut.
    """
    x = np.asarray(x_coords, dtype=np.float64)
    nx = x.size
    if nx < 2:
        raise ValueError("need at least 2 x cells to derive a spacing.")
    width = nx - (int(steps) - 1) * int(super_res_factor)
    if width <= 0:
        raise ValueError(
            f"moving_frame_steps={steps} with super_res_factor={super_res_factor} "
            f"exceeds x resolution {nx}."
        )
    behind, ahead = legacy_window_split(width)
    c = np.asarray(centres, dtype=np.float64)
    end = _tail_end(c.shape[0], stop)
    if end < int(steps):
        raise ValueError(
            f"beam track has {end} usable frames, fewer than moving_frame_steps={steps}."
        )
    tail = c[end - int(steps):end, 0]
    x0, dx = float(x[0]), float(x[1] - x[0])
    return [
        beam_following_x_slice(beam_cx=float(cx), x0=x0, dx=dx, nx=nx,
                               behind=behind, ahead=ahead)
        for cx in tail
    ]


def steady_entry_selection(
    entries, avg_window: int, stop_by_case: dict[int, int] | None = None
) -> tuple[list[int], list[int]]:
    """Flat indices of the LAST ``avg_window`` frames of every case, case-major.

    ``stop_by_case`` maps a case index to its `laser_on_frame_stop`; frames at or past it
    are excluded before the tail is taken, so an overrun case is scored on the same frames
    its steady target was built from.

    ``entries`` is a ``LazySampleDataset``-style sequence of ``(case_idx, time_idx)``.
    Returns ``(indices, case_ids)`` with one case id per selected index, ordered so a
    sequential DataLoader walks each case's window contiguously -- which is what lets the
    eval loop average a case and emit one scored frame, mirroring how the steady dataset
    itself is built (mean of the last ``avg_window`` beam-followed frames).
    """
    w = int(avg_window)
    if w < 1:
        raise ValueError(f"avg_window must be >= 1, got {avg_window!r}.")
    by_case: dict[int, list[tuple[int, int]]] = {}
    for flat, (case, t) in enumerate(entries):
        if t is None:
            raise ValueError(
                "steady transfer needs per-frame entries; this dataset serves whole "
                "cases (time_idx=None), which the eval loop cannot window."
            )
        by_case.setdefault(int(case), []).append((int(t), flat))
    indices: list[int] = []
    case_ids: list[int] = []
    for case in sorted(by_case):
        frames = sorted(by_case[case])
        if stop_by_case and case in stop_by_case:
            frames = [f for f in frames if f[0] < int(stop_by_case[case])]
        if len(frames) < w:
            raise ValueError(
                f"case {case} has {len(frames)} frames, fewer than avg_window={w}."
            )
        for _, flat in frames[-w:]:
            indices.append(flat)
            case_ids.append(case)
    return indices, case_ids


def fixed_window_slice(*, nx: int, behind: int, ahead: int) -> slice:
    """Crop an ALREADY beam-centred grid (a steady/moving-ref native one) to a window.

    The legacy derivation pins the beam ``legacy_window_split(nx)[0]`` cells in, so the
    crop is a fixed x-slice -- no per-frame beam position needed. Used to put a
    steady-native model's 61-wide prediction on the same 32+16 support as the
    transferred lab/moving-ref ones.
    """
    width = int(behind) + int(ahead)
    if width > int(nx):
        raise ValueError(f"window width {width} exceeds the grid's {nx} cells.")
    legacy_behind, _ = legacy_window_split(int(nx))
    start = int(np.clip(legacy_behind - int(behind), 0, int(nx) - width))
    return slice(start, start + width)
