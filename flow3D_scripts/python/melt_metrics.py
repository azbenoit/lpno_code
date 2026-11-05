#!/usr/bin/env python3
"""Melt-pool geometry metrics — the single definition used by every scorer.

Implements `validation/ACCEPTANCE.md` verbatim. Import these rather than
re-deriving them: during the 2026-07-28/30 calibration campaign each analysis
script carried its own copy of this arithmetic, the copies drifted, and the drift
produced retracted results — an aspect ratio that took width from a different
frame than depth, and four separate comparisons of a simulated vapour depression
against a metallographic fusion depth.

Array layout is the NPZ export's: ``[t, x, y, z]`` with ``z`` ascending and metal
occupying the cells at ``z < 0``.

    from melt_metrics import fusion_geometry, depression, domain_validity, score_point

CLI, for a quick look at one exported case::

    python3 python/melt_metrics.py runs/<case>/npz/<case>.npz
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

#: Pre-registered acceptance band, as a fraction. Derived in ACCEPTANCE.md from
#: the sources' own scatter (Galiegue's ~10% along-track variability, its
#: single-location measurements, its own 30% r0 inversion, Gan's equal depths at two
#: speeds, the 12%
#: cross-lab spread) -- not chosen for convenience. Tightening it below ~0.10
#: would be fitting measurement noise.
DEFAULT_BAND = 0.30

#: Laser-on fraction of the exported frames, and the tail of that treated as steady.
LASER_ON_FRACTION = 0.85
STEADY_TAIL_FRACTION = 0.30

#: Domain-validity thresholds: a hot floor or a melt pool touching the side walls
#: means the domain, not the physics, set the answer.
MAX_FLOOR_K = 400.0
MAX_MELT_SPAN_FRACTION = 0.8

VALID_DEPTH_KINDS = ("fusion", "depression")


class DepthKindMismatch(ValueError):
    """Raised when a fusion depth is compared against a vapour-depression depth.

    Metallography anchors report the fusion boundary; X-ray anchors report the
    vapour cavity, and fusion is always the larger. This is a bug in the caller,
    never a judgement call -- it inflated one reported gap from ~2x to "4x".
    """


@dataclass(frozen=True)
class Geometry:
    depth_um: float
    width_um: float

    @property
    def aspect(self) -> float:
        return self.depth_um / self.width_um if self.width_um else 0.0


@dataclass(frozen=True)
class Depression(Geometry):
    depth_sd_um: float = 0.0


@dataclass(frozen=True)
class Validity:
    floor_k: float
    melt_span_fraction: float
    ok: bool
    # Melt reached the bottom of the exported array. For a refined run that floor
    # is the refinement box, not the substrate (assumptions.md: a refined run
    # exports the fine block only), so this catches a pool that outgrew its box.
    depth_truncated: bool = False
    # Melt touched an x face, i.e. an end wall in the scan direction. Adiabatic
    # walls cannot conduct heat away, so a pool against one runs deeper than the
    # same pool in a larger box -- and depth is usually the scored quantity.
    # REPORTED, not fatal: `ok` deliberately ignores it, because flipping it
    # would retroactively invalidate every scanned case in the back-catalogue.
    # Promoting it into `ok` is left as a separate, deliberate decision.
    end_wall_contact: bool = False
    # How far the floor rose above the substrate's OWN initial temperature.
    #
    # `floor_k` against MAX_FLOOR_K answers "did heat reach the bottom" only for a
    # substrate starting near ambient. A case preheated to 500-770 K has its floor
    # above 400 K on frame 0, before the laser fires, so the absolute test flags it
    # invalid by construction: measured 2026-08-22, 47 of 60 cases in BOTH
    # ti64_preheat60 batches read ok=0, and invalidity tracks the preheat axis
    # exactly (every case >= 400 K preheat invalid, every case below it valid)
    # while the floor's median rise above its own initial is +0.2 K -- untouched.
    # Only 9-10 of 60 per batch have heat genuinely reaching the bottom.
    #
    # REPORTED, not fatal, for the same reason as `end_wall_contact`: promoting it
    # into `ok` would retroactively revalidate every preheated case in the
    # back-catalogue, which is a human's call. See tests/test_floor_rise.py.
    floor_rise_k: float = 0.0


@dataclass(frozen=True)
class PointScore:
    sim_depth_um: float
    measured_depth_um: float
    error_pct: float
    within_band: bool
    depth_kind: str


@dataclass(frozen=True)
class LadderSummary:
    n: int
    n_within_band: int
    mean_signed_error_pct: float
    systematic: bool


def _cell_sizes(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, int, float, float]:
    z = np.asarray(data["z"])
    metal = z < 0.0
    dz = abs(float(z[1] - z[0])) * 1e6
    dy = abs(float(np.asarray(data["y"])[1] - np.asarray(data["y"])[0])) * 1e6
    return metal, int(metal.sum()), dz, dy


def melt_mask(data: Mapping[str, np.ndarray], *,
              solidus_k: float | None = None) -> np.ndarray | None:
    """Molten cells per frame, from `Melt Region` or, failing that, temperature.

    Datasets ingested with the ML training profile carry only Fraction Of Fluid,
    Pressure, Temperature and Velocity, so `Melt Region` is absent and every melt
    metric returned nothing for them. `ti64_varspeed150` is 150 such cases.

    `Melt Region` is the INSTANTANEOUS molten fraction, and it is not a
    temperature threshold: per frame it agrees with `T > solidus` at only
    IoU 0.36-0.49, because it follows the solidus-liquidus fraction rather than a
    hard cut. What fusion geometry uses is the time UNION, and that reproduces at
    IoU 0.965. At the metric level, measured on four multi-beam cases, fusion
    depth was identical in all four and width identical in two with the other two
    one cell out.

    So the fallback is sound for FUSION GEOMETRY and is not a general substitute
    for the field: anything reading instantaneous melt still needs the real one.
    `solidus_k` must be given explicitly, because guessing it would silently score
    one alloy with another's melting point.
    """
    mr = data.get("Melt Region")
    if mr is not None:
        return np.asarray(mr) > 0.5
    temp = data.get("Temperature")
    if temp is None or solidus_k is None:
        return None
    return np.asarray(temp) > float(solidus_k)


def beam_track(data: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """Beam position per frame, taken from the deposition field itself.

    `Heat Absorption Rate` is where the laser is putting power, so its peak on
    the metal surface is the beam. Reading the track from the solved field rather
    than from the config means it works for any path, needs no config alongside
    the NPZ, and reflects where the beam actually went.

    Returns an (n_frames, 2) array of (x, y) in metres, with NaN on frames that
    received no power (before laser-on, after turn-off). None when the field is
    absent, which is the case for exports that did not request it.
    """
    har = data.get("Heat Absorption Rate")
    if har is None:
        return None
    har = np.asarray(har)
    x = np.asarray(data["x"]); y = np.asarray(data["y"])
    flat = har.reshape(har.shape[0], har.shape[1] * har.shape[2], -1).sum(axis=2)
    out = np.full((har.shape[0], 2), np.nan)
    live = flat.max(axis=1) > 0
    idx = np.argmax(flat, axis=1)
    ix, iy = np.divmod(idx, har.shape[2])
    out[live, 0] = x[ix[live]]
    out[live, 1] = y[iy[live]]
    return out


def fusion_geometry_at_beam(data: Mapping[str, np.ndarray], *,
                            half_width_um: float = 400.0,
                            solidus_k: float | None = None) -> Geometry:
    """Cross-section geometry in the plane perpendicular to the beam's travel.

    `fusion_geometry` slices a band at mid-x, which assumes the track runs along
    x through the centre of the domain. A random or multi-beam path has no such
    track, so that band lands wherever the domain centre happens to be: on the
    2026-08-05 multi-beam batch it returned depth 0 AND width 0 for `mb40_008`
    and `mb40_023` while both carried over 2 % melt.

    This takes the beam's own position (see `beam_track`), picks the mid-track
    frame, and unions melt over time in the plane through that point
    perpendicular to the local direction of travel. Metallography sections a
    solidified track at one place, so unioning over time is still what a section
    shows; only the choice of plane changes. For a straight x-track that plane is
    the y-z plane at mid-x, so this reproduces `fusion_geometry`.
    """
    track = beam_track(data)
    if track is None:
        return Geometry(0.0, 0.0)
    live = np.where(np.isfinite(track[:, 0]))[0]
    if live.size < 2:
        return Geometry(0.0, 0.0)

    pts = track[live]
    # Smooth before measuring. The deposition peak hops around the cavity walls:
    # on t2_nist195 the median frame-to-frame step is 22 um while the beam only
    # advances 3.5 um per frame, so raw arc length accumulates jitter instead of
    # travel and the direction came out perpendicular to the true one.
    if len(pts) >= 9:
        k = min(9, len(pts) // 2 * 2 + 1)
        ker = np.ones(k) / k
        pad = k // 2
        sm = np.empty_like(pts)
        for c in (0, 1):
            padded = np.r_[np.full(pad, pts[0, c]), pts[:, c], np.full(pad, pts[-1, c])]
            sm[:, c] = np.convolve(padded, ker, mode="valid")[:len(pts)]
        pts = sm
    step = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    mid = int(np.searchsorted(step, step[-1] / 2.0))
    mid = min(max(mid, 0), len(pts) - 1)
    x0, y0 = pts[mid]
    # Direction over a BASELINE, not between adjacent frames. The track comes from
    # the argmax of a cell-quantised field, so it jitters by a cell and stalls:
    # spot130_p100_v020_std has 235 zero-steps in 688 frames, and on the straight
    # x-track cocr_p200_v095 an adjacent-frame difference read dy = -30 um against
    # dx = +20 um. Taking the direction from adjacent frames therefore pointed the
    # normal ALONG the track and measured its length as width (810 um against a
    # true 130 um on rep3_in718_p245). A baseline of several cells averages the
    # jitter out.
    base = max(5.0 * abs(float(np.asarray(data["y"])[1] - np.asarray(data["y"])[0])),
               step[-1] * 0.05)
    lo = int(np.searchsorted(step, max(step[mid] - base, 0.0)))
    hi = int(np.searchsorted(step, min(step[mid] + base, step[-1])))
    lo, hi = min(lo, len(pts) - 1), min(hi, len(pts) - 1)
    if hi == lo:                          # too short to have a direction
        lo, hi = 0, len(pts) - 1
    dx, dy_ = pts[hi] - pts[lo]
    norm = float(np.hypot(dx, dy_))
    if norm == 0.0:                      # stationary beam: any perpendicular does
        px, py = 0.0, 1.0
    else:
        px, py = -dy_ / norm, dx / norm  # unit normal to the direction of travel

    metal, n_metal, dz, dy_um = _cell_sizes(data)
    melt = melt_mask(data, solidus_k=solidus_k)
    if melt is None:
        return Geometry(0.0, 0.0)
    ever = melt.any(axis=0)[:, :, metal]
    x = np.asarray(data["x"]); y = np.asarray(data["y"])
    # Sample the union field along the perpendicular at the native lateral pitch,
    # so the reported width keeps the mesh's own resolution.
    pitch = abs(float(y[1] - y[0]))
    n = int(half_width_um * 1e-6 / pitch)
    s = np.arange(-n, n + 1) * pitch
    xs = np.clip(np.searchsorted(x, x0 + s * px), 0, len(x) - 1)
    ys = np.clip(np.searchsorted(y, y0 + s * py), 0, len(y) - 1)
    line = ever[xs, ys, :]               # (samples, n_metal)
    if not line.any():
        return Geometry(0.0, 0.0)
    depth = (n_metal - int(np.where(line.any(axis=0))[0].min())) * dz
    hit = np.where(line.any(axis=1))[0]
    width = (int(hit.max() - hit.min()) + 1) * pitch * 1e6
    return Geometry(float(depth), float(width))


def fusion_geometry_auto(data: Mapping[str, np.ndarray], *,
                         solidus_k: float | None = None) -> Geometry:
    """Fusion geometry, measured at the beam when that is possible.

    This is what scorers should call. It prefers `fusion_geometry_at_beam`, which
    sections perpendicular to the beam's own direction of travel and so works for
    any path, and falls back to the mid-track `fusion_geometry` when the run
    carries no deposition field to locate the beam with.

    Switching is safe for the straight-track anchors: across seven real cases
    spanning four alloys the two agree exactly in depth and within 0-20 um
    (0-2 cells) in width. It matters for everything else -- on random paths the
    mid-track slice reported the track's extent rather than the pool's, by up to
    530 um, and occasionally missed the track and returned zero.
    """
    if beam_track(data) is not None:
        g = fusion_geometry_at_beam(data, solidus_k=solidus_k)
        if g.depth_um > 0.0 or g.width_um > 0.0:
            return g
    return fusion_geometry(data, solidus_k=solidus_k)


def fusion_geometry(data: Mapping[str, np.ndarray], *, track_half_cells: int = 5,
                    solidus_k: float | None = None) -> Geometry:
    """Time-UNION melt extent in a band at mid-track — what metallography sees.

    A cross-section shows the maximum extent ever melted, not the melt at one
    instant, so this unions over time rather than picking a frame.

    Reads melt through `melt_mask`, so the temperature fallback reaches this
    path too. It used to index `Melt Region` directly, which meant a case
    without that field raised KeyError HERE even though `melt_mask` could have
    scored it — `fusion_geometry_auto` falls back to this function whenever the
    run carries no deposition field to locate the beam with, so the fallback was
    only half-wired: every ML-profile export with no `Heat Absorption Rate`
    (ti64_varspeed150, 150 cases) still got nothing.
    """
    metal, n_metal, dz, dy = _cell_sizes(data)
    melt = melt_mask(data, solidus_k=solidus_k)
    if melt is None:
        return Geometry(0.0, 0.0)
    xc = melt.shape[1] // 2
    lo, hi = max(0, xc - track_half_cells), xc + track_half_cells + 1
    ever = melt[:, lo:hi, :, :].any(axis=0)[:, :, metal]
    if not ever.any():
        return Geometry(0.0, 0.0)
    depth = (n_metal - int(np.where(ever.any(axis=(0, 1)))[0].min())) * dz
    ys = np.where(ever.any(axis=(0, 2)))[0]
    width = (int(ys.max() - ys.min()) + 1) * dy
    return Geometry(float(depth), float(width))


def _gas_in_metal(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, int, float, float]:
    """Interior gas cells below the original surface, x/y edge columns excluded.

    The edge columns are dropped because a boundary cell can read as gas for
    reasons that have nothing to do with the keyhole.
    """
    metal, n_metal, dz, dy = _cell_sizes(data)
    gas = np.asarray(data["Fraction Of Fluid"])[:, :, :, metal] < 0.5
    gas[:, [0, -1], :, :] = False
    gas[:, :, [0, -1], :] = False
    return gas, n_metal, dz, dy


#: Cells needed across a feature before a length measured on it means anything --
#: and, more seriously, before the solver can be expected to sustain the feature at
#: all. A matched mesh pair on 2026-07-30 gave a 2.3-cell cavity at 10 um and a
#: 24.7-cell cavity at 5 um from identical physics (+164% fusion depth), so at this
#: size "the model under-predicts" and "the mesh cannot support it" are the same
#: observation. See validation/ACCEPTANCE.md.
MIN_CELLS_PER_FEATURE = 5.0


@dataclass(frozen=True)
class MeshAdequacy:
    feature_um: float
    cell_um: float
    cells: float
    required_cell_um: float
    ok: bool


def mesh_adequate(*, feature_um: float, cell_um: float,
                  min_cells: float = MIN_CELLS_PER_FEATURE) -> MeshAdequacy:
    """Is this cell size fine enough to resolve a feature of this size?

    The criterion is on the FEATURE, not the beam. Cells-per-beam-radius does not
    predict adequacy for this campaign: every Gan case ran at 4.75 cells/radius and
    passed, every Galiegue rung at 6.0 cells/radius and failed -- the finer-per-radius
    cases are the failures. What separates them is cells across the vapour cavity.

    **This is diagnostic, not predictive, and the distinction matters.** Feeding it
    the cavity depth the Gan scaling law predicts does NOT anticipate our failures:
    at the Galiegue transition conditions the law predicts 86-130 um, i.e. 9-13
    cells at 10 um, so an a-priori check passes every case that in fact failed. The
    deficit is in what the simulation produces (20 um), not in what the physics
    calls for.

    The implication is that the limitation is in cavity *initiation* rather than in
    resolving a final depth: a cavity must grow from zero through the 1-2 cell
    regime before reaching a resolved one, and where the drive is marginal the grid
    never lets it start. No static criterion on the final feature size captures
    that, so use this to interpret a completed run -- never to certify a mesh in
    advance.
    """
    cells = feature_um / cell_um if cell_um > 0 else 0.0
    return MeshAdequacy(feature_um, cell_um, cells,
                        feature_um / min_cells if min_cells else float("inf"),
                        cells >= min_cells)


def cell_size_um(data: Mapping[str, np.ndarray]) -> float:
    """This case's mesh cell size, in microns, read from its own z coordinates.

    Never hardcode this. The reproducibility notebook carried ``CELL_UM = 10.0``
    while 14 of 96 campaign cases ran at 6, 25 or 50 um, so every length it
    reported for them was wrong by 1.67x, 2.5x or 5x. It is also the resolution
    floor on every length: see the quantization rule in ACCEPTANCE.md.
    """
    return _cell_sizes(data)[2]


def depression_frame(data: Mapping[str, np.ndarray], frame: int) -> Geometry:
    """Vapour-cavity geometry in ONE frame — for imagery, not for scoring.

    Single-frame numbers sit inside the oscillation amplitude (measured +-30-42 um
    on keyhole cases), so quote `depression()` instead when comparing to anything.
    """
    gas, n_metal, dz, dy = _gas_in_metal(data)
    g = gas[frame]
    if not g.any():
        return Geometry(0.0, 0.0)
    depth = (n_metal - int(np.where(g.any(axis=(0, 1)))[0].min())) * dz
    ys = np.where(g.any(axis=(0, 2)))[0]
    return Geometry(float(depth), float((int(ys.max() - ys.min()) + 1) * dy))


def steady_window(n_frames: int, *, tail_fraction: float = STEADY_TAIL_FRACTION,
                  laser_off_s: float | None = None,
                  times_s: "np.ndarray | None" = None) -> slice:
    """Frames treated as steady: the tail of the laser-on portion.

    Pass ``laser_off_s`` (from the case manifest's ``derived.laser_off_time_s``)
    together with the export's ``time_s`` whenever you have them; the window is
    then the tail of the frames that are actually laser-on.

    ``LASER_ON_FRACTION`` is only a fallback for when the laser-off time is
    unknown. It is a heuristic fitted to this campaign's cases, which turn the
    laser off at ~80% of the run -- and it is WRONG for datasets that run the laser
    to the end. The 84-case straight-path sweep sets ``finish_time == travel
    time``, so the laser is on for 100% of every run and the fallback silently
    discards the final 15% of the track, its deepest part, as a cooling tail that
    does not exist.
    """
    if laser_off_s is not None and times_s is not None and len(times_s):
        on = int(np.searchsorted(np.asarray(times_s), laser_off_s, side="right"))
        on = max(1, min(on, n_frames))
    else:
        on = int(n_frames * LASER_ON_FRACTION)
    return slice(on - max(1, int(on * tail_fraction)), on)


def depression_series(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-frame deepest interior gas cell below z = 0, in microns."""
    gas, n_metal, dz, _ = _gas_in_metal(data)
    return np.asarray([(n_metal - int(np.where(g.any(axis=(0, 1)))[0].min())) * dz
                       if g.any() else 0.0 for g in gas])


def depression(data: Mapping[str, np.ndarray], *,
               tail_fraction: float = STEADY_TAIL_FRACTION,
               laser_off_s: float | None = None) -> Depression:
    """Steady-window vapour depression — what in-situ X-ray sees.

    Depth AND width come from the SAME window. Taking width from the globally
    widest frame is the round-4 bug: for dwell cases that frame falls after
    laser-off, when the cavity has collapsed and spread.
    """
    gas, _, dz, dy = _gas_in_metal(data)
    series = depression_series(data)
    win = steady_window(len(series), tail_fraction=tail_fraction,
                        laser_off_s=laser_off_s, times_s=data.get("time_s"))
    widths = []
    for g in gas[win]:
        if g.any():
            ys = np.where(g.any(axis=(0, 2)))[0]
            widths.append((int(ys.max() - ys.min()) + 1) * dy)
    return Depression(depth_um=float(series[win].mean()),
                      width_um=float(np.mean(widths)) if widths else 0.0,
                      depth_sd_um=float(series[win].std()))


def domain_validity(data: Mapping[str, np.ndarray], *,
                    solidus_k: float | None = None) -> Validity:
    """A clipped case counts for nothing — in either direction.

    It cannot support a finding and it cannot overturn one. Invalidating a
    measurement is not the same as invalidating the finding it supported: re-run
    a valid case before retracting.
    """
    temperature = np.asarray(data["Temperature"])
    floor = float(temperature[:, :, :, 0].max())
    metal = np.asarray(data["z"]) < 0.0
    # The substrate's own initial temperature, read from the export rather than a
    # config, so this stays self-contained and works for any caller.
    initial = (float(np.median(temperature[0][:, :, metal]))
               if metal.any() and temperature.size else 0.0)
    floor_rise = floor - initial
    # Metal only: the question is whether the POOL reached the side walls. Measuring
    # over all z counts ejected melt in the gas layer, which falsely invalidated
    # three SS304L 456 W cases and a CoCr case (2026-07-30).
    # Through melt_mask for the same reason as fusion_geometry: a case with no
    # `Melt Region` used to raise KeyError instead of using the temperature
    # fallback, so clipping could not be checked for it at all.
    melt_all = melt_mask(data, solidus_k=solidus_k)
    if melt_all is None:
        return Validity(floor, 0.0, floor < MAX_FLOOR_K, False,
                        floor_rise_k=floor_rise)
    melt = np.asarray(melt_all)[:, :, :, metal]
    ys = np.where(melt.any(axis=0).any(axis=(0, 2)))[0]
    span = ((ys.max() - ys.min() + 1) / len(np.asarray(data["y"]))) if ys.size else 0.0
    # Lateral span is a fraction of the array's own width, so it already scales to a
    # refined run's fine block. Depth had no check at all: melt touching the lowest
    # exported cell is measured against a boundary rather than against solid, which
    # under-reports depth.
    depth_truncated = bool(melt[..., 0].any()) if melt.size else False
    # End walls in the SCAN direction. The y-span check above collapses x
    # (`axis=(0, 2)`), so nothing looked at the x faces; an x-span fraction is
    # the wrong test for a track, which is legitimately long in x, but CONTACT
    # with either face is unambiguous. All 7 Galiegue spot60 cases had it and
    # passed (2026-08-16).
    end_wall_contact = bool(melt[:, 0].any() or melt[:, -1].any()) if melt.size else False
    return Validity(floor, float(span),
                    floor < MAX_FLOOR_K and span < MAX_MELT_SPAN_FRACTION
                    and not depth_truncated,
                    depth_truncated, end_wall_contact,
                    floor_rise_k=floor_rise)


# --------------------------------------------------------------------------
# Absorptance. Everything that turns a solver log into a coupling number lives
# in this block: the audit parser, the window scalar, the time-resolved trace,
# and the evaporative reconstruction with its guards. Keep it that way. Each of
# these was at some point re-implemented at a call site -- the manuscript
# notebook hand-rolled the parser twice and the trace twice -- and a hand-rolled
# copy is what put a wrong absorptance in a figure (see gross_absorptance).
# An external reviewer should be able to read this one block and check every
# absorptance the project reports.
# --------------------------------------------------------------------------

def hd3out_audit(hd3out_path: Path | str, quantity: str) -> "np.ndarray":
    """Time series of one audited quantity from a FLOW-3D ``hd3out.*`` log.

    FLOW-3D prints its "mass and energy audits" as blocks under a ``time =``
    header, so a value belongs to the most recent time printed before it. Known
    quantities include ``total fluid mass``, ``total fluid #1 thermal energy``
    and ``cumulative fluid mass source``.

    Returns an ``(n, 2)`` array of ``(time_s, value)``, empty when the file or
    the quantity is absent -- a diagnostic must not abort a scoring sweep.
    Repeated times are collapsed to their first sample, because hd3out reprints
    a block at the same time on restarts and a duplicate makes any derivative
    divide by zero.
    """
    import re
    import numpy as np

    try:
        txt = Path(hd3out_path).read_text(errors="ignore")
    except OSError:
        return np.empty((0, 2))
    times = [(m.start(), float(m.group(1)))
             for m in re.finditer(r"\btime\s*=\s*([0-9]\.[0-9Ee+-]+)", txt)]
    if not times:
        return np.empty((0, 2))
    pattern = re.escape(quantity) + r"\s*=\s*([0-9.Ee+-]+)"
    rows: list[tuple[float, float]] = []
    ti = 0
    for m in re.finditer(pattern, txt):
        while ti + 1 < len(times) and times[ti + 1][0] < m.start():
            ti += 1
        if not rows or times[ti][1] != rows[-1][0]:
            rows.append((times[ti][1], float(m.group(1))))
    return np.array(rows, dtype=float) if rows else np.empty((0, 2))


def absorbed_fraction(hd3out_path: Path | str, power_w: float,
                      t0: float = 2e-4, t1: float = 8e-4) -> float:
    """Net absorbed fraction from the solver's own energy audit.

    Reads ``total fluid #1 thermal energy`` against ``time`` in ``hd3out.*`` and
    divides the gain over the window by the energy delivered. Returns NaN rather
    than raising when the file is missing or the window holds fewer than two
    samples, because this is a diagnostic and a missing audit must not abort a
    scoring sweep.

    This is **net** absorptance: it counts energy still in the metal, so it reads
    lower than a measured **gross** absorptance, which includes what later leaves
    by radiation, convection and vaporization. Comparing net against a gross
    measurement understates the model -- the NIST 50.8% figure is gross, so the
    true gap there is smaller than the raw difference suggests. Say which one you
    are quoting.

    Lived only in the reproducibility notebook until 2026-07-30; it is the metric
    behind the NIST absorptance diagnostic and belongs with the others.
    """
    rows = hd3out_audit(hd3out_path, "total fluid #1 thermal energy")
    window = [(t, e) for t, e in rows if t0 <= t <= t1]
    if len(window) < 2:
        return float("nan")
    (ta, ea), (tb, eb) = window[0], window[-1]
    if power_w <= 0 or tb <= ta:
        return float("nan")
    return (eb - ea) / (power_w * (tb - ta))


@dataclass(frozen=True)
class AbsorptanceSeries:
    """Retained fraction of incident power, resolved in time.

    `retained` is per frame of `time_s`; `scale` converts one unit of the
    deposition field into a fraction of incident power, and `window_mean` is
    what `absorbed_fraction` returns for the same log and window.
    """
    time_s: "np.ndarray"
    retained: "np.ndarray"
    scale: float
    window_mean: float


def absorbed_fraction_series(time_s, deposited, hd3out_path: Path | str,
                             power_w: float, t0: float = 2e-4,
                             t1: float = 8e-4) -> AbsorptanceSeries:
    """`absorbed_fraction` resolved in time, for a per-frame comparison.

    The energy audit is printed a handful of times per run, far too coarsely to
    trace a drilling transient, while the ``Heat Absorption Rate`` field is
    exported every frame but in the solver's own units. So the SHAPE comes from
    the field -- pass `deposited`, its per-frame sum over the domain -- and the
    LEVEL from the audit: the trace is scaled so that its mean over ``t0..t1``
    equals `absorbed_fraction` on the same log. Averaging the result over that
    window and getting the scalar back is the contract, and it is a test.

    Like `absorbed_fraction` this is the RETAINED share, a lower bound on a
    calorimeter's gross absorptance. Do not add an evaporative correction from
    the mass audit; `gross_absorptance` explains why.

    Returns NaN fields rather than raising when the audit is missing. Raises
    ValueError only for a caller error, a `deposited` of the wrong length,
    which would otherwise broadcast into a plausible but wrong trace.
    """
    import numpy as np

    time_s = np.asarray(time_s, dtype=float)
    deposited = np.asarray(deposited, dtype=float)
    if deposited.shape != time_s.shape:
        raise ValueError(
            f"deposited has shape {deposited.shape}, time_s has {time_s.shape}; "
            "pass the per-frame total of the deposition field for THESE frames")

    nan = np.full(time_s.shape, float("nan"))
    mean = absorbed_fraction(hd3out_path, power_w, t0=t0, t1=t1)
    inside = (time_s >= t0) & (time_s <= t1)
    if mean != mean or not inside.any():
        return AbsorptanceSeries(time_s, nan, float("nan"), mean)
    ref = float(np.mean(deposited[inside]))
    if not np.isfinite(ref) or ref == 0.0:
        return AbsorptanceSeries(time_s, nan, float("nan"), mean)
    scale = mean / ref
    return AbsorptanceSeries(time_s, deposited * scale, scale, mean)


@dataclass(frozen=True)
class Absorptance:
    """Coupling audit, all fractions of incident laser power.

    `retained` is what `absorbed_fraction` returns. `gross` is the lower bound
    on the quantity a calorimeter measures. `plausible` is False when the
    reconstruction exceeds the incident power, which means the mass audit is
    ejection-dominated and the number must not be quoted.
    """
    retained: float
    evaporative: float
    gross: float
    plausible: bool


# Ti-6Al-4V. The latent term dominates; cp only heats the evaporated mass to the
# vaporization anchor and moves the total by well under a point.
TI64_LATENT_VAPORIZATION_J_KG = 9.83e6
TI64_CP_J_KG_K = 700.0
TI64_TV_K = 3520.0

# Above this the reconstruction is not believable. Conservation alone (<= 1.0) is
# far too permissive: it passes t2_206w at 98 %, which would mean a Ti-6Al-4V
# keyhole reflecting essentially nothing. The anchor is our own reference set --
# the highest absorptance ever measured in it is the 73.5 % peak of the NIST
# Ti-6Al-4V scan -- and 0.85 leaves headroom above that for a deeper cavity.
MAX_CREDIBLE_ABSORPTANCE = 0.85

# Largest single rise the mass audit may show inside the window, as a fraction of
# its net loss, before the series is refused as an evaporation record. On
# t2_nist195 the audit bottoms at 640 us and climbs 2.4 % of the net back by the
# end of the run. Across 2861 runs it is non-monotone in 50.7 % of cases, and in
# 19.8 % the largest rise exceeds the entire net loss.
MASS_REVERSAL_TOLERANCE = 0.01


def gross_absorptance(hd3out_path: Path | str, power_w: float,
                      t0: float = 2e-4, t1: float = 8e-4,
                      latent_j_kg: float = TI64_LATENT_VAPORIZATION_J_KG,
                      cp_j_kg_k: float = TI64_CP_J_KG_K,
                      t_vaporization_k: float = TI64_TV_K,
                      t_ambient_k: float = 293.0,
                      max_credible: float = MAX_CREDIBLE_ABSORPTANCE) -> Absorptance:
    """Absorptance comparable with integrating-sphere calorimetry.

    `absorbed_fraction` counts energy still in the metal. A calorimeter counts
    every watt that crossed the surface, including what later left by radiation,
    convection and evaporation, so the two are different quantities and the raw
    difference overstates the disagreement. This adds the evaporative part back,
    charged at latent plus sensible heat against the mass the audit says left.

    **Check `plausible` and believe it.** Two failure modes below put most of
    our reference cases on the wrong side of it, including the NIST Ti-6Al-4V
    scan this was built for, so a caller that ignores the flag will quote a
    number the function itself rejects. `absorbed_fraction` alone, reported as a
    lower bound, is the safe fallback and is what the LP-NO manuscript uses.

    Where it does hold, it is still a LOWER bound. Radiation and convection to
    the void are not in the printed audit, so the true gross is a little higher.

    **Do not quote the result when `plausible` is False.** The mass audit cannot
    separate evaporated mass from ejected droplets and spatter, which leave the
    domain without paying the latent heat this charges them. In strong keyholing
    that inflates the total beyond anything a metal absorbs: measured 2026-08-15,
    `t2_206w` reconstructs to 98 % and `rep4_in718_s049_blind` to 94 %, against a
    physical 49 % for `t2_nist195` and a 73.5 % measured peak. Separating the two needs a vapour-species
    audit or a flux through a control surface at the free boundary; neither is in
    `hd3out`.

    **`total fluid mass` is a NET phase-change balance, not an evaporation
    record — so never evaluate this per frame.** Every deck this repo renders
    carries `rsize_cond=1` and `acondens=1.0`, so recondensation is active at
    full accommodation and returns mass to the fluid. Phase change does not
    appear in `cumulative fluid mass source`, which counts source terms, so that
    counter reading zero on all 2861 runs says nothing either way. What the audit
    reports is evaporation MINUS recondensation, and charging that difference to
    carried-off enthalpy is wrong whichever term dominates.

    It shows: measured 2026-08-17, `total fluid mass` on `t2_nist195` moves 0.2 %
    across the solve, falls until 640 us and then RISES; across 2861 runs it is
    non-monotone in 50.7 % of cases and in 19.8 % the largest rise exceeds the
    entire net loss. Its early rate on `t2_nist195`, 2.5e-6 kg/s or 30 W of a
    195 W beam, is several times a Langmuir flux at the boiling point, so it is
    not a credible evaporation signal on magnitude either.

    Endpoints across such a window still net out to a believable loss, which is
    why `plausible` now inspects the SERIES for a reversal
    (`MASS_REVERSAL_TOLERANCE`) instead of only the total. A time-resolved trace
    built from this recipe is worse still: one added ~15 points during the
    drilling transient, lifting a simulated absorptance onto a measured peak the
    model does not reach (LP-NO manuscript Fig. 9b, removed 2026-08-17). Use
    `absorbed_fraction` or `absorbed_fraction_series` and report the lower bound.

    A window holding only two audit samples cannot show a reversal at all, so
    `plausible` is weaker there. Widen `t0`/`t1` if the run is short.

    Returns NaN fields rather than raising when the audit is missing, matching
    `absorbed_fraction`: a diagnostic must not abort a scoring sweep.
    """
    retained = absorbed_fraction(hd3out_path, power_w, t0=t0, t1=t1)
    if retained != retained:                       # NaN: no audit to work from
        return Absorptance(retained, float("nan"), float("nan"), False)

    rows = hd3out_audit(hd3out_path, "total fluid mass")
    window = [(t, m) for t, m in rows if t0 <= t <= t1]

    evaporative = 0.0
    if len(window) >= 2:
        (ta, ma), (tb, mb) = window[0], window[-1]
        # Only a DECREASE is carried-off mass. A rising audit is not negative
        # evaporation, and subtracting it would invent absorbed power.
        lost = max(ma - mb, 0.0)
        if power_w > 0 and tb > ta:
            enthalpy = latent_j_kg + cp_j_kg_k * (t_vaporization_k - t_ambient_k)
            evaporative = lost * enthalpy / (power_w * (tb - ta))

    # Endpoints alone cannot see a series that falls and then climbs back, and a
    # series that climbs is recording net phase change, not carried-off mass.
    # Flag it, because the net across such a window still looks like a plausible
    # loss. The threshold sits an order of magnitude above the audit's own print
    # quantization (~0.1 % of a typical net loss at six significant digits).
    biggest_rise = max((b - a for (_, a), (_, b) in zip(window, window[1:])),
                       default=0.0)
    net_loss = window[0][1] - window[-1][1] if len(window) >= 2 else 0.0
    reverses = biggest_rise > MASS_REVERSAL_TOLERANCE * max(net_loss, 0.0)

    gross = retained + evaporative
    return Absorptance(retained, evaporative, gross,
                       gross <= max_credible and not reverses)


def score_point(*, sim_depth_um: float, measured_depth_um: float,
                sim_kind: str, measured_kind: str,
                band: float = DEFAULT_BAND) -> PointScore:
    """Signed error against one measured point, refusing cross-kind comparison."""
    for kind in (sim_kind, measured_kind):
        if kind not in VALID_DEPTH_KINDS:
            raise ValueError(f"depth_kind must be one of {VALID_DEPTH_KINDS}, got {kind!r}")
    if sim_kind != measured_kind:
        raise DepthKindMismatch(
            f"refusing to compare a {sim_kind} depth against a {measured_kind} depth; "
            "metallography reports the fusion boundary, X-ray the vapour cavity, "
            "and fusion is always the larger")
    err = 100.0 * (sim_depth_um - measured_depth_um) / measured_depth_um
    return PointScore(sim_depth_um, measured_depth_um, err, abs(err) <= band * 100.0,
                      measured_kind)


def ladder_summary(errors_pct: Sequence[float], *, band: float = DEFAULT_BAND) -> LadderSummary:
    """Pass count AND mean signed error — never the pass count alone.

    A one-sign offset across a whole ladder is a finding even when every
    individual point sits inside the band.
    """
    errs = list(errors_pct)
    within = sum(1 for e in errs if abs(e) <= band * 100.0)
    mean = float(np.mean(errs)) if errs else 0.0
    systematic = bool(errs) and (all(e > 0 for e in errs) or all(e < 0 for e in errs))
    return LadderSummary(len(errs), within, mean, systematic)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="+")
    args = ap.parse_args()
    for path in args.npz:
        data = np.load(path)
        fus, dep, val = fusion_geometry_auto(data), depression(data), domain_validity(data)
        print(f"{path.name}: fusion {fus.depth_um:.0f}/{fus.width_um:.0f} um "
              f"(aspect {fus.aspect:.2f}) | depression {dep.depth_um:.0f}+-{dep.depth_sd_um:.0f}"
              f"/{dep.width_um:.0f} um (aspect {dep.aspect:.2f}) | "
              f"floor {val.floor_k:.0f} K, melt span {val.melt_span_fraction:.2f} -> "
              f"{'VALID' if val.ok else 'DOMAIN-INVALID'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
