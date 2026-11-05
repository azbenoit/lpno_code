#!/usr/bin/env python3
"""Build constant-speed laser scan paths for the rectangular moving-laser case.

FLOW-3D moves the beam by integrating a velocity time-table from the beam origin
(``xb0lbm``/``yb0lbm``).  A straight scan is one constant-velocity leg; an
arbitrary in-plane path is the same idea with more rows: sample the path at
*equal arc length*, so every leg has the same length and (at constant scan
speed) the same duration, and emit one ``utlbm``/``vtlbm``/``timlbm`` row per
sample.  Equal arc length is what keeps the scan speed constant, and the beam
then follows the sampled polyline exactly.

Four path specs are supported (``laser.path.type``):

- ``waypoints``: an ordered ``points_m`` list, joined ``linear`` (polyline) or
  ``spline`` (Catmull-Rom through the points);
- ``sinusoidal``: a preset sine wave travelling along +x;
- ``random``: a smooth, seeded random walk biased to steer toward the parts of
  the domain it has visited least;
- ``variable_speed``: explicit ``knots_m`` with per-knot arrival ``times_s`` (and
  optional per-leg ``power_w``); knots are kept verbatim with no arc-length
  resample, so the speed varies leg to leg (see ``LaserPath`` and
  ``_build_variable_speed``).

A hard check rejects any path whose samples come within ``4 * beam_radius`` of a
domain edge (the keep-out margin the caller passes in).

Stdlib only, matching ``run_rectangular_case.py``; no numpy/scipy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Cap on path samples.  Each sample becomes a constant-velocity plateau in the
# beam table (two rows per leg, see velocity_time_tables), so the emitted table is
# ~2x this.  The FLOW-3D beam time-table holds at most 500 rows -- MEASURED, not
# guessed: row 501 is rejected at preprocessing (`MAX_BEAM_TABLE_ROWS` in
# run_rectangular_case.py; mb40_007 died that way on 2026-08-03).  So 500 samples
# admits ~999 rows, about 2x over the cap; only multi_beam_tables guards the row
# count, and a single-beam path past ~250 samples renders a deck the preprocessor
# refuses.  No existing run is near it (max laser_path_num_samples on disk is 61).
# ponytail: fixed cap; raise MAX_PATH_SAMPLES if a real run needs a finer/longer path.
MAX_PATH_SAMPLES = 500

Point = tuple[float, float]


class PathError(ValueError):
    """A ``laser.path`` spec is missing, invalid, or leaves the safe region."""


@dataclass
class LaserPath:
    """A resolved in-plane path.

    Constant-speed paths (waypoints/sinusoidal/random) are resampled to equal
    arc length and leave ``times_s``/``powers_w`` as ``None`` (the caller applies
    one scan speed).  A ``variable_speed`` path instead keeps its knots verbatim
    and carries the per-knot arrival ``times_s`` (and optional per-leg
    ``powers_w``), so the beam speed varies leg to leg and a repeated knot with
    advancing time is a true dwell.
    """

    samples: list[Point]  # (x, y) in metres, first == start
    length_m: float  # total travelled length (sum of chords)
    spacing_m: float  # mean chord length between consecutive samples
    type: str
    interpolation: str  # "linear" or "spline"
    times_s: list[float] | None = None  # variable_speed: arrival time per knot, len == len(samples)
    powers_w: list[float] | None = None  # variable_speed: power per leg, len == len(samples)-1

    @property
    def start(self) -> Point:
        return self.samples[0]

    @property
    def end(self) -> Point:
        return self.samples[-1]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _polyline_length(points: list[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _trim_polyline(points: list[Point], target: float) -> list[Point]:
    """Cut a polyline at arc length ``target`` (interpolating the final segment).

    Used to give ``random`` paths an exact requested length.  If ``target`` is
    longer than the polyline, the whole polyline is returned unchanged.
    """
    out = [points[0]]
    acc = 0.0
    for a, b in zip(points, points[1:]):
        seg = math.dist(a, b)
        if seg <= 0.0:
            continue
        if acc + seg >= target:
            frac = (target - acc) / seg
            out.append((a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1])))
            return out
        acc += seg
        out.append(b)
    return out


def resample_by_arclength(points: list[Point], n_samples: int) -> list[Point]:
    """Return ``n_samples`` points spaced equally by arc length along ``points``.

    Endpoints are preserved exactly.  Equal arc-length spacing is what makes the
    per-leg speed constant once divided by a uniform time step.
    """
    if n_samples < 2:
        raise PathError("a path needs at least two samples")
    if len(points) < 2:
        raise PathError("cannot resample a path with fewer than two points")
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(a, b))
    total = cumulative[-1]
    if total <= 0.0:
        raise PathError("path has zero length; start and end coincide")
    out: list[Point] = [points[0]]
    seg = 0
    for i in range(1, n_samples - 1):
        target = total * i / (n_samples - 1)
        while seg < len(cumulative) - 2 and cumulative[seg + 1] < target:
            seg += 1
        span = cumulative[seg + 1] - cumulative[seg]
        frac = 0.0 if span <= 0.0 else (target - cumulative[seg]) / span
        ax, ay = points[seg]
        bx, by = points[seg + 1]
        out.append((ax + frac * (bx - ax), ay + frac * (by - ay)))
    out.append(points[-1])
    return out


def catmull_rom(control: list[Point], per_segment: int = 24) -> list[Point]:
    """Densify a smooth Catmull-Rom curve through ``control`` points.

    Endpoints are handled by reflecting a phantom point, so the curve starts and
    ends exactly on the first and last control points.
    """
    if len(control) < 2:
        raise PathError("a spline needs at least two control points")
    if len(control) == 2:
        return list(control)
    pts = list(control)
    first = (2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1])
    last = (2 * pts[-1][0] - pts[-2][0], 2 * pts[-1][1] - pts[-2][1])
    ext = [first, *pts, last]
    out: list[Point] = [pts[0]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for step in range(1, per_segment + 1):
            t = step / per_segment
            t2, t3 = t * t, t * t * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    return out


# --------------------------------------------------------------------------- #
# Path specs
# --------------------------------------------------------------------------- #
def _xy(value: object, label: str) -> Point:
    if not isinstance(value, list) or len(value) not in (2, 3):
        raise PathError(f"{label} must be an [x, y] or [x, y, 0] list in metres")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise PathError(f"{label} must contain finite numbers") from error
    if len(value) == 3 and abs(float(value[2])) > 1.0e-15:
        raise PathError(f"{label} z component must be 0 (the beam focal plane)")
    if not (math.isfinite(x) and math.isfinite(y)):
        raise PathError(f"{label} must contain finite numbers")
    return (x, y)


def path_start_point(spec: dict) -> tuple[float, float, float]:
    """Start (x, y, 0) a path implies, before the domain bounds are known.

    ``random`` returns ``None`` for x/y when no start is given so the caller can
    default it to the domain centre.
    """
    kind = _require_type(spec)
    if kind == "waypoints":
        points = spec.get("points_m")
        if not isinstance(points, list) or len(points) < 2:
            raise PathError("laser.path.points_m must list at least two [x, y] points")
        x, y = _xy(points[0], "laser.path.points_m[0]")
        return (x, y, 0.0)
    if kind == "variable_speed":
        knots = spec.get("knots_m")
        if not isinstance(knots, list) or len(knots) < 2:
            raise PathError("laser.path.knots_m must list at least two [x, y] points")
        x, y = _xy(knots[0], "laser.path.knots_m[0]")
        return (x, y, 0.0)
    if kind == "sinusoidal":
        x, y = _xy(spec.get("start_point_m", [0.0, 0.0, 0.0]), "laser.path.start_point_m")
        return (x, y, 0.0)
    # random: start may be omitted and defaulted to the domain centre later.
    start = spec.get("start_point_m")
    if start is None:
        return (None, None, 0.0)  # type: ignore[return-value]
    x, y = _xy(start, "laser.path.start_point_m")
    return (x, y, 0.0)


def _require_type(spec: dict) -> str:
    if not isinstance(spec, dict):
        raise PathError("laser.path must be an object")
    kind = spec.get("type")
    if kind not in ("waypoints", "sinusoidal", "random", "variable_speed"):
        raise PathError("laser.path.type must be waypoints, sinusoidal, random, or variable_speed")
    return kind


def _positive(spec: dict, key: str) -> float:
    value = spec.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise PathError(f"laser.path.{key} must be a positive number")
    return float(value)


def _positive_default(spec: dict, key: str, default: float) -> float:
    """Positive number for an optional field, defaulting when absent (clean
    PathError on a non-numeric value rather than a raw float() crash)."""
    if key not in spec:
        return default
    return _positive(spec, key)


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _sinusoidal_dense(spec: dict, radius: float) -> list[Point]:
    x0, y0 = _xy(spec.get("start_point_m", [0.0, 0.0, 0.0]), "laser.path.start_point_m")
    length = _positive(spec, "length_m")
    amplitude = spec.get("amplitude_m")
    if not isinstance(amplitude, (int, float)) or isinstance(amplitude, bool) or amplitude < 0:
        raise PathError("laser.path.amplitude_m must be a non-negative number")
    has_wl, has_periods = "wavelength_m" in spec, "periods" in spec
    if has_wl == has_periods:
        raise PathError("laser.path.sinusoidal needs exactly one of wavelength_m or periods")
    wavelength = _positive(spec, "wavelength_m") if has_wl else length / _positive(spec, "periods")
    # Resolve the sine finely enough that the later equal-arc resample is accurate.
    dx = min(radius / 4.0, wavelength / 60.0)
    n = max(2, int(math.ceil(length / dx)) + 1)
    two_pi = 2.0 * math.pi
    return [
        (x0 + length * i / (n - 1), y0 + amplitude * math.sin(two_pi * (length * i / (n - 1)) / wavelength))
        for i in range(n)
    ]


def _random_control(spec: dict, allowed: tuple[float, float, float, float], start: Point, radius: float) -> list[Point]:
    """Coverage-biased random walk: at each step steer, within the ``max_turn``
    smoothness cap, toward the least-visited neighbourhood so the path fills
    unexplored domain, walking until it reaches ``length_m``.

    ``max_turn`` caps the per-step turn in open space; recovering from a domain
    wall needs a sharper turn, so a wall-peel junction may exceed it.  In a
    domain smaller than ``length_m`` the walk self-intersects to reach the length
    (self-intersection is allowed for now)."""
    seed = spec.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PathError("laser.path.seed must be an integer")
    target_len = _positive(spec, "length_m")
    step = _positive_default(spec, "step_m", 4.0 * radius)
    max_turn = math.radians(_positive_default(spec, "max_turn_deg", 45.0))
    ax_min, ax_max, ay_min, ay_max = allowed
    if ax_max <= ax_min or ay_max <= ay_min:
        raise PathError("domain is too small for a random path at this beam radius")

    rng = random.Random(seed)
    # Visitation grid over the allowed region; one cell ~ one step.
    gx = max(1, int(math.ceil((ax_max - ax_min) / step)))
    gy = max(1, int(math.ceil((ay_max - ay_min) / step)))
    visits = [[0.0] * gy for _ in range(gx)]

    def cell(p: Point) -> tuple[int, int]:
        ix = min(gx - 1, max(0, int((p[0] - ax_min) / step)))
        iy = min(gy - 1, max(0, int((p[1] - ay_min) / step)))
        return ix, iy

    def mark(p: Point) -> None:
        cx, cy = cell(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ix, iy = cx + dx, cy + dy
                if 0 <= ix < gx and 0 <= iy < gy:
                    visits[ix][iy] += 1.0 / (1 + abs(dx) + abs(dy))

    def score(p: Point) -> float:
        # Average visitation over the in-region neighbour cells.  Averaging (not
        # summing) is deliberate: the forbidden region outside the cropped domain
        # contributes no cells, so an edge candidate is not made artificially
        # attractive by having fewer neighbours -- the walk is not biased to hug
        # the boundary.
        cx, cy = cell(p)
        total = 0.0
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ix, iy = cx + dx, cy + dy
                if 0 <= ix < gx and 0 <= iy < gy:
                    total += visits[ix][iy]
                    count += 1
        return total / count if count else 0.0

    def inside(p: Point) -> bool:
        return ax_min <= p[0] <= ax_max and ay_min <= p[1] <= ay_max

    cx, cy = (ax_min + ax_max) / 2.0, (ay_min + ay_max) / 2.0
    controls = [start]
    mark(start)
    # Head generally inward (toward the region centre) from the start, with spread,
    # so an edge start (e.g. left-middle) walks into the region instead of along it.
    heading = math.atan2(cy - start[1], cx - start[0]) + rng.uniform(-1.0, 1.0)
    travelled = 0.0
    # Wall-peel: when boxed in against an edge, rotate toward the interior by up to
    # max_turn WITHOUT stepping and retry, instead of giving up. This lets the walk
    # slide along a wall and keep exploring until it reaches the target length
    # (self-intersecting if the domain is smaller than the requested length).
    # `stuck` bounds the rotate-in-place retries so a genuinely trapped walk still
    # terminates; a full turn toward the interior always frees an interior point.
    max_controls = min(MAX_PATH_SAMPLES * 4, 8000)
    max_stuck = int(math.ceil(2.0 * math.pi / max_turn)) + 2
    stuck = 0
    while travelled < target_len and len(controls) < max_controls:
        cur = controls[-1]
        best = None
        # Candidate headings spread across the smoothness cap.  Every candidate
        # turn is clamped to +/- max_turn so the documented cap always holds.
        for k in range(12):
            delta = max_turn * (2.0 * k / 11.0 - 1.0) + rng.uniform(-0.05, 0.05) * max_turn
            delta = max(-max_turn, min(max_turn, delta))
            angle = heading + delta
            cand = (cur[0] + step * math.cos(angle), cur[1] + step * math.sin(angle))
            if not inside(cand):
                continue
            s = score(cand) + rng.uniform(0.0, 0.25)  # tie-break / exploration noise
            if best is None or s < best[0]:
                best = (s, cand, angle)
        if best is None:
            # Boxed in: rotate toward the interior (up to max_turn) and retry; do
            # not step or append, so the smoothness cap holds and the walk peels
            # along the wall rather than dead-ending.
            stuck += 1
            if stuck > max_stuck:
                break  # degenerate domain; should not happen from an interior point
            diff = _wrap_pi(math.atan2(cy - cur[1], cx - cur[0]) - heading)
            heading += max_turn if abs(diff) < 1.0e-9 else math.copysign(min(max_turn, abs(diff)), diff)
            continue
        stuck = 0
        _, nxt, heading = best
        controls.append(nxt)
        mark(nxt)
        travelled += math.dist(cur, nxt)
    if len(controls) < 2:
        raise PathError("random path could not take a step; enlarge the domain or reduce step_m")
    return controls


# --------------------------------------------------------------------------- #
# Hard safety check
# --------------------------------------------------------------------------- #
def check_within_margin(samples: list[Point], bounds: tuple[float, float, float, float], margin: float) -> None:
    """Reject any sample within ``margin`` of a domain edge (the 4-radius rule)."""
    x_min, x_max, y_min, y_max = bounds
    lo_x, hi_x, lo_y, hi_y = x_min + margin, x_max - margin, y_min + margin, y_max - margin
    if lo_x > hi_x or lo_y > hi_y:
        raise PathError(
            f"domain is smaller than 2x the {margin:g} m keep-out margin (4 beam radii); "
            "enlarge geometry.length_m/width_m"
        )
    eps = 1.0e-12 * max(1.0, abs(x_max - x_min), abs(y_max - y_min))
    for x, y in samples:
        if x < lo_x - eps or x > hi_x + eps or y < lo_y - eps or y > hi_y + eps:
            raise PathError(
                f"laser path point ({x:g}, {y:g}) m is within 4 beam radii ({margin:g} m) of a domain edge; "
                f"the path must stay inside x=[{lo_x:g}, {hi_x:g}], y=[{lo_y:g}, {hi_y:g}]"
            )


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #
def build_path(
    spec: dict,
    *,
    radius: float,
    bounds: tuple[float, float, float, float],
) -> LaserPath:
    """Resolve a ``laser.path`` spec to an equal-arc-length sample list.

    ``bounds`` is the mesh rectangle ``(x_min, x_max, y_min, y_max)``; the path
    must stay ``4 * radius`` inside it.  The constant scan speed is applied later,
    when the samples are turned into a velocity table (``velocity_time_tables``).
    """
    kind = _require_type(spec)
    if kind == "variable_speed":
        return _build_variable_speed(spec, radius=radius, bounds=bounds)
    margin = 4.0 * radius
    x_min, x_max, y_min, y_max = bounds
    clamp_region: tuple[float, float, float, float] | None = None

    if kind == "waypoints":
        raw = spec.get("points_m")
        if not isinstance(raw, list) or len(raw) < 2:
            raise PathError("laser.path.points_m must list at least two [x, y] points")
        control = [_xy(p, f"laser.path.points_m[{i}]") for i, p in enumerate(raw)]
        interpolation = str(spec.get("interpolation", "linear")).strip().lower()
        if interpolation not in ("linear", "spline"):
            raise PathError("laser.path.interpolation must be linear or spline")
        dense = control if interpolation == "linear" else catmull_rom(control)
    elif kind == "sinusoidal":
        interpolation = "spline"
        dense = _sinusoidal_dense(spec, radius)
    else:  # random
        interpolation = "spline"
        # The space-filling walk considers only the cropped region inside the
        # 4-radius keep-out (the forbidden margin is not treated as unexplored, so
        # it does not bias the walk outward).  Catmull-Rom can overshoot the
        # control polygon, so samples are clamped back into this region below.
        crop = (x_min + margin, x_max - margin, y_min + margin, y_max - margin)
        if crop[1] <= crop[0] or crop[3] <= crop[2]:
            raise PathError(
                f"domain is smaller than 2x the {margin:g} m keep-out margin (4 beam radii); "
                "enlarge geometry.length_m/width_m"
            )
        clamp_region = crop
        # Default start: left edge, mid height, of the cropped region.
        start = (crop[0], 0.5 * (crop[2] + crop[3]))
        given = spec.get("start_point_m")
        if given is not None:
            gx, gy = _xy(given, "laser.path.start_point_m")
            start = (min(max(gx, crop[0]), crop[1]), min(max(gy, crop[2]), crop[3]))
        # length_m defaults to the domain length (x extent = cropped length + the
        # two 4-radius keep-out margins).  Fill it into the spec the walk reads.
        target = _positive_default(spec, "length_m", x_max - x_min)
        spec = {**spec, "length_m": target}
        control = _random_control(spec, crop, start, radius)
        # The walk over-shoots the requested length (it stops once the control
        # polygon reaches it, and the spline is longer still); trim the smoothed
        # curve to the exact length so laser.path.length_m controls it precisely.
        dense = _trim_polyline(catmull_rom(control), target)

    length = _polyline_length(dense)
    if length <= 0.0:
        raise PathError("laser.path resolves to zero length")

    spacing = spec.get("sample_spacing_m")
    if spacing is None:
        spacing = radius / 2.0  # two samples per beam radius, matching the mesh floor
    else:
        spacing = _positive(spec, "sample_spacing_m")
    n_samples = max(2, int(round(length / spacing)) + 1)
    if n_samples > MAX_PATH_SAMPLES:
        raise PathError(
            f"laser.path needs {n_samples} samples (> {MAX_PATH_SAMPLES}); "
            "increase laser.path.sample_spacing_m or shorten the path"
        )
    samples = resample_by_arclength(dense, n_samples)
    if clamp_region is not None:
        cx0, cx1, cy0, cy1 = clamp_region
        samples = [(min(max(x, cx0), cx1), min(max(y, cy0), cy1)) for x, y in samples]
    check_within_margin(samples, bounds, margin)
    resampled_len = _polyline_length(samples)
    return LaserPath(
        samples=samples,
        length_m=resampled_len,
        spacing_m=resampled_len / (n_samples - 1),
        type=kind,
        interpolation=interpolation,
    )


def _build_variable_speed(
    spec: dict,
    *,
    radius: float,
    bounds: tuple[float, float, float, float],
) -> LaserPath:
    """Resolve a ``variable_speed`` path: explicit position-time knots.

    ``knots_m`` are the beam positions verbatim (no arc-length resample -- the
    per-leg timing IS the speed control), ``times_s`` the arrival time at each
    knot (strictly increasing, first == 0), and optional ``power_w`` a per-leg
    power (len == knots-1) for de-rating dwells.  A repeated knot with advancing
    time is a true dwell (zero velocity).  The 4-radius keep-out still applies to
    every knot.
    """
    raw = spec.get("knots_m")
    if not isinstance(raw, list) or len(raw) < 2:
        raise PathError("laser.path.knots_m must list at least two [x, y] points")
    knots = [_xy(p, f"laser.path.knots_m[{i}]") for i, p in enumerate(raw)]

    times = spec.get("times_s")
    if not isinstance(times, list) or len(times) != len(knots):
        raise PathError("laser.path.times_s must have one time per knot (same length as knots_m)")
    try:
        times = [float(t) for t in times]
    except (TypeError, ValueError) as error:
        raise PathError("laser.path.times_s must contain finite numbers") from error
    if not all(math.isfinite(t) for t in times):
        raise PathError("laser.path.times_s must contain finite numbers")
    if abs(times[0]) > 1.0e-15:
        raise PathError("laser.path.times_s must start at 0")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise PathError("laser.path.times_s must be strictly increasing")

    powers = spec.get("power_w")
    if powers is not None:
        if not isinstance(powers, list) or len(powers) != len(knots) - 1:
            raise PathError("laser.path.power_w must have one value per leg (len(knots_m) - 1)")
        try:
            powers = [float(p) for p in powers]
        except (TypeError, ValueError) as error:
            raise PathError("laser.path.power_w must contain finite numbers") from error
        if any(not math.isfinite(p) or p < 0 for p in powers):
            raise PathError("laser.path.power_w must be finite and non-negative")

    check_within_margin(knots, bounds, 4.0 * radius)
    length = _polyline_length(knots)
    if length <= 0.0:
        raise PathError("laser.path variable_speed knots resolve to zero length (all points coincide)")
    return LaserPath(
        samples=knots,
        length_m=length,
        spacing_m=length / (len(knots) - 1),
        type="variable_speed",
        interpolation="linear",
        times_s=times,
        powers_w=powers,
    )


def variable_speed_time_tables(
    default_power: float,
    samples: list[Point],
    times: list[float],
    turn_off: bool,
    powers: list[float] | None = None,
) -> dict[str, str]:
    """Render the beam table for a variable-speed path from explicit knot times.

    Each leg ``i`` runs from ``samples[i]`` at ``times[i]`` to ``samples[i+1]`` at
    ``times[i+1]``; its velocity is ``chord / (times[i+1] - times[i])`` (zero for a
    dwell, where the two knots coincide).  Same plateau idiom as
    ``velocity_time_tables`` -- two rows per leg at the leg velocity/power, an
    ``eps`` step between plateaus -- but the row times come straight from the
    knots instead of a single constant speed.  ``turn_off`` appends a final zero
    row to stop the beam.
    """
    if len(samples) < 2 or len(times) != len(samples):
        raise PathError("variable_speed needs >=2 knots and one time per knot")
    legs = list(zip(samples, samples[1:]))
    dts = [times[i + 1] - times[i] for i in range(len(legs))]
    if any(dt <= 0.0 for dt in dts):
        raise PathError("variable_speed times must be strictly increasing")
    scan_time = times[-1] - times[0]
    # eps switches velocity between plateaus: below the shortest leg (strict
    # monotonic times) and below the runner's laser-off tolerance (scan_time*1e-9).
    eps = min(min(dts) * 1.0e-6, scan_time * 1.0e-9)

    def fmt(v: float) -> str:
        return format(v, ".12g")

    out_times, us, vs, out_powers = [], [], [], []
    for i, ((ax, ay), (bx, by)) in enumerate(legs):
        dt = dts[i]
        ux, uy = (bx - ax) / dt, (by - ay) / dt  # 0,0 for a dwell
        p = default_power if powers is None else powers[i]
        out_times += [times[i] if i == 0 else times[i] + eps, times[i + 1]]
        us += [ux, ux]
        vs += [uy, uy]
        out_powers += [p, p]
    if turn_off:
        out_times.append(times[-1] + eps)
        us.append(0.0)
        vs.append(0.0)
        out_powers.append(0.0)

    power_lines = [f"    powlbm({r + 1},1)={fmt(p)}," for r, p in enumerate(out_powers)]
    vel_lines = [f"    utlbm({r + 1},1)={fmt(u)}," for r, u in enumerate(us)]
    vel_lines += [f"    vtlbm({r + 1},1)={fmt(v)}," for r, v in enumerate(vs)]
    vel_lines += [f"    wtlbm({r + 1},1)=0.0," for r in range(len(us))]
    time_lines = [f"    timlbm({r + 1})={fmt(t)}," for r, t in enumerate(out_times)]
    return {
        "__LASER_POWER_TABLE__": "\n".join(power_lines),
        "__LASER_VELOCITY_TABLE__": "\n".join(vel_lines),
        "__LASER_TIME_TABLE__": "\n".join(time_lines),
    }


def velocity_time_series(
    power: float, samples: list[Point], speed: float, turn_off: bool
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Raw ``(times, us, vs, powers)`` for a path -- the values behind the tables.

    Split out so multi-beam runs can resample several independent paths onto one
    shared time grid: FLOW-3D indexes ``powlbm``/``utlbm``/``vtlbm`` by
    ``(row, beam)`` but ``timlbm`` by row alone, so beams share a time base.
    ``velocity_time_tables`` formats this for the single-beam case.
    """
    return _series(power, samples, speed, turn_off)


def velocity_time_tables(power: float, samples: list[Point], speed: float, turn_off: bool) -> dict[str, str]:
    """Render the ``powlbm``/``utlbm``/``vtlbm``/``timlbm`` rows for a path.

    Each leg between consecutive samples is written as a constant-velocity
    *plateau*: two rows at the same velocity, at the leg's start and end times,
    with the leg duration set to ``chord / speed`` so the beam moves at exactly
    ``speed`` on every leg (equal-arc-length resampling alone is not enough — a
    leg's straight chord is shorter than its arc at a corner).  A tiny time step
    (``eps``) separates one plateau from the next.  This matches the straight
    scan's plateau idiom (``laser_time_tables``, A6): under FLOW-3D's linear
    interpolation of the beam tables the plateau holds the velocity constant and
    the beam traces the sampled polyline; under piecewise-constant tables the
    duplicate rows are simply redundant.  So the trajectory is correct either
    way.  With ``turn_off`` a final zero row stops the beam on the last sample.
    """
    times, us, vs, powers = _series(power, samples, speed, turn_off)

    def fmt(v: float) -> str:
        return format(v, ".12g")

    power_lines = [f"    powlbm({r + 1},1)={fmt(p)}," for r, p in enumerate(powers)]
    vel_lines = [f"    utlbm({r + 1},1)={fmt(u)}," for r, u in enumerate(us)]
    vel_lines += [f"    vtlbm({r + 1},1)={fmt(v)}," for r, v in enumerate(vs)]
    vel_lines += [f"    wtlbm({r + 1},1)=0.0," for r in range(len(us))]
    time_lines = [f"    timlbm({r + 1})={fmt(t)}," for r, t in enumerate(times)]
    return {
        "__LASER_POWER_TABLE__": "\n".join(power_lines),
        "__LASER_VELOCITY_TABLE__": "\n".join(vel_lines),
        "__LASER_TIME_TABLE__": "\n".join(time_lines),
    }


def _series(power: float, samples: list[Point], speed: float, turn_off: bool):
    # Drop consecutive duplicate points so every leg has positive length.
    pts = [samples[0]]
    for p in samples[1:]:
        if math.dist(pts[-1], p) > 0.0:
            pts.append(p)
    if len(pts) < 2:
        raise PathError("path collapses to a single point after resampling")

    def fmt(v: float) -> str:
        return format(v, ".12g")

    legs = list(zip(pts, pts[1:]))
    dts = [math.dist(a, b) / speed for a, b in legs]  # constant speed => dt per chord
    scan_time = sum(dts)
    # eps switches velocity between plateaus.  Kept below the shortest leg (strict
    # monotonic times) and below the runner's laser-off tolerance (scan_time*1e-9)
    # so the off event still lands before finish_time.
    eps = min(min(dts) * 1.0e-6, scan_time * 1.0e-9)

    times, us, vs, powers = [], [], [], []
    t = 0.0
    for i, ((ax, ay), (bx, by)) in enumerate(legs):
        dt = dts[i]
        ux, uy = (bx - ax) / dt, (by - ay) / dt  # |(ux, uy)| == speed exactly
        times += [t if i == 0 else t + eps, t + dt]  # plateau start, plateau end
        us += [ux, ux]
        vs += [uy, uy]
        powers += [power, power]
        t += dt
    if turn_off:
        times.append(t + eps)
        us.append(0.0)
        vs.append(0.0)
        powers.append(0.0)
    # else: the last plateau holds the final velocity (beam keeps moving), the
    # same as a straight scan with turn_off_at_end=false.
    return times, us, vs, powers
