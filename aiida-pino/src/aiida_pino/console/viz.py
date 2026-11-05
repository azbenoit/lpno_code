"""Per-case visualization + melt-pool statistics for the web console.

Plot style follows the web demo (inferno, fixed T range, spline-zoomed
slices with nearest interpolation, melt contour overlay, top view = topmost
metal projection). Stats follow lpno'
materials_dataset_exploration notebook and are cached in the simulation's
extras under ``meltpool_stats`` (mutable, versioned).
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

MELT_CONTOUR_K = 1928.0  # Ti64 liquidus fallback; per-case liquidus overrides when known.
_DISPLAY_ZOOM = 3
STATS_VERSION = 8  # v8: melt depth/width ARE the time-union fusion geometry (peak-frame kept as melt_peak_frame_*)
REGIME_KEYWORDS = ("lof", "conduction", "keyhole")
STATS_EXTRA_KEY = "meltpool_stats"

_FIELD_UNITS = {
    "Temperature": "K",
    "Pressure": "Pa",
    "Fraction Of Fluid": "-",
    "Melt Region": "-",
    "Solid Fraction": "-",
    "Velocity": "m/s",
}

VIEWS = ("surface", "long", "cross")


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------- array access
# Stacks above this size never enter the LRU: four ~1.5 GB frames-x-grid
# stacks (400-frame keyhole150 cases) pinned at once OOM-killed the process.
_CACHE_STACK_MAX_BYTES = 400 * 1024 * 1024


def _load_field_stack(uuid: str, alias: str) -> np.ndarray:
    from aiida.orm import load_node

    sim = load_node(uuid)
    arr = np.asarray(sim.get_array(alias))
    if arr.ndim == 5 and arr.shape[1] > 1:  # vector field -> magnitude
        arr = np.sqrt(np.sum(arr.astype(np.float32) ** 2, axis=1))
    elif arr.ndim == 5:
        arr = arr[:, 0]
    return np.asarray(arr, dtype=np.float32)


@lru_cache(maxsize=4)
def _field_stack_lru(uuid: str, alias: str) -> np.ndarray:
    return _load_field_stack(uuid, alias)


def _field_stack_cached(uuid: str, alias: str) -> np.ndarray:
    """(T, nx, ny, nz) float32 stack for one field of one simulation.

    Velocity collapses to magnitude. Small stacks go through a 4-entry LRU
    (repeat frames of the animated viewer); oversized stacks are loaded,
    used, and dropped so a big case cannot pin gigabytes.
    """
    from aiida.orm import load_node

    sim = load_node(uuid)
    shapes = sim.base.attributes.get("array_shapes") or {}
    shape = shapes.get(alias) or []
    est = 4 * int(np.prod([s for s in shape if s])) if shape else 0
    if est > _CACHE_STACK_MAX_BYTES:
        return _load_field_stack(uuid, alias)
    return _field_stack_lru(uuid, alias)


def _reduce_axis(a: np.ndarray, alias: str) -> np.ndarray:
    """Canonical axes are broadcast to (T, 1, nx, ny, nz); take the 1-D axis.

    The result is COPIED, and that copy is the whole point. Basic slicing
    returns a VIEW which keeps the entire float64 grid alive through `.base`,
    and these results go straight into a 64-entry `lru_cache`. Measured on
    one case: four axes holding 0.0023 MiB of data pinned 0.660 GiB, so the
    cache had a 42 GiB ceiling for arrays nothing would read again. That is
    what grew `pino-stats-autocompute` to 11 GiB over four cases, with
    `gc.collect()` freeing none of it -- the cache holds a strong reference,
    so the objects were reachable and gc was never going to help.
    """
    if a.ndim == 5:
        if alias == "x":
            out = a[0, 0, :, 0, 0]
        elif alias == "y":
            out = a[0, 0, 0, :, 0]
        elif alias == "z":
            out = a[0, 0, 0, 0, :]
        else:
            out = a[:, 0, 0, 0, 0]  # time_s
    else:
        out = a.reshape(-1)
    return np.array(out, dtype=np.float64, copy=True)


@lru_cache(maxsize=64)
def _axes_cached(uuid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1-D x, y, z axes (m) + time axis (s)."""
    from aiida.orm import load_node

    sim = load_node(uuid)

    def axis(alias: str) -> np.ndarray:
        return _reduce_axis(np.asarray(sim.get_array(alias), dtype=np.float64), alias)

    return axis("x"), axis("y"), axis("z"), axis("time_s")


@lru_cache(maxsize=16)
def _field_range_cached(uuid: str, alias: str) -> tuple[float, float]:
    stack = _field_stack_cached(uuid, alias)
    return float(np.nanmin(stack)), float(np.nanmax(stack))


def clear_caches() -> None:
    _field_stack_lru.cache_clear()
    _axes_cached.cache_clear()
    # was missing: a clear_caches() that leaves a cache populated is a trap for
    # whoever next tries to free memory by calling it.
    _field_range_cached.cache_clear()
    _field_range_cached.cache_clear()


def _zoom_slice(arr2d: np.ndarray, factor: int = _DISPLAY_ZOOM) -> np.ndarray:
    if factor <= 1:
        return np.asarray(arr2d)
    try:
        from scipy.ndimage import zoom

        return zoom(np.asarray(arr2d, dtype=np.float32), factor, order=1)
    except Exception:  # noqa: BLE001 - scipy optional; fall back to native grid
        return np.asarray(arr2d)


def _top_metal_projection(vol: np.ndarray, fof: np.ndarray | None) -> np.ndarray:
    """Value at the topmost metal cell per (x, y) column (fof >= 0.5), else the
    top slice — the demo's melt-surface view (vapor gap not shown)."""
    if fof is None:
        return vol[:, :, -1]
    nx, ny, nz = vol.shape
    metal = fof >= 0.5
    # index of highest z with metal; columns without metal fall back to top slice
    rev = metal[:, :, ::-1]
    idx_top = nz - 1 - np.argmax(rev, axis=2)
    has_metal = metal.any(axis=2)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    out = vol[ii, jj, np.where(has_metal, idx_top, nz - 1)]
    return out


def frame_info(sim) -> dict[str, Any]:
    _, _, _, times = _axes_cached(str(sim.uuid))
    return {
        "num_frames": int(times.size),
        "times_s": [float(t) for t in times],
    }


def render_case_frame_png(
    sim,
    *,
    field: str,
    view: str = "long",
    frame: int = -1,
    contour: bool = True,
) -> bytes:
    """One PNG frame: surface (xy, topmost metal), long (xz @ mid-y) or cross
    (yz @ melt-pool x) slice, demo-style."""
    plt = _matplotlib()
    uuid = str(sim.uuid)
    aliases = sim.base.attributes.get("array_aliases") or {}
    if field not in aliases:
        raise ValueError(f"unknown field {field!r}; available: {sorted(k for k in aliases if k not in ('x','y','z','P','V_scan','time_s'))}")
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")

    stack = _field_stack_cached(uuid, field)
    x, y, z, times = _axes_cached(uuid)
    T = stack.shape[0]
    fi = T - 1 if frame is None or int(frame) < 0 else max(0, min(int(frame), T - 1))
    vol = stack[fi]

    fof = None
    if "Fraction Of Fluid" in aliases:
        fof = _field_stack_cached(uuid, "Fraction Of Fluid")[fi]
    temp = _field_stack_cached(uuid, "Temperature")[fi] if "Temperature" in aliases else None

    # Stable color range over the whole animation (no per-frame flicker).
    vmin, vmax = _field_range_cached(uuid, field)
    if field == "Temperature":
        vmin = min(max(vmin, 250.0), 300.0)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0

    um = 1e6
    if view == "surface":
        sl = _top_metal_projection(vol, fof).T  # (ny, nx)
        t_sl = _top_metal_projection(temp, fof).T if temp is not None else None
        extent = [x[0] * um, x[-1] * um, y[0] * um, y[-1] * um]
        xlabel, ylabel, title = "x (µm)", "y (µm)", "surface (top metal)"
    elif view == "long":
        # Slice through the BEAM, not the middle of the box. On a random path the
        # beam is rarely at mid-y, so the old mid-y slice showed undisturbed metal
        # and the pool was simply absent from the view.
        beam = beam_xy_index(uuid, aliases, fi)
        yi = beam[1] if beam else vol.shape[1] // 2
        sl = vol[:, yi, :].T  # (nz, nx)
        t_sl = temp[:, yi, :].T if temp is not None else None
        extent = [x[0] * um, x[-1] * um, z[0] * um, z[-1] * um]
        where = "beam" if beam else "mid-domain"
        xlabel, ylabel, title = "x (µm)", "z (µm)", f"longitudinal @ y={y[yi]*um:.0f} µm ({where})"
    else:  # cross: yz plane through the beam, else the hottest column, else mid-x
        beam = beam_xy_index(uuid, aliases, fi)
        if beam:
            xi = beam[0]
        elif temp is not None:
            xi = int(np.unravel_index(np.argmax(temp), temp.shape)[0])
        else:
            xi = vol.shape[0] // 2
        sl = vol[xi, :, :].T  # (nz, ny)
        t_sl = temp[xi, :, :].T if temp is not None else None
        extent = [y[0] * um, y[-1] * um, z[0] * um, z[-1] * um]
        xlabel, ylabel, title = "y (µm)", "z (µm)", f"transverse @ x={x[xi]*um:.0f} µm"

    dpi = 100
    fig = plt.figure(figsize=(6.4, 4.4), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        _zoom_slice(sl), origin="lower", cmap="inferno", aspect="equal",
        extent=extent, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    t_ms = times[fi] * 1e3
    ax.set_title(f"{field} — {title} — t = {t_ms:.3f} ms (frame {fi + 1}/{T})", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    unit = _FIELD_UNITS.get(field, "")
    cbar.set_label(f"{field} [{unit}]" if unit else field)

    if contour and t_sl is not None:
        try:
            xg = np.linspace(extent[0], extent[1], _zoom_slice(t_sl).shape[1])
            vg = np.linspace(extent[2], extent[3], _zoom_slice(t_sl).shape[0])
            cs = ax.contour(*np.meshgrid(xg, vg), _zoom_slice(t_sl), levels=[MELT_CONTOUR_K], colors="cyan", linewidths=1.2)
            if len(cs.allsegs[0]):
                ax.plot([], [], color="cyan", linewidth=1.2, label="meltpool edge")
                ax.legend(loc="lower right", fontsize=8, frameon=True)
        except Exception:  # noqa: BLE001 - contour failure must not kill the frame
            pass

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------ meltpool stats
# Cavity/melt masks are the materials_dataset_exploration.ipynb::case_metrics
# semantics; the regime rule is path-agnostic since v2 (ellipse-fit radii).
#: The fields `compute_case_metrics` pulls in as whole (T, X, Y, Z) stacks.
#: The background worker sizes a case against this list before loading it, so
#: it must stay in step with what the function below actually reads -- hence one
#: definition here rather than a second copy in `utils/stats_autocompute.py`.
FULL_STACK_FIELDS = ("Fraction Of Fluid", "Melt Region", "Temperature")

FOF_GAS = 0.5        # Fraction Of Fluid below this = gas/void
MELT_FRAC = 0.5      # Melt Region above this = melted metal
SURFACE_Z = 0.0      # metal/gas interface (m)
LOF_DEPTH_CELLS = 2  # depression depth < this (cells) => lack of fusion
# Keyhole = a cavity that is deeper than it is narrow *on average over time*
# (mean depth / ellipse-fit r_min above 1), and deep enough in absolute terms
# to be a keyhole at all: at least KEYHOLE_MIN_DEPTH_RADII beam radii.
#
# Calibrated 2026-07-20 on the original Ti64 straight-path family (83 cases,
# H 5.0-12.2) with the boundary confirmed visually on steady-state
# longitudinal sections: H=12.2/a=2.26 is a deep narrow keyhole (-200 um),
# H=8.6/a=1.22 an early keyhole with a steep front wall, H=7.7/a=1.01 a
# transitional bowl, H=6.8/a=0.55 plainly conduction. The mean-aspect=1
# crossing alone sits at H~8 (keyhole >= 7.7, conduction <= 8.6) but marks
# 72% of the family keyhole; the depth gate brings it to 30/83 = 36% (about
# a third, as expected) and its keyhole band starts at H=10.4.
#
# The depth gate is also what separates the 62 fixed-P,V random-path cases:
# their tracks leave shallow speckled depressions (40-100 um) whose small
# connected components read mean aspect 1.05-1.42, i.e. nominally above the
# aspect boundary, but visually they are conduction — the gate classifies all
# 62 as conduction, matching the surface plots.
#
# Recalibrated 2026-07-23 (2.5 -> 2.1 radii) against four hand-labelled
# straight-path cases: 130W_0561mps and 90W_0323mps are keyhole at max cavity
# depth 120 um, while 110W_0589mps (90 um) and 70W_0298mps (70 um) are
# conduction. All four have aspect_mean > 1,
# so only the gate separates them: it must sit above 90 and at most 120 um.
# Cavity depths are quantized to the 10 um cell, leaving 95/105/115 as the
# candidates; 105 um = 2.1 x 50 um beam radius is the maximum-margin choice and
# also stays above the 100 um ceiling of the random-path family above (a gate of
# 100 would have flipped all 62). Keyhole share of the 454 classified cases
# moves 56 -> 90 (12% -> 20%).
KEYHOLE_ASPECT_MEAN = 1.0
KEYHOLE_MIN_DEPTH_RADII = 2.1
DEFAULT_BEAM_RADIUS_M = 5e-5  # fallback when a case records no beam radius
T0_K = 293.15        # reference temperature for normalized enthalpy


def _footprint_radii_um(footprint: np.ndarray, cell_um: float) -> tuple[float, float]:
    """Min/max radii (µm) of an ellipse fitted to a 2-D cavity footprint mask.

    Second-moment fit: for a filled ellipse the coordinate variance along a
    principal axis is (semi-axis)²/4, so semi-axis = 2·σ. A per-cell variance
    of cell²/12 is added so a single cell fits a cell-sized ellipse instead of
    a point. Works for any footprint shape, including curved random tracks —
    the fit follows the elongation direction rather than the x/y axes.
    """
    idx = np.argwhere(footprint)
    if not idx.size:
        return 0.0, 0.0
    pts = idx.astype(np.float64) * cell_um
    cov = np.cov(pts, rowvar=False, bias=True) if pts.shape[0] > 1 else np.zeros((2, 2))
    cov = np.atleast_2d(cov) + np.eye(2) * (cell_um**2 / 12.0)
    eig = np.linalg.eigvalsh(cov)
    rmin, rmax = 2.0 * math.sqrt(float(eig[0])), 2.0 * math.sqrt(float(eig[1]))
    return rmin, rmax


#: Self-describing label for the cached `regime`, built FROM the constants so a
#: threshold change cannot leave a stale description behind. This is the project's
#: regime rule (aiida-pino assumptions.md A18).
REGIME_CRITERION = (
    f"classify_regime: keyhole iff aspect_mean > {KEYHOLE_ASPECT_MEAN} and "
    f"cavity_depth_max >= {KEYHOLE_MIN_DEPTH_RADII}*r0; LoF iff depth_cells < "
    f"{LOF_DEPTH_CELLS}; else conduction. See aiida-pino assumptions.md A18."
)


def classify_regime(
    *, depth_cells: int, aspect_mean: float, cavity_depth_max_um: float, min_depth_um: float
) -> str:
    """LoF / keyhole / conduction from the cavity metrics. See the constants above."""
    if depth_cells < LOF_DEPTH_CELLS:
        return "LoF"
    if aspect_mean > KEYHOLE_ASPECT_MEAN and cavity_depth_max_um >= min_depth_um:
        return "keyhole"
    return "conduction"


def _cavity_component_metrics(g: np.ndarray, nzm: int, cell_um: float) -> tuple[int, float, float]:
    """(depth_cells, rmin_um, rmax_um) of the connected cavity containing the
    deepest depressed column of one frame's gas mask ``g`` (X, Y, Zmetal)."""
    from scipy import ndimage

    fp = g.any(axis=2)
    if not fp.any():
        return 0, 0.0, 0.0
    col_min = np.where(g, np.arange(g.shape[2])[None, None, :], nzm).min(axis=2)
    labels, count = ndimage.label(fp)
    deepest = np.unravel_index(np.argmin(np.where(fp, col_min, nzm + 1)), col_min.shape)
    comp = labels == labels[deepest]
    depth_cells = nzm - int(col_min[comp].min())
    rmin_um, rmax_um = _footprint_radii_um(comp, cell_um)
    return depth_cells, rmin_um, rmax_um


def beam_xy_index(uuid: str, aliases: dict, frame: int) -> tuple[int, int] | None:
    """Beam column (ix, iy) at one frame, from the field that says where it is.

    `Heat Absorption Rate` is where the laser deposits power, so its peak is the
    beam. Temperature is the fallback: the hottest column is under the beam for a
    moving source, though it lags on a fast turn. Returns None when neither field
    was exported, and the caller then keeps the old domain-centre behaviour.

    Reading the beam from the solved field rather than the config means this works
    for random and multi-beam paths, and needs nothing beside the stored arrays.
    """
    for alias, reducer in (("Heat Absorption Rate", np.sum), ("Temperature", np.max)):
        if alias not in aliases:
            continue
        try:
            vol = _field_stack_cached(uuid, alias)[int(frame)]
        except Exception:
            continue
        plane = reducer(vol, axis=2)
        if not np.isfinite(plane).any() or plane.max() <= 0:
            continue
        ix, iy = np.unravel_index(int(np.argmax(plane)), plane.shape)
        return int(ix), int(iy)
    return None


def _across_track_width_um(mg: np.ndarray, track: np.ndarray, cell_um: float) -> float:
    """Melt width measured ACROSS the beam path, not along the domain axes.

    The footprint ellipse fit and the x/y bounding box both describe the whole
    swept track. For a straight scan that is the pool, but for a random path the
    track wanders and those numbers report how far the beam travelled rather than
    how wide its pool is: on the 2026-08-05 multi-beam batch the domain-axis
    numbers reached 690 um for a pool ~440 um across.

    This takes the mid-track point, steps along the normal to the local direction
    of travel, and measures the melt run through it. For a straight track the
    normal is the y axis, so it reduces to the transverse width.
    """
    live = track[np.isfinite(track[:, 0])]
    if live.shape[0] < 2 or not mg.any():
        return 0.0
    step = np.r_[0.0, np.cumsum(np.hypot(*np.diff(live, axis=0).T))]
    if step[-1] <= 0:
        return 0.0
    mid = int(np.searchsorted(step, step[-1] / 2.0))
    mid = min(max(mid, 1), len(live) - 1)
    d = live[mid] - live[mid - 1]
    n = float(np.hypot(*d))
    px, py = (0.0, 1.0) if n == 0 else (-d[1] / n, d[0] / n)
    x0, y0 = live[mid]
    foot = mg.any(axis=2)
    span = int(max(foot.shape))
    hits = 0
    for s in range(-span, span + 1):
        ix, iy = int(round(x0 + s * px)), int(round(y0 + s * py))
        if 0 <= ix < foot.shape[0] and 0 <= iy < foot.shape[1] and foot[ix, iy]:
            hits += 1
    return float(hits) * cell_um


class _SimFields(Mapping):
    """An AiiDA simulation presented as the mapping `melt_metrics` expects.

    `melt_metrics` reaches fields only through `[]` and `.get()`, so nothing has
    to be materialised up front: each field loads on first access through the
    existing LRU cache. Building a real dict would load every field of every
    case (a 331-frame case is expensive) and throw away the streaming the viewer
    depends on.
    """

    _AXES = ("x", "y", "z", "time_s")

    def __init__(self, sim):
        self._uuid = str(sim.uuid)
        self._aliases = dict(sim.base.attributes.get("array_aliases") or {})

    def __getitem__(self, key):
        if key in self._AXES:
            return _axes_cached(self._uuid)[self._AXES.index(key)]
        if key not in self._aliases:
            raise KeyError(key)
        return _field_stack_cached(self._uuid, key)

    def __iter__(self):
        return iter(list(self._aliases) + list(self._AXES))

    def __len__(self):
        return len(self._aliases) + len(self._AXES)


@lru_cache(maxsize=1)
def _melt_metrics_module():
    """`flow3D_scripts/python/melt_metrics.py`, or None with a reason.

    Same optional-import shape as `service.config_schema_fields`: gated on
    `flow3d_scripts_root`, degrading with a note rather than failing, so a
    headless or CI deployment without the sibling checkout still starts.
    """
    import importlib.util
    import sys

    from aiida_pino.console.config import Settings

    root = str(getattr(Settings(), "flow3d_scripts_root", "") or "").strip()
    candidates = []
    if root:
        candidates.append(Path(root) / "python" / "melt_metrics.py")
    # The sibling checkout beside this repo. `flow3d_scripts_root` exists to pin
    # WHICH checkout a submitted run uses, and defaults to empty; reading metric
    # DEFINITIONS is not a submission, so falling back to the sibling keeps the
    # stats path working on a normal dev box while a configured root still wins.
    candidates.append(Path(__file__).resolve().parents[4] / "flow3D_scripts" / "python" / "melt_metrics.py")
    module_path = next((c for c in candidates if c.is_file()), None)
    if module_path is None:
        return None, (
            "melt_metrics.py not found; set AIIDA_PINO_WEB_FLOW3D_SCRIPTS_ROOT "
            "to a flow3D_scripts checkout"
        )
    try:
        name = "_flow3d_melt_metrics"
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        # Register BEFORE exec: melt_metrics defines @dataclass types, and
        # dataclasses resolves each class's own module via
        # `sys.modules.get(cls.__module__).__dict__`. Unregistered that is None
        # and the import dies with a bare AttributeError. (The config_schema
        # precedent never hit this because it defines no dataclass.)
        sys.modules[name] = module
        sys.path.insert(0, str(module_path.parent))
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        finally:
            sys.path.remove(str(module_path.parent))
        return module, ""
    except Exception as exc:  # noqa: BLE001 - a missing sibling is never fatal
        return None, f"melt_metrics unavailable: {type(exc).__name__}"


def _solidus_k(sim) -> float | None:
    """The alloy's solidus from the case manifest.

    `melt_metrics.melt_mask` demands this explicitly rather than guessing,
    because scoring one alloy with another's melting point is a silent error.
    None means the temperature fallback stays off for this case.
    """
    # The queryable extra first: it is what `material_facts` stamped from the
    # manifest, and for cases whose own manifest records no material it carries
    # the value established from a same-case sibling (material_source says
    # which, so a derived value is never mistaken for a first-hand one).
    stamped = sim.base.extras.get("solidus_k", None)
    if isinstance(stamped, (int, float)) and math.isfinite(stamped) and stamped > 0:
        return float(stamped)
    meta = sim.base.attributes.get("simulation_metadata") or {}
    manifest = meta.get("flow3d_case_manifest") or {}
    material = (manifest.get("material") or {}).get("properties") or manifest.get("material") or {}
    for name in ("solidus_k", "T_solidus", "tsolidus", "solidus_temperature_k", "TS1"):
        value = material.get(name)
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value)
    return None


def fusion_geometry_from_melt_metrics(sim) -> tuple[dict[str, Any], str]:
    """Fusion depth/width for one sim, computed by the shared definition.

    Returns ({}, reason) when the module is unavailable or the case carries
    neither `Melt Region` nor a usable temperature fallback -- an empty result
    with a stated reason, never a silently different number.
    """
    module, note = _melt_metrics_module()
    if module is None:
        return {}, note
    fields = _SimFields(sim)
    try:
        if module.melt_mask(fields, solidus_k=_solidus_k(sim)) is None:
            return {}, "no Melt Region and no solidus for the temperature fallback"
        geometry = module.fusion_geometry_auto(fields, solidus_k=_solidus_k(sim))
    except Exception as exc:  # noqa: BLE001 - stats must not break the console
        return {}, f"{type(exc).__name__}: {exc}"
    if geometry.depth_um <= 0.0 and geometry.width_um <= 0.0:
        return {}, "fusion geometry measured zero extent"
    return {
        "fusion_depth_um": float(geometry.depth_um),
        "fusion_width_um": float(geometry.width_um),
        "fusion_aspect": float(geometry.aspect),
        "fusion_source": "melt_metrics",
    }, ""


def compute_case_metrics(sim) -> dict[str, Any]:
    """Melt-pool metrics for one simulation, notebook-identical semantics.

    Depression (recoil cavity) metrics come from Fraction Of Fluid < 0.5 inside
    the metal (z < 0), interior columns only, at the argmax-volume frame; the
    true melt pool from Melt Region > 0.5 at its own argmax frame; tmax over
    the whole case. Raw metrics in cells + dz; derived um/regime added here.
    """
    uuid = str(sim.uuid)
    aliases = sim.base.attributes.get("array_aliases") or {}
    x, y, zc, _times = _axes_cached(uuid)
    dz = float(zc[1] - zc[0]) if zc.size > 1 else 0.0
    metal = zc < SURFACE_Z
    nzm = int(metal.sum())
    out: dict[str, Any] = {"version": STATS_VERSION, "dz_m": dz, "nz_metal": nzm}

    if "Fraction Of Fluid" in aliases and nzm:
        F = _field_stack_cached(uuid, "Fraction Of Fluid")  # (T, X, Y, Z)
        gm = F[:, :, :, metal] < FOF_GAS
        gm[:, [0, -1], :, :] = False
        gm[:, :, [0, -1], :] = False
        volt = gm.sum(axis=(1, 2, 3))
        pf = int(volt.argmax())
        g = gm[pf]
        vol = int(g.sum())
        cell_um = dz * 1e6
        if vol:
            zpl = np.where(g.any(axis=(0, 1)))[0]
            depth = nzm - int(zpl.min())
            xs = np.where(g.any(axis=(1, 2)))[0]
            ys = np.where(g.any(axis=(0, 2)))[0]
            xlen = int(xs.max() - xs.min() + 1)
            ylen = int(ys.max() - ys.min() + 1)
            _, rmin_um, rmax_um = _cavity_component_metrics(g, nzm, cell_um)
        else:
            depth = xlen = ylen = 0
            rmin_um = rmax_um = 0.0
        # Per-frame cavity aspect (depth / ellipse-fit r_min of the connected
        # cavity around the deepest point — the whole-footprint fit would take
        # in the trailing track depression and inflate r_min on curved paths).
        # The regime uses the max over time: a transient keyhole at any frame
        # counts. Frames shallower than the LoF threshold are ignored; a
        # newborn one-cell dimple otherwise always reads "deeper than wide".
        aspect_max, aspect_max_frame = 0.0, None
        aspects: list[float] = []
        depth_max_cells = 0
        for fi in np.where(volt > 0)[0]:
            depth_f, rmin_f, _ = _cavity_component_metrics(gm[int(fi)], nzm, cell_um)
            if depth_f < LOF_DEPTH_CELLS or rmin_f <= 0:
                continue
            depth_max_cells = max(depth_max_cells, depth_f)
            aspect_f = depth_f * cell_um / rmin_f
            aspects.append(aspect_f)
            if aspect_f > aspect_max:
                aspect_max, aspect_max_frame = aspect_f, int(fi)
        out.update(
            vol_cells=vol, depth_cells=depth, xlen_cells=xlen, ylen_cells=ylen, peak_frame=pf,
            cavity_rmin_um=rmin_um, cavity_rmax_um=rmax_um,
            aspect_max=aspect_max, aspect_max_frame=aspect_max_frame,
            aspect_mean=float(np.mean(aspects)) if aspects else 0.0,
            cavity_depth_max_um=depth_max_cells * cell_um,
        )

    if "Melt Region" in aliases and nzm:
        M = _field_stack_cached(uuid, "Melt Region")[:, :, :, metal] > MELT_FRAC
        mvolt = M.sum(axis=(1, 2, 3))
        mpf = int(mvolt.argmax())
        mg = M[mpf]
        mvol = int(mg.sum())
        mdepth = (nzm - int(np.where(mg.any(axis=(0, 1)))[0].min())) if mvol else 0
        out.update(melt_vol_cells=mvol, melt_depth_cells=mdepth, melt_peak_frame=mpf)
        # Same ellipse fit as the cavity, on the melt footprint at its peak
        # frame: r_min is the half-width, r_max the half-length, both following
        # the pool's own elongation rather than the x/y axes.
        if mvol:
            mfp = mg.any(axis=2)
            mrmin, mrmax = _footprint_radii_um(mfp, dz * 1e6)
            mxs = np.where(mfp.any(axis=1))[0]
            mys = np.where(mfp.any(axis=0))[0]
            out.update(
                melt_rmin_um=mrmin, melt_rmax_um=mrmax,
                melt_xlen_cells=int(mxs.max() - mxs.min() + 1),
                melt_ylen_cells=int(mys.max() - mys.min() + 1),
            )
            # Width ACROSS the beam path. The three numbers above describe the
            # swept track, which equals the pool only when the scan is straight.
            track = np.array([beam_xy_index(uuid, aliases, fi) or (np.nan, np.nan)
                              for fi in range(M.shape[0])], float)
            across = _across_track_width_um(mg, track, dz * 1e6)
            if across > 0:
                out["melt_across_track_um"] = across
        else:
            out.update(melt_rmin_um=0.0, melt_rmax_um=0.0, melt_xlen_cells=0, melt_ylen_cells=0)

    if "Temperature" in aliases:
        out["tmax_k"] = float(_field_stack_cached(uuid, "Temperature").max())

    # Derived (um, aspect, regime) — notebook cell 7 semantics, cubic cells.
    dz_um = dz * 1e6
    for key in ("depth", "xlen", "ylen", "melt_depth", "melt_xlen", "melt_ylen"):
        if f"{key}_cells" in out:
            out[f"{key}_um"] = out[f"{key}_cells"] * dz_um
    # Width/length are the full ellipse axes (twice the fitted radii), which is
    # what "melt pool width/length" means in the welding literature.
    for prefix in ("cavity", "melt"):
        if f"{prefix}_rmin_um" in out:
            out[f"{prefix}_width_um"] = 2.0 * out[f"{prefix}_rmin_um"]
            out[f"{prefix}_length_um"] = 2.0 * out[f"{prefix}_rmax_um"]
    if "vol_cells" in out:
        out["vol_um3"] = out["vol_cells"] * dz_um**3
    if "melt_vol_cells" in out:
        out["melt_vol_um3"] = out["melt_vol_cells"] * dz_um**3
    if "depth_cells" in out and "ylen_cells" in out:
        out["aspect_depth_over_ylen"] = (out["depth_cells"] / out["ylen_cells"]) if out["ylen_cells"] else 0.0
        # Path-agnostic regime: time-averaged cavity aspect (depth divided by
        # the ellipse-fit min radius of the laser-local cavity) plus an
        # absolute depth gate in beam radii. See the constants above.
        beam_radius_m = sim.base.extras.get("beam_radius", None)
        try:
            beam_radius_um = float(beam_radius_m) * 1e6
        except (TypeError, ValueError):
            beam_radius_um = DEFAULT_BEAM_RADIUS_M * 1e6
        if not beam_radius_um or beam_radius_um <= 0:
            beam_radius_um = DEFAULT_BEAM_RADIUS_M * 1e6
        min_depth_um = KEYHOLE_MIN_DEPTH_RADII * beam_radius_um
        out["keyhole_min_depth_um"] = min_depth_um
        out["regime_criterion"] = REGIME_CRITERION
        out["regime"] = classify_regime(
            depth_cells=out["depth_cells"],
            aspect_mean=out.get("aspect_mean", 0.0),
            cavity_depth_max_um=out.get("cavity_depth_max_um", 0.0),
            min_depth_um=min_depth_um,
        )

    # Fusion geometry from the SHARED definition rather than a second local
    # copy. This is what gives melt-pool numbers to cases with no `Melt Region`
    # (the ML training profile exports only FoF/Pressure/Temperature/Velocity;
    # all 150 cases of the variable-speed set are such cases), because
    # `melt_mask` falls back to T > solidus there.
    fusion, why = fusion_geometry_from_melt_metrics(sim)
    if fusion:
        out.update(fusion)
        # The headline melt numbers ARE the time union from v8 on. A section
        # shows the maximum extent ever melted, not the melt at one instant, so
        # the union is the quantity that compares against metallography; the
        # peak-frame values answer a different question ("largest instantaneous
        # pool") that nothing is measured against. Equivalence over 12 cases:
        # 6 of 7 comparable depths identical to the micron, one a single cell
        # apart, across-track width agreeing to <0.001% where both existed.
        #
        # Nothing is lost: whatever the local computation produced is preserved
        # under melt_peak_frame_*, so a v6/v7 number can still be recovered and
        # compared.
        # ONLY the two quantities the equivalence pass actually compared:
        # depth and across-track width. `melt_width_um`/`melt_length_um` are the
        # ellipse axes of the melt FOOTPRINT, which has no union counterpart in
        # melt_metrics -- overwriting them with the section width would swap one
        # definition for a different one with no measurement behind the swap.
        for key in ("melt_depth_um", "melt_across_track_um"):
            if key in out:
                out[f"{key.replace('melt_', 'melt_peak_frame_', 1)}"] = out[key]
        out["melt_depth_um"] = fusion["fusion_depth_um"]
        out["melt_across_track_um"] = fusion["fusion_width_um"]
        out["melt_geometry_basis"] = "time_union (depth, across-track width)"
    elif why:
        out["fusion_note"] = why
        if "melt_depth_um" in out:
            # Says plainly that these are the older, peak-frame numbers rather
            # than leaving the basis to be guessed from their absence.
            out["melt_geometry_basis"] = "peak_frame"

    h = _normalized_enthalpy(sim)
    if h is not None:
        out["H"] = h
    return out


def _normalized_enthalpy(sim) -> float | None:
    """King-Rubenchik normalized enthalpy from the case manifest material block;
    None when the material properties are not all available."""
    meta = sim.base.attributes.get("simulation_metadata") or {}
    manifest = meta.get("flow3d_case_manifest") or {}
    material = (manifest.get("material") or {}).get("properties") or manifest.get("material") or {}
    P = meta.get("laser_power") or sim.base.attributes.get("laser_power")
    V = meta.get("V_scan") or sim.base.attributes.get("V_scan")

    def pick(*names):
        for n in names:
            v = material.get(n)
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        return None

    rho = pick("density_kg_m3", "rho", "density")
    cp = pick("specific_heat_j_kg_k", "cp", "specific_heat")
    k = pick("thermal_conductivity_w_m_k", "k", "thermal_conductivity")
    lat = pick("latent_heat_j_kg", "latent_heat_fusion_j_kg", "latent_heat", "latent")
    tl = pick("liquidus_temperature_k", "Tl", "liquidus")
    # Real absorptivity lives in the material optical block (the input.laser
    # value is 1.0 when Fresnel handles reflection); derived block as fallback.
    optical = (manifest.get("material") or {}).get("optical") or {}
    derived = manifest.get("derived") or {}
    absorp = (
        optical.get("normal_absorptivity")
        or derived.get("laser_normal_absorptivity_from_ereflct")
        or pick("absorptivity", "A")
    )
    r = 5e-5
    if not all(isinstance(v, (int, float)) and v and v > 0 for v in (P, V, rho, cp, k, lat, tl, absorp)):
        return None
    D = k / (rho * cp)
    k1 = rho * (cp * (tl - T0_K) + lat) * math.sqrt(math.pi * D * r**3) / absorp
    return float(P / (k1 * math.sqrt(V)))


def apply_regime_keyword(sim, regime: str | None) -> None:
    """Stamp the melt regime as a simulation keyword (one of REGIME_KEYWORDS).

    Regime labels are mutually exclusive, so the other two are removed; any
    non-regime keywords the user set are left untouched.
    """
    label = (regime or "").lower()
    if label not in REGIME_KEYWORDS:
        return
    current = list(sim.base.extras.get("keywords", None) or [])
    kept = [k for k in current if str(k).lower() not in REGIME_KEYWORDS]
    if kept + [label] != current:
        sim.base.extras.set("keywords", [*kept, label])


def get_or_compute_case_stats(sim, *, recompute: bool = False) -> dict[str, Any]:
    cached = sim.base.extras.get(STATS_EXTRA_KEY, None)
    if not recompute and isinstance(cached, dict) and cached.get("version") == STATS_VERSION:
        return cached
    stats = compute_case_metrics(sim)
    sim.base.extras.set(STATS_EXTRA_KEY, stats)
    apply_regime_keyword(sim, stats.get("regime"))
    return stats


REGIME_COLORS = {"LoF": "#c44e52", "conduction": "#4c72b0", "keyhole": "#55a868"}
DATASET_PLOT_KINDS = ("regime", "depth_vs_width", "melt_hist", "pv_map")


def render_dataset_stats_png(rows: list[dict[str, Any]], *, kind: str = "regime") -> bytes:
    """Dataset summary plot from cached per-case stats rows (notebook style)."""
    plt = _matplotlib()
    rows = [r for r in rows if r.get("stats")]
    fig = plt.figure(figsize=(6.4, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    if kind == "regime":
        order = ["LoF", "conduction", "keyhole"]
        counts = {k: 0 for k in order}
        for r in rows:
            reg = r["stats"].get("regime")
            if reg in counts:
                counts[reg] += 1
        ax.bar(order, [counts[k] for k in order], color=[REGIME_COLORS[k] for k in order])
        ax.set_ylabel("cases")
        ax.set_title(f"Melt regime mix ({len(rows)} cases)")
    elif kind == "depth_vs_width":
        # Since stats v2: ellipse-fit cavity r_min vs depth; the boundary
        # r_min = KEYHOLE_RMIN_FACTOR * depth separates keyhole (below) from
        # conduction (above), independent of scan-path direction.
        # Regime plane: time-averaged aspect vs max cavity depth, with both
        # boundaries drawn (aspect 1 and the beam-radii depth gate).
        mx = 1.0
        gates = []
        for reg, color in REGIME_COLORS.items():
            xs = [r["stats"].get("cavity_depth_max_um", 0.0) for r in rows if r["stats"].get("regime") == reg]
            ys = [r["stats"].get("aspect_mean", 0.0) for r in rows if r["stats"].get("regime") == reg]
            if xs:
                mx = max(mx, max(xs))
                ax.scatter(xs, ys, s=16, alpha=0.75, color=color, label=f"{reg} ({len(xs)})")
        gates = [r["stats"].get("keyhole_min_depth_um") for r in rows if r["stats"].get("keyhole_min_depth_um")]
        ax.axhline(KEYHOLE_ASPECT_MEAN, color="k", linestyle="--", linewidth=1, label="mean aspect = 1")
        if gates:
            ax.axvline(min(gates), color="#8c4a12", linestyle=":", linewidth=1,
                       label=f"depth = {KEYHOLE_MIN_DEPTH_RADII:g}·beam radius")
        ax.set_xlim(0, mx * 1.05)
        ax.set_xlabel("max cavity depth (µm)")
        ax.set_ylabel("mean cavity aspect (depth / r_min)")
        ax.set_title("Keyhole criterion: mean aspect > 1 and deep enough")
        ax.legend(fontsize=8)
    elif kind == "melt_hist":
        vals = [r["stats"].get("melt_vol_um3", 0.0) / 1e9 for r in rows if r["stats"].get("melt_vol_um3") is not None]
        ax.hist(vals, bins=30, color="#4c72b0")
        ax.set_xlabel("melt pool volume (×10⁹ µm³)")
        ax.set_ylabel("cases")
        ax.set_title("True melt-pool volume")
    elif kind == "pv_map":
        for reg, color in REGIME_COLORS.items():
            xs = [r.get("V_scan") for r in rows if r["stats"].get("regime") == reg and r.get("V_scan") is not None]
            ys = [r.get("laser_power") for r in rows if r["stats"].get("regime") == reg and r.get("laser_power") is not None]
            if xs:
                ax.scatter(xs, ys, s=16, alpha=0.75, color=color, label=reg)
        ax.set_xlabel("scan speed (m/s)")
        ax.set_ylabel("laser power (W)")
        ax.set_title("P–V map by regime")
        ax.legend(fontsize=8)
    else:
        raise ValueError(f"kind must be one of {DATASET_PLOT_KINDS}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------- model curves
_STEP_LOSS_RE = re.compile(r"\[step:\s*(\d+)\]\s*loss:\s*([0-9.eE+-]+)")


def parse_stdout_loss(text: str) -> list[tuple[int, float]]:
    """The dense `[step: N] loss: X` series a training job prints to stdout."""
    out: list[tuple[int, float]] = []
    for step, value in _STEP_LOSS_RE.findall(text or ""):
        try:
            out.append((int(step), float(value)))
        except ValueError:
            continue
    return out


def parse_tfevents(path: str) -> dict[str, list[tuple[int, float]]]:
    """Scalar-ish series from a tfevents file, keyed by tag.

    physicsnemo logs these as TB *tensors*, not scalars, so `Scalars()` comes
    back empty and `Tensors()` is what carries the values.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util

    acc = EventAccumulator(path, size_guidance={"tensors": 0, "scalars": 0})
    acc.Reload()
    series: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("tensors", []):
        points = []
        for event in acc.Tensors(tag):
            try:
                points.append((int(event.step), float(tensor_util.make_ndarray(event.tensor_proto))))
            except Exception:  # noqa: BLE001 - non-scalar tensors are not curves
                continue
        if points:
            series[tag] = sorted(points)
    for tag in acc.Tags().get("scalars", []):
        points = [(int(e.step), float(e.value)) for e in acc.Scalars(tag)]
        if points:
            series.setdefault(tag, sorted(points))
    return series


def render_model_curves_png(curves: dict[str, Any], *, kind: str = "validation", t_factor: float | None = None) -> bytes:
    """Validation or loss curves for one model.

    Temperature series are drawn in Kelvin when the model's scale is known and
    in normalised units otherwise — the axis label says which, so a curve can
    never be read as Kelvin by accident.
    """
    plt = _matplotlib()
    tags = curves.get("tags") or {}
    fig = plt.figure(figsize=(6.6, 4.2), dpi=100)
    if kind == "loss":
        ax = fig.add_subplot(111)
        stdout = curves.get("loss_stdout") or []
        drew = False
        if len(stdout) >= 2:
            ax.plot([p[0] for p in stdout], [p[1] for p in stdout], lw=1.2, label="loss (job output)")
            drew = True
        for tag, points in sorted(tags.items()):
            if not tag.startswith("Train/loss") or len(points) < 1:
                continue
            ax.plot([p[0] for p in points], [p[1] for p in points], "x", ms=6,
                    label=tag.split("/", 1)[-1])
            drew = True
        if not drew:
            ax.text(0.5, 0.5, "no loss series recorded", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.set_yscale("log")
            ax.legend(fontsize=7)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training loss")
    else:
        # Two axes, not one twin axis: Kelvin and dimensionless on a shared
        # y-scale is unreadable, and a right-hand axis invites misreading.
        left = fig.add_subplot(121)
        right = fig.add_subplot(122)
        # Validator MAE/RMSE only. `Train/loss_*` lives on a different scale
        # (1e7 vs 1e-2) and would flatten these to the axis; `l2_relative_*` is
        # a ratio that must never be multiplied into Kelvin.
        val = {t: p for t, p in tags.items()
               if t.startswith("Validators/") and ("mae_" in t or "rmse_" in t)}
        t_tags = {t: p for t, p in val.items() if t.endswith("_T")}
        other = {t: p for t, p in val.items() if t.endswith(("_alpha", "_fl"))}
        scale = t_factor or 1.0
        for tag, points in sorted(t_tags.items()):
            left.plot([p[0] for p in points], [p[1] * scale for p in points], marker=".", ms=3,
                      label=tag.rsplit("/", 1)[-1])
        left.set_ylabel("K" if t_factor else "normalised")
        left.set_title("Validation error (T)")
        for tag, points in sorted(other.items()):
            right.plot([p[0] for p in points], [p[1] for p in points], marker=".", ms=3,
                       label=tag.rsplit("/", 1)[-1])
        right.set_ylabel("normalised")
        right.set_title("Validation error (α, fl)")
        for ax, series in ((left, t_tags), (right, other)):
            ax.set_xlabel("step")
            if series:
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "not recorded", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def render_eval_preview_png(arrays: dict[str, Any], *, field: str = "T", z: int = -1,
                            t_factor: float | None = None) -> bytes:
    """Ground truth, prediction and |error| for one stored preview case."""
    plt = _matplotlib()
    true = np.asarray(arrays[f"true_{field}"])
    pred = np.asarray(arrays[f"pred_{field}"])
    while true.ndim > 3:
        true, pred = true[0], pred[0]
    nz = true.shape[-1]
    z = nz // 2 if z is None or z < 0 or z >= nz else int(z)
    scale = t_factor if (field == "T" and t_factor) else 1.0
    a, b = true[..., z] * scale, pred[..., z] * scale
    err = np.abs(b - a)
    unit = "K" if (field == "T" and t_factor) else "normalised"

    fig = plt.figure(figsize=(9.0, 3.2), dpi=100)
    vmin, vmax = float(np.nanmin([a.min(), b.min()])), float(np.nanmax([a.max(), b.max()]))
    for i, (data, title, kw) in enumerate((
        (a, "ground truth", {"vmin": vmin, "vmax": vmax}),
        (b, "prediction", {"vmin": vmin, "vmax": vmax}),
        (err, "|error|", {}),
    )):
        ax = fig.add_subplot(1, 3, i + 1)
        im = ax.imshow(np.asarray(data).T, origin="lower", aspect="auto", **kw)
        ax.set_title(f"{field} {title}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(unit, fontsize=7)
    fig.suptitle(f"z-slice {z + 1}/{nz}", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
