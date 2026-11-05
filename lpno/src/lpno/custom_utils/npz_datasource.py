"""Reader for consolidated single-file NPZ cases with data-derived geometry.

The per-timestep VTK path (``openFoam_datasource.VTKOpenFOAMDataSource``) crops
and KNN-resamples scattered cell data onto a grid fixed by ``custom.geometry`` in
the config. That hardcodes one rectangular domain: a case with a different
physical extent or resolution gets silently cropped/extrapolated onto the config
box.

A consolidated NPZ (one ``.npz`` per case, e.g. from the flow3D_scripts compact
export) already stores a uniform rectangular grid: explicit ``x``/``y``/``z``
axes plus every field over time. There is nothing to resample, so this reader
takes the grid straight from the file and lets arbitrary rectangular geometries
flow through. ``derive_geometry_from_npz`` feeds those bounds/resolution back into
the config so the rest of the pipeline (model, plotters, FNO mode caps) sizes
itself to the data instead of the hardcoded block.

Kept numpy-only (no pyvista/torch) so it imports cheaply during config loading
and is unit-testable on CPU.

Expected NPZ keys: ``x(nx,) y(ny,) z(nz,) time_s(nt,)``, scalars ``power_w``,
``scan_speed_m_per_s``, ``case_name``, and fields shaped ``(nt, nx, ny, nz)``
(scalar) or ``(nt, nx, ny, nz, 3)`` (vector, e.g. ``Velocity``).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# Keys that identify a consolidated case NPZ and distinguish it from the
# datasource's own ``var_*.npz`` resample cache (which has x/y/z/P/V_scan only).
_SIGNATURE_KEYS = {"power_w", "scan_speed_m_per_s", "case_name"}

# NPZ stores the 3-vector "Velocity"; the config may still request the split
# X/Y/Z-velocity scalars. They are just its components, so synthesize them.
_VELOCITY_COMPONENTS = {"X-velocity": 0, "Y-velocity": 1, "Z-velocity": 2}


# --------------------------------------------------------------------------- #
# Moving-frame post-process (numpy-only mirror of openFoam_datasource's block).
# Duplicated deliberately so this module does not import pyvista/torch.
# ponytail: 2 small funcs mirrored, not shared, to keep the tested VTK hot path
# untouched; both copies are covered by tests.
# --------------------------------------------------------------------------- #
def _crop_shift_moving_frame(data: np.ndarray, Nsteps: int, super_res_factor: int = 1) -> np.ndarray:
    """(B,T,C,nX,nY,nZ) -> (B,Nsteps,C,nX-(Nsteps-1)*sr,nY,nZ): keep last Nsteps, shift each by its index."""
    if data.ndim != 6:
        raise ValueError("data must be 6D (B, T, C, nX, nY, nZ)")
    B, T, C, nX, nY, nZ = data.shape
    if not 1 <= Nsteps <= T:
        raise ValueError("Nsteps must be in [1, T]")
    new_nX = nX - (Nsteps - 1) * super_res_factor
    if new_nX <= 0:
        raise ValueError("Nsteps must be smaller than nX to keep a positive x-length")
    cropped_t = data[:, T - Nsteps:, ...]
    out = np.empty((B, Nsteps, C, new_nX, nY, nZ), dtype=data.dtype)
    for t in range(Nsteps):
        out[:, t] = cropped_t[:, t, :, super_res_factor * t: super_res_factor * t + new_nX, :, :]
    return out


def _moving_average_time(data: np.ndarray, window: int, axis: int = 1) -> np.ndarray:
    """Time-only sliding mean: (B,T,C,X,Y,Z) -> (B,T-window+1,C,X,Y,Z)."""
    from numpy.lib.stride_tricks import sliding_window_view

    arr = np.asarray(data)
    T = arr.shape[axis]
    if not (1 <= window <= T):
        raise ValueError(f"window must be in [1, {T}]")
    view = sliding_window_view(arr, window_shape=window, axis=axis)
    return view.mean(axis=-1, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Detection and geometry derivation
# --------------------------------------------------------------------------- #
def _open_case_safely(path):
    """`pack_slice.open_case`, degrading to the path itself if unavailable."""
    try:
        from .pack_slice import open_case
    except Exception:  # noqa: BLE001 - a real path needs no translation
        return path
    return open_case(path)


def _npz_keys(path: Path) -> set[str]:
    try:
        with np.load(_open_case_safely(path)) as data:
            return set(data.files)
    except Exception:
        return set()


def _has_signature(path: Path) -> bool:
    return _SIGNATURE_KEYS.issubset(_npz_keys(path))


def is_consolidated_npz(path: str | Path) -> bool:
    """Return whether ``path`` is a consolidated case NPZ (file) or a dir of them."""
    p = Path(path)
    if p.is_file():
        return p.suffix == ".npz" and _has_signature(p)
    if p.is_dir():
        if any(
            _has_signature(f)
            for f in sorted(p.glob("*.npz"))
            if not f.name.startswith("var_")
        ):
            return True
        # Zero-copy staging (default since 2026-08-14) materializes no real .npz
        # files -- the cases exist only as pack_index.json byte ranges. Detection
        # must see them exactly like `_case_files` does, or the format dispatcher
        # falls through to the arraydir store and rejects the npz_file member
        # manifests (run 46347: "Unsupported runtime format ... aiida_npz_file_v1").
        return any(_has_signature(s) for s in _pack_index_entries(p))
    return False


def _pack_index_entries(directory: Path) -> List[Path]:
    """Pack-slice cases for a staged dir, or [] if the feature is unavailable.

    Zero-copy staging is an *accelerator*: a dataset staged as real files must
    keep loading even if this module is missing, broken, or from a mismatched
    checkout. An earlier version imported it at the top of `_case_files`, so a
    tree carrying this file's changes without `pack_slice.py` broke case
    discovery outright — an optional feature must never be able to do that.
    """
    try:
        from .pack_slice import load_pack_index
    except Exception:  # noqa: BLE001 - absent or broken: fall back to real files
        return []
    return load_pack_index(directory)


def _case_files(path: str | Path) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = [
            f for f in sorted(p.glob("*.npz"))
            if not f.name.startswith("var_") and _has_signature(f)
        ]
        # Cases staged as pack byte ranges rather than copies (see pack_slice).
        # A real file wins a name collision: it is what a repair materialized.
        real_names = {f.name for f in files}
        sliced = [s for s in _pack_index_entries(p) if s.name not in real_names and _has_signature(s)]
        merged = sorted(files + sliced, key=lambda f: f.name)
        if merged:
            return merged
    raise ValueError(f"[NPZ] No consolidated case NPZ found at '{path}'.")


def derive_geometry_from_npz(path: str | Path) -> Dict[str, Any]:
    """Read x/y/z from the first case and return a config ``geometry`` dict."""
    first = _case_files(path)[0]
    with np.load(_open_case_safely(first)) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)
        z = np.asarray(data["z"], dtype=np.float64)
    return {
        "x_range": (float(x.min()), float(x.max())),
        "y_range": (float(y.min()), float(y.max())),
        "z_range": (float(z.min()), float(z.max())),
        "resolution": (int(x.size), int(y.size), int(z.size)),
    }


def derive_geometry_from_data(path: str | Path, software: str = "flow3D") -> Dict[str, Any]:
    """Single entry point for config-time geometry derivation.

    Consolidated NPZ carries an explicit grid, and a staged arraydir runtime records
    the grid it serves; both are auto-derived. VTK cases must set ``custom.geometry``
    numerically (they define a curated crop, e.g. dropping the gas layer, that is not
    simply the data extent).
    """
    if is_consolidated_npz(path):
        return derive_geometry_from_npz(path)
    # A staged arraydir runtime carries the grid it serves in its own manifest; the
    # arraydir loader never crops or resamples, so that grid is the only correct one.
    from lpno.models.pipeline.lazy_runtime import (  # local: avoids a cycle
        is_arraydir_runtime,
        runtime_grid_params,
    )

    if is_arraydir_runtime(path):
        grid = runtime_grid_params(path)
        if grid is not None:
            return grid
        raise ValueError(
            f"Arraydir runtime '{path}' records no usable dataset_metadata.grid_params; "
            "it was staged by an aiida-pino too old to write one. Re-stage it."
        )
    raise ValueError(
        "Automatic geometry derivation is only supported for consolidated NPZ "
        f"datasets and staged arraydir runtimes; '{path}' is neither. Set "
        "custom.geometry ranges/resolution explicitly (VTK data is cropped/resampled "
        "onto that box)."
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _field_to_bt_cxyz(name: str, data, nt: int, nx: int, ny: int, nz: int) -> np.ndarray:
    """Reshape a raw ``(nt,nx,ny,nz[,3])`` field to ``(1,nt,C,nx,ny,nz)``."""
    if name in data.files:
        arr = np.asarray(data[name], dtype=np.float32)
    elif name in _VELOCITY_COMPONENTS and "Velocity" in data.files:
        arr = np.asarray(data["Velocity"], dtype=np.float32)[..., _VELOCITY_COMPONENTS[name]]
    else:
        available = sorted(k for k in data.files if k not in _SIGNATURE_KEYS)
        raise KeyError(
            f"[NPZ] Requested field '{name}' not found. Available fields: {available}"
        )
    if arr.shape[:4] != (nt, nx, ny, nz):
        raise ValueError(
            f"[NPZ] Field '{name}' has shape {arr.shape}; expected leading (nt,nx,ny,nz)="
            f"{(nt, nx, ny, nz)}."
        )
    if arr.ndim == 4:  # scalar
        return arr.reshape(1, nt, 1, nx, ny, nz)
    if arr.ndim == 5 and arr.shape[4] == 3:  # vector (nt,nx,ny,nz,3) -> (1,nt,3,nx,ny,nz)
        return np.moveaxis(arr, 4, 1)[None, ...]
    raise ValueError(f"[NPZ] Field '{name}' has unsupported shape {arr.shape}.")


# --------------------------------------------------------------------------- #
# Laser-path conditioning channels (arbitrary weld path -> model inputs)
#
# The straight-line model told the network where the beam was via the moving
# reference frame. For an arbitrary path in the lab frame we instead pass the
# trajectory explicitly, derived from the sidecar case_manifest.json:
#   beam_x, beam_y : instantaneous beam centre at each frame (broadcast scalars)
#   beam_dist      : instantaneous Gaussian beam footprint (a spatial field)
#   path_dose      : accumulated beam exposure up to the frame time -- the
#                    thermal-memory channel, since the field at time t depends on
#                    the whole path travelled so far, not just the current spot.
# All are computed from the geometric path + timing only (no solver output), so
# they are legitimate model inputs rather than leaked targets.
# --------------------------------------------------------------------------- #
PATH_CHANNEL_NAMES = (
    "beam_x",
    "beam_y",
    "beam_dist",
    "path_dose",
    "path_dose_recent",
    # Local path coordinates: signed along-track and cross-track offsets in
    # beam-radius units, plus instantaneous path speed as a spatial field.
    "beam_s",
    "beam_n",
    "beam_speed",
    # Curvilinear (Frenet) frame: the same coordinates measured along the arc
    # instead of the tangent line, so the wake stays "behind" through a bend.
    # beam_s_arc is the signed arc distance from a point's nearest spot on the
    # trajectory to the current beam; beam_n_arc the signed offset from the
    # track itself. On a straight path both equal beam_s / beam_n exactly.
    "beam_s_arc",
    "beam_n_arc",
    # Effective sweep speed v(1 - n*kappa_signed): the kinematic curvature
    # correction -- the inner side of a bend sees a slower sweep (more energy
    # per unit length). Signed curvature from the trajectory itself; reduces to
    # the broadcast speed on a straight track.
    "beam_veff",
    # Exact linear-conduction response G*f (spectral stepping, image source at
    # the free surface): the whole dose family's limit, with real z-structure.
    # Per-case compute is cached on disk (PINO_CONDUCTION_CACHE, default
    # ~/.pino/conduction_cache) keyed by case geometry, so the cost is paid
    # once per case, not per epoch. Gas planes (z > surface) are zero.
    "conduction_field",
    # Depth-aware beam fields. ``*_sigma`` uses Gaussian beam propagation
    # sigma(z); ``*_sigma_exp`` additionally applies exp(-|z-z_surface|/delta).
    "beam_dist_sigma",
    "path_dose_sigma",
    "path_dose_recent_sigma",
    "beam_dist_sigma_exp",
    "path_dose_sigma_exp",
    "path_dose_recent_sigma_exp",
    # Case-constant broadcasts for the PV-LHS beam/preheat sweep (dataset 38427
    # family). Raw SI values, like V_scan and P: FiLM moves broadcast-constant
    # scalar inputs into conditioning, so no ref-scale is baked in here.
    "preheat_T0",           # input.simulation.initial_temperature_k, K (293.15 if absent)
    "beam_shape_uniform",   # 1.0 when input.laser.heat_distribution == "uniform"
    "focus_factor",         # input._design.focus_factor (1.0 if absent)
    "beam_radius",          # input.laser.beam_radius_m, metres (5e-5 if absent)
    # Analytic boundary soak. The PV-LHS sweep's configured preheat never reached the
    # metal -- the legacy_halfspace renderer set tvoid/tbct but not tempi -- so the
    # block starts at ambient and heats INWARD from its faces at T0 for the whole run.
    # A broadcast constant states the asymptote, not the state at time t. Conduction is
    # linear, so the transient has a closed form (separation of variables, odd-harmonic
    # series per axis); this channel is that solution, exact and simulation-free.
    "soak_field",
    # Normalized enthalpy against the LOCAL, time-varying soak temperature rather than a
    # global ambient. The stored `H_norm` (P/sqrt(V), split-normalised) is preheat-blind;
    # the design's own h_norm_ti64 is not, because melting enthalpy is measured FROM T0:
    #     H* = A P / (rho (cp (T_l - T0) + L_f) sqrt(pi D v r^3))
    # Substituting T0 -> soak_field(x, t) turns preheat from a background offset into what
    # it physically is: a modulation of the LOCAL melting threshold. Soaked material needs
    # less enthalpy to melt, so H* rises where the block has warmed.
    "h_norm_local",
)

# --------------------------------------------------------------------------- #
# Multi-timescale thermal memory.
#
# `path_dose` is a pure accumulator (tau -> infinity, so it carries only the total)
# and `path_dose_recent` is a leaky integrator at ONE tau (1 ms). Together they are a
# two-number summary of the whole exposure history. That is nearly sufficient for a
# constant-speed path, where the history is parameterised by (P, V) and geometry, but
# it ALIASES a variable-speed path: "fast then slow" and "slow then fast" deposit the
# same total and nearly the same leaky integral while leaving different thermal states.
#
# Measured on ti64_varspeed150 (scripts/probe_varspeed_memory.py): mean speed CV 0.205,
# median segment dwell 190.8 us, and 100% of segments shorter than tau = 1 ms -- about
# 5.2 segments inside one tau window. Distinguishing histories that vary on timescale d
# over a window tau needs roughly tau/d exponential basis functions: ~5 demanded, 2
# supplied.
#
# The bank is the truncated basis. The linear heat equation's Green's function is well
# approximated by a sum of exponentials in time, so leaky integrators at several tau
# span the diffusive memory kernel and leave the network only the nonlinear corrections
# (latent heat, convection, temperature-dependent properties). Each tau corresponds to
# a diffusion length sqrt(alpha*tau): for Ti (alpha ~ 3e-6 m^2/s) 30 us ~ 9 um,
# 100 us ~ 17 um, 300 us ~ 30 um, 1 ms ~ 55 um, 3 ms ~ 95 um.
#
# tau = 1 ms is deliberately absent: `path_dose_recent` already provides it, so a
# config asking for the bank plus `path_dose_recent` gets five distinct timescales
# with no duplicate channel.
_DOSE_TAU_BANK_S = {
    "path_dose_tau30us": 30e-6,
    "path_dose_tau100us": 100e-6,
    "path_dose_tau300us": 300e-6,
    "path_dose_tau3ms": 3e-3,
}
# Same prefix convention as the rest of the family: bare, `_sigma`, `_sigma_exp`.
_TAU_BANK_NAMES = tuple(
    name if prefix == "" else f"{name}_{prefix}"
    for prefix in ("", "sigma", "sigma_exp")
    for name in _DOSE_TAU_BANK_S
)
PATH_CHANNEL_NAMES = PATH_CHANNEL_NAMES + _TAU_BANK_NAMES


def _tau_bank_for_prefix(prefix: str) -> dict[str, float]:
    """{channel name: tau} for one footprint family."""
    suffix = "" if prefix == "" else f"_{prefix}"
    return {f"{base}{suffix}": tau for base, tau in _DOSE_TAU_BANK_S.items()}


def _leaky_dose(footprint, dt, on, tau: float):
    """Exponentially weighted exposure history: (1/tau) * integral f(s) e^{-(t-s)/tau} ds.

    Discretised as a leaky accumulator, which is what makes a bank of these cheap: one
    O(1)-memory recursion per tau, computed once at datasource level, rather than
    holding k previous frames. Dividing by tau normalises the kernel so channels at
    different tau carry comparable magnitude instead of scaling with the window.
    """
    import numpy as _np

    out = _np.empty_like(footprint, dtype=_np.float32)
    acc = _np.zeros(footprint.shape[1:], dtype=_np.float32)
    for i in range(footprint.shape[0]):
        decay = _np.exp(-float(dt[i]) / tau) if i > 0 else 0.0
        acc = acc * _np.float32(decay) + footprint[i].astype(_np.float32) * _np.float32(dt[i] * on[i])
        out[i] = acc / _np.float32(tau)
    return out

# --------------------------------------------------------------------------- #
# Material conditioning channels: per-case scalar material properties, read from
# the case manifest and broadcast to constant full-field channels so a single
# operator can be conditioned on *which material* it is welding. Each maps to a
# leaf in manifest["material"]; they are constant in space and time (a material
# property is defined everywhere), unlike the path channels which vary per frame.
# Values are physical (SI); the pipeline non-dimensionalises them via ref_scales.
# --------------------------------------------------------------------------- #
# channel -> path within the manifest (tuple of nested keys)
_MATERIAL_CHANNEL_SOURCES = {
    "mat_rho": ("material", "properties", "density_kg_m3"),
    "mat_cp": ("material", "properties", "specific_heat_j_kg_k"),
    "mat_k": ("material", "properties", "thermal_conductivity_w_m_k"),
    "mat_latent": ("material", "properties", "latent_heat_j_kg"),
    "mat_Tl": ("material", "properties", "liquidus_temperature_k"),
    # Solidus completes the melt window. `fl` is the melt region, so scoring it
    # needs BOTH ends per case, not just the liquidus.
    "mat_Ts": ("material", "properties", "solidus_temperature_k"),
    # Vaporisation completes the trio: it is the alpha-mask FILL temperature, and a
    # fixed Ti64 3520 K fill on a 2068 K material teaches "hot = gas" even harder.
    "mat_Tv": ("material", "properties", "vaporization_temperature_k"),
    "mat_A": ("material", "optical", "normal_absorptivity"),
}

# Overlay channels: per-case scalars that the manifest CANNOT supply, keyed by
# `case_name` in a JSON side table built by
# `flow3D_scripts/python/material_conditioning.py`.
#
# Why an overlay and not another manifest path. The manifest carries the 15
# material scalars, but for the property CURVES it stores only row COUNTS
# (`material.table_rows`), so k(T) at the liquidus is unreachable from a stored
# simulation. Two of the others are operating-point x material composites that no
# manifest leaf holds.
#
# What these buy, measured on a gradient-boosting proxy for log fusion depth over
# the 298 melting ck300 cases (CV R^2, on top of the six manifest channels):
#   mat_k_liq                     0.356 -> 0.669   (all six OTHER hidden scalars: 0.385)
#   + mat_H_king, mat_X_keyhole   -> 0.812 with tv_ratio
#   + mat_theta_v, mat_theta_m    -> 0.828
#
# `mat_H_king` and `mat_X_keyhole` are NOT redundant despite correlating at
# r = +0.904. Pooled, adding H on top of X buys +0.003; split by measured regime,
# H adds +0.067 on conduction and X adds +0.203 on keyhole. They normalise on
# melting and on vaporisation respectively, which is exactly the regime split.
#
# `mat_H_king` is ALSO not the `H_norm` channel. `utils.h_norm_raw` computes
# P/sqrt(|V| r0^3) and lets a split-max denominator absorb the material constants,
# which its own docstring notes is exact only while a split shares ONE material.
# On a 300-material split at fixed r0 that channel is exactly P/sqrt(V) rescaled
# (measured r = +1.000) and carries no material information at all.
_MATERIAL_OVERLAY_SOURCES = {
    "mat_k_liq": "k_liquidus_w_m_k",
    "mat_H_king": "h_king",
    "mat_X_keyhole": "x_keyhole",
    "mat_tv_ratio": "tv_ratio",
    "mat_theta_v": "theta_v",
    "mat_theta_m": "theta_m",
}

#: Environment variable naming the overlay JSON. An env var rather than a config
#: field because the value must survive into a remote AiiDA job, where the config
#: is rendered on the submitting machine but this code runs on the compute box.
MATERIAL_OVERLAY_ENV = "PINO_MATERIAL_OVERLAY"

# The manifest-backed six. This name keeps its original meaning -- every channel
# in it is readable from a stored simulation with no external file -- because
# callers use it both as a membership set and as "give me all of them", and the
# overlay channels cannot satisfy the second reading.
MATERIAL_CHANNEL_NAMES = (
    "mat_rho", "mat_cp", "mat_k", "mat_latent", "mat_Tl", "mat_A",
)

#: The melt-WINDOW channels. Deliberately NOT in MATERIAL_CHANNEL_NAMES: that tuple
#: doubles as "give me all of them", and older manifests carry no solidus or
#: vaporisation key, so widening it made `_dig` raise on them
#: (tests/test_arraydir_material_channels.py). Requestable by name; never implied.
MATERIAL_WINDOW_CHANNEL_NAMES = ("mat_Ts", "mat_Tv")

#: The overlay-backed six. Requesting one of these without the overlay is an
#: error, so they are deliberately NOT part of ``MATERIAL_CHANNEL_NAMES``.
MATERIAL_OVERLAY_CHANNEL_NAMES = tuple(_MATERIAL_OVERLAY_SOURCES)

#: Label channels: a per-case constant that NAMES the alloy instead of describing it.
#: `mat_alloy_flag` is 1 for Inconel 718 and 0 for any other material, read from the
#: manifest's material name, so it needs no side table. It exists for two-alloy
#: training, where a model given only process parameters cannot tell the alloys apart.
#: With two alloys the six property channels would carry the same information as six
#: perfectly correlated constants; one binary label says it once.
#: Kept out of MATERIAL_CHANNEL_NAMES so that tuple stays "the manifest-backed six".
MATERIAL_LABEL_CHANNEL_NAMES = ("mat_alloy_flag",)
_ALLOY_FLAG_PATTERN = re.compile(r"(inconel|\bin)[ _-]?718", re.IGNORECASE)


def _alloy_flag(manifest: Dict[str, Any]) -> float:
    """1.0 when the case's material is Inconel 718, else 0.0 (see MATERIAL_LABEL_CHANNEL_NAMES).

    Raises when the manifest names no material at all: reading 0 there would label an
    unnamed Inconel case as the other alloy, silently, as `_overlay_value` explains.
    """
    material = manifest.get("material") or manifest.get("input", {}).get("material") or {}
    names = " ".join(str(material.get(k) or "") for k in ("name", "flow3d_label", "file")).strip()
    if not names:
        raise KeyError("mat_alloy_flag needs the manifest's material name; this case has none")
    return 1.0 if _ALLOY_FLAG_PATTERN.search(names) else 0.0


#: Every name that IS a material channel. Use this for membership filtering
#: against a config's declared input keys; use it never as "build all of these".
ALL_MATERIAL_CHANNEL_NAMES = (
    MATERIAL_CHANNEL_NAMES + MATERIAL_WINDOW_CHANNEL_NAMES + MATERIAL_OVERLAY_CHANNEL_NAMES
    + MATERIAL_LABEL_CHANNEL_NAMES
)


def served_material_channels(run_cfg, input_keys: Iterable[str]) -> List[str]:
    """The material channels a loader must serve for this run.

    The declared inputs, plus the manifest six when `h_norm_include_material` is set.
    Those six are AUXILIARY then: King's material factor reads them inside H_norm, and
    input assembly indexes by the declared keys, so they never reach the network as
    channels of their own (the same arrangement as `mat_Tv` for the mask fill).
    """
    out = [k for k in input_keys if k in ALL_MATERIAL_CHANNEL_NAMES]
    if bool(getattr(run_cfg, "h_norm_include_material", False)):
        out += [k for k in MATERIAL_CHANNEL_NAMES if k not in out]
    return out


# FLOW-3D's default initial fluid temperature: what the metal actually starts at in
# every legacy_halfspace case, whatever the config asked for.
_AMBIENT_T_K = 273.15


def _dig_or(manifest: Dict[str, Any], path: tuple, default: float) -> float:
    """`_dig` with a default: absent keys mean the sweep did not vary this axis."""
    node: Any = manifest
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return float(default)
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return float(default)


def _dig(manifest: Dict[str, Any], path: tuple) -> float:
    node: Any = manifest
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(".".join(path))
        node = node[key]
    return float(node)


@lru_cache(maxsize=4)
def _load_material_overlay(path: str) -> Dict[str, Dict[str, float]]:
    """Load and cache an overlay side table keyed by ``case_name``."""
    import json

    with open(path) as fh:
        table = json.load(fh)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{MATERIAL_OVERLAY_ENV}={path} is not a non-empty case table")
    return table


def _overlay_value(manifest: Dict[str, Any], channel: str) -> float:
    """Resolve one overlay channel for this case, or raise.

    Deliberately raises instead of defaulting. A missing overlay entry would
    otherwise broadcast the SAME constant to every case, which trains happily and
    silently answers a different question -- the conditioning channel becomes a
    bias term. That failure is invisible in the loss curve, so it has to be loud
    here.
    """
    path = os.environ.get(MATERIAL_OVERLAY_ENV)
    if not path:
        raise KeyError(
            f"channel {channel!r} needs the per-case overlay; set "
            f"{MATERIAL_OVERLAY_ENV} to the JSON written by "
            f"flow3D_scripts/python/material_conditioning.py"
        )
    case = manifest.get("case_name")
    if not case:
        raise KeyError(f"channel {channel!r} needs manifest['case_name']; manifest has none")
    table = _load_material_overlay(path)
    if case not in table:
        raise KeyError(
            f"case {case!r} is absent from the overlay {path} "
            f"({len(table)} cases); the overlay does not cover this dataset"
        )
    field = _MATERIAL_OVERLAY_SOURCES[channel]
    if field not in table[case]:
        raise KeyError(f"overlay entry for {case!r} has no {field!r} (channel {channel!r})")
    return float(table[case][field])


def _build_material_channels(
    requested: Iterable[str], manifest: Dict[str, Any], shape: tuple
) -> Dict[str, np.ndarray]:
    """Build requested material channels as constant ``shape`` float32 fields."""
    requested = [k for k in requested if k in ALL_MATERIAL_CHANNEL_NAMES]
    out: Dict[str, np.ndarray] = {}
    for key in requested:
        if key in _MATERIAL_OVERLAY_SOURCES:
            value = _overlay_value(manifest, key)
        elif key == "mat_alloy_flag":
            value = _alloy_flag(manifest)
        else:
            value = _dig(manifest, _MATERIAL_CHANNEL_SOURCES[key])
        out[key] = np.full(shape, value, dtype=np.float32)
    return out


def _load_manifest(npz_path: Path) -> Optional[Dict[str, Any]]:
    """Return the case manifest sitting next to an NPZ, or None if absent."""
    sidecar = npz_path.with_name(f"{npz_path.stem}.manifest.json")
    if not sidecar.is_file():
        legacy = npz_path.with_name("case_manifest.json")
        sidecar = legacy if legacy.is_file() else sidecar
    if not sidecar.is_file():
        return None
    import json

    with open(sidecar) as fh:
        return json.load(fh)


def _beam_path_over_time(manifest: Dict[str, Any], t_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate beam centre, direction and local speed at each frame time.

    Constant-speed paths are sampled by arc length. Variable-speed paths carry
    explicit per-knot arrival times in ``derived.laser_path_times_s``; those are
    used directly so the conditioning channels follow dwells and speed changes.

    Returns ``(centres, directions, speeds)`` with shapes ``(nt, 2)``, ``(nt, 2)``,
    and ``(nt,)``. Direction is zero for dwell/off frames.
    """
    derived = manifest.get("derived", {})
    xy = np.asarray(derived["laser_path_xy_m"], dtype=np.float64)  # (K, 2)
    tt = np.asarray(t_seconds, dtype=np.float64)
    # Present-but-NULL means "never turns off", same as absent: the default
    # argument does not fire on an explicit None, so float(None) raised and
    # killed both arms on ti64_straight_path_calibrated_with_keyholes at step 0
    # (24 of 123 members record null; they carry turn_off_at_end=False and
    # scan_time_s == the run's effective_end_time_s, i.e. no cooling tail).
    off_raw = derived.get("laser_off_time_s")
    # `turn_off_at_end: false` means the solver holds the last laser row until
    # finish_time: the beam stays ON and, for a scan, KEEPS MOVING past the declared
    # path end. Post-2026-08-15 manifests say so outright (`laser_on_past_declared_time`,
    # `laser_overrun_distance_m`); earlier ones wrote `laser_off_time_s = scan_time`
    # regardless -- a turn-off event the deck never contained
    # (2026-08-15) -- so for those the off time is itself the lie and
    # `laser_turn_off_at_end` is the field to trust. Either way the position must
    # keep going, not freeze at the last knot: freezing it put the anchorbox steady
    # target 5-12 cells AHEAD of its own beam (eval 83535 of model 82910), a state no
    # keyhole can be in, and left speed/direction non-zero on frames where the
    # centre stood still. No flag at all (legacy fixtures) keeps the old clamp.
    keeps_going = bool(derived.get("laser_on_past_declared_time")) or (
        derived.get("laser_turn_off_at_end") is False
    )
    off = np.inf if (off_raw is None or keeps_going) else float(off_raw)
    active_t = np.minimum(tt, off)

    times_raw = derived.get("laser_path_times_s")
    if times_raw is not None:
        times = np.asarray(times_raw, dtype=np.float64)
        if times.shape != (xy.shape[0],):
            raise ValueError(
                "derived.laser_path_times_s must have one arrival time per laser_path_xy_m point"
            )
        cx = np.interp(active_t, times, xy[:, 0])
        cy = np.interp(active_t, times, xy[:, 1])
        leg_idx = np.searchsorted(times[1:], active_t, side="right")
        leg_idx = np.clip(leg_idx, 0, max(0, xy.shape[0] - 2))
        dxy = np.diff(xy, axis=0)
        dt_leg = np.diff(times)
        speeds_leg = np.divide(
            np.linalg.norm(dxy, axis=1),
            dt_leg,
            out=np.zeros_like(dt_leg, dtype=np.float64),
            where=dt_leg > 0,
        )
        directions_leg = np.divide(
            dxy,
            np.linalg.norm(dxy, axis=1, keepdims=True),
            out=np.zeros_like(dxy, dtype=np.float64),
            where=np.linalg.norm(dxy, axis=1, keepdims=True) > 0,
        )
        if keeps_going and xy.shape[0] >= 2:
            # Past the last knot the solver holds the last velocity row: continue
            # along the last leg at its speed (a terminal dwell has speed 0 and stays).
            over_t = np.maximum(active_t - times[-1], 0.0)
            cx = cx + directions_leg[-1, 0] * speeds_leg[-1] * over_t
            cy = cy + directions_leg[-1, 1] * speeds_leg[-1] * over_t
        speeds = speeds_leg[leg_idx]
        directions = directions_leg[leg_idx]
        off_mask = tt > off
        speeds = np.where(off_mask, 0.0, speeds)
        directions = np.where(off_mask.reshape(-1, 1), 0.0, directions)
        return np.stack([cx, cy], axis=1), directions, speeds

    speed = float(manifest["input"]["laser"]["scan_speed_m_per_s"])
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])  # (K,)
    total = float(arc[-1])
    dist = speed * active_t
    dxy = np.diff(xy, axis=0)
    directions_leg = np.divide(
        dxy,
        np.linalg.norm(dxy, axis=1, keepdims=True),
        out=np.zeros_like(dxy, dtype=np.float64),
        where=np.linalg.norm(dxy, axis=1, keepdims=True) > 0,
    )
    # Arc distance beyond the declared path. Zero unless the laser keeps going, in
    # which case the centre continues along the last leg at scan speed -- exactly
    # `laser_overrun_distance_m` at finish_time, since |v| == speed on every leg.
    over = np.maximum(dist - total, 0.0) if (keeps_going and xy.shape[0] >= 2) else 0.0
    dist = np.clip(dist, 0.0, total)
    cx = np.interp(dist, arc, xy[:, 0]) + directions_leg[-1, 0] * over
    cy = np.interp(dist, arc, xy[:, 1]) + directions_leg[-1, 1] * over
    leg_idx = np.searchsorted(arc[1:], dist, side="right")
    leg_idx = np.clip(leg_idx, 0, max(0, xy.shape[0] - 2))
    directions = directions_leg[leg_idx]
    off_mask = tt > off
    speeds = np.where(off_mask, 0.0, speed)
    directions = np.where(off_mask.reshape(-1, 1), 0.0, directions)
    return np.stack([cx, cy], axis=1), directions, speeds


class _MultibeamSpec:
    """One beam of a multi-beam case, dressed as a single-beam manifest."""

    __slots__ = ("manifest", "power_fraction", "turn_off_time_s")

    def __init__(self, manifest, power_fraction, turn_off_time_s):
        self.manifest = manifest
        self.power_fraction = float(power_fraction)
        self.turn_off_time_s = float(turn_off_time_s)


_MULTIBEAM_ANNOUNCED = set()


def _multibeam_specs(manifest: Dict[str, Any]):
    """Per-beam specs for a multi-beam case, or None for single-beam.

    Multi-beam manifests (mbc40) carry `derived.beams[i]` with per-beam
    `path_xy_m`/`path_times_s` merged from the prepin velocity tables at
    runtime-export time (the ingest thinned the original tables, and the
    top-level `laser_path_xy_m` is a 2-point degenerate stub for these cases —
    building channels from it silently produces a wrong single-beam framing).
    Each spec's manifest reuses the existing single-beam interpolation
    machinery verbatim; weights are power fractions so one beam is identity.
    """
    beams = (manifest.get("derived") or {}).get("beams") or []
    with_paths = [b for b in beams if b.get("path_xy_m") and b.get("path_times_s")]
    if len(beams) >= 2 and len(with_paths) < 2:
        # Falling through to the single-beam branch here is never right: the
        # top-level path is a degenerate stub for these cases, so the framing
        # would be silently wrong (or die on KeyError 'scan_speed_m_per_s',
        # eval calc 46611). Seen live when a runtime cache was staged minutes
        # before the aiida-pino beam-path merge fix (4649a5a) and reused as-is.
        raise RuntimeError(
            f"multibeam manifest has {len(beams)} beams but per-beam paths for only "
            f"{len(with_paths)}; refusing the single-beam fallback. If this dataset "
            "was staged before aiida-pino 4649a5a, delete its "
            "~/.cache/aiida-pino/runtime/<uuid> entry (with no job queued or running "
            "against it) and re-stage."
        )
    if len(with_paths) < 2:
        return None
    total = sum(float(b.get("power_w") or 0.0) for b in with_paths)
    if total <= 0:
        raise RuntimeError(
            f"multibeam manifest has {len(with_paths)} beams with paths but zero total "
            "power_w; cannot weight beams, and the single-beam fallback would frame the "
            "case wrongly."
        )
    specs = []
    for b in with_paths:
        off = float(b.get("turn_off_time_s") or np.inf)
        sub = {
            "derived": {
                "laser_path_xy_m": b["path_xy_m"],
                "laser_path_times_s": b["path_times_s"],
                "laser_off_time_s": off,
            },
            "input": {"laser": {"scan_speed_m_per_s": float(b.get("scan_speed_m_per_s") or 0.0)}},
        }
        specs.append(_MultibeamSpec(sub, float(b.get("power_w") or 0.0) / total, off))
    return specs


def _beam_weights_at(specs, t_seconds: np.ndarray) -> np.ndarray:
    """(nt, nbeams) weights: power fraction while a beam is on, 0 after."""
    tt = np.asarray(t_seconds, dtype=np.float64)
    return np.stack(
        [np.where(tt <= s.turn_off_time_s, s.power_fraction, 0.0) for s in specs], axis=1
    )


def _beam_centres_over_time(manifest: Dict[str, Any], t_seconds: np.ndarray) -> np.ndarray:
    """Interpolate the beam (x,y) centre at each frame time.

    Multi-beam aware: the lazy store calls this directly (for the moving-frame
    reference), bypassing `_build_path_channels`, so it must apply the same
    power-weighted-centroid combination. Without this a multi-beam manifest fell
    through to the single-path branch and raised
    `KeyError: 'scan_speed_m_per_s'` -- mbc40 manifests carry per-beam speeds
    only, with no top-level `input.laser.scan_speed_m_per_s`.
    """
    specs = _multibeam_specs(manifest)
    if specs is not None:
        weights = _beam_weights_at(specs, t_seconds)
        wn = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        per_beam = np.stack(
            [_beam_path_over_time(s.manifest, t_seconds)[0] for s in specs], axis=2
        )                                             # (nt, 2, nbeams)
        return np.sum(per_beam * wn[:, None, :], axis=2)
    centres, _directions, _speeds = _beam_path_over_time(manifest, t_seconds)
    return centres


_P_REF_W = 100.0  # reference laser power (W) for power-weighted path_dose scaling
# ponytail: thermal recency time-constant for path_dose_recent (leaky-integrator
# dose). ~1 ms ~ the melt-pool diffusion time L^2/alpha for Ti (L~0.1mm, alpha~3e-6);
# tune if the recency channel proves too short/long-memoried. Env override for sweeps.
_DOSE_RECENT_TAU_S = float(os.environ.get("PINO_DOSE_RECENT_TAU_S", "1e-3"))
_ABS_DEPTH_M = float(os.environ.get("PINO_BEAM_ABS_DEPTH_M", "5e-5"))
_LASER_WAVELENGTH_M = float(os.environ.get("PINO_LASER_WAVELENGTH_M", "1.064e-6"))


def _rayleigh_range_m(manifest: Dict[str, Any], sigma0: float) -> float:
    laser = manifest.get("input", {}).get("laser", {})
    derived = manifest.get("derived", {})
    override = os.environ.get("PINO_BEAM_RAYLEIGH_M")
    if override is not None:
        return float(override)
    wavelength = float(derived.get("laser_wavelength_m") or laser.get("wavelength_m") or _LASER_WAVELENGTH_M)
    return float(np.pi * sigma0 * sigma0 / wavelength)


def _focus_z_m(manifest: Dict[str, Any]) -> float:
    derived = manifest.get("derived", {})
    laser = manifest.get("input", {}).get("laser", {})
    override = os.environ.get("PINO_BEAM_FOCUS_Z_M")
    if override is not None:
        return float(override)
    focus = laser.get("focus_z_m")
    if focus is not None:
        return float(focus)
    return float(derived.get("surface_z_m", 0.0))


def _abs_depth_m(manifest: Dict[str, Any]) -> float:
    laser = manifest.get("input", {}).get("laser", {})
    override = os.environ.get("PINO_BEAM_ABS_DEPTH_M")
    if override is not None:
        return float(override)
    depth = laser.get("absorption_depth_m")
    return float(depth if depth is not None else _ABS_DEPTH_M)


def _broadcast_time_scalar(values: np.ndarray, shape: tuple) -> np.ndarray:
    nt = values.size
    return np.broadcast_to(values.astype(np.float32).reshape(1, nt, 1, 1, 1, 1), shape).astype(np.float32)


def _power_scale(manifest, enabled: bool) -> float:
    """``power_w / _P_REF_W`` when ``enabled``, else 1.0.

    ONE expression, shared by every channel family that opts into power weighting.
    It was inline in `_emit_beam_family` and nowhere else, which is the shape the
    2026-08-14 dose bug had: a rule stated in one builder and not carried to the
    next. `conduction_field` is emitted outside that helper and so was power-blind
    for the same structural reason.
    """
    if not enabled:
        return 1.0
    return float(
        manifest.get("input", {}).get("laser", {}).get("power_w", _P_REF_W)
    ) / _P_REF_W


def _emit_beam_family(
    out: Dict[str, np.ndarray],
    requested: set[str],
    *,
    prefix: str,
    footprint: np.ndarray,
    shape: tuple,
    on: np.ndarray,
    dt: np.ndarray,
    power_weighted: bool,
    manifest: Dict[str, Any],
) -> None:
    """Emit beam, cumulative dose and recent dose for one footprint family."""
    # (leaky integrator shared by `path_dose_recent` and the multi-tau bank)
    beam_key = "beam_dist" if prefix == "" else f"beam_dist_{prefix}"
    dose_key = "path_dose" if prefix == "" else f"path_dose_{prefix}"
    recent_key = "path_dose_recent" if prefix == "" else f"path_dose_recent_{prefix}"

    if beam_key in requested:
        fp = footprint * on.reshape((-1,) + (1,) * (footprint.ndim - 1))
        if footprint.ndim == 3:
            fp = fp[:, :, :, None]
        out[beam_key] = np.broadcast_to(fp.astype(np.float32)[None, :, None], shape).astype(np.float32)

    # One factor for EVERY dose channel in this family. A dose is deposited energy,
    # so power belongs in all of them: until 2026-08-14 only `path_dose` carried it
    # and `path_dose_recent` plus the whole tau bank stayed power-blind with the flag
    # ON, which on a power-varying set leaves a 68 W and a 500 W case with the same
    # track identical in every memory channel but one. `beam_dist` is deliberately
    # excluded -- it is the normalized footprint (geometry), and power is already its
    # own conditioning channel.
    p_scale = _power_scale(manifest, power_weighted)

    if dose_key in requested:
        weight = (dt * on).reshape((-1,) + (1,) * (footprint.ndim - 1))
        dose = np.cumsum(footprint * weight, axis=0)
        denom = float((dt * on).sum())
        if denom > 0:
            dose = dose / denom
        dose = dose * p_scale
        if footprint.ndim == 3:
            dose = dose[:, :, :, None]
        out[dose_key] = np.broadcast_to(dose.astype(np.float32)[None, :, None], shape).astype(np.float32)

    if recent_key in requested:
        dose_r = _leaky_dose(footprint, dt, on, _DOSE_RECENT_TAU_S) * p_scale
        if footprint.ndim == 3:
            dose_r = dose_r[:, :, :, None]
        out[recent_key] = np.broadcast_to(dose_r.astype(np.float32)[None, :, None], shape).astype(np.float32)

    # Multi-timescale bank: same leaky integrator, several tau. See _DOSE_TAU_BANK_S.
    for bank_key, tau in _tau_bank_for_prefix(prefix).items():
        if bank_key not in requested:
            continue
        banked = _leaky_dose(footprint, dt, on, tau) * p_scale
        if footprint.ndim == 3:
            banked = banked[:, :, :, None]
        out[bank_key] = np.broadcast_to(banked.astype(np.float32)[None, :, None], shape).astype(np.float32)


def _soak_field(manifest: Dict[str, Any], x, y, z, t_seconds) -> np.ndarray:
    """Dirichlet-face soak of a block from ambient toward T0. Shape (nt, nx, ny, nz).

    Separation of variables for dT/dt = alpha grad^2 T with T = T0 on every face and
    T = T_ambient uniformly at t = 0:

        (T - T0)/(Ti - T0) = prod_d (4/pi) sum_{n odd} (1/n) sin(n pi u_d)
                                                 exp(-alpha (n pi / L_d)^2 t)

    The z axis is treated as driven on both faces too: the substrate sits on a heated
    base and the free surface exchanges with the void, which the deck sets to the same
    T0 (tvoid). N_TERMS=15 odd harmonics converges to <0.5 % everywhere except the first
    frame, where the series is slowest and the answer is ambient by construction.
    """
    t0 = _dig_or(manifest, ("input", "simulation", "initial_temperature_k"), _AMBIENT_T_K)
    alpha = _material_diffusivity_m2_s(manifest)
    tt = np.asarray(t_seconds, dtype=np.float64).reshape(-1, 1)
    if abs(t0 - _AMBIENT_T_K) < 1e-9:
        nt = tt.size
        return np.full((nt, x.size, y.size, z.size), np.float32(t0), dtype=np.float32)

    from scipy.special import erf

    def axis_profile(coord: np.ndarray) -> np.ndarray:
        """Fraction of the ORIGINAL (ambient) state still remaining, per axis.

        Product of the two semi-infinite solutions, one per face (Newman's rule for the
        linear heat equation). Chosen over the odd-harmonic series because that series
        converges as 1/n and is ~7 % wrong at t=0 with any affordable term count, while
        this form is EXACT at t=0 (erf(inf)=1 in the interior, 0 at the faces) and
        accurate wherever the two faces' influences barely overlap -- sqrt(alpha t) is
        <=92 um here against a 200 um half-width, so that holds throughout the sweep.
        """
        c = np.asarray(coord, dtype=np.float64)
        span = float(c[-1] - c[0])
        if span <= 0:
            return np.ones((tt.size, c.size), dtype=np.float64)
        d_lo = (c - c[0]).reshape(1, -1)
        d_hi = (c[-1] - c).reshape(1, -1)
        denom = 2.0 * np.sqrt(np.maximum(alpha * tt, 1e-30))
        return np.clip(erf(d_lo / denom) * erf(d_hi / denom), 0.0, 1.0)

    px = axis_profile(x)[:, :, None, None]
    py = axis_profile(y)[:, None, :, None]
    pz = axis_profile(z)[:, None, None, :]
    remaining = px * py * pz                            # product solution
    return (t0 + (_AMBIENT_T_K - t0) * remaining).astype(np.float32)


def _h_norm_local(manifest: Dict[str, Any], soak: np.ndarray) -> np.ndarray:
    """Normalized enthalpy evaluated against the local soak temperature.

        H*(x,t) = A P / (rho (cp (T_l - T_soak(x,t)) + L_f) sqrt(pi D v r^3))

    Same closed form as the batch generator's `h_norm_ti64`, with the global T0 replaced
    by the soak field. Clamped to 1 K of remaining superheat so a cell the soak has driven
    to the liquidus cannot divide by zero -- that regime is the solver's business.
    """
    props = ("material", "properties")
    rho = _dig_or(manifest, props + ("density_kg_m3",), 4420.0)
    cp = _dig_or(manifest, props + ("specific_heat_j_kg_k",),
                 _dig_or(manifest, props + ("solid_specific_heat_j_kg_k",), 546.0))
    lf = _dig_or(manifest, props + ("latent_heat_j_kg",), 345000.0)
    tl = _dig_or(manifest, props + ("liquidus_temperature_k",), 1923.15)
    absorb = _dig_or(manifest, ("material", "optical", "normal_absorptivity"), 0.35)
    power = _dig_or(manifest, ("input", "laser", "power_w"), 100.0)
    speed = abs(_dig_or(manifest, ("input", "laser", "scan_speed_m_per_s"), 0.5))
    r0 = _dig_or(manifest, ("input", "laser", "beam_radius_m"), 5e-5)
    alpha = _material_diffusivity_m2_s(manifest)
    superheat = np.maximum(tl - soak.astype(np.float64), 1.0)
    denom = rho * (cp * superheat + lf) * np.sqrt(np.pi * alpha * max(speed, 1e-6) * r0 ** 3)
    field = absorb * power / np.maximum(denom, 1e-30)
    # Emitted as the RATIO to the same quantity at ambient, not the raw H*. Two reasons.
    # (1) Scale: raw H* lands at 25-32 here, ~30x every other conditioning channel and the
    #     same failure mode raw Kelvin had. The ratio is 1.0 in unsoaked material and
    #     rises where the block has warmed, so it is O(1) by construction.
    # (2) Non-redundancy: the P, V and r factors cancel in the ratio, and those are already
    #     three separate channels. What is left is exactly the part they cannot express --
    #     how much easier melting has become because this cell soaked.
    ambient_superheat = max(tl - _AMBIENT_T_K, 1.0)
    ref = rho * (cp * ambient_superheat + lf) * np.sqrt(np.pi * alpha * max(speed, 1e-6) * r0 ** 3)
    return (field * (ref / max(absorb * power, 1e-30))).astype(np.float32)


def _material_diffusivity_m2_s(manifest: Dict[str, Any]) -> float:
    """alpha = k / (rho * cp) from the embedded material block; Ti64-ish fallback."""
    # `input.material` is {"file": "..."} in the calibrated batches -- truthy but with no
    # "properties", so an `or` on it silently swallowed the real block and every case fell
    # back to the hardcoded default, making alpha a CONSTANT on any multi-material set.
    # Prefer the top-level material block and fall back only if it is genuinely absent.
    props = (manifest.get("material", {}).get("properties")
             or manifest.get("input", {}).get("material", {}).get("properties", {}))
    try:
        return (float(props["thermal_conductivity_w_m_k"])
                / (float(props["density_kg_m3"]) * float(props["specific_heat_j_kg_k"])))
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        return 3.0e-6


def _conduction_field(manifest, x, y, z, t_seconds, centres, dt, on, sigma):
    """Exact linear conduction G*f on the case grid, cached on disk per case.

    Spectral stepping (one k-space multiply per frame, one iFFT per export)
    with a mirrored-z image source at the free surface, per-material
    diffusivity from the manifest, and the same on-gate as the dose channels.
    Units follow the dose convention (q = 1 per unit time, power-blind); gas
    planes above the surface are zero. This function stays power-blind on purpose:
    `custom.power_weighted_conduction` is applied by the CALLER after the return,
    because the peak normalisation below would divide it out and the disk cache
    key carries no power term. float16 on disk: the channel is
    normalised downstream, and the cache is dataset-sized.
    """
    import hashlib
    import json
    from pathlib import Path as _Path

    from . import linear_response as _lr

    alpha = _material_diffusivity_m2_s(manifest)
    surface = float(manifest.get("derived", {}).get("surface_z_m", 0.0))
    nx, ny, nz, nt = x.size, y.size, z.size, t_seconds.size
    # The source term f is (centres, on): two beams of one multi-beam case share
    # every scalar below, so a key without the path made all per-beam calls hash
    # identically -- beam 1 read beam 0's cached field and the superposed channel
    # was silently N x beam 0 (2026-08-13). Hash the actual source.
    src_h = hashlib.sha1(
        np.ascontiguousarray(np.asarray(centres, np.float64).round(9)).tobytes()
        + np.ascontiguousarray(np.asarray(on, np.float64).round(6)).tobytes()
    ).hexdigest()[:12]
    key = hashlib.sha1(json.dumps([
        str(manifest.get("case_name", "?")), int(nt), int(nx), int(ny), int(nz),
        round(alpha, 12), round(float(sigma), 9), float(t_seconds[0]),
        float(t_seconds[-1]), round(surface, 9), src_h,
    ]).encode()).hexdigest()[:20]
    cache_dir = _Path(os.environ.get("PINO_CONDUCTION_CACHE",
                                     _Path.home() / ".pino" / "conduction_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"conduction_{key}.npy"
    if cached.exists():
        try:
            return np.load(cached).astype(np.float32)
        except (ValueError, EOFError, OSError):
            # torn file from an interrupted writer: recompute and rewrite
            pass

    metal = np.where(z <= surface + 1e-12)[0]
    nz_m = int(metal.size)
    dz = float(np.median(np.abs(np.diff(z)))) if z.size > 1 else 1e-5
    sim = _lr.SpectralConduction(
        (nx, ny, nz_m), (float(x[1] - x[0]), float(y[1] - y[0]), dz),
        alpha=alpha, sigma0=float(sigma))
    outf = np.zeros((nt, nx, ny, nz), dtype=np.float16)
    # grid z ascends: metal plane j is depth index (surface at metal[-1])
    for f in range(nt):
        sim.step(float(dt[f]), float(centres[f, 0] - x[0]),
                 float(centres[f, 1] - y[0]), q=float(on[f]))
        field = sim.field()                      # (nx, ny, nz_m), [:,:,0] = surface
        outf[f, :, :, metal[::-1]] = np.moveaxis(field, 2, 0).astype(np.float16)
    # per-case [0, 1] normalisation, the path_dose convention: raw units are
    # arbitrary (q = 1 dose scale, ~1e-4), and unscaled the channel would be
    # numerically invisible next to the O(1) inputs
    peak = float(outf.max())
    if peak > 0:
        outf = (outf.astype(np.float32) / peak).astype(np.float16)
    # atomic publish: concurrent dataloader workers (or two training runs) may
    # build the same cold case; a torn np.save would crash every later reader.
    # Unique tmp name + os.replace makes the last full write win cleanly.
    tmp = cached.parent / f".{cached.stem}.{os.getpid()}.npy"
    np.save(tmp, outf)          # ends in .npy, so np.save appends nothing
    os.replace(tmp, cached)
    return outf.astype(np.float32)


#: Path channels that depend only on the case manifest -- no beam kinematics, no
#: coordinates, no time axis. Kept separate so a caller that HAS a manifest but cannot
#: reconstruct the beam path (a staged derived member) can still supply them.
CONSTANT_PATH_CHANNELS = ("preheat_T0", "beam_shape_uniform", "focus_factor", "beam_radius")


def build_constant_path_channels(requested, manifest: dict, shape) -> Dict[str, np.ndarray]:
    """The case-constant path channels, broadcast to ``shape``.

    One implementation, two callers: `_build_path_channels` (the full path, which also
    needs the beam trajectory) and the staged-member runtime store (which has the manifest
    but no per-frame beam path). Duplicated builders drift apart: an earlier preheat bug
    lived in the third of three near-identical builders.
    """
    requested_set = set(requested)
    out: Dict[str, np.ndarray] = {}

    def _const(value: float) -> np.ndarray:
        return np.full(shape, np.float32(value), dtype=np.float32)

    if "preheat_T0" in requested_set:
        out["preheat_T0"] = _const(
            _dig_or(manifest, ("input", "simulation", "initial_temperature_k"), 293.15))
    if "beam_shape_uniform" in requested_set:
        dist = str(((manifest.get("input") or {}).get("laser") or {})
                   .get("heat_distribution", "gaussian")).lower()
        out["beam_shape_uniform"] = _const(1.0 if dist == "uniform" else 0.0)
    if "focus_factor" in requested_set:
        out["focus_factor"] = _const(
            _dig_or(manifest, ("input", "_design", "focus_factor"), 1.0))
    if "beam_radius" in requested_set:
        # Emitted in METRES; `_preprocess_var` divides it by ref_scales["L"] like every
        # other length-valued channel. Needed because the plain H_norm is P/sqrt(|V|) and
        # carries no radius term, and `beam_dist` -- which encodes the footprint width
        # spatially -- is a trajectory channel a derived steady member does not have.
        out["beam_radius"] = _const(
            _dig_or(manifest, ("input", "laser", "beam_radius_m"), 5e-5))
    return out


def _build_path_channels(
    requested: Iterable[str],
    manifest: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t_seconds: np.ndarray,
    power_weighted: bool = False,
    power_weighted_conduction: bool = False,
    conduction_sigma_factor: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Build requested path channels as ``(1, nt, 1, nx, ny, nz)`` float32 arrays.

    ``conduction_sigma_factor`` sets the standard deviation of the ``conduction_field`` source to that
    multiple of the beam radius. The solver's Gaussian beam falls to 1/e^2 at the beam radius r0, so its
    intensity has a standard deviation of r0/2: 0.5 matches it. The default 1.0 is the original channel (a
    source twice as wide), kept so every model trained on it is fed the same input at evaluation.

    ``power_weighted`` scales EVERY dose channel of the family -- ``path_dose``
    (after its per-case dwell normalization), ``path_dose_recent`` and the multi-tau
    bank -- by ``power_w / _P_REF_W``, so the channel amplitude encodes deposited
    power. Without it (default) the dose channels are per-case [0,1] geometric+dwell
    shapes that are power-blind — fine for a fixed-power dataset, but on a
    power-varying dataset a constant-per-case power cancels in the normalization, so
    the network cannot read "this case ran hotter" from the memory channels.

    ``beam_dist`` stays power-blind on purpose: it is the normalized footprint
    (where the beam is), and power already has its own ``P`` conditioning channel.

    Left off by default so an already-trained fixed-power model is not fed rescaled
    inputs at eval time — NOT because a power-varying set should go without it. Set
    ``custom.power_weighted_dose: true`` (or ``PINO_POWER_WEIGHTED_DOSE=1``) on any
    dataset whose power varies.
    """
    requested = [k for k in requested if k in PATH_CHANNEL_NAMES]
    if not requested:
        return {}
    nx, ny, nz, nt = x.size, y.size, z.size, t_seconds.size
    mb_specs = _multibeam_specs(manifest)
    mb_paths = None
    mb_weights = None
    if mb_specs is not None:
        # Per-beam kinematics through the SAME single-beam machinery; combined
        # here as power-weighted quantities so one beam reproduces the old
        # channels exactly. beam_x/beam_y become the power-weighted centroid of
        # the still-active beams; the footprint/dose/conduction combination
        # happens below at their construction sites.
        mb_paths = [_beam_path_over_time(s.manifest, t_seconds) for s in mb_specs]
        mb_weights = _beam_weights_at(mb_specs, t_seconds)          # (nt, nb)
        norm = np.maximum(mb_weights.sum(axis=1, keepdims=True), 1e-12)
        wn = mb_weights / norm
        centres = np.sum(
            np.stack([p[0] for p in mb_paths], axis=2) * wn[:, None, :], axis=2
        )
        directions = mb_paths[int(np.argmax([s.power_fraction for s in mb_specs]))][1]
        speeds = np.sum(np.stack([p[2] for p in mb_paths], axis=1) * wn, axis=1)
        if not _MULTIBEAM_ANNOUNCED:
            _MULTIBEAM_ANNOUNCED.add(1)
            print(f"[multibeam] {len(mb_specs)} beams, power fractions "
                  f"{[round(s.power_fraction, 3) for s in mb_specs]}; beam_x/beam_y are the "
                  "power-weighted centroid, footprint/dose/conduction are per-beam sums",
                  flush=True)
    else:
        centres, directions, speeds = _beam_path_over_time(manifest, t_seconds)
    out: Dict[str, np.ndarray] = {}
    shape = (1, nt, 1, nx, ny, nz)

    if "beam_x" in requested:
        out["beam_x"] = np.broadcast_to(
            centres[:, 0].astype(np.float32).reshape(1, nt, 1, 1, 1, 1), shape
        ).astype(np.float32)
    if "beam_y" in requested:
        out["beam_y"] = np.broadcast_to(
            centres[:, 1].astype(np.float32).reshape(1, nt, 1, 1, 1, 1), shape
        ).astype(np.float32)

    requested_set = set(requested)

    # Case-constant broadcasts (no beam geometry needed). Raw SI: the FiLM path
    # routes broadcast-constant channels into conditioning, matching V_scan/P.
    def _const(value: float) -> np.ndarray:
        return np.full(shape, np.float32(value), dtype=np.float32)

    out.update(build_constant_path_channels(requested_set, manifest, shape))
    if requested_set & {"soak_field", "h_norm_local"}:
        soak = _soak_field(manifest, x, y, z, t_seconds)
        if "soak_field" in requested_set:
            out["soak_field"] = soak.reshape(shape)
        if "h_norm_local" in requested_set:
            out["h_norm_local"] = _h_norm_local(manifest, soak).reshape(shape)

    local_keys = {"beam_s", "beam_n", "beam_speed", "beam_s_arc", "beam_n_arc",
                  "beam_veff", "conduction_field"}
    family_keys = {
        "beam_dist",
        "path_dose",
        "path_dose_recent",
        "beam_dist_sigma",
        "path_dose_sigma",
        "path_dose_recent_sigma",
        "beam_dist_sigma_exp",
        "path_dose_sigma_exp",
        "path_dose_recent_sigma_exp",
    }
    # The tau bank is BUILT inside the block below, so its names have to be able to
    # OPEN it. They were in `PATH_CHANNEL_NAMES` and in neither key set, so a request
    # naming only tau channels fell straight through and returned {} -- no error, no
    # channel, a model training on a silently truncated input. All twelve
    # (bare/_sigma/_sigma_exp), not just the four bare ones. Live configs escaped only
    # by also requesting `beam_dist`; the tau-only ablation anyone would write to
    # isolate the bank is exactly the case that breaks. Reported by random paths,
    # 2026-09-08; reproduced before fixing.
    family_keys = family_keys | set(_TAU_BANK_NAMES)
    if requested_set & (local_keys | family_keys):
        sigma = float(manifest["input"]["laser"]["beam_radius_m"])
        # Same present-but-NULL case as in `_beam_path_over_time` above.
        _off_raw = manifest.get("derived", {}).get("laser_off_time_s")
        off = np.inf if _off_raw is None else float(_off_raw)
        tt = np.asarray(t_seconds, dtype=np.float64)
        dt = np.gradient(tt) if nt > 1 else np.ones(1)
        on = (tt <= off).astype(np.float64)
        xg = x.astype(np.float64).reshape(1, nx, 1)
        yg = y.astype(np.float64).reshape(1, 1, ny)
        cx = centres[:, 0].reshape(nt, 1, 1)
        cy = centres[:, 1].reshape(nt, 1, 1)
        dx = xg - cx
        dy = yg - cy
        if "beam_s" in requested_set or "beam_n" in requested_set:
            vhx = directions[:, 0].reshape(nt, 1, 1)
            vhy = directions[:, 1].reshape(nt, 1, 1)
            if "beam_s" in requested_set:
                s = (dx * vhx + dy * vhy) / sigma
                out["beam_s"] = np.broadcast_to(s.astype(np.float32)[None, :, None, :, :, None], shape).astype(np.float32)
            if "beam_n" in requested_set:
                n = (-dx * vhy + dy * vhx) / sigma
                out["beam_n"] = np.broadcast_to(n.astype(np.float32)[None, :, None, :, :, None], shape).astype(np.float32)
        if "beam_speed" in requested_set:
            out["beam_speed"] = _broadcast_time_scalar(speeds, shape)
        if requested_set & {"beam_s_arc", "beam_n_arc"}:
            # Static foot map onto the trajectory polyline (the path does not
            # move, so the nearest-point assignment is frame-invariant; only
            # the beam's own arc position changes per frame). Vertex-level
            # nearest point is enough: vertex spacing is V*dt, far below sigma
            # at the export cadences used here. At a self-intersection the
            # nearest branch wins, which keeps the frame single-valued.
            traj = centres.astype(np.float64)                      # (nt, 2)
            seg_v = np.diff(traj, axis=0)
            seg_l = np.hypot(seg_v[:, 0], seg_v[:, 1])
            arc = np.concatenate([[0.0], np.cumsum(seg_l)])        # (nt,)
            tang = seg_v / np.maximum(seg_l, 1e-30)[:, None]       # (nt-1, 2)
            gx = x.astype(np.float64)
            gy = y.astype(np.float64)
            foot = np.empty((nx, ny), dtype=np.int64)
            for i0 in range(0, nx, 32):                            # bound memory
                d2 = ((gx[i0:i0 + 32, None, None] - traj[None, None, :, 0]) ** 2
                      + (gy[None, :, None] - traj[None, None, :, 1]) ** 2)
                foot[i0:i0 + 32] = np.argmin(d2, axis=2)
            tj = tang[np.minimum(foot, len(tang) - 1)]             # (nx, ny, 2)
            dxf = gx[:, None] - traj[foot, 0]
            dyf = gy[None, :] - traj[foot, 1]
            if "beam_s_arc" in requested_set:
                # signed arc distance foot -> beam, plus the along-tangent
                # remainder at the foot so the straight-path limit is exact
                s_arc = (arc[foot][None, :, :] - arc[:, None, None]
                         + (dxf * tj[..., 0] + dyf * tj[..., 1])[None, :, :]) / sigma
                out["beam_s_arc"] = np.broadcast_to(
                    s_arc.astype(np.float32)[None, :, None, :, :, None], shape
                ).astype(np.float32)
            if "beam_n_arc" in requested_set:
                n_arc = (-dxf * tj[..., 1] + dyf * tj[..., 0]) / sigma
                out["beam_n_arc"] = np.broadcast_to(
                    n_arc.astype(np.float32)[None, None, None, :, :, None], shape
                ).astype(np.float32)
        if "beam_veff" in requested_set:
            # signed curvature of the trajectory at each frame: heading change
            # per unit arc, positive for a left turn (the beam_n convention)
            th = np.arctan2(directions[:, 1], directions[:, 0])
            dth = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
            darc = np.maximum(
                np.hypot(np.diff(centres[:, 0]), np.diff(centres[:, 1])), 1e-12)
            kap = np.zeros(nt)
            kap[1:-1] = ((dth[:-1] + dth[1:]) / (darc[:-1] + darc[1:]))
            n_m = (-(xg - centres[:, 0].reshape(nt, 1, 1)) * directions[:, 1].reshape(nt, 1, 1)
                   + (yg - centres[:, 1].reshape(nt, 1, 1)) * directions[:, 0].reshape(nt, 1, 1))
            nk = np.clip(n_m * kap.reshape(nt, 1, 1), -0.8, 0.8)
            veff = (speeds.reshape(nt, 1, 1) * (1.0 - nk)).astype(np.float32)
            out["beam_veff"] = np.broadcast_to(
                veff[None, :, None, :, :, None], shape).astype(np.float32)
        if "conduction_field" in requested_set:
            _cond_sigma_factor = float(conduction_sigma_factor)
            if not (_cond_sigma_factor > 0.0 and np.isfinite(_cond_sigma_factor)):
                raise ValueError(f"conduction_sigma_factor must be a positive number, got {conduction_sigma_factor!r}")
            # Applied AFTER `_conduction_field` returns, never inside it, for two
            # reasons that both bite: the builder's per-case peak normalisation
            # divides any pre-scaling straight back out (a no-op), and its disk
            # cache key carries no power term, so scaling the cached artifact would
            # serve a field at the wrong power on every hit -- the 2026-08-13
            # multibeam key-collision failure class. Post-read scaling keeps every
            # existing cache entry valid.
            _cond_scale = _power_scale(manifest, power_weighted_conduction)
            if mb_specs is not None:
                # Heat conduction is linear: the multi-beam field is the sum of
                # per-beam single-source fields, each source scaled by its power
                # fraction and gated by its own turn-off (folded into q via on).
                cond = None
                for b, (c_b, _d, _s) in enumerate(mb_paths):
                    term = _conduction_field(
                        manifest, x, y, z, t_seconds, c_b, dt, mb_weights[:, b], sigma * _cond_sigma_factor
                    )
                    cond = term if cond is None else cond + term
                out["conduction_field"] = (
                    cond * _cond_scale).reshape(shape).astype(np.float32)
            else:
                out["conduction_field"] = (
                    _conduction_field(manifest, x, y, z, t_seconds, centres, dt, on,
                                      sigma * _cond_sigma_factor)
                    * _cond_scale
                ).reshape(shape).astype(np.float32)

        r2 = (xg - cx) ** 2 + (yg - cy) ** 2  # (nt, nx, ny)
        heat_dist = str(((manifest.get("input") or {}).get("laser") or {})
                        .get("heat_distribution", "gaussian")).lower()
        if heat_dist == "uniform":
            # Flat-top footprint for a uniform (top-hat) beam: ~1 inside the beam
            # radius, smooth tanh edge of width sigma/4 (a hard indicator aliases
            # frame-to-frame as the beam crosses cell boundaries; the smooth edge
            # spans ~1 cell at the sweep's 10 um grids). Gaussian cases keep the
            # legacy formula BIT-IDENTICAL -- every earlier dataset reads it.
            r = np.sqrt(r2)
            footprint = (0.5 * (1.0 - np.tanh(4.0 * (r - sigma) / sigma))).astype(np.float32)
        else:
            footprint = np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)  # peak 1.0
        if mb_specs is not None:
            # Power-fraction-weighted sum of per-beam Gaussians, each gated by
            # its own turn-off (in the weights). `on` collapses to all-ones so
            # the shared emitters below do not double-gate.
            # Flat-top is NOT implemented for multibeam: every multibeam sweep to
            # date is Gaussian. Refuse loudly rather than silently replacing a
            # uniform case's flat-top with a Gaussian sum.
            if heat_dist == "uniform":
                raise ValueError(
                    "multibeam + heat_distribution='uniform' is unsupported: the "
                    "multibeam footprint is a sum of Gaussians and would silently "
                    "misrepresent a flat-top beam. Add a flat-top branch here if a "
                    "sweep ever needs it."
                )
            fp = np.zeros((nt, nx, ny), dtype=np.float64)
            for b, (c_b, _d, _s) in enumerate(mb_paths):
                r2_b = (xg - c_b[:, 0].reshape(nt, 1, 1)) ** 2 + (yg - c_b[:, 1].reshape(nt, 1, 1)) ** 2
                fp += mb_weights[:, b].reshape(nt, 1, 1) * np.exp(-r2_b / (2.0 * sigma * sigma))
            footprint = fp.astype(np.float32)
            on = np.ones_like(on)
        # On-state gate: past laser_off_time_s the beam is OFF, and the
        # instantaneous footprint must vanish -- cooling-tail frames
        # (material_ck300: finish = 1.25*track/V) otherwise show a full-power
        # beam parked at the track end while the target field cools. The dose
        # channels already gated on this; beam_dist must too.
        _emit_beam_family(
            out, requested_set, prefix="", footprint=footprint, shape=shape, on=on, dt=dt,
            power_weighted=power_weighted, manifest=manifest,
        )
        # Same omission one level down, and it survived the outer fix: these gates list
        # the family by hand too, so a request for only `path_dose_tau*_sigma` opened the
        # outer block and then skipped the sigma footprint that builds them. Found by the
        # requested-vs-returned guard below the moment the outer gate was fixed -- which
        # is the argument for the guard over another hand-maintained name list.
        if requested_set & ({"beam_dist_sigma", "path_dose_sigma", "path_dose_recent_sigma",
                             "beam_dist_sigma_exp", "path_dose_sigma_exp",
                             "path_dose_recent_sigma_exp"}
                            | set(_tau_bank_for_prefix("sigma"))
                            | set(_tau_bank_for_prefix("sigma_exp"))):
            zz = z.astype(np.float64).reshape(1, 1, 1, nz)
            z0 = _focus_z_m(manifest)
            z_r = _rayleigh_range_m(manifest, sigma)
            sigma_z = sigma * np.sqrt(1.0 + ((zz - z0) / z_r) ** 2)
            # Same shape dispatch as the base footprint above: a flat-top case's
            # sigma channels must not contradict its own beam_dist (they did until
            # 2026-08-14 -- the sigma family was unconditionally Gaussian). The
            # gaussian branch is BIT-IDENTICAL to the legacy formula.
            if heat_dist == "uniform":
                r_z = np.sqrt(r2)[:, :, :, None]
                footprint_sigma = (
                    0.5 * (1.0 - np.tanh(4.0 * (r_z - sigma_z) / sigma_z))
                ).astype(np.float32)
            else:
                footprint_sigma = np.exp(-r2[:, :, :, None] / (2.0 * sigma_z * sigma_z)).astype(np.float32)
            _emit_beam_family(
                out, requested_set, prefix="sigma", footprint=footprint_sigma, shape=shape, on=on, dt=dt,
                power_weighted=power_weighted, manifest=manifest,
            )
            if requested_set & ({"beam_dist_sigma_exp", "path_dose_sigma_exp",
                                 "path_dose_recent_sigma_exp"}
                                | set(_tau_bank_for_prefix("sigma_exp"))):
                surface = float(manifest.get("derived", {}).get("surface_z_m", 0.0))
                delta = max(_abs_depth_m(manifest), np.finfo(np.float64).tiny)
                absorb = np.exp(-np.abs(zz - surface) / delta).astype(np.float32)
                _emit_beam_family(
                    out, requested_set, prefix="sigma_exp", footprint=footprint_sigma * absorb,
                    shape=shape, on=on, dt=dt, power_weighted=power_weighted, manifest=manifest,
                )
    # Every requested channel must come back, or say which did not. The tau-bank bug
    # above was silent for exactly one reason -- nothing compared the request against
    # the result -- and the same hole reopens for the NEXT name added to
    # `PATH_CHANNEL_NAMES` and forgotten in the key sets, which is how this one arose.
    # A full request of all of PATH_CHANNEL_NAMES returns all of them, so this refuses
    # only genuine misses.
    _missing = [k for k in requested_set if k not in out]
    if _missing:
        raise KeyError(
            f"path channels {sorted(_missing)} were requested but not built. They are "
            "recognised names, so this is a gap in `_build_path_channels` -- most likely "
            "the name is absent from `local_keys`/`family_keys`, which gate whether the "
            "builder block runs at all. Returning them silently would train the model on "
            "a truncated input."
        )
    return out


def _apply_moving_frame(
    var: Dict[str, np.ndarray],
    frame_steps: Optional[int],
    window: int,
    super_res_factor: int,
    *,
    centres: Optional[np.ndarray] = None,
    x0: Optional[float] = None,
    dx: Optional[float] = None,
    stops: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Crop every field to a laser-tracking moving frame; optional time-average.

    ``stops`` is the per-case `frame_transfer.laser_on_frame_stop` (``(B,)``); the tail
    of ``frame_steps`` frames then ends there instead of at the export end, so a laser
    that overruns its path is windowed on its declared track, not against the +x wall.

    ``centres`` is the per-case, per-frame beam position, ``(B, T, 2)`` in the same units
    as ``x0``/``dx``. It is REQUIRED, because the alternative -- the old
    ``_crop_shift_moving_frame`` index shift, which advances the window exactly one cell
    per frame -- follows the beam only when ``V_scan * dt == dx``. Measured on the spkh123
    train split (2026-08-20) the beam advances 0.051 to 0.482 cells per frame against that
    assumed 1.000, so over a 30-frame window it drifts 21-27 cells (4.2-5.4 beam radii)
    out of the window and the time average smears the melt pool across the drift. Guessing
    is what produced those datasets, so this refuses instead. See
    ``models.pipeline.frame_transfer`` and ``tests/test_frame_transfer.py``.

    The output grid keeps the legacy width ``nX - (Nsteps - 1) * super_res_factor`` so
    configs and latent lattices sized against the old crop still fit; only where the window
    SITS changes.
    """
    # Absolute: tests load this module standalone via importlib, where a relative
    # import beyond the package fails.
    from lpno.models.pipeline.frame_transfer import (
        beam_following_x_slice,
        legacy_window_split,
    )

    ref = var[next(iter(var))]
    if ref.ndim != 6:
        raise ValueError(f"[NPZ] Expected 6D time-axis arrays; got {ref.shape}.")
    B, T, nX = ref.shape[0], ref.shape[1], ref.shape[3]
    Nsteps = T if frame_steps is None else int(frame_steps)
    if not (1 <= Nsteps <= T):
        raise ValueError(f"[NPZ] moving_frame_steps must be in [1, {T}]; got {Nsteps}.")
    if not (1 <= window <= Nsteps):
        raise ValueError(f"[NPZ] avg_window_size must be in [1, {Nsteps}]; got {window}.")
    if centres is None or x0 is None or dx is None:
        raise ValueError(
            "[NPZ] the moving frame needs the beam track (centres, x0, dx). Without it the "
            "only option is the one-cell-per-frame index shift, which is wrong on any "
            "dataset whose export cadence is not exactly one cell per frame -- measured "
            "0.051-0.482 cells/frame on spkh123, smearing a 30-frame average over 4-5 beam "
            "radii. Load the case manifests so the beam path is available."
        )
    centres = np.asarray(centres, dtype=np.float64)
    if centres.shape[:2] != (B, T):
        raise ValueError(
            f"[NPZ] centres must be (B, T, 2) = {(B, T, 2)}; got {centres.shape}."
        )

    width = nX - (Nsteps - 1) * int(super_res_factor)
    if width <= 0:
        raise ValueError("[NPZ] moving_frame_steps must be smaller than nX.")
    behind, ahead = legacy_window_split(width)
    # The window is placed from the beam position of the frames that survive the crop.
    ends = np.full(B, T, dtype=int) if stops is None else np.asarray(stops, dtype=int).reshape(-1)
    if ends.shape != (B,):
        raise ValueError(f"[NPZ] stops must have one entry per case ({B}), got {ends.shape}.")
    if (ends < Nsteps).any() or (ends > T).any():
        raise ValueError(
            f"[NPZ] frame stops {ends.tolist()} must lie in [{Nsteps}, {T}] "
            f"(moving_frame_steps={Nsteps} frames are needed before each stop)."
        )
    slices = [
        [beam_following_x_slice(beam_cx=float(centres[b, ends[b] - Nsteps + t, 0]),
                                x0=float(x0), dx=float(dx), nx=nX, behind=behind, ahead=ahead)
         for t in range(Nsteps)]
        for b in range(B)
    ]

    processed: Dict[str, np.ndarray] = {}
    for name, arr in var.items():
        cropped = np.empty(
            (B, Nsteps, arr.shape[2], width, arr.shape[4], arr.shape[5]), dtype=arr.dtype
        )
        for b in range(B):
            tail_arr = arr[b, ends[b] - Nsteps:ends[b]]
            for t in range(Nsteps):
                cropped[b, t] = tail_arr[t, :, slices[b][t]]
        if window > 1:
            # `[:, -1]`, not `[:, -1:]`: indexing DROPS the time axis, slicing keeps
            # it as size 1. This function's contract (see load_consolidated_npz) is
            # that a moving frame with avg_window_size>1 removes the axis, and the
            # arch layers depend on it -- a surviving T=1 reaches a 2D FNO as
            # (B,1,C,X,Z) and trips "Only 4D tensors accepted for 2D FNO". Regressed
            # in a540be1 (2026-08-20) and invisible to every real training, which
            # reads through NpzDirStore rather than this eager path; CI's smoke is
            # the only consumer, so it broke there alone.
            processed[name] = _moving_average_time(
                cropped, window=window, axis=1
            ).astype(np.float32)[:, -1]
        else:
            processed[name] = cropped
    return processed


def load_consolidated_npz(
    input_dir: str | Path,
    requested_fields: Optional[Iterable[str]] = None,
    *,
    super_res_factor: int = 1,
    time_axis: bool = True,
    avg_window_size: Optional[int] = 1,
    moving_frame_steps: Optional[int] = None,
    moving_ref_frame: bool = True,
    path_channels: Optional[Iterable[str]] = None,
    material_channels: Optional[Iterable[str]] = None,
    power_weighted: bool = False,
    power_weighted_conduction: bool = False,
    conduction_sigma_factor: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Load consolidated NPZ case(s) into the pipeline var dict.

    ``power_weighted`` / ``power_weighted_conduction`` are forwarded to
    `_build_path_channels`, spelled as that function spells them. Until 2026-09-08 this
    loader took no config at all and called it with six positional arguments, so both
    flags sat at their False default and `custom.power_weighted_dose: true` was silently
    INERT on this path while the lazy store honoured it -- one config, two different
    input channels, depending on which loader ran.

    Returns raw FLOW-3D field names (``Temperature``, ``Velocity``, ...) plus
    ``x/y/z/P/V_scan/time_s`` shaped like ``VTKOpenFOAMDataSource.parse_directory``
    output, so the existing ``_preprocess_var`` tail (rename + non-dim) applies
    unchanged. In time-axis mode arrays are ``(B,T,C,X,Y,Z)`` before the moving
    frame; the moving frame trims X and (if ``avg_window_size>1``) removes the
    time axis, matching the VTK path.
    """
    # PINO_LAZY_CROP_Z_TOP is a LAZY-store setting; this eager path does not honour
    # it. Saying so, because the silent version is the duplicated-builder trap: the
    # run would train on uncropped data while its profile says otherwise, and the
    # only symptom is a number.
    import os as _os

    if _os.environ.get("PINO_LAZY_CROP_Z_TOP", "").strip() not in ("", "0"):
        print(
            "WARNING: PINO_LAZY_CROP_Z_TOP is set but this run uses the EAGER "
            "consolidated-NPZ path, which does not apply it -- the gas-side z-crop "
            "is a read-time feature of NpzDirStore. This run trains on the FULL z "
            "extent. Use a lazy dataset (PINO_LAZY_NPZ=1) for the crop to take "
            "effect, or drop the variable to silence this.",
            flush=True,
        )
    if requested_fields is None:
        requested_fields = ["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"]
    field_names = list(requested_fields)

    entries: List[Dict[str, np.ndarray]] = []

    case_centres: list = []
    case_stops: list = []

    case_grids: list = []
    for case in _case_files(input_dir):
        with np.load(_open_case_safely(case)) as data:   # PackSlicePath-safe, as at :92/:161
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
            z = np.asarray(data["z"], dtype=np.float32)
            nx, ny, nz = int(x.size), int(y.size), int(z.size)
            t = np.asarray(data["time_s"], dtype=np.float32)
            nt = int(t.size)
            power = float(np.asarray(data["power_w"]))
            speed = float(np.asarray(data["scan_speed_m_per_s"]))

            shape = (1, nt, 1, nx, ny, nz)
            entry: Dict[str, np.ndarray] = {
                "x": np.broadcast_to(x.reshape(1, 1, 1, nx, 1, 1), shape).astype(np.float32),
                "y": np.broadcast_to(y.reshape(1, 1, 1, 1, ny, 1), shape).astype(np.float32),
                "z": np.broadcast_to(z.reshape(1, 1, 1, 1, 1, nz), shape).astype(np.float32),
                "P": np.full(shape, power, dtype=np.float32),
                "V_scan": np.full(shape, speed, dtype=np.float32),
                "time_s": np.broadcast_to(t.reshape(1, nt, 1, 1, 1, 1), shape).astype(np.float32),
            }
            for name in field_names:
                entry[name] = _field_to_bt_cxyz(name, data, nt, nx, ny, nz)
        need_track = bool(time_axis and moving_ref_frame)
        if path_channels or material_channels or need_track:
            manifest = _load_manifest(case)
            if manifest is None:
                requested = list(path_channels or []) + list(material_channels or [])
                if requested:
                    raise ValueError(
                        f"[NPZ] manifest channels {requested} requested but no "
                        f"'{case.stem}.manifest.json' (or case_manifest.json) beside '{case.name}'."
                    )
                raise ValueError(
                    f"[NPZ] moving_ref_frame needs the beam path, but there is no "
                    f"'{case.stem}.manifest.json' (or case_manifest.json) beside "
                    f"'{case.name}'. The window has to follow the beam's real position; "
                    "see models.pipeline.frame_transfer for why an index shift will not do."
                )
            if path_channels:
                entry.update(_build_path_channels(
                    path_channels, manifest, x, y, z, t,
                    power_weighted=power_weighted,
                    power_weighted_conduction=power_weighted_conduction,
                    conduction_sigma_factor=conduction_sigma_factor,
                ))
            if material_channels:
                entry.update(_build_material_channels(material_channels, manifest, shape))
            if need_track:
                from lpno.models.pipeline.frame_transfer import laser_on_frame_stop

                case_centres.append(_beam_path_over_time(manifest, t)[0])
                case_stops.append(laser_on_frame_stop(manifest, t))
                case_grids.append((float(x[0]), float(x[1] - x[0])))
        entries.append(entry)

    if not entries:
        raise ValueError(f"[NPZ] No cases loaded from '{input_dir}'.")

    # Cases must share the timestep count to stack along batch.
    t_counts = {e["time_s"].shape[1] for e in entries}
    if len(t_counts) != 1:
        raise ValueError(f"[NPZ] Cases have differing timestep counts {sorted(t_counts)}; cannot stack.")
    var: Dict[str, np.ndarray] = {
        k: np.concatenate([e[k] for e in entries], axis=0) for k in entries[0]
    }

    disable_shift_avg = time_axis and (avg_window_size is None) and (moving_frame_steps is None)
    if time_axis and moving_ref_frame and not disable_shift_avg:
        window = 1 if avg_window_size is None else int(avg_window_size)
        grids = set(case_grids)
        if len(grids) != 1:
            raise ValueError(
                f"[NPZ] the moving frame needs one shared x grid across cases; got {sorted(grids)}."
            )
        gx0, gdx = case_grids[0]
        var = _apply_moving_frame(
            var, moving_frame_steps, window, super_res_factor,
            centres=np.stack(case_centres, axis=0), x0=gx0, dx=gdx,
            stops=np.asarray(case_stops, dtype=int) if case_stops else None,
        )
    elif not time_axis:
        # Collapse to the final timestep (mirrors the VTK non-time "latest file" path).
        var = {k: arr[:, -1] for k, arr in var.items()}

    return var
