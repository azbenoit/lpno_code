"""Time-dependent random-path inference adapter for /infer_path.

Serves the free-form-path checkpoints (time_as_batch: each frame is an independent
sample; the frame count follows the requested frame_dt_s). Path conditioning channels
(beam_x/beam_y/beam_dist/path_dose) are built by the submodule's own
``npz_datasource`` helpers from either a preset case manifest or a drawn,
clamped/resampled polyline.

Models load lazily on first request per model and stay cached. The service GPU
lock serializes normal requests, but it is released when a request times out
while its worker thread keeps running, so lazy loading is additionally guarded
by an in-process load lock (see ``_load_lock``).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .arch_build import build_gaot_kwargs, build_rigno_net, is_fno_config, is_gaot
from .config import Settings
from . import submodule
from .gt import T_FILL_K, T_LIQUIDUS_K, T_SOLIDUS_K, load_axes, mask_fields
from .model_physics import (  # noqa: F401 - re-exported for tests and callers
    MAX_PATH_LENGTH_M,
    P_WATTS,
    V_SCAN_M_PER_S,
    ModelPhysics,
    physics_config_mismatches,
    physics_from_meta as _physics_from_meta,
)
from .model_registry import evict_to_cap
from . import model_registry

logger = logging.getLogger(__name__)


class PathInputError(ValueError):
    """Invalid /infer_path input detected past pydantic (e.g. unknown preset)."""


# Physics knobs the calibrated channel builders read out of the case manifest.
# Read from a real calibrated case (runs/batches/ti64_rand150_newphys_20260817/
# runs/r150np_000/case_manifest.json): initial_temperature_k is 293.15, while
# heat_distribution and _design.focus_factor are ABSENT, so the builders' own
# defaults ("gaussian" -> 0.0, and 1.0) are what the training data used. They are
# stated explicitly rather than inherited: a silent default is exactly how a
# conditioning channel ends up carrying a value nobody chose.
# NOTE 293.15 != npz_datasource._AMBIENT_T_K (273.15), so the soak field does NOT
# short-circuit and h_norm_local sits slightly above 1.0 -- which is what the
# training cases do too.
DRAWN_INITIAL_T_K = 293.15
DRAWN_HEAT_DISTRIBUTION = "gaussian"
DRAWN_FOCUS_FACTOR = 1.0
DRAWN_SURFACE_Z_M = 0.0

# Fixed physics for drawn paths (matches the training/test sweep generation).
# P_WATTS / V_SCAN_M_PER_S / MAX_PATH_LENGTH_M live in app.model_physics (the
# ModelPhysics defaults) and are re-exported here.
BEAM_SIGMA_M = 5.0e-5  # Gaussian sigma == generator beam_radius_m
RESAMPLE_SPACING_M = 2.5e-5
N_FRAMES = 20  # legacy fixed count, still the straight page's frame budget

# /infer_path spaces frames by a TIME STEP rather than a fixed count. A fixed
# count made Δt an accident of the request: a 1.5 mm weld and a 6 mm weld both
# got 20 frames, so the same animation covered very different physical
# durations and two runs could not be compared frame for frame.
#
# The default is calibrated so the reference weld — 1.5 mm at the variable-path
# model's default 0.6 m/s, i.e. 2.5 ms — comes out at exactly 20 frames:
#   round(2.5e-3 / 1.3e-4) + 1 = 20
# 130 µs is also 13x the varpath training dt (1e-5), so rendered frames land
# exactly on the dose grid instead of being rounded onto it.
DEFAULT_FRAME_DT_S = 1.3e-4
# Floor keeps an animation meaningful; ceiling keeps one request from asking
# for thousands of renders (rendering, not inference, dominates the cost).
MIN_FRAMES = 5
MAX_FRAMES = 100
# Fixed-duration models were trained on this one window; the step subdivides it.
FIXED_WINDOW_S = 2.0e-3
# Their window never varies, so Δt was never ambiguous for them — their default
# step is simply their training cadence, which reproduces the historical 20
# frames exactly. Only duration-scaled models had the problem this fixes.
FIXED_FRAME_DT_S = FIXED_WINDOW_S / (N_FRAMES - 1)


def default_frame_dt_s(duration_scaled: bool) -> float:
    return DEFAULT_FRAME_DT_S if duration_scaled else FIXED_FRAME_DT_S


def frame_count_for(total_s: float, frame_dt_s: float, n_available: int | None = None) -> int:
    """How many frames span ``total_s`` at ``frame_dt_s``, endpoints included.

    ``n_available`` caps the count to the samples that actually exist (the
    training-dt dose grid, or a preset's stored frames) — asking for more would
    duplicate frames rather than resolve anything new.
    """
    dt = float(frame_dt_s) if frame_dt_s and frame_dt_s > 0.0 else DEFAULT_FRAME_DT_S
    n = int(round(float(total_s) / dt)) + 1
    n = max(MIN_FRAMES, min(MAX_FRAMES, n))
    if n_available is not None:
        n = min(n, int(n_available))
    return max(2, n)

# Substrate keep-out box: substrate x/y bounds inset by 4 beam radii, the same
# margin the training-path generator enforced (flow3D_scripts laser_path.py).
KEEPOUT_X = (-5.5e-4, 5.5e-4)
KEEPOUT_Y = (-3.0e-4, 3.0e-4)

# Normalization / material constants, verified against both run dirs'
# .hydra/config.yaml (custom.ref_scales, custom.material_params). Also asserted
# against the loaded hydra config at model-load time.
REF_L = 1.0e-4
REF_T = 3000.0
REF_V = 0.1
REF_P = 10.0
# T_SOLIDUS_K / T_LIQUIDUS_K / T_FILL_K live in app.gt (shared with the GT loaders).
# H_norm = (P/sqrt(|V|)) / max-over-split; every random-path case shares
# P=150 W, V=0.8 m/s, so the split max equals the value and H_norm == 1.0.
H_NORM = 1.0

# custom.FNO_input_keys in the hydra configs; the training pipeline appends
# "t_hat" (= time_s * V/L) because time_as_batch=true (config_loading add_t_hat).
BASE_INPUT_KEYS = ["V_scan", "P", "H_norm", "x", "y", "z", "beam_x", "beam_y", "beam_dist", "path_dose"]
# Channels this adapter knows how to build; a checkpoint may use any subset
# (order per its own config). path_dose_recent is the recency (exp-decay) dose.
# Calibrated ("newphys"/spkh123) models add five more. Every one is built by the
# SUBMODULE's own builder (all five are in npz_datasource.PATH_CHANNEL_NAMES), never
# reimplemented here -- a re-derived formula that drifts from training would produce
# confidently wrong physics with no error.
CALIBRATED_CHANNELS = ("preheat_T0", "beam_shape_uniform", "focus_factor",
                       "h_norm_local", "conduction_field")
# NOTE the calibrated five are admitted only when the lpno code in use
# (ext/lpno, or LPFNO_PINO_SRC; see app/submodule.py) can actually build them --
# see `supported_input_keys()`. Advertising a channel that builder does not know
# would fail with a KeyError mid-request; computing the set against the module in
# use turns that into a clean "cannot build" refusal at model load.
SUPPORTED_INPUT_KEYS = set(BASE_INPUT_KEYS) | {"path_dose_recent"}
INPUT_KEYS = BASE_INPUT_KEYS + ["t_hat"]
OUTPUT_KEYS = ["T", "alpha"]




def prepare_path(
    points_xy_m: list[list[float]], max_length_m: float | None = MAX_PATH_LENGTH_M
) -> tuple[np.ndarray, float, bool]:
    """Clamp a drawn polyline into the keep-out box, optionally trim to a max arc
    length, and resample arc-length-uniform at ~25 um spacing.

    ``max_length_m=None`` disables the trim entirely — variable-length models
    trained path length as a free variable, so trimming would defeat the point.
    The keep-out box still bounds where the path may go.

    Returns (resampled_points (N,2), path_length_m, clamped) where ``clamped``
    is True when any point was moved into the box or the path was trimmed.
    """
    raw = np.asarray(points_xy_m, dtype=np.float64).reshape(-1, 2)
    pts = np.column_stack([
        np.clip(raw[:, 0], KEEPOUT_X[0], KEEPOUT_X[1]),
        np.clip(raw[:, 1], KEEPOUT_Y[0], KEEPOUT_Y[1]),
    ])
    clamped = bool(np.any(pts != raw))

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = seg > 0.0  # drop zero-length segments so arc length is strictly increasing
    if not np.any(keep):
        # Degenerate path: all points identical after clamping.
        return np.repeat(pts[:1], 2, axis=0), 0.0, clamped
    dedup = np.concatenate([pts[:1], pts[1:][keep]], axis=0)
    seg = seg[keep]
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])

    length = total if max_length_m is None else min(total, max_length_m)
    trimmed = max_length_m is not None and total > max_length_m
    n = max(2, int(round(length / RESAMPLE_SPACING_M)) + 1)
    s = np.linspace(0.0, length, n)
    out = np.column_stack([np.interp(s, arc, dedup[:, 0]), np.interp(s, arc, dedup[:, 1])])
    return out, length, clamped or trimmed


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def calibrated_channel_builders(ch: dict[str, Any]) -> dict[str, Any]:
    """Builders for the five calibrated channels, shared by BOTH adapters.

    The straight adapter keeps its own `_build_invar` (different grid, different
    physics source), so without one definition these five would be written twice
    and drift -- the exact shape of the 2026-08-10 preheat bug, where the third of
    three near-identical builders was the one that dropped a field.

    preheat_T0 is Kelvin-valued and scaled like T
    (preprocessing_datasets.py:538-544); the other four are already O(1) by
    construction and pass through raw.
    """
    return {
        "preheat_T0": lambda: (ch["preheat_T0"] / REF_T).astype(np.float32),
        "beam_shape_uniform": lambda: ch["beam_shape_uniform"],
        "focus_factor": lambda: ch["focus_factor"],
        "h_norm_local": lambda: ch["h_norm_local"],
        "conduction_field": lambda: ch["conduction_field"],
    }


def _synthetic_manifest(
    points: np.ndarray,
    laser_off_time_s: float,
    *,
    v_scan_mps: float = V_SCAN_M_PER_S,
    p_watts: float = P_WATTS,
) -> dict[str, Any]:
    """Manifest-shaped dict for the submodule path-channel builders.

    scan_speed drives the beam-advance timing (arclength(speed*t)); power_w is
    read by the power-weighted dose builder for variable-power models.
    """
    return {
        "derived": {
            "laser_path_xy_m": points.tolist(),
            "laser_off_time_s": laser_off_time_s,
            # conduction_field treats z <= surface_z_m as metal; without this it
            # would fall back to an upstream default and stamp the source on the
            # wrong plane.
            "surface_z_m": DRAWN_SURFACE_Z_M,
        },
        "input": {
            "laser": {
                "scan_speed_m_per_s": v_scan_mps,
                "power_w": p_watts,
                "beam_radius_m": BEAM_SIGMA_M,
                # Absent in the training manifests, so the builder's own default
                # ("gaussian") is the trained value -- pinned here so a change to
                # that default cannot silently move the served physics.
                "heat_distribution": DRAWN_HEAT_DISTRIBUTION,
            },
            "simulation": {"initial_temperature_k": DRAWN_INITIAL_T_K},
            "_design": {"focus_factor": DRAWN_FOCUS_FACTOR},
        },
    }


@dataclass(slots=True)
class PathCase:
    """A resolved /infer_path request: polyline + timing + optional preset GT."""

    points: np.ndarray  # (N, 2) polyline actually used, metres
    path_length_m: float
    laser_off_time_s: float
    times: np.ndarray  # (n_frames,) rendered frame times, seconds
    clamped: bool
    preset: str | None  # preset case name, or None for drawn paths
    manifest: dict[str, Any]
    # Effective laser physics for this request (defaults = the fixed constants).
    p_watts: float = P_WATTS
    v_scan_mps: float = V_SCAN_M_PER_S
    physics: ModelPhysics = ModelPhysics()
    physics_clamped: bool = False
    # Fine time grid the dose integrals are computed on (duration-scaled models
    # use the training dt; fixed models: times_fine is times). frame_idx maps the
    # rendered frames into times_fine.
    times_fine: np.ndarray | None = None
    frame_idx: np.ndarray | None = None

    @property
    def gt_available(self) -> bool:
        return self.preset is not None


class RandomPathAdapter:
    """Lazy-loading adapter for /infer_path. ``stub=True`` fabricates fields."""

    def __init__(self, settings: Settings, *, stub: bool = False) -> None:
        self.settings = settings
        self.stub = stub
        self._grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._presets: dict[str, list[dict[str, Any]]] = {}  # keyed by presets subdir ("" = root)
        self._models: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._imports: dict[str, Any] | None = None
        # The service GPU lock is released when a request times out while its
        # worker thread keeps running, so a retried first request could load
        # the same model concurrently; this lock serializes lazy loading.
        self._load_lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Submodule access
    # ------------------------------------------------------------------ #
    def _npz_datasource(self) -> Any:
        """numpy-only submodule module; cheap to import, safe in stub mode."""
        submodule.ensure_submodule_on_path()
        from lpno.custom_utils import npz_datasource

        return npz_datasource

    def _ensure_imports(self) -> dict[str, Any]:
        """Heavy (torch/physicsnemo) imports, only for real inference."""
        if self._imports is not None:
            return self._imports
        with self._load_lock:
            return self._ensure_imports_locked()

    def _ensure_imports_locked(self) -> dict[str, Any]:
        if self._imports is not None:
            return self._imports
        submodule.ensure_submodule_on_path()

        import torch
        from omegaconf import OmegaConf

        from lpno.custom_utils.inferencer import FNOGridInferencer
        from lpno.custom_utils.utils import (
            fill_where_mask_smooth_np,
            smooth_liquid_fraction_numpy,
        )
        from lpno.models import register_custom_hydra_inputs
        from physicsnemo.sym.hydra import instantiate_arch
        from physicsnemo.sym.key import Key

        self._imports = {
            "torch": torch,
            "OmegaConf": OmegaConf,
            "FNOGridInferencer": FNOGridInferencer,
            "fill_where_mask_smooth_np": fill_where_mask_smooth_np,
            "smooth_liquid_fraction_numpy": smooth_liquid_fraction_numpy,
            "register_custom_hydra_inputs": register_custom_hydra_inputs,
            "instantiate_arch": instantiate_arch,
            "Key": Key,
        }
        return self._imports

    # ------------------------------------------------------------------ #
    # Presets / grid (numpy-only; works in stub mode)
    # ------------------------------------------------------------------ #
    def _data_dir(self, subdir: str | None = None) -> Path:
        """Preset/ground-truth dir. ``subdir`` (a model's physics.presets_dir) selects
        that model's own test split; None = the legacy shared root. The scan is
        non-recursive, so a per-model subdir stays invisible to the root."""
        d = Path(self.settings.path_data_dir)
        if subdir:
            # meta.json is repo data, but keep it defensive: no traversal escapes.
            safe = Path(subdir).name
            if safe != subdir or safe in ("", ".", ".."):
                raise PathInputError(f"Invalid presets_dir '{subdir}'")
            d = d / safe
        if not d.is_dir():
            raise FileNotFoundError(f"path_data_dir not found: {d}")
        return d

    def _case_files(self, subdir: str | None = None) -> list[Path]:
        root = self._data_dir(subdir)
        files = sorted(f for f in root.glob("*.npz") if not f.name.startswith("var_"))
        if not files:
            raise FileNotFoundError(f"No preset case NPZ files in {root}")
        return files

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The served grid: the axes of the first ground-truth case of the FIRST
        model's own presets directory. It used to read the first NPZ at the
        ground-truth ROOT, which is the legacy fixed-physics set no served model
        uses -- an asset bundle that (rightly) omits it left a fresh install with
        no grid at all (FileNotFoundError), and a root set with a different mesh
        would have silently changed the served grid."""
        if self._grid is None:
            from .model_registry import discover_models

            subdirs = [(m.get("physics") or {}).get("presets_dir") for m in discover_models(self._models_dir())]
            subdirs = [d for d in subdirs if d] + [None]
            last: Exception | None = None
            for sub in subdirs:
                try:
                    with np.load(self._case_files(sub)[0]) as data:
                        self._grid = load_axes(data)
                    break
                except FileNotFoundError as exc:
                    last = exc
            if self._grid is None:
                self._grid = self._page_grid()
            if self._grid is None:
                raise FileNotFoundError(f"no ground-truth case and no grid.npz to take the grid from: {last}")
        return self._grid

    def _page_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The page's served grid from ``data/<page>/grid.npz`` (x, y, z cell-centre axes in metres, the mesh of
        the simulations the models were trained on). Ground truth is optional (it is distributed only in the
        restricted-access record), so an install from the open record takes its grid from this file."""
        gp = Path(self._models_dir()).parent / "grid.npz"
        if not gp.is_file():
            return None
        with np.load(gp) as data:
            return load_axes(data)

    def _default_presets_dir(self) -> str | None:
        """presets_dir of the page's default model (else the first that has one).
        A request without ``model`` gets these, so a fresh install -- which
        ships only the served models' cases, not the legacy root set -- answers
        the same way the page's initial state does."""
        from .model_registry import discover_models

        models = discover_models(self._models_dir())
        ordered = [m for m in models if m.get("default")] + [m for m in models if not m.get("default")]
        for m in ordered:
            sub = ((m.get("physics") or {}).get("presets_dir") or "").strip()
            if sub:
                return sub
        return None

    def presets(self, model: str | None = None) -> list[dict[str, Any]]:
        """Ground-truth preset cases for ``model`` — its own test split when its
        meta.json sets physics.presets_dir; without a model, the default model's.
        Cached per directory. A model is only ever offered presets it was trained
        for."""
        if model:
            self._validate_model(model)          # unknown id -> 422, as on /infer_path
        subdir = self.physics_for(model).presets_dir if model else self._default_presets_dir()
        key = subdir or ""
        if key not in self._presets:
            nds = self._npz_datasource()
            entries = []
            try:
                files = self._case_files(subdir)
            except FileNotFoundError:
                files = []  # no ground truth installed: no presets, freehand paths still work
            for f in files:
                manifest = nds._load_manifest(f)
                if manifest is None:
                    logger.warning("Preset %s has no manifest; skipping", f.stem)
                    continue
                laser = manifest.get("input", {}).get("laser", {})
                entries.append(
                    {
                        "name": f.stem,
                        "npz_path": f,
                        "manifest": manifest,
                        "points_xy_m": [list(map(float, p)) for p in manifest["derived"]["laser_path_xy_m"]],
                        "p_watts": float(laser.get("power_w", P_WATTS)),
                        "v_scan_mps": float(laser.get("scan_speed_m_per_s", V_SCAN_M_PER_S)),
                    }
                )
            self._presets[key] = entries
        return self._presets[key]

    def served_grid_shape(self, model: str) -> tuple[int, int, int]:
        """(nx, ny, nz) of the grid this page feeds `model`: its own grid.npz where the subclass offers one
        (StraightPathAdapter.model_grid), else the page's dataset grid."""
        own = self.model_grid(model) if hasattr(self, "model_grid") else None
        x, y, z = own if own is not None else self.grid()
        return (int(np.asarray(x).size), int(np.asarray(y).size), int(np.asarray(z).size))

    def physics_for(self, model: str) -> ModelPhysics:
        """The model's laser-physics capabilities, parsed once by the registry.

        An unknown model id yields the fixed defaults instead of raising, so the
        later ``_validate_model`` produces the proper 422 (not a 500 here)."""
        return model_registry.physics_for(self._models_dir(), model)

    @staticmethod
    def _effective_pv(
        physics: ModelPhysics, p_watts: float | None, v_scan_mps: float | None
    ) -> tuple[float, float, bool]:
        """Apply requested P/V against the model's capability: fixed models reject
        explicit values (422); variable models clamp to the trained range."""
        clamped = False
        if not physics.variable:
            if p_watts is not None or v_scan_mps is not None:
                raise PathInputError(
                    "This model was trained at fixed laser physics "
                    f"(P={physics.p_default:g} W, V={physics.v_default:g} m/s); "
                    "p_watts/v_scan_mps cannot be set for it."
                )
            return physics.p_default, physics.v_default, False
        p = physics.p_default if p_watts is None else float(p_watts)
        v = physics.v_default if v_scan_mps is None else float(v_scan_mps)
        if physics.p_range is not None:
            p2 = min(max(p, physics.p_range[0]), physics.p_range[1])
            clamped |= p2 != p
            p = p2
        if physics.v_range is not None:
            v2 = min(max(v, physics.v_range[0]), physics.v_range[1])
            clamped |= v2 != v
            v = v2
        return p, v, clamped

    @staticmethod
    def _scaled_times(
        length_m: float, v_scan_mps: float, dt_s: float, frame_dt_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Duration-scaled grids: fine dose grid at the training dt over the real
        weld duration length/speed, plus render frames spaced by ``frame_dt_s``.
        Returns (times_fine, frame_idx)."""
        total = max(length_m / v_scan_mps, 2.0 * dt_s)  # degenerate paths: >=2 steps
        n_fine = int(np.ceil(total / dt_s)) + 1
        times_fine = np.arange(n_fine, dtype=np.float64) * dt_s
        n_render = frame_count_for(total, frame_dt_s, n_available=n_fine)
        frame_idx = np.unique(np.linspace(0, n_fine - 1, n_render).round().astype(int))
        return times_fine, frame_idx

    def resolve(
        self,
        *,
        points_xy_m: list[list[float]] | None,
        preset: str | None,
        physics: ModelPhysics | None = None,
        p_watts: float | None = None,
        v_scan_mps: float | None = None,
        model: str | None = None,
        frame_dt_s: float | None = None,
    ) -> PathCase:
        physics = physics if physics is not None else ModelPhysics()
        # One effective step for every branch below, so a preset, a drawn path
        # and a fixed-window model all mean the same thing by "frame_dt_s".
        dt_frame = (
            float(frame_dt_s) if frame_dt_s and frame_dt_s > 0.0
            else default_frame_dt_s(physics.duration_scaled_times)
        )
        if preset is not None:
            # Presets replay a stored GT case: physics comes from ITS manifest
            # (times from its NPZ), so explicit overrides are rejected.
            if p_watts is not None or v_scan_mps is not None:
                raise PathInputError("p_watts/v_scan_mps cannot be set for a preset (its physics is fixed by the stored case).")
            available = self.presets(model)
            entry = next((p for p in available if p["name"] == preset), None)
            if entry is None:
                raise PathInputError(
                    f"Unknown preset '{preset}'. Available: {[p['name'] for p in available]}"
                )
            manifest = entry["manifest"]
            points = np.asarray(manifest["derived"]["laser_path_xy_m"], dtype=np.float64)
            with np.load(entry["npz_path"]) as data:
                stored = np.asarray(data["time_s"], dtype=np.float64)
            # The stored grid IS the case's real (training-dt) time grid, so it is
            # the fine grid: dose integrals use it whole, and we render frames
            # spaced by frame_dt_s from it. A preset cannot resolve finer than
            # its stored frames, so the step is capped by what was simulated
            # (GT is sliced to match).
            if stored.size < 2:
                raise PathInputError(f"Preset '{preset}' has {stored.size} frames; need >=2")
            span = float(stored[-1] - stored[0])
            n_render = frame_count_for(span, dt_frame, n_available=stored.size)
            frame_idx = np.unique(np.linspace(0, stored.size - 1, n_render).round().astype(int))
            laser = manifest.get("input", {}).get("laser", {})
            return PathCase(
                points=points,
                path_length_m=_polyline_length(points),
                laser_off_time_s=float(manifest["derived"]["laser_off_time_s"]),
                times=stored[frame_idx],
                clamped=False,
                preset=preset,
                manifest=manifest,
                p_watts=float(laser.get("power_w", P_WATTS)),
                v_scan_mps=float(laser.get("scan_speed_m_per_s", V_SCAN_M_PER_S)),
                physics=physics,
                times_fine=stored,
                frame_idx=frame_idx,
            )

        assert points_xy_m is not None
        p_eff, v_eff, pv_clamped = self._effective_pv(physics, p_watts, v_scan_mps)
        points, length, clamped = prepare_path(points_xy_m, max_length_m=physics.max_path_length_m)
        off = length / v_eff
        if physics.duration_scaled_times:
            # Duration-varying models: the time grid spans the REAL weld duration
            # (length/speed) at the training dt — the fixed window truncates long
            # welds and extrapolates short ones (corrected 2026-07-15).
            times_fine, frame_idx = self._scaled_times(length, v_eff, physics.dt_s, dt_frame)
            times = times_fine[frame_idx]
        else:
            # Fixed-duration models (rp3 & friends): trained on one ~2 ms
            # window, so the SPAN stays put whatever the step — only how finely
            # it is sampled follows frame_dt_s. A different span is OOD.
            n_render = frame_count_for(FIXED_WINDOW_S, dt_frame)
            times = np.linspace(0.0, FIXED_WINDOW_S, n_render)
            times_fine, frame_idx = times, np.arange(n_render)
        return PathCase(
            points=points,
            path_length_m=length,
            laser_off_time_s=off,
            times=times,
            clamped=clamped,
            preset=None,
            manifest=_synthetic_manifest(points, off, v_scan_mps=v_eff, p_watts=p_eff),
            p_watts=p_eff,
            v_scan_mps=v_eff,
            physics=physics,
            physics_clamped=pv_clamped,
            times_fine=times_fine,
            frame_idx=frame_idx,
        )


    def beam_centres(self, case: PathCase) -> np.ndarray:
        nds = self._npz_datasource()
        return nds._beam_centres_over_time(case.manifest, case.times)  # (nt, 2)

    def _path_channels(self, case: PathCase, keys: list[str] | None = None) -> dict[str, np.ndarray]:
        """(nt,1,nx,ny,nz) float32 conditioning channels via the submodule builder.

        Cumulative channels (path_dose, path_dose_recent) are time integrals, so
        for duration-scaled cases they are computed on the FINE grid (training dt)
        and sliced to the rendered frames — computing them on 20 coarse frames
        would under-resolve the dose integral. power_weighted follows the model's
        training convention (meta.json physics.power_weighted_dose).
        """
        nds = self._npz_datasource()
        # Per-model grid override (StraightCase.grid_xyz, e.g. the z-cropped
        # rigno_cal); PathCase has no such field -> the shared dataset grid.
        case_grid = getattr(case, "grid_xyz", None)
        x, y, z = case_grid if case_grid is not None else self.grid()
        requested = list(keys) if keys is not None else list(nds.PATH_CHANNEL_NAMES)
        # getattr-duck-typed: StraightPathAdapter routes its StraightCase (no
        # fine-grid/physics fields) through this same builder.
        fine = getattr(case, "times_fine", None)
        if fine is None:
            fine = case.times
        idx = getattr(case, "frame_idx", None)
        if idx is None:
            idx = np.arange(case.times.size)
        physics = getattr(case, "physics", None)
        if physics is None and any(k in ("path_dose", "path_dose_recent") for k in requested):
            logger.warning(
                "building %s with no model physics: dose is UNWEIGHTED (every served model trains "
                "power-weighted) -- pass physics=adapter.physics_for(model) into resolve()",
                [k for k in requested if k.startswith("path_dose")],
            )
        ch = nds._build_path_channels(
            requested, case.manifest, x, y, z, fine,
            power_weighted=bool(physics.power_weighted_dose) if physics is not None else False,
            power_weighted_conduction=bool(physics.power_weighted_conduction) if physics is not None else False,
            conduction_sigma_factor=float(physics.conduction_sigma_factor) if physics is not None else 1.0,
        )
        # drop the case axis and slice the fine grid to the rendered frames
        return {k: v[0][idx] for k, v in ch.items()}  # (n_frames,1,nx,ny,nz)

    # ------------------------------------------------------------------ #
    # Model loading (real runtime)
    # ------------------------------------------------------------------ #
    def _run_dir(self, model: str) -> Path:
        from .model_registry import resolve_model_dir

        return resolve_model_dir(self.settings.path_models_dir, model)

    def supported_input_keys(self) -> set[str]:
        """Channels this adapter can really build with the lpno code in use.

        The calibrated channels are added only if that code's
        ``npz_datasource.PATH_CHANNEL_NAMES`` actually contains them, so the
        contract check refuses a model it cannot serve at load time, before a
        request is in flight. When the lpno code gains channels this set widens
        on its own, with no edit here.
        """
        keys = set(SUPPORTED_INPUT_KEYS)
        try:
            available = set(self._npz_datasource().PATH_CHANNEL_NAMES)
        except Exception:  # noqa: BLE001 -- an import problem must not mask the real error
            return keys
        return keys | (set(CALIBRATED_CHANNELS) & available)

    def release_models(self, keep: str | None = None) -> list[str]:
        """Drop every cached model except ``keep``; return the ids dropped.

        Used by the service's VRAM headroom guard. Safe to call at any time: a
        forward in flight holds its own reference to the entry, and a dropped
        model simply reloads on next use (its neighbour graphs are cached on
        disk, so that is far cheaper than a cold start).
        """
        dropped = [m for m in list(self._models) if m != keep]
        for m in dropped:
            self._models.pop(m, None)
        return dropped

    def _ensure_model(self, model: str) -> dict[str, Any]:
        if model in self._models:
            self._models.move_to_end(model)
            return self._models[model]
        with self._load_lock:
            return self._ensure_model_locked(model)

    def _ensure_model_locked(self, model: str) -> dict[str, Any]:
        if model in self._models:
            self._models.move_to_end(model)
            return self._models[model]

        imports = self._ensure_imports()
        torch = imports["torch"]
        OmegaConf = imports["OmegaConf"]
        Key = imports["Key"]

        run_dir = self._run_dir(model)
        cfg_path = submodule.find_hydra_config_path(run_dir)
        if cfg_path is None:
            raise FileNotFoundError(f"Hydra config not found for run dir {run_dir}")
        imports["register_custom_hydra_inputs"]()
        cfg = OmegaConf.load(cfg_path)

        # Sanity: the hydra config must carry the constants this adapter assumes.
        rs = cfg.custom.ref_scales
        expected = {"L": REF_L, "T": REF_T, "V": REF_V, "P": REF_P}
        for key, val in expected.items():
            if not np.isclose(float(rs[key]), val):
                raise RuntimeError(f"ref_scales.{key}={rs[key]} in {cfg_path} != expected {val}")
        mp = cfg.custom.material_params
        if float(mp["T_s"]) != T_SOLIDUS_K or float(mp["T_l"]) != T_LIQUIDUS_K:
            raise RuntimeError(f"material T_s/T_l in {cfg_path} != expected {T_SOLIDUS_K}/{T_LIQUIDUS_K}")
        # The dose and conduction channels are built from meta.json's physics block; refuse a folder
        # whose block disagrees with the checkpoint's own training settings (it would serve wrong inputs).
        mismatches = physics_config_mismatches(self.physics_for(model), cfg.custom)
        if mismatches:
            raise RuntimeError(f"meta.json physics of model {model!r} disagrees with {cfg_path}: "
                               + "; ".join(mismatches))
        # Model-driven input contract: the checkpoint's own FNO_input_keys decide
        # which channels this adapter must build (any subset/order of the
        # supported set); t_hat is appended by the training pipeline when
        # time_as_batch, so mirror that here.
        cfg_input_keys = [str(k) for k in cfg.custom.FNO_input_keys]
        supported = self.supported_input_keys()
        unsupported = [k for k in cfg_input_keys if k not in supported]
        if unsupported:
            missing_upstream = sorted(set(unsupported) & set(CALIBRATED_CHANNELS))
            hint = (
                f" -- {missing_upstream} are calibrated channels the lpno code in "
                f"ext/lpno cannot build; update it to serve this model"
                if missing_upstream else ""
            )
            raise RuntimeError(
                f"FNO_input_keys in {cfg_path} contain channels this adapter cannot "
                f"build: {unsupported}; supported: {sorted(supported)}{hint}"
            )
        if not bool(cfg.custom.time_as_batch):
            raise RuntimeError(f"Expected custom.time_as_batch=true in {cfg_path} (adds the t_hat input)")
        model_input_keys = cfg_input_keys + ["t_hat"]

        input_keys = [Key(k) for k in model_input_keys]
        output_keys = [Key(k) for k in OUTPUT_KEYS]
        # Pick the architecture from the checkpoint's own hydra config, not the
        # model id, so any discovered folder loads by what it actually is (see
        # app/arch_build.py for why custom.model_family wins over the arch blocks).
        if is_fno_config(cfg):
            # Grid 75x50x40 with modes (20,18,8) is below every Nyquist cap, so no
            # fno_modes adjustment is needed (unlike the super-resolved /infer path).
            instantiate_arch = imports["instantiate_arch"]
            decoder = instantiate_arch(cfg=cfg.arch.decoder, output_keys=output_keys)
            net = instantiate_arch(
                cfg=cfg.arch.fno, decoder_net=decoder, input_keys=input_keys, output_keys=output_keys
            )
        elif is_gaot(cfg):
            from lpno.models.gaot_arch import GAOTArch

            net = GAOTArch(**build_gaot_kwargs(
                cfg=cfg, cache_dir=submodule.graph_cache_dir(self.settings),
                input_keys=input_keys, output_keys=output_keys,
            ))
        else:
            net = build_rigno_net(
                cfg=cfg, cache_dir=submodule.graph_cache_dir(self.settings),
                input_keys=input_keys, output_keys=output_keys,
            )

        submodule.load_checkpoint_strict(net, run_dir / "u_network.0.pth", torch)
        if is_fno_config(cfg):
            # An FNO trained on a strided grid and served on a finer one needs the evaluation's repair (lpno
            # fno_superres: padding rescaled, spectral weights remapped onto the frequencies the trained grid
            # held; it reads trained_grid.json beside the checkpoint, and does nothing on the trained grid).
            # Without it the straight page's uni_fno_lab_b8_32k_s42 read 254 K MAE on a test frame its
            # tabled evaluation scored at 39 K.
            from lpno.models.pipeline.fno_superres import maybe_apply_fno_superres

            maybe_apply_fno_superres(net, run_dir, self.served_grid_shape(model))

        from .gpu import pick_device

        device = pick_device(torch, self.settings.cuda_device)
        net.to(device)
        net.eval()

        results_dir = Path(self.settings.runtime_results_dir) / "infer_path_vti"
        infer = imports["FNOGridInferencer"](
            model=net,
            input_keys=model_input_keys,
            output_names=OUTPUT_KEYS,
            results_dir=results_dir,
            device=str(device),
            requires_grad=False,
            mask_fn=None,
            log_iter=False,
        )
        # Frames are independent samples (time_as_batch), so batching them only
        # changes GPU memory use, not results; the hydra configs pin batch 1.
        batch_size = max(1, int(self.settings.path_infer_batch))
        entry = {
            "infer": infer,
            "batch_size": batch_size,
            "device": str(device),
            "input_keys": model_input_keys,
        }
        self._models[model] = entry
        self._models.move_to_end(model)
        logger.info("Loaded random-path model '%s' from %s on %s", model, run_dir, device)
        evicted = evict_to_cap(self._models, self.settings.model_cache_entries)
        if evicted:
            logger.info("Evicted model(s) %s to cap resident GPU models", evicted)
            from .gpu import empty_cache

            empty_cache()
        return entry

    # ------------------------------------------------------------------ #
    # Inference + postprocess
    # ------------------------------------------------------------------ #
    def _build_invar(self, case: PathCase, input_keys: list[str] | None = None) -> dict[str, np.ndarray]:
        x, y, z = self.grid()
        nx, ny, nz = x.size, y.size, z.size
        nt = case.times.size
        shape = (nt, 1, nx, ny, nz)
        keys = list(input_keys) if input_keys is not None else INPUT_KEYS
        nds = self._npz_datasource()
        path_keys = [k for k in keys if k in set(nds.PATH_CHANNEL_NAMES)]
        ch = self._path_channels(case, keys=path_keys)

        # Effective physics: per-request for variable models, the historical
        # constants otherwise. H_norm follows the training convention
        # (raw P[W]/sqrt(V[m/s]) / train-split max); a single-P/V split's max is
        # its own value, so fixed models keep H_norm == 1.0.
        denom = case.physics.h_norm_denom
        h_norm = (case.p_watts / np.sqrt(case.v_scan_mps)) / denom if denom else H_NORM

        builders = {
            "V_scan": lambda: np.full(shape, case.v_scan_mps / REF_V, dtype=np.float32),
            "P": lambda: np.full(shape, case.p_watts / REF_P, dtype=np.float32),
            "H_norm": lambda: np.full(shape, h_norm, dtype=np.float32),
            "x": lambda: np.broadcast_to((x / REF_L).reshape(1, 1, nx, 1, 1), shape).astype(np.float32),
            "y": lambda: np.broadcast_to((y / REF_L).reshape(1, 1, 1, ny, 1), shape).astype(np.float32),
            "z": lambda: np.broadcast_to((z / REF_L).reshape(1, 1, 1, 1, nz), shape).astype(np.float32),
            "beam_x": lambda: (ch["beam_x"] / REF_L).astype(np.float32),
            "beam_y": lambda: (ch["beam_y"] / REF_L).astype(np.float32),
            "beam_dist": lambda: ch["beam_dist"],
            "path_dose": lambda: ch["path_dose"],
            "path_dose_recent": lambda: ch["path_dose_recent"],
            # preheat_T0 is Kelvin-valued and MUST be scaled like T
            # (preprocessing_datasets.py:538-544). The other four are left raw:
            # a 0/1 flag, an O(1) design factor, a ratio that is 1.0 at ambient,
            # and a field the builder already normalises to [0, 1].
            **calibrated_channel_builders(ch),
            # t_hat = time_s * (V_ref / L_ref), one value per frame (time_as_batch)
            "t_hat": lambda: np.broadcast_to(
                (case.times * (REF_V / REF_L)).astype(np.float32).reshape(nt, 1, 1, 1, 1), shape
            ).astype(np.float32),
        }
        return {k: builders[k]() for k in keys}

    def _models_dir(self) -> str:
        return self.settings.path_models_dir

    def _validate_model(self, model: str) -> None:
        """Reject a model id that has no folder under the page's models dir, in
        both stub and real mode, with a 422 (PathInputError) instead of a later
        FileNotFoundError/500 (or a silent stub success)."""
        ids = {m["id"] for m in model_registry.discover_models(self._models_dir())}
        if model not in ids:
            raise PathInputError(f"Unknown model '{model}'. Available: {sorted(ids) or ['(none)']}")

    def infer(self, model: str, case: PathCase) -> dict[str, np.ndarray]:
        """Return postprocessed fields, each (nt, nx, ny, nz): T in Kelvin, fl/alpha in 0..1."""
        self._validate_model(model)
        if self.stub:
            return self._stub_infer(case)

        imports = self._ensure_imports()
        entry = self._ensure_model(model)
        invar = self._build_invar(case, input_keys=entry["input_keys"])
        preds = entry["infer"].run(
            invar, name=f"path_{model}", step=0, save_filetypes=(), batch_size=entry["batch_size"]
        )

        # Postprocess exactly like eval (run_inference_and_metrics), in normalized units:
        # fl from unmasked T, then mask T (fill to boiling), alpha, then fl with the
        # updated alpha — same key order as the eval's ds_output dict (T, alpha, fl).
        smooth_fl = imports["smooth_liquid_fraction_numpy"]
        fill = imports["fill_where_mask_smooth_np"]
        t_s, t_l = T_SOLIDUS_K, T_LIQUIDUS_K
        preds["fl"] = smooth_fl(preds["T"], t_s / REF_T, t_l / REF_T)
        for key in ("T", "alpha", "fl"):
            d = T_FILL_K / REF_T if key == "T" else 0.0
            preds[key] = fill(preds[key], preds["alpha"], threshold=0.5, d=d, k=20, inplace=False)

        return {
            "T": np.asarray(preds["T"][:, 0]) * REF_T,
            "alpha": np.asarray(preds["alpha"][:, 0]),
            "fl": np.asarray(preds["fl"][:, 0]),
        }

    def _stub_infer(self, case: PathCase) -> dict[str, np.ndarray]:
        """Fabricated moving-Gaussian T field along the resampled path (no GPU)."""
        # Same per-model grid override as _path_channels: the channels are built
        # on the case's grid, so taking z from the shared dataset grid here
        # would broadcast a mismatched depth against them.
        case_grid = getattr(case, "grid_xyz", None)
        x, y, z = case_grid if case_grid is not None else self.grid()
        ch = self._path_channels(case)  # real path-channel code, numpy-only
        beam = ch["beam_dist"][:, 0]  # (nt, nx, ny, nz), constant over z
        dose = ch["path_dose"][:, 0]
        depth = np.exp(-np.abs(z) / 1.0e-4).reshape(1, 1, 1, -1).astype(np.float32)
        t_field = 300.0 + (2600.0 * beam + 500.0 * dose) * depth
        alpha = np.broadcast_to((z < 0.0).astype(np.float32).reshape(1, 1, 1, -1), t_field.shape).copy()
        phi = (t_field - T_SOLIDUS_K) / (T_LIQUIDUS_K - T_SOLIDUS_K)
        fl = (0.5 * (np.tanh(2.5 * (phi - 0.5)) + 1.0) * alpha).astype(np.float32)
        return {"T": t_field.astype(np.float32), "alpha": alpha, "fl": fl}

    # ------------------------------------------------------------------ #
    # Ground truth for presets (computed from the raw test NPZ; verified
    # bit-identical to the eval pipeline's ground_truth.npz for T and alpha)
    # ------------------------------------------------------------------ #
    def ground_truth(self, case: PathCase, model: str | None = None) -> dict[str, np.ndarray] | None:
        if case.preset is None:
            return None
        entry = next(p for p in self.presets(model) if p["name"] == case.preset)
        # Slice the stored frames to the rendered ones so GT lines up with the
        # prediction frame-for-frame (variable-path cases store 22..396 frames but
        # render <=20; legacy 20-frame sets slice to themselves).
        idx = case.frame_idx if case.frame_idx is not None else slice(None)
        with np.load(entry["npz_path"]) as data:
            t_raw = np.asarray(data["Temperature"][idx], dtype=np.float32)[:, None]  # (nt,1,nx,ny,nz)
            a_raw = np.asarray(data["Fraction Of Fluid"][idx], dtype=np.float32)[:, None]

        # Same gate math as fill_where_mask_smooth_np(k=20) without importing torch-heavy utils.
        return {k: v[:, 0] for k, v in mask_fields(t_raw, a_raw).items()}

