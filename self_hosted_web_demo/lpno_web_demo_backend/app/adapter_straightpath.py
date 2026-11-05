"""Time-dependent straight-track inference adapter for /infer_straight.

A straight case is a laser moving along +x at y=0 across the substrate at a
user-chosen power P and scan speed V. The served checkpoints are the paper's
calibrated laboratory-frame models (16 conditioning channels, time_as_batch),
trained on ti64_straight_path_calibrated_with_keyholes: grid 90x40x40 (the
RIGNO on its 90x40x30 training crop via grid.npz), per-case P/V, the H_norm
denominator fitted on the train split (meta.json physics.h_norm_denom). The
input-channel contract is the random-path one, so this adapter subclasses
``RandomPathAdapter`` and only overrides the grid, run dirs, P/V-driven invar,
ground truth, and the P-V ground-truth catalog.

The track geometry is uniform across every case: start (-0.300 mm, 0), end
(+0.29724 mm, 0), length 0.59724 mm, total sim time = length / V (laser on
throughout). Ground truth is served for a requested (P, V) only when it matches
a catalog case (the frontend's clickable catalog points send the exact case
P/V); free P/V predicts without GT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapter_randompath import (
    BEAM_SIGMA_M,
    N_FRAMES,
    REF_L,
    REF_P,
    REF_V,
    PathInputError,
    RandomPathAdapter,
    _polyline_length,
)
from .config import Settings
from .gt import load_axes, mask_fields, nearest_pv

logger = logging.getLogger(__name__)

# Fixed straight-track geometry (identical for every resampled_straight_path_data case).
STRAIGHT_START_XY_M = (-3.0e-4, 0.0)
STRAIGHT_END_XY_M = (2.9724e-4, 0.0)
STRAIGHT_LENGTH_M = STRAIGHT_END_XY_M[0] - STRAIGHT_START_XY_M[0]  # 0.59724 mm

# H_norm reference for the LEGACY straight models (custom.ref_scales.H in the
# hydra configs). The random-path models pinned H_norm==1.0 because P/V were
# fixed; here P/V vary, so H_norm = (P/sqrt(|V|)) / H_REF is a real input.
#
# This is a fallback, not the general answer, and the calibrated models do NOT
# use it. `ref_scales.H` is only the denominator when `custom.h_norm_mode` is
# "ref_scale"; absent that (every straight config, legacy and calibrated), the
# pipeline FITS the denominator as the train split's max of P/sqrt(|V|)
# (lazy_runtime.compute_h_norm_denom) and BUG-04 reuses that same value at eval.
# For ti64_straight_path_calibrated_with_keyholes that max is 275.0539301382
# (case kh40_014, P=215 W, V=0.611 m/s), which is what the calibrated metas
# carry in physics.h_norm_denom. Serving those models at 203 would feed every
# H_norm 35 % high with no error anywhere.
STRAIGHT_H_REF = 203.0

# Physics the 16-channel calibrated builders read out of the case manifest.
# Measured across all 123 cases of ti64_straight_path_calibrated_with_keyholes
# (2026-09-03): every one carries these same values.
#
# 273.15 K is the one that matters and the one that is easy to get wrong: the
# random-path page pins 293.15, and 273.15 IS npz_datasource._AMBIENT_T_K, so
# here the soak field short-circuits to a uniform constant instead of running the
# diffusion profile. That is what the training data did; pinning 293.15 would
# build a preheat_T0 channel with no error and the wrong physics.
STRAIGHT_INITIAL_T_K = 273.15
STRAIGHT_HEAT_DISTRIBUTION = "gaussian"   # -> beam_shape_uniform 0.0
STRAIGHT_FOCUS_FACTOR = 1.0               # absent in the manifests = the builder default
STRAIGHT_SURFACE_Z_M = 0.0                # conduction_field treats z <= this as metal

# Export cadence of ti64_straight_path_calibrated_with_keyholes: the stored time
# axis steps 4.05-5.19 us (median 5.004 us, measured on kh40_000). The dose and
# conduction channels are TIME INTEGRALS; training built them on that axis, so
# serving must integrate on a grid at least this fine and only then pick the 20
# rendered frames out of it. Integrating on the 20 frames themselves (dt up to
# 250 us) put path_dose 2x off and grew the error monotonically along the track
# (kh40_000: 14 -> 121 K substrate over the sweep, 2026-09-14).
STRAIGHT_FRAME_DT_S = 5.0e-6


@dataclass(slots=True)
class StraightCase:
    """A resolved /infer_straight request: straight polyline + timing + P/V."""

    points: np.ndarray  # (2, 2) polyline used, metres
    p_watts: float
    v_scan_mps: float
    path_length_m: float
    laser_off_time_s: float
    times: np.ndarray  # (20,) frame times, seconds
    clamped: bool
    manifest: dict[str, Any]
    gt_name: str | None  # matched resampled_straight_path_data case name, or None
    gt_npz: Path | None
    split: str | None  # train | val | test of the matched case, or None
    # H_norm denominator of the serving model (meta.json physics.h_norm_denom;
    # the legacy straight checkpoints share STRAIGHT_H_REF, the calibrated ones
    # carry their own train-split max).
    h_ref: float = STRAIGHT_H_REF
    # Per-model grid override (folder grid.npz); None = the legacy dataset grid.
    grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    # The serving model's physics block (meta.json): power_weighted_dose decides
    # how path_dose is built. None would silently serve UNWEIGHTED dose to a
    # model trained on weighted dose (every lab model), which is what happened
    # before this field existed.
    physics: Any = None
    # Fine time grid the cumulative channels are integrated on, and the index of
    # each rendered frame in it (mirrors PathCase.times_fine / frame_idx).
    times_fine: np.ndarray | None = None
    frame_idx: np.ndarray | None = None

    @property
    def gt_available(self) -> bool:
        return self.gt_npz is not None


def _straight_manifest(
    points: np.ndarray,
    laser_off_time_s: float,
    v_scan: float,
    p_watts: float | None = None,
) -> dict[str, Any]:
    """Manifest-shaped dict for the submodule path-channel builders (per-case P/V).

    Carries the calibrated dataset's physics constants, which the 16-channel
    models read: preheat_T0 <- initial_temperature_k, beam_shape_uniform <-
    heat_distribution, focus_factor <- _design.focus_factor, conduction_field <-
    surface_z_m. The 10-channel legacy models read none of them, so one builder
    serves both. Used for free (P, V); a request that matched a real case uses
    that case's OWN manifest instead -- see `_gt_manifest`.
    """
    return {
        "derived": {
            "laser_path_xy_m": points.tolist(),
            "laser_off_time_s": laser_off_time_s,
            "surface_z_m": STRAIGHT_SURFACE_Z_M,
        },
        "input": {
            "laser": {
                "scan_speed_m_per_s": v_scan,
                "power_w": p_watts,
                "beam_radius_m": BEAM_SIGMA_M,
                "heat_distribution": STRAIGHT_HEAT_DISTRIBUTION,
            },
            "simulation": {"initial_temperature_k": STRAIGHT_INITIAL_T_K},
            "_design": {"focus_factor": STRAIGHT_FOCUS_FACTOR},
        },
    }


def _gt_manifest(npz_path: Path) -> dict[str, Any] | None:
    """The case's own FLOW-3D manifest from `<case>.manifest.json`, if shipped.

    Preferred over `_straight_manifest` for a matched case: it is the same dict
    the training pipeline read, so every conditioning channel is built from the
    case's real physics rather than from a constant that can go stale when the
    served dataset changes. The legacy ground truth ships no sidecars, so absent
    is normal and falls back.
    """
    side = npz_path.with_suffix("").with_suffix(".manifest.json")
    if not side.is_file():
        side = npz_path.parent / f"{npz_path.stem}.manifest.json"
    if not side.is_file():
        return None
    try:
        import json

        man = json.loads(side.read_text())
    except (OSError, ValueError):
        logger.warning("Unreadable GT manifest %s; using the pinned constants", side)
        return None
    return man if isinstance(man, dict) else None



def _fine_time_grid(times: np.ndarray, laser_off_time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(times_fine, frame_idx): a STRAIGHT_FRAME_DT_S grid over the scan and where
    each rendered frame sits in it (nearest fine sample)."""
    off = float(laser_off_time_s)
    fine = np.arange(0.0, off, STRAIGHT_FRAME_DT_S)
    fine = np.append(fine, off) if fine.size == 0 or fine[-1] < off else fine   # end exactly at laser-off:
    if fine.size < 2:                                                            # never a sample past it
        fine = np.array([0.0, off])
    idx = np.abs(fine[:, None] - np.asarray(times, dtype=np.float64)[None, :]).argmin(axis=0)
    return fine, idx

class StraightPathAdapter(RandomPathAdapter):
    """Lazy-loading adapter for /infer_straight (free P/V, straight track)."""

    def __init__(self, settings: Settings, *, stub: bool = False) -> None:
        super().__init__(settings, stub=stub)
        self._catalog: list[dict[str, Any]] | None = None
        self._model_grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}

    # -- run dirs / grid --------------------------------------------------- #
    def _models_dir(self) -> str:
        return self.settings.straight_models_dir

    def _run_dir(self, model: str) -> Path:
        from .model_registry import resolve_model_dir

        return resolve_model_dir(self.settings.straight_models_dir, model)

    def _data_root(self) -> Path:
        d = Path(self.settings.straight_data_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"straight_data_dir not found: {d}")
        return d

    def _all_case_files(self) -> list[tuple[str, Path]]:
        """(split, npz_path) for every straight case across train/val/test."""
        out: list[tuple[str, Path]] = []
        for split in ("train", "val", "test"):
            d = self._data_root() / split
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.npz")):
                if not f.name.startswith("var_"):
                    out.append((split, f))
        if not out:
            raise FileNotFoundError(f"No straight case NPZ files under {self._data_root()}")
        return out

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._grid is None:
            try:
                _split, f0 = self._all_case_files()[0]
                with np.load(f0) as data:
                    self._grid = load_axes(data)
            except FileNotFoundError:
                # no ground truth installed: the page's grid.npz (see _page_grid)
                self._grid = self._page_grid()
                if self._grid is None:
                    raise
        return self._grid

    # -- ground-truth catalog (P-V points, colored by split client-side) --- #
    def gt_catalog(self) -> list[dict[str, Any]]:
        """One entry per straight-path ground-truth case: name, P, V, split, npz path. Empty when no ground
        truth is installed, so every request is then served as a free P-V prediction."""
        if self._catalog is None:
            entries: list[dict[str, Any]] = []
            try:
                files = self._all_case_files()
            except FileNotFoundError:
                files = []
            for split, f in files:
                try:
                    with np.load(f) as data:
                        p = float(data["power_w"])
                        v = float(data["scan_speed_m_per_s"])
                except Exception:  # noqa: BLE001 - a malformed case must not sink the catalog
                    logger.warning("Straight case %s missing power/speed; skipping", f.stem)
                    continue
                entries.append(
                    {
                        "name": f.stem,
                        "p_watts": p,
                        "v_scan_mps": v,
                        "split": split,
                        "npz_path": f,
                    }
                )
            self._catalog = entries
        return self._catalog

    def _match_gt(self, p_watts: float, v_scan_mps: float) -> dict[str, Any] | None:
        return nearest_pv(self.gt_catalog(), p_watts, v_scan_mps)

    def model_grid(self, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Optional per-model grid from ``<model folder>/grid.npz`` (x, y, z
        arrays in metres). A checkpoint trained with `crop_z_top` (rigno_cal,
        90x40x30) serves fewer z planes than the dataset stores, so a folder can
        pin its own grid; absent -> None = the shared dataset grid. Cached per model."""
        if model not in self._model_grids:
            grid = None
            try:
                gp = self._run_dir(model) / "grid.npz"
                if gp.is_file():
                    with np.load(gp) as data:
                        grid = load_axes(data)
            except FileNotFoundError:
                grid = None
            self._model_grids[model] = grid
        return self._model_grids[model]

    # -- resolve a request into a case ------------------------------------- #
    def resolve(
        self,
        *,
        p_watts: float,
        v_scan_mps: float,
        h_ref: float | None = None,
        grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        physics: Any = None,
    ) -> StraightCase:
        p = float(p_watts)
        v = float(v_scan_mps)
        if not np.isfinite(p) or not np.isfinite(v) or v <= 0.0:
            raise PathInputError("p_watts and v_scan_mps must be finite and v_scan_mps > 0")

        match = self._match_gt(p, v)
        if match is not None:
            with np.load(match["npz_path"]) as data:
                times = np.asarray(data["time_s"], dtype=np.float64)
                p = float(data["power_w"])
                v = float(data["scan_speed_m_per_s"])
            if times.size != N_FRAMES:
                raise PathInputError(
                    f"Straight GT case '{match['name']}' has {times.size} frames; expected {N_FRAMES}"
                )
            pts = np.array([STRAIGHT_START_XY_M, STRAIGHT_END_XY_M], dtype=np.float64)
            off = float(times[-1])
            fine, fidx = _fine_time_grid(times, off)
            return StraightCase(
                points=pts,
                p_watts=p,
                v_scan_mps=v,
                path_length_m=_polyline_length(pts),
                laser_off_time_s=off,
                times=times,
                clamped=False,
                manifest=_gt_manifest(match["npz_path"]) or _straight_manifest(pts, off, v, p),
                gt_name=match["name"],
                gt_npz=match["npz_path"],
                split=match["split"],
                h_ref=h_ref if h_ref else STRAIGHT_H_REF,
                grid_xyz=grid_xyz,
                physics=physics,
                times_fine=fine,
                frame_idx=fidx,
            )

        pts = np.array([STRAIGHT_START_XY_M, STRAIGHT_END_XY_M], dtype=np.float64)
        total = STRAIGHT_LENGTH_M / max(v, 1e-9)
        times = np.linspace(0.0, total, N_FRAMES)
        fine, fidx = _fine_time_grid(times, total)
        return StraightCase(
            points=pts,
            p_watts=p,
            v_scan_mps=v,
            path_length_m=STRAIGHT_LENGTH_M,
            laser_off_time_s=total,
            times=times,
            clamped=False,
            manifest=_straight_manifest(pts, total, v, p),
            gt_name=None,
            gt_npz=None,
            split=None,
            h_ref=h_ref if h_ref else STRAIGHT_H_REF,
            grid_xyz=grid_xyz,
            physics=physics,
            times_fine=fine,
            frame_idx=fidx,
        )

    # -- model inputs (real P/V/H_norm on the straight grid) --------------- #
    def _build_invar(
        self, case: StraightCase, input_keys: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Model-driven like the parent: build exactly the checkpoint's own
        input channels (the parent's infer() passes entry["input_keys"]), with
        the straight page's H_ref and per-case P/V. Regression note: this
        override once took only (case,), so every real straight inference
        TypeError'd after the parent went model-driven."""
        from .adapter_randompath import INPUT_KEYS, calibrated_channel_builders

        x, y, z = case.grid_xyz if case.grid_xyz is not None else self.grid()
        nx, ny, nz = x.size, y.size, z.size
        nt = case.times.size
        shape = (nt, 1, nx, ny, nz)
        keys = list(input_keys) if input_keys is not None else INPUT_KEYS
        nds = self._npz_datasource()
        path_keys = [k for k in keys if k in set(nds.PATH_CHANNEL_NAMES)]
        ch = self._path_channels(case, keys=path_keys)

        h_norm = (case.p_watts / (abs(case.v_scan_mps) ** 0.5)) / case.h_ref
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
            "t_hat": lambda: np.broadcast_to(
                (case.times * (REF_V / REF_L)).astype(np.float32).reshape(nt, 1, 1, 1, 1), shape
            ).astype(np.float32),
            **calibrated_channel_builders(ch),
        }
        return {k: builders[k]() for k in keys}

    # -- ground truth from the matched resampled_straight_path_data case ----------------- #
    def ground_truth(self, case: StraightCase) -> dict[str, np.ndarray] | None:
        if case.gt_npz is None:
            return None
        with np.load(case.gt_npz) as data:
            t_raw = np.asarray(data["Temperature"], dtype=np.float32)[:, None]  # (nt,1,nx,ny,nz)
            a_raw = np.asarray(data["Fraction Of Fluid"], dtype=np.float32)[:, None]

        # A model trained with `crop_z_top` serves fewer z planes than the ground
        # truth stores, and the renderer indexes prediction and GT with ONE z
        # index -- so an uncropped GT would put the two panels on different
        # planes and score the difference as model error. The crop drops the
        # HIGH-index (gas-side) planes, matching npz_lazy_store's `[..., :zk]`,
        # so the kept planes are the leading ones and no interpolation is needed.
        if case.grid_xyz is not None:
            nz_model = int(case.grid_xyz[2].size)
            nz_gt = t_raw.shape[-1]
            if nz_model < nz_gt:
                t_raw = t_raw[..., :nz_model]
                a_raw = a_raw[..., :nz_model]
            elif nz_model > nz_gt:
                logger.warning(
                    "model grid has %d z planes but GT has %d; serving prediction-only",
                    nz_model, nz_gt,
                )
                return None

        # Same eval masking as the random-path GT (fill_where_mask_smooth_np k=20).
        return {k: v[:, 0] for k, v in mask_fields(t_raw, a_raw).items()}
