from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json
import logging
import re
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pyvista as pv
import torch
from scipy.spatial import cKDTree


try:
    from physicsnemo_curator.etl.data_sources import DataSource  # pyright: ignore[reportMissingImports]
except ImportError:  # physicsnemo-curator is not published on PyPI
    # Base class only; see `dataset_loading.py` for the same fallback and why.
    class DataSource:  # type: ignore[no-redef]
        """Minimal stand-in for the curator base class."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D107
            pass


# Per-file/-timestep/-field diagnostics go through logging at DEBUG so they stay
# silent by default (they once flooded a CI job past its 4 MB log cap). Enable
# with logging.getLogger("...openFoam_datasource").setLevel(logging.DEBUG).
logger = logging.getLogger(__name__)


def _safe_unlink(path: Path) -> None:
    """Remove a stale cache file if the data dir allows it. A read-only data dir
    (e.g. a shared dataset in CI) must not abort the parse — the stale file is
    just left in place and the result is recomputed."""
    try:
        path.unlink()
    except OSError as exc:
        logger.debug(f"[VTK] Could not remove stale cache '{path}' ({exc}); rebuilding without deleting it.")


class GridResolutionMismatchError(ValueError):
    """Source data forms a uniform grid whose resolution differs from the
    configured ``grid_params`` resolution.

    Loading it would silently KNN-interpolate real grid data onto the wrong size
    (e.g. a case exported at a different resolution than the rest of a sweep, or a
    truncated/misexported case), poisoning training without a trace. Fix the case
    export or the ``grid_params`` resolution, or opt into the resample with
    ``allow_grid_interpolation=True`` (config ``custom.allow_grid_interpolation``).
    """


def crop_shift_moving_frame(data: np.ndarray, Nsteps: int = 30, super_res_factor: int = 1) -> np.ndarray:
    """
    data: ndarray shaped (B, T, C, nX, nY, nZ)
    Nsteps: number of final time steps to keep/shift

    Returns: ndarray shaped (B, Nsteps, C, nX - Nsteps, nY, nZ)
    """
    if data.ndim != 6:
        raise ValueError("data must be 6D (B, T, C, nX, nY, nZ)")
    B, T, C, nX, nY, nZ = data.shape
    if not 1 <= Nsteps <= T:
        raise ValueError("Nsteps must be in [1, T]")
    new_nX = nX - (Nsteps-1) * super_res_factor
    if new_nX <= 0:
        raise ValueError("Nsteps must be smaller than nX to keep a positive x-length")

    # Crop to the last Nsteps in time
    cropped_t = data[:, T - Nsteps :, ...]  # shape (B, Nsteps, C, nX, nY, nZ)

    out = np.empty((B, Nsteps, C, new_nX, nY, nZ), dtype=data.dtype)
    for t in range(Nsteps):
        # shift -x by t (take slice starting at x=2*t)
        out[:, t] = cropped_t[:, t, :, super_res_factor*t : super_res_factor*t + new_nX, :, :]
    return out


def moving_average_time(data: np.ndarray, window: int, axis: int = 1) -> np.ndarray:
    """
    Time-only sliding mean.

    Input shape: (B, T, C, X, Y, Z) by default (axis=1).
    Output shape: (B, T - window + 1, C, X, Y, Z)
    """
    arr = np.asarray(data)
    T = arr.shape[axis]
    if not (1 <= window <= T):
        raise ValueError(f"window must be in [1, {T}]")
    view = sliding_window_view(arr, window_shape=window, axis=axis)
    return view.mean(axis=-1, dtype=np.float64)

class VTKOpenFOAMDataSource(DataSource):
    """
    OpenFOAM VTK/VTU datasource (cell arrays) -> invar/outvar with (B,C,X,Y,Z).

    Filename convention (example):
      results_V_scan_0p01_laser_power_5_395.vtk
        V_scan      -> 0.01 (m/s)  [0p01 => 0.01]
        laser_power -> 5    (W)
        trailing '395' => time_s = 395 * 1e-7 = 3.95e-5 s
      Other params like 'beam_radius_*' are also parsed if present.

    By default selects the *latest* time among files sharing the same base params.
    If cfg.target_time_s (or cfg.time_s) is provided, will select the file whose
    encoded time is closest to that target.
    """

    def __init__(
        self,
        cfg: Any = None,
        input_dir: str = "tutorial_data",
        target_time_s: Optional[float] = None,
        time_s: Optional[float] = None,
        time_axis: bool = False,
        avg_window_size: int = 1,
        moving_frame_steps: Optional[int] = None,
        moving_ref_frame: bool = True,
    ):
        """
        cfg is optional; retained for backward compatibility with DataSource interface.
        """
        super().__init__(cfg)
        if cfg is not None:
            input_dir = getattr(cfg, "input_dir", input_dir)
            if target_time_s is None:
                target_time_s = getattr(cfg, "target_time_s", None)
            if time_s is None:
                time_s = getattr(cfg, "time_s", None)
            time_axis = getattr(cfg, "time_axis", time_axis)
            avg_window_size = getattr(cfg, "avg_window_size", avg_window_size)
            moving_frame_steps = getattr(cfg, "moving_frame_steps", moving_frame_steps)
            moving_ref_frame = getattr(cfg, "moving_ref_frame", moving_ref_frame)
        self.input_dir = Path(input_dir)
        self.target_time_s = target_time_s if target_time_s is not None else time_s
        self.time_axis = time_axis
        self.avg_window_size = int(avg_window_size)
        self.moving_frame_steps = moving_frame_steps
        self.moving_ref_frame = bool(moving_ref_frame)

    # --------------------------- #
    # DataSource interface methods
    # --------------------------- #

    def get_file_list(self) -> List[str]:
        """
        Return stems selected by time:
        - group files by 'base without trailing time'
        - choose latest if no target time, else closest to target_time_s
        """
        paths = sorted(list(self.input_dir.glob("*.vtu")) + list(self.input_dir.glob("*.vtk")))
        groups: Dict[str, List[Path]] = {}
        for p in paths:
            groups.setdefault(self._base_without_time(p.stem), []).append(p)

        selected: List[str] = []
        for base, plist in groups.items():
            chosen = self._choose_by_time(plist, self.target_time_s)
            selected.append(chosen.stem)
        logger.debug(f"[VTK] Selected {len(selected)} file(s) after time filtering")
        return selected

    def read_file(self, filename: str | Path) -> Dict[str, Any]:
        """Read one VTK/VTU file and return raw cell-centers + cell fields (no resampling)."""
        path = self._resolve_file_path(filename)
        ds = pv.read(path)  # reads VTK/VTU using VTK’s readers :contentReference[oaicite:3]{index=3}

        # cell centers in the dataset’s cell order :contentReference[oaicite:4]{index=4}
        centers = ds.cell_centers().points.astype(np.float32)

        fields = self._collect_cell_fields_pyvista(ds)  # dict name -> (N,) or (N,C)
        for fname, arr in fields.items():
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"[VTK] Field '{fname}' in '{path.name}' contains non-finite values "
                    f"(NaN/inf); refusing to ingest them into the training tensors."
                )

        params = self._parse_params(Path(filename).stem)
        self._require_scan_parameters(params, path.name)
        meta = {
            "filename": path.name,
            "num_cells": int(centers.shape[0]),
            "power": float(params["laser_power"]),
            "velocity": float(params["V_scan"]),
            "time_s": float(params.get("time_s")) if "time_s" in params else None,
            "params": params,
        }
        out: Dict[str, Any] = {"coordinates": centers, "metadata": meta}
        out.update(fields)
        return out
    
    def _collect_cell_fields_pyvista(self, ds) -> Dict[str, np.ndarray]:
        """
        Collect cell arrays from a PyVista dataset.
        If a field only exists on points, convert point->cell using VTK averaging.
        """
        out: Dict[str, np.ndarray] = {}

        # Primary: cell_data :contentReference[oaicite:5]{index=5}
        for name in ds.cell_data.keys():
            arr = np.asarray(ds.cell_data[name])
            out[name] = self._normalize_field_array(arr).astype(np.float32)

        # Also bring in point-only arrays by converting point->cell (VTK filter) :contentReference[oaicite:6]{index=6}
        if ds.point_data and len(ds.point_data.keys()) > 0:
            ds_cells = ds.point_data_to_cell_data(pass_point_data=False)
            for name in ds_cells.cell_data.keys():
                if name not in out:
                    arr = np.asarray(ds_cells.cell_data[name])
                    out[name] = self._normalize_field_array(arr).astype(np.float32)

        return out

    def write(self, data: Dict[str, Any], filename: str) -> None:
        pass

    def should_skip(self, filename: str) -> bool:
        return False
    
    def _get_output_path(self, filename: str):
        # Not used for VTK reader; satisfy abstract interface
        return None
 
    def _write_impl_temp_file(self, path, data):
        # Not used; satisfy abstract interface
        raise NotImplementedError("VTKOpenFOAMDataSource is read-only.")
    
    def _detect_uniform_grid(
        self,
        coords: np.ndarray,
        Xmin: float, Xmax: float,
        Ymin: float, Ymax: float,
        Zmin: float, Zmax: float,
        nx: int, ny: int, nz: int,
        tol: float = 1e-6,
        allow_grid_interpolation: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if the cells in the given [X,Y,Z] range already form a uniform tensor grid
        of shape (nx,ny,nz). If yes, return mask + sorting info to crop/reshape
        without interpolation. If not, return None.
        """
        # Round coords to avoid floating point issues
        coords = np.round(coords, decimals=8) 
        # Unique coordinates along each axis
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        mask = (Xmin <= x) & (Xmax >= x) & (Ymin <= y) & (Ymax >= y) & (Zmin <= z) & (Zmax >= z)
        x = x[mask]
        y = y[mask]
        z = z[mask]
        xs = np.unique(x)
        ys = np.unique(y)
        zs = np.unique(z)
        nx_found = xs.size
        ny_found = ys.size
        nz_found = zs.size
        logger.debug(f"[VTK] uniform-grid detection: found grid sizes nx={nx_found}, ny={ny_found}, nz={nz_found}; expected nx={nx}, ny={ny}, nz={nz}")
        if (nx_found != nx) or (ny_found != ny) or (nz_found != nz):
            # The ROI cells may still form a *complete* rectilinear grid, just at a
            # different resolution than the config requested. Any axis-aligned crop
            # of a rectilinear grid stays rectilinear, so unique-axis counts whose
            # product equals the ROI cell count means "gridded, wrong size" — as
            # opposed to genuinely scattered data (product >> cell count), for which
            # KNN interpolation is the intended path. Silently interpolating gridded
            # data onto (nx,ny,nz) resamples it to the wrong size and poisons
            # training without a trace, so refuse unless the caller opts in.
            roi_cells = int(mask.sum())
            is_complete_grid = roi_cells > 0 and (nx_found * ny_found * nz_found == roi_cells)
            if is_complete_grid and not allow_grid_interpolation:
                raise GridResolutionMismatchError(
                    f"[VTK] Data forms a uniform {nx_found}x{ny_found}x{nz_found} grid in the "
                    f"configured ROI x=[{Xmin},{Xmax}] y=[{Ymin},{Ymax}] z=[{Zmin},{Zmax}], but "
                    f"grid_params requests {nx}x{ny}x{nz}. Loading it would silently interpolate "
                    f"real grid data onto the wrong resolution. Fix the case export or the "
                    f"grid_params resolution, or pass allow_grid_interpolation=True "
                    f"(custom.allow_grid_interpolation) to resample it on purpose."
                )
            return None

        # Build mask of cells inside these ranges (with small tolerance)
        span = max(
            1.0,
            float(abs(xs[-1] - xs[0])),
            float(abs(ys[-1] - ys[0])),
            float(abs(zs[-1] - zs[0])),
        )
        tol_pos = tol * span

        mask = (
            (coords[:, 0] >= xs[0] - tol_pos) & (coords[:, 0] <= xs[-1] + tol_pos) &
            (coords[:, 1] >= ys[0] - tol_pos) & (coords[:, 1] <= ys[-1] + tol_pos) &
            (coords[:, 2] >= zs[0] - tol_pos) & (coords[:, 2] <= zs[-1] + tol_pos)
        )
        sub_coords = coords[mask]
        n_expected = nx * ny * nz
        if sub_coords.shape[0] != n_expected:
            logger.debug(
                f"[VTK] uniform-grid candidate has {sub_coords.shape[0]} cells, "
                f"expected {n_expected}; falling back to interpolation."
            )
            return None

        # Sort by (x,y,z) so reshape(nx,ny,nz) matches (gx,gy,gz) with indexing='ij'
        sort_idx = np.lexsort((sub_coords[:, 2], sub_coords[:, 1], sub_coords[:, 0]))

        logger.debug("[VTK] Detected uniform grid in ROI; using crop-only path.")
        return {
            "mask": mask,
            "sort_idx": sort_idx,
            "gx": xs.astype(np.float32),
            "gy": ys.astype(np.float32),
            "gz": zs.astype(np.float32),
        }



    @staticmethod
    def _coordinate_channels(uniform_info, templates, shape):
        """x/y/z channels for one sample, matching where the fields actually sit.

        On the crop-only path the fields are the ROI's own cells, sampled
        at their centers, but the channels were always linspace(min, max, n) --
        offset ~half a cell from the centers, so the field grid and the
        coordinates the PDE loss differentiates against described different
        points. When that path is taken, the detector's true axes are used
        instead. The interpolation path resamples *onto* the linspace grid, so
        its templates already agree with its fields and are returned unchanged.
        """

        nx, ny, nz = shape
        if uniform_info is None:
            return templates
        gx = np.asarray(uniform_info["gx"], dtype=np.float32)
        gy = np.asarray(uniform_info["gy"], dtype=np.float32)
        gz = np.asarray(uniform_info["gz"], dtype=np.float32)
        if (gx.size, gy.size, gz.size) != (nx, ny, nz):
            # The crop path falls back to interpolation on a count mismatch; the
            # channels have to fall back with it.
            return templates
        full = (1, 1, nx, ny, nz)
        return (
            np.broadcast_to(gx.reshape(1, 1, nx, 1, 1), full),
            np.broadcast_to(gy.reshape(1, 1, 1, ny, 1), full),
            np.broadcast_to(gz.reshape(1, 1, 1, 1, nz), full),
        )

    def build_knn_for_grid(self, coords: np.ndarray, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, k: int = 8):
        """
        coords: (Ncells, 3) cell centers
        gx, gy, gz: 1D target axes
        returns (idxs, dists, nx, ny, nz):
        idxs  -> (Ngrid, k) int64 neighbor indices into coords
        dists -> (Ngrid, k) float32 distances
        """
        nx, ny, nz = len(gx), len(gy), len(gz)
        Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
        Q = np.column_stack([Gx.ravel(), Gy.ravel(), Gz.ravel()])  # (Ngrid, 3)

        tree = cKDTree(coords)
        dists, idxs = tree.query(Q, k=k)  # shapes: (Ngrid, k)
        if dists.ndim == 1:  # k == 1 case
            dists = dists[:, None]
            idxs = idxs[:, None]
        return idxs.astype(np.int64), dists.astype(np.float32), nx, ny, nz
    
    def apply_knn_scalar_torch(self,
            values: np.ndarray,       # (Ncells,)
            idxs: np.ndarray,         # (Ngrid, k)
            dists: np.ndarray,        # (Ngrid, k)
            shape_xyz: tuple[int,int,int],
            device: str = "cuda",
            eps: float = 1e-12,
        ) -> np.ndarray:
        """
        Compute interpolated scalar on GPU: out[i] = sum_j w_ij * values[idxs[i,j]]
        with w_ij = 1/(dist+eps) normalized across neighbors.
        Returns (nx,ny,nz) NumPy array.
        """
        nx, ny, nz = shape_xyz
        vals_t = torch.from_numpy(values).to(device=device, dtype=torch.float32)          # (Ncells,)
        idx_t  = torch.from_numpy(idxs).to(device=device, dtype=torch.long)               # (Ngrid,k)
        dst_t  = torch.from_numpy(dists).to(device=device, dtype=torch.float32)          # (Ngrid,k)

        w = 1.0 / (dst_t + eps)
        w = w / (w.sum(dim=1, keepdim=True) + eps)                                       # (Ngrid,k)
        gathered = vals_t[idx_t]                                                         # (Ngrid,k)
        out = (gathered * w).sum(dim=1)                                                  # (Ngrid,)
        return out.detach().cpu().numpy().reshape(nx, ny, nz).astype(np.float32)


    def apply_knn_vec3_torch(self,
        vec3: np.ndarray,         # (Ncells, 3)
        idxs: np.ndarray,         # (Ngrid, k)
        dists: np.ndarray,        # (Ngrid, k)
        shape_xyz: tuple[int,int,int],
        device: str = "cuda",
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Vector version. Returns (3, nx, ny, nz) NumPy array.
        """
        nx, ny, nz = shape_xyz
        vec_t = torch.from_numpy(vec3).to(device=device, dtype=torch.float32)            # (Ncells,3)
        idx_t = torch.from_numpy(idxs).to(device=device, dtype=torch.long)               # (Ngrid,k)
        dst_t = torch.from_numpy(dists).to(device=device, dtype=torch.float32)           # (Ngrid,k)

        w = 1.0 / (dst_t + eps)
        w = w / (w.sum(dim=1, keepdim=True) + eps)                                       # (Ngrid,k)

        # gather per component
        gathered = vec_t[idx_t]                                                          # (Ngrid,k,3)
        out = (gathered * w.unsqueeze(-1)).sum(dim=1)                                    # (Ngrid,3)
        out = out.permute(1, 0).reshape(3, nx, ny, nz).detach().cpu().numpy()            # (3,nx,ny,nz)
        return out.astype(np.float32)
            
    def crop_and_resample_scalar(self,
        coords, values, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz, k=8, device="cuda",
        flip_y: bool = False,
        uniform_info: Optional[Dict[str, Any]] = None,
    ):
        # ---- fast path: already uniform grid in ROI -> crop + reshape only ----
        if uniform_info is not None:
            mask = uniform_info["mask"]
            sort_idx = uniform_info["sort_idx"]
            gx = uniform_info["gx"]
            gy = uniform_info["gy"]
            gz = uniform_info["gz"]

            vals_sel = values[mask]
            n_expected = gx.size * gy.size * gz.size
            if vals_sel.shape[0] == n_expected:
                vals_sorted = vals_sel[sort_idx]
                grid_vals = vals_sorted.reshape(gx.size, gy.size, gz.size).astype(np.float32)
                if flip_y:
                    grid_vals = grid_vals[:, ::-1, :]
                    gy = gy[::-1].copy()
                logger.debug("[VTK] Used fast crop-only path for scalar field.")
                return gx, gy, gz, grid_vals
            else:
                logger.debug(
                    f"[VTK] uniform_info present but values length {vals_sel.shape[0]} "
                    f"!= expected {n_expected}; falling back to interpolation."
                )
        # ---- general path: build KNN and interpolate ----
        gx = np.linspace(Xmin, Xmax, nx, dtype=np.float32)
        gy = np.linspace(Ymin, Ymax, ny, dtype=np.float32)
        gz = np.linspace(Zmin, Zmax, nz, dtype=np.float32)
        idxs, dists, nx_, ny_, nz_ = self.build_knn_for_grid(coords, gx, gy, gz, k=k)
        grid_vals = self.apply_knn_scalar_torch(values.astype(np.float32), idxs, dists, (nx_, ny_, nz_), device=device)
        if flip_y:
            grid_vals = grid_vals[:, ::-1, :]
            gy = gy[::-1].copy()
        return gx, gy, gz, grid_vals
    
    def crop_and_resample_vec3(self,
        coords, vec3, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz, k=8, device="cuda", flip_y: bool = False,
        uniform_info: Optional[Dict[str, Any]] = None,
    ):
        # ---- fast path: already uniform grid in ROI ----
        if uniform_info is not None:
            mask = uniform_info["mask"]
            sort_idx = uniform_info["sort_idx"]
            gx = uniform_info["gx"]
            gy = uniform_info["gy"]
            gz = uniform_info["gz"]

            vec_sel = vec3[mask]  # (Ncells_roi, 3)
            n_expected = gx.size * gy.size * gz.size
            if vec_sel.shape[0] == n_expected:
                v_sorted = vec_sel[sort_idx]  # (Ncells_roi, 3)
                arr = v_sorted.reshape(gx.size, gy.size, gz.size, 3)  # (nx,ny,nz,3)
                U = np.moveaxis(arr, 3, 0).astype(np.float32)         # (3,nx,ny,nz)
                if flip_y:
                    U = U[:, :, ::-1, :]
                    gy = gy[::-1].copy()
                logger.debug("[VTK] Used fast crop-only path for vector field.")
                return gx, gy, gz, U
            else:
                logger.debug(
                    f"[VTK] uniform_info present but vec length {vec_sel.shape[0]} "
                    f"!= expected {n_expected}; falling back to interpolation."
                )
        gx = np.linspace(Xmin, Xmax, nx, dtype=np.float32)
        gy = np.linspace(Ymin, Ymax, ny, dtype=np.float32)
        gz = np.linspace(Zmin, Zmax, nz, dtype=np.float32)
        idxs, dists, nx_, ny_, nz_ = self.build_knn_for_grid(coords, gx, gy, gz, k=k)
        U = self.apply_knn_vec3_torch(vec3.astype(np.float32), idxs, dists, (nx_, ny_, nz_), device=device)  # (3,nx,ny,nz)
        if flip_y:
            U = U[:, :, ::-1, :]
            gy = gy[::-1].copy()
        return gx, gy, gz, U

    def parse_directory(
        self,
        grid_params,
        input_dir: Optional[str] = None,
        requested_fields = None,
        load_all_fields: bool = False,
        super_res_factor: int = 1,
        debug_params: bool = False,
        force_recompute: bool = False,
        flip_axes: bool = True,
        time_axis: Optional[bool] = None,
        avg_window_size: Optional[int] = None,
        moving_frame_steps: Optional[int] = None,
        moving_ref_frame: Optional[bool] = None,
        categorical_fields: Optional[Iterable[str]] = None,
        allow_grid_interpolation: bool = False,
    ) -> Dict[str, Any]:
        """
        Same usage pattern as HDF5 datasource:
        - grid_params: {'x_range':(..),'y_range':(..),'z_range':(..),'resolution':(nx,ny,nz)}
        - input_dir: directory containing VTK/VTU files
        - requested_fields: iterable of field names to load
        - load_all_fields: if True, ignore requested_fields and load everything present
        - force_recompute: ignore/delete cached NPZ and rebuild
        - categorical_fields: field names to resample with pure nearest neighbour
          (k=1) instead of the default k=8 inverse-distance blend, so
          categorical/flag fields keep their original discrete values. Default
          None leaves every field blended as before.
        - avg_window_size: sliding window for temporal averaging after moving-frame shift (1 disables averaging)
        - moving_frame_steps: number of final timesteps to keep/shift along x (defaults to total T)
        - moving_ref_frame: enable/disable moving-frame shift+crop (defaults to datasource setting)
        - returns a single var dict with arrays shaped (B,C,X,Y,Z); in time_axis mode a moving-frame
          shift reduces X to (nX - moving_frame_steps). If avg_window_size > 1 the time axis is
          averaged out and removed (keeps the last averaged timestep).
        """
        # Set/override input directory for this parse.
        if input_dir is not None:
            self.input_dir = Path(input_dir)

        if time_axis is not None and not isinstance(time_axis, bool):
            raise ValueError("time_axis must be a boolean when provided.")
        use_time_axis = self.time_axis if time_axis is None else time_axis
        use_moving_ref_frame = (
            self.moving_ref_frame if moving_ref_frame is None else bool(moving_ref_frame)
        )
        disable_shift_avg = use_time_axis and (avg_window_size is None) and (moving_frame_steps is None)
        window = None if disable_shift_avg else (self.avg_window_size if avg_window_size is None else int(avg_window_size))
        frame_steps = None if disable_shift_avg else (self.moving_frame_steps if moving_frame_steps is None else moving_frame_steps)
        if use_time_axis and not use_moving_ref_frame:
            frame_steps = 1
            window = 1
        load_all_fields = load_all_fields or (requested_fields is None)
        cat_fields = frozenset(str(f) for f in categorical_fields) if categorical_fields else frozenset()

        # Determine which additional VTK fields to load (fallback to legacy defaults if not loading all).
        fields_to_load = requested_fields
        base_required = {"x", "y", "z", "P", "V_scan"}
        if use_time_axis:
            base_required.add("time_s")
        # Key the cache on grid geometry + flip so a changed grid/range/flip can
        # never silently reuse a stale NPZ (old un-tokened cache names simply
        # stop matching and are recomputed).
        geom_token = hashlib.sha1(
            json.dumps(
                {
                    "x_range": [float(v) for v in grid_params["x_range"]],
                    "y_range": [float(v) for v in grid_params["y_range"]],
                    "z_range": [float(v) for v in grid_params["z_range"]],
                    "resolution": [int(v) for v in grid_params["resolution"]],
                    "flip_axes": bool(flip_axes),
                    "categorical_fields": sorted(cat_fields),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:8]
        if use_time_axis:
            if not use_moving_ref_frame:
                time_suffix = "time_noref"
            elif disable_shift_avg:
                time_suffix = "time_raw"
            else:
                steps_token = int(frame_steps) if frame_steps is not None else "all"
                time_suffix = f"time_steps{steps_token}_avg{window}"
            var_npz = self.input_dir / f"var_{super_res_factor}_{time_suffix}_{geom_token}.npz"
        else:
            var_npz = self.input_dir / f"var_{super_res_factor}_{geom_token}.npz"
        # If cached NPZ exist, ensure they contain the required fields; otherwise rebuild.
        if var_npz.exists() and not force_recompute:
            with np.load(var_npz) as cached:
                cached_keys = set(cached.files)
                missing_base = [f for f in base_required if f not in cached_keys]
                if load_all_fields:
                    if missing_base:
                        logger.debug(f"[VTK] Cached NPZ missing base fields {missing_base}; deleting and rebuilding.")
                        _safe_unlink(var_npz)
                    else:
                        self.extra_fields = sorted(k for k in cached_keys if k not in base_required)
                        logger.debug("[VTK] Found cached NPZ; using them (all fields from cache).")
                        return dict(np.load(var_npz))
                else:
                    self.extra_fields = [str(f) for f in fields_to_load] if fields_to_load is not None else []
                    required_npz_fields = base_required | set(self.extra_fields)
                    missing_cached = [f for f in required_npz_fields if f not in cached_keys]
                    if missing_cached:
                        logger.debug(f"[VTK] Cached NPZ missing fields {missing_cached}; deleting and rebuilding.")
                        _safe_unlink(var_npz)
                    else:
                        logger.debug("[VTK] Found cached NPZ; using them.")
                        return dict(np.load(var_npz))

        Xmin, Xmax = grid_params["x_range"]
        Ymin, Ymax = grid_params["y_range"]
        Zmin, Zmax = grid_params["z_range"]
        nx, ny, nz = grid_params["resolution"]

        # Precompute axes for broadcasting (constant across all batches/timesteps)
        gx = np.linspace(Xmin, Xmax, nx, dtype=np.float32)
        gy = np.linspace(Ymin, Ymax, ny, dtype=np.float32)
        gz = np.linspace(Zmin, Zmax, nz, dtype=np.float32)
        x = gx.reshape(1, 1, -1, 1, 1)
        y = gy.reshape(1, 1, 1, -1, 1)
        z = gz.reshape(1, 1, 1, 1, -1)
        xB_template = np.broadcast_to(x, (1, 1, nx, ny, nz))
        yB_template = np.broadcast_to(y, (1, 1, nx, ny, nz))
        zB_template = np.broadcast_to(z, (1, 1, nx, ny, nz))

        if use_time_axis:
            flat_paths = sorted(list(self.input_dir.glob("*.vtu")) + list(self.input_dir.glob("*.vtk")))
            groups = self._group_files_by_base(paths=flat_paths) if flat_paths else self._group_files_by_subdir()
            if not groups:
                raise ValueError("[VTK] No VTK/VTU files found for time-axis parsing.")
            group_lengths = {len(v) for v in groups.values()}
            if len(group_lengths) != 1:
                raise ValueError(f"[VTK] Time-axis mode requires equal timestep counts per base. Found counts: {sorted(group_lengths)}")
            # If loading all fields, peek at the first file (first base, first timestep).
            first_data_cache: Optional[Dict[str, Any]] = None
            if load_all_fields:
                first_base = sorted(groups.keys())[0]
                first_path = sorted(groups[first_base], key=self._time_sort_key)[0]
                first_data_cache = self.read_file(first_path)
                self.extra_fields = sorted(
                    k for k in first_data_cache.keys()
                    if k not in ("coordinates", "metadata")
                )
                logger.debug(f"[VTK] Loading all available fields (time axis): {self.extra_fields}")
            else:
                self.extra_fields = [str(f) for f in fields_to_load] if fields_to_load is not None else []
        else:
            filenames = self.get_file_list()

            # If loading all fields, peek at the first file to set the list.
            first_data_cache = None
            if load_all_fields:
                if not filenames:
                    raise ValueError("[VTK] No files found to infer fields from.")
                first_data_cache = self.read_file(filenames[0])
                self.extra_fields = sorted(
                    k for k in first_data_cache.keys()
                    if k not in ("coordinates", "metadata")
                )
                logger.debug(f"[VTK] Loading all available fields: {self.extra_fields}")
            else:
                self.extra_fields = [str(f) for f in fields_to_load] if fields_to_load is not None else []

        var_list = []
        # load_all_fields locks the channel set to the first file; track fields
        # seen only in later files so we can warn once instead of dropping them
        # silently.
        dropped_fields: Dict[str, str] = {}

        def _track_dropped(data: Dict[str, Any], source_name: str) -> None:
            if not load_all_fields:
                return
            for key in data.keys():
                if key in ("coordinates", "metadata") or key in self.extra_fields:
                    continue
                dropped_fields.setdefault(key, source_name)

        if use_time_axis:
            grouped_items = [(b, sorted(plist, key=self._time_sort_key)) for b, plist in groups.items()]
            grouped_items.sort(key=lambda x: x[0])
            for b_idx, (base, plist) in enumerate(grouped_items):
                base_entries = []
                uniform_info_base: Optional[Dict[str, Any]] = None
                for t_idx, path in enumerate(plist):
                    use_cached = load_all_fields and (b_idx == 0) and (t_idx == 0) and (first_data_cache is not None)
                    data = first_data_cache if use_cached else self.read_file(path)
                    _track_dropped(data, path.name)
                    coords = data["coordinates"]  # (N,3)
                    if flip_axes:
                        coords[:, 1] = -coords[:, 1]  # Flip Y to match expected orientation
                        coords = coords[:, [0, 2, 1]]  # swap Y and Z to match expected (X,Y,Z) ordering
                    meta = data["metadata"]
                    if uniform_info_base is None:
                        uniform_info_base = self._detect_uniform_grid(
                            coords, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz,
                            allow_grid_interpolation=allow_grid_interpolation,
                        )
                    xB, yB, zB = self._coordinate_channels(
                        uniform_info_base, (xB_template, yB_template, zB_template), (nx, ny, nz)
                    )
                    logger.debug(f"[VTK] Broadcasted 'x','y','z' shapes (time axis): {xB.shape}, {yB.shape}, {zB.shape}")

                    P = float(meta.get("power", 0.0))
                    Vscan = float(meta.get("velocity", 0.0))
                    Pgrid = np.full((1, 1, nx, ny, nz), P, dtype=np.float32)
                    Vgrid = np.full((1, 1, nx, ny, nz), Vscan, dtype=np.float32)
                    logger.debug(f"[VTK] Created 'Pgrid' and 'Vgrid' with shapes: {Pgrid.shape}, {Vgrid.shape}")
                    time_val = meta.get("time_s")
                    if time_val is None:
                        raise ValueError(
                            f"[VTK] No time parsed from '{path.stem}' in time-axis mode; a NaN "
                            f"time channel would poison training. Use time-suffixed filenames "
                            f"like 'case_123' (integer, 1e-6 s) or 'case_0.00123' (seconds)."
                        )
                    time_grid = np.full((1, 1, nx, ny, nz), float(time_val), dtype=np.float32)
                    # Optional scalar inputs (kept simple—only include if present)
                    opt_in = {}
                    for sname in self.extra_fields:
                        if sname not in data:
                            available = [k for k in data.keys() if k not in ("coordinates", "metadata")]
                            raise KeyError(
                                f"[VTK] Required field '{sname}' not found in '{path.stem}'. "
                                f"Available fields: {sorted(available)}"
                            )
                        vals = data[sname]
                        # Categorical fields use pure nearest neighbour (k=1).
                        knn_kwargs = {"k": 1} if sname in cat_fields else {}

                        # Scalar field: (N,)
                        if vals.ndim == 1:
                            _, _, _, sgrid = self.crop_and_resample_scalar(
                                coords, vals, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax,
                                nx, ny, nz, uniform_info=uniform_info_base, **knn_kwargs
                            )
                            opt_in[sname] = sgrid.reshape(1, 1, nx, ny, nz).astype(np.float32)
                            logger.debug(f"[VTK] Resampled scalar '{sname}' -> {opt_in[sname].shape}")

                        # Vector field: (N,3)  (e.g. gradT, Marangoni)
                        elif vals.ndim == 2 and vals.shape[1] == 3:
                            if flip_axes:
                                vals = vals[:, [0, 2, 1]]  # Swap Y and Z in raw data
                            _, _, _, vgrid = self.crop_and_resample_vec3(
                                coords, vals, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax,
                                nx, ny, nz, uniform_info=uniform_info_base, **knn_kwargs
                            )
                            opt_in[sname] = vgrid.reshape(1, 3, nx, ny, nz).astype(np.float32)
                            # Flip Z component to match expected orientation
                            if flip_axes:
                                opt_in[sname][0, 2, :, :, :] = -opt_in[sname][0, 2, :, :, :]
                            logger.debug(f"[VTK] Resampled vector '{sname}' -> {opt_in[sname].shape}")

                        else:
                            raise ValueError(
                                f"[VTK] Field '{sname}' has unsupported shape {vals.shape}; expected (N,) or (N,3)."
                            )

                    base_entries.append({
                        "x": xB, "y": yB, "z": zB,
                        "P": Pgrid, "V_scan": Vgrid,
                        "time_s": time_grid,
                        **opt_in,
                    })

                # Stack timesteps for this base: (1,T,C,X,Y,Z)
                stacked_base = {k: np.stack([entry[k] for entry in base_entries], axis=1) for k in base_entries[0]}
                var_list.append(stacked_base)
        else:
            for i, stem in enumerate(filenames):
                if (i == 0) and (first_data_cache is not None):
                    data = first_data_cache
                else:
                    data = self.read_file(stem)
                _track_dropped(data, str(stem))
                coords = data["coordinates"]  # (N,3)
                if flip_axes:
                    coords[:, 1] = -coords[:, 1]  # Flip Y to match expected orientation
                    coords = coords[:, [0, 2, 1]]  # swap Y and Z to match expected (X,Y,Z) ordering
                meta = data["metadata"]
                uniform_info = self._detect_uniform_grid(
                    coords, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz,
                    allow_grid_interpolation=allow_grid_interpolation,
                )
                # --- Broadcast x,y,z the same way as in the HDF5 datasource ---
                xB, yB, zB = self._coordinate_channels(
                    uniform_info, (xB_template, yB_template, zB_template), (nx, ny, nz)
                )
                logger.debug(f"[VTK] Broadcasted 'x','y','z' shapes: {xB.shape}, {yB.shape}, {zB.shape}")

                P = float(meta.get("power", 0.0))
                Vscan = float(meta.get("velocity", 0.0))
                Pgrid = np.full((1, 1, nx, ny, nz), P, dtype=np.float32)
                Vgrid = np.full((1, 1, nx, ny, nz), Vscan, dtype=np.float32)
                logger.debug(f"[VTK] Created 'Pgrid' and 'Vgrid' with shapes: {Pgrid.shape}, {Vgrid.shape}")
                # Optional scalar inputs (kept simple—only include if present)
                opt_in = {}
                for sname in self.extra_fields:
                    if sname not in data:
                        available = [k for k in data.keys() if k not in ("coordinates", "metadata")]
                        raise KeyError(
                            f"[VTK] Required field '{sname}' not found in '{stem}'. "
                            f"Available fields: {sorted(available)}"
                        )
                    vals = data[sname]
                    # Categorical fields use pure nearest neighbour (k=1).
                    knn_kwargs = {"k": 1} if sname in cat_fields else {}

                    # Scalar field: (N,)
                    if vals.ndim == 1:
                        _, _, _, sgrid = self.crop_and_resample_scalar(
                            coords, vals, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax,
                            nx, ny, nz, uniform_info=uniform_info,  # if you added the uniform fast path
                            **knn_kwargs
                        )
                        opt_in[sname] = sgrid.reshape(1, 1, nx, ny, nz).astype(np.float32)
                        logger.debug(f"[VTK] Resampled scalar '{sname}' -> {opt_in[sname].shape}")

                    # Vector field: (N,3)  (e.g. gradT, Marangoni)
                    elif vals.ndim == 2 and vals.shape[1] == 3:
                        if flip_axes:
                            vals = vals[:, [0, 2, 1]]  # Swap Y and Z in raw data
                        _, _, _, vgrid = self.crop_and_resample_vec3(
                            coords, vals, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax,
                            nx, ny, nz, uniform_info=uniform_info, **knn_kwargs
                        )
                        opt_in[sname] = vgrid.reshape(1, 3, nx, ny, nz).astype(np.float32)
                        # Flip Z component to match expected orientation
                        if flip_axes:
                            opt_in[sname][0, 2, :, :, :] = -opt_in[sname][0, 2, :, :, :]
                        logger.debug(f"[VTK] Resampled vector '{sname}' -> {opt_in[sname].shape}")

                    else:
                        raise ValueError(
                            f"[VTK] Field '{sname}' has unsupported shape {vals.shape}; expected (N,) or (N,3)."
                        )

                var_list.append({
                    "x": xB, "y": yB, "z": zB,
                    "P": Pgrid, "V_scan": Vgrid,
                    **opt_in,
                })

        if dropped_fields:
            listing = ", ".join(
                f"'{name}' (e.g. in '{source}')" for name, source in sorted(dropped_fields.items())
            )
            warnings.warn(
                f"[VTK] load_all_fields locked the channel set to the first file "
                f"({sorted(self.extra_fields)}); dropping fields present only in later "
                f"files: {listing}. Pass requested_fields to load them explicitly.",
                stacklevel=2,
            )

        # Concatenate along batch axis to form (B,T,C,X,Y,Z) if time_axis else (B,C,X,Y,Z)
        if len(var_list) == 0:
            raise ValueError("[VTK] No data loaded; var_list is empty, make sure your data directory and file patterns are correct. Perhaps you are dealing with temporal data and forgot +include_time=True?")
        var = {k: np.concatenate([d[k] for d in var_list], axis=0) for k in var_list[0]}

        if use_time_axis and not disable_shift_avg and use_moving_ref_frame:
            sample_key = next(iter(var))
            ref = var[sample_key]
            if ref.ndim != 6:
                raise ValueError(f"[VTK] Expected time-axis arrays to be 6D; got {ref.shape} for '{sample_key}'.")
            B, T, C, nX, nY, nZ = ref.shape
            Nsteps = T if frame_steps is None else int(frame_steps)
            if not (1 <= Nsteps <= T):
                raise ValueError(f"[VTK] moving_frame_steps must be in [1, {T}]; got {Nsteps}.")
            # Must be the formula crop_shift_moving_frame actually uses. The
            # pre-check used nX - Nsteps and ignored super_res_factor entirely,
            # so the two diverged as the factor grew: a request that passed here
            # still aborted inside the crop, with a bare ValueError instead of
            # this labelled one.
            new_nX = nX - (Nsteps - 1) * super_res_factor
            if new_nX <= 0:
                raise ValueError(
                    f"[VTK] moving_frame_steps ({Nsteps}) leaves no x-length at "
                    f"super_res_factor={super_res_factor}: "
                    f"nX ({nX}) - (Nsteps-1)*super_res_factor = {new_nX}."
                )
            if not (1 <= window <= Nsteps):
                raise ValueError(f"[VTK] avg_window_size must be in [1, {Nsteps}]; got {window}.")

            processed: Dict[str, np.ndarray] = {}
            for name, arr in var.items():
                if arr.ndim != 6:
                    raise ValueError(f"[VTK] Field '{name}' expected 6D array in time-axis mode; got shape {arr.shape}.")
                if arr.shape[1] != T:
                    raise ValueError(f"[VTK] Field '{name}' has unexpected timestep count {arr.shape[1]} (expected {T}).")
                shifted = crop_shift_moving_frame(arr, Nsteps, super_res_factor=super_res_factor)
                if window > 1:
                    averaged = moving_average_time(shifted, window=window, axis=1).astype(np.float32)
                    processed[name] = averaged[:, -1]
                else:
                    processed[name] = shifted
            var = processed

        # Cache for reuse. Best-effort: the data dir may be read-only (e.g. a
        # shared dataset in CI), and the cache is only an optimization — a failed
        # write must not abort an otherwise-complete parse.
        try:
            np.savez_compressed(var_npz, **var)  # pyright: ignore[reportArgumentType]
        except OSError as exc:
            warnings.warn(
                f"[VTK] Could not write resample cache '{var_npz.name}' ({exc}); "
                f"continuing without caching (each run recomputes it).",
                stacklevel=2,
            )
        return var

    # --------------------------- #
    # Internals
    # --------------------------- #

    def _resolve_file_path(self, stem: str | Path) -> Path:
        """
        Resolve a VTK/VTU path. Accepts either a bare stem or a full/relative path with or without extension.
        In time-axis subdir mode, callers may supply paths that include a subdirectory.
        """
        p_in = Path(stem)
        # If caller passed a full filename with extension, try it directly (and also try under input_dir).
        if p_in.suffix in (".vtu", ".vtk"):
            if p_in.exists():
                return p_in
            alt = self.input_dir / p_in.name
            if alt.exists():
                return alt
            raise FileNotFoundError(f"Could not find '{p_in}' or '{alt}'")

        # Otherwise treat as stem and resolve under the given parent (if any) or input_dir.
        parent = p_in.parent if p_in.parent != Path("") else self.input_dir
        stem_only = p_in.stem
        p_vtu = parent / f"{stem_only}.vtu"
        p_vtk = parent / f"{stem_only}.vtk"
        if p_vtu.exists(): return p_vtu
        if p_vtk.exists(): return p_vtk
        raise FileNotFoundError(f"Could not find '{p_vtu}' or '{p_vtk}'")

    @staticmethod
    def _base_without_time(stem: str) -> str:
        m = re.match(r"^(.*)_(\d+)$", stem)
        return m.group(1) if m else stem

    def _choose_by_time(self, paths: List[Path], target_time_s: Optional[float]) -> Path:
        # parse times; None sorts lowest
        parsed = []
        for p in paths:
            t = self._parse_time(p.stem)
            parsed.append((p, -1e30 if t is None else t))
        if target_time_s is None:
            # latest time
            return max(parsed, key=lambda x: x[1])[0]
        # closest to target
        return min(parsed, key=lambda x: abs(x[1] - target_time_s))[0]

    @staticmethod
    def _parse_time(stem: str) -> Optional[float]:
        m = re.search(r"_([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)$", stem)
        if not m:
            return None
        token = m.group(1)
        if any(ch in token for ch in ".eE"):
            return float(token)
        # Legacy integer-only VTK names encode time using a 1e-6 scale.
        return int(token) * 1e-6

    @staticmethod
    def _float_token(tok: str) -> Optional[float]:
        if tok is None: return None
        tok = tok.replace("p", ".")  # 0p01 -> 0.01
        try:
            return float(tok)
        except Exception:
            return None

    @staticmethod
    def _require_scan_parameters(params: Dict[str, Any], filename: str) -> None:
        """Refuse a file whose name carries no laser power / scan speed (A5).

        These become the P and V_scan channels, and H_norm = P/sqrt(|V_scan|).
        Defaulting either to 0.0 fed the network 0/0 = NaN on a scan speed no
        welding case can have, and trained on it without a word.
        """

        missing = [token for token in ("laser_power", "V_scan") if params.get(token) is None]
        if missing:
            raise ValueError(
                f"[VTK] '{filename}': could not read {', '.join(missing)} from the filename. "
                f"Expected tokens like 'V_scan_0p01_laser_power_5'; refusing to substitute 0.0, "
                f"which makes H_norm = P/sqrt(|V_scan|) undefined."
            )

    def _parse_params(self, stem: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        m = re.search(r"(?:^|_)V[_-]?scan_([0-9pEe+\-]+)", stem)
        if m:
            v = self._float_token(m.group(1))
            if v is not None: params["V_scan"] = v
        m = re.search(r"(?:^|_)laser[_-]?power_([0-9pEe+\-]+)", stem)
        if m:
            p = self._float_token(m.group(1))
            if p is not None: params["laser_power"] = p
        m = re.search(r"(?:^|_)beam[_-]?radius_([0-9pEe+\-]+)", stem)
        if m:
            br = self._float_token(m.group(1))
            if br is not None: params["beam_radius"] = br
        t = self._parse_time(stem)
        if t is not None: params["time_s"] = t
        return params

    @staticmethod
    def _normalize_field_array(data: np.ndarray) -> np.ndarray:
        """
        Normalize a raw cell/point array to either:
          - (N,)        for scalar fields
          - (N, C)      for vector/multicomponent fields (C >= 2)
        """
        data = np.asarray(data)
        if data.ndim == 1:
            return data
        if data.ndim == 2:
            if data.shape[1] == 1:
                return data[:, 0]
            return data  # (N, C) -> keep as is
        # For higher dims, flatten trailing dims into components
        n = data.shape[0]
        return data.reshape(n, -1)

    def _group_files_by_base(self, paths: Optional[List[Path]] = None) -> Dict[str, List[Path]]:
        """
        Group all VTK/VTU files by base stem (without trailing time token), sorted by time.
        """
        paths = sorted(paths if paths is not None else (list(self.input_dir.glob("*.vtu")) + list(self.input_dir.glob("*.vtk"))))
        groups: Dict[str, List[Path]] = {}
        for p in paths:
            groups.setdefault(self._base_without_time(p.stem), []).append(p)
        # Sort each group by time (ascending), fallback to name if time missing.
        for base, plist in groups.items():
            groups[base] = sorted(plist, key=self._time_sort_key)
        return groups

    def _group_files_by_subdir(self) -> Dict[str, List[Path]]:
        """
        Group VTK/VTU files by immediate subdirectory name; each subdir is one batch.
        """
        groups: Dict[str, List[Path]] = {}
        for d in sorted([p for p in self.input_dir.iterdir() if p.is_dir()]):
            plist = sorted(list(d.glob("*.vtu")) + list(d.glob("*.vtk")), key=self._time_sort_key)
            if plist:
                groups[d.name] = plist
        return groups

    def _time_sort_key(self, path: Path) -> tuple:
        t = self._parse_time(path.stem)
        return (1 if t is None else 0, -1e30 if t is None else t, path.name)
