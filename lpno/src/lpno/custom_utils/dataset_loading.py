from __future__ import annotations
import os
import h5py
import re
import torch
import numpy as np
from scipy.interpolate import CubicSpline


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    from physicsnemo_curator.etl.data_sources import DataSource  # pyright: ignore[reportMissingImports]
except ImportError:  # physicsnemo-curator is not published on PyPI
    # `DataSource` is used here only as a base class, so a local stand-in keeps
    # this module importable on a machine that cannot obtain the package. It is
    # not on any index: `pip index versions physicsnemo-curator` finds nothing,
    # which made the whole distribution uninstallable for anyone but us.
    class DataSource:  # type: ignore[no-redef]
        """Minimal stand-in for the curator base class."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D107
            pass

from .resampling import create_grid, resample_linear_then_nearest

# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false

def fit_poly_from_txt(path, degree=1, delimiter=",", max_T=None, temp_first=False):
    arr = np.loadtxt(path, dtype=float, delimiter=delimiter)
    if arr.ndim == 1:  # single row case
        arr = arr[None, :]
    if temp_first:
        T, y = arr[:, 0], arr[:, 1]
    else:
        y, T = arr[:, 0], arr[:, 1]
    if max_T is not None:
        mask = T <= max_T
        T, y = T[mask], y[mask]
    coeffs = np.polyfit(T, y, degree)   # y ≈ poly(T)
    return coeffs  # returns array of coefficients, highest degree first


def get_c_piecewise_func(param_file, T_min=0, T_max=4000):
    """
    Build piecewise specific-heat functions for Ti-6Al-4V.

    Returns
    -------
    c_fun(T), dc_dT_fun(T), d2c_dT2_fun(T)
        Each accepts a numpy array/scalar or a torch tensor and returns the
        same type (torch out if torch in).

    Piecewise definition
    --------------------
    - T <= 1000 K: linear fit
    - 1000 K < T <= 2500 K: shape-preserving cubic Hermite (PCHIP)
    - T > 2500 K: constant (last value of the spline segment)
    """

    # --- Load and window the table ---
    c_tbl = np.loadtxt(param_file, delimiter=",")
    c_tbl = c_tbl[:, ::-1]  # your file is [c, T]; flip to [T, c]
    T_c, c_vals = c_tbl[:, 0], c_tbl[:, 1]
    mask = (T_c >= T_min) & (T_c <= T_max)
    T_c, c_vals = T_c[mask], c_vals[mask]

    # --- Masks by region for fitting ---
    mask_linear = T_c <= 1000.0
    mask_spline = (T_c > 1000.0) & (T_c <= 2500.0)

    # --- Fits ---
    # Linear region
    coeffs_linear = np.polyfit(T_c[mask_linear], c_vals[mask_linear], 1)  # [a1, a0] -> a1*T + a0
    slope_linear = coeffs_linear[0]
    const_linear = coeffs_linear[1]

    # PCHIP region (provides derivative() out of the box)
    spline = CubicSpline(T_c[mask_spline], c_vals[mask_spline],
                     bc_type='natural', extrapolate=True)
    spline_d1 = spline.derivative(1)
    spline_d2 = spline.derivative(2)

    # Constant region (use last spline value if available, else last table value)
    c_const_val = c_vals[mask_spline][-1] if len(c_vals[mask_spline]) > 0 else c_vals[-1]

    # --- helpers to keep torch/numpy symmetry ---
    def _to_numpy(x):
        is_torch = isinstance(x, torch.Tensor)
        x_np = x.detach().cpu().numpy() if is_torch else np.asarray(x)
        return is_torch, x_np

    def _to_like(x_in, x_np_out):
        if isinstance(x_in, torch.Tensor):
            return torch.from_numpy(np.asarray(x_np_out)).to(x_in.device, dtype=x_in.dtype)
        return x_np_out

    # --- c(T) ---
    def c_piecewise(T):
        is_torch, T_np = _to_numpy(T)
        out = np.empty_like(T_np, dtype=float)

        m1 = T_np <= 1000.0
        m2 = (T_np > 1000.0) & (T_np <= 2500.0)
        m3 = T_np > 2500.0

        # Linear
        out[m1] = slope_linear * T_np[m1] + const_linear
        # PCHIP (safe for array, NaN outside since extrapolate=False)
        vals2 = spline(T_np[m2])
        out[m2] = vals2
        # Constant
        out[m3] = c_const_val

        return _to_like(T, out)

    # --- dc/dT(T) ---
    def dc_dT_piecewise(T):
        is_torch, T_np = _to_numpy(T)
        out = np.empty_like(T_np, dtype=float)

        m1 = T_np <= 1000.0
        m2 = (T_np > 1000.0) & (T_np <= 2500.0)
        m3 = T_np > 2500.0

        # Linear slope is constant
        out[m1] = slope_linear
        # PCHIP first derivative
        out[m2] = spline_d1(T_np[m2])
        # Constant region derivative is zero
        out[m3] = 0.0

        return _to_like(T, out)

    # --- d²c/dT²(T) ---
    def d2c_dT2_piecewise(T):
        is_torch, T_np = _to_numpy(T)
        out = np.empty_like(T_np, dtype=float)

        m1 = T_np <= 1000.0
        m2 = (T_np > 1000.0) & (T_np <= 2500.0)
        m3 = T_np > 2500.0

        # Linear has zero curvature
        out[m1] = 0.0
        # PCHIP second derivative
        out[m2] = spline_d2(T_np[m2])
        # Constant region curvature is zero
        out[m3] = 0.0

        return _to_like(T, out)

    return c_piecewise, dc_dT_piecewise, d2c_dT2_piecewise

class H5AbaqusDataSource(DataSource):
    """
    DataSource for Abaqus/ODB-style HDF5 exports.

    Expected schema (minimal):
      /instances/<INSTANCE>/nodes                (N,3) float32/64
      /instances/<INSTANCE>/node_labels          (N,)  int32        [optional but recommended]
      /instances/<INSTANCE>/fields/<FIELD>/
          frame_000001/values                    (Nf,) float32/64
          frame_000001/labels                    (Nf,) int32        [optional, enables robust reindex]
          frame_000002/...
      /steps/<STEP>/frame_times                  (T,)  float64       [optional]

    Required fields and options are hard-coded (see class constants).
    """
    DEFAULT_INSTANCE = "BRONZEPART-1"   # set to your fixed instance (None to auto-pick first)
    FIELD_NAME = "NT11"                 # Abaqus temperature field
    FRAME = "last"                      # "first" | "last" | "frame_000001"
    STEP_NAME = "LaserHeating"          # step used for frame_times (or None)
    REMAP_BY_LABELS = True              # reorder values using labels if available

    def __init__(self, input_dir: str = "tutorial_data"):
        super().__init__(cfg=None)
        self.input_dir = Path(input_dir)
        self.instance_name = self.DEFAULT_INSTANCE
        self.field_name = self.FIELD_NAME
        self.frame_req = self.FRAME
        self.step_name = self.STEP_NAME
        self.remap_by_labels = self.REMAP_BY_LABELS

    # --------------------------- #
    # DataSource interface methods
    # --------------------------- #
    def _get_output_path(self, filename: str):
        # Not used for VTK reader; satisfy abstract interface
        return None
 
    def _write_impl_temp_file(self, path, data):
        # Not used; satisfy abstract interface
        raise NotImplementedError("VTKOpenFOAMDataSource is read-only.")
    
    def get_file_list(self) -> List[str]:
        """List *.h5 files (stem only, sorted)."""
        h5_files = sorted(self.input_dir.glob("*.h5"))
        filenames = [f.stem for f in h5_files]
        print(f"[AbaqusDataSource] Found {len(filenames)} HDF5 files: {filenames}")
        return filenames

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Read one Abaqus-style HDF5 file and return a dict with arrays + metadata."""
        path = self.input_dir / f"{filename}.h5"
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Hard-coded config
        instance_name = self.instance_name
        field_name = self.field_name
        frame_req = self.frame_req      # "first" | "last" | "frame_000001"
        step_name = self.step_name
        remap_by_labels = self.remap_by_labels
        # --- parse power & speed from filename stem ---
        power_val, speed_val = self._parse_power_speed_from_stem(Path(filename).stem)
        # A None here used to reach np.full((1,1,nx,ny,nz), None), which builds a
        # dtype=object array. That concatenates and even saves without complaint,
        # and only blows up much later in torch.from_numpy -- by which point the
        # filename that could not be parsed is long out of the traceback.
        if power_val is None or speed_val is None:
            raise ValueError(
                f"Could not parse laser power and scan speed from filename stem "
                f"'{Path(filename).stem}'. Expected '<anything>_<power>_<speed>', "
                f"where the last two underscore-separated tokens are numbers."
            )

        print(f"[AbaqusDataSource] Reading {path}")
        with h5py.File(path, "r") as f:
            # --- instances ---
            if "instances" not in f:
                raise ValueError("HDF5 is missing '/instances' group (not an Abaqus-style file).")
            inst_root = f["instances"]
            inst_keys = sorted(list(inst_root.keys()))
            if not inst_keys:
                raise ValueError("No instances found under '/instances'.")
            if instance_name is None:
                instance_name = inst_keys[0]
            if instance_name not in inst_root:
                raise ValueError(f"Requested instance '{instance_name}' not found. Available: {inst_keys}")

            g_inst = inst_root[instance_name]

            # nodes (+ labels)
            if "nodes" not in g_inst:
                raise ValueError(f"'/instances/{instance_name}/nodes' not found.")
            nodes = np.array(g_inst["nodes"])  # (N,3)
            node_labels = np.array(g_inst["node_labels"]) if "node_labels" in g_inst else None
            N = nodes.shape[0]
            if nodes.ndim != 2 or nodes.shape[1] != 3:
                raise ValueError(f"nodes must be (N,3); got {nodes.shape}")

            # --- field frames ---
            if "fields" not in g_inst:
                raise ValueError(f"'/instances/{instance_name}/fields' not found.")
            g_fields = g_inst["fields"]
            if field_name not in g_fields:
                raise ValueError(f"Field '{field_name}' not found under '/instances/{instance_name}/fields'. "
                                 f"Available: {list(g_fields.keys())}")

            g_field = g_fields[field_name]
            frame_keys = sorted([k for k in g_field.keys() if k.startswith("frame_")])
            if not frame_keys:
                raise ValueError(f"No 'frame_xxxxxx' groups under '/instances/{instance_name}/fields/{field_name}'.")

            # pick frame
            if frame_req == "first":
                frame_key = frame_keys[0]
            elif frame_req == "last":
                frame_key = frame_keys[-1]
            else:
                frame_key = frame_req
                if frame_key not in frame_keys:
                    raise ValueError(f"Requested frame '{frame_key}' not found. Available: {frame_keys[:5]}...")

            g_frame = g_field[frame_key]
            if "values" not in g_frame: # pyright: ignore[reportOperatorIssue]
                raise ValueError(f"'{frame_key}/values' missing for field '{field_name}'.")

            values = np.array(g_frame["values"])  # (Nf,)
            # Optional labels for this frame
            frame_labels = np.array(g_frame["labels"]) if "labels" in g_frame else None

            # Reindex values to match instance node order (if labels available)
            if remap_by_labels and (node_labels is not None) and (frame_labels is not None):
                # map: frame_labels -> values; then index by node_labels order
                sorter = np.argsort(frame_labels)
                # indices of node_labels within sorted frame_labels
                try:
                    indx = np.searchsorted(frame_labels[sorter], node_labels)
                    values = values[sorter][indx]
                except Exception as e:
                    print(f"[AbaqusDataSource] Warning: label-based reindex failed ({e}); proceeding without reindex.")
            else:
                # Assume values are already in node order (common in some exports)
                if values.shape[0] != N:
                    print(f"[AbaqusDataSource] Warning: values count ({values.shape[0]}) != nodes ({N}).")

            # --- step/frame times (optional) ---
            frame_time = None
            if "steps" in f:
                steps_root = f["steps"]
                step_keys = sorted(list(steps_root.keys()))
                if step_keys:
                    if step_name is None:
                        step_name = step_keys[0]
                    if step_name in steps_root and "frame_times" in steps_root[step_name]:
                        frame_times = np.array(steps_root[step_name]["frame_times"])
                        # infer index from suffix if possible (frame_000001 -> 1-based)
                        try:
                            idx = int(frame_key.split("_")[-1]) - 1
                            if 0 <= idx < len(frame_times):
                                frame_time = float(frame_times[idx])
                        except Exception:
                            pass

            # Build output dict similar to physicsNemo tutorial style
            out: Dict[str, Any] = {
                "coordinates": nodes.astype(np.float32),
                "temperature": values.astype(np.float32),
                "node_labels": node_labels.astype(np.int32) if node_labels is not None else None,
                "metadata": {
                    "instance": instance_name,
                    "field": field_name,
                    "frame": frame_key,
                    "frame_time": frame_time,
                    "num_points": int(N),
                    "power": power_val,
                    "velocity": speed_val,
                },
                "filename": filename,
            }

            # include element connectivity if present (useful for later meshing/VTK)
            try:
                # heuristic: first element family found
                if "elements" in g_inst:
                    g_elems = g_inst["elements"]
                    fams = list(g_elems.keys())
                    if fams:
                        fam0 = g_elems[fams[0]]
                        for cell_kind in fam0.keys():  # e.g., 'n8'
                            if "connectivity" in fam0[cell_kind]:
                                out["connectivity"] = np.array(fam0[cell_kind]["connectivity"])
                                out["element_labels"] = np.array(fam0[cell_kind]["element_labels"]) \
                                    if "element_labels" in fam0[cell_kind] else None
                                break
            except Exception:
                # ignore if not present
                pass

            return out

    def crop_and_resample(self, data: Dict[str, Any],Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz, mirror = False) -> Dict[str, Any]:
        """Crop to mask and resample to regular grid."""
        coords = data["coordinates"]  # (N,3)
        values = data["temperature"]  # (N,)
        if mirror:
            # Find points where y != 0 for mirroring
            mask_nonzero = np.abs(coords[:, 1]) > 1e-6  # tolerance for floating point

            coords_to_mirror = coords[mask_nonzero]
            theta_to_mirror  = values[mask_nonzero]

            # Mirror about y=0
            coords_mirrored = coords_to_mirror.copy()
            coords_mirrored[:, 1] = -coords_mirrored[:, 1]

            # Concatenate original and mirrored (excluding y=0 duplicates)
            coords = np.concatenate([coords, coords_mirrored], axis=0)
            values = np.concatenate([values, theta_to_mirror], axis=0)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        mask = (x >= Xmin) & (x <= Xmax) & (y >= Ymin) & (y <= Ymax) & (z >= Zmin) & (z <= Zmax)

        # Apply mask
        coords_masked = coords[mask]
        values_masked = values[mask]

        # Create regular grid
        grid_x, grid_y, grid_z = create_grid(Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz)
        # Resample using linear interpolation followed by nearest to fill holes
        grid_values, mask = resample_linear_then_nearest(
            coords_masked, values_masked, grid_x, grid_y, grid_z
        )
        # Update data dict
        data_resampled = data.copy()
        data_resampled["x"] = grid_x
        data_resampled["y"] = grid_y
        data_resampled["z"] = grid_z
        data_resampled["temperature"] = grid_values
        return data_resampled

    def parse_directory(self, grid_params, super_res_factor = 1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # If npz files are present, use them instead
        if os.path.exists(self.input_dir / f"invar_{super_res_factor}.npz"):
            print("[AbaqusDataSource] Found invar.npz in input directory; using that instead of HDF5 parsing.")
            invar = np.load(self.input_dir / f"invar_{super_res_factor}.npz")
            outvar = np.load(self.input_dir / f"outvar_{super_res_factor}.npz")
            # param = np.load(self.input_dir / f"params_{super_res_factor}.npz")
            return dict(invar), dict(outvar)
        filenames = self.get_file_list()
        invar_list = []
        outvar_list = []
        # param_list = []
        Xmin, Xmax = grid_params.get('x_range')
        Ymin, Ymax = grid_params.get('y_range')
        Zmin, Zmax = grid_params.get('z_range')
        nx, ny, nz = grid_params.get('resolution')
        # print(Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz)
        for fname in filenames:
            # try:
            data = self.read_file(fname)
            data_resampled = self.crop_and_resample(data, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, nx, ny, nz, mirror=True)
            x = data_resampled["x"]
            x_reshaped = x.reshape(1, 1, -1, 1, 1)         # shape (1, 1, nx, 1, 1)
            x_broadcasted = np.broadcast_to(x_reshaped, (1, 1, nx, ny, nz))  # shape (1, 1, nx, ny, nz)
            y = data_resampled["y"]
            y_reshaped = y.reshape(1, 1, 1, -1, 1)         # shape (1, 1, 1, ny, 1)
            y_broadcasted = np.broadcast_to(y_reshaped, (1, 1, nx, ny, nz))  # shape (1, 1, nx, ny, nz)
            z = data_resampled["z"]
            z_reshaped = z.reshape(1, 1, 1, 1, -1)         # shape (1, 1, 1, 1, nz)
            z_broadcasted = np.broadcast_to(z_reshaped, (1, 1, nx, ny, nz))  # shape (1, 1, nx, ny, nz)
            power = data_resampled["metadata"]["power"]
            power_array = np.full((1, 1, nx, ny, nz), power)
            velocity = data_resampled["metadata"]["velocity"]
            velocity_array = np.full((1, 1, nx, ny, nz), velocity)

            theta = data_resampled["temperature"] # shape (nx, ny, nz)
            # theta, temp_min, temp_max =scale_min_max(temp, new_min=-1, new_max=1)
            theta = theta.reshape(1, 1, nx, ny, nz)  # scale to [-1, 1] and add batch/channel dims

            # Collect dictionaries for this file
            invar_list.append({
                "x": x_broadcasted,
                "y": y_broadcasted,
                "z": z_broadcasted,
                "P": power_array,
                "V_scan": velocity_array,
                "u": np.zeros((1, 1, nx, ny, nz), dtype=np.float32),
                "v": np.zeros((1, 1, nx, ny, nz), dtype=np.float32),
                "w": np.zeros((1, 1, nx, ny, nz), dtype=np.float32),
            })
            outvar_list.append({
                "T": theta,
            })
            # param_list.append({
            #     "temp_min": temp_min,
            #     "temp_max": temp_max,
            # })
            # except Exception as e:
            #     print(f"[AbaqusDataSource] Error processing '{fname}': {e}")
        # Concatenate each variable along batch axis (axis=0)
        invar = {k: np.concatenate([d[k] for d in invar_list], axis=0) for k in invar_list[0]}
        outvar = {k: np.concatenate([d[k] for d in outvar_list], axis=0) for k in outvar_list[0]}
        # param = {k: np.array([d[k] for d in param_list]) for k in param_list[0]}
        # Save to npz for future use
        np.savez_compressed(self.input_dir / f"invar_{super_res_factor}", **invar) # pyright: ignore[reportArgumentType]
        np.savez_compressed(self.input_dir / f"outvar_{super_res_factor}", **outvar) # pyright: ignore[reportArgumentType]
        # np.savez_compressed(self.input_dir / f"params.npz_{super_res_factor}", **param) # pyright: ignore[reportArgumentType]
        return invar, outvar

    # Not used in this pipeline, but required by the interface
    def write(self, data: Dict[str, Any], filename: str) -> None:
        pass

    def should_skip(self, filename: str) -> bool:
        return False
    
    def read_all_files(self) -> List[Dict[str, Any]]:
        """Read all files in the input directory."""
        file_list = self.get_file_list()
        all_data = []
        for fname in file_list:
            try:
                data = self.read_file(fname)
                all_data.append(data)
            except Exception as e:
                print(f"[AbaqusDataSource] Error reading '{fname}': {e}")
        return all_data
    
    @staticmethod
    def _parse_power_speed_from_stem(stem: str) -> tuple[Optional[float], Optional[float]]:
        """
        Parse '<anything>_<power>_<speed>' from filename stem.
        Accepts floats/ints/scientific (e.g., '2.5e2').
        Returns (power, speed) or (None, None) if parsing fails.
        """
        # split by underscores; take the last two tokens
        parts = stem.split("_")
        if len(parts) < 3:
            return (None, None)

        # robust float extractor (handles '250', '2.5', '2.5e2', possibly with stray units)
        def to_float(tok: str) -> Optional[float]:
            m = re.search(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?", tok)
            if not m:
                return None
            try:
                return float(m.group(0))
            except Exception:
                return None

        power_tok = parts[-2]
        speed_tok = parts[-1]
        power = to_float(power_tok)
        speed = to_float(speed_tok)
        return (power, speed)
