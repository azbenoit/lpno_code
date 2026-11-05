from typing import Tuple
from scipy.interpolate import griddata
import numpy as np

def create_grid(Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,nx,ny,nz):
    x_lin = np.linspace(Xmin, Xmax, nx)
    y_lin = np.linspace(Ymin, Ymax, ny)
    z_lin = np.linspace(Zmin, Zmax, nz)
    return x_lin, y_lin, z_lin

def resample_linear_then_nearest(
    coords: np.ndarray,      # (N,3) sample points
    values: np.ndarray,      # (N,)  scalar field
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray,   # uniform axes
):
    """
    Returns:
      V   : (Nx,Ny,Nz) float32, interpolated volume
      mask: (Nx,Ny,Nz) uint8, 1=linear used, 0=nearest fallback used
    """
    # Create query grid
    X, Y, Z = np.meshgrid(ux, uy, uz, indexing="ij")
    XI = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # (Nx*Ny*Nz, 3)

    # 1) Linear interpolation (best inside convex hull)
    V_lin = griddata(coords, values, XI, method="linear")
    mask = ~np.isnan(V_lin)  # True where linear succeeded

    # 2) Fill remaining with nearest
    if not mask.all():
        V_nn = griddata(coords, values, XI[~mask], method="nearest")
        V_lin[~mask] = V_nn
    
    V = V_lin.reshape(len(ux), len(uy), len(uz)).astype(np.float32)
    mask = mask.reshape(len(ux), len(uy), len(uz)).astype(np.uint8)
    return V, mask