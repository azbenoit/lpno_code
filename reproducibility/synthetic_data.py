"""Synthetic straight-track laser-melting "simulations" for the training tutorial.

THESE ARE NOT FLOW-3D RESULTS. The temperature is a regularised Rosenthal
moving-point-source solution (a thick plate, constant properties), the metal
fraction is a flat surface with a small Gaussian depression under the beam, and
there is no fluid flow. The data only exercise the training and evaluation
pipeline; a model trained on them says nothing about laser melting.

Two writers, one per tutorial route:

* ``write_steady_split``: one beam-centred frame per case (the "steady"
  layout the ``config_flow3D_*_pvlhs_steady`` configs read), as consolidated
  NPZ files ``<split>/<case>.npz`` with a ``<case>.manifest.json`` sidecar.
  This is what ``lpno``' NPZ loader reads directly (route A).
* ``write_flow3d_runs``: the same cases as laboratory-frame time series in the
  run-directory layout ``flow3D_scripts`` writes (``<case>/case_manifest.json``
  and ``<case>/npz/<case>.npz``). This is what aiida-pino ingests (route B);
  aiida-pino then derives the steady layout itself (beam-following window and
  time average), as it did for the published steady models.

NPZ keys (``lpno/custom_utils/npz_datasource.py``): ``x(nx) y(ny)
z(nz) time_s(nt)``, the scalars ``power_w``, ``scan_speed_m_per_s`` and
``case_name``, and fields ``(nt, nx, ny, nz)``: ``Temperature`` (K) and
``Fraction Of Fluid`` (metal volume fraction, 1 in the metal, 0 in the gas).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Material and grid (Ti-6Al-4V-like round numbers; only orders of magnitude matter)
# --------------------------------------------------------------------------- #
T0_K = 300.0             # ambient / initial temperature
T_SOLIDUS_K = 1873.0     # matches material_params.T_s of the shipped configs
T_LIQUIDUS_K = 1923.0    # matches material_params.T_l
T_CAP_K = 3300.0         # soft ceiling, a little above boiling (Rosenthal is singular)
ABSORPTIVITY = 0.35
K_EFF_W_MK = 20.0        # effective conductivity (stands in for melt-pool convection)
RHO_KG_M3 = 4420.0
CP_J_KGK = 700.0
DIFFUSIVITY_M2_S = K_EFF_W_MK / (RHO_KG_M3 * CP_J_KGK)
BEAM_RADIUS_M = 5.0e-5
LATENT_HEAT_J_KG = 2.86e5

DX_M = 20.0e-6           # uniform cell size in x, y and z
NX_WINDOW = 40           # x cells of the beam-centred (steady) window
LASER_COL = 27           # beam column in that window: 2/3 behind, the aiida-pino convention
NY = 16
N_METAL_Z = 12           # cells at or below the surface z = 0
N_GAS_Z = 10             # gas cells above it: every FLOW-3D export here carries ten,
                         # and aiida-pino crops ten by default (crop_z_top)


@dataclass(frozen=True)
class Case:
    name: str
    power_w: float
    speed_m_s: float
    split: str


def default_cases(n_train: int = 8, n_val: int = 2, n_test: int = 2, seed: int = 0) -> list[Case]:
    """A small Latin-hypercube-like spread over power and scan speed."""
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    powers = np.linspace(120.0, 320.0, n)
    speeds = np.linspace(0.3, 1.0, n)
    rng.shuffle(speeds)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    order = rng.permutation(n)
    return [
        Case(f"synth_{i:03d}", float(powers[j]), float(speeds[j]), splits[i])
        for i, j in enumerate(order)
    ]


def axes(nx: int = NX_WINDOW, x_origin_col: int = LASER_COL):
    """Cell-centre axes in metres; x = 0 at column ``x_origin_col``, z = 0 at the surface."""
    x = (np.arange(nx) - x_origin_col) * DX_M
    y = (np.arange(NY) - (NY - 1) / 2.0) * DX_M
    z = (np.arange(N_METAL_Z + N_GAS_Z) - (N_METAL_Z - 1)) * DX_M
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def rosenthal_temperature(xi, y, z, power_w, speed_m_s):
    """Moving point source on a semi-infinite plate, regularised at the beam.

    ``xi`` is the distance ahead of the beam along the scan direction. Gas cells
    (z > 0) take the surface value below them.
    """
    zm = np.minimum(z, 0.0)
    r = np.sqrt(xi**2 + y**2 + zm**2)
    r_reg = np.sqrt(r**2 + BEAM_RADIUS_M**2)
    q = ABSORPTIVITY * power_w / (2.0 * np.pi * K_EFF_W_MK)
    dT = q / r_reg * np.exp(-speed_m_s * (xi + r) / (2.0 * DIFFUSIVITY_M2_S))
    span = T_CAP_K - T0_K
    return T0_K + span * (1.0 - np.exp(-dT / span))


def metal_fraction(xi, y, z, power_w, speed_m_s):
    """Metal volume fraction of each cell: a flat surface plus a shallow dimple."""
    h_norm = power_w / np.sqrt(speed_m_s)            # W (s/m)^0.5, a crude intensity measure
    depth = 1.0e-6 * max(0.0, h_norm - 150.0) / 10.0  # 0 to about 45 um over default_cases()
    surface = -depth * np.exp(-(xi**2 + y**2) / (2.0 * BEAM_RADIUS_M**2))
    lower = z - DX_M / 2.0
    return np.clip((surface - lower) / DX_M, 0.0, 1.0)


def _fields(x, y, z, beam_x, power_w, speed_m_s):
    X, Y, Z = np.meshgrid(x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), indexing="ij")
    xi = X - beam_x
    temp = rosenthal_temperature(xi, Y, Z, power_w, speed_m_s)
    alpha = metal_fraction(xi, Y, Z, power_w, speed_m_s)
    return temp.astype(np.float32), alpha.astype(np.float32)


def _manifest(case: Case, *, path_xy, scan_time_s, finish_time_s, steady: bool) -> dict:
    """A case manifest with the keys the loaders read (``npz_datasource``, aiida-pino import)."""
    return {
        "case_name": case.name,
        "schema_version": 3,
        "units": "SI",
        "synthetic": True,
        "description": "Synthetic Rosenthal field for the training tutorial; not a FLOW-3D run.",
        "input": {
            "laser": {
                "power_w": case.power_w,
                "scan_speed_m_per_s": case.speed_m_s,
                "beam_radius_m": BEAM_RADIUS_M,
                "heat_distribution": "gaussian",
            },
            "simulation": {"finish_time_s": finish_time_s, "initial_temperature_k": T0_K},
            "mesh": {"cell_size_m": DX_M},
            "material": "synthetic_ti64",
        },
        "derived": {
            "laser_path_xy_m": path_xy,
            "scan_time_s": scan_time_s,
            "laser_off_time_s": None if steady else scan_time_s,
            "laser_turn_off_at_end": True,
            "beam_velocity_m_per_s": case.speed_m_s,
            "surface_z_m": 0.0,
            "steady_beam_centred": steady,
        },
        "material": {
            "name": "synthetic_ti64",
            "properties": {
                "density_kg_m3": RHO_KG_M3,
                "specific_heat_j_kg_k": CP_J_KGK,
                "thermal_conductivity_w_m_k": K_EFF_W_MK,
                "latent_heat_j_kg": LATENT_HEAT_J_KG,
                "solidus_temperature_k": T_SOLIDUS_K,
                "liquidus_temperature_k": T_LIQUIDUS_K,
            },
            "optical": {"normal_absorptivity": ABSORPTIVITY},
        },
    }


def _save(path: Path, x, y, z, t, case: Case, temp, alpha) -> None:
    np.savez_compressed(
        path,
        x=x, y=y, z=z, time_s=np.asarray(t, dtype=np.float64),
        power_w=np.float64(case.power_w),
        scan_speed_m_per_s=np.float64(case.speed_m_s),
        case_name=np.array(case.name),
        Temperature=temp,
        **{"Fraction Of Fluid": alpha},
    )


def write_steady_split(root, cases=None) -> dict[str, list[str]]:
    """Write ``root/{train,val,test}/<case>.npz`` (+ sidecar manifest), one frame each.

    The beam sits at x = 0 (column ``LASER_COL``) and moves towards +x, so the
    melt-pool tail extends to negative x. Returns {split: [case names]}.
    """
    root = Path(root)
    cases = cases or default_cases()
    x, y, z = axes()
    out: dict[str, list[str]] = {}
    for case in cases:
        d = root / case.split
        d.mkdir(parents=True, exist_ok=True)
        temp, alpha = _fields(x, y, z, 0.0, case.power_w, case.speed_m_s)
        _save(d / f"{case.name}.npz", x, y, z, [0.0], case, temp[None], alpha[None])
        man = _manifest(case, path_xy=[[0.0, 0.0], [0.0, 0.0]], scan_time_s=0.0,
                        finish_time_s=0.0, steady=True)
        (d / f"{case.name}.manifest.json").write_text(json.dumps(man, indent=1))
        out.setdefault(case.split, []).append(case.name)
    return out


def write_flow3d_runs(root, cases=None, *, n_frames: int = 8, steady_frames: int = 5) -> list[str]:
    """Write laboratory-frame time series in the flow3D_scripts run layout.

    ``root/<case>/case_manifest.json`` and ``root/<case>/npz/<case>.npz``. The
    beam moves one cell per stored frame along +x. The field at each frame is
    the quasi-steady Rosenthal field around the current beam position (the
    start-up transient is ignored).

    The x extent is chosen for aiida-pino's steady derivation with
    ``moving_frame_steps = time_avg_window = steady_frames``: its beam-following
    window is ``nx - (steady_frames - 1)`` cells wide with the beam at 2/3 of it,
    so here it is exactly the ``NX_WINDOW``-cell window of ``write_steady_split``
    with the beam at column ``LASER_COL``, on each of the last ``steady_frames``
    frames.
    """
    if not 1 <= steady_frames <= n_frames:
        raise ValueError("steady_frames must be in [1, n_frames]")
    root = Path(root)
    cases = cases or default_cases()
    nx = NX_WINDOW + steady_frames - 1
    x, y, z = axes(nx=nx, x_origin_col=0)
    start_col = LASER_COL - (n_frames - steady_frames)
    if start_col < 0:
        raise ValueError("n_frames - steady_frames must not exceed LASER_COL")
    x0 = float(x[start_col])
    names = []
    for case in cases:
        dt = DX_M / case.speed_m_s
        t = np.arange(n_frames) * dt
        scan_time = float(t[-1])
        x1 = x0 + case.speed_m_s * scan_time
        case_dir = root / case.name
        (case_dir / "npz").mkdir(parents=True, exist_ok=True)
        temp = np.empty((n_frames, nx, NY, z.size), np.float32)
        alpha = np.empty_like(temp)
        for k, tk in enumerate(t):
            temp[k], alpha[k] = _fields(x, y, z, x0 + case.speed_m_s * tk, case.power_w, case.speed_m_s)
        _save(case_dir / "npz" / f"{case.name}.npz", x, y, z, t, case, temp, alpha)
        man = _manifest(case, path_xy=[[x0, 0.0], [x1, 0.0]], scan_time_s=scan_time,
                        finish_time_s=scan_time, steady=False)
        (case_dir / "case_manifest.json").write_text(json.dumps(man, indent=1))
        names.append(case.name)
    return names


def summary(npz_path) -> dict:
    """Peak temperature and a melt-pool size estimate (cells with T above liquidus, in the metal)."""
    with np.load(npz_path) as d:
        temp = d["Temperature"][-1]
        alpha = d["Fraction Of Fluid"][-1]
        z = d["z"]
        p, v = float(d["power_w"]), float(d["scan_speed_m_per_s"])
    melt = (temp > T_LIQUIDUS_K) & (alpha > 0.5)
    depth_cells = int(melt[:, :, z <= 0].any(axis=(0, 1)).sum())
    length_cells = int(melt.any(axis=(1, 2)).sum())
    return {"power_w": p, "speed_m_s": v, "T_max_K": float(temp.max()),
            "melt_cells": int(melt.sum()), "melt_depth_cells": depth_cells,
            "melt_length_cells": length_cells}
