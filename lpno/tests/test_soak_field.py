"""Analytic boundary-soak channel for the PV-LHS sweep.

The sweep's `initial_temperature_k` never reached the metal: the legacy_halfspace
renderer wrote `tvoid`/`tbct` (void and boundary) but not `tempi` (initial fluid), so
every case starts at FLOW-3D's 273.1 K default and the block then heats INWARD from its
faces at T0 for the whole run. A broadcast constant is the wrong channel for that: it
states the asymptote, not the state at time t.

The soak is a pure linear-conduction problem with a uniform initial condition and
Dirichlet faces, so it has a closed form -- separation of variables, one odd-harmonic
series per axis, multiplied:

    (T - T0)/(Ti - T0) = prod_d  (4/pi) sum_{n odd} (1/n) sin(n pi x_d / L_d)
                                          exp(-alpha n^2 pi^2 t / L_d^2)

No simulation, no cache, a handful of terms. These tests pin the physics rather than
the implementation: limits, monotonicity, symmetry, and the diffusion-length scale.
"""
from __future__ import annotations

import numpy as np

from lpno.custom_utils import npz_datasource as nds


def _manifest(t0=773.15, alpha=2.9e-6):
    xs = np.linspace(0.0, 1.0e-3, 61)
    xy = np.stack([xs, np.zeros_like(xs)], axis=1).tolist()
    return {
        "input": {
            "laser": {"scan_speed_m_per_s": 0.5, "beam_radius_m": 5e-5,
                      "heat_distribution": "gaussian"},
            "simulation": {"initial_temperature_k": t0},
        },
        "derived": {"laser_path_xy_m": xy, "scan_time_s": 2.0e-3,
                    "laser_off_time_s": 2.0e-3, "surface_z_m": 0.0},
        "material": {"properties": {"thermal_diffusivity_m2_s": alpha}},
    }


def _grid(nx=90, ny=40, nz=50):
    x = np.linspace(0.0, (nx - 1) * 10e-6, nx)
    y = np.linspace(0.0, (ny - 1) * 10e-6, ny)
    z = np.linspace(-(nz - 1) * 10e-6, 0.0, nz)
    t = np.array([0.0, 5e-4, 1.5e-3, 3.0e-3])
    return x, y, z, t


def test_starts_at_ambient_everywhere():
    """t=0 is the initial condition: the block is at 273.15 K, not at T0."""
    x, y, z, t = _grid()
    f = nds._build_path_channels(["soak_field"], _manifest(t0=773.15), x, y, z, t)["soak_field"]
    interior = f[0, 0, 0, 2:-2, 2:-2, 2:-2]
    assert np.allclose(interior, 273.15, atol=2.0), \
        f"t=0 interior should be ambient, got {interior.mean():.1f}"


def test_boundary_is_held_at_T0():
    x, y, z, t = _grid()
    t0 = 773.15
    f = nds._build_path_channels(["soak_field"], _manifest(t0=t0), x, y, z, t)["soak_field"]
    edge = f[0, -1, 0, 0, :, :]      # last frame, x = 0 face
    assert edge.max() > 0.9 * t0, "the driven face must approach T0"


def test_interior_warms_monotonically_and_stays_below_T0():
    x, y, z, t = _grid()
    t0 = 773.15
    f = nds._build_path_channels(["soak_field"], _manifest(t0=t0), x, y, z, t)["soak_field"]
    centre = f[0, :, 0, len(x) // 2, len(y) // 2, len(z) // 2]
    assert np.all(np.diff(centre) >= -1e-3), f"soak must be monotone, got {centre}"
    assert centre[-1] <= t0 + 1e-6, "cannot exceed the driving temperature"
    assert centre[0] < centre[-1], "the interior must warm at all"


def test_ambient_case_is_flat():
    """T0 == ambient means no soak: the field is constant everywhere, all times."""
    x, y, z, t = _grid()
    f = nds._build_path_channels(["soak_field"], _manifest(t0=273.15), x, y, z, t)["soak_field"]
    assert np.allclose(f, 273.15, atol=1e-3)


def test_penetration_matches_the_diffusion_length():
    """At 3 ms, sqrt(alpha t) ~ 93 um: the centre of a 400 um-wide block is barely warmed
    while a point one diffusion length in is substantially warmed."""
    x, y, z, t = _grid()
    t0 = 773.15
    f = nds._build_path_channels(["soak_field"], _manifest(t0=t0), x, y, z, t)["soak_field"]
    last = f[0, -1, 0]
    rise = lambda v: (v - 273.15) / (t0 - 273.15)
    near_face = rise(float(last[len(x) // 2, 9, len(z) // 2]))    # ~90 um from the y face
    centre = rise(float(last[len(x) // 2, len(y) // 2, len(z) // 2]))
    assert near_face > centre, "the soak must decay inward"
    assert near_face > 0.05, f"one diffusion length in should be warmed, got {near_face:.3f}"


def test_symmetric_across_the_block():
    x, y, z, t = _grid()
    f = nds._build_path_channels(["soak_field"], _manifest(), x, y, z, t)["soak_field"]
    last = f[0, -1, 0, len(x) // 2, :, len(z) // 2]
    assert np.allclose(last, last[::-1], rtol=1e-4), "y faces are driven equally"
