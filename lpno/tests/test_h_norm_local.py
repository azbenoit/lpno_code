"""Local normalized enthalpy: preheat as a melting-threshold modulation, not an offset.

`H_norm` as stored is P/sqrt(V) normalised by a split max -- an exact function of two
channels the network already has, and completely preheat-blind. The design's own
h_norm_ti64 measures melting enthalpy FROM T0, so it is preheat-aware. `h_norm_local`
is that form with T0 replaced by the soak field, making it a spatial, time-varying
field rather than a per-case scalar.
"""
from __future__ import annotations

import numpy as np

from lpno.custom_utils import npz_datasource as nds


def _manifest(t0=773.15, power=150.0, speed=0.5, r0=5e-5):
    xs = np.linspace(0.0, 1.0e-3, 61)
    return {
        "input": {
            "laser": {"scan_speed_m_per_s": speed, "beam_radius_m": r0,
                      "power_w": power, "heat_distribution": "gaussian"},
            "simulation": {"initial_temperature_k": t0},
        },
        "derived": {"laser_path_xy_m": np.stack([xs, np.zeros_like(xs)], 1).tolist(),
                    "scan_time_s": 2.0e-3, "laser_off_time_s": 2.0e-3, "surface_z_m": 0.0},
        "material": {
            "properties": {"density_kg_m3": 4420.0, "specific_heat_j_kg_k": 546.0,
                           "latent_heat_j_kg": 345000.0, "liquidus_temperature_k": 1923.15,
                           "thermal_conductivity_w_m_k": 7.0},
            "optical": {"normal_absorptivity": 0.35},
        },
    }


def _grid():
    x = np.linspace(0.0, 890e-6, 90)
    y = np.linspace(0.0, 390e-6, 40)
    z = np.linspace(-490e-6, 0.0, 50)
    return x, y, z, np.array([0.0, 1.0e-3, 3.0e-3])


def test_rises_where_the_block_has_soaked():
    """Soaked material needs less enthalpy to melt, so H* must be LARGER near the faces."""
    x, y, z, t = _grid()
    h = nds._build_path_channels(["h_norm_local"], _manifest(t0=773.15), x, y, z, t)["h_norm_local"]
    last = h[0, -1, 0]
    near_face = float(last[len(x) // 2, 1, len(z) // 2])
    centre = float(last[len(x) // 2, len(y) // 2, len(z) // 2])
    assert near_face > centre, f"soaked edge {near_face:.4g} should exceed cold centre {centre:.4g}"


def test_grows_with_time_as_the_soak_advances():
    x, y, z, t = _grid()
    h = nds._build_path_channels(["h_norm_local"], _manifest(t0=773.15), x, y, z, t)["h_norm_local"]
    probe = h[0, :, 0, len(x) // 2, 3, len(z) // 2]
    assert np.all(np.diff(probe) >= -1e-12), f"H* must be non-decreasing as heat arrives: {probe}"
    assert probe[-1] > probe[0]


def test_uniform_when_there_is_no_preheat():
    """T0 == ambient: no soak, so the field is the classic per-case scalar everywhere."""
    x, y, z, t = _grid()
    h = nds._build_path_channels(["h_norm_local"], _manifest(t0=273.15), x, y, z, t)["h_norm_local"]
    assert np.allclose(h, h.flat[0], rtol=1e-6), "no preheat means no spatial structure"


def test_is_exactly_one_at_ambient():
    """The channel is the RATIO to the ambient-T0 value, so an unsoaked block reads 1.0."""
    x, y, z, t = _grid()
    man = _manifest(t0=273.15, power=150.0, speed=0.5, r0=5e-5)
    h = nds._build_path_channels(["h_norm_local"], man, x, y, z, t)["h_norm_local"]
    assert np.allclose(h, 1.0, rtol=1e-5), f"ambient must read 1.0, got {float(h.flat[0]):.6g}"


def test_ratio_cancels_power_and_speed():
    """P, V and r are already separate channels; the ratio must not re-encode them, so
    only the soak-dependent part survives."""
    x, y, z, t = _grid()
    g = lambda **kw: nds._build_path_channels(
        ["h_norm_local"], _manifest(t0=773.15, **kw), x, y, z, t)["h_norm_local"]
    assert np.allclose(g(power=300.0), g(power=150.0), rtol=1e-6)
    assert np.allclose(g(speed=2.0), g(speed=0.5), rtol=1e-6)


def test_stays_order_one_across_the_sweeps_preheat_range():
    x, y, z, t = _grid()
    for t0 in (293.15, 523.15, 773.15):
        h = nds._build_path_channels(["h_norm_local"], _manifest(t0=t0), x, y, z, t)["h_norm_local"]
        assert 1.0 - 1e-6 <= h.min() and h.max() < 3.0, \
            f"T0={t0}: range {h.min():.3f}-{h.max():.3f} is not O(1)"


def test_no_divide_by_zero_when_soak_reaches_the_liquidus():
    """A pathological T0 above the liquidus must clamp, not explode."""
    x, y, z, t = _grid()
    h = nds._build_path_channels(["h_norm_local"], _manifest(t0=2500.0), x, y, z, t)["h_norm_local"]
    assert np.all(np.isfinite(h)) and h.max() < 1e6
