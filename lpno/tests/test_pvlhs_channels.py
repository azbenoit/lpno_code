"""Channels for the PV-LHS beam/preheat sweep (dataset 38427 family).

The 6-D sweep varies preheat (293-773 K), beam shape (gaussian vs uniform) and focus
alongside P/V/radius. Three additions cover them:

  * ``preheat_T0``           broadcast per-case constant, raw Kelvin (like V_scan/P,
                             FiLM moves broadcast scalars to conditioning)
  * ``beam_shape_uniform``   0.0/1.0 flag broadcast
  * shape-aware ``beam_dist``: uniform-beam cases get a flat-top footprint instead of
                             a Gaussian that is wrong at exactly the beam-box scale

Regression guarantee: gaussian-manifest cases are BIT-IDENTICAL to the previous
behaviour -- the sweep's other half and every earlier dataset read the same channel
they always did. Numpy-only, no GPU/license.
"""
from __future__ import annotations

import numpy as np

from lpno.custom_utils import npz_datasource as nds


def _manifest(heat="gaussian", t0=293.15, focus=1.0, beam_r=5e-5, speed=0.8, length=1.0e-3):
    xs = np.linspace(0.0, length, 61)
    xy = np.stack([xs, np.zeros_like(xs)], axis=1).tolist()
    scan_time = length / speed
    return {
        "input": {
            "laser": {
                "scan_speed_m_per_s": speed,
                "beam_radius_m": beam_r,
                "heat_distribution": heat,
            },
            "simulation": {"initial_temperature_k": t0},
            "_design": {"focus_factor": focus},
        },
        "derived": {
            "laser_path_xy_m": xy,
            "scan_time_s": scan_time,
            "laser_off_time_s": scan_time,
        },
    }


def _grid():
    x = np.linspace(-2.0e-4, 1.2e-3, 90).astype(np.float64)
    y = np.linspace(-2.0e-4, 2.0e-4, 40).astype(np.float64)
    z = np.linspace(-4.0e-4, 1.0e-4, 10).astype(np.float64)
    t = np.linspace(0.0, 1.0e-3, 5)
    return x, y, z, t


def test_preheat_broadcast_value_and_shape():
    x, y, z, t = _grid()
    ch = nds._build_path_channels(["preheat_T0"], _manifest(t0=647.3), x, y, z, t)
    a = ch["preheat_T0"]
    assert a.shape == (1, len(t), 1, len(x), len(y), len(z))
    assert np.allclose(a, 647.3), "broadcast must be the raw Kelvin value everywhere"


def test_preheat_defaults_to_ambient_when_absent():
    x, y, z, t = _grid()
    man = _manifest()
    del man["input"]["simulation"]
    ch = nds._build_path_channels(["preheat_T0"], man, x, y, z, t)
    assert np.allclose(ch["preheat_T0"], 293.15), "older manifests carry no preheat: ambient"


def test_shape_flag_gaussian_zero_uniform_one():
    x, y, z, t = _grid()
    g = nds._build_path_channels(["beam_shape_uniform"], _manifest("gaussian"), x, y, z, t)
    u = nds._build_path_channels(["beam_shape_uniform"], _manifest("uniform"), x, y, z, t)
    assert np.allclose(g["beam_shape_uniform"], 0.0)
    assert np.allclose(u["beam_shape_uniform"], 1.0)


def test_focus_factor_broadcast_and_default():
    x, y, z, t = _grid()
    ch = nds._build_path_channels(["focus_factor"], _manifest(focus=1.23), x, y, z, t)
    assert np.allclose(ch["focus_factor"], 1.23)
    man = _manifest()
    del man["input"]["_design"]
    ch = nds._build_path_channels(["focus_factor"], man, x, y, z, t)
    assert np.allclose(ch["focus_factor"], 1.0)


def test_gaussian_footprint_bit_identical_to_legacy():
    """The gaussian path must not change AT ALL: every earlier dataset reads it."""
    x, y, z, t = _grid()
    man = _manifest("gaussian")
    got = nds._build_path_channels(["beam_dist"], man, x, y, z, t)["beam_dist"]
    # legacy formula, computed independently
    centres = nds._beam_centres_over_time(man, np.asarray(t))
    sigma = 5e-5
    r2 = ((x[None, :, None] - centres[:, 0][:, None, None]) ** 2
          + (y[None, None, :] - centres[:, 1][:, None, None]) ** 2)
    legacy = np.exp(-r2 / (2 * sigma * sigma)).astype(np.float32)
    assert np.array_equal(
        got[0, :, 0, :, :, 0], np.broadcast_to(legacy[:, :, :, None], got.shape[1:2] + got.shape[3:])[..., 0]
    ) or np.allclose(got[0, :, 0, :, :, 0], legacy, rtol=0, atol=0), \
        "gaussian footprint changed -- regression against every prior dataset"


def test_uniform_footprint_is_flat_topped():
    x, y, z, t = _grid()
    beam_r = 5e-5
    ch = nds._build_path_channels(["beam_dist"], _manifest("uniform", beam_r=beam_r), x, y, z, t)
    fp = ch["beam_dist"][0, 0, 0, :, :, 0]     # first frame, one z-slab
    centres = nds._beam_centres_over_time(_manifest("uniform"), np.asarray(t))
    cx, cy = centres[0]
    r = np.hypot(x[:, None] - cx, y[None, :] - cy)
    inside = fp[r < 0.6 * beam_r]
    outside = fp[r > 2.0 * beam_r]
    assert inside.min() > 0.9, "flat-top must be ~1 well inside the radius"
    assert inside.max() - inside.min() < 0.1, "flat-top must be FLAT inside (no Gaussian dome)"
    assert outside.max() < 0.05, "flat-top must vanish outside the beam"
    # and it must differ substantially from the gaussian at the mid-radius
    g = nds._build_path_channels(["beam_dist"], _manifest("gaussian", beam_r=beam_r), x, y, z, t)
    gfp = g["beam_dist"][0, 0, 0, :, :, 0]
    # the two shapes differ most OUTSIDE the radius: the flat-top has fallen to ~0
    # while a Gaussian with sigma=R still carries 0.2-0.6 there
    band = (r > 1.1 * beam_r) & (r < 1.8 * beam_r)
    assert np.abs(fp[band] - gfp[band]).max() > 0.3


def test_uniform_dose_uses_the_flat_top():
    """The dose family integrates the footprint, so shape-awareness must propagate."""
    x, y, z, t = _grid()
    u = nds._build_path_channels(["path_dose"], _manifest("uniform"), x, y, z, t)["path_dose"]
    g = nds._build_path_channels(["path_dose"], _manifest("gaussian"), x, y, z, t)["path_dose"]
    assert not np.allclose(u, g), "dose must inherit the shape-aware footprint"


def test_sigma_family_respects_the_beam_shape():
    """`beam_dist_sigma` must be flat-topped for a uniform case, like `beam_dist`.

    Found 2026-08-14: the sigma(z) footprint family was unconditionally Gaussian,
    so a flat-top case's sigma channels contradicted its own `beam_dist`. At this
    test's scale the Rayleigh range (~7 mm) dwarfs the z-extent (0.5 mm), so
    sigma_z ~= sigma everywhere and the sigma footprint must closely match the
    shape-aware base footprint -- for BOTH shapes.
    """
    x, y, z, t = _grid()
    for heat in ("uniform", "gaussian"):
        ch = nds._build_path_channels(["beam_dist", "beam_dist_sigma"], _manifest(heat), x, y, z, t)
        base = ch["beam_dist"][0, 0, 0]        # (nx, ny, nz)
        sig = ch["beam_dist_sigma"][0, 0, 0]
        assert np.abs(base - sig).max() < 0.05, (
            f"{heat}: sigma-family footprint diverges from the shape-aware base"
        )
