"""Oracle tests for the linear-response conditioning machinery (E7/E8 prototypes).

The load-bearing check is the spectral-conduction one: a single deposited pulse
must spread to the analytic Gaussian width sqrt(sigma0^2 + 2*alpha*T), because
that IS the heat-kernel physics the channel claims to carry.
"""
import math

import numpy as np
import pytest

from lpno.custom_utils import linear_response as lr


def test_sigma_tau_limits():
    assert lr.sigma_tau(50e-6, 3e-6, 0.0) == pytest.approx(50e-6)
    # 3 ms blob for Ti: sqrt(50um^2 + 2*3e-6*3e-3) ~ 143 um
    assert lr.sigma_tau(50e-6, 3e-6, 3e-3) == pytest.approx(
        math.sqrt((50e-6) ** 2 + 2 * 3e-6 * 3e-3), rel=1e-12
    )


def _second_moment_sigma(field, x):
    """Fitted Gaussian width from the second moment along x through the peak."""
    w = field / field.sum()
    mu = (w * x).sum()
    return math.sqrt(((x - mu) ** 2 * w).sum())


def test_spectral_pulse_spreads_to_heat_kernel():
    nx, ny, nz = 96, 96, 32
    dx = 10e-6
    alpha, sigma0 = 3e-6, 30e-6
    sim = lr.SpectralConduction((nx, ny, nz), (dx, dx, dx), alpha=alpha, sigma0=sigma0)
    # one unit pulse at the centre of the surface, then pure diffusion
    sim.step(1e-9, cx=nx // 2 * dx, cy=ny // 2 * dx, q=1.0)
    T = 200e-6
    nsteps = 40
    for _ in range(nsteps):
        sim.step(T / nsteps, cx=0.0, cy=0.0, q=0.0)
    field = sim.field()
    surface = field[:, :, 0]
    line = surface[:, ny // 2]
    x = np.arange(nx) * dx
    expected = math.sqrt(sigma0**2 + 2 * alpha * T)
    got = _second_moment_sigma(line, x)
    assert got == pytest.approx(expected, rel=0.05)


def test_spectral_conserves_energy_and_adds_preheat():
    # conservation at preheat=0: subtracting a 300 K offset from ~1e-11-scale
    # heat first would drown the invariant in floating-point cancellation
    sim = lr.SpectralConduction((32, 32, 48), (10e-6,) * 3, alpha=3e-6, sigma0=30e-6)
    sim.step(1e-9, cx=160e-6, cy=160e-6, q=2.0)

    def phys_total():
        nz = sim.shape[2]
        f = sim.field(full=True)
        return f[:, :, 0].sum() / 2 + f[:, :, 1:nz].sum() + f[:, :, nz].sum() / 2

    total0 = phys_total()
    for _ in range(20):
        sim.step(20e-6, cx=0.0, cy=0.0, q=0.0)
    # adiabatic box: the trapezoidal integral over the physical half is exact
    assert phys_total() == pytest.approx(total0, rel=1e-9)

    warm = lr.SpectralConduction((16, 16, 8), (10e-6,) * 3, alpha=3e-6,
                                 sigma0=30e-6, preheat=300.0)
    assert warm.field().min() == pytest.approx(300.0)
    warm.step(1e-9, cx=80e-6, cy=80e-6, q=1.0)
    assert warm.field().max() > 300.0


def test_tabular_footprint_degenerates_to_gaussian():
    n = 128
    x = np.linspace(-300e-6, 300e-6, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r0, bf = 100e-6, 0.35
    tab = lr.tabular_footprint(X, Y, 0.0, 0.0, r0,
                               samples=[{"radius_fraction": 0.0, "weight": 1.0}],
                               beamlet_radius_fraction=bf)
    gauss = lr.gaussian_footprint(X, Y, 0.0, 0.0, bf * r0)
    assert np.allclose(tab, gauss, atol=1e-6)


def test_tabular_ring_peaks_off_axis():
    n = 200
    x = np.linspace(-300e-6, 300e-6, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    ring = lr.tabular_footprint(X, Y, 0.0, 0.0, 100e-6,
                                samples=[{"radius_fraction": 0.8, "weight": 1.0}],
                                beamlet_radius_fraction=0.2)
    centre = ring[n // 2, n // 2]
    # analytic-peak normalisation: on-grid max approaches 1 from below
    assert 0.98 < ring.max() <= 1.0 + 1e-6
    assert centre < 0.5 * ring.max()          # a ring, so a hole in the middle
    peak_r = math.hypot(X.flat[np.argmax(ring)], Y.flat[np.argmax(ring)])
    assert peak_r == pytest.approx(80e-6, rel=0.1)


def test_local_enthalpy_matches_case_constant_and_scales():
    mat = dict(eta=0.4, rho=4420.0, cp=670.0, dtl=1655.0, lv=9.7e6,
               alpha=3e-6, r0=50e-6)
    h_case = lr.local_normalized_enthalpy(200.0, 0.5, **mat)
    assert h_case > 0
    # halving the instantaneous speed raises H* by sqrt(2): the varspeed point
    assert lr.local_normalized_enthalpy(200.0, 0.25, **mat) == pytest.approx(
        h_case * math.sqrt(2.0), rel=1e-12
    )
    # a dwell: clamped at the diffusion-time speed alpha/r0, never divergent
    v_star = mat["alpha"] / mat["r0"]
    assert lr.local_normalized_enthalpy(200.0, 0.0, v_floor=v_star, **mat) ==         pytest.approx(lr.local_normalized_enthalpy(200.0, v_star, **mat))
    with pytest.raises(ValueError):
        lr.local_normalized_enthalpy(200.0, 0.0, **mat)


def test_diffused_dose_wider_than_frozen():
    n = 180
    x = np.linspace(-0.9e-3, 0.9e-3, n)
    X, Y = np.meshgrid(x, np.linspace(-0.6e-3, 0.6e-3, n), indexing="ij")
    tpts = np.linspace(0.0, 4e-3, 160)
    px = -0.6e-3 + (1.2e-3 / 4e-3) * tpts
    py = np.zeros_like(px)
    dt = np.gradient(tpts)
    frozen = lr.leaky_dose_2d(X, Y, px, py, dt, sigma=50e-6, tau=3e-3)
    diffused = lr.leaky_dose_2d(X, Y, px, py, dt, sigma=50e-6, tau=3e-3,
                                alpha=3e-6)
    # cross-track width behind the beam: diffused must be wider
    row = np.argmin(np.abs(x - 0.0))
    y = np.linspace(-0.6e-3, 0.6e-3, n)
    w_frozen = _second_moment_sigma(frozen[row], y)
    w_diff = _second_moment_sigma(diffused[row], y)
    assert w_diff > 1.3 * w_frozen


def test_weighted_dose_reduces_to_plain_and_paints_history():
    n = 160
    x = np.linspace(-0.9e-3, 0.9e-3, n)
    X, Y = np.meshgrid(x, np.linspace(-0.6e-3, 0.6e-3, n), indexing="ij")
    tpts = np.linspace(0.0, 4e-3, 200)
    px = -0.6e-3 + (1.2e-3 / 4e-3) * tpts
    py = np.zeros_like(px)
    dt = np.gradient(tpts)
    plain = lr.leaky_dose_2d(X, Y, px, py, dt, sigma=50e-6, tau=3e-3)
    ones = lr.leaky_dose_2d(X, Y, px, py, dt, sigma=50e-6, tau=3e-3,
                            weights=np.ones_like(px))
    assert np.allclose(plain, ones)
    # weight only the first half of the track: the field must localise there
    w = (px < 0.0).astype(float)
    painted = lr.leaky_dose_2d(X, Y, px, py, dt, sigma=50e-6, tau=3e-3, weights=w)
    left = painted[x < -0.2e-3].sum()
    right = painted[x > 0.2e-3].sum()
    assert left > 5 * right


def test_path_curvature_oracles():
    # straight line: zero curvature
    tpts = np.linspace(0, 1e-3, 50)
    px = tpts * 0.5
    py = np.zeros_like(px)
    k = lr.path_curvature(px, py)
    assert np.abs(k).max() < 1e-6
    # circle of radius R: kappa = 1/R at every interior vertex
    R = 200e-6
    th = np.linspace(0, np.pi, 100)
    k2 = lr.path_curvature(R * np.cos(th), R * np.sin(th))
    interior = k2[2:-2]
    assert np.allclose(interior, 1.0 / R, rtol=0.02)


def test_hstar_dose_lights_up_the_dwell():
    mat = dict(eta=0.4, rho=4420.0, cp=670.0, dtl=1655.0, lv=9.7e6,
               alpha=3e-6, r0=50e-6)
    n = 160
    x = np.linspace(-0.9e-3, 0.9e-3, n)
    X, Y = np.meshgrid(x, np.linspace(-0.6e-3, 0.6e-3, n), indexing="ij")
    # constant-speed track except a slow middle third
    npts = 300
    px = np.linspace(-0.6e-3, 0.6e-3, npts)
    py = np.zeros_like(px)
    speed = np.where(np.abs(px) < 0.2e-3, 0.05, 0.5)
    seg = np.gradient(px)
    dt = np.abs(seg) / speed
    h = lr.hstar_dose_2d(X, Y, px, py, dt, speeds=speed, power_w=200.0,
                         sigma=50e-6, tau=30e-3, **mat)
    plain = lr.leaky_dose_2d(X, Y, px, py, dt, sigma=50e-6, tau=30e-3)
    mid = np.abs(x) < 0.15e-3
    out = x < -0.4e-3
    # the dwell is slow (more dwell time) AND high-H* (1/sqrt(v)); the ratio
    # of H*-painted to plain dose isolates the regime weighting alone
    ratio_mid = h[mid].max() / plain[mid].max()
    ratio_out = h[out].max() / plain[out].max()
    assert ratio_mid > 2.0 * ratio_out
    # dwell guard: v=0 in the series must not produce inf/nan
    speed0 = speed.copy(); speed0[len(speed0)//2] = 0.0
    h0 = lr.hstar_dose_2d(X, Y, px, py, dt, speeds=speed0, power_w=200.0,
                          sigma=50e-6, tau=30e-3, **mat)
    assert np.isfinite(h0).all()


def test_effective_speed_field_kinematics():
    n = np.linspace(-200e-6, 200e-6, 81)          # left-positive offset, metres
    v = lr.effective_speed_field(n, kappa=0.0, speed=0.4)
    assert np.allclose(v, 0.4)                     # straight track: broadcast v
    # LEFT turn (signed kappa > 0): the left side (n > 0) is the INSIDE of the
    # bend, radius R - n, swept SLOWER (omega * r); the outside faster
    R = 500e-6
    v = lr.effective_speed_field(n, kappa=1.0 / R, speed=0.4)
    inner = v[n > 0]; outer = v[n < 0]
    assert inner.max() < 0.4 < outer.min()
    assert np.isclose(v[-1], 0.4 * (1 - 200e-6 / R))   # r = R - n exactly
    # signed curvature of a counterclockwise arc is positive (left turn):
    # (R,0) -> (0,R) with increasing angle turns left
    th = np.linspace(0, np.pi / 2, 60)
    ks = lr.path_curvature(R * np.cos(th), R * np.sin(th), signed=True)
    assert (ks[2:-2] > 0).all()
    assert np.allclose(np.abs(ks[2:-2]), 1.0 / R, rtol=0.02)
    # clamp: tight bend never produces negative or runaway speeds
    v = lr.effective_speed_field(n, kappa=1.0 / 50e-6, speed=0.4, clamp=0.8)
    assert v.min() >= 0.4 * 0.2 - 1e-9 and v.max() <= 0.4 * 1.8 + 1e-9
