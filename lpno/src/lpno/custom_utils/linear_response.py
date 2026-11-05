"""Linear-response conditioning machinery: E7/E8 prototypes and channel candidates.

Prototype home for the recipe experiments (LP-NO manuscript, sec:formulation).
Everything here is computable from the case declaration alone -- never from the
solution -- so each function is a legitimate model input. A function graduates
into ``npz_datasource`` once its experiment earns it a place in a training
config; until then this module keeps the datasource untouched.

Contents:
  sigma_tau                  E7 spreading law sigma(tau) = sqrt(sigma0^2 + 2 a tau)
  gaussian_footprint         peak-1 Gaussian, the beam_dist convention
  leaky_dose_2d              leaky integrator; pass alpha to get the E7 variant
                             (each stamp deposited at the width its age implies)
  SpectralConduction         E8: exact linear conduction G*f by spectral stepping,
                             image source at the free surface, preheat as the
                             additive constant of the linear solution
  local_normalized_enthalpy  H* from the INSTANTANEOUS speed (varspeed channel),
                             same form as validate_scaling_law._ke_enthalpy
  tabular_footprint          radial_table/tabular beam profile rasterised as a
                             peak-1 footprint (ring x beamlet closed form)

Benchmark (E8 cost question): python -m lpno.custom_utils.linear_response
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# E7: the spreading law and the diffused dose
# --------------------------------------------------------------------------- #
def sigma_tau(sigma0: float, alpha: float, tau: float) -> float:
    """Width of a heat blob of age ~tau: sqrt(sigma0^2 + 2*alpha*tau).

    The frozen-width bank stamps every contribution at sigma0; physically a
    contribution the kernel remembers for ~tau has already spread to this width
    (Ti, sigma0=50um: 77um at 1ms, 143um at 3ms).
    """
    return math.sqrt(sigma0 * sigma0 + 2.0 * alpha * tau)


def gaussian_footprint(X: np.ndarray, Y: np.ndarray, cx: float, cy: float,
                       sigma: float) -> np.ndarray:
    """Peak-1 Gaussian footprint, the ``beam_dist`` convention."""
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma * sigma))


def leaky_dose_2d(X: np.ndarray, Y: np.ndarray,
                  px: Sequence[float], py: Sequence[float],
                  dt: Sequence[float], *, sigma: float, tau: float,
                  alpha: float | None = None,
                  weights: Sequence[float] | None = None) -> np.ndarray:
    """Final-frame leaky dose on a 2D grid; the generic history painter.

    With ``alpha`` None every stamp uses sigma, mirroring
    ``npz_datasource._leaky_dose``; with ``alpha`` given, stamps are deposited
    at sigma_tau(tau), the E7 kernel-consistent variant.

    ``weights`` paints an arbitrary per-step quantity into space instead of
    plain dwell time: weights[i] multiplies step i's deposit. weights=None is
    exactly weights=1 (a pure exposure clock). This one knob is what turns the
    dose into a regime-history channel (weights = H*(t), see ``hstar_dose_2d``)
    or a geometry-history channel (weights = kappa(t)*sigma).
    """
    stamp_sigma = sigma if alpha is None else sigma_tau(sigma, alpha, tau)
    acc = np.zeros_like(X, dtype=np.float64)
    for i in range(len(px)):
        decay = math.exp(-float(dt[i]) / tau) if i else 0.0
        w = 1.0 if weights is None else float(weights[i])
        acc = acc * decay + gaussian_footprint(X, Y, px[i], py[i], stamp_sigma) * (w * float(dt[i]))
    return (acc / tau).astype(np.float32)


def path_curvature(px: Sequence[float], py: Sequence[float],
                   signed: bool = False) -> np.ndarray:
    """Discrete curvature (1/m) at each vertex: turn angle / arc step.

    ``signed=True`` returns the signed curvature, positive for a LEFT turn --
    the convention that pairs with the left-positive cross-track offset
    ``beam_n``, and the one ``effective_speed_field`` requires. Endpoints get
    their neighbour's value so the series has the path's length. Multiply by
    the beam radius for the dimensionless kappa*sigma (measured on dataset
    pk 1601: |kappa|*sigma median 0.105, p99 1.10).
    """
    px = np.asarray(px, dtype=np.float64); py = np.asarray(py, dtype=np.float64)
    v = np.stack([np.diff(px), np.diff(py)], axis=1)
    seg = np.linalg.norm(v, axis=1)
    ang = np.arctan2(v[:, 1], v[:, 0])
    dth = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    if not signed:
        dth = np.abs(dth)
    step = np.maximum(0.5 * (seg[:-1] + seg[1:]), 1e-12)
    k_int = dth / step
    return np.concatenate([[k_int[0]], k_int, [k_int[-1]]]) if k_int.size else np.zeros(px.size)


def effective_speed_field(beam_n: np.ndarray, kappa: float, speed: float,
                          *, clamp: float = 0.8) -> np.ndarray:
    """The speed the moving source APPEARS to have from each point: v(1 - n*kappa).

    Kinematics of a source on a curved track: with the left-positive offset n
    (the ``beam_n`` convention) and the SIGNED curvature kappa (positive for a
    left turn, ``path_curvature(signed=True)``), a point sees the source sweep
    past at v_eff = v * (1 - n * kappa): the inner side of the bend (n and
    kappa same sign) sees a slower sweep -- more energy per unit length,
    hotter -- and the outer side a faster one. Sign discipline matters: an
    unsigned kappa puts the hot side of the pool on the wrong edge. Reduces
    exactly to the broadcast speed on a straight track (kappa = 0).

    ``beam_n`` here is the signed cross-track offset in metres (the channel
    value TIMES the beam radius); ``clamp`` bounds |n*kappa| because the
    construction is only valid inside the osculating radius (|n| < 1/|kappa|),
    and the frame itself has broken down well before that.
    """
    nk = np.clip(np.asarray(beam_n, dtype=np.float64) * float(kappa),
                 -float(clamp), float(clamp))
    return (float(speed) * (1.0 - nk)).astype(np.float32)


def hstar_dose_2d(X: np.ndarray, Y: np.ndarray,
                  px: Sequence[float], py: Sequence[float],
                  dt: Sequence[float], *, speeds: Sequence[float],
                  power_w: float, sigma: float, tau: float,
                  eta: float, rho: float, cp: float, dtl: float, lv: float,
                  alpha: float, r0: float,
                  v_floor: float | None = None) -> np.ndarray:
    """Regime history painted into space: dose weighted by the instantaneous H*.

    Each step deposits footprint * H*(v_i) * dt, so a point remembers not only
    that the beam passed but what regime the beam was in when it did -- a dwell
    is both a long exposure AND a high-H* (keyhole-leaning) excursion, and this
    channel is the one that keeps the two entangled the way the physics does.
    The dwell guard is on by default: v_floor falls back to alpha/r0 (the speed
    whose interaction time equals the diffusion time), so a stop on the path
    saturates instead of diverging.
    """
    if v_floor is None:
        v_floor = alpha / r0
    w = [local_normalized_enthalpy(power_w, float(v), eta=eta, rho=rho, cp=cp,
                                   dtl=dtl, lv=lv, alpha=alpha, r0=r0,
                                   v_floor=v_floor)
         for v in speeds]
    return leaky_dose_2d(X, Y, px, py, dt, sigma=sigma, tau=tau, weights=w)


# --------------------------------------------------------------------------- #
# E8: exact linear conduction by spectral stepping
# --------------------------------------------------------------------------- #
class SpectralConduction:
    """The exact linear-conduction field G*f for an arbitrary path, cheaply.

    Solves dT/dt = alpha * laplacian(T) + f spectrally: one state array in
    k-space, per-step cost = one multiply (diffusion propagator, exact for the
    linear equation at any dt) plus one analytic Gaussian stamp. A physical
    field costs one inverse FFT, paid only when a frame is exported.

    The free surface is nearly insulating, so it is handled by an image source:
    the z axis is mirrored (state size 2*nz) and sources sit on the mirror
    plane, which makes dT/dz = 0 at the surface exact. x/y are periodic, which
    is fine while tracks keep their usual margins from the domain edge.

    ``preheat`` enters exactly where the linear solution puts it: the additive
    constant the response relaxes toward, applied at export. This is the
    principled home for a preheat input (the broadcast scalar remains the
    trivial alternative).

    Units: q is deposited "dose" per unit time in the same arbitrary scale the
    dose channels use; the pipeline normalises channels, so only relative
    structure matters.
    """

    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float],
                 *, alpha: float, sigma0: float, preheat: float = 0.0):
        nx, ny, nz = shape
        dx, dy, dz = spacing
        self.shape, self.spacing = shape, spacing
        self.alpha, self.sigma0, self.preheat = float(alpha), float(sigma0), float(preheat)
        # mirrored-z working domain
        self._nz2 = 2 * nz
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
        kz = 2 * np.pi * np.fft.rfftfreq(self._nz2, d=dz)
        self._kx = kx[:, None, None]
        self._ky = ky[None, :, None]
        self._kz = kz[None, None, :]
        self._k2 = self._kx**2 + self._ky**2 + self._kz**2
        # analytic FT of the unit-peak 3D Gaussian stamp (space part; the
        # centre phase is applied per step). Surface plane z=0 is the mirror
        # plane, so the stamp is already even in z.
        s2 = self.sigma0 * self.sigma0
        self._stamp_k = np.exp(-0.5 * s2 * self._k2) * (
            (2 * np.pi * s2) ** 1.5 / (dx * dy * dz)
        ) / (nx * ny * self._nz2)
        # state layout: full FFT over x and y, rfft over the mirrored z axis
        self._state = np.zeros((nx, ny, self._nz2 // 2 + 1), dtype=np.complex128)

    def step(self, dt: float, cx: float, cy: float, q: float = 1.0) -> None:
        """Advance by dt (exact propagator), then deposit q*dt at (cx, cy)."""
        self._state *= np.exp(-self.alpha * self._k2 * float(dt))
        if q:
            phase = np.exp(-1j * (self._kx * cx + self._ky * cy))
            self._state += (q * float(dt)) * self._stamp_k * phase

    def field(self, full: bool = False) -> np.ndarray:
        """Physical field on the plate half, plus the preheat constant.

        ``full=True`` returns the whole mirrored-z working domain (2*nz planes),
        which is what the exact conservation invariant needs: the physical
        integral is trapezoidal over planes 0..nz, the two mirror planes at
        half weight.
        """
        f = np.fft.irfftn(self._state, s=(self.shape[0], self.shape[1], self._nz2),
                          axes=(0, 1, 2)) * (self.shape[0] * self.shape[1] * self._nz2)
        return f + self.preheat if full else f[:, :, : self.shape[2]] + self.preheat


# --------------------------------------------------------------------------- #
# Channel candidates with an existing dataset already varying the knob
# --------------------------------------------------------------------------- #
def local_normalized_enthalpy(power_w: float, speed_m_s: float, *, eta: float,
                              rho: float, cp: float, dtl: float, lv: float,
                              alpha: float, r0: float,
                              v_floor: float | None = None) -> float:
    """H* from the INSTANTANEOUS speed: the varspeed regime-indicator channel.

    Same form as ``validate_scaling_law._ke_enthalpy`` (h = cp*dTl + Lv):
    eta*P / (h * pi * rho * sqrt(alpha * v * r0^3)). A BROADCAST per-frame
    scalar, constant over the domain like ``beam_speed``: it describes the beam
    at this instant, and carries no spatial structure away from the path (the
    dose channels carry that).

    The quasi-steady formula diverges as v -> 0, and its derivation stops
    holding once the interaction time exceeds the diffusion time r0^2/alpha, so
    a dwell needs ``v_floor``: speeds are clamped to it, and the natural choice
    is alpha/r0 (the speed whose interaction time IS the diffusion time,
    ~0.06 m/s for Ti at 50 um). Without a floor, v = 0 raises.
    """
    if v_floor is not None:
        speed_m_s = max(speed_m_s, v_floor)
    if speed_m_s <= 0:
        raise ValueError("local_normalized_enthalpy: v=0 needs a v_floor "
                         "(the quasi-steady form diverges at a dwell)")
    h = cp * dtl + lv
    return eta * power_w / (h * math.pi * rho * math.sqrt(alpha * speed_m_s * r0**3))


def tabular_footprint(X: np.ndarray, Y: np.ndarray, cx: float, cy: float,
                      r0: float, *, samples: Iterable[Dict[str, float]],
                      beamlet_radius_fraction: float = 0.25) -> np.ndarray:
    """A ``radial_table``/``tabular`` beam profile rasterised as a peak-1 footprint.

    Mirrors the flow3D_scripts convention (config_schema: ``laser.beam_profile``):
    ``samples`` is a list of {radius_fraction, weight} shells on radius
    ``radius_fraction * r0``, realised with beamlets of width
    ``beamlet_radius_fraction * r0``. Azimuthally averaged, a shell convolved
    with its beamlet Gaussian has the closed form
    exp(-(r - rs)^2 / 2 sb^2) * i0e(r rs / sb^2), which is what a single
    Gaussian footprint cannot represent for a ring or top-hat.
    """
    from scipy.special import i0e

    sb = beamlet_radius_fraction * r0
    shells = [(float(s["radius_fraction"]) * r0, float(s["weight"]))
              for s in samples if float(s["weight"]) > 0]
    if not shells:
        raise ValueError("tabular_footprint: weights must be positive in total")

    def profile(r):
        out = np.zeros_like(r, dtype=np.float64)
        for rs, w in shells:
            if rs == 0.0:
                out += w * np.exp(-(r * r) / (2 * sb * sb))
            else:
                out += w * np.exp(-((r - rs) ** 2) / (2 * sb * sb)) * i0e(r * rs / (sb * sb))
        return out

    # normalise by the ANALYTIC radial peak, matching the beam_dist convention
    # (peak 1 in the continuum, on-grid values <= 1); the on-grid max would
    # depend on where the pixels happen to fall
    r_max = max(rs for rs, _ in shells) + 5 * sb
    peak = profile(np.linspace(0.0, r_max, 4096)).max()
    r = np.hypot(X - cx, Y - cy)
    return (profile(r) / peak).astype(np.float32)


# --------------------------------------------------------------------------- #
# E8 cost benchmark
# --------------------------------------------------------------------------- #
def _bench(shape: Tuple[int, int, int], n_steps: int, export_every: int) -> Dict[str, float]:
    import time

    sim = SpectralConduction(shape, (10e-6,) * 3, alpha=3e-6, sigma0=50e-6)
    t0 = time.perf_counter()
    for i in range(n_steps):
        sim.step(1e-5, cx=(i * 4e-6) % 1e-3, cy=0.0, q=1.0)
    t_steps = time.perf_counter() - t0
    n_exports = max(n_steps // export_every, 1)
    t0 = time.perf_counter()
    for _ in range(n_exports):
        sim.field()
    t_exports = time.perf_counter() - t0
    return {"per_step_ms": 1e3 * t_steps / n_steps,
            "per_export_ms": 1e3 * t_exports / n_exports,
            "case_s": t_steps + t_exports}


def main() -> int:
    print("E8 spectral-conduction cost (400-frame case, export every frame):")
    for shape in ((90, 40, 30), (180, 80, 80)):
        r = _bench(shape, n_steps=400, export_every=1)
        per150 = 150 * r["case_s"] / 60
        print(f"  grid {shape}: {r['per_step_ms']:.1f} ms/step, "
              f"{r['per_export_ms']:.1f} ms/export, {r['case_s']:.1f} s/case, "
              f"{per150:.0f} min for 150 cases (1 CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
