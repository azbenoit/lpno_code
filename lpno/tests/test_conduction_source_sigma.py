"""`custom.conduction_source_sigma_factor`: the width of the conduction_field source.

The solver's Gaussian beam falls to 1/e^2 of its peak at the configured beam radius r0, so its intensity has a
standard deviation of r0/2. The conduction channel's source was a Gaussian of standard deviation r0, twice as
wide. The factor sets the source standard deviation to factor * r0: 1.0 (the default) keeps every existing model's
input as it trained on it, 0.5 matches the solver's beam.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lpno.custom_utils import npz_datasource as nds
from lpno.models.pipeline import npz_lazy_store as store


def _manifest(beam_r=5e-5, speed=0.4, length=1.0e-3, k=61):
    xs = np.linspace(0.0, length, k)
    scan_time = length / speed
    return {
        "case_name": "conduction_sigma_test",
        "input": {"laser": {"scan_speed_m_per_s": speed, "beam_radius_m": beam_r},
                  "material": {"properties": {"density_kg_m3": 4420.0, "specific_heat_j_kg_k": 670.0,
                                              "thermal_conductivity_w_m_k": 8.9}}},
        "derived": {"laser_path_xy_m": np.stack([xs, np.zeros_like(xs)], axis=1).tolist(),
                    "scan_time_s": scan_time, "laser_off_time_s": scan_time, "surface_z_m": 0.0},
    }


def _grid(man):
    x = np.linspace(0.0, 1.0e-3, 32)
    y = np.linspace(-2e-4, 2e-4, 40)
    z = np.linspace(-3.0e-4, 5e-5, 15)
    t = np.linspace(0.0, man["derived"]["scan_time_s"], 21)
    return x, y, z, t


def _build(man, **kw):
    x, y, z, t = _grid(man)
    return nds._build_path_channels(["conduction_field"], man, x, y, z, t, **kw)["conduction_field"][0]


def test_default_factor_reproduces_the_existing_channel(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_CONDUCTION_CACHE", str(tmp_path))
    man = _manifest()
    a = _build(man)
    b = _build(man, conduction_sigma_factor=1.0)
    assert np.array_equal(a, b)
    assert len(list(tmp_path.glob("*.npy"))) == 1      # one cache entry: the same key


def test_factor_reaches_the_spectral_source(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_CONDUCTION_CACHE", str(tmp_path))
    from lpno.custom_utils import linear_response as lr
    seen = []
    real = lr.SpectralConduction

    def spy(*args, **kwargs):
        seen.append(kwargs["sigma0"])
        return real(*args, **kwargs)

    monkeypatch.setattr(lr, "SpectralConduction", spy)
    _build(_manifest(beam_r=5e-5), conduction_sigma_factor=0.5)
    assert seen == [pytest.approx(2.5e-5)]


def test_half_width_source_is_narrower_and_cached_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_CONDUCTION_CACHE", str(tmp_path))
    man = _manifest()
    x, y, z, t = _grid(man)
    wide = _build(man, conduction_sigma_factor=1.0)[-5, 0]  # a late frame, (nx, ny, nz)
    narrow = _build(man, conduction_sigma_factor=0.5)[-5, 0]
    assert len(list(tmp_path.glob("*.npy"))) == 2       # different key: no stale reuse
    surf = int(np.where(z <= 1e-9)[0].max())

    def width(g):
        ix = int(np.argmax(g[:, :, surf].max(axis=1)))
        prof = g[ix, :, surf].astype(np.float64)
        prof = prof / prof.sum()
        mu = (prof * y).sum()
        return float(np.sqrt((prof * (y - mu) ** 2).sum()))

    assert width(narrow) < width(wide)


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan")])
def test_nonpositive_factor_is_refused(bad, tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_CONDUCTION_CACHE", str(tmp_path))
    with pytest.raises(ValueError, match="conduction_sigma_factor"):
        _build(_manifest(), conduction_sigma_factor=bad)


def test_resolver_reads_the_run_config():
    assert store.conduction_source_sigma_factor(SimpleNamespace()) == 1.0
    assert store.conduction_source_sigma_factor(SimpleNamespace(conduction_source_sigma_factor=0.5)) == 0.5


def test_baked_channels_carry_the_factor():
    old = {"power_weighted_dose": True, "power_weighted_conduction": False}   # a bake from before the factor
    ok = dict(power_weighted_dose=True, power_weighted_conduction=False)
    assert store._baked_channels_usable(old, **ok, conduction_source_sigma_factor=1.0)
    assert not store._baked_channels_usable(old, **ok, conduction_source_sigma_factor=0.5)
    new = dict(old, conduction_source_sigma_factor=0.5)
    assert store._baked_channels_usable(new, **ok, conduction_source_sigma_factor=0.5)
    assert not store._baked_channels_usable(new, **ok, conduction_source_sigma_factor=1.0)
