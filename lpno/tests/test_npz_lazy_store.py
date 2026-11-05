"""Unit tests for the lazy per-case NPZ store (NpzDirStore).

Verifies the RuntimeStore interface the lazy training path uses (sample_count,
time_steps, spatial_shape, scalar_value, read_case) and that read_case emits the
same per-timestep tensors as the eager loader would: sliced fields, constant
material channels, path channels, coords/laser scalars. Numpy-only, no GPU.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lpno.models.pipeline.npz_lazy_store import NpzDirStore
from lpno.custom_utils import npz_datasource as nds

_PROPS = dict(rho=4124.66, cp=752.99, k=78.43, latent=248130.7, Tl=1388.33, A=0.38554)


def _manifest(length=6e-4, speed=0.9):
    xs = np.linspace(-3e-4, length - 3e-4, 41)
    xy = np.stack([xs, np.zeros_like(xs)], axis=1).tolist()
    return {
        "input": {"laser": {"scan_speed_m_per_s": speed, "beam_radius_m": 5e-5}},
        "derived": {"laser_path_xy_m": xy, "scan_time_s": length / speed,
                    "laser_off_time_s": length / speed},
        "material": {
            "properties": {"density_kg_m3": _PROPS["rho"], "specific_heat_j_kg_k": _PROPS["cp"],
                           "thermal_conductivity_w_m_k": _PROPS["k"], "latent_heat_j_kg": _PROPS["latent"],
                           "liquidus_temperature_k": _PROPS["Tl"]},
            "optical": {"normal_absorptivity": _PROPS["A"]},
        },
    }


def _write_cases(tmp_path, n=2, nx=10, ny=8, nz=6, nt=7):
    for i in range(n):
        fields = {
            "x": np.linspace(-3e-4, 3e-4, nx).astype(np.float32),
            "y": np.linspace(-2e-4, 2e-4, ny).astype(np.float32),
            "z": np.linspace(-3e-4, 0.0, nz).astype(np.float32),
            "time_s": np.linspace(0.0, 6e-4, nt).astype(np.float32),
            "power_w": np.float64(120.0 + i), "scan_speed_m_per_s": np.float64(0.9),
            "case_name": np.array(f"mlhs_{i:03d}"),
            "Temperature": (np.arange(nt)[:, None, None, None] +
                            np.zeros((nt, nx, ny, nz))).astype(np.float32),  # frame value == t index
            "Velocity": np.random.rand(nt, nx, ny, nz, 3).astype(np.float32),
            "Fraction Of Fluid": np.random.rand(nt, nx, ny, nz).astype(np.float32),
            "Pressure": np.random.rand(nt, nx, ny, nz).astype(np.float32),
        }
        np.savez(tmp_path / f"mlhs_{i:03d}.npz", **fields)
        (tmp_path / f"mlhs_{i:03d}.manifest.json").write_text(json.dumps(_manifest()))
    return nx, ny, nz, nt


def _store(tmp_path, **kw):
    return NpzDirStore(
        tmp_path,
        requested_fields=["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"],
        path_channels=list(nds.PATH_CHANNEL_NAMES),
        material_channels=list(nds.MATERIAL_CHANNEL_NAMES),
        **kw,
    )


def test_metadata_interface(tmp_path):
    nx, ny, nz, nt = _write_cases(tmp_path, n=3)
    s = _store(tmp_path)
    assert s.sample_count() == 3
    assert s.has_time_axis() and s.has_uniform_timesteps()
    assert s.time_steps(case_idx=0) == nt
    assert s.spatial_shape() == (nx, ny, nz)
    assert s.scalar_value("P", 1) == pytest.approx(121.0)
    assert s.scalar_value("V_scan", 0) == pytest.approx(0.9)


def test_read_case_timestep_shapes_and_values(tmp_path):
    nx, ny, nz, nt = _write_cases(tmp_path)
    s = _store(tmp_path)
    t_idx = 3
    raw = s.read_case(0, time_idx=t_idx)
    # scalar field -> (1,1,X,Y,Z); vector -> (1,3,X,Y,Z)
    assert raw["Temperature"].shape == (1, 1, nx, ny, nz)
    assert raw["Velocity"].shape == (1, 3, nx, ny, nz)
    # Temperature frame value == frame index by construction
    assert np.allclose(raw["Temperature"], float(t_idx))
    # coords + laser scalars present, constant material channels equal manifest
    for k in ("x", "y", "z", "P", "V_scan", "time_s", *nds.PATH_CHANNEL_NAMES, *nds.MATERIAL_CHANNEL_NAMES):
        assert k in raw, k
        assert raw[k].shape == (1, 1, nx, ny, nz), (k, raw[k].shape)
    assert np.allclose(raw["mat_k"], _PROPS["k"], rtol=1e-5)
    assert np.allclose(raw["mat_A"], _PROPS["A"], rtol=1e-5)
    assert np.ptp(raw["mat_rho"]) == 0.0
    # beam advances in +x: beam_x at a later frame >= earlier frame
    bx_late = s.read_case(0, time_idx=nt - 1)["beam_x"].flat[0]
    bx_early = s.read_case(0, time_idx=0)["beam_x"].flat[0]
    assert bx_late >= bx_early


def test_read_case_key_subset(tmp_path):
    _write_cases(tmp_path)
    s = _store(tmp_path)
    raw = s.read_case(0, time_idx=2, keys=["Temperature", "mat_k"])
    assert set(raw) == {"Temperature", "mat_k"}


def test_lru_cache_bounds_memory(tmp_path):
    _write_cases(tmp_path, n=3)
    s = _store(tmp_path, cache_size=1)
    s.read_case(0, time_idx=0)
    s.read_case(1, time_idx=0)
    s.read_case(2, time_idx=0)
    assert len(s._cache) == 1  # only the most-recent case retained


def test_missing_manifest_raises(tmp_path):
    _write_cases(tmp_path, n=1)
    (tmp_path / "mlhs_000.manifest.json").unlink()
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.read_case(0, time_idx=0)
