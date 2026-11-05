"""Unit tests for material conditioning channels (per-case scalar properties).

Covers the manifest read + broadcast (constant full-field channels) and the
datasource plumbing (attach when requested, absent otherwise, loud error when the
manifest is missing). Numpy-only, no GPU/license.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lpno.custom_utils import npz_datasource as nds

# Realized values for one synthetic material_lhs_300 case (mlhs_000).
_PROPS = {
    "mat_rho": 4124.66,
    "mat_cp": 752.99,
    "mat_k": 78.43,
    "mat_latent": 248130.7,
    "mat_Tl": 1388.33,
    "mat_A": 0.38554,
}


def _material_manifest(props=_PROPS):
    """Minimal manifest carrying the material block the channels read."""
    return {
        "material": {
            "properties": {
                "density_kg_m3": props["mat_rho"],
                "specific_heat_j_kg_k": props["mat_cp"],
                "thermal_conductivity_w_m_k": props["mat_k"],
                "latent_heat_j_kg": props["mat_latent"],
                "liquidus_temperature_k": props["mat_Tl"],
            },
            "optical": {"normal_absorptivity": props["mat_A"]},
        }
    }


def test_build_material_channels_constant_fields_with_right_values():
    man = _material_manifest()
    shape = (1, 5, 1, 12, 10, 6)
    ch = nds._build_material_channels(nds.MATERIAL_CHANNEL_NAMES, man, shape)
    assert set(ch) == set(nds.MATERIAL_CHANNEL_NAMES)
    for k, v in ch.items():
        assert v.shape == shape, k
        assert v.dtype == np.float32, k
        assert np.allclose(v, _PROPS[k], rtol=1e-5), k  # constant in space and time
        assert np.ptp(v) == 0.0, k  # truly uniform


def test_build_material_channels_subset_only():
    ch = nds._build_material_channels(["mat_k", "mat_A"], _material_manifest(), (1, 3, 1, 4, 4, 2))
    assert set(ch) == {"mat_k", "mat_A"}


def test_missing_material_key_raises():
    bad = {"material": {"properties": {"density_kg_m3": 4000.0}}}  # no optical/normal_absorptivity
    with pytest.raises(KeyError):
        nds._build_material_channels(["mat_A"], bad, (1, 2, 1, 3, 3, 2))


def _write_case(tmp_path, nx=12, ny=10, nz=6, nt=5):
    fields = {
        "x": np.linspace(-1e-4, 8e-4, nx).astype(np.float32),
        "y": np.linspace(-2e-4, 2e-4, ny).astype(np.float32),
        "z": np.linspace(-3e-4, 0.0, nz).astype(np.float32),
        "time_s": np.linspace(0.0, 6e-4, nt).astype(np.float32),
        "power_w": np.float64(122.6), "scan_speed_m_per_s": np.float64(0.976),
        "case_name": np.array("mlhs_000"),
        "Temperature": np.random.rand(nt, nx, ny, nz).astype(np.float32),
    }
    np.savez(tmp_path / "mlhs_000.npz", **fields)
    return nx, ny, nz, nt


def test_load_consolidated_npz_attaches_material_and_requires_manifest(tmp_path):
    nx, ny, nz, nt = _write_case(tmp_path)
    kw = dict(requested_fields=["Temperature"], time_axis=True, avg_window_size=None,
              moving_frame_steps=None, moving_ref_frame=False)

    # No manifest yet -> requesting material channels must fail loudly (no silent zeros).
    with pytest.raises(ValueError):
        nds.load_consolidated_npz(tmp_path, material_channels=["mat_rho", "mat_k"], **kw)

    (tmp_path / "mlhs_000.manifest.json").write_text(json.dumps(_material_manifest()))
    var = nds.load_consolidated_npz(
        tmp_path, material_channels=list(nds.MATERIAL_CHANNEL_NAMES), **kw
    )
    for k in nds.MATERIAL_CHANNEL_NAMES:
        assert k in var, k
        assert var[k].shape == (1, nt, 1, nx, ny, nz), (k, var[k].shape)
        assert np.allclose(var[k], _PROPS[k], rtol=1e-5), k


def test_no_material_channels_when_not_requested(tmp_path):
    _write_case(tmp_path)
    (tmp_path / "mlhs_000.manifest.json").write_text(json.dumps(_material_manifest()))
    var = nds.load_consolidated_npz(
        tmp_path, requested_fields=["Temperature"], time_axis=True, avg_window_size=None,
        moving_frame_steps=None, moving_ref_frame=False, material_channels=None,
    )
    assert not any(k in var for k in nds.MATERIAL_CHANNEL_NAMES)
