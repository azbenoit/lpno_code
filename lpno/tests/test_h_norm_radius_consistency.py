"""`h_norm_include_radius=True` must be honoured consistently, or refused loudly.

The King-form term only cancels when the SAME convention builds the numerator and the
denominator. An audit on 2026-08-21 found four ways they could still disagree with the
flag on, each silent:

  1. `NpzDirStore.scalar_value` served only P/V_scan, so the denominator raised on the
     very datasets the term exists for (ti64_preheat60_fixed59, 44 distinct radii).
  2. `h_norm_mode="ref_scale"` divides by the constant `ref_scales["H"]`, which carries
     no radius, while the numerator does -- the k-fold path.
  3. An explicit `custom.h_norm_denom` (live in every alsi_* AL run) is returned before
     `include_radius` is ever consulted; it was chosen radius-blind.
  4. If `beam_radius` is absent from the sample, the numerator silently drops the term
     (`var.get`) while the denominator raises -- an inverted asymmetry, and the stashed
     `h_norm_denom_radius_aware` marker then records False for a radius-aware denom.

1 must WORK. 2-4 must FAIL LOUDLY: a wrong H_norm is worth six orders of magnitude and
returns a plausible number rather than an error.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lpno.models.pipeline.preprocessing_datasets import _preprocess_var

SHP = (1, 1, 2, 2, 2)
R0 = 45e-6


def _var(with_radius=True):
    v = {
        "Temperature": np.full(SHP, 1000.0, np.float32),
        "Fraction Of Fluid": np.ones(SHP, np.float32),
        "Velocity": np.zeros((1, 3, 2, 2, 2), np.float32),
        "Pressure": np.zeros(SHP, np.float32),
        "x": np.linspace(0.0, 1e-4, 2, dtype=np.float32),
        "y": np.linspace(0.0, 1e-4, 2, dtype=np.float32),
        "z": np.linspace(0.0, 1e-4, 2, dtype=np.float32),
        "P": np.full(SHP, 100.0, np.float32),
        "V_scan": np.full(SHP, 0.25, np.float32),
    }
    if with_radius:
        v["beam_radius"] = np.full(SHP, R0, np.float32)
    return v


def _cfg(**over):
    base = dict(
        software="flow3D", sharp_alpha=False, alpha_masking=False,
        material_params={"T_s": 1878.0, "T_l": 1928.0}, laser_params={},
        ref_scales={"L": 1e-4, "T": 3000.0, "V": 0.1, "P": 200.0, "Rho": 4000.0, "H": 5.6},
        pde_loss=False, vof_pde_loss=False, use_2d=False,
        time_as_batch=False, include_time=False, h_norm_include_radius=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_ref_scale_mode_refuses_radius():
    """ref_scale divides by a constant that carries no radius -- must not silently differ."""
    with pytest.raises(ValueError, match="h_norm_include_radius"):
        _preprocess_var(_var(), _cfg(), h_norm_mode="ref_scale",
                        set_runtime_params=False, h_norm_denom=None)


def test_explicit_denom_refuses_radius():
    """An explicit denom was chosen under the radius-blind convention."""
    with pytest.raises(ValueError, match="h_norm_include_radius"):
        _preprocess_var(_var(), _cfg(h_norm_denom=216.386), h_norm_mode="max",
                        set_runtime_params=False, h_norm_denom=216.386)


def test_internal_fitted_denominator_is_allowed():
    """The pipeline passes its OWN fitted denom through the same parameter -- allow it.

    `_prepare_runtime_reference` calls _preprocess_var(h_norm_denom=train_h_norm) with the
    radius-AWARE value it just fitted. Guarding on the parameter rather than the config
    field refused that path and killed run 61121 (fno_preheat59_steady_32k_radiusON), the
    training this flag exists to enable.
    """
    from lpno.custom_utils.utils import h_norm_raw

    fitted = float(h_norm_raw(100.0, 0.25, R0))          # what the pipeline would fit
    cfg = _cfg()                                          # h_norm_include_radius=True, no config denom
    out = _preprocess_var(_var(), cfg, h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=fitted)
    assert np.allclose(out["H_norm"], 1.0), (
        f"internal fitted denom was refused or mis-scaled: {float(np.max(out['H_norm'])):.6g}"
    )


def test_missing_radius_refuses_rather_than_dropping_it():
    """No beam_radius in the sample: the numerator must not silently go radius-blind."""
    with pytest.raises(ValueError, match="beam_radius"):
        _preprocess_var(_var(with_radius=False), _cfg(), h_norm_mode="max",
                        set_runtime_params=False, h_norm_denom=None)


def test_radius_off_is_unaffected_by_all_three():
    """The default path must keep working in every one of those situations."""
    off = dict(h_norm_include_radius=False)
    assert _preprocess_var(_var(False), _cfg(**off), h_norm_mode="ref_scale",
                           set_runtime_params=False, h_norm_denom=None) is not None
    out = _preprocess_var(_var(False), _cfg(**off), h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=200.0)
    assert np.allclose(out["H_norm"], 1.0)


def test_npz_dir_store_serves_beam_radius(tmp_path):
    """NpzDirStore must serve the constant path channels, not only P/V_scan."""
    from lpno.models.pipeline.npz_lazy_store import NpzDirStore

    case = tmp_path / "c0.npz"
    np.savez(case, x=np.zeros(2, np.float32), y=np.zeros(2, np.float32),
             z=np.zeros(2, np.float32), time_s=np.zeros(1, np.float32),
             power_w=np.float32(100.0), scan_speed_m_per_s=np.float32(0.25),
             case_name=np.array('c0'),
             **{"Temperature": np.zeros((1, 2, 2, 2), np.float32)})
    (tmp_path / "c0.manifest.json").write_text(json.dumps(
        {"input": {"laser": {"beam_radius_m": R0}, "simulation": {"initial_temperature_k": 400.0}}}))
    store = NpzDirStore(tmp_path)
    assert store.scalar_value("P", 0) == pytest.approx(100.0)
    assert store.scalar_value("beam_radius", 0) == pytest.approx(R0, rel=1e-6)
    with pytest.raises(KeyError):
        store.scalar_value("not_a_channel", 0)
