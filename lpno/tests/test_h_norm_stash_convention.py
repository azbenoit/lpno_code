"""A stashed H_norm denominator must be used with the numerator convention it was fitted with.

The r0^3 term was added to the H_norm numerator (c2e66c4) with the justification that a
CONSTANT radius "cancels exactly", because H_norm is divided by its own max over the
split. That holds only when numerator and denominator are computed in the SAME pass.

It is false across the stash boundary. BUG-04 makes the authoritative train pass fit the
denominator once and stash it on the runtime params, so every val/eval pass of a model
trained BEFORE the r0 change divides a radius-CARRYING numerator by a radius-BLIND
stashed denominator. The leftover factor is 1/sqrt(r0^3) -- 2.83e6 at r0 = 50 um --
and it is silent: no error, just a network input six orders of magnitude off.

Measured 2026-08-21: re-evaluating model 57990 (trained 08-20 14:58, denominator fitted
radius-blind) fed in_H_norm = 1.28e6 and returned a T MAE of 9.6e7 K.
"""

from types import SimpleNamespace

import numpy as np

from lpno.models.pipeline.preprocessing_datasets import _preprocess_var

SHP = (1, 1, 2, 2, 2)
R0 = 50e-6


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


def _run_cfg(**over):
    base = dict(
        software="flow3D",
        sharp_alpha=False,
        alpha_masking=False,
        material_params={"T_s": 1878.0, "T_l": 1928.0},
        laser_params={},
        ref_scales={"L": 1e-4, "T": 3000.0, "V": 0.1, "P": 200.0, "Rho": 4000.0},
        pde_loss=False,
        vof_pde_loss=False,
        use_2d=False,
        time_as_batch=False,
        include_time=False,
        h_norm_include_radius=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_legacy_radius_blind_stash_is_honoured():
    """A stash with no convention marker is legacy (radius-blind): H_norm must stay O(1)."""
    # 100/sqrt(0.25) = 200, the value a pre-c2e66c4 train pass would have fitted.
    cfg = _run_cfg(laser_params={"h_norm_denom": 200.0})
    out = _preprocess_var(_var(), cfg, h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=None)
    assert np.allclose(out["H_norm"], 1.0), (
        f"legacy stash mis-scaled: H_norm={float(np.max(out['H_norm'])):.6g}, "
        "expected 1.0 (the 1/sqrt(r0^3) factor leaked through)"
    )


def test_radius_aware_stash_round_trips():
    """A denom stashed by a radius-aware train pass must be reused radius-aware.

    Simulates the train pass by writing the stash the way _preprocess_var does, rather
    than calling it with set_runtime_params=True (which needs a full param_dir run_cfg).
    """
    from lpno.custom_utils.utils import h_norm_raw

    fitted = float(np.max(h_norm_raw(100.0, 0.25, R0)))
    stash = {"h_norm_denom": fitted, "h_norm_denom_radius_aware": True}
    out = _preprocess_var(_var(), _run_cfg(laser_params=stash), h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=None)
    assert np.allclose(out["H_norm"], 1.0), (
        f"radius-aware stash did not round-trip: {float(np.max(out['H_norm'])):.6g}"
    )


def test_explicit_denom_is_refused_with_radius_on():
    """An explicit denom + the King term is now REFUSED, not silently reconciled.

    This file originally pinned the explicit denominator as "current convention", on the
    reasoning that it is supplied deliberately per run. The 2026-08-21 audit showed that
    is wrong in practice: every run that sets custom.h_norm_denom (the alsi_* AL family,
    216.386) chose it under the radius-BLIND convention, so honouring the flag there
    restores the exact 1/sqrt(r0^3) leak the flag exists to prevent. Refusing is the only
    reading that cannot silently produce a wrong number.
    """
    import pytest

    from lpno.custom_utils.utils import h_norm_raw

    explicit = 4.0 * float(h_norm_raw(100.0, 0.25, R0))
    # The guard keys on the CONFIG field (user intent), not the parameter -- the pipeline
    # passes its own fitted denom through the parameter and that path must stay open.
    with pytest.raises(ValueError, match="h_norm_include_radius"):
        _preprocess_var(_var(), _run_cfg(h_norm_denom=explicit), h_norm_mode="max",
                        set_runtime_params=False, h_norm_denom=explicit)


def test_explicit_denom_still_wins_with_radius_off():
    """The default path keeps the explicit-override behaviour unchanged."""
    cfg = _run_cfg()
    cfg.h_norm_include_radius = False
    out = _preprocess_var(_var(with_radius=False), cfg, h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=800.0)
    assert np.allclose(out["H_norm"], 0.25)


def test_radius_is_off_by_default():
    """Without `custom.h_norm_include_radius`, H_norm is the legacy P/sqrt(V).

    This is the property that makes every model trained before the King-form change
    evaluate correctly, and it is why the flag defaults off.
    """
    cfg = _run_cfg()
    cfg.h_norm_include_radius = False
    out = _preprocess_var(_var(), cfg, h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=200.0)
    assert np.allclose(out["H_norm"], 1.0), (
        f"default path is not radius-blind: {float(np.max(out['H_norm'])):.6g}"
    )
