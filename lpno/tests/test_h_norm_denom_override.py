"""The explicit `custom.h_norm_denom` override must pin the H_norm scale.

Active-learning runs retrain on a growing train set and evaluate on candidate
grids whose P/sqrt(V) maxima differ from the train split's, so the data-fitted
denominator changes between runs and the same physical (P, V) maps to a
different network input. An explicit denom keeps the scale constant across the
whole experiment (BUG-04 anticipated this: "an explicit h_norm_denom already
share a denom").
"""

from types import SimpleNamespace

import numpy as np

from lpno.models.pipeline.preprocessing_datasets import (
    _effective_h_norm_denom,
    _preprocess_var,
)

SHP = (1, 1, 2, 2, 2)


def _var():
    return {
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
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_explicit_denom_wins_over_data_max():
    # P/sqrt(V) = 100/0.5 = 200; explicit denom 800 -> H_norm = 0.25 (not 1.0).
    out = _preprocess_var(_var(), _run_cfg(), h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=800.0)
    assert np.allclose(out["H_norm"], 0.25)


def test_data_max_fallback_and_stash_reuse():
    out = _preprocess_var(_var(), _run_cfg(), h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=None)
    assert np.allclose(out["H_norm"], 1.0)  # fitted max on this split
    stashed = _run_cfg(laser_params={"h_norm_denom": 400.0})
    out = _preprocess_var(_var(), stashed, h_norm_mode="max",
                          set_runtime_params=False, h_norm_denom=None)
    assert np.allclose(out["H_norm"], 0.5)  # train-pass stash reused


def test_effective_denom_prefers_run_cfg_over_store():
    class _Store:
        def has_time_axis(self):
            return False

        def scalar_value(self, key, case_idx, time_idx=None):
            return {"P": 100.0, "V_scan": 0.25}[key]

    run_cfg = _run_cfg(h_norm_denom=216.386)
    assert _effective_h_norm_denom(run_cfg, _Store(), [0]) == 216.386
    run_cfg = _run_cfg(h_norm_denom=None)
    assert _effective_h_norm_denom(run_cfg, _Store(), [0]) == 200.0  # 100/sqrt(0.25)
