"""The melt window must follow the material, not a hardcoded Ti64 pair.

ck300 spans liquidus 868-1898 K. Scoring all 300 against Ti64's 1873/1923 makes
the fl target near-empty for every low-melting material: on ckc_072 (T_l=868 K)
FLOW-3D's own Melt Region has 1080 cells in a mid-y slice and the Ti64 window
finds 4. A model that predicts nothing then scores well, which is what
fno_ck300_lab_32k_cond12 (69617) did -- fl IoU 0.0015 with a sane T field.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from lpno.custom_utils.utils import smooth_liquid_fraction_numpy
from lpno.models.pipeline.preprocessing_datasets import _preprocess_var

SHP = (1, 1, 2, 2, 2)


def _var(**extra):
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
    v.update(extra)
    return v


def _run_cfg(**over):
    base = dict(
        software="flow3D", sharp_alpha=False, alpha_masking=False,
        material_params={"T_s": 1873.0, "T_l": 1923.0},
        laser_params={},
        ref_scales={"L": 1e-4, "T": 3000.0, "V": 0.1, "P": 200.0, "Rho": 4000.0,
                    "mat_Ts": 1800.0, "mat_Tl": 1800.0,
                    "mat_Tv": 3000.0},
        pde_loss=False, vof_pde_loss=False, use_2d=False,
        time_as_batch=False, include_time=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_array_valued_window_broadcasts():
    """A per-cell window must work; today float() on an array raises."""
    T = np.array([900.0, 900.0], dtype=np.float32)
    Ts = np.array([825.0, 1873.0], dtype=np.float32)   # two different materials
    Tl = np.array([868.0, 1923.0], dtype=np.float32)
    fl = smooth_liquid_fraction_numpy(T, Ts, Tl)
    assert fl[0] > 0.9, "900 K is well above an 868 K liquidus -> molten"
    assert fl[1] < 0.1, "900 K is far below a 1873 K solidus -> solid"


def test_preprocess_uses_per_case_window_when_channels_present():
    """With mat_Ts/mat_Tl in var, fl must follow THAT material, not run_cfg."""
    T = 1000.0  # molten for ckc_072 (825/868), solid for Ti64 (1873/1923)
    var = _var(
        Temperature=np.full(SHP, T, np.float32),
        mat_Ts=np.full(SHP, 825.0, np.float32),
        mat_Tl=np.full(SHP, 868.0, np.float32),
    )
    out = _preprocess_var(var, _run_cfg(), h_norm_mode="max", set_runtime_params=False)
    assert out["fl"].mean() > 0.9, (
        f"1000 K against an 868 K liquidus must read molten; got {out['fl'].mean():.3f} "
        "(Ti64 window would give ~0)")


def test_preprocess_falls_back_to_run_cfg_without_channels():
    """No per-case channels -> old behaviour, so existing runs are unchanged."""
    var = _var(Temperature=np.full(SHP, 1000.0, np.float32))
    out = _preprocess_var(var, _run_cfg(), h_norm_mode="max", set_runtime_params=False)
    assert out["fl"].mean() < 0.1, "Ti64 window: 1000 K is solid"


def test_fill_temperature_is_per_case_when_mat_Tv_present():
    """The alpha-mask fill must follow the material, not Ti64's 3520 K.

    ckc_072 vaporises at 2068 K; filling its gas with 3520 K both mislabels the
    gas layer and strengthens the "hot means gas" association that leaves
    predicted melt empty inside metal.
    """
    var = _var(
        Temperature=np.full(SHP, 1000.0, np.float32),
        mat_Tv=np.full(SHP, 2068.0, np.float32),
    )
    var["Fraction Of Fluid"] = np.zeros(SHP, np.float32)  # all gas -> all filled
    out = _preprocess_var(var, _run_cfg(alpha_masking=True, material_params={"T_s": 1873.0, "T_l": 1923.0, "T_boil": 3520.0}),
                          h_norm_mode="max", set_runtime_params=False)
    filled = out["T"].mean() * 3000.0
    assert abs(filled - 2068.0) < 60.0, f"expected ~2068 K per-case fill, got {filled:.0f} K"
