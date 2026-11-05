"""Material-aware H_norm: opt-in, and both sides must use the SAME form.

2026-08-21: the radius term went into the NUMERATOR only and `compute_h_norm_denom`
stayed blind, feeding the network in_H_norm = 2.5e6 instead of 0.886 (exactly
1/sqrt(r0^3)). The eager path self-cancels so nothing looked wrong. These tests pin
the symmetry directly rather than trusting either side alone.
"""
import numpy as np
import pytest

from lpno.custom_utils.utils import h_norm_raw, king_material_factor


def test_material_factor_divides_out_material_variation():
    """Two materials, same laser: the material-aware form must separate them."""
    soft = king_material_factor(rho=3292.0, cp=630.0, k=21.1, latent=2.0e5, T_l=868.0, T_0=293.0, A=0.35)
    hard = king_material_factor(rho=4506.0, cp=700.0, k=7.0,  latent=3.45e5, T_l=1923.0, T_0=293.0, A=0.35)
    h_soft = h_norm_raw(200.0, 1.0, material_factor=soft)
    h_hard = h_norm_raw(200.0, 1.0, material_factor=hard)
    assert h_soft > h_hard, "the easier-to-melt material must read a HIGHER normalized enthalpy"
    blind = h_norm_raw(200.0, 1.0)
    assert np.isclose(h_soft, blind / soft) and np.isclose(h_hard, blind / hard)


def test_flag_off_is_bit_identical():
    """Default must not perturb any existing run."""
    assert h_norm_raw(200.0, 1.0, None, None) == h_norm_raw(200.0, 1.0)


def test_numerator_and_denominator_agree():
    """The split max of the numerator, divided by the fitted denom, must be 1.0.

    This is the 2026-08-21 invariant: if either side gains a term the other lacks,
    the ratio leaves 1.0 by exactly that factor.
    """
    from lpno.models.pipeline.lazy_runtime import compute_h_norm_denom

    cases = [(200.0, 1.0, 3292.0, 630.0, 21.1, 2.0e5, 868.0),
             (150.0, 0.5, 4506.0, 700.0, 7.0, 3.45e5, 1923.0)]

    class _Store:
        def has_time_axis(self): return False
        def time_steps(self, case_idx=0): return 1
        def scalar_value(self, name, case_idx, time_idx=None):
            P, V, rho, cp, k, lf, Tl = cases[case_idx]
            return {"P": P, "V_scan": V, "mat_rho": rho, "mat_cp": cp, "mat_k": k,
                    "mat_latent": lf, "mat_Tl": Tl, "mat_A": 0.35}[name]

    denom = compute_h_norm_denom(_Store(), [0, 1], time_as_batch=False, include_material=True)
    numers = [float(h_norm_raw(P, V, material_factor=king_material_factor(
        rho=rho, cp=cp, k=k, latent=lf, T_l=Tl, T_0=293.0, A=0.35)))
        for (P, V, rho, cp, k, lf, Tl) in cases]
    assert np.isclose(max(numers) / denom, 1.0, rtol=1e-9), (
        f"numerator max {max(numers):.6g} / denom {denom:.6g} != 1.0 -- the two sides disagree")


def test_numerator_refuses_without_the_material_channels():
    """Missing channels must raise, not fall back to a blind numerator.

    A silent fallback would leave the denominator material-aware and the numerator
    blind -- the 2026-08-20 failure with the sign flipped.
    """
    from types import SimpleNamespace
    from lpno.models.pipeline.preprocessing_datasets import _preprocess_var

    SHP = (1, 1, 2, 2, 2)
    var = {
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
    cfg = SimpleNamespace(
        software="flow3D", sharp_alpha=False, alpha_masking=False,
        material_params={"T_s": 1873.0, "T_l": 1923.0}, laser_params={},
        ref_scales={"L": 1e-4, "T": 3000.0, "V": 0.1, "P": 200.0, "Rho": 4000.0},
        pde_loss=False, vof_pde_loss=False, use_2d=False,
        time_as_batch=False, include_time=False, h_norm_include_material=True,
    )
    with pytest.raises(ValueError, match="h_norm_include_material"):
        _preprocess_var(var, cfg, h_norm_mode="max", set_runtime_params=False)
