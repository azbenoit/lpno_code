"""H_norm must carry the beam radius, because the normalized enthalpy does.

`H_norm` was `P / sqrt(|V_scan|)` -- a P-V proxy. The physical normalized enthalpy, the
same closed form `_h_norm_local` already implements and the batch generator's `h_norm_ti64`
uses, is

    H* = A P / (rho (cp dT + L_f) sqrt(pi D v r0^3))

so it scales as `P / sqrt(v * r0^3)`. Dropping r0 was harmless while every case shared a
radius and fatal once one did not: ti64_preheat60_fixed59 has 44 distinct radii, and with
r0 absent two cases differing only in beam radius produced the SAME H_norm.

OPT-IN since 2026-08-21 via `custom.h_norm_include_radius`, default OFF. The original
claim in this file -- that it was "safe to make unconditional" because a constant radius
cancels against the split max -- was WRONG, and cost a morning of void results. It holds
only when numerator and denominator are computed in the SAME pass. They are not: BUG-04
stashes the denominator at train time and reuses it at eval, and the lazy denominator
reads the radius through `store.scalar_value`, which cannot serve a synthesised
_CONSTANT_PATH_CHANNELS member at all. Both asymmetries were silent.

r0 remains available to the network as its own input channel regardless, so nothing is
lost by default. Turn the flag on for a varying-radius dataset once the denominator can
see the radius on that dataset's store type.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lpno.custom_utils.utils import h_norm_raw
from lpno.models.pipeline.preprocessing_datasets import _preprocess_var


def _run_cfg():
    return SimpleNamespace(
        software="flow3D",
        material_params={"T_s": 1873.0, "T_l": 1923.0},
        pde_weight=1.0, sharp_alpha=False, alpha_masking=False,
        pde_loss=False, vof_pde_loss=False, laser_params={},
        ref_scales={"L": 1e-4, "T": 3000.0, "V": 1.0, "P": 200.0, "H": 5.6},
        h_norm_include_radius=True,
    )


def _run_cfg_default():
    cfg = _run_cfg()
    cfg.h_norm_include_radius = False
    return cfg


def _var(P, V, r0=None):
    shape = (1, 1, 2, 2, 2)
    v = {
        "T": np.full(shape, 1000.0, dtype=np.float32),
        "alpha.metal": np.ones(shape, dtype=np.float32),
        "P": np.full(shape, P, dtype=np.float32),
        "V_scan": np.full(shape, V, dtype=np.float32),
        "x": np.zeros(shape, dtype=np.float32),
        "y": np.zeros(shape, dtype=np.float32),
        "z": np.zeros(shape, dtype=np.float32),
    }
    if r0 is not None:
        v["beam_radius"] = np.full(shape, r0, dtype=np.float32)
    return v


def _h(P, V, r0=None, denom=None):
    out = _preprocess_var(_var(P, V, r0), _run_cfg(), h_norm_mode="max",
                          set_runtime_params=False, preserve_time_axis=False,
                          h_norm_denom=denom)
    return float(np.asarray(out["H_norm"]).reshape(-1)[0])


def _num(P, V, r0=None):
    """The raw numerator, via the shared helper both sides now use.

    Formula checks go through h_norm_raw rather than `_preprocess_var(..., denom=1.0)`:
    since the 2026-08-21 guard, combining h_norm_include_radius=True with an EXPLICIT
    denominator is refused, because such a denominator was always chosen radius-blind.
    Pinned by tests/test_h_norm_radius_consistency.py.
    """
    return float(np.asarray(h_norm_raw(P, V, r0)).reshape(-1)[0])


def test_radius_enters_h_norm():
    """Same P and V, different radius -> different H_norm. This is the whole point."""
    a = _num(150.0, 0.5, r0=35.5e-6)
    b = _num(150.0, 0.5, r0=71.0e-6)
    assert not np.isclose(a, b), "beam radius must change H_norm"
    # r0^-3/2: doubling the radius drops H* by 2^1.5
    assert np.isclose(a / b, 2.0 ** 1.5, rtol=1e-3)


def test_a_constant_radius_cancels_under_the_shared_denominator():
    """A constant radius cancels -- but ONLY when both sides are fitted in one pass.

    This is the claim that was over-read into "safe to make unconditional" and cost a
    morning of void results. It holds here because `_preprocess_var` fits the denominator
    on the same array it just built. It does NOT hold across a stash boundary, an explicit
    denominator, or a ref_scale denominator -- all three are now refused outright.
    """
    cfg_off = _run_cfg()
    cfg_off.h_norm_include_radius = False
    without = float(np.asarray(_preprocess_var(          # legacy P/sqrt(V), self-normalised
        _var(150.0, 0.5), cfg_off, h_norm_mode="max", set_runtime_params=False,
        preserve_time_axis=False, h_norm_denom=None)["H_norm"]).reshape(-1)[0])
    with_r = _h(150.0, 0.5, r0=50e-6)              # radius present, self-normalised
    assert np.isclose(without, with_r, rtol=1e-6), "a constant radius must cancel"


def test_absent_radius_is_refused_not_silently_dropped():
    """No beam_radius with the flag on must RAISE.

    It used to fall back to the legacy form silently, which is the inverted half of the
    original defect: a radius-blind numerator over a radius-aware denominator.
    """
    import pytest

    with pytest.raises(ValueError, match="beam_radius"):
        _h(150.0, 0.5)  # _run_cfg() sets h_norm_include_radius=True


def test_scaling_in_p_and_v_is_unchanged():
    assert np.isclose(_num(300.0, 0.5, r0=50e-6) / _num(150.0, 0.5, r0=50e-6), 2.0, rtol=1e-6)
    assert np.isclose(_num(150.0, 2.0, r0=50e-6) / _num(150.0, 0.5, r0=50e-6), 0.5, rtol=1e-6)
