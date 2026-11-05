"""A material channel with no ref scale must be refused, not silently unscaled.

Regression for 68893: 32k steps, exit 0, and a model that never reached melt
(peak 1211 K against an 1898 K threshold) because every mat_* channel was
divided by a default of 1.0 and entered the lifting layer as a raw SI value.
"""
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_missing_material_scale_raises():
    var = _var(mat_latent=np.full(SHP, 3.0e5, np.float32))
    with pytest.raises(ValueError, match="mat_latent"):
        _preprocess_var(var, _run_cfg(), h_norm_mode="max", set_runtime_params=False)


def test_present_scale_divides_to_order_one():
    cfg = _run_cfg(ref_scales={**_run_cfg().ref_scales, "mat_latent": 3.0e5})
    out = _preprocess_var(
        _var(mat_latent=np.full(SHP, 3.0e5, np.float32)),
        cfg, h_norm_mode="max", set_runtime_params=False,
    )
    assert np.allclose(out["mat_latent"], 1.0)


def test_served_but_undeclared_channel_is_not_refused():
    """The store serves the manifest six on every keyless read; a config that
    never asked for them must not be refused for lacking their scales.

    Evals 75405/75410 (m57990/m58734, single-material era) died on
    `material channels ['mat_rho', ...] have no custom.ref_scales entry` without
    requesting a single material channel: every production read of the arraydir
    store is keyless, so `wanted = material_requested or MATERIAL_CHANNEL_NAMES`
    injected all six and the guard read "served" as "used".
    """
    cfg = _run_cfg(keys=SimpleNamespace(str_fno_input_keys=["T", "P", "V_scan"]))
    out = _preprocess_var(
        _var(mat_rho=np.full(SHP, 4420.0, np.float32)),
        cfg, h_norm_mode="max", set_runtime_params=False,
    )
    # Left raw rather than divided by a silent 1.0 -- it never reaches the network.
    assert np.allclose(out["mat_rho"], 4420.0)


def test_declared_channel_without_scale_still_raises():
    """68893 protection survives the relaxation: ASKING for a channel with no
    scale is still refused."""
    cfg = _run_cfg(keys=SimpleNamespace(str_fno_input_keys=["T", "mat_latent"]))
    with pytest.raises(ValueError, match="mat_latent"):
        _preprocess_var(
            _var(mat_latent=np.full(SHP, 3.0e5, np.float32)),
            cfg, h_norm_mode="max", set_runtime_params=False,
        )
