"""`beam_radius` is a conditioning channel, because it VARIES and nothing else carries it.

Measured 2026-08-20 on ti64_preheat60_fixed59 (44 train cases): 44 distinct beam radii and
a 31/13 gaussian/uniform split. Nothing in the input set carried either:

  - `H_norm` is `P / sqrt(|V_scan|)` (preprocessing_datasets.py:278) -- no radius term. The
    physically complete H* in `_h_norm_local` DOES contain r0^3, but that is a different,
    optional channel; the plain H_norm every arm uses does not.
  - `beam_dist` encodes the footprint width spatially, but it is a TRAJECTORY channel and
    `materialize_derived_dataset` drops it, so a steady arm cannot have it.
  - `focus_factor` is constant 1.0 on every case here -- genuinely redundant.

So on the preheat axis two cases could differ by 2x in beam radius and present identical
inputs. Same defect as the missing preheat_T0, and the same consequence: the model learns
the mean over an invisible variable.

Emitted non-dimensionally (r0 / ref_scales["L"]) like every other length-valued channel --
raw metres are ~3.5e-5 and would be numerically invisible next to O(1) inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from lpno.custom_utils.npz_datasource import (
    CONSTANT_PATH_CHANNELS,
    PATH_CHANNEL_NAMES,
    build_constant_path_channels,
)


def _manifest(r0=35.5e-6, dist="gaussian"):
    return {"input": {"laser": {"beam_radius_m": r0, "heat_distribution": dist},
                      "simulation": {"initial_temperature_k": 500.0}}}


def test_beam_radius_is_a_known_channel():
    assert "beam_radius" in PATH_CHANNEL_NAMES
    assert "beam_radius" in CONSTANT_PATH_CHANNELS


def test_beam_radius_is_built_from_the_manifest():
    out = build_constant_path_channels(["beam_radius"], _manifest(r0=35.5e-6), (1, 1, 2, 2, 2))
    assert np.allclose(out["beam_radius"], 35.5e-6)


def test_distinct_radii_give_distinct_channels():
    """The whole point: two cases differing only in radius must differ in inputs."""
    a = build_constant_path_channels(["beam_radius"], _manifest(r0=35.5e-6), (1, 1, 2, 2, 2))
    b = build_constant_path_channels(["beam_radius"], _manifest(r0=70.0e-6), (1, 1, 2, 2, 2))
    assert not np.allclose(a["beam_radius"], b["beam_radius"])


def test_missing_radius_falls_back_to_the_campaign_default():
    out = build_constant_path_channels(["beam_radius"], {"input": {"laser": {}}}, (1, 1, 2, 2, 2))
    assert np.allclose(out["beam_radius"], 5e-5)


def test_it_is_scaled_by_the_length_reference_not_left_in_metres():
    """Raw metres (3.5e-5) would be numerically invisible beside O(1) channels."""
    from types import SimpleNamespace

    from lpno.models.pipeline.preprocessing_datasets import _preprocess_var

    shape = (1, 1, 2, 2, 2)
    var = {
        "T": np.full(shape, 1000.0, dtype=np.float32),
        "alpha.metal": np.ones(shape, dtype=np.float32),
        "P": np.full(shape, 150.0, dtype=np.float32),
        "V_scan": np.full(shape, 0.5, dtype=np.float32),
        "x": np.zeros(shape, dtype=np.float32),
        "y": np.zeros(shape, dtype=np.float32),
        "z": np.zeros(shape, dtype=np.float32),
        "beam_radius": np.full(shape, 35.5e-6, dtype=np.float32),
    }
    # same minimal shape test_preprocessing_velocity_scale established for this function
    run_cfg = SimpleNamespace(
        software="flow3D",
        material_params={"T_s": 1878.0, "T_l": 1923.0},
        pde_weight=1.0,
        sharp_alpha=False,
        alpha_masking=False,
        pde_loss=False,
        vof_pde_loss=False,
        laser_params={},
        ref_scales={"L": 1e-4, "T": 3000.0, "V": 1.0, "P": 200.0, "H": 5.6},
    )
    out = _preprocess_var(var, run_cfg, h_norm_mode="ref_scale", set_runtime_params=False,
                          preserve_time_axis=False)
    assert np.allclose(out["beam_radius"], 35.5e-6 / 1e-4), "must be divided by ref_scales['L']"
    assert 0.01 < float(np.abs(out["beam_radius"]).max()) < 100.0, "must land O(1)"
