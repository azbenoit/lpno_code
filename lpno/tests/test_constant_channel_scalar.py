"""A case-CONSTANT path channel must be readable as a SCALAR, not only via read_case.

`beam_radius`, `preheat_T0`, `beam_shape_uniform` and `focus_factor` are
_CONSTANT_PATH_CHANNELS: synthesised from the case manifest inside read_case, never
stored as arrays. Every other read path went through `get_array`, so
`store.scalar_value("beam_radius", ...)` raised KeyError.

That asymmetry is the whole 2026-08-21 H_norm defect. `compute_h_norm_denom` needs the
radius as a cheap scalar, got the KeyError, and a bare `except Exception` turned it into
a silently radius-blind denominator while the numerator (which reads the frame) kept the
radius -- leaving 1/sqrt(r0^3) = 2.83e6 in the network input.

Removing the silent except made it fail loudly instead, which is better but left
`custom.h_norm_include_radius=true` unusable: measured 2026-08-21 on
ti64_preheat60_fixed59 (44 distinct radii, the dataset the King-form term exists for),
turning the flag on died with KeyError: 'beam_radius'. So the scalar path must actually
serve these channels.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lpno.models.pipeline import lazy_runtime


def _make_member(root, radius_m=45e-6, preheat_k=500.0):
    root.mkdir(parents=True, exist_ok=True)
    arr = np.full((3, 2, 2, 2), 7.0, dtype=np.float32)
    np.save(root / "Temperature.npy", arr)
    np.save(root / "P.npy", np.float32(120.0))
    np.save(root / "V_scan.npy", np.float32(0.4))
    (root / "manifest.json").write_text(json.dumps({
        "format": lazy_runtime.RUNTIME_FORMAT_V2,
        "source_kind": "simulation",
        "array_entries": [
            {"original_name": "Temperature", "filename": "Temperature.npy",
             "shape": [3, 2, 2, 2]},
            {"original_name": "P", "filename": "P.npy", "shape": []},
            {"original_name": "V_scan", "filename": "V_scan.npy", "shape": []},
        ],
        "simulation_metadata": {
            "flow3d_case_manifest": {
                "input": {
                    "laser": {"beam_radius_m": radius_m, "power_w": 120.0,
                              "scan_speed_m_per_s": 0.4},
                    "simulation": {"initial_temperature_k": preheat_k},
                }
            }
        },
    }))
    return root


def test_scalar_value_serves_beam_radius(tmp_path):
    store = lazy_runtime.ArrayDirSimulationStore(_make_member(tmp_path / "sim"))
    got = store.scalar_value("beam_radius", 0)
    assert got == pytest.approx(45e-6, rel=1e-6), (
        f"beam_radius not served as a scalar: {got!r}"
    )


def test_stored_arrays_still_win(tmp_path):
    """The fallback must not shadow a genuinely stored array."""
    store = lazy_runtime.ArrayDirSimulationStore(_make_member(tmp_path / "sim"))
    assert store.scalar_value("P", 0) == pytest.approx(120.0)


def test_preheat_t0_also_served(tmp_path):
    """The fix is for the whole _CONSTANT_PATH_CHANNELS family, not just beam_radius."""
    store = lazy_runtime.ArrayDirSimulationStore(_make_member(tmp_path / "sim"))
    assert store.scalar_value("preheat_T0", 0) == pytest.approx(500.0)


def test_unknown_name_still_raises(tmp_path):
    """A name that is neither stored nor a constant channel must still fail loudly."""
    store = lazy_runtime.ArrayDirSimulationStore(_make_member(tmp_path / "sim"))
    with pytest.raises(KeyError):
        store.scalar_value("not_a_channel", 0)


def test_denominator_can_include_radius(tmp_path):
    """compute_h_norm_denom(include_radius=True) must work, not KeyError.

    This is the exact call that died on ti64_preheat60_fixed59.
    """
    from lpno.custom_utils.utils import h_norm_raw

    store = lazy_runtime.ArrayDirSimulationStore(_make_member(tmp_path / "sim"))
    denom = lazy_runtime.compute_h_norm_denom(
        store, [0], time_as_batch=False, include_radius=True)
    # float32: the channel is synthesised as float32, so the round-trip loses ~1e-7 rel.
    assert denom == pytest.approx(float(h_norm_raw(120.0, 0.4, 45e-6)), rel=1e-6)
