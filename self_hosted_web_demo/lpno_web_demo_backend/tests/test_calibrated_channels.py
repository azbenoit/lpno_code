"""The five extra input channels the calibrated ("newphys"/spkh123) models need.

Every one is produced by the SUBMODULE's own builder, never re-derived here: a
local ``conduction_field`` or ``h_norm_local`` that drifted from the training
definition would render a confident, wrong field with no error anywhere.

Which means the pin matters. ``ext/lpno`` is a pinned submodule, not
a view of the live repo, and at b196efb (2026-07-30) it carries 5 path channels
against upstream's 36 today. So the adapter advertises the calibrated channels
only when the module it actually loads can build them -- otherwise a 16-channel
model would pass the contract check and then KeyError mid-request.

These tests deliberately resolve the datasource THROUGH THE ADAPTER
(``adapter._npz_datasource()``) rather than importing ``lpno``
directly. An earlier version of this file imported it directly, picked up the
live repo instead of the submodule, and passed while the served code could not
build a single one of the five.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.adapter_randompath import (
    CALIBRATED_CHANNELS,
    DRAWN_FOCUS_FACTOR,
    DRAWN_HEAT_DISTRIBUTION,
    DRAWN_INITIAL_T_K,
    DRAWN_SURFACE_Z_M,
    REF_T,
    RandomPathAdapter,
    _synthetic_manifest,
)
from app.config import Settings
from tests.conftest import needs_served_gt

DRAWN = [[-4.0e-4, -1.0e-4], [4.0e-4, 1.0e-4]]


@pytest.fixture(scope="module")
def adapter() -> RandomPathAdapter:
    return RandomPathAdapter(Settings(startup_load_runtime=False, use_stub_runtime=False))


def _pinned_channels(adapter) -> set[str]:
    return set(adapter._npz_datasource().PATH_CHANNEL_NAMES)


def test_supported_keys_never_promise_more_than_the_pin_can_build(adapter) -> None:
    """The contract check must refuse what it cannot build, not admit and crash.

    This is the invariant that survives a submodule bump in either direction.
    """
    supported = adapter.supported_input_keys()
    buildable = _pinned_channels(adapter)
    over_promised = sorted((supported & set(CALIBRATED_CHANNELS)) - buildable)
    assert not over_promised, (
        f"advertised {over_promised} but the pinned submodule cannot build them; "
        "a model using one would KeyError mid-request"
    )


def test_calibrated_channels_are_admitted_exactly_when_buildable(adapter) -> None:
    supported = adapter.supported_input_keys()
    buildable = _pinned_channels(adapter)
    for chan in CALIBRATED_CHANNELS:
        assert (chan in supported) == (chan in buildable), chan


def test_synthetic_manifest_pins_the_physics_the_builders_read() -> None:
    """A drawn path has no case file, so these values come from us.

    Read off a real calibrated case (runs/batches/ti64_rand150_newphys_20260817/
    runs/r150np_000/case_manifest.json): initial_temperature_k 293.15, with
    heat_distribution and _design.focus_factor ABSENT -- so the builders' own
    defaults are the trained values. Pinned explicitly so a change to an upstream
    default cannot silently move what the demo serves.
    """
    m = _synthetic_manifest(np.asarray(DRAWN, dtype=float), 1.0e-3, v_scan_mps=0.6, p_watts=100.0)
    assert m["input"]["simulation"]["initial_temperature_k"] == DRAWN_INITIAL_T_K == 293.15
    assert m["input"]["laser"]["heat_distribution"] == DRAWN_HEAT_DISTRIBUTION == "gaussian"
    assert m["input"]["_design"]["focus_factor"] == DRAWN_FOCUS_FACTOR == 1.0
    assert m["derived"]["surface_z_m"] == DRAWN_SURFACE_Z_M == 0.0


def test_pinned_values_match_the_builder_defaults_they_replace(adapter) -> None:
    """Pinning must make the physics EXPLICIT without CHANGING it.

    If a builder default ever diverges from what we pin, that is a real
    behaviour change and this is where it should surface.
    """
    nds = adapter._npz_datasource()
    if not hasattr(nds, "build_constant_path_channels"):
        pytest.skip("pinned submodule has no build_constant_path_channels")
    got = nds.build_constant_path_channels(
        {"preheat_T0", "beam_shape_uniform", "focus_factor"}, {}, (1, 1, 1, 1, 1, 1)
    )
    assert float(got["preheat_T0"].ravel()[0]) == pytest.approx(DRAWN_INITIAL_T_K)
    assert float(got["focus_factor"].ravel()[0]) == pytest.approx(DRAWN_FOCUS_FACTOR)
    assert float(got["beam_shape_uniform"].ravel()[0]) == pytest.approx(0.0)  # "gaussian"


@needs_served_gt(Settings().path_data_dir)
def test_channels_build_end_to_end_once_the_pin_supports_them(adapter) -> None:
    """The real check, active as soon as the submodule is bumped.

    preheat_T0 is divided by ref_scales["T"] like every Kelvin-valued channel
    (preprocessing_datasets.py:538-544); the other four pass through untouched --
    a 0/1 flag, an O(1) design factor, a ratio that is 1.0 in unsoaked material,
    and a field the builder already normalises to [0, 1].
    """
    buildable = _pinned_channels(adapter)
    keys = [c for c in ("preheat_T0", "beam_shape_uniform", "focus_factor", "h_norm_local")
            if c in buildable]
    if not keys:
        pytest.skip("pinned submodule cannot build the calibrated channels yet")

    case = adapter.resolve(points_xy_m=DRAWN, preset=None, physics=None)
    ch = adapter._build_invar(case, keys)
    if "preheat_T0" in keys:
        assert np.allclose(ch["preheat_T0"], DRAWN_INITIAL_T_K / REF_T)
    if "beam_shape_uniform" in keys:
        assert np.allclose(ch["beam_shape_uniform"], 0.0)
    if "focus_factor" in keys:
        assert np.allclose(ch["focus_factor"], 1.0)
    if "h_norm_local" in keys:
        assert float(ch["h_norm_local"].min()) >= 1.0 - 1e-6
        assert float(ch["h_norm_local"].max()) < 2.0
