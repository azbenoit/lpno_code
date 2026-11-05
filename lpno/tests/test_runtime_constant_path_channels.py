"""A staged member can supply the CASE-CONSTANT path channels from its own manifest.

`preheat_T0`, `beam_shape_uniform` and `focus_factor` are classed as path channels, so
`materialize_derived_dataset` drops them: a staged steady variant holds 17 arrays and none
of them. That is fatal on the preheat axis specifically -- without `preheat_T0` two cases
with the same (P, V) and different initial temperature present IDENTICAL inputs with
different targets, so the axis is unlearnable and the run answers a different question (the
2026-08-10 preheat batch is the recorded precedent for exactly that failure).

Unlike the trajectory channels these need no beam kinematics, only the manifest -- which
the staged member already carries under `simulation_metadata.flow3d_case_manifest`, and
which this store already reads for the material payload. So they can be built at LOAD time
without waiting on the materialiser.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lpno.models.pipeline.lazy_runtime import (
    RUNTIME_FORMAT_V2,
    ArrayDirSimulationStore,
)


def _member(tmp_path, *, t0=620.7, dist="uniform", focus=1.3, with_manifest=True):
    """A staged arraydir member: one stored array plus the case manifest, no material block.

    No material payload on purpose -- `_material_manifest` treats its presence as "build
    every material channel", which is a separate code path from the one under test.
    """
    root = tmp_path / "member"
    root.mkdir(parents=True, exist_ok=True)
    T = np.linspace(300.0, 3000.0, 2 * 1 * 3 * 4 * 5, dtype=np.float32).reshape(2, 1, 3, 4, 5)
    np.save(root / "Temperature.npy", T)
    meta = {
        "array_entries": [{"original_name": "Temperature", "stored_name": "Temperature",
                           "filename": "Temperature.npy", "shape": list(T.shape),
                           "dtype": "float32"}],
        "array_aliases": {},
        "format": RUNTIME_FORMAT_V2,
        "source_kind": "simulation",
        "simulation_identifier": "c0",
    }
    if with_manifest:
        meta["simulation_metadata"] = {"flow3d_case_manifest": {
            "input": {
                "simulation": {"initial_temperature_k": t0},
                "laser": {"heat_distribution": dist, "beam_radius_m": 5e-5},
                "_design": {"focus_factor": focus},
            },
        }}
    (root / "manifest.json").write_text(json.dumps(meta))
    return root


def test_preheat_t0_is_built_from_the_member_manifest(tmp_path):
    store = ArrayDirSimulationStore(_member(tmp_path, t0=620.7))
    out = store.read_case(0, time_idx=0, keys=["Temperature", "preheat_T0"])
    assert "preheat_T0" in out, "preheat_T0 must be synthesised, not KeyError"
    assert np.allclose(out["preheat_T0"], 620.7)
    assert out["preheat_T0"].shape == out["Temperature"].shape


def test_preheat_varies_between_members(tmp_path):
    """The whole point of the axis: two members must present DIFFERENT preheat."""
    a = ArrayDirSimulationStore(_member(tmp_path / "a", t0=296.5))
    b = ArrayDirSimulationStore(_member(tmp_path / "b", t0=760.9))
    va = a.read_case(0, time_idx=0, keys=["Temperature", "preheat_T0"])["preheat_T0"]
    vb = b.read_case(0, time_idx=0, keys=["Temperature", "preheat_T0"])["preheat_T0"]
    assert not np.allclose(va, vb)
    assert np.allclose(va, 296.5) and np.allclose(vb, 760.9)


def test_other_case_constant_channels_are_available(tmp_path):
    store = ArrayDirSimulationStore(_member(tmp_path, dist="uniform", focus=1.3))
    out = store.read_case(0, time_idx=0,
                          keys=["Temperature", "beam_shape_uniform", "focus_factor"])
    assert np.allclose(out["beam_shape_uniform"], 1.0)
    assert np.allclose(out["focus_factor"], 1.3)


def test_a_trajectory_channel_still_refuses_clearly(tmp_path):
    """beam_x needs per-frame kinematics this store cannot reconstruct: name that."""
    store = ArrayDirSimulationStore(_member(tmp_path))
    with pytest.raises(KeyError, match="beam_x"):
        store.read_case(0, time_idx=0, keys=["Temperature", "beam_x"])


def test_no_manifest_refuses_rather_than_silently_defaulting(tmp_path):
    """Silently substituting 293.15 K is how a preheat sweep trains on one temperature."""
    store = ArrayDirSimulationStore(_member(tmp_path, with_manifest=False))
    with pytest.raises(KeyError, match="preheat_T0"):
        store.read_case(0, time_idx=0, keys=["Temperature", "preheat_T0"])


def test_unrequested_constants_are_not_injected(tmp_path):
    store = ArrayDirSimulationStore(_member(tmp_path))
    out = store.read_case(0, time_idx=0, keys=["Temperature"])
    assert set(out) == {"Temperature"}


def test_the_two_constant_channel_lists_stay_in_step():
    """lazy_runtime's tuple and npz_datasource's must not drift apart.

    They are two names for one set: the store uses its copy to decide what to split out of
    the array loop, the datasource uses its own to decide what to build. If one gains a
    channel and the other does not, the store either KeyErrors on a stored-array lookup or
    silently drops a requested channel. An earlier preheat bug lived in one of three
    near-identical builders.
    """
    from lpno.custom_utils.npz_datasource import CONSTANT_PATH_CHANNELS
    from lpno.models.pipeline.lazy_runtime import _CONSTANT_PATH_CHANNELS

    assert set(_CONSTANT_PATH_CHANNELS) == set(CONSTANT_PATH_CHANNELS)


def test_constants_are_still_built_by_the_full_path_builder():
    """The extraction must not have changed the full path: same values, same source."""
    import numpy as np

    from lpno.custom_utils.npz_datasource import _build_path_channels

    manifest = {
        "input": {"simulation": {"initial_temperature_k": 543.2},
                  "laser": {"heat_distribution": "uniform", "beam_radius_m": 5e-5,
                            "scan_speed_m_per_s": 0.5},
                  "_design": {"focus_factor": 0.8}},
        "derived": {"laser_path_xy_m": [[0.0, 0.0], [1e-3, 0.0]],
                    "scan_time_s": 2e-3, "laser_off_time_s": 2e-3},
    }
    x = np.linspace(0.0, 1e-3, 4, dtype=np.float64)
    y = np.linspace(-1e-4, 1e-4, 2, dtype=np.float64)
    z = np.linspace(-1e-4, 0.0, 2, dtype=np.float64)
    t = np.linspace(0.0, 2e-3, 3, dtype=np.float64)
    out = _build_path_channels(["preheat_T0", "focus_factor", "beam_shape_uniform"],
                               manifest, x, y, z, t)
    assert np.allclose(out["preheat_T0"], 543.2)
    assert np.allclose(out["focus_factor"], 0.8)
    assert np.allclose(out["beam_shape_uniform"], 1.0)


def test_default_read_includes_constants_the_config_asks_for(tmp_path):
    """The training sample builder calls read_case() with NO keys argument.

    That defaults to `list_array_names()` -- the STORED arrays -- so a synthesised channel
    is never requested and the builder dies later with a bare KeyError from
    `{key: processed[key] for key in str_fno_input_keys}`. Measured: run 58791, the first
    preheat steady arm, exit 300 on `KeyError: 'preheat_T0'` after the store had been
    taught to build it. Teaching the store was necessary and not sufficient; the caller
    has to ask.
    """
    store = ArrayDirSimulationStore(_member(tmp_path, t0=655.0))
    default = store.read_case(0, time_idx=0)
    assert "preheat_T0" in default, (
        "a constant path channel must be served by a keyless read, or the sample builder "
        "never sees it"
    )
    assert np.allclose(default["preheat_T0"], 655.0)


def test_keyless_read_does_not_invent_trajectory_channels(tmp_path):
    """Only the constants are free. beam_x must not appear from nowhere."""
    store = ArrayDirSimulationStore(_member(tmp_path))
    default = store.read_case(0, time_idx=0)
    assert "beam_x" not in default
    assert "path_dose" not in default
