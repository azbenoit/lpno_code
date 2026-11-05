"""`stride` as a first-class derivation, materialized physically.

`PINO_LAZY_STRIDE` strides *after* a full-resolution zlib inflate, so it saves
GPU memory and (measured 2026-07-28) **zero loader time**: a 1601 run sat at
2.5 s/iteration with the GPU at 0% and the dataloader workers pegged, because
every `__getitem__` inflates 4.77 GB/case before decimating. Striding into the
stored arrays instead cuts the inflate itself (4.77 -> 1.19 GB/case).

Making it a derivation rather than an env var also removes the documented trap:
pointing at a pre-strided copy while `PINO_LAZY_STRIDE` is still set strides
twice and silently trains on 38x25x40 instead of 75x50x40.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.utils.dataset_derivation import (
    DEFAULT_DERIVATION_PROFILE,
    apply_dataset_layout_to_training_profile,
    build_derivation_profile_from_training_profile,
    validate_derivation_profile,
)
from aiida_pino.utils.stride_npz import stride_arrays


# --------------------------------------------------------------------------- #
# Profile plumbing
# --------------------------------------------------------------------------- #
def test_stride_defaults_to_no_op():
    assert DEFAULT_DERIVATION_PROFILE["stride"] == [1, 1, 1]
    assert validate_derivation_profile({})["stride"] == [1, 1, 1]


def test_stride_is_normalized_to_a_list_of_three_ints():
    assert validate_derivation_profile({"stride": (2, 2, 1)})["stride"] == [2, 2, 1]


@pytest.mark.parametrize("bad", [[2, 2], [2, 2, 1, 1], [0, 1, 1], [-2, 1, 1], "2,2,1", 2])
def test_unusable_stride_is_rejected(bad):
    """A silent fallback here would train at a resolution nobody asked for."""
    with pytest.raises(ValueError):
        validate_derivation_profile({"stride": bad})


def test_stride_changes_the_derivation_identity():
    """Two strides must not collide, or a run reuses the wrong resolution."""
    a = validate_derivation_profile({"stride": [1, 1, 1]})
    b = validate_derivation_profile({"stride": [2, 2, 1]})
    assert a != b


def test_lazy_stride_env_becomes_derivation_profile_stride():
    profile = build_derivation_profile_from_training_profile(
        {
            "include_time": True,
            "time_as_batch": True,
            "time_as_dim": False,
            "env": {"PINO_LAZY_STRIDE": "2,2,1"},
        }
    )
    assert profile["stride"] == [2, 2, 1]


def test_strided_dataset_layout_removes_lazy_stride_env():
    training_profile = {
        "include_time": True,
        "time_as_batch": True,
        "time_as_dim": False,
        "env": {"PINO_LAZY_STRIDE": "2,2,1", "PINO_LAZY_CACHE": "1"},
        "hydra_overrides": [],
    }
    metadata = {
        "time_axis": True,
        "time_axis_mode": {"include_time": True, "time_as_batch": True, "time_as_dim": False},
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "avg_window_size": 1,
        "super_res_factor": 1,
        "stride": [2, 2, 1],
    }

    resolved = apply_dataset_layout_to_training_profile(training_profile, metadata)

    assert resolved["env"] == {"PINO_LAZY_CACHE": "1"}


# --------------------------------------------------------------------------- #
# The decimation itself: must match `PINO_LAZY_STRIDE` exactly, or a model
# trained on the derived copy is not comparable to one trained with the env var.
# --------------------------------------------------------------------------- #
def _case(nt=4, nx=6, ny=4, nz=3):
    rng = np.random.default_rng(0)
    return {
        "x": np.arange(nx, dtype=np.float64),
        "y": np.arange(ny, dtype=np.float64),
        "z": np.arange(nz, dtype=np.float64),
        "time_s": np.arange(nt, dtype=np.float64),
        "Temperature": rng.random((nt, nx, ny, nz)),
        "Velocity": rng.random((nt, nx, ny, nz, 3)),
        "power": np.array(250.0),
    }


def test_axes_and_fields_are_decimated_start_at_zero():
    src = _case()
    out = stride_arrays(src, (2, 2, 1))
    assert np.array_equal(out["x"], src["x"][::2])
    assert np.array_equal(out["y"], src["y"][::2])
    assert np.array_equal(out["z"], src["z"][::1])
    assert np.array_equal(out["Temperature"], src["Temperature"][:, ::2, ::2, ::1])


def test_vector_fields_keep_their_component_axis():
    src = _case()
    out = stride_arrays(src, (2, 2, 1))
    assert out["Velocity"].shape == (4, 3, 2, 3, 3)
    assert np.array_equal(out["Velocity"], src["Velocity"][:, ::2, ::2, ::1])


def test_time_and_scalars_pass_through_untouched():
    """Spatial stride must not touch the time axis or per-case scalars."""
    src = _case()
    out = stride_arrays(src, (2, 2, 1))
    assert np.array_equal(out["time_s"], src["time_s"])
    assert np.array_equal(out["power"], src["power"])


def test_unit_stride_is_the_identity():
    src = _case()
    out = stride_arrays(src, (1, 1, 1))
    for key, value in src.items():
        assert np.array_equal(out[key], value), key


def test_canonical_simulation_arrays_are_physically_strided(pino_simulation_factory):
    from aiida_pino.data import derive_simulation_arrays

    source = np.arange(2 * 1 * 6 * 4 * 3, dtype=np.float32).reshape(2, 1, 6, 4, 3)
    simulation = pino_simulation_factory(
        "stride-canonical",
        arrays={
            "Temperature": source,
            "x": np.broadcast_to(np.arange(6, dtype=np.float32)[None, None, :, None, None], source.shape),
            "y": np.broadcast_to(np.arange(4, dtype=np.float32)[None, None, None, :, None], source.shape),
            "z": np.broadcast_to(np.arange(3, dtype=np.float32)[None, None, None, None, :], source.shape),
        },
    )

    arrays, time_values = derive_simulation_arrays(simulation, {"stride": [2, 2, 1]})

    assert arrays["Temperature"].shape == (2, 1, 3, 2, 3)
    assert np.array_equal(arrays["Temperature"], source[..., ::2, ::2, ::1])
    assert np.array_equal(arrays["x"][0, 0, :, 0, 0], np.array([0, 2, 4], dtype=np.float32))
    assert np.array_equal(arrays["y"][0, 0, 0, :, 0], np.array([0, 2], dtype=np.float32))
    assert time_values is not None
    assert time_values.shape == (2,)


def test_canonical_simulation_arrays_stride_after_time_drop(pino_simulation_factory):
    from aiida_pino.data import derive_simulation_arrays

    source = np.arange(2 * 1 * 6 * 4 * 3, dtype=np.float32).reshape(2, 1, 6, 4, 3)
    simulation = pino_simulation_factory("stride-no-time", arrays={"Temperature": source})

    arrays, time_values = derive_simulation_arrays(simulation, {"include_time": False, "stride": [2, 2, 1]})

    assert arrays["Temperature"].shape == (1, 3, 2, 3)
    assert np.array_equal(arrays["Temperature"], source[-1, ..., ::2, ::2, ::1])
    assert time_values is not None
    assert time_values.shape == (1,)


def test_collection_stride_metadata_uses_header_shapes(pino_dataset_factory):
    from aiida import orm
    from aiida_pino.workflows.derive_dataset import materialize_derived_dataset

    dataset = pino_dataset_factory(
        "stride-metadata",
        arrays={"Temperature": np.zeros((2, 1, 6, 4, 3), dtype=np.float32)},
    )

    derived = materialize_derived_dataset(
        dataset,
        orm.Dict(
            dict={
                "include_time": True,
                "time_as_batch": True,
                "time_as_dim": False,
                "moving_ref_frame": False,
                "moving_frame_steps": 1,
                "time_avg_window": 1,
                "super_res_factor": 1,
                "stride": [2, 2, 1],
            }
        ),
    )
    metadata = derived.get_dataset_metadata()

    assert metadata["array_shapes"]["Temperature"] == [2, 1, 3, 2, 3]
    assert metadata["spatial_shape"] == [3, 2, 3]
    assert metadata["grid_params"]["resolution"] == [3, 2, 3]


# --------------------------------------------------------------------------- #
# Metadata round-trip. Without it `find_reusable_derived_dataset` compares a
# stored variant's layout (stride missing -> [1,1,1]) against a [2,2,1] request,
# never matches, and rebuilds the 17 GB derivation on every submit.
# --------------------------------------------------------------------------- #
def test_stride_round_trips_through_dataset_metadata():
    from aiida_pino.utils.dataset_derivation import dataset_layout_profile_from_metadata

    profile = validate_derivation_profile({"stride": [2, 2, 1]})
    metadata = {
        "time_axis": True,
        "time_axis_mode": {"include_time": True, "time_as_batch": True, "time_as_dim": False},
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "avg_window_size": 1,
        "super_res_factor": 1,
        "stride": profile["stride"],
    }
    assert dataset_layout_profile_from_metadata(metadata)["stride"] == [2, 2, 1]


def test_datasets_predating_stride_read_back_as_unstrided():
    """Older collections have no `stride` key; they were built at full resolution."""
    from aiida_pino.utils.dataset_derivation import dataset_layout_profile_from_metadata

    metadata = {
        "time_axis": True,
        "time_axis_mode": {"include_time": True, "time_as_batch": True, "time_as_dim": False},
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "avg_window_size": 1,
        "super_res_factor": 1,
    }
    assert dataset_layout_profile_from_metadata(metadata)["stride"] == [1, 1, 1]


# --------------------------------------------------------------------------- #
# Hash back-compatibility. Adding `stride` to the profile re-hashed every
# *unstrided* derivation, so find_reusable_derived_dataset stopped matching
# stored variants and built duplicates -- one test split was created as a second copy
# of another with the same 53 sims and the same layout, differing only in hash.
# --------------------------------------------------------------------------- #
def test_default_stride_does_not_change_the_legacy_hash():
    from aiida_pino.utils.dataset_derivation import derivation_hash

    legacy = {
        "include_time": True, "time_as_batch": True, "time_as_dim": False,
        "moving_ref_frame": False, "moving_frame_steps": 1,
        "time_avg_window": 1, "super_res_factor": 1,
    }
    assert derivation_hash(legacy) == derivation_hash({**legacy, "stride": [1, 1, 1]})


def test_a_real_stride_does_change_the_hash():
    """A strided copy IS a different dataset and must not be reused for a full-res request."""
    from aiida_pino.utils.dataset_derivation import derivation_hash

    base = {"include_time": True, "time_as_batch": True, "time_as_dim": False,
            "moving_ref_frame": False, "moving_frame_steps": 1,
            "time_avg_window": 1, "super_res_factor": 1}
    assert derivation_hash(base) != derivation_hash({**base, "stride": [2, 2, 1]})
    assert derivation_hash({**base, "stride": [2, 2, 1]}) != derivation_hash({**base, "stride": [4, 4, 1]})
