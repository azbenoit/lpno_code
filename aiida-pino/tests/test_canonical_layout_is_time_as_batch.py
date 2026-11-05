"""The canonical base layout is `time_as_batch`, not `time_as_dim`.

`time_as_dim` folds the trajectory into a grid axis, which *requires* every case
to have the same frame count. Real FLOW-3D random-path data is ragged (dataset
2579: 14 to 400 frames), so `time_as_dim` cannot represent it at all — yet it was
the canonical default, and 6 collections were stamped with it despite
`uniform_timesteps=False`, an impossible layout that no derivation could satisfy.

`time_as_batch` (one sample per frame) is well defined for every dataset, ragged
or not, and is what every FLOW-3D model here actually trains on. That makes it
the correct lossless base.

The layout is a *reading* instruction, not a storage format: a member's arrays
are `(T, C, X, Y, Z)` on disk either way, and `derive_simulation_arrays` applies
the layout at export. So flipping the default reinterprets nothing on disk.
"""
from __future__ import annotations

from aiida_pino.data import is_canonical_metadata
from aiida_pino.utils import canonical_base_layout_profile, validate_derivation_profile


def test_canonical_profile_is_time_as_batch():
    canonical = canonical_base_layout_profile()
    assert canonical["time_as_batch"] is True
    assert canonical["time_as_dim"] is False
    assert canonical["include_time"] is True
    assert canonical["moving_ref_frame"] is False
    assert canonical["super_res_factor"] == 1


def test_canonical_profile_survives_validation_unchanged():
    """validate_derivation_profile must not rewrite the canonical layout.

    It forces time_as_batch=False whenever time_as_dim is set, so a canonical
    profile that still said time_as_dim would silently come back different.
    """
    canonical = canonical_base_layout_profile()
    assert validate_derivation_profile(canonical) == canonical


def test_is_canonical_metadata_agrees_with_the_profile():
    """The stamp check and the profile constant must not drift apart."""
    canonical = canonical_base_layout_profile()
    metadata = {
        "time_axis": True,
        "time_axis_mode": {
            "include_time": canonical["include_time"],
            "time_as_batch": canonical["time_as_batch"],
            "time_as_dim": canonical["time_as_dim"],
        },
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "avg_window_size": 1,
        "super_res_factor": 1,
    }
    assert is_canonical_metadata(metadata)


def test_time_as_dim_metadata_is_no_longer_canonical():
    metadata = {
        "time_axis": True,
        "time_axis_mode": {"include_time": True, "time_as_batch": False, "time_as_dim": True},
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "avg_window_size": 1,
        "super_res_factor": 1,
    }
    assert not is_canonical_metadata(metadata)


def test_a_ragged_dataset_can_satisfy_the_canonical_layout():
    """The point of the change: ragged data has a valid canonical variant now.

    `native_source_incompatibility_reason` rejects time_as_dim on a collection
    whose members differ in frame count. With time_as_batch canonical, a ragged
    base is usable without an explicit derivation profile.
    """
    from aiida_pino.utils.dataset_derivation import _variant_token

    canonical = canonical_base_layout_profile()
    # the token feeds dataset_identifier; it must describe a time_as_batch variant
    assert "time" in _variant_token(canonical)


def test_canonical_layout_token_reflects_time_as_batch():
    from aiida_pino.data.pino_dataset import PinoDatasetData

    assert "batch" in PinoDatasetData.CANONICAL_LAYOUT
    assert "dim" not in PinoDatasetData.CANONICAL_LAYOUT
