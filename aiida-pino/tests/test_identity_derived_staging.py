"""An identity derivation must stage by pack index, not by full copy.

Zero-copy pack-index staging has been the default since 9d6d8bd and works for
base/composed collections (a 42-member composed collection stages as a 4 MB
index instead of ~10 GB). `_export_one_serial` routes every `is_derived`
collection down a writer that never receives `pack_index`, so it always makes
real copies -- and the train/val/test subsets every training submission
materialises are derived. Measured on the live cache: of 260
post-flip staged dirs (55.7 GB), every real-copy entry of any size is
`dataset_role=derived`, e.g. `ti64_preheat60_fixed59_train_split` 4.53 GB and
`ti64_multibeam40_capped_test_split` 4.81 GB -- roughly three full dataset
copies per training submission, on a root that is 94% full.

Those subsets carry an all-default derivation profile (stride [1,1,1], no
resample, no moving frame): the payload is byte-identical to the source
simulation's, which is exactly the case the index can reference. A profile that
really transforms the arrays must still be written out.
"""

from __future__ import annotations

import pytest

from aiida_pino.data.pino_dataset_collection import derivation_changes_payload


def test_identity_profiles_do_not_change_the_payload():
    """These are the profiles the split subsets actually carry (taken from a live dataset)."""
    # `crop_z_top: 0` is stated explicitly because the DEFAULT became 10 on 2026-08-26.
    # Production builds a subset profile through `dataset_layout_profile_from_metadata`,
    # which reads a stored dataset as uncropped -- verified on live subsets 39420, 46553,
    # 47323 and 62134, all crop 0 and still staged by index. A literal dict here does not
    # go through that path, so it has to say so, or it tests the default rather than the
    # property.
    assert derivation_changes_payload({"crop_z_top": 0}) is False
    assert derivation_changes_payload(None) is False
    assert derivation_changes_payload({
        "stride": [1, 1, 1], "time_as_dim": False, "include_time": True,
        "time_as_batch": True, "time_avg_window": 1, "moving_ref_frame": False,
        "super_res_factor": 1, "moving_frame_steps": 1, "crop_z_top": 0,
    }) is False, "the live split-subset profile is a no-op and must stage by index"


def test_real_transforms_still_force_a_written_copy():
    """Anything that rewrites the arrays cannot be served from the source pack."""
    assert derivation_changes_payload({"stride": [2, 2, 1]}) is True
    assert derivation_changes_payload({"super_res_factor": 2}) is True
    # Moving-frame resampling only takes effect outside batch/dim time layout,
    # so the flags have to be set in a combination that actually applies.
    assert derivation_changes_payload(
        {"time_as_batch": False, "moving_ref_frame": True}
    ) is True
    assert derivation_changes_payload(
        {"time_as_batch": False, "moving_ref_frame": True, "time_avg_window": 4}
    ) is True


def test_flags_the_validator_normalises_away_are_not_transforms():
    """A key that cannot take effect in its own profile changes no bytes.

    `time_avg_window` and `moving_ref_frame` are forced back to their defaults
    when the time layout is batch or dim (the validator would otherwise raise
    "time_avg_window > 1 requires moving_ref_frame=True"). Comparing VALIDATED
    profiles therefore compares effective semantics, which is the point: a
    request that the model normalises into a no-op must not cost a full copy.
    """
    # `crop_z_top: 0` is stated explicitly because the DEFAULT became 10 on 2026-08-26.
    # Production builds a subset profile through `dataset_layout_profile_from_metadata`,
    # which reads a stored dataset as uncropped -- verified on live subsets 39420, 46553,
    # 47323 and 62134, all crop 0 and still staged by index. A literal dict here does not
    # go through that path, so it has to say so, or it tests the default rather than the
    # property.
    assert derivation_changes_payload({"time_avg_window": 4, "crop_z_top": 0}) is False
    # A moving frame alone is a real request since 2026-10-30: it infers time_as_batch=False
    # (the steady layout) instead of being switched off, so it does transform.
    assert derivation_changes_payload({"moving_ref_frame": True, "crop_z_top": 0}) is True


def test_unparseable_profile_is_treated_as_transforming():
    """Fail safe: if we cannot prove it is a no-op, write the copy."""
    assert derivation_changes_payload({"totally_unknown_key": 7}) is True
