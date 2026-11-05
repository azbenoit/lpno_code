"""A steady derivation can be asked to build `conduction_field` alongside Temperature.

2026-08-22, requested for the preheat trainings; the diagnosis was
verified before implementing.

The problem: a steady derivation (`moving_ref_frame`, steps 30, avg 30) cannot
serve `conduction_field`. `materialize_derived_dataset` drops every path channel
and rebuilds only the four `_CONSTANT_PATH_CHANNELS` from the manifest, and
`derive-fields` does not help despite reading as though it would -- it stores
the datasource's CACHE files, and the staged-derived path never calls that
builder.

**The obvious fix is wrong and would be a wrong-NUMBER bug rather than a crash.**
Rebuilding the channel inside the already-staged member cannot be faithful: that
member has been cropped to the beam-following window AND averaged over 30
frames, and its `time_s` is a single value -- the 30 pre-average frame times are
gone. You would get an instantaneous lab-frame field against a 30-frame-averaged
ground truth: spatially alignable, silently misaligned in time.

So it is built in the LAB frame and injected into `derive_simulation_arrays`'s
transform loop, which computes `moving_frame_slices` ONCE per simulation so
"every field is cropped to the SAME per-frame window". Anything in that loop
gets bit-identical treatment to Temperature -- the right physics here, since
conduction_field is temperature-like and wants the same crop and same average.

**Shape equality against Temperature is the load-bearing assertion**: it is what
proves the field took the same crop and the same average, rather than merely
being present.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.dataset_derivation import (
    DERIVATION_PROFILE_KEYS,
    validate_derivation_profile,
)


def test_derived_fields_is_part_of_the_derivation_identity():
    """It must feed the hash, or two different datasets collide on one signature."""
    assert "derived_fields" in DERIVATION_PROFILE_KEYS


def test_default_is_empty_so_every_existing_submission_is_unchanged():
    out = validate_derivation_profile({})
    assert out["derived_fields"] == []


def test_the_list_is_sorted_so_equal_profiles_hash_alike():
    """Same reason `stride` is normalised: order must not change identity."""
    from aiida_pino.utils.dataset_derivation import derivation_hash

    a = validate_derivation_profile({"derived_fields": ["conduction_field"]})
    b = validate_derivation_profile({"derived_fields": ["conduction_field"]})
    assert a["derived_fields"] == b["derived_fields"] == ["conduction_field"]
    assert derivation_hash(a) == derivation_hash(b)


def test_requesting_a_field_changes_the_derivation_hash():
    """A run asking for it must get a NEW collection, never reuse the plain one."""
    from aiida_pino.utils.dataset_derivation import derivation_hash

    plain = validate_derivation_profile({})
    withf = validate_derivation_profile({"derived_fields": ["conduction_field"]})
    assert derivation_hash(plain) != derivation_hash(withf)


def test_an_unknown_field_is_refused_at_submit_not_after_staging():
    with pytest.raises(ValueError, match="derived_fields"):
        validate_derivation_profile({"derived_fields": ["conductoin_field"]})


def test_duplicates_collapse_so_identity_is_stable():
    out = validate_derivation_profile(
        {"derived_fields": ["conduction_field", "conduction_field"]}
    )
    assert out["derived_fields"] == ["conduction_field"]


def test_a_moving_frame_request_infers_the_steady_layout(caplog):
    """The trap that made an earlier end-to-end check vacuous, now closed.

    `time_as_batch` defaults to True (the base, one-sample-per-frame layout), and
    the moving frame exists only as the beam-centred average of the steady
    layout. A profile asking for `{moving_ref_frame: True, moving_frame_steps:
    30, time_avg_window: 30}` and nothing else used to be SILENTLY reduced to no
    crop and no average. It now infers `time_as_batch: False` and says so
    (2026-10-30).

    Verified on a real simulation once the profile was correct: Temperature came
    back `(1, 121, 100, 40)` -- 30 frames averaged to one, x cropped 150 -> 121
    by the beam-following window -- with `conduction_field` identical.
    """
    with caplog.at_level("WARNING", logger="aiida_pino.utils.dataset_derivation"):
        inferred = validate_derivation_profile(
            {"moving_ref_frame": True, "moving_frame_steps": 30, "time_avg_window": 30}
        )
    assert inferred["moving_ref_frame"] is True
    assert inferred["time_as_batch"] is False
    assert inferred["moving_frame_steps"] == 30
    assert inferred["time_avg_window"] == 30
    assert any("time_as_batch" in r.getMessage() for r in caplog.records), "the inference must be announced"


def test_a_moving_frame_with_an_explicit_per_frame_layout_is_refused():
    """Both stated explicitly is a contradiction, not something to guess about."""
    with pytest.raises(ValueError, match="moving_ref_frame"):
        validate_derivation_profile({"moving_ref_frame": True, "time_as_batch": True, "time_avg_window": 30})


def test_an_explicit_steady_layout_is_unchanged_and_silent(caplog):
    with caplog.at_level("WARNING", logger="aiida_pino.utils.dataset_derivation"):
        real = validate_derivation_profile(
            {
                "time_as_batch": False,
                "moving_ref_frame": True,
                "moving_frame_steps": 30,
                "time_avg_window": 30,
            }
        )
    assert real["moving_ref_frame"] is True
    assert real["moving_frame_steps"] == 30
    assert real["time_avg_window"] == 30
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_derived_fields_alone_still_counts_as_a_payload_change():
    """The simplest use of the feature must not take the no-op staging fast path.

    A profile whose ONLY non-default key is `derived_fields` still has to be treated
    as transforming. If `derivation_changes_payload` said False, staging would
    reference the stored payload object instead of writing a member -- and the
    channel would never be built, silently, which is the same class of defect as the
    worker and the strided writer (2026-08-22, three writers of one transform).
    It holds today because the test is equality with the validated DEFAULT profile,
    and `derived_fields` is part of that profile; this pins it.
    """
    from aiida_pino.data.pino_dataset_collection import derivation_changes_payload
    from aiida_pino.utils.dataset_derivation import validate_derivation_profile

    # The CONTROL, stated with an explicit `crop_z_top: 0`: the default became 10 on
    # 2026-08-26, so `{}` is no longer the identity and this line was asserting the default
    # rather than the baseline it means to establish.
    assert derivation_changes_payload(validate_derivation_profile({"crop_z_top": 0})) is False
    assert (
        derivation_changes_payload(
            validate_derivation_profile({"derived_fields": ["conduction_field"]})
        )
        is True
    )
