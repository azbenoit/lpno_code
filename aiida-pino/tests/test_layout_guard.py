"""The submission-path layout guard, which had no test until it refused a feature.

2026-08-22. `derived_fields` landed as a derivation-profile key, was verified at
the transform level, and was **unreachable from a submission**: three arms died
at exit 303 within a second.

`validate_inputs` compares the layout implied by the TRAINING profile against
the selected derivation profile with strict dict equality. But
`build_derivation_profile_from_training_profile` can only ever produce the
DEFAULT `derived_fields`, because a training profile has no field expressing it.
So any non-empty value made the equality unsatisfiable and every such
submission was refused.

The precedent for the fix was already in that function: `stride` is popped from
both sides when the loader applies it, because demanding a match "is not a
contradiction but the normal way that switch is used" -- it had refused 77
historical trainings. `derived_fields` is stronger still: it is a CONTENT choice
(which arrays the staged member holds), not a frame LAYOUT choice, so it does
not belong in a layout comparison at all and is popped unconditionally.

The general shape, and why this file exists: **a new profile key meeting an
older strict-equality guard that enumerates what it knows and rejects the rest.**
The same family as the VRAM smoke inheriting a recording cadence it could not
satisfy, hours earlier the same day. The transform tests were correct and
covered nothing here, because the guard is on the submission path.
"""

from __future__ import annotations

from aiida_pino.workflows.train_model import layouts_compatible
from aiida_pino.utils.dataset_derivation import (
    DEFAULT_DERIVATION_PROFILE,
    validate_derivation_profile,
)


def _layout(**over):
    return validate_derivation_profile({**over})


def test_a_requested_derived_field_does_not_refuse_the_submission():
    """The exact 303: builder yields [], the selected variant asks for a field."""
    requested = _layout()                                     # what the training profile implies
    selected = _layout(derived_fields=["conduction_field"])   # what the run asked to stage
    assert requested["derived_fields"] == []
    assert selected["derived_fields"] == ["conduction_field"]
    assert layouts_compatible(requested, selected, env_stride="") is True


def test_a_real_layout_mismatch_is_still_refused():
    """The guard must keep doing its job; popping one key is not disabling it."""
    requested = _layout()
    selected = _layout(time_as_batch=False, moving_ref_frame=True, moving_frame_steps=30)
    assert layouts_compatible(requested, selected, env_stride="") is False


def test_the_loader_stride_escape_hatch_still_works():
    """Pre-existing behaviour: env stride is applied by the loader, not the dataset."""
    requested = _layout(stride=[2, 2, 1])
    selected = _layout(stride=[1, 1, 1])
    assert layouts_compatible(requested, selected, env_stride="") is False
    assert layouts_compatible(requested, selected, env_stride="2,2,1") is True


def test_identical_layouts_are_compatible():
    assert layouts_compatible(_layout(), _layout(), env_stride="") is True


def test_every_derivation_key_is_either_compared_or_deliberately_exempt():
    """A new key must not silently join the comparison and refuse submissions.

    This is the check that would have caught the bug when `derived_fields` was
    added: it fails for any key that is neither compared nor listed as exempt,
    forcing a decision rather than defaulting to strict equality.
    """
    from aiida_pino.workflows.train_model import LAYOUT_COMPARISON_EXEMPT_KEYS

    for key in DEFAULT_DERIVATION_PROFILE:
        if key in LAYOUT_COMPARISON_EXEMPT_KEYS:
            continue
        differing = dict(DEFAULT_DERIVATION_PROFILE)
        differing[key] = "sentinel-value-that-cannot-match"
        assert layouts_compatible(
            dict(DEFAULT_DERIVATION_PROFILE), differing, env_stride=""
        ) is False, f"{key} is neither compared nor exempt; decide which it is"


#: A training-profile value for each COMPARED key that must reach the built layout. A key
#: with no entry fails the test below, which is the point: it forces the author to say
#: whether the training profile can express the key at all.
PRODUCIBLE_FROM_TRAINING_PROFILE = {
    "include_time": False,
    "time_as_batch": True,
    "time_as_dim": True,
    "moving_ref_frame": True,
    "moving_frame_steps": 5,
    "time_avg_window": 3,
    "super_res_factor": 2,
    "stride": [2, 2, 1],
}


def test_every_compared_key_can_actually_be_PRODUCED_by_a_training_profile():
    """The half the original guard missed, and the half that keeps breaking.

    Asserting a key is COMPARED is not enough. `build_derivation_profile_from_training_profile`
    reads eight named fields and nothing else, so a compared key it cannot produce is
    compared against its DEFAULT forever -- and every submission setting that key is
    refused, with two profiles that are identical key-for-key.

    The original guard passed for `crop_z_top` for exactly this reason: the key IS
    compared. It just cannot be produced, so the comparison was permanently 0 vs 10 and it
    blocked the gas-cropped training the user had asked for. Same defect as
    `derived_fields` before it, and `time_frames` after it.
    """
    from aiida_pino.utils.dataset_derivation import build_derivation_profile_from_training_profile
    from aiida_pino.workflows.train_model import LAYOUT_COMPARISON_EXEMPT_KEYS

    compared = [k for k in DEFAULT_DERIVATION_PROFILE if k not in LAYOUT_COMPARISON_EXEMPT_KEYS]
    missing = [k for k in compared if k not in PRODUCIBLE_FROM_TRAINING_PROFILE]
    assert not missing, (
        f"no training-profile value recorded for compared key(s) {missing}. Either the "
        f"training profile can express it -- add the value here and make the builder read "
        f"it -- or it cannot, and it belongs in LAYOUT_COMPARISON_EXEMPT_KEYS."
    )

    for key in compared:
        value = PRODUCIBLE_FROM_TRAINING_PROFILE[key]
        profile = {key: value}
        if key == "stride":
            profile = {"env": {"PINO_LAZY_STRIDE": "2,2,1"}}
        elif key in ("time_avg_window", "moving_frame_steps"):
            # These are only VALID alongside a moving frame, so the probe has to be a
            # coherent profile rather than the key in isolation.
            profile = {key: value, "moving_ref_frame": True}
        built = build_derivation_profile_from_training_profile(profile)
        assert built[key] == value, (
            f"a training profile setting {key}={value!r} produced {built[key]!r}: the "
            f"builder cannot express this key, so comparing it refuses every submission "
            f"that sets it"
        )
