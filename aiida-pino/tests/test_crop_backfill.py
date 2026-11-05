"""The phantom-crop correction: what it rewrites, and what it deliberately does not."""

from types import SimpleNamespace

import pytest

from aiida_pino.utils.crop_backfill import _phantom_crop, apply_crop_correction


def _node(*, role="base", crop=10, spatial=(150, 100, 30), member_z=40, shapes=None):
    metadata = {
        "spatial_shape": list(spatial),
        "array_shapes": shapes if shapes is not None else {"Temperature": [190, 1, *spatial]},
        "member_spatial_shapes": [[spatial[0], spatial[1], member_z]],
        "derivation_profile": {"crop_z_top": crop},
    }
    attributes = {
        "dataset_role": role,
        "derivation_profile": {"crop_z_top": crop},
        "dataset_metadata": metadata,
    }
    return SimpleNamespace(
        base=SimpleNamespace(
            attributes=SimpleNamespace(get=lambda key, default=None: attributes.get(key, default)),
            extras=SimpleNamespace(get=lambda key, default=None: None),
        )
    )


def test_phantom_crop_restores_the_member_extent():
    correction = _phantom_crop(_node())
    assert correction is not None
    assert correction["effective_crop_z_top"] == 0
    assert correction["effective_spatial_shape"] == [150, 100, 40]
    assert correction["effective_array_shapes"]["Temperature"] == [190, 1, 150, 100, 40]
    assert correction["crop_correction"]["from"] == 10
    assert correction["crop_correction"]["to"] == 0
    assert correction["crop_correction"]["recorded_z"] == 30
    assert correction["crop_correction"]["actual_z"] == 40


def test_derived_collections_are_left_alone():
    """`derived` genuinely bakes the crop for at least some read paths."""
    assert _phantom_crop(_node(role="derived")) is None


def test_a_record_that_already_tells_the_truth_is_left_alone():
    """The crop was refused at creation (no gas layer on this grid)."""
    assert _phantom_crop(_node(spatial=(4, 4, 4), member_z=4)) is None


def test_a_record_that_disagrees_with_itself_is_left_alone():
    """One array cropped, another not: not ours to guess at."""
    node = _node(shapes={"Temperature": [190, 1, 150, 100, 30], "P": [190, 1, 150, 100, 33]})
    assert _phantom_crop(node) is None


def test_no_crop_stamped_means_no_correction():
    assert _phantom_crop(_node(crop=0)) is None


def test_overlay_replaces_shapes():
    metadata = {"spatial_shape": [150, 100, 30], "array_shapes": {"T": [190, 1, 150, 100, 30]}}
    extras = {
        "crop_correction": {"from": 10, "to": 0},
        "effective_spatial_shape": [150, 100, 40],
        "effective_array_shapes": {"T": [190, 1, 150, 100, 40]},
    }
    out = apply_crop_correction(metadata, extras)
    assert out["spatial_shape"] == [150, 100, 40]
    assert out["array_shapes"]["T"] == [190, 1, 150, 100, 40]


def test_overlay_preserves_the_recorded_intent():
    """`crop_z_top` stays as recorded: it is what was REQUESTED, truthfully.

    Rewriting it would change what a derived child inherits, which is a change to
    the data, not to the record.
    """
    metadata = {"spatial_shape": [150, 100, 30], "derivation_profile": {"crop_z_top": 10}}
    extras = {"crop_correction": {"from": 10, "to": 0}, "effective_spatial_shape": [150, 100, 40]}
    out = apply_crop_correction(metadata, extras)
    assert out["derivation_profile"]["crop_z_top"] == 10


def test_overlay_is_a_no_op_without_a_correction():
    metadata = {"spatial_shape": [150, 100, 30]}
    assert apply_crop_correction(dict(metadata), {})["spatial_shape"] == [150, 100, 30]
