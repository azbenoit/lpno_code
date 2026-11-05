"""`crop_z_top` restricts the evaluated domain to the one a model was trained on.

2026-08-25. Model 63965 trained on `z_range [-300, 0] um`, 30 cells -- the gas deliberately
cropped out. Every current NPZ export carries exactly 10 gas layers, 100 um above z=0, in
all four regimes (straight/random/multi-beam nz=40, preheat nz=50). So evaluating that
model on current data feeds it 10 layers of void it has never seen, and every metric is
void: one eval reported substrate 26.90 K, beam box 229.22 K, fl IoU 0.2552, and was
read as a model pathology before anyone checked the geometry. It is not a pathology --
`pred_alpha ~ 1` is correct behaviour on a domain the model was never built for.

Why a derivation key rather than reviving `custom.geometry`: the archived config of an
NPZ-trained model already records a geometry the run did NOT use. Model 65230's archived
`custom.geometry` says `z_range [-0.0003, 0], resolution [90, 40, 30]` while it actually
trained on the derived 40-layer grid, because `grid_params` is replaced by a local that is
never written back. 14 of 61 packaged configs carry an explicit geometry block and none
say `auto`, so making the config authoritative would silently CROP current 40-layer models
to 30 -- the same failure pointed the other way. An opt-in key reinterprets nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.utils.dataset_derivation import (
    DERIVATION_PROFILE_KEYS,
    derivation_hash,
    validate_derivation_profile,
)
from aiida_pino.utils.runtime_export import apply_derivation

IDENTITY = {"stride": [1, 1, 1], "include_time": True, "moving_ref_frame": False,
            "moving_frame_steps": 1, "time_avg_window": 1, "super_res_factor": 1}


def test_it_is_part_of_the_dataset_identity():
    """Two datasets differing only by the crop must not collide on one signature.

    Compares 0 against the default rather than `{}` against 10: since 2026-08-26 the default
    IS 10, so `{}` and `{"crop_z_top": 10}` are the same profile and hashing alike is
    correct. The property under test is that the crop still separates identities.
    """
    assert "crop_z_top" in DERIVATION_PROFILE_KEYS
    a = validate_derivation_profile({"crop_z_top": 0})
    b = validate_derivation_profile({})
    assert a["crop_z_top"] != b["crop_z_top"]
    assert derivation_hash(a) != derivation_hash(b)


def test_the_default_crops_the_gas_and_opting_out_is_explicit():
    """Since 2026-08-26 the gas crop is the default.

    The old name asserted a zero default "so every existing dataset is unchanged". That
    property still holds, by a different mechanism: stored datasets are not re-derived, and
    an evaluation reconstructs the crop a model RECORDED (`unrecorded_crop_z_top=0`), so the
    449 of 451 stored models that name no crop are still scored on the full extent.
    """
    assert validate_derivation_profile({})["crop_z_top"] == 10
    assert validate_derivation_profile({"crop_z_top": 0})["crop_z_top"] == 0


def test_a_negative_or_junk_crop_is_refused():
    for bad in (-1, "many", 1.5j):
        with pytest.raises(ValueError):
            validate_derivation_profile({"crop_z_top": bad})


def test_it_drops_the_TOP_layers_not_the_bottom():
    """The gas is at high z, which is the tail of the canonical `(nt,1,nx,ny,nz)` shape.

    Dropping from the wrong end would remove the substrate the model exists to predict and
    keep the void -- and the shape would look right either way.
    """
    array = np.arange(2 * 1 * 3 * 2 * 5, dtype=np.float32).reshape(2, 1, 3, 2, 5)
    out = apply_derivation(array, {**IDENTITY, "crop_z_top": 2})
    assert out.shape == (2, 1, 3, 2, 3)
    np.testing.assert_array_equal(out, array[..., :3])


def test_the_ten_gas_layers_of_a_real_export_come_off():
    """The motivating case: 40 -> 30 z layers, the model's own domain."""
    array = np.zeros((1, 1, 90, 40, 40), dtype=np.float32)
    assert apply_derivation(array, {**IDENTITY, "crop_z_top": 10}).shape == (1, 1, 90, 40, 30)


def test_cropping_everything_crops_NOTHING_rather_than_returning_an_empty_grid():
    """The empty grid is still what must never happen; the response to it changed.

    This refused with a ValueError until the gas crop became the DEFAULT, at which point a
    grid-blind 10 hit every coarse array in the suite -- 201 tests -- and refusing would have
    made the default unusable on any grid shallower than the crop. A crop that would take the
    whole extent means there is NO GAS on this grid, so nothing is cropped and it says so.
    The array is returned intact, which is the property the original name was protecting.
    """
    array = np.zeros((1, 1, 2, 2, 3), dtype=np.float32)
    out = apply_derivation(array, {**IDENTITY, "crop_z_top": 3})
    assert out.shape == array.shape
    np.testing.assert_array_equal(out, array)


def test_zero_is_a_true_no_op():
    array = np.arange(24, dtype=np.float32).reshape(1, 1, 2, 3, 4)
    np.testing.assert_array_equal(apply_derivation(array, {**IDENTITY, "crop_z_top": 0}), array)


def test_the_crop_precedes_striding():
    """Order matters: crop-then-stride and stride-then-crop give different grids.

    The crop defines the DOMAIN, so it runs first and everything else operates on the
    restricted one.
    """
    array = np.arange(1 * 1 * 1 * 1 * 8, dtype=np.float32).reshape(1, 1, 1, 1, 8)
    out = apply_derivation(array, {**IDENTITY, "crop_z_top": 2, "stride": [1, 1, 2]})
    # crop to 6 then stride 2 -> indices 0,2,4 ; stride-then-crop would give 0,2
    np.testing.assert_array_equal(out.ravel(), np.array([0.0, 2.0, 4.0]))
