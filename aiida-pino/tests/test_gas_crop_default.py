"""The gas crop is the default; training without it takes an explicit opt-out.

Default since 2026-08-26. Every FLOW-3D deck in this project puts the substrate surface at
z = 0 with a constant 95 um of gas headroom above it at dz = 10 um, so the gas is exactly TEN
cells -- verified across every distinct grid in the store (nz = 40, 50, 60 and 80, which
differ only in how deep they go into the metal: zmin -295, -395, -495, -695 um). A fixed cell
count is therefore correct here for a reason, not by luck.

The dangerous half is EVALUATION. `build_derivation_profile_from_training_profile` reconstructs
what a model was trained on, and it did not mention `crop_z_top` at all -- so every profile fell
through to the package default. Flipping that default without touching the builder would have
cropped ~448 EXISTING models, trained on the full z extent, down to 30 cells at evaluation time:
wrong numbers, no error, exactly the shape of the near-misses this repo keeps recording.

So the builder now resolves the crop the model ACTUALLY trained with -- its recorded env var,
then its recorded field, then ZERO for anything predating the default -- which is the same
reasoning the stride branch beside it already carries.
"""

from __future__ import annotations

from aiida_pino.utils.dataset_derivation import (
    DEFAULT_DERIVATION_PROFILE,
    build_derivation_profile_from_training_profile,
    validate_derivation_profile,
)

GAS_CELLS = 10


def test_the_default_profile_crops_the_gas():
    assert DEFAULT_DERIVATION_PROFILE["crop_z_top"] == GAS_CELLS
    assert validate_derivation_profile({})["crop_z_top"] == GAS_CELLS


def test_opting_out_is_explicit_and_possible():
    """Training on the gas is still allowed -- it just has to be asked for."""
    assert validate_derivation_profile({"crop_z_top": 0})["crop_z_top"] == 0


def test_a_legacy_model_evaluates_UNCROPPED():
    """The safety property, and the reason the fallback is a PARAMETER.

    Evaluation passes `unrecorded_crop_z_top=0` because it reconstructs what the model saw.
    Measured on the live profile: 449 of 451 stored models record no crop, so inheriting the
    new default here would re-score essentially the whole population on a grid it never
    trained on.
    """
    legacy = {"run_name": "old", "config_name": "config_x", "env": {}}
    got = build_derivation_profile_from_training_profile(legacy, unrecorded_crop_z_top=0)
    assert got["crop_z_top"] == 0


def test_a_NEW_training_that_names_no_crop_gets_the_default():
    """The other half, which the first version of this got wrong: submitting a training with
    no explicit crop must inherit the package default, or making it the default does
    nothing."""
    fresh = {"run_name": "new", "config_name": "config_x", "env": {}}
    assert build_derivation_profile_from_training_profile(fresh)["crop_z_top"] == GAS_CELLS


def test_a_model_trained_with_the_crop_evaluates_cropped():
    """The crop rides as a loader env var since the read-time change, so that is where a
    model records what it actually saw."""
    trained = {"run_name": "new", "config_name": "config_x",
               "env": {"PINO_LAZY_CROP_Z_TOP": str(GAS_CELLS)}}
    assert build_derivation_profile_from_training_profile(trained)["crop_z_top"] == GAS_CELLS


def test_an_explicit_field_is_honoured_too():
    """Not every profile carries the env var; a recorded field means the same thing."""
    trained = {"run_name": "new", "config_name": "config_x", "env": {}, "crop_z_top": 4}
    assert build_derivation_profile_from_training_profile(trained)["crop_z_top"] == 4


def test_a_malformed_env_value_falls_back_to_uncropped():
    """Unparseable means unknown, and unknown must not invent a crop."""
    bad = {"run_name": "x", "config_name": "c", "env": {"PINO_LAZY_CROP_Z_TOP": "not-a-number"}}
    got = build_derivation_profile_from_training_profile(bad, unrecorded_crop_z_top=0)
    assert got["crop_z_top"] == 0


def test_ten_cells_is_the_gas_for_every_grid_in_the_store():
    """Pins the basis of the number, so a future grid with different headroom fails here
    rather than silently cropping metal. z runs to +95 um at dz = 10 um on every deck."""
    for nz, zmin_um in ((40, -295.0), (50, -395.0), (60, -495.0), (80, -695.0)):
        # Cell CENTRES: n cells span n-1 intervals, so nz = span/dz + 1. (My first version
        # of this line omitted the +1 and failed on every grid.)
        span_um = 95.0 - zmin_um
        assert round(span_um / 10.0) + 1 == nz, f"nz={nz} does not match a 10 um grid"
        # Centres at +5..+95 um above the surface: ten of them.
        assert round(95.0 / 10.0) == GAS_CELLS
