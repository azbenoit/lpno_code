"""A read-time crop and a baked crop must not both apply, and the pairing is not the user's job.

2026-08-27, friction report after ~26 failed submissions across four
rounds. `PINO_LAZY_CROP_Z_TOP` is REQUIRED for one class of run and FATAL for another, and
which class you are in is decided by something else:

  - plain training       -> resolves to an UNCROPPED collection -> the env var is required,
                            and dropping it silently changes the geometry
  - eval of a model      -> resolves to a collection that BAKES the crop -> fatal
  - training with a set  -> the workchain reuses a baked-crop collection -> fatal
    `PINO_LAZY_STRIDE`

The loader refuses the double-crop correctly, but only after submission. That cost a ten-arm
batch a day after the eval half of the rule had been written down -- knowing the
rule was not enough, because which half applies is decided by dataset resolution.

So the workchain reconciles: when the collection already bakes the crop the env var is
dropped and the reason reported. Only when the two AGREE -- then the intent is unambiguous.
When they disagree nothing is guessed and the loader's guard still refuses, naming both.
"""

from __future__ import annotations

from aiida_pino.utils import reconcile_crop_env


def test_a_baked_crop_drops_the_matching_env_var():
    env, note = reconcile_crop_env({"PINO_LAZY_CROP_Z_TOP": "10"}, {"crop_z_top": 10})
    assert "PINO_LAZY_CROP_Z_TOP" not in env
    assert "already bakes crop_z_top=10" in note
    assert "20 cells instead of 10" in note, "the note must say what the double crop would do"


def test_an_uncropped_collection_keeps_the_env_var():
    """The other half of the pairing: dropping it here changes the geometry silently."""
    env, note = reconcile_crop_env({"PINO_LAZY_CROP_Z_TOP": "10"}, {"crop_z_top": 0})
    assert env["PINO_LAZY_CROP_Z_TOP"] == "10"
    assert note == ""


def test_disagreeing_crops_are_left_to_the_loader():
    """Two different crops is not an unambiguous intent, so nothing is guessed."""
    env, note = reconcile_crop_env({"PINO_LAZY_CROP_Z_TOP": "5"}, {"crop_z_top": 10})
    assert env["PINO_LAZY_CROP_Z_TOP"] == "5"
    assert note == ""


def test_the_rest_of_the_env_is_untouched():
    """Only the one key moves. The last time an env was rebuilt here it deleted the training
    env, including the silent `PINO_LAZY_MOVING_REF`."""
    env, _ = reconcile_crop_env(
        {"PINO_LAZY_CROP_Z_TOP": "10", "PINO_LAZY_MOVING_REF": "40,20",
         "PINO_MATERIAL_OVERLAY": "/p.json"},
        {"crop_z_top": 10},
    )
    assert env == {"PINO_LAZY_MOVING_REF": "40,20", "PINO_MATERIAL_OVERLAY": "/p.json"}


def test_no_env_var_and_junk_are_both_no_ops():
    assert reconcile_crop_env({}, {"crop_z_top": 10}) == ({}, "")
    assert reconcile_crop_env(None, None) == ({}, "")
    env, note = reconcile_crop_env({"PINO_LAZY_CROP_Z_TOP": "nonsense"}, {"crop_z_top": 10})
    assert env["PINO_LAZY_CROP_Z_TOP"] == "nonsense" and note == ""


def test_both_workchains_reconcile_after_resolving_the_dataset():
    """It can only be decided once the dataset has resolved, so it must run after that."""
    from pathlib import Path

    import aiida_pino

    root = Path(aiida_pino.__file__).parent / "workflows"
    for name in ("train_model.py", "evaluate_model.py"):
        assert "reconcile_crop_env(" in (root / name).read_text(), f"{name} does not reconcile"
