"""`reconcile_crop_env` must find the baked crop where derived collections actually keep it.

The function drops `PINO_LAZY_CROP_Z_TOP` when the resolved dataset already bakes the same
crop, because applying both removes 20 cells where 10 were meant. It read
`dataset_metadata["crop_z_top"]` -- a key a derived collection does NOT have at the top
level. `PinoDatasetCollectionData.get_dataset_metadata()` for one such collection exposes
`derivation_profile` (holding `crop_z_top: 10`) and no top-level `crop_z_top`, so `baked`
computed as 0, the reconciliation silently did nothing, and the loader refused at run time:

    ValueError: [NpzDirStore] the dataset staged at '...' is already cropped crop_z_top=10
    by its derivation profile, and PINO_LAZY_CROP_Z_TOP='10' would crop it again

The guard is right; the reconciliation that exists to prevent it was looking in the wrong
place. Reported 2026-09-02 as an env merge/replace defect after
two evals failed identically -- but an eval env that omits the key DOES drop
the inherited one (`merge_model_training_with_evaluation_profile` replaces `env`). The
variable is re-added from `ctx.demoted_env`, which is why omitting it from the evaluation
profile changed nothing, and why a sentinel for "unset an inherited key" would not have
fixed this.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.dataset_derivation import reconcile_crop_env

ENV = {"PINO_LAZY_CROP_Z_TOP": "10", "PINO_LAZY_CACHE": "1"}


class NestedCropTest(unittest.TestCase):
    def test_a_crop_baked_in_the_derivation_profile_is_found(self):
        # A DERIVED collection: `materialize_derived_dataset` wrote the crop into the
        # staged arrays, so carrying the env var too would crop twice.
        env, note = reconcile_crop_env(
            ENV, {"dataset_role": "derived", "derivation_profile": {"crop_z_top": 10}}
        )
        self.assertNotIn("PINO_LAZY_CROP_Z_TOP", env, "the double crop was not reconciled")
        self.assertIn("PINO_LAZY_CACHE", env, "unrelated variables must survive")
        self.assertIn("10", note)

    def test_the_top_level_key_still_works(self):
        env, note = reconcile_crop_env(ENV, {"crop_z_top": 10})
        self.assertNotIn("PINO_LAZY_CROP_Z_TOP", env)
        self.assertTrue(note)

    def test_disagreeing_crops_are_still_left_to_the_loader(self):
        """Nothing is guessed when the two differ; the loader names both numbers."""
        env, note = reconcile_crop_env(
            ENV, {"dataset_role": "derived", "derivation_profile": {"crop_z_top": 4}}
        )
        self.assertEqual(env.get("PINO_LAZY_CROP_Z_TOP"), "10")
        self.assertEqual(note, "")

    def test_an_uncropped_collection_keeps_the_env_var(self):
        """A plain training resolves to an uncropped collection and NEEDS the read-time crop."""
        for metadata in ({}, {"dataset_role": "derived", "derivation_profile": {}},
                         {"dataset_role": "derived", "derivation_profile": {"crop_z_top": 0}}):
            env, note = reconcile_crop_env(ENV, metadata)
            self.assertEqual(env.get("PINO_LAZY_CROP_Z_TOP"), "10", f"dropped for {metadata}")
            self.assertEqual(note, "")

    def test_no_env_crop_is_a_no_op(self):
        env, note = reconcile_crop_env(
            {"PINO_LAZY_CACHE": "1"},
            {"dataset_role": "derived", "derivation_profile": {"crop_z_top": 10}},
        )
        self.assertEqual(env, {"PINO_LAZY_CACHE": "1"})
        self.assertEqual(note, "")


    def test_a_base_collection_s_crop_record_is_NOT_treated_as_baked(self):
        """The base-collection case, and the reason this guard is narrow.

        `dataset_role: base` with `crop_z_top: 10`, but its 13 members serve RAW 40-cell
        arrays -- read off the payload, not inferred -- and an FNO eval measured
        `substrate_fraction` 0.75 (30 metal of 40). The crop is a REQUEST for read time, so
        `PINO_LAZY_CROP_Z_TOP` is the mechanism that applies it. Dropping it would leave the
        eval scoring an uncropped domain against baselines taken on a cropped one: a wrong
        number instead of a crash. 399 base collections carry such a record.
        """
        env, note = reconcile_crop_env(
            ENV, {"dataset_role": "base", "derivation_profile": {"crop_z_top": 10}}
        )
        self.assertEqual(env.get("PINO_LAZY_CROP_Z_TOP"), "10",
                         "a base collection's crop record must not un-crop the run")
        self.assertEqual(note, "")

    def test_an_unknown_role_is_treated_conservatively(self):
        # subset/composed may or may not materialise; keeping the env var restores the
        # loader's loud refusal rather than risking a silent un-crop.
        for role in ("subset", "composed", "", None):
            env, _ = reconcile_crop_env(
                ENV, {"dataset_role": role, "derivation_profile": {"crop_z_top": 10}}
            )
            self.assertEqual(env.get("PINO_LAZY_CROP_Z_TOP"), "10", f"un-cropped for {role!r}")


if __name__ == "__main__":
    unittest.main()
