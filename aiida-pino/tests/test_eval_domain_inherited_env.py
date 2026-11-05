"""The domain guard must mirror how the eval profile's `env` is merged.

`domain_extent_mismatch` refuses an eval that would score a model on a different domain
extent. It reads the crop from the eval derivation and the eval profile's env -- but the
crop has a third source: `merge_model_training_with_evaluation_profile` deep-copies the
TRAINING profile, and an evaluation profile REPLACES `env` wholesale rather than merging
key-by-key. An evaluation profile with no `env` therefore inherits the training crop and
serves the right domain.

Measured over the last 120 TrainEvalWorkChains: 73 supply no evaluation_profile at all and
0 supply one without an env -- so ignoring that source refused 73 of 120 submissions once
their model carried a crop, which the default gas crop makes the normal case.
"""

import unittest

from aiida_pino.utils.dataset_derivation import domain_extent_mismatch

CROPPED = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
UNCROPPED = {"env": {}}


class InheritedEnvTest(unittest.TestCase):
    def test_no_eval_env_inherits_the_training_crop_and_is_allowed(self):
        # The majority shape: `train_eval` with no evaluation_profile at all.
        self.assertIsNone(domain_extent_mismatch(CROPPED, {"crop_z_top": 0}, {}))
        self.assertIsNone(domain_extent_mismatch(CROPPED, {"crop_z_top": 0}, None))

    def test_the_arm_that_motivated_the_guard_is_still_refused(self):
        # 72419: the evaluation profile DID declare an env, with no crop in it. That is how
        # the wholesale replacement dropped the training crop and served 40 cells to a model
        # trained on 30. An env present and silent about the crop still means no crop.
        msg = domain_extent_mismatch(
            CROPPED, {"crop_z_top": 0},
            {"env": {"PINO_LAZY_CACHE": "1", "PINO_LOADER_WORKERS": "2"}},
        )
        self.assertIsNotNone(msg)
        self.assertIn("crop_z_top", msg)

    def test_an_explicitly_uncropped_eval_of_a_cropped_model_is_refused(self):
        self.assertIsNotNone(
            domain_extent_mismatch(CROPPED, {"crop_z_top": 0},
                                   {"env": {"PINO_LAZY_CROP_Z_TOP": "0"}})
        )

    def test_the_other_direction_is_refused_too(self):
        # Uncropped model, cropped eval: denied cells it did see.
        self.assertIsNotNone(domain_extent_mismatch(UNCROPPED, {"crop_z_top": 10}, {}))

    def test_the_correct_spellings_of_a_matching_domain_stay_allowed(self):
        self.assertIsNone(domain_extent_mismatch(CROPPED, {"crop_z_top": 10}, {}))
        self.assertIsNone(
            domain_extent_mismatch(CROPPED, {"crop_z_top": 0},
                                   {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}})
        )

    def test_a_model_with_no_recorded_crop_is_unaffected(self):
        # ~449 stored models record no crop; they compare 0 against 0.
        self.assertIsNone(domain_extent_mismatch({}, {"crop_z_top": 0}, {}))
        self.assertIsNone(domain_extent_mismatch({}, {}, {}))


if __name__ == "__main__":
    unittest.main()
