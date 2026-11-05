"""An explicit derivation profile must not silently re-domain the model.

`evaluate_model` reconstructs "what the MODEL saw" only when no `derivation_profile`
input is supplied. When one IS supplied it is trusted outright -- and on 2026-08-27 a
`train_eval` handed the TRAINING derivation (`crop_z_top: 0`, chosen so the read-time crop
would not trip the double-crop guard) straight to the evaluation, which scored a model
trained on a 30-deep cropped domain against the uncropped 40-deep grid. Whole-domain MAE
read 816 K against 91 K for a comparable arm while SUBSTRATE MAE was within 2 % -- a
plausible number, not an error, and it read as a broken architecture for half a day.

A RESOLUTION difference (stride) is the norm and must stay allowed. A DOMAIN-EXTENT difference (crop_z_top) is not.
"""
import unittest

from aiida_pino.utils.dataset_derivation import domain_extent_mismatch


#: The eval profile that 72419 actually carried: an `env` is PRESENT and silent about the
#: crop. That is what drops the training crop, because the merge replaces `env` wholesale
#: rather than merging key-by-key. Omitting the argument entirely no longer expresses this
#: -- an absent `env` key inherits the training env, crop included, and serves the right
#: domain, which is the shape 73 of the last 120 TrainEvalWorkChains submit. These three
#: cases were written with the two-argument shorthand before that distinction existed; the
#: behaviour they pin is unchanged, only how it is spelled.
EVAL_ENV_WITHOUT_CROP = {"env": {"PINO_LAZY_CACHE": "1", "PINO_LOADER_WORKERS": "2"}}


class DomainExtentMismatch(unittest.TestCase):
    def test_same_crop_is_fine(self):
        tp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
        self.assertIsNone(domain_extent_mismatch(tp, {"crop_z_top": 10}))

    def test_model_cropped_eval_uncropped_is_refused(self):
        """The exact 2026-08-27 case."""
        tp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
        msg = domain_extent_mismatch(tp, {"crop_z_top": 0}, EVAL_ENV_WITHOUT_CROP)
        self.assertIsNotNone(msg)
        self.assertIn("10", msg)
        self.assertIn("0", msg)

    def test_model_uncropped_eval_cropped_is_refused(self):
        """The opposite direction is equally wrong: scoring on less than it saw."""
        msg = domain_extent_mismatch({"env": {}}, {"crop_z_top": 10})
        self.assertIsNotNone(msg)

    def test_stride_difference_alone_is_allowed(self):
        """Trained strided, evaluated full-res is the NORM -- must not be refused."""
        tp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10", "PINO_LAZY_STRIDE": "2,2,1"}}
        self.assertIsNone(domain_extent_mismatch(tp, {"crop_z_top": 10, "stride": [1, 1, 1]}))

    def test_model_predating_the_crop_is_not_refused_against_zero(self):
        """449 of 451 stored models name no crop; scoring them uncropped is correct."""
        self.assertIsNone(domain_extent_mismatch({"env": {}}, {"crop_z_top": 0}))

    def test_missing_crop_key_in_eval_profile_reads_as_zero(self):
        tp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
        self.assertIsNotNone(domain_extent_mismatch(tp, {}, EVAL_ENV_WITHOUT_CROP))

    def test_message_names_the_override(self):
        """A deliberate re-domaining must be told how to proceed."""
        msg = domain_extent_mismatch({"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}, {"crop_z_top": 0},
                                     EVAL_ENV_WITHOUT_CROP)
        self.assertIn("allow_domain_change", msg)

    def test_non_integer_crop_does_not_crash(self):
        self.assertIsNone(domain_extent_mismatch({"env": {"PINO_LAZY_CROP_Z_TOP": "junk"}}, {"crop_z_top": 0}))


if __name__ == "__main__":
    unittest.main()


class EvalEnvCarriesTheCrop(unittest.TestCase):
    """The crop can ride in the eval profile's env instead of the derivation field.

    That is the CORRECT way to run it -- `crop_z_top: 0` in the derivation with the crop in
    the env is how you avoid `_resolve_read_time_crop`'s double-crop refusal. A guard reading
    only the derivation field calls that a mismatch and refuses a configuration that serves
    exactly the right domain. Found before it grounded 12 in-flight arms.
    """

    def test_env_crop_satisfies_the_match(self):
        tp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
        deriv = {"crop_z_top": 0, "stride": [2, 2, 1]}
        evalp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
        self.assertIsNone(domain_extent_mismatch(tp, deriv, evalp))

    def test_env_absent_still_refuses(self):
        tp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
        self.assertIsNotNone(domain_extent_mismatch(tp, {"crop_z_top": 0}, {"env": {}}))

    def test_derivation_crop_still_counts_when_env_is_empty(self):
        tp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
        self.assertIsNone(domain_extent_mismatch(tp, {"crop_z_top": 10}, {"env": {}}))

    def test_env_crop_on_an_uncropped_model_is_refused(self):
        self.assertIsNotNone(
            domain_extent_mismatch({"env": {}}, {"crop_z_top": 0}, {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}))


class BakedCropIsAlsoTheTrainedDomain(unittest.TestCase):
    """A crop BAKED into the training derivation is still the domain the model saw.

    `domain_extent_mismatch` read the crop only from the training profile's env. That is
    correct for read-time crops, but a layout that cannot lazy-load (`moving_ref_frame`)
    MUST bake the crop into the derivation and remove the env var -- so the training profile
    is legitimately crop-free while the model trained on a 30-deep domain. The guard then
    read trained=0 against a served 10 and refused an eval whose domains actually agree.

    Caught 2026-08-29 on the three steady-block crop arms, which would have trained 32000
    steps each and then been refused.
    """

    def test_baked_training_derivation_counts_as_trained_crop(self):
        tp = {"env": {}}                      # crop is NOT in the env: it was baked
        trained_deriv = {"crop_z_top": 10, "moving_ref_frame": True, "time_avg_window": 30}
        self.assertIsNone(domain_extent_mismatch(
            tp, {"crop_z_top": 10}, {}, training_derivation_profile=trained_deriv))

    def test_baked_training_vs_uncropped_eval_still_refuses(self):
        """The eval must carry an env for this to be a real mismatch.

        With NO env key the eval INHERITS the training env, so `served` picks the trained
        crop up and the domains agree by construction -- that inheritance is what cleared 51
        blockers on 2026-08-29 and must not be undone. The genuine mismatch shape is an eval
        env that is PRESENT and silent about the crop, which is exactly how 72419 scored a
        30-deep model on 40 cells.
        """
        tp = {"env": {}}
        trained_deriv = {"crop_z_top": 10, "moving_ref_frame": True}
        msg = domain_extent_mismatch(
            tp, {"crop_z_top": 0}, {"env": {"PINO_LOADER_WORKERS": "2"}},
            training_derivation_profile=trained_deriv)
        self.assertIsNotNone(msg)
        self.assertIn("10", msg)

    def test_env_crop_still_wins_when_no_baked_crop(self):
        tp = {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}}
        self.assertIsNone(domain_extent_mismatch(
            tp, {"crop_z_top": 0}, {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}},
            training_derivation_profile={"crop_z_top": 0}))
