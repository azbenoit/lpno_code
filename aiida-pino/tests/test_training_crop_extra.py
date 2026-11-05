"""A model trained on a cropped domain must SAY so, in a queryable extra.

The stride story, repeating on a second axis: 18 of 158 models trained at 2x2x1 while
nothing on the node said so, which is why `training_stride` exists. Since the gas crop
became the default, every new model trains on 30 of 40 z-layers, and the only records were
`training_profile.env` (an attribute, invisible to a QueryBuilder extras filter) and
`trained_grid.json` in the artifacts. The question (2026-08-28) was whether a default crop gets logged; the answer
was "not queryably", so now it is.
"""

import unittest

from aiida_pino.data.pino_model import _effective_crop_z_top, _env_crop, queryable_model_extras


def _extras(profile, metadata=None):
    md = dict(metadata or {})
    md["training_profile"] = profile or {}
    return queryable_model_extras(md)


class CropExtraTest(unittest.TestCase):
    def test_the_default_gas_crop_is_stamped_from_the_env(self):
        extras = _extras({"env": {"PINO_LAZY_CROP_Z_TOP": "10"}})
        self.assertEqual(extras["training_crop_z_top"], 10)
        self.assertEqual(extras["training_crop_source"], "env.PINO_LAZY_CROP_Z_TOP")

    def test_a_baked_crop_is_stamped_from_the_derivation(self):
        extras = _extras({}, {"derivation_profile": {"crop_z_top": 10}})
        self.assertEqual(extras["training_crop_z_top"], 10)
        self.assertEqual(extras["training_crop_source"], "derivation_profile.crop_z_top")

    def test_the_env_wins_when_both_are_present(self):
        # Same precedence as the stride: the env var is applied by the datasource itself.
        self.assertEqual(_effective_crop_z_top(
            {"env": {"PINO_LAZY_CROP_Z_TOP": "10"}},
            {"derivation_profile": {"crop_z_top": 0}}), 10)

    def test_unknown_stays_unknown(self):
        # A pre-crop model and a defaulted import both assert nothing; stamping 0 would
        # claim "trained uncropped" on no evidence -- the exact mistake the stride rule
        # exists to prevent.
        self.assertNotIn("training_crop_z_top", _extras({}))
        self.assertNotIn("training_crop_z_top", _extras(
            {}, {"derivation_profile": {"crop_z_top": 10},
                 "training_summary": {"derivation_profile_defaulted": True}}))

    def test_an_explicit_zero_is_a_recorded_fact_not_an_absence(self):
        extras = _extras({"env": {"PINO_LAZY_CROP_Z_TOP": "0"}})
        self.assertEqual(extras["training_crop_z_top"], 0)

    def test_full_resolution_still_means_stride_only(self):
        # Documented contract: its consumers read it as "no stride". A cropped, unstrided
        # model stays full_resolution=True and carries the crop in its own extra.
        extras = _extras({"env": {"PINO_LAZY_CROP_Z_TOP": "10", "PINO_LAZY_STRIDE": "1,1,1"}})
        self.assertTrue(extras["full_resolution"])
        self.assertEqual(extras["training_crop_z_top"], 10)

    def test_garbage_env_values_stamp_nothing(self):
        self.assertIsNone(_env_crop({"env": {"PINO_LAZY_CROP_Z_TOP": "junk"}}))
        self.assertNotIn("training_crop_z_top", _extras({"env": {"PINO_LAZY_CROP_Z_TOP": "junk"}}))


if __name__ == "__main__":
    unittest.main()
