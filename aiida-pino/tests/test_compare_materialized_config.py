"""`compare-models` judges what Hydra MATERIALIZED, not the flag text that produced it.

Reported 2026-08-27: many exact repeats read as confounded although they give the same
model in the end. Two causes, both measured on live pairs.
"""

import unittest

from aiida_pino.utils import model_compare as mc


class FactorCollapseTest(unittest.TestCase):
    def test_the_same_knob_from_two_sources_is_one_factor(self):
        # `override.X` is the input, `config.X` is what Hydra made of it. Counting them
        # separately doubled the factor count of every pair that stated a knob explicitly on
        # one side and inherited it on the other.
        self.assertEqual(mc._factor("override.custom.include_time"), "custom.include_time")
        self.assertEqual(mc._factor("config.custom.include_time"), "custom.include_time")
        self.assertEqual(mc._factor("profile.power_weighted_dose"), "power_weighted_dose")

    def test_the_explicit_map_still_wins(self):
        # Fields that surface under several unrelated names keep their curated factor.
        self.assertEqual(mc._factor("override.seed"), "seed")
        self.assertEqual(mc._factor("steps_trained"), "training length")

    def test_an_unprefixed_field_is_unchanged(self):
        self.assertEqual(mc._factor("gpu"), "gpu")


class OverrideSpellingTest(unittest.TestCase):
    """An override that resolves to the same value is spelling, not an experiment."""

    def _facts(self, overrides, config, archived=True):
        return {"overrides": overrides, "config_flat": config if archived else None,
                "env": {}, "has_calcjob": True, "profile": {}}

    def test_diff_maps_can_be_forced_to_a_severity(self):
        rows = mc._diff_maps("override", {"a": "1"}, {}, set(), severity="benign")
        self.assertEqual([r["severity"] for r in rows], ["benign"])

    def test_the_absent_sentinel_is_not_a_stored_none(self):
        # A key genuinely set to None must not compare equal to a key that is missing.
        self.assertIsNot(mc._ABSENT, None)


class VerdictTest(unittest.TestCase):
    def test_zero_confounds_with_unknowns_is_not_confounded(self):
        # The bug behind the report. `not confound_factors and unknown_factors` fell into
        # the CONFOUNDED branch and printed "0 confounded factor(s)" -- the strongest word in
        # this vocabulary on the weakest evidence. Of six pairs whose materialized configs
        # are byte-identical, three read that way.
        import inspect
        src = inspect.getsource(mc.compare_models)
        self.assertIn("elif not confound_factors:", src)
        self.assertIn("NO MEASURED DIFFERENCE", src)
        # And it must sit BEFORE the catch-all, or it can never be reached.
        self.assertLess(src.index("elif not confound_factors:"),
                        src.index('verdict = "CONFOUNDED: "'))

    def test_the_clean_branch_still_requires_no_unknowns(self):
        # "CLEAN" must keep meaning "nothing differs AND everything was checkable"; the new
        # branch is for the weaker claim, and collapsing them would overstate the evidence.
        import inspect
        src = inspect.getsource(mc.compare_models)
        self.assertIn("if not confound_factors and not unknown_factors:", src)


if __name__ == "__main__":
    unittest.main()
