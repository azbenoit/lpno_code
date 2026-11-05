"""An eval records the grid it ACTUALLY served, measured rather than recomputed.

Root cause found 2026-09-09. Nine evals on one test subset all record `derivation_profile.crop_z_top = 0`; three of
them demonstrably served 90x40x30 with crop 10, and only the calcjob log said so. The
corroborating observable was `beambox_n` 397546 against 680638 -- a 1.712 ratio visible in the
metrics while every metadata field claimed the domains matched.

ROOT CAUSE. `derivation_profile` on the eval node is the derivation REQUESTED for the dataset.
The crop that actually applied came from somewhere else: the eval inherits the model's training
env, and `ctx.demoted_env` is overlaid ON TOP of it after the merge, so a crop the submitter
explicitly blanked is put back. Verified on the four nodes -- the three that cropped have eval
env `PINO_LAZY_CROP_Z_TOP = ''` (blanked by the submitter) while their MODEL env carries '10'
and `training_crop_z_top` is 10; the six that did not have None on both. So the one field that
distinguishes the groups spells NO crop on exactly the three that cropped. The profile is not
lying, it is answering a different question than every reader assumes.

THE FIX IS A CAPTURE, NOT A COMPUTATION. lpno `4722344` already records what the
store served (`NpzDirStore.served_grid`) into the metrics payload. This consumes it: the eval
node now carries `served_grid`, and `eval_stride` / `eval_crop_z_top` are taken from it when
present. A measured value beats a computed one -- the same rule that settled the eval_stride
labels on 2026-09-05, where a computed label disagreed with counted cells and the count won.

Historical evals cannot be repaired from their nodes: they predate the upstream capture, so
their log remains the only record. That is stated rather than papered over.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.model_registry import served_grid_facts

SERVED = {"nx": 90, "ny": 40, "nz": 30, "raw": [90, 40, 40],
          "stride": [1, 1, 1], "crop_z_top": 10, "moving_window": None}


class ServedGridTest(unittest.TestCase):
    def test_it_reads_the_grid_from_the_metrics_payload(self):
        facts = served_grid_facts({"served_grid": dict(SERVED)})
        self.assertEqual(facts["served_grid"]["nz"], 30)
        self.assertEqual(facts["eval_crop_z_top"], 10)
        self.assertEqual(facts["eval_stride"], [1, 1, 1])

    def test_absent_is_None_not_a_guessed_zero(self):
        """A pre-4722344 eval has no capture; unknown must not read as 'no crop'."""
        facts = served_grid_facts({})
        self.assertIsNone(facts["served_grid"])
        self.assertIsNone(facts["eval_crop_z_top"])
        self.assertIsNone(facts["eval_stride"])

    def test_a_malformed_capture_is_ignored_rather_than_fatal(self):
        for bad in ({"served_grid": "90x40x30"}, {"served_grid": {"nz": "x"}},
                    {"served_grid": {"crop_z_top": None}}, {"served_grid": []}):
            facts = served_grid_facts(bad)
            self.assertIsNone(facts["eval_crop_z_top"], f"{bad!r} produced a value")

    def test_it_looks_in_the_summary_too(self):
        facts = served_grid_facts({}, eval_summary={"served_grid": dict(SERVED)})
        self.assertEqual(facts["eval_crop_z_top"], 10)

    def test_the_measured_crop_survives_a_blanked_env(self):
        """The exact failure: submitter blanked the crop, demotion put it back, node said 0."""
        facts = served_grid_facts({"served_grid": dict(SERVED)})
        self.assertEqual(facts["eval_crop_z_top"], 10,
                         "the node would still claim an uncropped domain")


class WiringTest(unittest.TestCase):
    def test_the_recorder_stamps_them(self):
        import inspect

        from aiida_pino.utils.model_registry import build_evaluation_metadata

        src = inspect.getsource(build_evaluation_metadata)
        self.assertIn("served_grid_facts", src)

    def test_the_extras_writer_exposes_the_crop(self):
        import inspect

        from aiida_pino.data import pino_model_evaluation as mod

        src = inspect.getsource(mod)
        self.assertIn("eval_crop_z_top", src)
        self.assertIn("served_grid", src)


if __name__ == "__main__":
    unittest.main()
