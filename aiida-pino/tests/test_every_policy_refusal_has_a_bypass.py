"""A refusal that reflects a JUDGEMENT must be overridable (decided 2026-09-07).

"Keep warning, don't refuse. And any refusal should be possible to bypass."

Two kinds of refusal live in the eval path and only one is covered by that rule:

- FACTUAL — the job cannot run at all. Exit 303 (the dataset has no such variant; a ragged
  dataset has no `time_as_dim` form) and exit 307 (the collection is not staged on the target
  box, and a remote job cannot build it). Overriding these would just move the failure later,
  so they stay.
- POLICY — a judgement about what is wise. Exit 306 domain extent (`allow_domain_change`) and
  the train/test overlap guard (`allow_training_data`) already had escape hatches. The eval
  VRAM guard did NOT, and it is the one with a recorded cost: one model declared 14000 by
  hand "to skip the whole-card smoke", was refused at 8000, and was forced to a figure nothing
  on the box could free -- so it could not start at all, with no way to confirm the booking was intended.

Since 2026-09-06 that guard CORRECTS a measured under-booking rather than refusing. What
remains is the case with no measured peak, where it still refuses on the training BOOKING --
"whatever a human or a clone put there" -- which is exactly the 66067 shape.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.model_registry import (
    EVALUATION_OVERRIDE_KEYS,
    _check_graph_eval_vram_against_training as check,
    validate_evaluation_profile,
)

RIGNO = {"config_name": "config_flow3D_RIGNO_randompath", "gpu_vram_mb": 14000}


class BypassTest(unittest.TestCase):
    def test_the_vram_refusal_can_be_bypassed(self):
        merged = {"gpu_vram_mb": 8000, "time_avg_window": 1, "allow_vram_below_training": True}
        check(RIGNO, merged, None)  # no measured peak: the refusing branch
        self.assertEqual(merged["gpu_vram_mb"], 8000, "the bypass must not also rewrite it")

    def test_without_the_bypass_it_still_refuses(self):
        merged = {"gpu_vram_mb": 8000, "time_avg_window": 1}
        with self.assertRaises(ValueError):
            check(RIGNO, merged, None)

    def test_the_bypass_is_an_accepted_profile_key(self):
        self.assertIn("allow_vram_below_training", EVALUATION_OVERRIDE_KEYS)
        got = validate_evaluation_profile({"allow_vram_below_training": True})
        self.assertIs(got["allow_vram_below_training"], True)

    def test_it_must_be_a_boolean_not_a_truthy_string(self):
        """`bool("false")` is True; a silent misread here defeats the guard entirely."""
        with self.assertRaises(ValueError):
            validate_evaluation_profile({"allow_vram_below_training": "maybe"})

    def test_the_bypass_does_not_disable_the_measured_correction(self):
        """Opting out of a REFUSAL is not opting out of a correction that cannot hurt.

        Raising a booking to a figure that was actually measured only makes the job more
        likely to run, so the bypass deliberately does not suppress it.
        """
        merged = {"gpu_vram_mb": 4000, "time_avg_window": 1, "allow_vram_below_training": True}
        check(RIGNO, merged, 9000)
        self.assertEqual(merged["gpu_vram_mb"], 9000)


class FactualRefusalsStayTest(unittest.TestCase):
    def test_staging_and_variant_refusals_have_no_bypass_key(self):
        """Recorded as a decision, not an oversight: these are not judgements."""
        for key in ("allow_unstaged", "allow_missing_variant"):
            self.assertNotIn(key, EVALUATION_OVERRIDE_KEYS)


if __name__ == "__main__":
    unittest.main()
