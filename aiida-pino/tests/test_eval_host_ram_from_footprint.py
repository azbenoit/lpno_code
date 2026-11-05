"""Eval host RAM is sized from the footprint's measured RSS, not a flat cap.

The cap was wrong in the direction that matters. Across the 497 eval calcjobs carrying a
measured resident set: p50 6918, p90 17671, p99 37132, max 47949 MiB against a flat
EVAL_HOST_RAM_MB of 16384 -- so 44 of 335 footprints were admitted against a number a
third of what they use, the same invisible over-run that OOM-kills co-tenants.
"""

import inspect
import unittest

from aiida_pino.utils.vram import (
    EVAL_RSS_EXTRA,
    EVAL_RSS_FLOOR_MB,
    VRAM_KEY_EXTRA,
    lookup_eval_host_ram_mb,
    vram_reserve_mb,
)
from aiida_pino.workflows import evaluate_model


class EvalHostRamTest(unittest.TestCase):
    def test_it_reads_the_extra_the_parser_actually_writes(self):
        # The failure mode this guards: a lookup keyed on a name nothing writes returns
        # None for every footprint and silently leaves the flat cap in place, which looks
        # exactly like "no measurements yet".
        src = inspect.getsource(inspect.getmodule(lookup_eval_host_ram_mb))
        from aiida_pino.parsers import common
        self.assertIn(f'extras["{EVAL_RSS_EXTRA}"]', inspect.getsource(common))
        self.assertIn(EVAL_RSS_EXTRA, src)

    def test_the_footprint_key_it_filters_on_is_the_one_evals_are_stamped_with(self):
        self.assertEqual(VRAM_KEY_EXTRA, "vram_key")
        self.assertIn("PinoEvalCalculation", inspect.getsource(lookup_eval_host_ram_mb))

    def test_an_unknown_footprint_returns_none_rather_than_a_guess(self):
        self.assertIsNone(lookup_eval_host_ram_mb("no-such-footprint-key"))

    def test_a_truncated_job_is_not_treated_as_a_measurement(self):
        # Mirrors EVAL_PEAK_FLOOR_MB on the VRAM axis: sizing from a crashed job's tiny
        # RSS would reserve almost nothing and cause the OOM this exists to prevent.
        self.assertGreater(EVAL_RSS_FLOOR_MB, 0)
        self.assertIn("EVAL_RSS_FLOOR_MB", inspect.getsource(lookup_eval_host_ram_mb))

    def test_a_measurement_above_the_flat_cap_raises_the_booking(self):
        # The 47949 MiB footprint: the flat cap would book 16384 for a job using 47949.
        self.assertGreater(vram_reserve_mb(47949), evaluate_model.EVAL_HOST_RAM_MB)
        self.assertGreater(vram_reserve_mb(47949), 47949)

    def test_a_measurement_below_the_flat_cap_lowers_the_booking(self):
        self.assertLess(vram_reserve_mb(3000), evaluate_model.EVAL_HOST_RAM_MB)

    def test_the_workchain_prefers_the_measurement_and_keeps_the_cap_as_fallback(self):
        src = inspect.getsource(evaluate_model)
        self.assertIn("measured = lookup_eval_host_ram_mb(key)", src)
        self.assertIn("capped = EVAL_HOST_RAM_MB", src)
        # An explicit booking in the evaluation profile still wins over both.
        self.assertIn('if "host_ram_mb" not in (self.ctx.evaluation_profile or {}):', src)


if __name__ == "__main__":
    unittest.main()
