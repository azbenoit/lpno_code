"""A finish_time_s that falls a picosecond short must say so as a DEFICIT.

On 2026-08-20, 8 cases of the 150-case `ti64_rand150_newphys` sweep failed config
validation with "finish_time_s is shorter than the laser travel time plus stop
event". The generator had derived finish_time from full-precision draws while
writing length_m and scan_speed_m_per_s rounded to 9 and 6 decimals, then rounded
the quotient DOWN -- leaving it short by 7e-11 to 1e-9 s.

The check itself is right and is deliberately not weakened here: those configs do
describe a path fractionally longer than the run. What was wrong is that both
numbers printed at %g as near-identical 6-significant-digit values, so a rounding
artifact was indistinguishable from a real design error and cost a diagnosis.

These tests pin the deficit being reported and the check still firing.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_rectangular_case as rectangular  # noqa: E402

SMOKE_CONFIG = ROOT / "configs" / "rectangular_moving_laser_smoke.json"


def _validate(cfg, name="deficit_check"):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cfg.json"
        p.write_text(json.dumps(cfg))
        return rectangular.load_and_validate(p, name)


class FinishTimeDeficitTest(unittest.TestCase):
    def _short_by(self, deficit_s):
        """A config whose finish_time_s is `deficit_s` under the travel time."""
        cfg = json.loads(SMOKE_CONFIG.read_text())
        laser = cfg["laser"]
        laser["turn_off_at_end"] = True
        start = laser.get("start_point_m") or [-1.0e-4, 0.0]
        end = laser.get("end_point_m") or [1.0e-4, 0.0]
        laser["start_point_m"], laser["end_point_m"] = start, end
        length = abs(end[0] - start[0])
        speed = laser["scan_speed_m_per_s"]
        cfg["simulation"]["finish_time_s"] = length / speed - deficit_s
        return cfg

    def test_tiny_deficit_still_rejected(self):
        """Not weakened: a config short of the travel time is still an error."""
        with self.assertRaises(rectangular.ConfigError):
            _validate(self._short_by(1.0e-9))

    def test_message_states_the_deficit(self):
        """The magnitude must be visible -- that is what %g was hiding."""
        with self.assertRaises(rectangular.ConfigError) as ctx:
            _validate(self._short_by(1.0e-9))
        msg = str(ctx.exception)
        self.assertIn("by ", msg)
        self.assertIn("e-", msg, "deficit should print in exponential form")
        self.assertIn("rounded down", msg,
                      "message should name the likely cause for a sub-timestep deficit")


    def test_message_states_the_value_that_would_pass(self):
        """A generator that rounds up must know BY HOW MUCH to clear the guard.

        The 2026-08-22 recurrence, 15 cases of the same 150-case sweep: the
        generator followed this message's own advice -- derive from the length
        and speed as written and round up -- by rounding the quotient up to a
        1e-12 grid. The guard's tolerance floor is ALSO 1e-12, so a round-up to
        that grid yields a margin between 0 and 1e-12 while the guard demands
        >= 1e-12. Rounding up on a grid no coarser than the required margin is
        not enough, and "round it up" does not say that.

        Naming the value that WOULD pass makes the target unambiguous: a
        generator can round up to any grid and compare.
        """
        with self.assertRaises(rectangular.ConfigError) as ctx:
            _validate(self._short_by(1.0e-9))
        msg = str(ctx.exception)
        self.assertIn("at least", msg,
                      "message should state the minimum finish_time_s that passes")
        # The stated minimum must itself be accepted, or the advice is wrong.
        import re
        m = re.search(r"at least ([0-9.eE+-]+)", msg)
        self.assertIsNotNone(m, f"no minimum found in: {msg}")
        cfg = self._short_by(1.0e-9)
        cfg["simulation"]["finish_time_s"] = float(m.group(1))
        case, replacements, manifest = _validate(cfg)
        self.assertIsNotNone(manifest, "the value the message names must validate")

    def test_adequate_finish_time_passes(self):
        """The obvious control: a comfortably long finish_time is accepted."""
        cfg = self._short_by(-1.0e-5)      # 10 us LONGER than the travel time
        case, replacements, manifest = _validate(cfg)
        self.assertIsNotNone(manifest)


if __name__ == "__main__":
    unittest.main()
