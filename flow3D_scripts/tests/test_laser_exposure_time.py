"""The manifest's laser timing must match the deck's, not the config's intent.

`laser.turn_off_at_end: false` renders a laser time table with ONE row. FLOW-3D
holds the last row, so the laser stays at full power -- and, for a scan, keeps
moving at constant velocity -- until `simulation.finish_time_s`. When finish_time
exceeds the declared travel or dwell, the solve delivers more exposure than the
config describes, and a dwell's pulse is simply not the length `dwell_time_s`
says it is.

The manifest recorded `laser_off_time_s = scan_time` regardless: a turn-off that
never happens. That is the 2026-08-10 preheat failure mode again -- manifest
records intent, prepin is actuality, downstream believes the manifest. Found
2026-08-15 in 67 tracked validation configs, which run 15-26 % longer than they
declare.

These tests pin the manifest to the deck.
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


def _render(mutate, name):
    cfg = json.loads(SMOKE_CONFIG.read_text())
    mutate(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cfg.json"
        p.write_text(json.dumps(cfg))
        case, replacements, manifest = rectangular.load_and_validate(p, name)
        text = rectangular.render(Path(tmp) / "case", case, replacements,
                                  manifest).read_text(encoding="utf-8")
    return text, manifest.get("derived", manifest), cfg


def _as_dwell(cfg, dwell_s, finish_s, turn_off):
    laser = cfg["laser"]
    laser.pop("end_point_m", None)
    laser.pop("scan_speed_m_per_s", None)
    laser["start_point_m"] = [0.0, 0.0, 0.0]
    laser["dwell_time_s"] = dwell_s
    laser["turn_off_at_end"] = turn_off
    cfg["simulation"]["finish_time_s"] = finish_s


class LaserExposureTimeTest(unittest.TestCase):
    def test_dwell_without_turn_off_reports_the_real_exposure(self):
        """A 1 ms dwell run for 1.15 ms is a 1.15 ms pulse. Say so."""
        text, d, _ = _render(lambda c: _as_dwell(c, 1.0e-3, 1.15e-3, False),
                             "exposure_dwell_noff")
        self.assertEqual(text.count("timlbm("), 1,
                         "turn_off_at_end false must render a single laser time row")
        self.assertAlmostEqual(d["laser_exposure_time_s"], 1.15e-3, places=9)
        self.assertIsNone(d["laser_off_time_s"],
                          "the laser never turns off, so there is no turn-off time")
        self.assertTrue(d["laser_on_past_declared_time"])

    def test_dwell_with_turn_off_matches_the_dwell(self):
        text, d, _ = _render(lambda c: _as_dwell(c, 1.0e-3, 1.15e-3, True),
                             "exposure_dwell_off")
        self.assertEqual(text.count("timlbm("), 3)
        self.assertIn("powlbm(3,1)=0.0", text)
        self.assertAlmostEqual(d["laser_exposure_time_s"], 1.0e-3, places=9)
        self.assertAlmostEqual(d["laser_off_time_s"], 1.0e-3, places=6)
        self.assertFalse(d["laser_on_past_declared_time"])

    def test_scan_without_turn_off_records_the_overrun(self):
        """The beam also keeps MOVING, so record how far past the end it gets."""
        def mutate(cfg):
            cfg["laser"]["turn_off_at_end"] = False
            cfg["laser"]["start_point_m"] = [-2.0e-4, 0.0, 0.0]
            cfg["laser"]["end_point_m"] = [2.0e-4, 0.0, 0.0]
            cfg["laser"]["scan_speed_m_per_s"] = 0.4          # travel = 1.0 ms
            cfg["simulation"]["finish_time_s"] = 1.25e-3

        _, d, _ = _render(mutate, "exposure_scan_noff")
        self.assertAlmostEqual(d["laser_exposure_time_s"], 1.25e-3, places=9)
        self.assertTrue(d["laser_on_past_declared_time"])
        # 0.25 ms of extra travel at 0.4 m/s = 100 um past the declared end.
        self.assertAlmostEqual(d["laser_overrun_distance_m"], 1.0e-4, places=7)

    def test_exposure_is_never_a_fiction(self):
        """Whatever the branch, the field is the deck's ON-time, not the intent."""
        for turn_off in (True, False):
            _, d, cfg = _render(
                lambda c, t=turn_off: c["laser"].__setitem__("turn_off_at_end", t),
                f"exposure_inv_{turn_off}")
            expect = d["scan_time_s"] if turn_off else cfg["simulation"]["finish_time_s"]
            self.assertAlmostEqual(d["laser_exposure_time_s"], expect, places=9,
                                   msg=f"turn_off_at_end={turn_off}")


if __name__ == "__main__":
    unittest.main()
