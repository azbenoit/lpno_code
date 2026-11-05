#!/usr/bin/env python3
"""Stationary (dwell) laser: `laser.dwell_time_s` with a single beam position.

A stationary spot is the standard experimental keyhole-onset test (e.g. Simonds
et al. 2021: vapour-depression onset ~96 W, keyhole onset ~120 W at 2 ms). Before
this, a zero-length path was rejected outright and the only workaround was a
quasi-stationary crawl, which is a different experiment.
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


def _config(**laser_over):
    cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
    cfg["laser"].update(laser_over)
    return cfg


class StationaryLaserTests(unittest.TestCase):
    def _render(self, cfg):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "case.json"
        path.write_text(json.dumps(cfg))
        _, replacements, manifest = rectangular.load_and_validate(path, "stat_case", None, None)
        return replacements, manifest

    def test_dwell_renders_zero_velocity_and_dwell_duration(self) -> None:
        cfg = _config(dwell_time_s=0.002, start_point_m=[0.0, 0.0, 0.0])
        cfg["laser"].pop("end_point_m", None)
        cfg["simulation"]["finish_time_s"] = 0.0023
        repl, manifest = self._render(cfg)
        d = manifest["derived"]
        self.assertEqual(d["scan_path_length_m"], 0.0)
        self.assertAlmostEqual(d["scan_time_s"], 0.002)
        self.assertTrue(d["laser_stationary"])
        # Beam tables now live in one __LASER_BEAM_BLOCK__ (3be3c0c).
        vel = repl["__LASER_BEAM_BLOCK__"]
        for line in vel.splitlines():
            if "utlbm" in line or "vtlbm" in line:
                self.assertIn("=0", line.replace(" ", ""), msg=vel)

    def test_dwell_requires_finish_time_to_cover_it(self) -> None:
        cfg = _config(dwell_time_s=0.002, start_point_m=[0.0, 0.0, 0.0])
        cfg["laser"].pop("end_point_m", None)
        cfg["simulation"]["finish_time_s"] = 0.001
        with self.assertRaises(rectangular.ConfigError):
            self._render(cfg)

    def test_dwell_rejects_a_moving_endpoint(self) -> None:
        cfg = _config(dwell_time_s=0.002, start_point_m=[0.0, 0.0, 0.0],
                      end_point_m=[0.0002, 0.0, 0.0])
        cfg["simulation"]["finish_time_s"] = 0.0023
        with self.assertRaises(rectangular.ConfigError):
            self._render(cfg)

    def test_moving_case_still_requires_distinct_points(self) -> None:
        cfg = _config(start_point_m=[0.0, 0.0, 0.0], end_point_m=[0.0, 0.0, 0.0])
        with self.assertRaises(rectangular.ConfigError):
            self._render(cfg)


if __name__ == "__main__":
    unittest.main()


class StationaryExportTests(unittest.TestCase):
    """A dwell manifest must survive the export path.

    Regression for 2026-07-28: `dwell_ti64_90w` solved successfully and then
    died with KeyError: 'scan_speed_m_per_s' during the NPZ export, because a
    stationary case legitimately omits that key. The solve had already
    completed.
    """

    def test_manifest_laser_block_has_no_scan_speed_for_a_dwell(self) -> None:
        cfg = _config(dwell_time_s=0.002, start_point_m=[0.0, 0.0, 0.0])
        cfg["laser"].pop("end_point_m", None)
        cfg["laser"].pop("scan_speed_m_per_s", None)
        cfg["simulation"]["finish_time_s"] = 0.0023
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "case.json"
        path.write_text(json.dumps(cfg))
        _, _, manifest = rectangular.load_and_validate(path, "stat_case", None, None)
        laser = manifest["input"]["laser"]
        # The export must read it defensively; assert the shape that broke it.
        self.assertNotIn("scan_speed_m_per_s", laser)
        self.assertEqual(laser.get("scan_speed_m_per_s", 0.0), 0.0)
        self.assertIn("power_w", laser)
