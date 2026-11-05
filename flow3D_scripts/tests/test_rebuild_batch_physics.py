#!/usr/bin/env python3
"""Rebuilding an old batch's configs onto the calibrated recipe.

The 2026-07-29 review found EVERY sweep config in the repo still renders
pre-calibration physics (no recoil_model, no pv0, old Ti64 anchor). Any
regeneration must rewrite them, and hand-patching 84-300 files per batch is
how silent misses happen.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import rebuild_batch_physics as rbp  # noqa: E402


class RebuildBatchPhysicsTests(unittest.TestCase):
    def _src(self, tmp: Path) -> Path:
        d = tmp / "src" / "configs"
        d.mkdir(parents=True)
        (d / "train_100W_0241mps.json").write_text(json.dumps({
            "schema_version": 1,
            "material": {"name": "ti_imi318_mills"},
            "laser": {"power_w": 100.0, "scan_speed_m_per_s": 0.241, "beam_radius_m": 5e-05},
            "simulation": {"finish_time_s": 0.001, "solver_profile": "legacy_case_fine_1"},
        }))
        return d.parent

    def test_applies_recipe_and_keeps_case_identity(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = self._src(tmp)
            out = tmp / "out"
            n = rbp.rebuild(src, out, material="/mat/ti64_v3_calibrated.json",
                            recoil_model=2, void_vapor_pressure_pa=101325.0)
            self.assertEqual(n, 1)
            c = json.loads((out / "configs" / "train_100W_0241mps.json").read_text())
            self.assertEqual(c["simulation"]["recoil_model"], 2)
            self.assertEqual(c["simulation"]["void_vapor_pressure_pa"], 101325.0)
            self.assertEqual(c["material"], {"file": "/mat/ti64_v3_calibrated.json"})
            # identity and process conditions must survive untouched
            self.assertEqual(c["laser"]["power_w"], 100.0)
            self.assertEqual(c["laser"]["scan_speed_m_per_s"], 0.241)
            self.assertEqual(c["case_name"], "train_100W_0241mps")

    def test_drops_a_pinned_vapor_pressure_b_so_the_material_governs(self) -> None:
        """A config that pins the OLD saturation curve must not keep it.

        101 of the bundle_gate configs hardcode `vapor_pressure_b: 18.120188`
        (tv1 = 3123.15 K). That override beats the material file, so rebuilding
        onto ti64_v3_calibrated (tv1 = 3520) while leaving it in place renders
        the pre-campaign vapour curve under a calibrated material name -- the
        exact silent miss this tool exists to prevent.
        """
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = self._src(tmp)
            cfg = src / "configs" / "train_100W_0241mps.json"
            c = json.loads(cfg.read_text())
            c["laser"]["vapor_pressure_b"] = 18.120188
            cfg.write_text(json.dumps(c))
            out = tmp / "out"
            rbp.rebuild(src, out, material="/mat/ti64_v3_calibrated.json",
                        recoil_model=2, void_vapor_pressure_pa=101325.0)
            got = json.loads((out / "configs" / "train_100W_0241mps.json").read_text())
            self.assertNotIn("vapor_pressure_b", got["laser"])

    def test_explicit_vapor_pressure_b_still_pins_the_curve(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = self._src(tmp)
            out = tmp / "out"
            rbp.rebuild(src, out, material="/mat/ti64_v3_calibrated.json",
                        recoil_model=2, void_vapor_pressure_pa=101325.0,
                        vapor_pressure_b=10.95913)
            got = json.loads((out / "configs" / "train_100W_0241mps.json").read_text())
            self.assertEqual(got["laser"]["vapor_pressure_b"], 10.95913)

    def test_strips_retired_plume_fields(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = self._src(tmp)
            c = json.loads((src / "configs" / "train_100W_0241mps.json").read_text())
            c["laser"]["plume_attenuation"] = 0.2
            c["laser"]["plume_radius_m"] = 3.8e-4
            (src / "configs" / "train_100W_0241mps.json").write_text(json.dumps(c))
            out = tmp / "out"
            rbp.rebuild(src, out, material="/mat/m.json", recoil_model=2,
                        void_vapor_pressure_pa=101325.0)
            got = json.loads((out / "configs" / "train_100W_0241mps.json").read_text())
            self.assertNotIn("plume_attenuation", got["laser"])
            self.assertNotIn("plume_radius_m", got["laser"])

    def test_reports_what_it_changed(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = self._src(tmp)
            out = tmp / "out"
            rbp.rebuild(src, out, material="/mat/m.json", recoil_model=2,
                        void_vapor_pressure_pa=101325.0)
            manifest = json.loads((out / "rebuild_manifest.json").read_text())
            self.assertEqual(manifest["cases"], 1)
            self.assertEqual(manifest["recipe"]["recoil_model"], 2)
            self.assertIn("source", manifest)


if __name__ == "__main__":
    unittest.main()
