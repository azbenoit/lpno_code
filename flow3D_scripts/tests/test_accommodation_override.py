#!/usr/bin/env python3
"""`simulation.accommodation_coefficient` overrides the solver profile's RSIZE.

RSIZE is FLOW-3D's evaporation accommodation coefficient. It sets the phase-change
rate AND, through the documented recoil recipe `A = RSIZE * PV1`, the drilling
pressure amplitude. The `legacy_case_fine_1` profile hardcodes 1.0 (the maximum);
literature values are 0.1-0.5, and the 2026-07-28 validation showed the keyhole
regime over-drilling, so it has to be tunable to calibrate against experiment.
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


def _render(cfg):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "case.json"
    path.write_text(json.dumps(cfg))
    try:
        return rectangular.load_and_validate(path, "acc_case", None, None)
    finally:
        tmp.cleanup()


def _config(**sim_over):
    cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
    cfg["simulation"].update(sim_over)
    return cfg


class AccommodationOverrideTests(unittest.TestCase):
    def test_default_keeps_the_profile_value(self) -> None:
        _, repl, manifest = _render(_config())
        self.assertEqual(repl["__RSIZE__"], "1")
        self.assertIsNone(manifest["derived"].get("accommodation_coefficient"))

    def test_override_replaces_rsize_and_is_recorded(self) -> None:
        _, repl, manifest = _render(_config(accommodation_coefficient=0.25))
        self.assertEqual(float(repl["__RSIZE__"]), 0.25)
        self.assertAlmostEqual(manifest["derived"]["accommodation_coefficient"], 0.25)

    def test_override_also_drives_rsize_cond(self) -> None:
        _, repl, _ = _render(_config(accommodation_coefficient=0.25))
        line = repl["__RSIZE_COND_LINE__"]
        if line.strip():
            self.assertIn("0.25", line)

    def test_rejects_non_positive_and_above_one(self) -> None:
        for bad in (0.0, -0.5, 1.5):
            with self.subTest(bad=bad), self.assertRaises(rectangular.ConfigError):
                _render(_config(accommodation_coefficient=bad))


if __name__ == "__main__":
    unittest.main()


class PlumeAttenuationTests(unittest.TestCase):
    """`laser.plume_attenuation` overrides APLUME (default 0.0 = transparent plume).

    APLUME is the fraction of beam power diverted into the vapour plume. The
    profiles ship 0.0, i.e. every watt reaches the surface, which is unphysical
    in the keyhole regime and a candidate for the over-absorption found on
    2026-07-28.
    """

    def _laser_config(self, **laser_over):
        cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        cfg["laser"].update(laser_over)
        return cfg

    def test_default_keeps_transparent_plume(self) -> None:
        _, repl, manifest = _render(self._laser_config())
        self.assertIn("aplume=0.0", repl["__WELD_LEGACY_AFTER_REFLECTION__"])
        self.assertIsNone(manifest["derived"].get("plume_attenuation"))

    def test_override_is_rendered_and_recorded(self) -> None:
        _, repl, manifest = _render(self._laser_config(plume_attenuation=0.2))
        block = repl["__WELD_LEGACY_AFTER_REFLECTION__"]
        self.assertIn("aplume=0.2", block.replace(" ", ""))
        self.assertNotIn("aplume=0.0", block)
        self.assertAlmostEqual(manifest["derived"]["plume_attenuation"], 0.2)

    def test_rejects_a_full_or_negative_fraction(self) -> None:
        for bad in (1.0, 1.5, -0.1):
            with self.subTest(bad=bad), self.assertRaises(rectangular.ConfigError):
                _render(self._laser_config(plume_attenuation=bad))


class PlumeRadiusTests(unittest.TestCase):
    """`laser.plume_radius_m` (RFPLUME) is what makes APLUME do anything.

    APLUME does not delete energy: the docs say the attenuated fraction becomes
    "an additional heat source" returned to the fluid inside the RFPLUME
    cylinder. With rfplume = 0 (the profile default) there is nowhere to return
    it, and on 2026-07-28 aplume 0.15 vs 0.30 gave bit-identical results.
    """

    def _cfg(self, **laser_over):
        cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        cfg["laser"].update(laser_over)
        return cfg

    def test_radius_is_rendered_with_attenuation(self) -> None:
        _, repl, manifest = _render(self._cfg(plume_attenuation=0.2, plume_radius_m=1.0e-4))
        block = repl["__WELD_LEGACY_AFTER_REFLECTION__"].replace(" ", "")
        self.assertIn("rfplume=0.0001", block)
        self.assertIn("aplume=0.2", block)
        self.assertNotIn("rfplume=0,", block)
        self.assertAlmostEqual(manifest["derived"]["plume_radius_m"], 1.0e-4)

    def test_radius_without_attenuation_is_rejected(self) -> None:
        with self.assertRaises(rectangular.ConfigError):
            _render(self._cfg(plume_radius_m=1.0e-4))

    def test_default_keeps_the_inert_profile_plume(self) -> None:
        _, repl, manifest = _render(self._cfg())
        self.assertIn("rfplume=0", repl["__WELD_LEGACY_AFTER_REFLECTION__"].replace(" ", ""))
        self.assertIsNone(manifest["derived"].get("plume_radius_m"))


class PlumeMasterSwitchTests(unittest.TestCase):
    """Requesting attenuation must enable IPLUMEOPT, the model's master switch.

    Regression for 2026-07-28: APLUME and RFPLUME are only parameters of the
    plume energy-scattering model. With IPLUMEOPT at its default 0 the solver
    never runs it, so aplume 0.15/0.30 and rfplume 0/99/198 um all produced
    bit-identical results -- four wasted solves before the preprocessor echo
    (`activation flag (iplumeopt) = 0`) was read.
    """

    def _cfg(self, **laser_over):
        cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        cfg["laser"].update(laser_over)
        return cfg

    def test_attenuation_enables_the_model(self) -> None:
        _, repl, _ = _render(self._cfg(plume_attenuation=0.2, plume_radius_m=1.0e-4))
        block = repl["__WELD_LEGACY_AFTER_REFLECTION__"].replace(" ", "")
        self.assertIn("iplumeopt=1", block)

    def test_no_attenuation_leaves_the_model_off(self) -> None:
        _, repl, _ = _render(self._cfg())
        self.assertNotIn("iplumeopt", repl["__WELD_LEGACY_AFTER_REFLECTION__"])


class RecoilModelAndVoidPressureTests(unittest.TestCase):
    """Overrides for the recoil form (IF_PRSRECOIL) and void vapour pressure (PV0).

    The 2026-07-28 adversarial review ranked the analytic recoil form
    (if_prsrecoil=1: full A*exp(B(1-Tv/T)) overpressure at Tv, no ambient
    netting) as the top candidate for the onset cliff, with if_prsrecoil=2
    (recoil from the actual evaporation mass flow) the natural test. pv0=50650
    is a 0.5*PV1 GUI relic in the same family as the old frozen avprslbm; it
    shifts the net-evaporation zero to ~3008 K.
    """

    def _cfg(self, **sim_over):
        cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        cfg["simulation"].update(sim_over)
        return cfg

    def test_recoil_model_2_is_rendered(self) -> None:
        _, repl, manifest = _render(self._cfg(recoil_model=2))
        self.assertEqual(repl["__IF_PRSRECOIL__"], "2")
        self.assertEqual(manifest["derived"]["recoil_model"], 2)

    def test_recoil_model_default_is_1(self) -> None:
        _, repl, manifest = _render(self._cfg())
        self.assertEqual(repl["__IF_PRSRECOIL__"], "1")
        self.assertIsNone(manifest["derived"].get("recoil_model"))

    def test_recoil_model_rejects_unknown(self) -> None:
        with self.assertRaises(rectangular.ConfigError):
            _render(self._cfg(recoil_model=3))

    def test_void_vapor_pressure_override(self) -> None:
        _, repl, manifest = _render(self._cfg(void_vapor_pressure_pa=0.0))
        self.assertEqual(float(repl["__PV0__"]), 0.0)
        self.assertEqual(manifest["derived"]["void_vapor_pressure_pa"], 0.0)


class AmbientPressureTests(unittest.TestCase):
    """`simulation.ambient_pressure_pa` renders a uniform chamber pressure."""

    def _cfg(self, **sim_over):
        cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        cfg["simulation"].update(sim_over)
        return cfg

    def test_ambient_pressure_reaches_initial_and_boundary_pressure(self) -> None:
        cfg = self._cfg(ambient_pressure_pa=200000.0)
        _, repl, manifest = _render(cfg)
        self.assertIn("pvoid=200000", repl["__FLUID_INITIALIZATION__"])
        self.assertIn("presi=200000", repl["__FLUID_INITIALIZATION__"])
        self.assertEqual(repl["__P_REFERENCE__"], "200000")
        self.assertEqual(repl["__AMBIENT_PRESSURE_PA__"], "200000")
        self.assertIn("pbct(1,6)=200000", repl["__BC_ZMAX_LINES__"])
        self.assertEqual(manifest["derived"]["ambient_pressure_pa"], 200000.0)

    def test_ambient_pressure_rejects_non_positive(self) -> None:
        for bad in (0.0, -1.0):
            with self.subTest(bad=bad), self.assertRaises(rectangular.ConfigError):
                _render(self._cfg(ambient_pressure_pa=bad))


class PlumeCondensationTests(unittest.TestCase):
    """`laser.plume_condensation` overrides ACONDENS (default 1.0).

    ACONDENS is the fraction of the plume-scattered energy that recondenses
    onto the fluid inside the RFPLUME cylinder. The keyhole profile hardcodes
    1.0; the 2026-07-28 bundle shipped it unablated, so a 0.0 arm is needed to
    show whether the returned energy matters at all.
    """

    def _cfg(self, **laser_over):
        cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        cfg["laser"].update(laser_over)
        return cfg

    def test_default_keeps_profile_value(self) -> None:
        _, repl, manifest = _render(self._cfg(plume_attenuation=0.2))
        self.assertIn("acondens=1.0", repl["__WELD_LEGACY_AFTER_REFLECTION__"])
        self.assertIsNone(manifest["derived"].get("plume_condensation"))

    def test_override_is_rendered_and_recorded(self) -> None:
        _, repl, manifest = _render(self._cfg(plume_attenuation=0.2, plume_condensation=0.0))
        block = repl["__WELD_LEGACY_AFTER_REFLECTION__"].replace(" ", "")
        self.assertIn("acondens=0,", block)
        self.assertNotIn("acondens=1.0", block)
        self.assertAlmostEqual(manifest["derived"]["plume_condensation"], 0.0)

    def test_requires_plume_attenuation(self) -> None:
        with self.assertRaises(rectangular.ConfigError):
            _render(self._cfg(plume_condensation=0.0))

    def test_rejects_out_of_range(self) -> None:
        for bad in (-0.1, 1.5):
            with self.subTest(bad=bad), self.assertRaises(rectangular.ConfigError):
                _render(self._cfg(plume_attenuation=0.2, plume_condensation=bad))


class CalibratedRecipeTests(unittest.TestCase):
    """Pin the full 2026-07-29 calibrated Ti64 recipe rendering.

    The regeneration-readiness review found every sweep config in the repo
    still rendered pre-calibration physics; this test pins material file +
    runner + template together so silent drift of any one of them fails fast.
    """

    def test_ti64_v3_recipe_renders_end_to_end(self) -> None:
        cfg = json.loads((ROOT / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        cfg["material"] = {"file": str(ROOT / "materials" / "ti64_v3_calibrated" / "ti64_v3_calibrated.json")}
        cfg["simulation"]["recoil_model"] = 2
        cfg["simulation"]["void_vapor_pressure_pa"] = 101325.0
        _, repl, manifest = _render(cfg)
        weld = repl["__WELD_LEGACY_AFTER_REFLECTION__"]
        self.assertEqual(repl["__IF_PRSRECOIL__"], "2")
        self.assertEqual(repl["__PV0__"], "101325")
        self.assertAlmostEqual(float(repl["__TVEXP__"]), 8.314462618 / (0.047867 * 9.83e6), places=9)
        self.assertAlmostEqual(manifest["material"]["properties"]["vaporization_temperature_k"], 3520.0)
        self.assertNotIn("iplumeopt=", weld)          # no plume in any recipe
        self.assertIn("aplume=0.0", weld)
