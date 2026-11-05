"""Regression: `simulation.initial_temperature_k` must reach the metal as `tempi`.

The 2026-08-10 preheat bug: three parallel `&fl` builders, and only two emitted
`tempi` (initial FLUID temperature). `legacy_halfspace` — the repo's default
initialization — wrote the configured temperature as `tvoid` (gas) and `tbct`
(boundary) but never as `tempi`, so a 230-case preheat sweep started every metal
block at FLOW-3D's 273.1 K default. The value appeared 11 times in the prepin, so
grepping for it succeeded; only the KEY was missing. These tests therefore assert
key=value, never mere presence, and pin ALL THREE branches so the same omission
cannot reappear in any of them.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_rectangular_case as rectangular  # noqa: E402

SMOKE_CONFIG = ROOT / "configs" / "rectangular_moving_laser_smoke.json"
PREHEAT_K = 623.15          # far from every default, so a hardcode cannot pass


def _render_with(mode: str, temperature_k: float) -> str:
    cfg = json.loads(SMOKE_CONFIG.read_text())
    cfg.setdefault("geometry", {})["initialization"] = mode
    cfg.setdefault("simulation", {})["initial_temperature_k"] = temperature_k
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cfg.json"
        p.write_text(json.dumps(cfg))
        case, replacements, manifest = rectangular.load_and_validate(p, f"t_{mode}")
        case_dir = Path(tmp) / "case"
        input_file = rectangular.render(case_dir, case, replacements, manifest)
        return input_file.read_text(encoding="utf-8")


def _fl_block(text: str) -> str:
    m = re.search(r"&fl\b(.*?)^\s*/\s*$", text, re.S | re.M)
    assert m, "no &fl namelist in rendered prepin"
    return m.group(1)


class InitialTemperatureReachesMetal(unittest.TestCase):
    def _assert_tempi(self, mode: str) -> None:
        fl = _fl_block(_render_with(mode, PREHEAT_K))
        m = re.search(r"tempi\s*=\s*([0-9.eE+-]+)", fl)
        self.assertIsNotNone(
            m, f"{mode}: &fl has no tempi= key; the metal would start at "
               f"FLOW-3D's default regardless of initial_temperature_k")
        self.assertAlmostEqual(
            float(m.group(1)), PREHEAT_K, places=1,
            msg=f"{mode}: tempi carries {m.group(1)}, not the configured value")

    def test_legacy_halfspace_emits_tempi(self) -> None:
        # The branch that shipped without it — the 230-case preheat sweep.
        self._assert_tempi("legacy_halfspace")

    def test_metal_region_emits_tempi(self) -> None:
        self._assert_tempi("substrate_with_gas")

    def test_filled_mesh_emits_tempi(self) -> None:
        self._assert_tempi("filled_mesh")

    def test_grep_for_the_value_is_not_the_test(self) -> None:
        """The failure mode: the value present many times under OTHER keys.

        Guards the guard: if someone simplifies the assertions above to a
        substring check, this documents why that is insufficient. tvoid and the
        tbct series legitimately carry the same number.
        """
        text = _render_with("legacy_halfspace", PREHEAT_K)
        occurrences = text.count("623.15")
        self.assertGreater(
            occurrences, 1,
            "the configured value should appear under several keys; if this "
            "ever fails the render changed shape and the semantic assertions "
            "above need re-checking")


if __name__ == "__main__":
    unittest.main()
