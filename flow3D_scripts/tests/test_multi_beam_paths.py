"""Independent beams, each on its own path, with its own turn-off time.

Distinct from `laser.beam_profile`, which splits ONE beam into co-moving beamlets
that share a path and conserve total power. Here each beam is independent: its own
path, its own power, its own turn-off. Paths may cross.

The binding constraint is the FLOW-3D table layout. `powlbm`, `utlbm`, `vtlbm` and
`wtlbm` are indexed `(row, beam)`, but **`timlbm` is indexed by row only** -- one
time array shared by every beam. So independent beams cannot carry independent time
bases; each beam's velocity and power must be sampled onto a shared grid. These
tests pin that: the union grid, per-beam sampling on it, and per-beam turn-off.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

import run_rectangular_case as rectangular  # noqa: E402


def _rows(text: str, name: str) -> dict[int, list[float]]:
    """{beam_index: [values]} for a (row, beam)-indexed table key."""
    out: dict[int, list[float]] = {}
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith(name + "("):
            continue
        idx, value = line.split("=", 1)
        inner = idx[idx.index("(") + 1: idx.index(")")]
        parts = [p.strip() for p in inner.split(",")]
        beam = int(parts[1]) if len(parts) > 1 else 1
        out.setdefault(beam, []).append(float(value))
    return out


def _config(**laser_extra) -> dict:
    raw = json.loads((REPO / "configs" / "rectangular_moving_laser_smoke.json").read_text())
    raw["geometry"] = {"length_m": 0.004, "width_m": 0.003,
                       "thickness_m": 0.0006, "gas_clearance_m": 0.0004}
    raw["mesh"] = {"cell_size_m": 2.0e-5}
    raw["laser"]["beam_radius_m"] = 6.0e-5
    raw["simulation"]["finish_time_s"] = 0.06
    raw["laser"].pop("end_point_m", None)
    raw["laser"].update(laser_extra)
    return raw


def _render(cfg: dict, name: str):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.json"
        p.write_text(json.dumps(cfg))
        return rectangular.load_and_validate(p, name)


TWO_BEAMS = [
    {"power_w": 100.0, "path": {"type": "random", "seed": 1, "length_m": 0.003}},
    {"power_w": 140.0, "path": {"type": "random", "seed": 2, "length_m": 0.002}},
]


class MultiBeamPathTests(unittest.TestCase):
    def test_each_beam_gets_its_own_indexed_velocity_and_power(self) -> None:
        _, repl, man = _render(_config(beams=TWO_BEAMS), "unit_mb_two")
        block = repl["__LASER_BEAM_BLOCK__"]
        self.assertIn("nlbeam=2,", block)
        for key in ("utlbm", "vtlbm", "powlbm"):
            beams = _rows(block, key)
            self.assertEqual(sorted(beams), [1, 2], msg=f"{key} not indexed per beam")
            self.assertEqual(len(beams[1]), len(beams[2]),
                             msg=f"{key} rows differ per beam; timlbm is shared so they must match")
        self.assertEqual(man["derived"]["beam_count"], 2)

    def test_time_grid_is_shared_and_strictly_increasing(self) -> None:
        """timlbm has no beam index, so there is exactly one time column."""
        _, repl, _ = _render(_config(beams=TWO_BEAMS), "unit_mb_time")
        block = repl["__LASER_BEAM_BLOCK__"]
        times = _rows(block, "timlbm")
        self.assertEqual(list(times), [1], "timlbm must not be beam-indexed")
        t = times[1]
        self.assertTrue(all(b > a for a, b in zip(t, t[1:])), "shared time grid must increase")
        # Every per-beam table must have exactly one value per shared time row.
        self.assertEqual(len(_rows(block, "utlbm")[1]), len(t))
        self.assertEqual(len(_rows(block, "utlbm")[2]), len(t))

    def test_per_beam_turn_off_zeroes_only_that_beam(self) -> None:
        beams = [
            {"power_w": 100.0, "turn_off_time_s": 0.004,
             "path": {"type": "random", "seed": 1, "length_m": 0.003}},
            {"power_w": 140.0, "path": {"type": "random", "seed": 2, "length_m": 0.003}},
        ]
        _, repl, _ = _render(_config(beams=beams), "unit_mb_off")
        block = repl["__LASER_BEAM_BLOCK__"]
        times = _rows(block, "timlbm")[1]
        p1, p2 = _rows(block, "powlbm")[1], _rows(block, "powlbm")[2]
        after = [i for i, t in enumerate(times) if t > 0.004 + 1e-9]
        self.assertTrue(after, "test needs rows after the turn-off time")
        self.assertTrue(all(p1[i] == 0.0 for i in after), "beam 1 must be off after its turn-off")
        self.assertTrue(any(p2[i] > 0.0 for i in after), "beam 2 must still be firing")

    def test_beams_may_cross(self) -> None:
        """Crossing is allowed: no keep-apart constraint between independent beams."""
        crossing = [
            # An X: the two tracks intersect near the domain centre.
            {"power_w": 100.0, "path": {"type": "waypoints",
                                        "points_m": [[0.0008, -0.0009], [0.0028, 0.0009]]}},
            {"power_w": 100.0, "path": {"type": "waypoints",
                                        "points_m": [[0.0008, 0.0009], [0.0028, -0.0009]]}},
        ]
        _, repl, man = _render(_config(beams=crossing), "unit_mb_cross")
        self.assertIn("nlbeam=2,", repl["__LASER_BEAM_BLOCK__"])
        self.assertEqual(man["derived"]["beam_count"], 2)

    def test_rejects_beams_together_with_beam_profile(self) -> None:
        """beam_profile splits one beam; beams are independent. Both at once is ambiguous."""
        cfg = _config(beams=TWO_BEAMS,
                      beam_profile={"type": "ring", "angular_samples": 3,
                                    "beamlet_radius_fraction": 0.25,
                                    "ring_radius_fraction": 0.5})
        with self.assertRaises(rectangular.ConfigError):
            _render(cfg, "unit_mb_conflict")

    def test_rejects_more_than_four_beams(self) -> None:
        many = [{"power_w": 50.0, "path": {"type": "random", "seed": i, "length_m": 0.002}}
                for i in range(5)]
        with self.assertRaisesRegex(rectangular.ConfigError, "beam"):
            _render(_config(beams=many), "unit_mb_many")

    def test_single_beam_config_is_unchanged(self) -> None:
        """The existing straight-scan path must render exactly as before."""
        raw = json.loads((REPO / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        _, repl, man = _render(raw, "unit_mb_legacy")
        self.assertIn("nlbeam=1,", repl["__LASER_BEAM_BLOCK__"])
        self.assertNotIn("beam_count", man["derived"])


if __name__ == "__main__":
    unittest.main()


class MultiBeamProvenanceTests(unittest.TestCase):
    """The NPZ provenance scalars must survive a config with no top-level power.

    `export_all_timesteps_npz` is handed `laser["power_w"]`. A multi-beam config has
    no top-level power -- each beam carries its own -- so the naive lookup raises
    KeyError *at export time*, i.e. after the solve has already been paid for. One
    scalar cannot describe 2-4 beams, so it records the total delivered power and the
    mean speed; the per-beam detail lives in case_manifest.json.
    """

    def test_provenance_scalars_resolve_without_a_top_level_power(self) -> None:
        cfg = _config(beams=TWO_BEAMS)
        laser = cfg["laser"]
        # The generated batch configs carry no top-level power/speed at all -- each beam
        # has its own. Reproduce that exactly; the smoke config this helper builds on
        # does define them.
        laser.pop("power_w", None)
        laser.pop("scan_speed_m_per_s", None)
        self.assertNotIn("power_w", laser, "this test is meaningless if the key is present")
        power, speed = rectangular.export_provenance_scalars(laser)
        self.assertAlmostEqual(power, 100.0 + 140.0)
        self.assertGreater(speed, 0.0)

    def test_single_beam_provenance_is_unchanged(self) -> None:
        raw = json.loads((REPO / "configs" / "rectangular_moving_laser_smoke.json").read_text())
        power, speed = rectangular.export_provenance_scalars(raw["laser"])
        self.assertAlmostEqual(power, raw["laser"]["power_w"])
        self.assertAlmostEqual(speed, raw["laser"]["scan_speed_m_per_s"])


class BeamTableRowLimitTests(unittest.TestCase):
    """FLOW-3D's beam tables hold at most 500 rows; row 501 is rejected.

    `mb40_007` rendered 529 rows and FLOW-3D refused the namelist at preprocessing:
    `error while reading name list "weld" ... unknown variable ...
    "powlbm(501,1)=0,"`. Nothing in the generator knew about the bound, so an
    over-long case failed at the solver instead of at config validation.

    Independent beams share one time column, so rows grow with beam count: measured
    107 (2 beams), 232/174 (3), 376 (4), 529 (4, rejected). The generator now thins
    each beam's samples to fit and records that it did.
    """

    def test_row_count_stays_within_the_solver_limit(self) -> None:
        # Four beams on long paths -- comfortably over 500 rows unthinned.
        beams = [{"power_w": 100.0, "scan_speed_m_per_s": 0.2,
                  "path": {"type": "random", "seed": 40 + i, "length_m": 0.0028}}
                 for i in range(4)]
        _, repl, man = _render(_config(beams=beams), "unit_mb_rowcap")
        rows = repl["__LASER_BEAM_BLOCK__"].count("timlbm(")
        self.assertLessEqual(rows, rectangular.MAX_BEAM_TABLE_ROWS,
                             f"{rows} rows exceeds the solver limit")
        self.assertGreater(rows, 50, "thinning must not collapse the trajectory")

    def test_thinning_is_recorded_when_it_happens(self) -> None:
        beams = [{"power_w": 100.0, "scan_speed_m_per_s": 0.2,
                  "path": {"type": "random", "seed": 60 + i, "length_m": 0.0028}}
                 for i in range(4)]
        _, _, man = _render(_config(beams=beams), "unit_mb_thinrec")
        self.assertTrue(man["derived"].get("beam_table_thinned"),
                        "a thinned case must say so in the manifest")

    def test_small_case_is_not_thinned(self) -> None:
        _, repl, man = _render(_config(beams=TWO_BEAMS), "unit_mb_nothin")
        self.assertLessEqual(repl["__LASER_BEAM_BLOCK__"].count("timlbm("),
                             rectangular.MAX_BEAM_TABLE_ROWS)
        self.assertFalse(man["derived"].get("beam_table_thinned"))
