#!/usr/bin/env python3
"""Unit tests for the constant-speed laser path builder (no license needed)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "rectangular_moving_laser_smoke.json"
sys.path.insert(0, str(ROOT / "python"))
import laser_path  # noqa: E402
import run_rectangular_case as rectangular  # noqa: E402


def path_config(path_spec: dict, **overrides) -> dict:
    """Smoke config with a wide domain and a laser.path (start/end removed)."""
    cfg = json.loads(SMOKE_CONFIG.read_text())
    cfg["geometry"] = {"length_m": 0.004, "width_m": 0.004, "thickness_m": 0.0004, "gas_clearance_m": 0.0004}
    cfg["geometry"].update(overrides)
    cfg["laser"].pop("start_point_m", None)
    cfg["laser"].pop("end_point_m", None)
    cfg["laser"]["path"] = path_spec
    cfg["simulation"]["finish_time_s"] = 1.0  # generous; real check is per-test
    return cfg


def load_cfg(cfg: dict, case_name: str):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.json"
        p.write_text(json.dumps(cfg))
        return rectangular.load_and_validate(p, case_name)

RADIUS = 0.00015
# A comfortably large domain so the geometry, not the margin, is under test.
BOUNDS = (-0.002, 0.002, -0.002, 0.002)


def _rows(table: str, name: str) -> list[float]:
    return [
        float(line.split("=")[1].rstrip(","))
        for line in table.splitlines()
        if line.strip().startswith(name)
    ]


def table_leg_speeds(samples: list, speed: float, turn_off: bool = True) -> tuple[dict, list[float]]:
    """Speeds actually written to the beam table (drops the turn-off row)."""
    tables = laser_path.velocity_time_tables(300.0, samples, speed, turn_off)
    us = _rows(tables["__LASER_VELOCITY_TABLE__"], "utlbm")
    vs = _rows(tables["__LASER_VELOCITY_TABLE__"], "vtlbm")
    speeds = [math.hypot(u, v) for u, v in zip(us, vs)]
    return tables, (speeds[:-1] if turn_off else speeds)


class ResampleTests(unittest.TestCase):
    def test_equal_spacing_and_endpoints(self) -> None:
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]  # L-shape, length 2
        out = laser_path.resample_by_arclength(pts, 5)
        self.assertEqual(out[0], (0.0, 0.0))
        self.assertEqual(out[-1], (1.0, 1.0))
        gaps = [math.dist(a, b) for a, b in zip(out, out[1:])]
        for g in gaps:
            self.assertAlmostEqual(g, 0.5, places=9)

    def test_catmull_rom_passes_through_controls(self) -> None:
        control = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
        dense = laser_path.catmull_rom(control, per_segment=20)
        for c in control:
            self.assertTrue(min(math.dist(c, d) for d in dense) < 1e-9)


class BuildPathTests(unittest.TestCase):
    def test_waypoints_linear_length_and_constant_speed(self) -> None:
        spec = {"type": "waypoints", "points_m": [[-0.001, 0.0], [0.0, 0.0005], [0.001, 0.0]]}
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        expected = math.dist((-0.001, 0.0), (0.0, 0.0005)) + math.dist((0.0, 0.0005), (0.001, 0.0))
        self.assertAlmostEqual(path.length_m, expected, places=6)
        _, speeds = table_leg_speeds(path.samples, 0.1)
        for s in speeds:
            self.assertAlmostEqual(s, 0.1, places=6)

    def test_sharp_corner_keeps_exact_speed(self) -> None:
        # Right-angle corner: a leg straddling the vertex has chord < arc, so a
        # uniform-dt table would dip ~28% here. Per-leg dt keeps every leg at speed.
        spec = {"type": "waypoints", "points_m": [[0.0, 0.0], [0.001, 0.0], [0.001, 0.001]]}
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        _, speeds = table_leg_speeds(path.samples, 0.1)
        self.assertAlmostEqual(min(speeds), 0.1, places=6)
        self.assertAlmostEqual(max(speeds), 0.1, places=6)

    def test_spline_curves_between_waypoints(self) -> None:
        spec = {
            "type": "waypoints",
            "interpolation": "spline",
            "points_m": [[-0.001, 0.0], [0.0, 0.0008], [0.001, 0.0]],
        }
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        # A spline bows out past the straight chord between the end points.
        straight = math.dist((-0.001, 0.0), (0.001, 0.0))
        self.assertGreater(path.length_m, straight)
        self.assertEqual(path.interpolation, "spline")

    def test_sinusoidal_reaches_amplitude(self) -> None:
        spec = {
            "type": "sinusoidal",
            "start_point_m": [-0.001, 0.0, 0.0],
            "length_m": 0.002,
            "amplitude_m": 0.0006,
            "periods": 2,
        }
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        ys = [y for _, y in path.samples]
        self.assertGreater(max(ys), 0.00055)
        self.assertLess(min(ys), -0.00055)
        # Wiggly path is longer than its straight projection.
        self.assertGreater(path.length_m, 0.002)

    def test_sinusoidal_wavelength_matches_periods(self) -> None:
        base = {"type": "sinusoidal", "start_point_m": [-0.001, 0.0, 0.0], "length_m": 0.002, "amplitude_m": 0.0005}
        by_periods = laser_path.build_path(
            {**base, "periods": 4}, radius=RADIUS, bounds=BOUNDS
        )
        by_wavelength = laser_path.build_path(
            {**base, "wavelength_m": 0.0005}, radius=RADIUS, bounds=BOUNDS
        )  # 0.002 / 4 == 0.0005
        self.assertAlmostEqual(by_periods.length_m, by_wavelength.length_m, places=6)

    def test_sample_spacing_controls_row_count(self) -> None:
        spec = {"type": "waypoints", "points_m": [[-0.001, 0.0], [0.001, 0.0]]}
        fine = laser_path.build_path(
            {**spec, "sample_spacing_m": 0.0001}, radius=RADIUS, bounds=BOUNDS
        )
        coarse = laser_path.build_path(
            {**spec, "sample_spacing_m": 0.0005}, radius=RADIUS, bounds=BOUNDS
        )
        self.assertGreater(len(fine.samples), len(coarse.samples))

    def test_too_many_rows_is_rejected(self) -> None:
        spec = {"type": "waypoints", "points_m": [[-0.0018, 0.0], [0.0018, 0.0]], "sample_spacing_m": 1e-6}
        with self.assertRaises(laser_path.PathError):
            laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)

    def test_random_respects_step_and_max_turn(self) -> None:
        spec = {"type": "random", "seed": 2, "length_m": 0.008, "step_m": 0.0005, "max_turn_deg": 30}
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        margin = 4.0 * RADIUS
        for x, y in path.samples:
            self.assertTrue(BOUNDS[0] + margin - 1e-9 <= x <= BOUNDS[1] - margin + 1e-9)
            self.assertTrue(BOUNDS[2] + margin - 1e-9 <= y <= BOUNDS[3] - margin + 1e-9)

    def test_random_control_respects_max_turn_away_from_walls(self) -> None:
        # Domain far larger than the target length: displacement < length < domain,
        # so the walk never reaches a wall and the max_turn cap holds on every step.
        allowed = (-0.0025, 0.0025, -0.0025, 0.0025)
        spec = {"type": "random", "seed": 3, "length_m": 0.002, "step_m": 0.0003, "max_turn_deg": 30}
        controls = laser_path._random_control(spec, allowed, (0.0, 0.0), RADIUS)
        cap = math.radians(30) + 1e-9
        for a, b, c in zip(controls, controls[1:], controls[2:]):
            h1 = math.atan2(b[1] - a[1], b[0] - a[0])
            h2 = math.atan2(c[1] - b[1], c[0] - b[0])
            self.assertLessEqual(abs(laser_path._wrap_pi(h2 - h1)), cap)

    def test_random_reaches_target_length_in_small_domain(self) -> None:
        # Target far longer than the domain: wall-peel keeps the walk going and it
        # self-intersects to reach the requested length. length_m is controlled to
        # a fraction of a percent at fine spacing (a few percent only at coarse
        # spacing with heavy wall contact, where the polyline under-cuts curves).
        bounds = (-0.0012, 0.0012, -0.0012, 0.0012)
        for ang in (15, 90):
            spec = {"type": "random", "seed": 1, "length_m": 0.008, "step_m": 0.0001,
                    "max_turn_deg": ang, "sample_spacing_m": 0.00003}
            path = laser_path.build_path(spec, radius=RADIUS, bounds=bounds)
            self.assertAlmostEqual(path.length_m, 0.008, delta=0.008 * 0.01)  # within 1%

    def test_random_length_defaults_to_domain_length(self) -> None:
        spec = {"type": "random", "seed": 1}  # no length_m -> domain length
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        domain_length = BOUNDS[1] - BOUNDS[0]
        self.assertAlmostEqual(path.length_m, domain_length, delta=domain_length * 0.03)

    def test_random_default_start_is_left_middle_of_cropped_region(self) -> None:
        spec = {"type": "random", "seed": 1, "length_m": 0.004}
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        margin = 4.0 * RADIUS
        self.assertAlmostEqual(path.samples[0][0], BOUNDS[0] + margin, places=9)  # left edge
        self.assertAlmostEqual(path.samples[0][1], 0.5 * (BOUNDS[2] + BOUNDS[3]), places=9)  # mid height

    def test_non_numeric_random_fields_raise_patherror(self) -> None:
        for bad in (
            {"type": "random", "seed": 1, "length_m": 0.005, "max_turn_deg": "lots"},
            {"type": "random", "seed": 1, "length_m": 0.005, "step_m": [1, 2]},
        ):
            with self.assertRaises(laser_path.PathError):
                laser_path.build_path(bad, radius=RADIUS, bounds=BOUNDS)

    def test_random_is_deterministic_and_seed_sensitive(self) -> None:
        spec_a = {"type": "random", "seed": 7, "length_m": 0.006}
        spec_b = {"type": "random", "seed": 7, "length_m": 0.006}
        spec_c = {"type": "random", "seed": 8, "length_m": 0.006}
        a = laser_path.build_path(spec_a, radius=RADIUS, bounds=BOUNDS)
        b = laser_path.build_path(spec_b, radius=RADIUS, bounds=BOUNDS)
        c = laser_path.build_path(spec_c, radius=RADIUS, bounds=BOUNDS)
        self.assertEqual(a.samples, b.samples)
        self.assertNotEqual(a.samples, c.samples)

    def test_random_stays_inside_four_radii_and_explores(self) -> None:
        bounds = (-0.003, 0.003, -0.003, 0.003)
        spec = {"type": "random", "seed": 3, "length_m": 0.03}
        path = laser_path.build_path(spec, radius=RADIUS, bounds=bounds)
        margin = 4.0 * RADIUS
        xs = [x for x, _ in path.samples]
        ys = [y for _, y in path.samples]
        for x, y in path.samples:
            self.assertGreaterEqual(x, bounds[0] + margin - 1e-9)
            self.assertLessEqual(x, bounds[1] - margin + 1e-9)
            self.assertGreaterEqual(y, bounds[2] + margin - 1e-9)
            self.assertLessEqual(y, bounds[3] - margin + 1e-9)
        # Coverage bias: a long walk should spread across both axes, not a line.
        reachable = (bounds[1] - margin) - (bounds[0] + margin)
        self.assertGreater(max(xs) - min(xs), 0.3 * reachable)
        self.assertGreater(max(ys) - min(ys), 0.3 * reachable)


class MarginCheckTests(unittest.TestCase):
    def test_path_too_close_to_edge_is_rejected(self) -> None:
        # End point sits 1 radius from the +x edge, well inside the 4-radius line.
        near = BOUNDS[1] - RADIUS
        spec = {"type": "waypoints", "points_m": [[0.0, 0.0], [near, 0.0]]}
        with self.assertRaises(laser_path.PathError):
            laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)

    def test_path_just_inside_margin_passes(self) -> None:
        ok = BOUNDS[1] - 4.0 * RADIUS
        spec = {"type": "waypoints", "points_m": [[0.0, 0.0], [ok, 0.0]]}
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        self.assertAlmostEqual(path.end[0], ok, places=9)


class VelocityTableTests(unittest.TestCase):
    def test_plateau_rows_times_and_turn_off(self) -> None:
        samples = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]  # 2 legs
        tables = laser_path.velocity_time_tables(300.0, samples, 0.1, turn_off=True)
        times = _rows(tables["__LASER_TIME_TABLE__"], "timlbm")
        us = _rows(tables["__LASER_VELOCITY_TABLE__"], "utlbm")
        vs = _rows(tables["__LASER_VELOCITY_TABLE__"], "vtlbm")
        powers = _rows(tables["__LASER_POWER_TABLE__"], "powlbm")
        # Two plateaus (2 rows each) + one turn-off row.
        self.assertEqual(len(times), 5)
        self.assertEqual(len(us), 5)
        self.assertEqual(len(powers), 5)
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(set(times)), len(times))  # strictly increasing
        # Every active leg runs at exactly the scan speed; final row zeroed.
        for u, v in list(zip(us, vs))[:-1]:
            self.assertAlmostEqual(math.hypot(u, v), 0.1, places=9)
        self.assertEqual((us[-1], vs[-1], powers[-1]), (0.0, 0.0, 0.0))
        self.assertIn("wtlbm(1,1)=0.0,", tables["__LASER_VELOCITY_TABLE__"])

    def test_no_turn_off_holds_last_velocity(self) -> None:
        samples = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]
        tables = laser_path.velocity_time_tables(300.0, samples, 0.1, turn_off=False)
        powers = _rows(tables["__LASER_POWER_TABLE__"], "powlbm")
        us = _rows(tables["__LASER_VELOCITY_TABLE__"], "utlbm")
        vs = _rows(tables["__LASER_VELOCITY_TABLE__"], "vtlbm")
        self.assertEqual(len(powers), 4)  # 2 legs * 2 rows, no turn-off row
        self.assertTrue(all(p == 300.0 for p in powers))
        self.assertAlmostEqual(math.hypot(us[-1], vs[-1]), 0.1, places=9)


class VariableSpeedTests(unittest.TestCase):
    def test_build_keeps_knots_and_times(self) -> None:
        knots = [[-0.001, 0.0], [0.0, 0.0005], [0.001, 0.0]]
        spec = {"type": "variable_speed", "knots_m": knots, "times_s": [0.0, 0.001, 0.0015]}
        path = laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)
        self.assertEqual([list(p) for p in path.samples], knots)  # exact, no arc resample
        self.assertEqual(path.times_s, [0.0, 0.001, 0.0015])
        self.assertEqual(path.type, "variable_speed")

    def test_per_leg_speed_matches_chord_over_dt(self) -> None:
        # leg 0: length 0.001 over 0.001 s -> 1 m/s; leg 1: length 0.001 over 0.0005 s -> 2 m/s
        knots = [[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]]
        times = [0.0, 0.001, 0.0015]
        tables = laser_path.variable_speed_time_tables(120.0, knots, times, turn_off=True)
        us = _rows(tables["__LASER_VELOCITY_TABLE__"], "utlbm")
        vs = _rows(tables["__LASER_VELOCITY_TABLE__"], "vtlbm")
        speeds = [math.hypot(u, v) for u, v in zip(us, vs)]
        self.assertAlmostEqual(speeds[0], 1.0, places=6)
        self.assertAlmostEqual(speeds[2], 2.0, places=6)  # first row of leg 1 (row idx 2)
        self.assertEqual((us[-1], vs[-1]), (0.0, 0.0))  # turn-off row

    def test_dwell_leg_has_zero_velocity_and_advances_time(self) -> None:
        # A repeated point with advancing time is a true dwell: u=v=0 across it.
        knots = [[0.0, 0.0], [0.001, 0.0], [0.001, 0.0], [0.002, 0.0]]
        times = [0.0, 0.001, 0.0013, 0.0023]  # leg 1 (0.001->0.001) is a 0.3 ms dwell
        tables = laser_path.variable_speed_time_tables(80.0, knots, times, turn_off=False)
        times_out = _rows(tables["__LASER_TIME_TABLE__"], "timlbm")
        us = _rows(tables["__LASER_VELOCITY_TABLE__"], "utlbm")
        vs = _rows(tables["__LASER_VELOCITY_TABLE__"], "vtlbm")
        # 3 legs * 2 plateau rows = 6 rows (no turn-off).
        self.assertEqual(len(times_out), 6)
        self.assertEqual(times_out, sorted(times_out))
        self.assertEqual(len(set(times_out)), 6)  # strictly increasing
        # Rows 2,3 are the dwell leg -> zero velocity, but time still advances.
        self.assertEqual((us[2], vs[2]), (0.0, 0.0))
        self.assertEqual((us[3], vs[3]), (0.0, 0.0))
        self.assertGreater(times_out[3], times_out[2])

    def test_uniform_power_by_default(self) -> None:
        knots = [[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]]
        tables = laser_path.variable_speed_time_tables(95.0, knots, [0.0, 0.001, 0.002], turn_off=True)
        powers = _rows(tables["__LASER_POWER_TABLE__"], "powlbm")
        self.assertTrue(all(p == 95.0 for p in powers[:-1]))
        self.assertEqual(powers[-1], 0.0)  # turn-off row

    def test_non_increasing_times_rejected(self) -> None:
        spec = {"type": "variable_speed", "knots_m": [[0.0, 0.0], [0.001, 0.0]], "times_s": [0.0, 0.0]}
        with self.assertRaises(laser_path.PathError):
            laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)

    def test_length_times_mismatch_rejected(self) -> None:
        spec = {"type": "variable_speed", "knots_m": [[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]],
                "times_s": [0.0, 0.001]}
        with self.assertRaises(laser_path.PathError):
            laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)

    def test_keepout_enforced(self) -> None:
        # y=0.0016 is inside the 4r margin of the +/-0.002 domain.
        spec = {"type": "variable_speed", "knots_m": [[-0.001, 0.0], [0.0, 0.0016]],
                "times_s": [0.0, 0.001]}
        with self.assertRaises(laser_path.PathError):
            laser_path.build_path(spec, radius=RADIUS, bounds=BOUNDS)


class RunnerIntegrationTests(unittest.TestCase):
    def test_waypoints_spline_renders_and_records_manifest(self) -> None:
        spec = {
            "type": "waypoints",
            "interpolation": "spline",
            "points_m": [[-0.001, 0.0], [0.0, 0.0006], [0.001, 0.0]],
        }
        case_name, replacements, manifest = load_cfg(path_config(spec), "unit_path_wp")
        d = manifest["derived"]
        self.assertEqual(d["laser_path_type"], "waypoints")
        self.assertEqual(d["laser_path_interpolation"], "spline")
        self.assertEqual(d["geometry_placement"], "centered")
        n = d["laser_path_num_samples"]
        self.assertGreater(n, 2)
        # Two plateau rows per leg + one turn-off row; times strictly increasing.
        # The beam tables are assembled into one __LASER_BEAM_BLOCK__ by the
        # multi-beam profile support (3be3c0c); they are no longer separate
        # replacement keys. Same rows, one container.
        times = _rows(replacements["__LASER_BEAM_BLOCK__"], "timlbm")
        rows = 2 * (n - 1) + 1
        self.assertEqual(len(times), rows)
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(set(times)), rows)
        # Renders to a complete prepin with no leftover tokens.
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            input_file = rectangular.render(case_dir, case_name, replacements, manifest)
            text = input_file.read_text()
        self.assertNotRegex(text, r"__[A-Z0-9_]+__")
        self.assertIn(f"utlbm({rows},1)=", text)

    def test_render_writes_path_png(self) -> None:
        spec = {"type": "sinusoidal", "start_point_m": [-0.001, 0.0, 0.0], "length_m": 0.002,
                "amplitude_m": 0.0006, "periods": 2}
        case_name, replacements, manifest = load_cfg(path_config(spec), "unit_path_png")
        self.assertGreater(len(manifest["derived"]["laser_path_xy_m"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            rectangular.render(case_dir, case_name, replacements, manifest)
            png = case_dir / "path.png"
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 1000)  # a real PNG, not empty
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")  # PNG magic

    def test_straight_scan_records_path_xy(self) -> None:
        # A straight scan stores start->end so it also gets a path.png.
        _, _, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_straight_xy")
        xy = manifest["derived"]["laser_path_xy_m"]
        self.assertEqual(len(xy), 2)
        self.assertEqual(manifest["derived"]["laser_path_type"], "linear")

    def test_sinusoidal_and_random_render(self) -> None:
        sinus = path_config(
            {"type": "sinusoidal", "start_point_m": [-0.001, 0.0, 0.0], "length_m": 0.002,
             "amplitude_m": 0.0006, "periods": 2}
        )
        _, _, man_s = load_cfg(sinus, "unit_path_sin")
        self.assertEqual(man_s["derived"]["laser_path_type"], "sinusoidal")

        rnd = path_config({"type": "random", "seed": 5, "length_m": 0.006})
        _, repl_a, man_r = load_cfg(rnd, "unit_path_rnd")
        self.assertEqual(man_r["derived"]["laser_path_type"], "random")
        # Deterministic: same seed -> identical velocity table.
        _, repl_b, _ = load_cfg(path_config({"type": "random", "seed": 5, "length_m": 0.006}), "unit_path_rnd2")
        self.assertEqual(repl_a["__LASER_BEAM_BLOCK__"], repl_b["__LASER_BEAM_BLOCK__"])

    def test_path_within_four_radii_is_rejected(self) -> None:
        # width 0.004 -> y bounds +/-0.002; 4r = 0.0006, so y=0.0016 is inside the margin.
        spec = {"type": "waypoints", "points_m": [[-0.001, 0.0], [0.0, 0.0016], [0.001, 0.0]]}
        with self.assertRaises(rectangular.ConfigError):
            load_cfg(path_config(spec), "unit_path_edge")

    def test_committed_sinusoidal_example_validates(self) -> None:
        cfg = ROOT / "configs" / "rectangular_curved_path_sinusoidal.json"
        _, _, manifest = rectangular.load_and_validate(cfg, "unit_example_sinus")
        self.assertEqual(manifest["derived"]["laser_path_type"], "sinusoidal")

    def test_variable_speed_renders_and_records_manifest(self) -> None:
        spec = {
            "type": "variable_speed",
            "knots_m": [[-0.001, 0.0], [0.0, 0.0], [0.0, 0.0], [0.001, 0.0]],
            "times_s": [0.0, 0.001, 0.0014, 0.0019],  # cruise, 0.4 ms dwell, cruise
        }
        cfg = path_config(spec)
        cfg["simulation"]["finish_time_s"] = 0.002
        case_name, replacements, manifest = load_cfg(cfg, "unit_path_vs")
        d = manifest["derived"]
        self.assertEqual(d["laser_path_type"], "variable_speed")
        self.assertTrue(d["laser_path_variable_speed"])
        self.assertAlmostEqual(d["scan_time_s"], 0.0019, places=9)
        # 3 legs -> 6 plateau rows + 1 turn-off row; times strictly increasing.
        # The beam tables are assembled into one __LASER_BEAM_BLOCK__ by the
        # multi-beam profile support (3be3c0c); they are no longer separate
        # replacement keys. Same rows, one container.
        times = _rows(replacements["__LASER_BEAM_BLOCK__"], "timlbm")
        self.assertEqual(len(times), 7)
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(set(times)), 7)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            input_file = rectangular.render(case_dir, case_name, replacements, manifest)
            text = input_file.read_text()
        self.assertNotRegex(text, r"__[A-Z0-9_]+__")

    def test_leading_margin_with_path_is_rejected(self) -> None:
        cfg = path_config({"type": "waypoints", "points_m": [[-0.001, 0.0], [0.001, 0.0]]},
                          placement="laser_leading_margin")
        with self.assertRaises(rectangular.ConfigError):
            load_cfg(cfg, "unit_path_lm")


if __name__ == "__main__":
    unittest.main()
