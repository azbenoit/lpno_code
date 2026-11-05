#!/usr/bin/env python3
"""Short, no-license tests for the batch builder and its task-file layout."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "python" / "run_rectangular_case.py"
BUILD_BATCH = ROOT / "python" / "build_batch.py"
REPRO_CONFIG = ROOT / "configs" / "case_fine_1_10um_reproduction.json"
SMOKE_CONFIG = ROOT / "configs" / "rectangular_moving_laser_smoke.json"

sys.path.insert(0, str(ROOT / "python"))
import batch  # noqa: E402


class BatchUnitTests(unittest.TestCase):
    def test_linspace_is_inclusive_and_evenly_spaced(self) -> None:
        self.assertEqual(batch.linspace(0.0, 1.0, 5), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(batch.linspace(40.0, 40.0, 1), [40.0])

    def test_set_dotted_creates_nested_keys(self) -> None:
        target: dict = {}
        batch.set_dotted(target, "laser.power_w", 90.0)
        self.assertEqual(target, {"laser": {"power_w": 90.0}})

    def test_zip_overrides_pairs_lists_and_rejects_length_mismatch(self) -> None:
        paired = batch.zip_overrides(power_w=[70, 90], scan_speed_m_per_s=[0.7, 0.3])
        self.assertEqual(paired[1], {"power_w": 90, "scan_speed_m_per_s": 0.3})
        with self.assertRaises(ValueError):
            batch.zip_overrides(a=[1, 2], b=[1])

    def test_product_overrides_is_cross_product(self) -> None:
        grid = batch.product_overrides(a=[1, 2], b=[10, 20])
        self.assertEqual(len(grid), 4)
        self.assertIn({"a": 2, "b": 10}, grid)

    def test_build_configs_deep_copies_and_names_uniquely(self) -> None:
        reference = json.loads(REPRO_CONFIG.read_text())
        overrides = batch.zip_overrides(
            **{"laser.power_w": [70.0, 90.0], "laser.scan_speed_m_per_s": [0.711, 0.323]}
        )
        configs = batch.build_configs(reference, overrides, batch.pv_name)
        self.assertEqual([c["case_name"] for c in configs], ["70W_0711mps", "90W_0323mps"])
        self.assertEqual(configs[0]["laser"]["power_w"], 70.0)
        self.assertEqual(configs[1]["laser"]["power_w"], 90.0)
        # Deep copy: editing one config must not touch the reference or siblings.
        self.assertEqual(reference["laser"]["power_w"], 70.0)
        self.assertIsNot(configs[0]["laser"], reference["laser"])

    def test_build_configs_rejects_duplicate_names(self) -> None:
        reference = json.loads(REPRO_CONFIG.read_text())
        dup = batch.zip_overrides(
            **{"laser.power_w": [70.0, 70.0], "laser.scan_speed_m_per_s": [0.711, 0.711]}
        )
        with self.assertRaises(ValueError):
            batch.build_configs(reference, dup, batch.pv_name)

    def test_write_jobs_emits_configs_and_one_command_per_case(self) -> None:
        reference = json.loads(SMOKE_CONFIG.read_text())
        overrides = batch.product_overrides(**{"laser.power_w": [200.0, 300.0]})
        configs = batch.build_configs(reference, overrides, batch.pv_name)
        with tempfile.TemporaryDirectory() as tmp:
            task_file = batch.write_jobs(configs, Path(tmp), cores=1, threads=1, export="npz")
            lines = task_file.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue((Path(tmp) / "configs" / "200W_0100mps.json").is_file())
            self.assertIn("--cores 1 --threads-per-rank 1 --export npz", lines[0])
            self.assertIn("run_rectangular_case.py", lines[0])
            # Run output is routed under the batch dir, not the shared runs/.
            self.assertIn(f"--out-root {Path(tmp).resolve()}", lines[0])

    def test_out_root_places_run_under_given_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_name = "unit_out_root"
            result = subprocess.run(
                ["python3", str(RUNNER), str(SMOKE_CONFIG), "--case-name", case_name,
                 "--out-root", tmp, "--prepare-only"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((Path(tmp) / case_name / f"prepin.{case_name}").is_file())


class BuildBatchCliTests(unittest.TestCase):
    def _build(self, *args: str) -> Path:
        out_dir = Path(self._tmp) / "batch"
        result = subprocess.run(
            ["python3", str(BUILD_BATCH), str(REPRO_CONFIG), str(out_dir), *args],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return out_dir

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_grid_linspace_shortcut_expands_cases(self) -> None:
        out_dir = self._build("--grid", "laser.power_w=40:200:5")
        configs = sorted((out_dir / "configs").glob("*.json"))
        self.assertEqual(len(configs), 5)

    def test_tsv_with_derived_finish_time_matches_track_over_speed(self) -> None:
        tsv = Path(self._tmp) / "params.tsv"
        tsv.write_text(
            "laser.power_w\tlaser.scan_speed_m_per_s\n70\t0.711\n90\t0.323\n", encoding="utf-8"
        )
        out_dir = self._build(
            "--params-tsv", str(tsv), "--derive-finish-time-from-track", "--name-prefix", "train_"
        )
        cfg = json.loads((out_dir / "configs" / "train_70W_0711mps.json").read_text())
        # track length is 0.00029724 - (-0.0003) = 0.00059724 m; /0.711 = 0.00084 s
        self.assertAlmostEqual(cfg["simulation"]["finish_time_s"], 0.00059724 / 0.711, places=9)

    def test_derive_finish_time_rejects_turn_off_at_end(self) -> None:
        # The derived finish time equals the scan time, too short for the
        # turn-off event, so the flag must reject a turn_off_at_end=true config.
        reference = json.loads(REPRO_CONFIG.read_text())
        reference["laser"]["turn_off_at_end"] = True
        ref_path = Path(self._tmp) / "turn_off_ref.json"
        ref_path.write_text(json.dumps(reference), encoding="utf-8")
        out_dir = Path(self._tmp) / "turn_off_out"
        result = subprocess.run(
            ["python3", str(BUILD_BATCH), str(ref_path), str(out_dir),
             "--grid", "laser.power_w=70:90:2", "--derive-finish-time-from-track"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("turn_off_at_end=false", result.stderr)

    def test_generated_config_passes_prepare_only(self) -> None:
        out_dir = self._build("--grid", "laser.power_w=120:160:2")
        config = sorted((out_dir / "configs").glob("*.json"))[0]
        case_name = "batch_prepare_check"
        case_dir = ROOT / "runs" / case_name
        try:
            result = subprocess.run(
                ["python3", str(RUNNER), str(config), "--case-name", case_name, "--prepare-only"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((case_dir / f"prepin.{case_name}").is_file())
        finally:
            import shutil
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_exactly_one_mode_required(self) -> None:
        out_dir = Path(self._tmp) / "batch"
        result = subprocess.run(
            ["python3", str(BUILD_BATCH), str(REPRO_CONFIG), str(out_dir)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one", result.stderr)


if __name__ == "__main__":
    unittest.main()
