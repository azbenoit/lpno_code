#!/usr/bin/env python3
"""Full solver-backed tests.

These tests assume a local FLOW-3D license endpoint has been configured in
F3D_LICENSE. If it is missing, they fail with setup instructions rather than
silently skipping, because full validation is intended to exercise the actual
user workflow.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "bash" / "run_rectangular_case.sh"
MELT_CONFIG = ROOT / "configs" / "rectangular_moving_laser_melt.json"
NPZ_VALIDATOR = ROOT / "python" / "validate_npz_series.py"

sys.path.insert(0, str(ROOT / "python"))
import run_rectangular_case as rectangular  # noqa: E402


def require_f3d_license() -> str:
    endpoint = rectangular.local_license_endpoint()
    if not endpoint:
        raise AssertionError(
            "No FLOW-3D license endpoint was found. Configure it in your local "
            "shell or in ignored license.env before running full solver-backed tests. Example:\n"
            "  export F3D_LICENSE='<your-license-endpoint>'\n"
            "or create license.env with:\n"
            "  F3D_LICENSE='<your-license-endpoint>'\n"
            "The endpoint format is PORT@HOST. Do not commit the real value."
        )
    return endpoint


class FullSolverWorkflowTests(unittest.TestCase):
    def test_two_core_moving_laser_melt_case_exports_and_melts(self) -> None:
        require_f3d_license()
        case_name = "full_solver_two_core_melt_test"
        case_dir = ROOT / "runs" / case_name
        shutil.rmtree(case_dir, ignore_errors=True)
        try:
            result = subprocess.run(
                [
                    str(RUN_SCRIPT),
                    str(MELT_CONFIG),
                    "--case-name",
                    case_name,
                    "--cores",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((case_dir / f"flsgrf1.{case_name}").is_file())
            self.assertTrue((case_dir / "runhyd.log").is_file())
            self.assertTrue((case_dir / "runpre_decompose.log").is_file())
            self.assertIn(
                "normal completion: finish time exceeded",
                (case_dir / "runhyd.log").read_text(encoding="utf-8", errors="ignore"),
            )
            self.assertTrue((case_dir / "export_npz.log").is_file())
            npz = case_dir / "npz" / f"{case_name}.npz"
            self.assertTrue(npz.is_file())
            validation = subprocess.run(
                [
                    "python3",
                    str(NPZ_VALIDATOR),
                    str(npz),
                    "--initial-temperature-k",
                    "293.15",
                    "--min-temperature-rise-k",
                    "100",
                    "--require-melt",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
