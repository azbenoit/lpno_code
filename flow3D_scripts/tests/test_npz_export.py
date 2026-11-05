#!/usr/bin/env python3
"""Short, no-license tests for the structured-array NPZ export path.

These exercise the NPZ validator and the reference comparator on small
in-memory fixtures, and check the exporter and run-script wiring statically
(the pvpython exporter itself needs FLOW-3D POST and a solver result, so it is
covered by the full solver-backed workflow test).
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "python" / "run_rectangular_case.py"
NPZ_EXPORTER = ROOT / "python" / "export_all_timesteps_to_npz.py"
NPZ_VALIDATOR = ROOT / "python" / "validate_npz_series.py"
NPZ_COMPARATOR = ROOT / "python" / "compare_npz_to_reference.py"

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # numpy is required for any NPZ handling
    HAS_NUMPY = False


def write_npz(path: Path, *, t_max: float, melt: float, frames: int = 3, scale: float = 1.0, nz: int = 2) -> None:
    nx, ny = 4, 3
    temperature = np.full((frames, nx, ny, nz), 293.15, dtype=np.float32)
    temperature[-1, 0, 0, 0] = t_max
    melt_region = np.zeros((frames, nx, ny, nz), dtype=np.float32)
    melt_region[-1, 0, 0, 0] = melt
    solid = np.ones((frames, nx, ny, nz), dtype=np.float32)
    solid[-1, 0, 0, 0] = 0.0
    pressure = np.full((frames, nx, ny, nz), 1.0e5 * scale, dtype=np.float32)
    np.savez_compressed(
        path,
        x=np.arange(nx, dtype=np.float32),
        y=np.arange(ny, dtype=np.float32),
        z=np.arange(nz, dtype=np.float32),
        time_s=np.linspace(0.0, 1.0e-3, frames).astype(np.float64),
        # The real exporter writes this on EVERY export as a 0-d array
        # (export_all_timesteps_to_npz.py:298). Omitting it here is what let the
        # comparator's IndexError on rank-0 keys stay green for so long: the
        # fixture was the only NPZ it ever saw without one.
        frame_stride=np.asarray(1, dtype=np.int64),
        Temperature=temperature,
        **{"Melt Region": melt_region, "Solid Fraction": solid},
        Pressure=pressure,
    )


@unittest.skipUnless(HAS_NUMPY, "numpy is required for the NPZ export path")
class NpzExportTests(unittest.TestCase):
    def test_run_script_defaults_export_to_npz(self) -> None:
        result = subprocess.run(
            ["python3", str(RUNNER), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.assertIn("--export", result.stdout)
        self.assertIn("npz", result.stdout)
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('default="npz"', source)
        self.assertIn("export_all_timesteps_npz", source)

    def test_npz_exporter_keeps_lab_frame_and_strips_ghosts_without_resampling(self) -> None:
        source = NPZ_EXPORTER.read_text(encoding="utf-8")
        # Lab frame, full grid, ghost strip, and no resampling / moving frame.
        self.assertIn("Noninertialreferenceframe = 0", source)
        self.assertIn("vtkGhostType", source)
        self.assertIn("vtkStructuredGrid", source)
        for forbidden in ("cKDTree", "moving_frame", "moving_average", "IsoValue", "Isovolume"):
            self.assertNotIn(forbidden, source)

    def test_validator_passes_on_heated_melted_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "case.npz"
            write_npz(npz, t_max=1800.0, melt=1.0)
            result = subprocess.run(
                ["python3", str(NPZ_VALIDATOR), str(npz),
                 "--initial-temperature-k", "293.15", "--min-temperature-rise-k", "100", "--require-melt"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("max_temperature_k=1800", result.stdout)

    def test_validator_check_initial_compares_frame0_to_the_configured_temperature(self) -> None:
        """README documents `--check-initial` / `--initial-tolerance-k`: frame 0's median
        metal temperature must sit within the tolerance of the configured initial."""
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "case.npz"
            write_npz(npz, t_max=1800.0, melt=1.0)      # frame 0 is 293.15 K everywhere
            base = ["python3", str(NPZ_VALIDATOR), str(npz), "--check-initial"]
            ok = subprocess.run(base + ["--initial-temperature-k", "293.15"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            bad = subprocess.run(base + ["--initial-temperature-k", "500", "--initial-tolerance-k", "25"],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(bad.returncode, 2, bad.stdout + bad.stderr)
            self.assertIn("frame", (bad.stdout + bad.stderr).lower())

    def test_validator_fails_without_required_melt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "case.npz"
            write_npz(npz, t_max=1800.0, melt=0.0)
            result = subprocess.run(
                ["python3", str(NPZ_VALIDATOR), str(npz), "--require-melt"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Melt Region", result.stderr)

    def test_comparator_passes_on_almost_identical_and_flags_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.npz"
            b = Path(tmp) / "b.npz"
            far = Path(tmp) / "far.npz"
            write_npz(a, t_max=1800.0, melt=1.0)
            write_npz(b, t_max=1801.0, melt=1.0)  # almost identical
            write_npz(far, t_max=9000.0, melt=1.0, scale=3.0)  # diverged
            ok = subprocess.run(
                ["python3", str(NPZ_COMPARATOR), str(a), str(b)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            self.assertIn("comparison passed", ok.stdout)
            bad = subprocess.run(
                ["python3", str(NPZ_COMPARATOR), str(a), str(far)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(bad.returncode, 2)

    def test_comparator_rejects_mismatched_grids(self) -> None:
        # A gas-gap-stripped reference has fewer z-cells; comparing global stats
        # across different cell populations must error, not silently pass.
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp) / "full.npz"
            stripped = Path(tmp) / "stripped.npz"
            write_npz(full, t_max=1800.0, melt=1.0, nz=4)
            write_npz(stripped, t_max=1800.0, melt=1.0, nz=2)  # same run, z-band cropped
            result = subprocess.run(
                ["python3", str(NPZ_COMPARATOR), str(full), str(stripped)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("different grids", result.stderr)

    def test_comparator_reports_a_requested_field_that_neither_file_has(self) -> None:
        """`--field` naming an absent array must raise the tool's own CompareError
        (exit 2, a sentence), not numpy's KeyError traceback (exit 1). The guard
        existed but ran AFTER the arrays were touched, so it could never fire."""
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.npz"
            b = Path(tmp) / "b.npz"
            write_npz(a, t_max=1800.0, melt=1.0)
            write_npz(b, t_max=1801.0, melt=1.0)
            result = subprocess.run(
                ["python3", str(NPZ_COMPARATOR), str(a), str(b), "--field", "Nope"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("Requested fields missing", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
