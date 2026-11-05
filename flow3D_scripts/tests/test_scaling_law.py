#!/usr/bin/env python3
"""Short, no-license tests for the Gan-2021 keyhole scaling-law consistency check.

Covers: the keyhole-number formula on a hand-computed Gan SD1 row, reference
JSON loading and fitted-curve monotonicity, the steady-window keyhole depth
metric on a synthetic NPZ, and check-mode wiring (run-dir and explicit-args
paths) without FLOW-3D.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import validate_scaling_law as vsl  # noqa: E402

REFERENCE = ROOT / "python" / "data" / "gan_scaling_reference.json"

# Gan et al. 2021 SD1, Ti64 spot 95 um, P=129.36 W, V=0.4 m/s row (their own
# property convention: liquid-at-melting values).
GAN_TI64_ROW = {
    "eta": 0.49972342414272736,
    "power_w": 129.36,
    "speed_m_per_s": 0.4,
    "r0_m": 4.75e-05,
    "rho_kg_m3": 3920.0,
    "cp_j_kg_k": 830.0,
    "alpha_m2_s": 1.0265552003934103e-05,
    "dtv_k": 3267.0,
}


def _gan_ke(**kw) -> float:
    return vsl.keyhole_number(
        eta=kw["eta"],
        power_w=kw["power_w"],
        speed_m_per_s=kw["speed_m_per_s"],
        r0_m=kw["r0_m"],
        rho_kg_m3=kw["rho_kg_m3"],
        cp_j_kg_k=kw["cp_j_kg_k"],
        alpha_m2_s=kw["alpha_m2_s"],
        dtv_k=kw["dtv_k"],
    )


def _write_synthetic_npz(path: Path, nz: int = 15, pocket_top_index: int = 5) -> None:
    """20-frame FoF series: pocket appears at frame 8, deepens after laser-off.

    z centers are (i - (nz/2 - 0.5 - extra)) ... simply (arange(nz) - (nz - 5.5)) * 1e-5
    so that 10 + (nz - 15) cells sit below the z=0 surface with 5 above.
    """
    nx, ny, nt = 8, 6, 20
    below = nz - 5  # cells below the surface
    z = (np.arange(nz) - (below - 0.5)) * 1e-5  # centers ...,-0.5e-5 | +0.5e-5,...
    x = (np.arange(nx) + 0.5) * 1e-5
    y = (np.arange(ny) + 0.5) * 1e-5
    fof = np.ones((nt, nx, ny, nz), dtype=np.float32)
    fof[:, :, :, below:] = 0.0  # gas above the surface
    fof[:, 0, :, :] = 0.0  # edge artifact column at the x=0 wall: must be excluded
    fof[:, :, 0, :] = 0.0  # and at the y=0 wall
    # keyhole pocket from frame 8 on, interior column, gas down to pocket_top_index
    fof[8:, 4, 3, pocket_top_index:] = 0.0
    # partially-filled cell just below the pocket: FoF=0.6 is fluid, not gas
    fof[8:, 4, 3, pocket_top_index - 1] = 0.6
    # post-laser deepening (frames 17..19) must not enter the steady window
    fof[17:, 4, 3, 1:] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        z=z.astype(np.float32),
        time_s=np.arange(nt, dtype=np.float64) * 1e-4,
        **{"Fraction Of Fluid": fof},
    )


def _write_case(root: Path, case: str, power_w: float, depth_cells_nz30: int = 9) -> Path:
    """A minimal run dir: npz/<case>.npz + case_manifest.json with Gan-like Ti64."""
    run_dir = root / case
    _write_synthetic_npz(run_dir / "npz" / f"{case}.npz", nz=30, pocket_top_index=depth_cells_nz30)
    manifest = {
        "schema_version": 1,
        "case_name": case,
        "input": {
            "laser": {
                "power_w": power_w,
                "scan_speed_m_per_s": 0.4,
                "beam_radius_m": 4.75e-05,
            },
            "simulation": {"initial_temperature_k": 293.15},
        },
        "derived": {"substrate_bounds_m": {"z": [-4e-4, 0.0]}},
        "material": {
            "name": "stub_gan_ti64",
            "file": "materials/does_not_exist/does_not_exist.json",
            "properties": {
                "density_kg_m3": 3920.0,
                "specific_heat_j_kg_k": 830.0,
                "thermal_conductivity_w_m_k": 33.4,
                "vaporization_temperature_k": 3560.15,
                "liquidus_temperature_k": 1923.15,
            },
        },
    }
    (run_dir / "case_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


class KeyholeNumberFormula(unittest.TestCase):
    def test_hand_computed_gan_row(self) -> None:
        # Independent arithmetic for the same definition:
        # Ke = eta*P / ((Tv-T0) * pi * rho * cp * sqrt(alpha * V * r0^3))
        k = GAN_TI64_ROW
        denominator = (
            k["dtv_k"] * math.pi * k["rho_kg_m3"] * k["cp_j_kg_k"]
            * math.sqrt(k["alpha_m2_s"] * k["speed_m_per_s"] * k["r0_m"] ** 3)
        )
        expected = k["eta"] * k["power_w"] / denominator
        self.assertAlmostEqual(_gan_ke(**k), expected, places=10)
        # and against the frozen value for this row so a formula edit cannot
        # silently rescale both sides of the identity above
        self.assertAlmostEqual(_gan_ke(**k), 2.918, delta=0.003)

    def test_scaling_identities(self) -> None:
        base = _gan_ke(**GAN_TI64_ROW)
        double_p = dict(GAN_TI64_ROW, power_w=2 * GAN_TI64_ROW["power_w"])
        self.assertAlmostEqual(_gan_ke(**double_p), 2 * base, places=10)
        quad_v = dict(GAN_TI64_ROW, speed_m_per_s=4 * GAN_TI64_ROW["speed_m_per_s"])
        self.assertAlmostEqual(_gan_ke(**quad_v), base / 2, places=10)


class ReferenceCurve(unittest.TestCase):
    def test_load_and_monotone(self) -> None:
        ref = vsl.load_reference(REFERENCE)
        self.assertEqual(len(ref["points"]), 87)
        self.assertEqual(len({p["group"] for p in ref["points"]}), 4)
        self.assertGreater(ref["fit"]["residual_sd"], 0.5)
        self.assertLess(ref["fit"]["residual_sd"], 1.5)
        ke_min, ke_max = ref["fit"]["ke_range"]
        grid = np.linspace(ke_min, ke_max, 200)
        values = np.array([vsl.eval_curve(ref, ke) for ke in grid])
        self.assertTrue(np.all(np.diff(values) > 0), "fitted curve must be monotone in Ke")

    def test_prediction_near_known_point(self) -> None:
        # Ti64 129.36 W / 0.4 m/s: measured e*=85.32/47.5=1.796, Ke=2.918;
        # the pooled curve should predict within one reference sd.
        ref = vsl.load_reference(REFERENCE)
        predicted = vsl.eval_curve(ref, 2.918)
        self.assertLess(abs(predicted - 1.796), ref["fit"]["residual_sd"])


class DepthMetric(unittest.TestCase):
    def test_synthetic_pocket_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "case.npz"
            _write_synthetic_npz(npz)  # nz=15: 10 below surface, pocket at index 5
            result = vsl.measure_depth_npz(npz)
        # pocket gas centers reach z index 5 => center (5-9.5)*1e-5 = -4.5e-5;
        # steady window (frames 12..16) is entirely inside the pocket-on span,
        # the deeper post-laser frames (17..19) and the wall columns are excluded.
        self.assertAlmostEqual(result["depth_m"], 4.5e-5, places=10)  # float32 z axis
        self.assertEqual(result["n_frames_total"], 20)
        self.assertEqual(result["steady_frames"], (12, 16))

    def test_no_pocket_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "flat.npz"
            nx, ny, nz, nt = 6, 6, 10, 10
            fof = np.ones((nt, nx, ny, nz), dtype=np.float32)
            fof[:, :, :, 5:] = 0.0
            np.savez_compressed(
                npz,
                x=np.arange(nx, dtype=np.float32),
                y=np.arange(ny, dtype=np.float32),
                z=((np.arange(nz) - 4.5) * 1e-5).astype(np.float32),
                time_s=np.arange(nt, dtype=np.float64),
                **{"Fraction Of Fluid": fof},
            )
            result = vsl.measure_depth_npz(npz)
        self.assertEqual(result["depth_m"], 0.0)


class CheckMode(unittest.TestCase):
    def test_run_dir_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_case(Path(tmp), "synth_pass", power_w=129.36)
            result = vsl.check_run_dir(run_dir, eta=GAN_TI64_ROW["eta"])
        # nz=30 fixture: 25 cells below the surface, pocket top at z index 9,
        # deepest gas center z[9] = (9-24.5)*1e-5 = -1.55e-4 m, so
        # e* = 1.55e-4/4.75e-5 = 3.26 vs ~2.29 predicted at Ke=2.918: |z|~1.
        self.assertAlmostEqual(result["ke"], 2.918, delta=0.003)
        self.assertEqual(result["props_source"], "manifest-scalar")
        self.assertAlmostEqual(result["depth_m"], 1.55e-4, places=10)
        self.assertLessEqual(abs(result["z"]), 2.0)
        self.assertEqual(result["status"], "PASS")

    def test_run_dir_fail_on_wrong_power(self) -> None:
        # Same measured depth, but the manifest claims 500 W: the curve expects
        # a much deeper keyhole, so the case must FAIL.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_case(Path(tmp), "synth_fail", power_w=500.0)
            result = vsl.check_run_dir(run_dir, eta=GAN_TI64_ROW["eta"])
        self.assertGreater(abs(result["z"]), 3.0)
        self.assertEqual(result["status"], "FAIL")

    def test_explicit_args_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat = Path(tmp) / "stub.json"
            mat.write_text(json.dumps({
                "schema_version": 1,
                "material_name": "stub",
                "properties": {
                    "density_kg_m3": 3920.0,
                    "specific_heat_j_kg_k": 830.0,
                    "thermal_conductivity_w_m_k": 33.4,
                    "vaporization_temperature_k": 3560.15,
                    "liquidus_temperature_k": 1923.15,
                },
            }), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "python" / "validate_scaling_law.py"), "check",
                 "--power-w", "129.36", "--speed-m-per-s", "0.4", "--radius-m", "4.75e-5",
                 "--material-json", str(mat), "--depth-um", "400",
                 "--eta", str(GAN_TI64_ROW["eta"])],
                capture_output=True, text=True,
            )
        # 400 um on 47.5 um spot radius is e*=8.4 vs ~2.3 predicted: FAIL, exit 1
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("FAIL", proc.stdout)

    def test_default_eta_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_case(Path(tmp), "synth_eta", power_w=129.36)
            result = vsl.check_run_dir(run_dir)  # no eta given
        self.assertEqual(result["eta_source"], "default")
        self.assertAlmostEqual(result["eta"], 0.35)


class SweepMode(unittest.TestCase):
    def test_sweep_summary_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            _write_case(batch, "case_ok", power_w=129.36)
            _write_case(batch, "case_bad", power_w=500.0)
            (batch / "not_a_case").mkdir()
            proc = subprocess.run(
                [sys.executable, str(ROOT / "python" / "validate_scaling_law.py"), "sweep",
                 str(batch), "--eta", str(GAN_TI64_ROW["eta"])],
                capture_output=True, text=True,
            )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("case_ok", proc.stdout)
        self.assertIn("case_bad", proc.stdout)
        self.assertIn("n=3", proc.stdout)  # 2 cases + 1 skipped non-case dir
        self.assertIn("pass=1", proc.stdout)
        self.assertIn("fail=1", proc.stdout)
        self.assertIn("skip=1", proc.stdout)


if __name__ == "__main__":
    unittest.main()


class SweepSystematicBiasTests(unittest.TestCase):
    """A sweep must surface systematic bias, not just per-case pass/fail.

    Found 2026-07-30 by sweeping the original straight-path dataset: 56/56
    cases PASSED while every z was negative (-0.17 to -1.21) and measured
    e/r0 ran 50-80% below the curve. The band is absolute in e/r0 units while
    e/r0 spans 0.1-10, so in the shallow regime even a 100% error passes. The
    per-case verdict is therefore blind to exactly the systematic shortfall the
    check is most wanted for; the summary must report mean z so it shows up.
    """

    def test_summary_reports_mean_z(self) -> None:
        line = vsl.summarize([-0.5, -0.9, -1.1], n_skip=2)
        self.assertIn("mean z=-0.83", line)
        self.assertIn("skip=2", line)

    def test_summary_flags_a_systematic_offset(self) -> None:
        biased = vsl.summarize([-0.6, -0.7, -0.8], n_skip=0)
        self.assertIn("SYSTEMATIC", biased)
        unbiased = vsl.summarize([-0.4, 0.1, 0.3], n_skip=0)
        self.assertNotIn("SYSTEMATIC", unbiased)
