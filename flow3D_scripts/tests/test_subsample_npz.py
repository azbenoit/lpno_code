#!/usr/bin/env python3
"""Tests for the NPZ time-subsampler (no license needed)."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import subsample_npz  # noqa: E402


def _make(tmp: str, T: int, nx: int = 5, ny: int = 4, nz: int = 3) -> Path:
    p = Path(tmp) / "r.npz"
    rng = np.random.default_rng(0)
    np.savez_compressed(
        p,
        x=np.arange(nx, dtype=float), y=np.arange(ny, dtype=float), z=np.arange(nz, dtype=float),
        time_s=np.linspace(0.0, 1.0, T),
        Temperature=rng.random((T, nx, ny, nz)),
        Velocity=rng.random((T, nx, ny, nz, 3)),
        power_w=np.array(150.0), case_name=np.array("c"),
    )
    return p


class SubsampleNpzTests(unittest.TestCase):
    def test_thins_time_axis_keeps_endpoints_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = _make(tmp, T=401)
            n = subsample_npz.subsample(p, 100)
            d = np.load(p, allow_pickle=True)
            self.assertLessEqual(n, 100)
            self.assertEqual(len(d["time_s"]), n)
            self.assertEqual(d["Temperature"].shape, (n, 5, 4, 3))
            self.assertEqual(d["Velocity"].shape, (n, 5, 4, 3, 3))
            self.assertAlmostEqual(float(d["time_s"][0]), 0.0)
            self.assertAlmostEqual(float(d["time_s"][-1]), 1.0)
            self.assertEqual(len(d["x"]), 5)  # coordinate axes untouched
            self.assertEqual(float(d["power_w"]), 150.0)

    def test_no_op_when_already_within_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = _make(tmp, T=50)
            self.assertEqual(subsample_npz.subsample(p, 100), 50)

    def test_rewrite_refreshes_an_existing_sha256_sidecar(self) -> None:
        """Thinning rewrites the NPZ in place; a sidecar written for the old bytes
        must be refreshed, or `check_file` (which prefers the sidecar) reports a
        healthy file as corrupt. `bash/finalize_run.sh` runs this on every case."""
        import verify_npz_checksums as vnc
        with tempfile.TemporaryDirectory() as tmp:
            p = _make(tmp, T=401)
            vnc.write_sidecar(p)
            subsample_npz.subsample(p, 4)
            self.assertEqual(vnc.check_file(p), (True, "sha256 ok"))

    def test_rewrite_without_a_sidecar_writes_none(self) -> None:
        """Pre-sidecar data keeps its CRC fallback; thinning must not invent one."""
        import verify_npz_checksums as vnc
        with tempfile.TemporaryDirectory() as tmp:
            p = _make(tmp, T=50)
            subsample_npz.subsample(p, 10)
            self.assertFalse(vnc.sidecar_path(p).exists())


if __name__ == "__main__":
    unittest.main()
