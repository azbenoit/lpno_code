"""Integrity checks for exported NPZ cases.

Regression origin (2026-08-05): `ti64_random150` held a case whose
`Temperature.npy` member failed its CRC. Nothing detected it until an eval
crashed days later, because the manifests carry run config and no checksum.
Each test here corresponds to a documented mode of `verify_npz_checksums.py`.
"""

import io
import struct
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import verify_npz_checksums as vnc  # noqa: E402


def _make_npz(path: Path) -> None:
    # Noise rather than arange: it compresses poorly, so the member's payload is
    # large enough that corrupting its middle cannot land in a header.
    rng = np.random.default_rng(0)
    np.savez_compressed(path, Temperature=rng.standard_normal((16, 16, 16)).astype(np.float32))


def _corrupt_member_payload(path: Path) -> None:
    """Flip a byte inside the compressed payload, leaving every header intact.

    This reproduces the observed failure: the recorded CRC still says one thing
    while the stored bytes decode to another, so the damage is invisible until a
    reader decompresses that member (`Bad CRC-32 for file 'Temperature.npy'`).
    Patching a header instead would not reproduce it -- testzip() validates
    against the central directory, not the local header.
    """
    raw = bytearray(path.read_bytes())
    start = raw.find(b"PK\x03\x04")
    name_len = struct.unpack_from("<H", raw, start + 26)[0]
    extra_len = struct.unpack_from("<H", raw, start + 28)[0]
    payload_start = start + 30 + name_len + extra_len
    end_of_central = raw.rfind(b"PK\x01\x02")
    target = payload_start + (end_of_central - payload_start) // 2
    raw[target] ^= 0xFF
    path.write_bytes(bytes(raw))


class VerifyNpzChecksumsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_clean_file_passes_without_sidecar(self):
        f = self.root / "case.npz"
        _make_npz(f)
        ok, reason = vnc.check_file(f)
        self.assertTrue(ok, reason)
        self.assertIn("crc ok", reason)

    def test_corrupt_member_is_detected_without_sidecar(self):
        """The real failure mode: no baseline exists, CRC alone must catch it."""
        f = self.root / "case.npz"
        _make_npz(f)
        _corrupt_member_payload(f)
        ok, reason = vnc.check_file(f)
        self.assertFalse(ok, f"corruption went undetected: {reason}")

    def test_sidecar_roundtrip_passes(self):
        f = self.root / "case.npz"
        _make_npz(f)
        digest = vnc.write_sidecar(f)
        self.assertTrue(vnc.sidecar_path(f).exists())
        self.assertEqual(vnc.read_sidecar(f), digest)
        ok, reason = vnc.check_file(f)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "sha256 ok")

    def test_sidecar_mismatch_is_detected(self):
        """Catches a changed file even when every member's CRC is self-consistent."""
        f = self.root / "case.npz"
        _make_npz(f)
        vnc.write_sidecar(f)
        # A perfectly valid archive with different contents: CRC alone would pass it.
        np.savez_compressed(f, Temperature=np.ones((16, 16, 16), dtype=np.float32))
        ok, reason = vnc.check_file(f)
        self.assertFalse(ok, reason)
        self.assertIn("sha256 mismatch", reason)

    def test_main_exit_status_gates_a_batch(self):
        good = self.root / "good.npz"
        bad = self.root / "bad.npz"
        _make_npz(good)
        _make_npz(bad)
        _corrupt_member_payload(bad)
        buf = io.StringIO()
        stdout, sys.stdout = sys.stdout, buf
        try:
            status = vnc.main([str(self.root), "--quiet"])
        finally:
            sys.stdout = stdout
        self.assertEqual(status, 1)
        self.assertIn("bad.npz", buf.getvalue())

    def test_main_write_mode_creates_sidecars(self):
        for name in ("a.npz", "b.npz"):
            _make_npz(self.root / name)
        buf = io.StringIO()
        stdout, sys.stdout = sys.stdout, buf
        try:
            status = vnc.main([str(self.root), "--write", "--quiet"])
        finally:
            sys.stdout = stdout
        self.assertEqual(status, 0)
        for name in ("a.npz", "b.npz"):
            self.assertTrue(vnc.sidecar_path(self.root / name).exists())


if __name__ == "__main__":
    unittest.main()
