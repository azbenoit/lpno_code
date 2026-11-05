"""Every training must leave a measured VRAM peak behind, or bookings stay folklore.

All four manuscript path donors record `gpu_vram_peak_mb = None`; every booking this week
was therefore a guess or an inherited guess, and the same day produced BOTH failure modes:
over-booking that idled three GPUs behind a full ledger (12000 booked vs 7.4-7.7 GiB used)
and an OOM from an inherited under-booking (74176: 11218 booked, died wanting 990 MiB).
"""
import json
import unittest
from unittest import mock

from lpno.models.pipeline.model_domain import record_vram_peak


class RecordVramPeak(unittest.TestCase):
    def test_writes_peak_json_and_returns_mb(self, ):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("torch.cuda.is_available", return_value=True), \
                 mock.patch("torch.cuda.max_memory_reserved", return_value=7919 * 2**20), \
                 mock.patch("torch.cuda.max_memory_allocated", return_value=7000 * 2**20):
                peak = record_vram_peak(d)
            self.assertEqual(peak, 7919)
            data = json.loads((pathlib.Path(d) / "vram_peak.json").read_text())
            self.assertEqual(data["gpu_vram_peak_mb"], 7919)
            self.assertEqual(data["gpu_vram_allocated_peak_mb"], 7000)

    def test_no_cuda_is_a_quiet_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("torch.cuda.is_available", return_value=False):
                self.assertIsNone(record_vram_peak(d))

    def test_failure_never_raises(self):
        """Recording is bookkeeping; it must not be able to kill a finished run."""
        with mock.patch("torch.cuda.is_available", side_effect=RuntimeError("boom")):
            self.assertIsNone(record_vram_peak("/nonexistent/dir"))


if __name__ == "__main__":
    unittest.main()
