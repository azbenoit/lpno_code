"""Raise-only reconciliation of a seat's host-RAM booking to its measured use.

Root cause it closes (2026-09-09): bookings were OPEN-LOOP -- declared at submit
from guesses or transferred measurements, never compared to what the run actually
used, while admission packs strictly by declaration. Six DataLoader workers were
SIGKILLed this campaign by under-declared co-tenants the ledger happily admitted.

Raise-only is deliberate: shrinking over-declarations would admit more co-tenants,
and each one evicts page cache the lazy loader depends on -- measured 2026-09-09 as
a 2x throughput loss on the incumbent (4.5 -> 8.7 s/step) with every anonymous-RAM
booking satisfied. Phantom headroom is accidentally protective; do not reclaim it here.
"""
import json, os, unittest, tempfile
from pathlib import Path

from lpno import gpu_spool as gs


class ReconcileLeaseRam(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "123-1.lease"
        self.fd = gs._hold(self.path, {"name": "seat:x", "vram_mb": 1000,
                                       "ram_mb": 40960, "since": "t"})

    def tearDown(self):
        try: os.close(self.fd)
        except OSError: pass
        self.tmp.cleanup()

    def read(self):
        return json.loads(self.path.read_text())

    def test_underdeclared_booking_is_raised_with_margin_and_marked(self):
        changed = gs._reconcile_lease_ram(self.fd, self.path, measured_mb=50000)
        self.assertTrue(changed)
        d = self.read()
        self.assertGreaterEqual(d["ram_mb"], 50000)          # at least the truth
        self.assertLessEqual(d["ram_mb"], int(50000 * 1.2))   # margin bounded
        self.assertEqual(d["ram_mb_measured"], 50000)
        self.assertEqual(d["ram_mb_declared"], 40960)         # the original survives

    def test_a_booking_that_already_covers_the_measurement_does_not_ratchet(self):
        """Regression: raising whenever measured*margin exceeded the BOOKING made the
        lease climb against a flat measurement -- 49152 -> 54948 -> 55948 -> 56007 while
        real use sat at 47.8-48.7 GB, which starved a 34 GB box into one-arm-at-a-time.
        The trigger must be the measurement exceeding the booking, not the margin doing so."""
        self.path.write_text(__import__("json").dumps(
            {"name": "seat:x", "vram_mb": 1000, "ram_mb": 49152, "since": "t"}))
        for measured in (47781, 48651, 48702):        # all BELOW the 49152 booking
            changed = gs._reconcile_lease_ram(self.fd, self.path, measured_mb=measured)
            self.assertFalse(changed, f"raised on measured={measured} under a 49152 booking")
            self.assertEqual(self.read()["ram_mb"], 49152)  # untouched, every round

    def test_overdeclared_booking_is_never_shrunk(self):
        changed = gs._reconcile_lease_ram(self.fd, self.path, measured_mb=20000)
        self.assertFalse(changed)
        d = self.read()
        self.assertEqual(d["ram_mb"], 40960)                  # raise-only, by design
        self.assertEqual(d.get("ram_mb_measured"), 20000)     # but the truth is recorded

    def test_raised_booking_is_visible_to_admission_arithmetic(self):
        gs._reconcile_lease_ram(self.fd, self.path, measured_mb=50000)
        entries = [json.loads(p.read_text()) for p in self.path.parent.glob("*.lease")]
        self.assertEqual(sum(e["ram_mb"] for e in entries), self.read()["ram_mb"])

    def test_monotonic_across_calls(self):
        gs._reconcile_lease_ram(self.fd, self.path, measured_mb=50000)
        first = self.read()["ram_mb"]
        gs._reconcile_lease_ram(self.fd, self.path, measured_mb=45000)  # dips later
        self.assertEqual(self.read()["ram_mb"], first)        # never lowered


if __name__ == "__main__":
    unittest.main()
