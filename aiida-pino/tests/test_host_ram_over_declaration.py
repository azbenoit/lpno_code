"""An over-declared host-RAM booking should be announced, not refused.

Host RAM is booked machine-wide, so a booking far above the measured peak does not fail
anything -- it halves the box. Recorded in `_check_host_ram_against_loader_settings`:
cropped arms booked 40960 MiB while peaking at 8188 and 9673, and against ~88542 MiB
usable that admitted only TWO concurrent arms on a four-card box.
"""

import unittest
from contextlib import redirect_stdout
from io import StringIO

from aiida_pino.utils import profiles


def _advice(profile, peak):
    original = profiles.recorded_host_ram_peak_mb
    profiles.recorded_host_ram_peak_mb = lambda *a, **k: peak
    try:
        out = StringIO()
        with redirect_stdout(out):
            profiles._warn_if_host_ram_is_over_declared(profile)
        return out.getvalue()
    finally:
        profiles.recorded_host_ram_peak_mb = original


class HostRamOverDeclarationTest(unittest.TestCase):
    def test_the_measured_incident_is_announced(self):
        # The cropped arms: booked 40960, peaked 9673.
        text = _advice(
            {"config_name": "rigno_1601", "host_ram_mb": 40960,
             "env": {"PINO_LAZY_CROP_Z_TOP": "10"}},
            9673,
        )
        self.assertIn("WARNING", text)
        self.assertIn("40960", text)
        self.assertIn("9673", text)
        self.assertIn(str(9673 * 2 + 4096), text)  # names the number to use instead

    def test_the_largest_legitimate_run_on_the_box_is_not_flagged(self):
        # GAOT_randompath peaks at 56861 MiB. This is why a flat 20480 ceiling was
        # rejected: it would refuse the biggest real run while passing the incident above.
        self.assertEqual(_advice({"config_name": "gaot_randompath", "host_ram_mb": 61440}, 56861), "")
        self.assertIn("WARNING", _advice({"config_name": "gaot_randompath", "host_ram_mb": 20480}, 4000))

    def test_silent_without_a_measurement_or_a_booking(self):
        # A new config has no recorded peak, and saying nothing is the required behaviour:
        # an advisory invented from no data is worse than no advisory.
        self.assertEqual(_advice({"config_name": "brand_new", "host_ram_mb": 65536}, None), "")
        self.assertEqual(_advice({"config_name": "rigno_1601"}, 4000), "")

    def test_the_lookup_is_inert_rather_than_fatal_without_a_database(self):
        # Called from profile validation, which runs in contexts with no loaded profile.
        self.assertIsNone(profiles.recorded_host_ram_peak_mb(None))

    def test_validation_runs_the_advisory(self):
        # The check is only worth anything if the submission path calls it.
        import inspect
        self.assertIn(
            "_warn_if_host_ram_is_over_declared(profile)",
            inspect.getsource(profiles.validate_training_profile),
        )


if __name__ == "__main__":
    unittest.main()


class GpuVramBandTest(unittest.TestCase):
    """A manual VRAM booking below the measured peak is admitted and then killed."""

    def _advice(self, profile, peak):
        original = profiles.recorded_gpu_vram_peak_mb
        profiles.recorded_gpu_vram_peak_mb = lambda *a, **k: peak
        try:
            out = StringIO()
            with redirect_stdout(out):
                profiles._warn_if_gpu_vram_is_under_declared(profile)
            return out.getvalue()
        finally:
            profiles.recorded_gpu_vram_peak_mb = original

    def test_a_quarter_sized_booking_is_announced_with_the_band(self):
        # The 2026-08-22 shape: six evals booked about a quarter of what was needed.
        text = self._advice({"config_name": "rigno_1601", "gpu_vram_mb": 4000}, 16000)
        self.assertIn("WARNING", text)
        self.assertIn("16000", text)
        self.assertIn("gpu_vram_mb", text)

    def test_an_adequate_or_absent_booking_says_nothing(self):
        # Absent is the GOOD path: it is auto-sized from the measured footprint.
        self.assertEqual(self._advice({"config_name": "rigno_1601", "gpu_vram_mb": 16000}, 16000), "")
        # The near-miss class: 10000 booked against a config whose largest peak is 19574.
        self.assertEqual(self._advice({"config_name": "rp", "gpu_vram_mb": 10000}, 19574), "")
        self.assertEqual(self._advice({"config_name": "rigno_1601"}, 16000), "")
        self.assertEqual(self._advice({"config_name": "new", "gpu_vram_mb": 1}, None), "")

    def test_the_band_it_prints_matches_the_automatic_sizing(self):
        # The advice must name the number the auto path would book, not one of its own.
        from aiida_pino.utils.vram import vram_reserve_mb
        self.assertIn(str(vram_reserve_mb(16000)),
                      self._advice({"config_name": "c", "gpu_vram_mb": 4000}, 16000))

    def test_validation_runs_both_advisories(self):
        import inspect
        src = inspect.getsource(profiles.validate_training_profile)
        self.assertIn("_warn_if_gpu_vram_is_under_declared(profile)", src)
        self.assertIn("_warn_if_host_ram_is_over_declared(profile)", src)
