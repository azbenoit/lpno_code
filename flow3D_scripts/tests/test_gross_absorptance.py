"""Gross absorptance reconstruction, so a simulated number can be compared
with integrating-sphere calorimetry without a units mismatch.

The measurement is an ABSORPTANCE: the fraction of incident laser power that
crosses the surface. `absorbed_fraction` returns something else -- the fraction
still held in the metal as thermal energy, after radiation, convection and
evaporating material have carried some back out. Comparing the two directly
understates the model by several percentage points, which is what this adds back.

It is a LOWER BOUND with two failure modes, and `plausible` is False in both:

  1. The mass audit cannot distinguish evaporated mass from ejected droplets, so
     in strong keyholing it over-charges latent heat and the total runs past any
     credible absorptance (t2_206w reconstructs to 98 %).
  2. `total fluid mass` is a NET phase-change balance. Every deck carries
     rsize_cond=1 and acondens=1.0, so recondensation returns mass to the fluid,
     and the audit reports evaporation MINUS recondensation. On t2_nist195 it
     falls, bottoms at 640 us and climbs back; across 2861 runs it is
     non-monotone in 50.7 % of cases. Endpoints across such a window still net
     out to a believable loss, so the guard has to read the series.

Between them those cover most of the reference set, including the NIST scan the
reconstruction was written for. A caller that ignores `plausible` will quote a
number this module rejects.
"""
import unittest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np

import melt_metrics as mm


def _hd3out(rows, path):
    """Minimal hd3out with the two audits the reconstruction reads."""
    out = []
    for t, energy, mass in rows:
        out.append(f" time = {t:.6E}   cycle =    1\n")
        out.append(f"   total fluid #1 thermal energy =  {energy:.6E}\n")
        out.append(f"   total fluid mass =  {mass:.6E}\n")
    path.write_text("".join(out))
    return path


class GrossAbsorptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_retained_matches_absorbed_fraction(self):
        """The retained term must equal the existing metric, or the two disagree
        about the same audit and one of the paper's numbers is wrong."""
        f = _hd3out([(2.0e-4, 0.0, 1.0e-6), (8.0e-4, 6.0e-2, 1.0e-6)],
                    self.dir / "hd3out.case")
        got = mm.gross_absorptance(f, power_w=195.0)
        want = mm.absorbed_fraction(f, 195.0)
        self.assertAlmostEqual(got.retained, want, places=9)

    def test_evaporated_mass_adds_latent_and_sensible_heat(self):
        """Mass that leaves took its enthalpy with it; gross must count it."""
        dm = 4.0e-10
        f = _hd3out([(2.0e-4, 0.0, 1.0e-6), (8.0e-4, 4.0e-2, 1.0e-6 - dm)],
                    self.dir / "hd3out.case")
        r = mm.gross_absorptance(f, power_w=195.0)
        self.assertGreater(r.evaporative, 0.0)
        self.assertAlmostEqual(r.gross, r.retained + r.evaporative, places=9)
        self.assertGreater(r.gross, r.retained)

    def test_no_mass_loss_leaves_gross_equal_to_retained(self):
        f = _hd3out([(2.0e-4, 0.0, 1.0e-6), (8.0e-4, 4.0e-2, 1.0e-6)],
                    self.dir / "hd3out.case")
        r = mm.gross_absorptance(f, power_w=195.0)
        self.assertEqual(r.evaporative, 0.0)
        self.assertAlmostEqual(r.gross, r.retained, places=9)

    def test_mass_gain_is_not_negative_evaporation(self):
        """A rising mass audit must not subtract energy: there is no mechanism
        by which the pool gains mass and loses absorbed power."""
        f = _hd3out([(2.0e-4, 0.0, 1.0e-6), (8.0e-4, 4.0e-2, 1.1e-6)],
                    self.dir / "hd3out.case")
        r = mm.gross_absorptance(f, power_w=195.0)
        self.assertEqual(r.evaporative, 0.0)

    def test_over_unity_is_flagged_not_returned_silently(self):
        """Strong keyholing ejects droplets the mass audit cannot tell from
        vapour, so the latent-heat charge is too big and the total exceeds the
        incident power. Observed on rep4_in718_s049_blind (104 %) and t2_206w
        (98 %). The number must be marked unusable, not quietly returned."""
        f = _hd3out([(2.0e-4, 0.0, 1.0e-6), (8.0e-4, 8.0e-2, 1.0e-6 - 6.0e-9)],
                    self.dir / "hd3out.case")
        r = mm.gross_absorptance(f, power_w=195.0)
        self.assertGreater(r.gross, 1.0)
        self.assertFalse(r.plausible)

    def test_physical_case_is_plausible(self):
        f = _hd3out([(2.0e-4, 0.0, 1.0e-6), (8.0e-4, 4.6e-2, 1.0e-6 - 4.2e-10)],
                    self.dir / "hd3out.case")
        r = mm.gross_absorptance(f, power_w=195.0)
        self.assertTrue(r.plausible)
        self.assertLess(r.gross, 1.0)

    def test_non_monotone_mass_is_flagged(self):
        """A mass audit that falls and then RISES is not an evaporation record.

        Recondensation is active in every deck (rsize_cond=1, acondens=1.0), so
        the audit is evaporation minus recondensation and can legitimately climb.
        Measured on t2_nist195 (2026-08-17): it moves 0.2 % over the solve,
        bottoms at 640 us and climbs afterwards. Taking endpoints across that
        reversal hides it -- the net still looks like a plausible loss -- so the
        reversal itself has to be what marks the number unusable.
        """
        f = _hd3out([(2.0e-4, 0.0, 1.000e-6), (4.0e-4, 1.5e-2, 0.9994e-6),
                     (6.4e-4, 3.0e-2, 0.9990e-6), (8.0e-4, 4.6e-2, 0.9993e-6)],
                    self.dir / "hd3out.case")
        r = mm.gross_absorptance(f, power_w=195.0)
        self.assertLess(r.gross, 1.0)          # net looks fine on its own
        self.assertFalse(r.plausible)          # but the series gives it away

    def test_monotone_loss_stays_plausible(self):
        """The guard must key on the reversal, not on having several samples."""
        f = _hd3out([(2.0e-4, 0.0, 1.000e-6), (4.0e-4, 1.5e-2, 0.9998e-6),
                     (6.4e-4, 3.0e-2, 0.9996e-6), (8.0e-4, 4.6e-2, 0.9995e-6)],
                    self.dir / "hd3out.case")
        r = mm.gross_absorptance(f, power_w=195.0)
        self.assertTrue(r.plausible)

    def test_missing_file_returns_nan_without_raising(self):
        """A diagnostic must never abort a scoring sweep, matching
        absorbed_fraction's contract."""
        r = mm.gross_absorptance(self.dir / "hd3out.absent", power_w=195.0)
        self.assertNotEqual(r.retained, r.retained)   # NaN
        self.assertFalse(r.plausible)


class Hd3outAuditTests(unittest.TestCase):
    """The audit parser is the primitive every absorptance number rests on, so
    it is public and tested directly rather than being re-implemented by each
    caller (it had been hand-rolled four times: twice here and twice in the
    manuscript notebook)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pairs_each_value_with_the_time_printed_before_it(self):
        f = _hd3out([(1.0e-4, 1.0, 9.0e-7), (2.0e-4, 2.0, 8.0e-7)],
                    self.dir / "hd3out.case")
        got = mm.hd3out_audit(f, "total fluid #1 thermal energy")
        self.assertEqual(got.shape, (2, 2))
        self.assertAlmostEqual(got[0][0], 1.0e-4)
        self.assertAlmostEqual(got[1][1], 2.0)

    def test_repeated_time_keeps_only_the_first_sample(self):
        """hd3out reprints a block at the same time on restarts and at some
        output steps; a duplicated time makes np.gradient divide by zero."""
        p = self.dir / "hd3out.case"
        p.write_text(" time = 1.000000E-04\n   total fluid mass =  9.000000E-07\n"
                     " time = 1.000000E-04\n   total fluid mass =  9.000000E-07\n"
                     " time = 2.000000E-04\n   total fluid mass =  8.000000E-07\n")
        got = mm.hd3out_audit(p, "total fluid mass")
        self.assertEqual(len(got), 2)

    def test_missing_file_and_absent_quantity_return_empty(self):
        f = _hd3out([(1.0e-4, 1.0, 9.0e-7)], self.dir / "hd3out.case")
        self.assertEqual(len(mm.hd3out_audit(self.dir / "hd3out.absent",
                                             "total fluid mass")), 0)
        self.assertEqual(len(mm.hd3out_audit(f, "cumulative fluid mass source")), 0)


class AbsorbedFractionSeriesTests(unittest.TestCase):
    """Time-resolved retained fraction. The manuscript needs a trace, not a
    window scalar, and hand-rolling it is how the evaporative-drift bug reached
    a figure. One implementation, pinned to the scalar."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # Energy rising linearly: a constant 50 % of a 200 W beam over the window.
        self.f = _hd3out([(t, 0.02 * (t - 1.0e-4) * 1.0e4, 1.0e-6)
                          for t in np.linspace(1.0e-4, 9.0e-4, 9)],
                         self.dir / "hd3out.case")

    def tearDown(self):
        self.tmp.cleanup()

    def test_window_mean_equals_the_scalar_metric(self):
        """The contract that makes the two auditable together: average the trace
        over the calibration window and you must get absorbed_fraction back. A
        units or scale slip in either one breaks this."""
        t = np.linspace(0.0, 9.0e-4, 60)
        dep = np.full_like(t, 3.7)                  # arbitrary field units
        s = mm.absorbed_fraction_series(t, dep, self.f, power_w=200.0)
        want = mm.absorbed_fraction(self.f, 200.0)
        inside = (t >= 2e-4) & (t <= 8e-4)
        self.assertAlmostEqual(float(np.mean(s.retained[inside])), want, places=9)

    def test_shape_follows_the_deposition_field(self):
        """Level comes from the audit, shape from the field. Doubling the field
        at one instant must double the trace there."""
        t = np.linspace(0.0, 9.0e-4, 60)
        dep = np.full_like(t, 2.0); dep[30] = 4.0
        s = mm.absorbed_fraction_series(t, dep, self.f, power_w=200.0)
        self.assertAlmostEqual(s.retained[30] / s.retained[0], 2.0, places=9)

    def test_missing_audit_gives_all_nan_without_raising(self):
        t = np.linspace(0.0, 9.0e-4, 10)
        s = mm.absorbed_fraction_series(t, np.ones_like(t),
                                        self.dir / "hd3out.absent", power_w=200.0)
        self.assertTrue(np.isnan(s.retained).all())
        self.assertTrue(np.isnan(s.scale))

    def test_length_mismatch_is_an_error_not_a_broadcast(self):
        """Silently broadcasting a wrong-length field would put a plausible but
        wrong trace in a figure."""
        with self.assertRaises(ValueError):
            mm.absorbed_fraction_series(np.zeros(10), np.zeros(9), self.f,
                                        power_w=200.0)


if __name__ == "__main__":
    unittest.main()
