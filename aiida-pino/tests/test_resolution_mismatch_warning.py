"""Scoring a model at a resolution it did not train at is WARNED, for every architecture.

Measured on three architectures, same weights each, scored at the model's own training stride
and at full resolution (2026-09-05..07). Verified here from each eval's
recorded `kelvin` block rather than recomputed:

    RIGNO          7.767 -> 8.607 K substrate  (x1.11)   IoU 0.926 -> 0.923
    GAOT           5.039 -> 11.379 K           (x2.26)   IoU 0.931 -> 0.806
    FNO            5.361 -> 55.437 K           (x10.3)   IoU 0.935 -> 0.541

It is a SPECTRUM, not a spectral/graph split. GAOT is a graph model and still degrades by a
factor of 2.3, so the interim rule first proposed -- "refuse for spectral, warn for the rest"
-- would have silently passed a GAOT eval that is 2.3x wrong. That is why this warns for every
architecture and refuses for none: a refusal needs a threshold, and the threshold is the
user's to set.

The failure is silent: the eval exits 0 and returns numbers that look real, and the FNO case
INVERTED the ranking of the two arms being compared before anyone noticed.

Caveat carried in the message itself, because it applies to all three rows: native and
full-res scorings cover different cell counts, so each ratio compares two means over different
samplings. That cannot manufacture x10.3, but these are effect sizes rather than precise
multipliers.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.model_registry import resolution_mismatch_warning as warn


class WarningTest(unittest.TestCase):
    def test_a_mismatch_warns_and_names_both_strides(self):
        msg = warn([2, 2, 1], [1, 1, 1])
        self.assertIsNotNone(msg)
        self.assertIn("[2, 2, 1]", msg)
        self.assertIn("[1, 1, 1]", msg)

    def test_a_match_is_silent(self):
        self.assertIsNone(warn([2, 2, 1], [2, 2, 1]))
        self.assertIsNone(warn([1, 1, 1], [1, 1, 1]))

    def test_an_unknown_on_either_side_is_silent(self):
        """Unknown must not read as a mismatch: most historical models carry no stride."""
        self.assertIsNone(warn(None, [1, 1, 1]))
        self.assertIsNone(warn([2, 2, 1], None))
        self.assertIsNone(warn(None, None))

    def test_it_carries_the_measured_evidence(self):
        """A warning nobody can size is a warning nobody acts on."""
        msg = warn([2, 2, 1], [1, 1, 1])
        self.assertIn("10.3", msg)     # the FNO case
        self.assertIn("2.3", msg)      # GAOT, the one a spectral-only rule would miss
        self.assertIn("architecture", msg.lower())

    def test_it_says_the_numbers_are_still_produced(self):
        """The danger is that it exits 0; the reader must know nothing will stop them."""
        msg = warn([2, 2, 1], [1, 1, 1])
        self.assertTrue(any(w in msg.lower() for w in ("exits 0", "no error", "silently")))

    def test_a_malformed_stride_is_silent_rather_than_fatal(self):
        for bad in ("2,2,1", [2, 2], 2, [1, 2, "x"]):
            self.assertIsNone(warn(bad, [1, 1, 1]), f"{bad!r} was accepted")
            self.assertIsNone(warn([1, 1, 1], bad), f"{bad!r} was accepted")


class WiringTest(unittest.TestCase):
    def test_the_workchain_warns_before_submitting(self):
        import inspect

        from aiida_pino.workflows import evaluate_model as mod

        src = inspect.getsource(mod.EvaluateModelWorkChain.run_evaluation)
        self.assertIn("resolution_mismatch_warning", src)
        self.assertLess(src.index("resolution_mismatch_warning"), src.index("self.submit("))


if __name__ == "__main__":
    unittest.main()
