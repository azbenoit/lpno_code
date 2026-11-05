"""Suppress physicsnemo's per-step DerivScalers warning.

`physicsnemo-sym/physicsnemo/sym/amp.py:577` warns

    "There are no scalers created in DerivScalers, please make sure there are no
     derivative nodes in the model."

from `DerivScalers.found_inf()`, which is called once per optimizer step. For a
DATA-ONLY model (no PDE/derivative nodes) with `amp.enabled: true` the scaler set
is always empty, so the warning fires every step for the whole run.

Measured on run 69912 (2026-08-26): **12707 of 13180 lines, 96.4 % of the log**,
1.6 MB at step 8700 of 32000. It buries every real message -- the `[step:` lines,
the `[grid] serving` line, and any traceback -- and the condition is normal for
these arms, not a fault to be told about 32000 times.

physicsnemo-sym is a read-only upstream checkout, so the fix is a filter on our
side. It must drop ONLY this message: the same logger carries real amp warnings
(inf/NaN in the scaler) that must still reach the log.
"""

import logging
import unittest

from lpno.quiet_upstream_logs import (
    DERIV_SCALER_SUBSTRING,
    quiet_deriv_scaler_warning,
)

LOGGER_NAME = "physicsnemo.sym.amp"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class TestQuietDerivScalerWarning(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(LOGGER_NAME)
        self.capture = _Capture()
        self.logger.addHandler(self.capture)
        self.logger.setLevel(logging.DEBUG)
        self._filters = list(self.logger.filters)

    def tearDown(self):
        self.logger.removeHandler(self.capture)
        self.logger.filters = self._filters

    def test_the_noisy_message_is_dropped(self):
        quiet_deriv_scaler_warning()
        self.logger.warning(
            "There are no scalers created in DerivScalers, "
            "please make sure there are no derivative nodes in the model."
        )
        self.assertEqual([], self.capture.messages)

    def test_other_warnings_from_the_same_logger_survive(self):
        """The filter must not mute real amp problems."""
        quiet_deriv_scaler_warning()
        self.logger.warning("Found inf in gradient scaler, skipping step")
        self.logger.error("scaler state is corrupt")
        self.assertEqual(
            ["Found inf in gradient scaler, skipping step", "scaler state is corrupt"],
            self.capture.messages,
        )

    def test_is_idempotent(self):
        """Called twice it must not stack two filters."""
        quiet_deriv_scaler_warning()
        n_after_first = len(logging.getLogger(LOGGER_NAME).filters)
        quiet_deriv_scaler_warning()
        self.assertEqual(n_after_first, len(logging.getLogger(LOGGER_NAME).filters))

    def test_substring_matches_the_upstream_text(self):
        """Pin the match against the real upstream string, so a reword is caught.

        Reads the installed physicsnemo-sym first, then a source checkout beside the
        repository, and skips only when neither is present.
        """
        import importlib.util
        from pathlib import Path

        candidates = []
        try:
            spec = importlib.util.find_spec("physicsnemo.sym.amp")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            candidates.append(Path(spec.origin))
        candidates.append(
            Path(__file__).resolve().parents[2]
            / "physicsnemo-sym"
            / "physicsnemo"
            / "sym"
            / "amp.py"
        )
        amp = next((p for p in candidates if p.is_file()), None)
        if amp is None:
            self.skipTest("physicsnemo-sym is neither installed nor checked out beside the repository")
        self.assertIn(
            DERIV_SCALER_SUBSTRING,
            amp.read_text(),
            "the upstream warning text changed; update DERIV_SCALER_SUBSTRING",
        )


if __name__ == "__main__":
    unittest.main()
