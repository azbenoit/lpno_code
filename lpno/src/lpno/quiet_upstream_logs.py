"""Filters for upstream log messages that are normal here and fire every step.

physicsnemo-sym is a read-only upstream checkout, so noise it emits has to be
filtered on our side rather than fixed at the source.
"""

from __future__ import annotations

import logging

#: The exact upstream text, matched as a substring. Pinned by
#: `tests/test_quiet_deriv_scaler_warning.py` against the real
#: `physicsnemo/sym/amp.py`, so a reword upstream fails the test rather than
#: silently un-muting the flood.
DERIV_SCALER_SUBSTRING = "There are no scalers created in DerivScalers"

_DERIV_SCALER_LOGGER = "physicsnemo.sym.amp"
_FILTER_ATTR = "_pino_deriv_scaler_filter"


class _DropDerivScalerWarning(logging.Filter):
    """Drop ONLY the empty-DerivScalers warning; keep every other amp record.

    The same logger carries real amp problems (inf/NaN in a scaler), which must
    still reach the log -- muting the whole logger would hide them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return DERIV_SCALER_SUBSTRING not in record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not kill training
            return True


def quiet_deriv_scaler_warning() -> None:
    """Silence physicsnemo's per-step empty-DerivScalers warning. Idempotent.

    `DerivScalers.found_inf()` (`physicsnemo/sym/amp.py:577`) warns whenever the
    scaler set is empty, and it is called once per optimizer step. For a data-only
    model (no PDE/derivative nodes) with `amp.enabled: true` that is every step of
    every run: measured on run 69912, **12707 of 13180 lines, 96.4 % of the log**
    at step 8700 of 32000. The condition is expected for these arms, so the
    warning carries no information and buries the `[step:` lines and any traceback.
    """
    logger = logging.getLogger(_DERIV_SCALER_LOGGER)
    if any(getattr(f, _FILTER_ATTR, False) for f in logger.filters):
        return
    handle = _DropDerivScalerWarning()
    setattr(handle, _FILTER_ATTR, True)
    logger.addFilter(handle)
