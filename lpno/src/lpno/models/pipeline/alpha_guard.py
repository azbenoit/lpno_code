"""Abort a run whose alpha head has stopped predicting the gas region.

Why this is an ABORT and not a `stop_criterion` entry: the standard criterion stops on
*no improvement* with patience, which is the wrong shape here. A collapsed alpha head does
not plateau at a mediocre value -- it sits at exactly zero recall forever while every
temperature metric looks normal, so a plateau rule would either fire on healthy runs whose
recall legitimately levels off, or never fire at all.

Measured motivation (2026-08-27, models 70965 vs 71052, identical config and data):
seed 43 predicted `alpha >= 0.5` in EVERY cell, catching 0 of 18489 true-gas cells, while
scoring `mae_alpha = 0.00421` -- better than a healthy baseline's 0.00914, because that
metric is a domain mean and the gas is 0.4 % of cells. The prediction-side boiling fill is
gated on the model's own alpha, so it never fired and the gas sat ~1257 K low, producing a
4x beam-box gap against its twin while substrate MAE differed by 11 %.
"""

from __future__ import annotations

__all__ = ("AlphaCollapseGuard",)


class AlphaCollapseGuard:
    """Track consecutive validations with zero gas recall.

    `update` returns None while healthy and a human-readable reason once the run should
    stop. A missing recall (metric not computable this validation) is neither healthy nor a
    collapse -- it is skipped, so a validator that could not build the mask cannot kill a
    run on its own.
    """

    #: Predicted-gas share of ALL cells above which the head is in all-gas mode. The
    #: healthy value is the keyhole fraction (well under 10 %); 0.95 sustained means the
    #: boiling fill fires across the whole domain. Measured on the 2026-08-27 pvlhs arms:
    #: mae_alpha ~0.97 against a ~constant-1 cropped target = ~0.03 emitted everywhere,
    #: recall 1.0, val beam box flat at ~2183 K -- invisible to the recall pole.
    ALL_GAS_FRACTION = 0.95

    def __init__(self, warmup_steps: int = 4000, patience: int = 3) -> None:
        self.warmup_steps = int(warmup_steps)
        self.patience = max(1, int(patience))
        self._strikes = 0
        self._gas_strikes = 0

    def update(self, step: int, recall: float | None, pred_fraction: float | None = None) -> str | None:
        if pred_fraction is not None:
            if pred_fraction < self.ALL_GAS_FRACTION:
                self._gas_strikes = 0
            elif int(step) >= self.warmup_steps:
                self._gas_strikes += 1
                if self._gas_strikes >= self.patience:
                    return (
                        f"alpha head collapsed the other way: the model calls "
                        f"{pred_fraction:.1%} of ALL cells gas, sustained for "
                        f"{self._gas_strikes} consecutive validations, most recently at "
                        f"step {int(step)} (warmup {self.warmup_steps}). It predicts "
                        "alpha < 0.5 in every cell, so gas recall reads a healthy 1.0 while "
                        "the boiling-temperature fill fires across the whole domain and the "
                        "beam box sits flat near the fill-vs-truth gap. Stopping rather "
                        "than spending the rest of the budget."
                    )
        if recall is None:
            return None
        if recall > 0.0:
            self._strikes = 0
            return None
        # Early training legitimately predicts no gas; only count strikes past warmup.
        if int(step) < self.warmup_steps:
            return None
        self._strikes += 1
        if self._strikes < self.patience:
            return None
        return (
            f"alpha head collapsed: gas recall has been 0.0 for {self._strikes} consecutive "
            f"validations, most recently at step {int(step)} (warmup {self.warmup_steps}). "
            "The model predicts alpha >= 0.5 in every cell, so the boiling-temperature fill "
            "gated on predicted alpha never fires and gas temperatures stay far below truth. "
            "Temperature metrics will look normal -- mae_alpha is a domain mean and the gas "
            "is well under 1 % of cells. Stopping rather than spending the rest of the budget."
        )
