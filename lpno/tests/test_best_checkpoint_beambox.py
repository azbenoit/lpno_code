"""The keeper ranks checkpoints by beambox_mae_T when the validator provides it.

Regression for 2026-08-13: domain mae_T kept a cold checkpoint (step 26000,
tail_deficit 0.368) over a healthy-tail one (step 22000, tail_deficit 0.019) on
gaot_pvlhs_base_32k, because the ambient far field dominates the domain average.
"""
from lpno.models.pipeline.best_checkpoint import _METRIC_PREFERENCE, _pick_metric


def test_beambox_is_the_first_preference():
    assert _METRIC_PREFERENCE[0] == "beambox_mae_T"


def test_beambox_wins_when_both_present():
    key, value = _pick_metric({"mae_T": 0.0345, "beambox_mae_T": 0.0422})
    assert key == "beambox_mae_T" and value == 0.0422


def test_falls_back_to_mae_when_beambox_absent():
    key, _ = _pick_metric({"mae_T": 0.05, "rmse_T": 0.08})
    assert key == "mae_T"


def test_nan_beambox_does_not_shadow_a_finite_fallback():
    key, value = _pick_metric({"beambox_mae_T": float("nan"), "mae_T": 0.05})
    assert key == "mae_T" and value == 0.05
