"""Validation must surface the tail deficit, because MAE cannot see a cold tail.

~99% of the substrate is quiescent metal, so val MAE describes the easy part.
`gaot_cond_disp_deep_v2` finished only 45% behind on val MAE while running 151 K colder at
the tail (p99.9 1771 K vs truth 1922 K). The melt threshold is 1898 K, so that tail forms
no melt however healthy the peak looks -- and `thermal_headroom_K` read +837 K on the same
model, which is why it is deliberately not surfaced during validation.

This test drives the patched validator wrapper directly, since a bug here would fail every
queued training run at its first validation step.
"""
import torch

from lpno.models.pipeline.model_domain import _augment_validator_metrics


def _wrapped():
    from physicsnemo.sym.domain.validator import GridValidator

    _augment_validator_metrics()
    return GridValidator._l2_relative_error


def test_tail_deficit_is_reported_for_T():
    f = _wrapped()
    true = {"T": torch.tensor([[0.1, 0.2, 0.6407]])}   # 1922 K / 3000
    pred = {"T": torch.tensor([[0.1, 0.2, 0.5903]])}   # 1771 K / 3000
    out = f(true, pred)
    assert "tail_deficit_max_T" in out
    # 1922 - 1771 = 151 K, i.e. 0.0504 in normalized units
    assert abs(float(out["tail_deficit_max_T"]) * 3000 - 151.2) < 1.0


def test_a_prediction_that_matches_the_tail_scores_zero():
    f = _wrapped()
    true = {"T": torch.tensor([[0.1, 0.9]])}
    pred = {"T": torch.tensor([[0.5, 0.9]])}  # bulk wrong, tail right
    out = f(true, pred)
    assert abs(float(out["tail_deficit_max_T"])) < 1e-6


def test_mae_cannot_see_what_tail_deficit_sees():
    """The failure mode in one assertion: same MAE, very different tails."""
    f = _wrapped()
    true = {"T": torch.tensor([[0.5, 0.5, 0.9]])}
    cold = {"T": torch.tensor([[0.5, 0.5, 0.6]])}   # tail 0.3 low
    spread = {"T": torch.tensor([[0.35, 0.35, 0.9]])}  # same total error, tail exact
    o_cold, o_spread = f(true, cold), f(true, spread)
    assert abs(float(o_cold["mae_T"]) - float(o_spread["mae_T"])) < 1e-6, "MAEs must match"
    assert float(o_cold["tail_deficit_max_T"]) > 0.29
    assert abs(float(o_spread["tail_deficit_max_T"])) < 1e-6


def test_only_T_gets_a_tail_deficit():
    """alpha and fl are fractions; a tail deficit on them is meaningless."""
    f = _wrapped()
    v = {"T": torch.tensor([[0.5]]), "alpha": torch.tensor([[0.5]])}
    out = f(v, v)
    assert "tail_deficit_max_T" in out
    assert "tail_deficit_max_alpha" not in out


def test_existing_metrics_are_unchanged():
    """This runs inside every training validation; it must not disturb mae/rmse."""
    f = _wrapped()
    true = {"T": torch.tensor([[0.0, 1.0]])}
    pred = {"T": torch.tensor([[0.0, 0.5]])}
    out = f(true, pred)
    assert abs(float(out["mae_T"]) - 0.25) < 1e-6
    assert abs(float(out["rmse_T"]) - 0.3535534) < 1e-5


def test_patch_is_idempotent():
    """_augment_validator_metrics is called on every domain build; double-wrapping would
    nest the wrapper and skew nothing but is a leak worth pinning."""
    from physicsnemo.sym.domain.validator import GridValidator

    _augment_validator_metrics()
    first = GridValidator._l2_relative_error
    _augment_validator_metrics()
    assert GridValidator._l2_relative_error is first
