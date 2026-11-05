"""An alpha head that never predicts gas must be visible and must stop the run.

Seed-43 (model 70965) predicted `alpha >= 0.5` in every one of 4.86M cells across a
45-frame test dump — catching 0 of 18489 true-gas cells — and still scored
`mae_alpha = 0.00421`, BETTER than a healthy baseline's 0.00914, because that metric is a
domain mean and the gas is 0.4 % of cells. The prediction-side boiling fill is gated on the
model's own alpha (`postprocessing.py:1335`), so the fill never fired: its gas T sat at
~1866 K against the correct 3123 K, 33 % of the model's total error and a 4x beam-box gap
(214.0 K vs 54.1 K) against a twin whose substrate MAE was within 11 %.
"""
import pytest

torch = pytest.importorskip("torch")

from lpno.models.pipeline.validation_beambox import gas_alpha_validation_metrics
from lpno.models.pipeline.alpha_guard import AlphaCollapseGuard


def _vars(true_alpha, pred_alpha):
    return ({"alpha": torch.tensor(true_alpha, dtype=torch.float32)},
            {"alpha": torch.tensor(pred_alpha, dtype=torch.float32)})


def test_reports_counts_and_recall_when_the_head_is_healthy():
    t, p = _vars([0.0, 0.1, 0.9, 1.0], [0.05, 0.2, 0.8, 1.0])
    m = gas_alpha_validation_metrics(t, p)
    assert float(m["gas_true_n"]) == 2
    assert float(m["gas_pred_n"]) == 2
    assert float(m["gas_recall_alpha"]) == pytest.approx(1.0)


def test_collapsed_head_reports_zero_recall_even_though_mae_is_small():
    """The seed-43 signature: every prediction above the threshold, tiny MAE."""
    t, p = _vars([0.0, 0.0, 1.0, 1.0], [0.66, 0.66, 1.0, 1.0])
    m = gas_alpha_validation_metrics(t, p)
    assert float(m["gas_true_n"]) == 2
    assert float(m["gas_pred_n"]) == 0
    assert float(m["gas_recall_alpha"]) == 0.0
    mae = float(torch.mean(torch.abs(t["alpha"] - p["alpha"])))
    assert mae < 0.35, "MAE stays small -- which is exactly why mae_alpha misses this"


def test_no_gas_in_truth_yields_no_metric_rather_than_a_false_alarm():
    t, p = _vars([1.0, 1.0], [1.0, 1.0])
    assert gas_alpha_validation_metrics(t, p) == {}


def test_guard_does_not_fire_before_the_warmup_step():
    g = AlphaCollapseGuard(warmup_steps=5000, patience=2)
    for step in (500, 1000, 2000, 4500):
        assert g.update(step, recall=0.0) is None


def test_guard_fires_only_after_repeated_collapse_past_warmup():
    g = AlphaCollapseGuard(warmup_steps=1000, patience=3)
    assert g.update(2000, recall=0.0) is None
    assert g.update(2500, recall=0.0) is None
    msg = g.update(3000, recall=0.0)
    assert msg is not None and "alpha" in msg.lower()
    assert "3000" in msg


def test_a_single_healthy_validation_resets_the_counter():
    g = AlphaCollapseGuard(warmup_steps=1000, patience=2)
    assert g.update(2000, recall=0.0) is None
    assert g.update(2500, recall=0.31) is None      # recovered
    assert g.update(3000, recall=0.0) is None       # counter restarted
    assert g.update(3500, recall=0.0) is not None


def test_guard_ignores_missing_recall():
    """A validation that could not compute the metric must not count as a collapse."""
    g = AlphaCollapseGuard(warmup_steps=0, patience=2)
    assert g.update(100, recall=None) is None
    assert g.update(200, recall=None) is None
    assert g.update(300, recall=None) is None


# --- wiring: a guard that is never reached is worse than no guard -------------------
# An earlier orphan-audit fix referenced a name that was only imported inside
# main(), so it raised NameError into its own `except` and silently passed everything.
# These tests assert the guard is actually reached from the validation callback.

def test_validation_callback_raises_once_the_guard_trips(monkeypatch):
    from lpno.models.pipeline import model_domain as md

    monkeypatch.setattr(md, "_ALPHA_GUARD", None, raising=False)
    monkeypatch.setenv("PINO_ALPHA_GUARD_WARMUP", "0")
    monkeypatch.setenv("PINO_ALPHA_GUARD_PATIENCE", "2")
    monkeypatch.delenv("PINO_ALPHA_GUARD", raising=False)

    losses = {"gas_recall_alpha": torch.tensor(0.0), "mae_T": torch.tensor(0.01)}
    assert md._check_alpha_collapse(losses, 1000) is None
    with pytest.raises(md.AlphaCollapseError) as exc:
        md._check_alpha_collapse(losses, 1500)
    assert "alpha head collapsed" in str(exc.value)


def test_env_var_disables_the_guard(monkeypatch):
    from lpno.models.pipeline import model_domain as md

    monkeypatch.setattr(md, "_ALPHA_GUARD", None, raising=False)
    monkeypatch.setenv("PINO_ALPHA_GUARD", "0")
    losses = {"gas_recall_alpha": torch.tensor(0.0)}
    for step in (10_000, 20_000, 30_000, 40_000):
        assert md._check_alpha_collapse(losses, step) is None


def test_healthy_recall_never_raises(monkeypatch):
    from lpno.models.pipeline import model_domain as md

    monkeypatch.setattr(md, "_ALPHA_GUARD", None, raising=False)
    monkeypatch.setenv("PINO_ALPHA_GUARD_WARMUP", "0")
    monkeypatch.setenv("PINO_ALPHA_GUARD_PATIENCE", "2")
    monkeypatch.delenv("PINO_ALPHA_GUARD", raising=False)
    losses = {"gas_recall_alpha": torch.tensor(0.93)}
    for step in range(1000, 9000, 1000):
        assert md._check_alpha_collapse(losses, step) is None


def test_the_metric_is_wired_into_the_validator_hook():
    """The metric must be merged where beam-box metrics are, or it never reaches the CSV."""
    import inspect
    from lpno.models.pipeline import validation_beambox as vb

    src = inspect.getsource(vb)
    body = src[src.index("def _wrap_l2"):]
    body = body[:body.index("\n    for cls in")]
    assert "gas_alpha_validation_metrics(true_var, pred_var)" in body, \
        "gas-alpha metric is defined but not merged into the validator output"


def test_end_to_end_through_the_real_validator_and_csv(tmp_path, monkeypatch):
    """The plumbing, not just the pieces.

    Unit tests proved the metric and the guard in isolation and an AST check proved the call
    site exists — none of that proves physicsnemo's validator actually delivers the scalar
    into `validation_metrics.csv`. This drives the REAL `GridValidator._l2_relative_error`
    under the patch, writes with the REAL csv appender, and feeds the result to the guard.
    """
    pytest.importorskip("physicsnemo.sym.domain.validator")
    from physicsnemo.sym.domain.validator import GridValidator
    from lpno.models.pipeline.validation_beambox import beam_box_validation
    from lpno.models.pipeline import model_domain as md

    true_alpha = torch.ones(1, 1, 8, 8, 8)
    true_alpha[..., :2] = 0.0                       # a genuine gas region
    pred_alpha = torch.full((1, 1, 8, 8, 8), 0.66)  # the seed-43 signature: never below 0.5
    true_var = {"alpha": true_alpha, "T": torch.rand(1, 1, 8, 8, 8)}
    pred_var = {"alpha": pred_alpha, "T": torch.rand(1, 1, 8, 8, 8)}

    with beam_box_validation(enabled=True):
        out = GridValidator._l2_relative_error(true_var, pred_var)

    assert "gas_recall_alpha" in out, "metric never reached the validator output"
    assert float(out["gas_true_n"]) == 128
    assert float(out["gas_pred_n"]) == 0
    assert float(out["gas_recall_alpha"]) == 0.0

    md._append_validation_metrics_csv(str(tmp_path), "val", out, step=5000)
    text = (tmp_path / "validation_metrics.csv").read_text()
    assert "gas_recall_alpha" in text, "metric missing from validation_metrics.csv"

    monkeypatch.setattr(md, "_ALPHA_GUARD", None, raising=False)
    monkeypatch.setenv("PINO_ALPHA_GUARD_WARMUP", "0")
    monkeypatch.setenv("PINO_ALPHA_GUARD_PATIENCE", "2")
    monkeypatch.delenv("PINO_ALPHA_GUARD", raising=False)
    md._check_alpha_collapse(out, 5000)
    with pytest.raises(md.AlphaCollapseError):
        md._check_alpha_collapse(out, 5500)


class TestAllGasCollapse:
    """The guard's original pole (recall 0 = all-metal) has a mirror: all-GAS.

    Measured 2026-08-27 on the pvlhs take-3 arms: mae_alpha ~0.97 against a ~constant-1
    cropped target means the head emits ~0.03 EVERYWHERE, so every cell is called gas,
    recall reads a healthy 1.0, the boiling fill fires across the whole domain, and the
    val beam box sits flat at ~2183 K. The recall-only guard can never fire on this.
    `gas_pred_fraction` (predicted-gas share of ALL cells) is the symmetric signal.
    """

    def _g(self):
        return AlphaCollapseGuard(warmup_steps=4000, patience=3)

    def test_all_gas_mode_aborts_after_patience(self):
        g = self._g()
        assert None is (g.update(5000, 1.0, pred_fraction=0.999))
        assert None is (g.update(5500, 1.0, pred_fraction=0.999))
        reason = g.update(6000, 1.0, pred_fraction=0.999)
        assert reason is not None
        assert "every cell" in reason

    def test_all_gas_during_warmup_does_not_count(self):
        g = self._g()
        for step in (500, 1000, 3999):
            assert None is (g.update(step, 1.0, pred_fraction=1.0))
        assert None is (g.update(4500, 1.0, pred_fraction=1.0))  # strike 1 only

    def test_healthy_fraction_resets_strikes(self):
        g = self._g()
        g.update(5000, 1.0, pred_fraction=0.99)
        g.update(5500, 1.0, pred_fraction=0.2)   # healthy: keyhole-sized gas
        assert None is (g.update(6000, 1.0, pred_fraction=0.99))
        assert None is (g.update(6500, 1.0, pred_fraction=0.99))

    def test_missing_fraction_keeps_old_behaviour(self):
        g = self._g()
        for step, recall in ((5000, 0.0), (5500, 0.0), (6000, 0.0)):
            reason = g.update(step, recall)
        assert reason is not None

    def test_two_poles_do_not_share_strikes(self):
        g = self._g()
        g.update(5000, 0.0, pred_fraction=0.5)   # recall strike 1
        g.update(5500, 1.0, pred_fraction=0.99)  # fraction strike 1, recall resets
        assert None is (g.update(6000, 0.0, pred_fraction=0.5))  # recall strike 1 again
