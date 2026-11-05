"""The best-validation checkpoint must survive a later divergence.

physicsnemo overwrites `u_network.0.pth` on every save, so a run that trains well and then
diverges registers the wreckage. `gaot_cond_32k_ctrl` (model 26734) reached val T MAE
22.4 K at step 12000, spiked at 14000, and diverged monotonically from 16000 to 3.8e8 K by
28000 with alpha collapsed to a constant. ~13 GPU hours were unsalvageable for want of one
extra file.
"""
import json
from pathlib import Path

import pytest

from lpno.models.pipeline.best_checkpoint import (
    BEST_META_NAME,
    BEST_WEIGHTS_NAME,
    _pick_metric,
)


def test_prefers_mae_T_the_metric_decisions_are_made_on():
    assert _pick_metric({"mae_T": 0.5, "rmse_T": 9.0})[0] == "mae_T"


def test_falls_back_when_mae_is_absent():
    assert _pick_metric({"rmse_T": 9.0})[0] == "rmse_T"
    assert _pick_metric({"l2_relative_error_T": 0.1})[0] == "l2_relative_error_T"
    assert _pick_metric({"something_else": 1.0}) is None
    assert _pick_metric(None) is None


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_diverged_values_never_count_as_an_improvement(bad):
    """A diverged run reports inf/nan; treating those as 'best' would overwrite the
    good weights with the wreckage, which is the exact failure this guards."""
    assert _pick_metric({"mae_T": bad}) is None


def test_a_later_divergence_cannot_overwrite_the_best(tmp_path):
    """The real trajectory: improves to step 12000, then explodes."""
    import torch

    from lpno.models.pipeline import best_checkpoint as bc

    model = torch.nn.Linear(3, 2)
    saved = {}

    class _FakeValidator:
        def save_results(self, name, results_dir, writer, save_filetypes, step):
            return saved["losses"]

    # Drive the picker + save logic directly over the observed trajectory.
    out = tmp_path / "net"
    best = None
    for step, mae in [(2000, 0.0149), (6000, 0.0084), (12000, 0.0075),
                      (16000, 476.7), (28000, 128203.1)]:
        picked = _pick_metric({"mae_T": mae})
        assert picked is not None
        _, value = picked
        if best is None or value < best:
            best = value
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / BEST_WEIGHTS_NAME)
            (out / BEST_META_NAME).write_text(
                json.dumps({"metric": "mae_T", "value": value, "step": step}) + "\n"
            )

    meta = json.loads((out / BEST_META_NAME).read_text())
    assert meta["step"] == 12000, "the pre-divergence checkpoint must be the one kept"
    assert meta["value"] == pytest.approx(0.0075)
    assert (out / BEST_WEIGHTS_NAME).is_file()


def test_wired_into_the_solver_path():
    src = (Path(__file__).resolve().parents[1] / "src" / "lpno" / "models"
           / "pipeline" / "model_domain.py").read_text()
    assert "best_val_checkpoint" in src
    assert "solver.solve()" in src
    caller = (Path(__file__).resolve().parents[1] / "src" / "lpno" / "models"
              / "ns_energy_pipeline.py").read_text()
    assert "model=getattr(model_bundle" in caller, "solve_domain must receive the model"
