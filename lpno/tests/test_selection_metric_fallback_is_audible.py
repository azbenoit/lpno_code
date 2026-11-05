"""Losing the beam box must not silently change what the checkpoint is selected on.

`_METRIC_PREFERENCE` takes the first AVAILABLE metric, so a run whose validator cannot
produce `beambox_mae_T` selects on whole-domain `mae_T` -- the bulk-diluted metric that
rewards a cold field -- while sibling arms select on the melt region. GAOT AR k=8 (model
86604, run 85613) finished 32k steps that way and read as a comparable arm: its config
inherits `config_flow3D_RIGNO`, whose input keys carry no `beam_dist`, while the arms it
was compared against used `*_randompath` configs that do. The degradation is deliberate;
its inaudibility was the defect (session 067cdc63, 2026-09-02).
"""
import logging

import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.sym.domain.validator import PointwiseValidator

from lpno.models.pipeline import validation_beambox as vb
from lpno.models.pipeline.best_checkpoint import best_val_checkpoint

SHAPE = (1, 1, 6, 5, 4)


class _FakeValidator(PointwiseValidator):
    def save_results(self, name, results_dir, writer, save_filetypes, step):
        return self._losses


def _run(tmp_path, losses, caplog):
    v = object.__new__(_FakeValidator); v._losses = losses
    caplog.set_level(logging.WARNING, logger="lpno.models.pipeline.best_checkpoint")
    with best_val_checkpoint(OmegaConf.create({}), torch.nn.Linear(1, 1), tmp_path):
        for step in (2000, 4000):
            _FakeValidator.save_results(v, "data", tmp_path, None, None, step)
    return caplog.text


def test_fallback_to_mae_is_warned_once(tmp_path, caplog):
    text = _run(tmp_path, {"mae_T": 0.05, "rmse_T": 0.08}, caplog)
    assert text.count("checkpoint selection is using") == 1, "once per run, not per validation"
    assert "'mae_T'" in text and "'beambox_mae_T'" in text


def test_no_warning_when_the_preferred_metric_is_there(tmp_path, caplog):
    text = _run(tmp_path, {"beambox_mae_T": 0.04, "mae_T": 0.05}, caplog)
    assert "checkpoint selection is using" not in text


def test_missing_beam_channels_announce_the_selection_consequence(capsys):
    """The cause, at the point it is detected, naming what it costs."""
    vb._UNAVAILABLE_ANNOUNCED.clear()
    vb._STASH.clear()
    vb._STASH["z"] = [torch.zeros(SHAPE)]          # z only: no beam_dist, no centre
    true = {"T": torch.ones(SHAPE)}; pred = {"T": torch.zeros(SHAPE)}
    assert vb.beam_box_validation_metrics(true, pred) == {}
    assert vb.beam_box_validation_metrics(true, pred) == {}
    out = capsys.readouterr().out
    assert out.count("[beam-box] beambox_* metrics are NOT available") == 1
    assert "CHANGES CHECKPOINT SELECTION" in out and "beam_dist" in out


def test_a_run_that_has_the_box_says_nothing(capsys):
    vb._UNAVAILABLE_ANNOUNCED.clear()
    vb._STASH.clear()
    for k in ("z", "beam_dist"):
        vb._STASH[k] = [torch.zeros(SHAPE)]
    vb.beam_box_validation_metrics({"T": torch.ones(SHAPE)}, {"T": torch.zeros(SHAPE)})
    assert "NOT available" not in capsys.readouterr().out
