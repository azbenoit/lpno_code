"""The selection metric must reach the LOG every validation, not only on improvement.

Run 82438 (`gaot_crossds_ph_w2_90k`, on the remote compute box) showed no beam-box number in the console at
step 29800 while `validation_metrics.csv` held `beambox_mae_T` at every step from 2000 on.
The metric was fine; the LOG was not. `best checkpoint: ... at step N` prints only when the
value improves, 82438 last improved at step 10000, and the console scrapes that line shape
from a 16 KB tail -- by then it was 34969 bytes back in a 61274-byte log.

For a remote job the log tail is the only LIVE channel (the CSV stays on the compute box
until retrieval), so a metric absent from the log is invisible however well it is computed.
"""
import logging

import torch
from omegaconf import OmegaConf
from physicsnemo.sym.domain.validator import PointwiseValidator

from lpno.models.pipeline.best_checkpoint import best_val_checkpoint


class _FakeValidator(PointwiseValidator):
    """Own `save_results` so the keeper's subclass walk patches it."""

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        return self._losses


def _run(tmp_path, sequence, caplog):
    v = object.__new__(_FakeValidator)
    model = torch.nn.Linear(1, 1)
    caplog.set_level(logging.INFO, logger="lpno.models.pipeline.best_checkpoint")
    with best_val_checkpoint(OmegaConf.create({}), model, tmp_path):
        for step, value in sequence:
            v._losses = {"beambox_mae_T": value, "mae_T": 0.5}
            _FakeValidator.save_results(v, "data", tmp_path, None, None, step)
    return caplog.text


def test_plateau_still_logs_the_current_value(tmp_path, caplog):
    # Improves once, then plateaus -- 82438's shape.
    text = _run(tmp_path, [(2000, 0.30), (4000, 0.20), (6000, 0.25), (8000, 0.26)], caplog)
    for step in (2000, 4000, 6000, 8000):
        assert f"at step {step}" in text, f"step {step} left no beam-box line in the log"
    # The console's regex must find a reading from the LAST validation, not a stale best.
    assert "beambox_mae_T=0.26 at step 8000" in text


def test_improvement_still_records_a_best_checkpoint(tmp_path, caplog):
    """The per-validation line must not displace the keeper's own behaviour."""
    text = _run(tmp_path, [(2000, 0.30), (4000, 0.20)], caplog)
    assert "best checkpoint: beambox_mae_T=0.2 at step 4000" in text
    assert (tmp_path / "u_network.best.pth").exists()
