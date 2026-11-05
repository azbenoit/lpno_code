"""Regression tests: the training-time grid validator must stream.

Upstream GridValidator.save_results accumulates every validation batch's
invar/true/pred on host RAM before computing metrics, which -9-kills runs with
large val splits. StreamingGridValidator computes the same metrics
(l2_relative_error, mae, rmse per key) from running sums and keeps only the
first batch for the file dump / plotter.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lpno.models.pipeline.model_domain import StreamingGridValidator


class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step, **kwargs):
        self.scalars[tag] = float(value)


def _make_validator(batches, forward, dump_files=False):
    validator = object.__new__(StreamingGridValidator)
    validator.dump_files = dump_files
    validator.dataloader = batches
    validator.forward = forward
    validator.device = "cpu"
    validator.requires_grad = False
    validator.plotter = None
    validator.dataset = SimpleNamespace(invar_keys=["x"], outvar_keys=["T"])
    return validator


def test_streaming_metrics_match_full_tensor_reference(tmp_path):
    torch.manual_seed(0)
    batches = []
    for _ in range(4):
        invar = {"x": torch.randn(2, 1, 3, 3)}
        true = {"T": torch.randn(2, 1, 3, 3)}
        batches.append((invar, true, None))

    def forward(invar):
        return {"T": invar["x"] * 0.5}

    validator = _make_validator(batches, forward)
    writer = _Writer()
    losses = validator.save_results("val", str(tmp_path) + "/", writer, "np", step=0)

    # Full-tensor reference (what upstream computes after concatenating).
    true_all = torch.cat([b[1]["T"] for b in batches])
    pred_all = torch.cat([forward(b[0])["T"] for b in batches])
    err = true_all - pred_all
    expected_l2 = torch.sqrt(torch.mean(err**2) / torch.var(true_all))
    expected_mae = torch.mean(torch.abs(err))
    expected_rmse = torch.sqrt(torch.mean(err**2))

    assert np.isclose(float(losses["l2_relative_error_T"]), float(expected_l2), rtol=1e-5)
    assert np.isclose(float(losses["mae_T"]), float(expected_mae), rtol=1e-5)
    assert np.isclose(float(losses["rmse_T"]), float(expected_rmse), rtol=1e-5)
    # Scalars reach the writer like upstream.
    assert any("l2_relative_error_T" in tag for tag in writer.scalars)
    csv_text = (tmp_path / "validation_metrics.csv").read_text(encoding="utf-8")
    assert "step,validator,metric,value" in csv_text
    assert "0,val,l2_relative_error_T," in csv_text


def test_streaming_dump_is_bounded_to_first_batch(tmp_path):
    batches = []
    for i in range(5):
        invar = {"x": torch.full((2, 1, 2, 2), float(i))}
        true = {"T": torch.full((2, 1, 2, 2), float(i))}
        batches.append((invar, true, None))

    # Dumping is opt-in since 2026-07-21; this test is about the bound that
    # applies *when* it is on, so it enables it explicitly.
    validator = _make_validator(batches, lambda invar: {"T": invar["x"]}, dump_files=True)
    validator.save_results("val", str(tmp_path) + "/", _Writer(), "np", step=0)

    saved = np.load(str(tmp_path / "val.npz"), allow_pickle=True)
    dumped = saved["arr_0"].item()
    # Only the first batch (2 samples), not all 10, is materialized on disk.
    assert dumped["true_T"].shape[0] == 2
    assert dumped["pred_T"].shape[0] == 2
