"""Validator prediction dumps are off by default (disk blowup guard).

Upstream `Validator.save_results` writes an npz of invar+true+pred every
`rec_validation_freq`. Those dumps reached 19 GB per run and 229 GB across
outputs/ before being deleted on 2026-07-21 -- they are debug artifacts, while
the metrics that matter (l2_relative_error/mae/rmse) go to tensorboard/wandb
regardless. So the dump is opt-in via `custom.validator_dumps`.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lpno.models.pipeline.model_domain import (
    StreamingGridValidator,
    _validator_dumps_enabled,
)


def _fake_validator(dump_files: bool) -> StreamingGridValidator:
    """A StreamingGridValidator with just the attributes save_results touches."""
    v = object.__new__(StreamingGridValidator)
    invar = {"x": torch.ones(1, 1, 2, 2, 2)}
    true_outvar = {"T": torch.ones(1, 1, 2, 2, 2)}
    lw = {"T": torch.ones(1, 1, 2, 2, 2)}
    v.dataloader = [(invar, true_outvar, lw)]
    v.device = torch.device("cpu")
    v.requires_grad = False
    v.plotter = None
    v.dump_files = dump_files
    v.forward = lambda inv: {"T": torch.zeros(1, 1, 2, 2, 2)}
    return v


class _Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def test_dumps_skipped_by_default(tmp_path) -> None:
    v = _fake_validator(dump_files=False)

    losses = v.save_results("val", f"{tmp_path}/", _Writer(), "np,vtk", 0)

    assert not (tmp_path / "val.npz").exists(), "validator wrote an npz dump while dumping is off"
    assert (tmp_path / "validation_metrics.csv").is_file(), "validator metrics must still be persisted"
    # Metrics must still be produced -- turning off dumps must not blind the run.
    assert "l2_relative_error_T" in losses
    assert "mae_T" in losses and "rmse_T" in losses


def test_dumps_written_when_opted_in(tmp_path) -> None:
    v = _fake_validator(dump_files=True)

    v.save_results("val", f"{tmp_path}/", _Writer(), "np", 0)

    assert (tmp_path / "val.npz").is_file()


@pytest.mark.parametrize(
    "cfg_dict,expected",
    [
        ({}, False),
        ({"custom": {"validator_dumps": False}}, False),
        ({"custom": {"validator_dumps": True}}, True),
    ],
)
def test_validator_dumps_enabled_defaults_off(cfg_dict, expected) -> None:
    assert _validator_dumps_enabled(OmegaConf.create(cfg_dict)) is expected
