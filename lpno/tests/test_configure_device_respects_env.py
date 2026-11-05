"""Regression: configure_device clobbered a CUDA_VISIBLE_DEVICES set by the caller.

cuda_bootstrap deliberately preserves an externally-set CUDA_VISIBLE_DEVICES --
a Slurm allocation, or gpu_spool pinning the seat it just leased -- but
configure_device then saw `cuda_visible_devices: null` in the config and set
CUDA_VISIBLE_DEVICES=str(cuda_device) unconditionally. Every shipped config
carries `cuda_device: 0`, so a run seated on GPU 1 was re-pointed at GPU 0
before CUDA initialized.

The env now wins unless the config names cuda_visible_devices explicitly, which
matches the precedence cuda_bootstrap already implements (CLI > env > config).
"""
from __future__ import annotations

import os

import pytest
from omegaconf import OmegaConf

from lpno.models.pipeline.config_loading import configure_device


def _cfg(cuda_device=0, cuda_visible_devices=None):
    return OmegaConf.create(
        {"custom": {"cuda_device": cuda_device, "cuda_visible_devices": cuda_visible_devices}}
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_a_pinned_seat_is_not_repointed(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    configure_device(_cfg(cuda_device=0), set_device=False)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_a_multi_gpu_allocation_is_preserved(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    configure_device(_cfg(cuda_device=0), set_device=False)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_an_empty_env_value_is_not_treated_as_a_pin(monkeypatch):
    # "" means no visible GPUs, not "the caller chose"; it is what an unset
    # variable often round-trips to, so fall through to the config.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    configure_device(_cfg(cuda_device=1), set_device=False)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_without_an_env_pin_the_config_device_is_used():
    configure_device(_cfg(cuda_device=1), set_device=False)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_an_explicit_config_value_still_overrides_the_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    configure_device(_cfg(cuda_device=0, cuda_visible_devices="0"), set_device=False)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_no_cuda_is_refused_unless_the_cpu_is_allowed(monkeypatch, capsys):
    # 2026-10-08: re-scoring a deposited model on a CPU-only cluster died here by design;
    # PINO_ALLOW_CPU=1 is the reader's explicit opt-in, and the default still refuses.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("PINO_ALLOW_CPU", raising=False)
    with pytest.raises(SystemExit, match="PINO_ALLOW_CPU"):
        configure_device(_cfg(cuda_device=0), set_device=True)
    monkeypatch.setenv("PINO_ALLOW_CPU", "1")
    configure_device(_cfg(cuda_device=0), set_device=True)
    assert "running on the CPU" in capsys.readouterr().out
