"""Parallel decompression for the lazy NPZ path.

`SupervisedGridConstraint` defaults to `num_workers=0` and lpno never
set it, so every lazy training decompressed one case per step on the main process
while the GPU sat idle. Measured on dataset 2579: 13.56 s to decompress a case,
against ~1.65 s of compute -- the loader was the entire step time.

Workers are the fix that keeps global shuffling (unlike a case-grouped sampler),
so a run stays protocol-comparable with the `rigno_ti64ds3` baseline.

The knob must stay bounded: workers each hold their own LRU, and this box also
runs FLOW-3D and CI, so an unbounded count would starve interactive work.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from lpno.models.pipeline.npz_lazy_store import dataloader_workers


def _cfg(**kw):
    return SimpleNamespace(**kw)


def test_defaults_to_serial_loading(monkeypatch):
    """Unset means the old behaviour: no worker processes."""
    monkeypatch.delenv("PINO_LOADER_WORKERS", raising=False)
    assert dataloader_workers(_cfg()) == 0


def test_config_value_is_used(monkeypatch):
    monkeypatch.delenv("PINO_LOADER_WORKERS", raising=False)
    assert dataloader_workers(_cfg(loader_workers=6)) == 6


def test_env_overrides_config(monkeypatch):
    """aiida-pino passes this through the training profile's env mapping."""
    monkeypatch.setenv("PINO_LOADER_WORKERS", "8")
    assert dataloader_workers(_cfg(loader_workers=2)) == 8


def test_worker_count_is_clamped_to_leave_cores_free(monkeypatch):
    """FLOW-3D and CI share this box; never ask for every core."""
    monkeypatch.delenv("PINO_LOADER_WORKERS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 20)
    assert dataloader_workers(_cfg(loader_workers=1000)) == 16  # 20 - 4 reserved


def test_negative_and_garbage_fall_back_to_serial(monkeypatch):
    monkeypatch.setenv("PINO_LOADER_WORKERS", "not-a-number")
    assert dataloader_workers(_cfg()) == 0
    monkeypatch.setenv("PINO_LOADER_WORKERS", "-3")
    assert dataloader_workers(_cfg()) == 0


@pytest.mark.parametrize("value,expected", [("0", 0), ("1", 1), ("4", 4)])
def test_env_values_parse(monkeypatch, value, expected):
    monkeypatch.setenv("PINO_LOADER_WORKERS", value)
    assert dataloader_workers(_cfg()) == expected
