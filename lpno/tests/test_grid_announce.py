"""The lazy store must announce the grid it actually serves, once, at first read.

On 2026-08-07/09 the effective training resolution of a run could only be
reconstructed by composing two AiiDA attributes (`derivation_profile.stride` on
the dataset node x `env.PINO_LAZY_STRIDE`); misreading one of them produced a
wrong "the stride was dropped" conclusion and a killed (queued) run. The log is
the ground truth the checkpoint actually saw, so the store prints it where no
attribute archaeology can get it wrong:

    [grid] serving 45x20x40 (raw 90x40x40, stride 2x2x1)

One line, first metadata read only -- per-case repeats would spam variable-length
datasets.
"""

from __future__ import annotations

import numpy as np
import pytest


def _write_case(path, nx=8, ny=6, nz=4, nt=3):
    np.savez(
        path,
        x=np.arange(nx, dtype=np.float32),
        y=np.arange(ny, dtype=np.float32),
        z=np.arange(nz, dtype=np.float32),
        time_s=np.arange(nt, dtype=np.float64),
        power_w=np.float64(100.0),
        scan_speed_m_per_s=np.float64(0.5),
        case_name=np.str_("case"),
        T=np.zeros((nt, nx, ny, nz), dtype=np.float32),
    )


@pytest.fixture()
def case_dir(tmp_path):
    _write_case(tmp_path / "case_000.npz")
    _write_case(tmp_path / "case_001.npz")
    return tmp_path


def _store(case_dir, stride):
    from lpno.models.pipeline.npz_lazy_store import NpzDirStore

    return NpzDirStore(str(case_dir), requested_fields=["T"], spatial_stride=stride)


def test_announces_raw_grid_stride_and_served_grid(case_dir, capsys):
    store = _store(case_dir, (2, 2, 1))
    store.spatial_shape()
    out = capsys.readouterr().out
    assert "[grid] serving 4x3x4 (raw 8x6x4, stride 2x2x1)" in out


def test_announces_full_resolution_without_a_stride(case_dir, capsys):
    store = _store(case_dir, (1, 1, 1))
    store.spatial_shape()
    out = capsys.readouterr().out
    assert "[grid] serving 8x6x4 (raw 8x6x4, stride 1x1x1)" in out


def test_announces_exactly_once(case_dir, capsys):
    store = _store(case_dir, (2, 2, 1))
    store.spatial_shape()
    store.time_steps(case_idx=1)  # second case: another _meta read
    store.array_shape("T")
    out = capsys.readouterr().out
    assert out.count("[grid] serving") == 1
