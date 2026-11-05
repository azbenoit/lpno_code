"""Per-frame moving-reference window in the lazy NPZ store.

The steady-state datasets crop to a laser-tracking window (`_apply_moving_frame`),
but that path assumes one grid cell of beam travel per frame and an eager load.
This is the lazy, per-frame version for time_as_batch: each emitted frame is an
x-window positioned around the beam's CURRENT cell (from the cached manifest
centres), clamped inside the domain so every sample has one constant shape --
what batching needs. The x coordinate is emitted beam-relative so every window
spans the same coordinate range; beam_x stays absolute so the model still knows
where the window sits (soak boundaries are absolute).

Contract: a windowed read equals the plain read sliced at [ix0, ix0+w), where
ix0 = clip(round((cx - x0)/dx) - behind, 0, nx - w).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lpno.custom_utils import npz_datasource as nds
from lpno.models.pipeline.npz_lazy_store import (
    NpzDirStore,
    maybe_build_npz_store,
)

from test_npz_lazy_store import _manifest, _write_cases


def _store(tmp_path, **kw):
    return NpzDirStore(
        tmp_path,
        requested_fields=["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"],
        path_channels=list(nds.PATH_CHANNEL_NAMES),
        material_channels=list(nds.MATERIAL_CHANNEL_NAMES),
        **kw,
    )


def _expected_ix0(tmp_path, frame, behind, ahead, nx=10, nt=7):
    x = np.linspace(-3e-4, 3e-4, nx)
    t = np.linspace(0.0, 6e-4, nt)
    centres = nds._beam_centres_over_time(_manifest(), t)
    ix = int(np.clip(np.round((centres[frame, 0] - x[0]) / (x[1] - x[0])), 0, nx - 1))
    return int(np.clip(ix - behind, 0, nx - (behind + ahead))), float(centres[frame, 0])


def test_windowed_read_equals_plain_slice(tmp_path):
    nx, ny, nz, nt = _write_cases(tmp_path)
    plain = _store(tmp_path)
    win = _store(tmp_path, moving_window=(4, 2))
    f = nt - 1  # beam well away from the start
    ix0, cx = _expected_ix0(tmp_path, f, 4, 2)
    a = plain.read_case(0, time_idx=f)
    b = win.read_case(0, time_idx=f)
    for key in ("Temperature", "Pressure", "beam_dist", "path_dose", "y", "z"):
        assert b[key].shape[-3] == 6, key
        assert np.array_equal(b[key], a[key][..., ix0:ix0 + 6, :, :]), key

def test_x_is_beam_relative_and_beam_x_absolute(tmp_path):
    nx, ny, nz, nt = _write_cases(tmp_path)
    plain = _store(tmp_path)
    win = _store(tmp_path, moving_window=(4, 2))
    f = nt - 1
    ix0, cx = _expected_ix0(tmp_path, f, 4, 2)
    a = plain.read_case(0, time_idx=f)
    b = win.read_case(0, time_idx=f)
    assert np.allclose(b["x"], a["x"][..., ix0:ix0 + 6, :, :] - np.float32(cx), atol=1e-9)
    assert np.allclose(b["beam_x"], a["beam_x"][..., ix0:ix0 + 6, :, :])

def test_window_clamped_at_domain_start(tmp_path):
    """Frame 0: beam at the path start; window must clamp to [0, w)."""
    nx, ny, nz, nt = _write_cases(tmp_path)
    win = _store(tmp_path, moving_window=(4, 2))
    plain = _store(tmp_path)
    ix0, _cx = _expected_ix0(tmp_path, 0, 4, 2)
    b = win.read_case(0, time_idx=0)
    a = plain.read_case(0, time_idx=0)
    assert np.array_equal(b["Temperature"], a["Temperature"][..., ix0:ix0 + 6, :, :])

def test_spatial_shape_reflects_window(tmp_path):
    nx, ny, nz, _ = _write_cases(tmp_path)
    win = _store(tmp_path, moving_window=(4, 2))
    assert win.spatial_shape() == (6, ny, nz)

def test_full_case_read_refused_under_window(tmp_path):
    _write_cases(tmp_path)
    win = _store(tmp_path, moving_window=(4, 2))
    with pytest.raises(ValueError, match="per-frame"):
        win.read_case(0, time_idx=None)

def test_window_plus_stride_refused(tmp_path):
    _write_cases(tmp_path)
    with pytest.raises(ValueError, match="stride"):
        _store(tmp_path, moving_window=(4, 2), spatial_stride=(2, 1, 1))

def test_window_wider_than_grid_refused(tmp_path):
    _write_cases(tmp_path, nx=10)
    with pytest.raises(ValueError, match="window"):
        _store(tmp_path, moving_window=(8, 4))

def test_env_parsing(tmp_path, monkeypatch):
    _write_cases(tmp_path)
    cfg = SimpleNamespace(
        data_directory_path=str(tmp_path),
        vtk_fields=["Temperature", "Pressure"],
        keys=SimpleNamespace(str_fno_input_keys=["V_scan", "P", "x", "y", "z"]),
        moving_ref_frame=False, time_as_batch=True, time_as_dim=False,
        include_time=True, lazy_npz=True,
    )
    monkeypatch.setenv("PINO_LAZY_MOVING_REF", "4,2")
    s = maybe_build_npz_store(tmp_path, cfg)
    assert s is not None and s.spatial_shape()[0] == 6
    monkeypatch.setenv("PINO_LAZY_MOVING_REF", "banana")
    with pytest.raises(ValueError, match="PINO_LAZY_MOVING_REF"):
        maybe_build_npz_store(tmp_path, cfg)
    monkeypatch.delenv("PINO_LAZY_MOVING_REF")
    assert maybe_build_npz_store(tmp_path, cfg).spatial_shape()[0] == 10  # noop default
