"""Regression: fast-path fields sat at cell centers while x/y/z used linspace.

_detect_uniform_grid returns the ROI's true cell-center axes and the crop-only
path samples the fields there, but parse_directory threw those axes away and
always built the coordinate channels from linspace(min, max, n). Cell centers
are offset ~half a cell from the domain box, so the field grid and the x/y/z
channels the PDE loss differentiates against described different points.

The interpolation path is self-consistent by construction (it resamples *onto*
the linspace grid), so only the crop-only path needed correcting.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista")

from lpno.custom_utils.openFoam_datasource import VTKOpenFOAMDataSource


def _uniform_info(nx: int, ny: int, nz: int):
    """Cell centers of a 1 m box divided into nx/ny/nz cells."""
    return {
        "gx": ((np.arange(nx) + 0.5) / nx).astype(np.float32),
        "gy": ((np.arange(ny) + 0.5) / ny).astype(np.float32),
        "gz": ((np.arange(nz) + 0.5) / nz).astype(np.float32),
    }


def _templates(nx: int, ny: int, nz: int):
    shape = (1, 1, nx, ny, nz)
    return (
        np.broadcast_to(np.linspace(0, 1, nx, dtype=np.float32).reshape(1, 1, nx, 1, 1), shape),
        np.broadcast_to(np.linspace(0, 1, ny, dtype=np.float32).reshape(1, 1, 1, ny, 1), shape),
        np.broadcast_to(np.linspace(0, 1, nz, dtype=np.float32).reshape(1, 1, 1, 1, nz), shape),
    )


def test_channels_follow_the_true_cell_centers():
    nx, ny, nz = 4, 3, 2
    xB, yB, zB = VTKOpenFOAMDataSource._coordinate_channels(
        _uniform_info(nx, ny, nz), _templates(nx, ny, nz), (nx, ny, nz)
    )
    assert np.allclose(xB[0, 0, :, 0, 0], [0.125, 0.375, 0.625, 0.875])
    assert np.allclose(yB[0, 0, 0, :, 0], [1 / 6, 0.5, 5 / 6])
    assert np.allclose(zB[0, 0, 0, 0, :], [0.25, 0.75])
    # ...and not the box edges linspace would have put there.
    assert not np.isclose(xB[0, 0, 0, 0, 0], 0.0)


def test_the_interpolation_path_keeps_its_linspace_axes():
    nx, ny, nz = 4, 3, 2
    templates = _templates(nx, ny, nz)
    out = VTKOpenFOAMDataSource._coordinate_channels(None, templates, (nx, ny, nz))
    for channel, template in zip(out, templates):
        assert channel is template


def test_a_mismatched_axis_length_falls_back_to_the_template():
    # Defensive: the crop path itself falls back to interpolation when the cell
    # count disagrees, and the channels must fall back with it.
    nx, ny, nz = 4, 3, 2
    templates = _templates(nx, ny, nz)
    info = _uniform_info(nx + 1, ny, nz)
    out = VTKOpenFOAMDataSource._coordinate_channels(info, templates, (nx, ny, nz))
    for channel, template in zip(out, templates):
        assert channel is template


def test_parse_directory_no_longer_pins_the_channels_to_the_template():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "lpno" /
           "custom_utils" / "openFoam_datasource.py").read_text(encoding="utf-8")
    assert "xB = xB_template" not in src, (
        "the coordinate channels must go through _coordinate_channels"
    )
    assert src.count("_coordinate_channels(") >= 3  # definition + both loops
