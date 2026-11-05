"""Regression: FNOGridInferencer built the mask meshgrid in (nz,ny,nx) order.

`mask_fn` is documented and called as mask_fn(x, y, z), and the checks below it
compare against (nx,ny,nz) -- but the meshgrid was built from
arange(nz), arange(ny), arange(nx). On a non-cubic grid both branches raised; on
a cubic one it "passed" with xx and zz transposed relative to the
[B,C,nx,ny,nz] prediction.

The active pipeline passes mask_fn=None, so this path is dead there; the test
exercises the coordinate contract directly.
"""
from __future__ import annotations

import numpy as np
import pytest

NX, NY, NZ = 7, 5, 3


def _meshgrid_as_shipped(nx: int, ny: int, nz: int):
    """The exact construction used in inferencer.py, read from source so the test
    tracks the shipped code rather than a copy of it."""
    from pathlib import Path
    import re

    src = (Path(__file__).resolve().parents[1] / "src" / "lpno" /
           "custom_utils" / "inferencer.py").read_text(encoding="utf-8")
    line = re.search(r"^\s*(\w+, \w+, \w+ = np\.meshgrid\(.*indexing=\"ij\"\))\s*$",
                     src, flags=re.MULTILINE)
    assert line, "could not find the mask meshgrid construction"
    scope: dict = {"np": np, "nx": nx, "ny": ny, "nz": nz}
    exec(line.group(1), scope)  # noqa: S102 - executing our own source line
    return scope["xx"], scope["yy"], scope["zz"]


def test_meshgrid_has_the_prediction_layout():
    xx, yy, zz = _meshgrid_as_shipped(NX, NY, NZ)
    for name, arr in (("xx", xx), ("yy", yy), ("zz", zz)):
        assert arr.shape == (NX, NY, NZ), f"{name} is {arr.shape}, expected {(NX, NY, NZ)}"


def test_each_axis_varies_along_its_own_dimension():
    # The transposition bug: xx must count along axis 0, zz along axis 2.
    xx, yy, zz = _meshgrid_as_shipped(NX, NY, NZ)
    assert np.array_equal(xx[:, 0, 0], np.arange(NX))
    assert np.array_equal(yy[0, :, 0], np.arange(NY))
    assert np.array_equal(zz[0, 0, :], np.arange(NZ))


def test_a_non_cubic_mask_fn_result_matches_the_expected_shape():
    # mask_fn(x, y, z) applied elementwise must come back as [nx, ny, nz], which
    # is the shape FNOGridInferencer then broadcasts to [B, nx, ny, nz].
    xx, yy, zz = _meshgrid_as_shipped(NX, NY, NZ)
    base = np.asarray((xx + yy + zz) % 2 == 0, dtype=bool)
    assert base.shape == (NX, NY, NZ)
