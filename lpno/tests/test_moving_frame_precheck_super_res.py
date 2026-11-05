"""Regression: the moving-frame pre-check ignored super_res_factor.

parse_directory validated `nX - Nsteps > 0`, but crop_shift_moving_frame raises
on `nX - (Nsteps-1)*super_res_factor <= 0`. The two diverge as the factor grows,
so a request that cleared the early, labelled check still aborted deep inside the
crop with a bare "Nsteps must be smaller than nX to keep a positive x-length".

Both now use the crop's formula, so whatever the pre-check accepts, the crop can
actually produce.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from lpno.custom_utils.openFoam_datasource import crop_shift_moving_frame

SRC = (Path(__file__).resolve().parents[1] / "src" / "lpno" /
       "custom_utils" / "openFoam_datasource.py").read_text(encoding="utf-8")


def _precheck_formula() -> str:
    """The pre-check's new_nX expression, read from the shipped source."""
    m = re.search(r"^\s*new_nX = (nX - .+)$", SRC, flags=re.MULTILINE)
    assert m, "could not find the pre-check new_nX expression"
    return m.group(1).strip()


def test_precheck_uses_the_crops_formula():
    # The crop's own expression, for comparison.
    crop = re.findall(r"^\s*new_nX = (nX - .+)$", SRC, flags=re.MULTILINE)
    assert len(crop) >= 2, "expected a new_nX in both the crop and the pre-check"
    # Compare the expressions, not their spacing.
    normalised = {re.sub(r"\s+", "", c) for c in crop}
    assert len(normalised) == 1, f"pre-check and crop disagree on new_nX: {sorted(normalised)}"
    assert "super_res_factor" in _precheck_formula()


@pytest.mark.parametrize("super_res_factor,nsteps,nx,ok", [
    (1, 3, 10, True),
    (4, 3, 10, True),    # 10 - 2*4 = 2 > 0
    (4, 4, 10, False),   # 10 - 3*4 = -2  -> the case the pre-check used to pass
    (8, 3, 10, False),   # 10 - 2*8 = -6
])
def test_crop_agrees_with_the_formula(super_res_factor: int, nsteps: int, nx: int, ok: bool):
    data = np.zeros((1, 5, 1, nx, 2, 2), dtype=np.float32)
    if ok:
        out = crop_shift_moving_frame(data, nsteps, super_res_factor=super_res_factor)
        assert out.shape[3] == nx - (nsteps - 1) * super_res_factor
    else:
        with pytest.raises(ValueError):
            crop_shift_moving_frame(data, nsteps, super_res_factor=super_res_factor)
