"""Regression: an unparseable filename produced object-dtype P/V_scan arrays.

`_parse_power_speed_from_stem` returns (None, None) on failure, and that None
reached `np.full((1,1,nx,ny,nz), None)` -- a dtype=object array. It concatenates
and saves without complaint and only raises much later, inside torch.from_numpy,
by which point the offending filename is nowhere near the traceback.

This is the H5/Abaqus analogue of the P/V default-to-0.0 problem (A5); this
one was silently mis-typed rather than silently defaulted.

The Abaqus path is deprecated -- its only consumer is under src/deprecated/ --
so this is a fail-loudly fix on a dead path.
"""
from __future__ import annotations

import numpy as np
import pytest

from lpno.custom_utils.dataset_loading import H5AbaqusDataSource


# The parser takes the last two underscore tokens as <power>_<speed>.
@pytest.mark.parametrize("stem", ["random_file", "noPowerHere", "", "case_pow_speed"])
def test_unparseable_stems_return_none(stem: str):
    assert H5AbaqusDataSource._parse_power_speed_from_stem(stem) == (None, None)


def test_none_metadata_would_have_produced_an_object_array():
    # The exact construction the guard now prevents -- pinned so the reason the
    # guard exists stays visible if anyone is tempted to drop it.
    arr = np.full((1, 1, 2, 2, 2), None)
    assert arr.dtype == object
    with pytest.raises(TypeError):
        import torch

        torch.from_numpy(arr)


def test_a_parseable_stem_still_yields_numbers():
    power, speed = H5AbaqusDataSource._parse_power_speed_from_stem("case_250_0.8")
    assert power is not None and speed is not None
    assert np.full((1, 1, 2, 2, 2), power).dtype != object
