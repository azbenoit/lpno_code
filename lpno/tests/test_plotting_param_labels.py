"""Regression: figure P/V_scan labels inverted the normalization with fixed factors.

Inputs are normalized as P/ref['P'] and V_scan/ref['V'], but the plotter
recovered a display value with hardcoded round(sample*10) and round(sample*100).
Those invert the normalization only when ref['P']==10 and ref['V']==0.1 -- true
of every live config, and false for the Abaqus one (P=10000, V=10), whose figure
titles and filenames therefore named a power off by three orders of magnitude.

The factors now come from the run's own ref_scales, so the labels follow the
config instead of one historical scale set. Live configs are unaffected: with
ref['P']==10 the value is identical to the old literal.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("physicsnemo.sym")

from lpno.plotting.validators import CustomSlicePlotter


def _param(value: float) -> np.ndarray:
    return np.full((1, 1), value, dtype=np.float32)


def test_default_scales_match_the_old_literals():
    plotter = CustomSlicePlotter(variables=["T"])
    # 150 W normalized by ref P=10 is 15.0; the label must read 150 again.
    assert plotter._label_power(_param(15.0), 0) == 150
    # 0.8 m/s normalized by ref V=0.1 is 8.0; the label reads mm/s.
    assert plotter._label_speed(_param(8.0), 0) == 800


def test_labels_follow_the_configured_ref_scales():
    plotter = CustomSlicePlotter(variables=["T"], ref_scales={"P": 10000.0, "V": 10.0})
    # 10 kW normalized by ref P=10000 is 1.0.
    assert plotter._label_power(_param(1.0), 0) == 10000
    # 2 m/s normalized by ref V=10 is 0.2 -> 2000 mm/s.
    assert plotter._label_speed(_param(0.2), 0) == 2000


def test_a_partial_ref_scales_mapping_keeps_the_defaults():
    plotter = CustomSlicePlotter(variables=["T"], ref_scales={"T": 3000.0})
    assert plotter._label_power(_param(15.0), 0) == 150


def test_no_hardcoded_denorm_factors_remain():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "lpno" /
           "plotting" / "validators.py").read_text(encoding="utf-8")
    assert "* 100))" not in src and "* 10))" not in src, (
        "P/V_scan labels must denormalize with ref_scales, not fixed factors"
    )
