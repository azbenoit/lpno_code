"""Regression: P/V_scan silently defaulted to 0.0 when a filename lacked the tokens.

`float(params.get('laser_power', 0.0))` and its V_scan sibling substituted zero
rather than failing, and those become the P/V_scan channels. H_norm is then
P/sqrt(|V_scan|) = 0/0 = NaN, and V_scan=0 is not a scan speed a welding case
can have. The run trained on it anyway.

Both are now refused: the VTK datasource raises on a stem it cannot parse, and
preprocessing refuses a non-positive power or speed whatever the source. All
8176 consolidated NPZ cases under data/flow3D carry non-zero values, so nothing
that loads today is affected.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

SRC = (Path(__file__).resolve().parents[1] / "src" / "lpno" /
       "custom_utils" / "openFoam_datasource.py").read_text(encoding="utf-8")
PREPROCESS = (Path(__file__).resolve().parents[1] / "src" / "lpno" / "models" /
              "pipeline" / "preprocessing_datasets.py").read_text(encoding="utf-8")


def test_the_datasource_no_longer_defaults_power_or_speed_to_zero():
    assert 'params.get("laser_power", 0.0)' not in SRC
    assert 'params.get("V_scan", 0.0)' not in SRC
    # It must say which file and which token, or the failure is unactionable.
    block = SRC.split("def _require_scan_parameters", 1)
    assert len(block) == 2, "expected a named guard, not an inline expression"
    body = block[1].split("\n    def ", 1)[0]
    assert "laser_power" in body and "V_scan" in body
    assert "raise ValueError" in body


def test_preprocessing_refuses_unphysical_power_or_speed():
    guard = re.search(r"def _require_physical_laser_parameters.*?(?=\ndef )", PREPROCESS, re.S)
    assert guard, "expected a laser-parameter guard before H_norm"
    assert "raise ValueError" in guard.group(0)
    # The guard has to run before H_norm is formed, not after.
    assert PREPROCESS.index("_require_physical_laser_parameters(var)") < PREPROCESS.index('var["H_norm"]')


@pytest.mark.parametrize("power,speed", [(0.0, 0.5), (100.0, 0.0), (-1.0, 0.5), (100.0, -0.0)])
def test_guard_rejects_zero_values(power: float, speed: float):
    from lpno.models.pipeline.preprocessing_datasets import (
        _require_physical_laser_parameters,
    )

    var = {
        "P": np.full((1, 1, 2, 2, 2), power, dtype=np.float32),
        "V_scan": np.full((1, 1, 2, 2, 2), speed, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="physically"):
        _require_physical_laser_parameters(var)


def test_a_reverse_direction_scan_is_still_a_scan():
    from lpno.models.pipeline.preprocessing_datasets import (
        _require_physical_laser_parameters,
    )

    var = {
        "P": np.full((1, 1, 2, 2, 2), 150.0, dtype=np.float32),
        "V_scan": np.full((1, 1, 2, 2, 2), -0.5, dtype=np.float32),
    }
    _require_physical_laser_parameters(var)  # negative speed = opposite direction


def test_guard_accepts_a_real_case():
    from lpno.models.pipeline.preprocessing_datasets import (
        _require_physical_laser_parameters,
    )

    var = {
        "P": np.full((1, 1, 2, 2, 2), 150.0, dtype=np.float32),
        "V_scan": np.full((1, 1, 2, 2, 2), 0.5, dtype=np.float32),
    }
    _require_physical_laser_parameters(var)  # must not raise
