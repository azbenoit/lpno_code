"""Regression: the off-metal T fill was hardcoded twice, against different scales.

Preprocessing fills off-metal T with raw 3123 K and normalizes by
``ref_scales['T']``; MaskNode filled with a literal ``3123/3000``. They agree
only while ref_scales.T is 3000 -- change it and the training target and the
model's own masked output disagree silently, with nothing raising.

Both now read one T_BOIL_K constant and divide by the configured T reference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lpno.custom_utils.utils import T_BOIL_K, boiling_fill_value
from lpno.equations.FNO_postprocess import MaskNode

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "lpno"


def test_fill_value_follows_the_configured_t_reference():
    assert boiling_fill_value({}, {"T": 3000.0}) == pytest.approx(T_BOIL_K / 3000.0)
    assert boiling_fill_value({}, {"T": 6000.0}) == pytest.approx(T_BOIL_K / 6000.0)


def test_a_configured_boiling_point_wins_over_the_default():
    assert boiling_fill_value({"T_boil": 2800.0}, {"T": 2800.0}) == pytest.approx(1.0)
    # A null in the yaml means "unset", not "fill with nothing".
    assert boiling_fill_value({"T_boil": None}, {"T": 3000.0}) == pytest.approx(T_BOIL_K / 3000.0)


def test_masknode_fills_t_at_the_configured_scale():
    ref_t = 6000.0
    fill = boiling_fill_value({}, {"T": ref_t})
    node = MaskNode(keys=["T"], threshold=0.5, d=0.0, k=20, t_fill=fill)
    T = torch.zeros(1, 1, 2, 2, 4)
    alpha = torch.zeros(1, 1, 2, 2, 4)
    alpha[..., :2] = 1.0
    out = node({"T": T, "alpha": alpha})["T"].numpy()
    assert np.allclose(out[..., 2:], T_BOIL_K / ref_t, atol=1e-5)
    assert np.allclose(out[..., :2], 0.0, atol=1e-5)


def test_masking_t_without_a_fill_value_is_refused():
    with pytest.raises(ValueError, match="t_fill"):
        MaskNode(keys=["T", "alpha"], threshold=0.5, d=0.0, k=20)
    # Nodes that never touch T do not need it.
    MaskNode(keys=["u", "v", "w"], threshold=0.5, d=0.0, k=20, mask_str_key="fl")


@pytest.mark.parametrize("relative_path", [
    "equations/FNO_postprocess.py",
    "models/pipeline/postprocessing.py",
    "models/pipeline/preprocessing_datasets.py",
])
def test_no_boiling_point_literal_left_in_the_pipeline(relative_path: str):
    text = (SRC_ROOT / relative_path).read_text(encoding="utf-8")
    assert "3123" not in text, f"{relative_path} still hardcodes the boiling point"
