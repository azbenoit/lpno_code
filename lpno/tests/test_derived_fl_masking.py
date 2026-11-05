"""Regression: fl derived from a predicted T reads 1 across the whole gas layer.

The network is trained on a T target whose off-metal cells are filled with the
boiling temperature, so its predicted T is ~T_boil in the gas *by construction*.
Any fl derived from that T is therefore 1 everywhere off the metal, while the
ground-truth fl is masked to 0 there -- measured on the random-path test set as
pred fl>0.5 on 24.9% of points against 0.11% in GT, iou_mean=0.0002, with the
same model's T and alpha metrics fine.

Deriving fl before the T fill does not help (the fill is in the prediction
already); the fix is to zero-fill the derived fl outside the metal region, which
is what the grid path was already doing for its own derived fl.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lpno.models.pipeline.postprocessing import (
    _apply_prediction_postprocessing,
    _derive_fl,
)

SRC = (Path(__file__).resolve().parents[1] / "src" / "lpno" / "models" /
       "pipeline" / "postprocessing.py").read_text(encoding="utf-8")


def _run_cfg(alpha_masking: bool = True):
    return SimpleNamespace(
        alpha_masking=alpha_masking,
        # The helper normalizes these by ref_scales["T"], so 1500/1800 K become
        # 0.5/0.6 against the normalized T tensors below.
        material_params={"T_s": 1500.0, "T_l": 1800.0},
        ref_scales={"T": 3000.0},
    )


def _gas_over_metal():
    """(1,1,2,2,4) tensors: z<2 is metal at 300 K, z>=2 is gas at T_boil."""
    T = np.full((1, 1, 2, 2, 4), 300.0 / 3000.0, dtype=np.float32)
    T[..., 2:] = 3123.0 / 3000.0
    alpha = np.zeros((1, 1, 2, 2, 4), dtype=np.float32)
    alpha[..., :2] = 1.0
    return T, alpha


#: The bug this file guards is fl reading ~1 off the metal. Since 2026-08-04 the
#: liquid fraction is a sigmoid rather than a tanh (so its gradient survives the
#: cold tail instead of being bit-zero below 1717.8 K), and a sigmoid never
#: returns exactly 0 -- 300 K metal now reads 1.7e-10 instead of a saturated 0.0.
#: Asserting exact equality would test the saturation, not the masking. This
#: tolerance is ~9 orders below the 0.5 melt threshold and 8 below the 1e-6 gate
#: floor, so a genuine regression (gas reading as melt) still fails loudly.
NEGLIGIBLE_FL = 1e-6


def test_derived_fl_is_zero_in_the_gas():
    T, alpha = _gas_over_metal()
    fl = _derive_fl({"T": T, "alpha": alpha}, _run_cfg())
    assert fl[..., 2:].max() < NEGLIGIBLE_FL, "gas cells must not read as melt"
    assert fl[..., :2].max() < NEGLIGIBLE_FL, "300 K metal is below T_s, so no melt either"
    # The regression this guards is fl ~ 1, which these bound out by ~6 orders.
    assert fl.max() < 0.5


def test_molten_metal_still_reads_as_melt():
    T, alpha = _gas_over_metal()
    T[..., :2] = 1.0  # normalized T above T_l -> fully liquid
    fl = _derive_fl({"T": T, "alpha": alpha}, _run_cfg())
    assert fl[..., :2].min() > 0.99
    assert fl[..., 2:].max() == 0.0


def test_without_alpha_masking_the_fill_is_not_applied():
    # A run trained without alpha_masking has no boiling fill in its target, so
    # zeroing the gas here would be inventing a mask the model never saw.
    T, alpha = _gas_over_metal()
    fl = _derive_fl({"T": T, "alpha": alpha}, _run_cfg(alpha_masking=False))
    assert fl[..., 2:].max() > 0.99


def test_postprocessing_leaves_no_gas_melt():
    T, alpha = _gas_over_metal()
    preds = _apply_prediction_postprocessing(
        {"T": T.copy(), "alpha": alpha.copy()}, _run_cfg(), target_keys=["T", "alpha"]
    )
    assert preds["fl"][..., 2:].max() == 0.0


def test_every_fl_derivation_goes_through_the_helper():
    # The bug was one unmasked derivation site out of several; a new one added
    # later would silently reintroduce it.
    body = SRC.split("def _derive_fl(", 1)[1]
    body = body.split("\ndef ", 1)[1]  # everything after the helper
    assert "smooth_liquid_fraction_numpy(" not in body, (
        "derive fl through _derive_fl so the off-metal zero-fill is not skipped"
    )
    # ...and the helper itself must actually mask.
    helper = SRC.split("def _derive_fl(", 1)[1].split("\ndef ", 1)[0]
    assert "fill_where_mask_smooth_np" in helper
    assert re.search(r"d=0\.0", helper)
