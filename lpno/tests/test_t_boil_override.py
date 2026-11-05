"""`custom.T_boil_k` must reach everything that reads the boiling temperature.

The mask fill, the MaskNode and the thermal-headroom metrics all read
`boiling_temperature(material_params)`, which falls back to the hardcoded
pre-calibration T_BOIL_K = 3123.0. The calibrated Ti64 material's own
vaporization temperature is 3520 K (materials/ti64_v3_calibrated), so on
calibrated data the fill sits ~400 K below the material's boiling point and
600-1200 K below real metal peaks -- the training target gets an artificial
cold cliff at every gas boundary and predicted-metal peaks are capped at the
fill wherever predicted alpha is wrong (measured: tail_deficit_max_T pinned at
(GT max - 3123)/3000 for 28k steps on the RIGNO and GAOT mref arms).

One injection point (material_params["T_boil"], the door the shared constant left) keeps the
fill, the mask and the metrics coherent.
"""
import numpy as np

from lpno.custom_utils.utils import boiling_temperature
from lpno.models.pipeline.config_loading import apply_t_boil_override


def test_default_is_the_legacy_constant():
    assert boiling_temperature({}) == 3123.0
    assert boiling_temperature(None) == 3123.0


def test_override_reaches_material_params():
    mp = {}
    apply_t_boil_override(mp, 3520.0)
    assert mp["T_boil"] == 3520.0
    assert boiling_temperature(mp) == 3520.0


def test_absent_override_leaves_material_params_untouched():
    mp = {}
    apply_t_boil_override(mp, None)
    assert "T_boil" not in mp
    assert boiling_temperature(mp) == 3123.0


def test_bogus_override_is_refused():
    import pytest

    mp = {}
    with pytest.raises(ValueError, match="T_boil_k"):
        apply_t_boil_override(mp, 0.0)
    with pytest.raises(ValueError, match="T_boil_k"):
        apply_t_boil_override(mp, -3520.0)
