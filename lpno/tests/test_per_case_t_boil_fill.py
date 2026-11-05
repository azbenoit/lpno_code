"""The off-metal T fill can follow each case's own material (custom.per_case_t_boil).

`adopt_material_t_boil` already makes a single-material dataset of any alloy fill at its own
vaporization temperature. A MIXED set (Ti64 3520 K + IN718 3186 K) got one fill for both,
which is wrong for half its cases either way. With the flag on, `mat_Tv` is read as an
AUXILIARY channel -- never a model input -- and every fill site (training MaskNode, the
rollout post-processor, the materialized eval) uses one formula, `boiling_fill_from_channel`,
so training and eval cannot diverge. Off by default because the fill is part of the target.

Also pins that `custom.*` flags reach these paths through `run_cfg.cfg`: `RunConfig` carries
explicit fields only, so an opt-out read with `getattr(run_cfg, ...)` alone is inert.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lpno.custom_utils import utils as U
from lpno.equations.FNO_postprocess import MaskNode
from lpno.models.pipeline import preprocessing_datasets as PD
from lpno.models.pipeline.postprocessing import _apply_prediction_postprocessing

RS = {"T": 3000.0, "mat_Tv": 1000.0}          # channel stored as K/1000
TI64, IN718 = 3520.0, 3186.0


def _cfg(per_case=None, **over):
    custom = {} if per_case is None else {"per_case_t_boil": per_case}
    base = dict(material_params={"T_boil": TI64, "T_s": 1873.0, "T_l": 1923.0},
                ref_scales=RS, alpha_masking=True, cfg=OmegaConf.create({"custom": custom}))
    base.update(over)
    return SimpleNamespace(**base)


def test_channel_formula_matches_on_numpy_and_torch():
    ch = np.array([TI64, IN718]) / RS["mat_Tv"]
    want = np.array([TI64, IN718]) / RS["T"]
    assert np.allclose(U.boiling_fill_from_channel(ch, RS), want)
    assert torch.allclose(U.boiling_fill_from_channel(torch.as_tensor(ch), RS), torch.as_tensor(want))


def test_flag_off_keeps_the_scalar_even_when_the_channel_is_there():
    U._PER_CASE_FILL_ANNOUNCED.clear()
    d = U.t_fill_field(_cfg(per_case=False), {"mat_Tv": np.array([[IN718 / RS["mat_Tv"]]])})
    assert d == pytest.approx(TI64 / RS["T"])


def test_flag_reaches_through_run_cfg_cfg_not_only_attributes(capsys):
    """The flag is set in Hydra `custom`; nothing copies it onto run_cfg. It must still work."""
    U._PER_CASE_FILL_ANNOUNCED.clear()
    ch = np.array([[TI64], [IN718]]) / RS["mat_Tv"]
    d = U.t_fill_field(_cfg(per_case=True), {"mat_Tv": ch})
    assert np.allclose(d, np.array([[TI64], [IN718]]) / RS["T"])
    assert "PER-CASE mask fill" in capsys.readouterr().out


def test_flag_on_without_the_channel_falls_back_loudly(capsys):
    U._PER_CASE_FILL_ANNOUNCED.clear()
    d = U.t_fill_field(_cfg(per_case=True), {"T": np.zeros(1)})
    assert d == pytest.approx(TI64 / RS["T"])
    assert "carries no mat_Tv" in capsys.readouterr().out


def test_masknode_fills_each_sample_with_its_own_boiling_point():
    shape = (2, 1, 3, 2, 2)
    T = torch.full(shape, 0.5)
    alpha = torch.zeros(shape)                      # everything is gas -> everything filled
    mat_tv = torch.tensor([TI64, IN718]).view(2, 1, 1, 1, 1).expand(shape) / RS["mat_Tv"]
    node = MaskNode(keys=["T"], threshold=0.5, d=0.0, k=20, t_fill=TI64 / RS["T"],
                    t_fill_key="mat_Tv", t_fill_scale=RS["mat_Tv"] / RS["T"])
    out = node({"T": T.clone(), "alpha": alpha, "mat_Tv": mat_tv})
    assert torch.allclose(out["T"][0], torch.full(shape[1:], TI64 / RS["T"]), atol=1e-4)
    assert torch.allclose(out["T"][1], torch.full(shape[1:], IN718 / RS["T"]), atol=1e-4)


def test_masknode_without_the_key_uses_the_scalar():
    shape = (2, 1, 2, 2, 2)
    node = MaskNode(keys=["T"], threshold=0.5, d=0.0, k=20, t_fill=TI64 / RS["T"], t_fill_key="mat_Tv")
    out = node({"T": torch.zeros(shape), "alpha": torch.zeros(shape)})
    assert torch.allclose(out["T"], torch.full(shape, TI64 / RS["T"]), atol=1e-4)


def test_aux_key_rides_along_only_under_the_flag():
    base = ["V_scan", "P", "x", "y", "z"]
    assert PD._aux_invar_keys(_cfg(per_case=False), base) == base
    assert PD._aux_invar_keys(_cfg(per_case=True), base) == base + ["mat_Tv"]
    assert PD._aux_invar_keys(_cfg(per_case=True), base + ["mat_Tv"]) == base + ["mat_Tv"], "no duplicate"


def test_rollout_postprocessing_fills_per_case_from_invar():
    U._PER_CASE_FILL_ANNOUNCED.clear()
    shape = (2, 1, 2, 2, 2)
    preds = {"T": np.full(shape, 0.5, np.float32), "alpha": np.zeros(shape, np.float32)}
    invar = {"mat_Tv": np.array([TI64, IN718], np.float32).reshape(2, 1, 1, 1, 1) / RS["mat_Tv"]
             * np.ones(shape, np.float32)}
    out = _apply_prediction_postprocessing(preds, _cfg(per_case=True), target_keys=["T"], invar=invar)
    assert np.allclose(out["T"][0], TI64 / RS["T"], atol=1e-4)
    assert np.allclose(out["T"][1], IN718 / RS["T"], atol=1e-4)


def test_melt_window_opt_out_is_honoured_through_cfg(tmp_path):
    """Regression for the inert opt-out shipped in 49a430a: set only in Hydra custom."""
    import json
    (tmp_path / "c.manifest.json").write_text(json.dumps({"material": {"properties": {
        "solidus_temperature_k": 1523.15, "liquidus_temperature_k": 1608.15}}}))
    cfg = _cfg(); cfg.cfg = OmegaConf.create({"custom": {"melt_window_from_manifest": False}})
    PD.adopt_metric_melt_window(cfg, tmp_path)
    assert getattr(cfg, "metric_melt_window", None) is None
