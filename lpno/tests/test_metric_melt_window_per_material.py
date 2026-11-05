"""Melt metrics must threshold against the DATA's material, not the config's.

`melt_threshold_K`, `thermal_headroom_K` and `can_reach_melt` take the midpoint
`(T_s+T_l)/2` from `material_params`. Evaluating a Ti64-trained model on IN718 therefore
measured all three against 1898.0 K -- Ti64's midpoint -- where IN718's is 1565.65, 332 K
out (self-improve session, ds 87432, 2026-09-04). `fl` was already correct: `_derive_fl`
prefers the per-case mat_Ts/mat_Tl channels.

The fix is measurement-only by construction: it writes `run_cfg.metric_melt_window`, never
`material_params`, because that dict also drives `fl`, which is a TARGET.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lpno.custom_utils.utils import melt_window_from_manifest
from lpno.models.pipeline.postprocessing import metric_melt_threshold_K
from lpno.models.pipeline.preprocessing_datasets import adopt_metric_melt_window

TI64 = (1873.15, 1923.15)     # midpoint 1898.15
IN718 = (1523.15, 1608.15)    # midpoint 1565.65


def _manifest(window, nest=False):
    inner = {"material": {"properties": {
        "solidus_temperature_k": window[0], "liquidus_temperature_k": window[1],
        "vaporization_temperature_k": 3186.0}}}
    return {"simulation_metadata": {"flow3d_case_manifest": inner}} if nest else inner


def _cfg(**over):
    base = dict(material_params={"T_s": TI64[0], "T_l": TI64[1]},
                ref_scales={"T": 3000.0, "mat_Ts": 1.0, "mat_Tl": 1.0})
    base.update(over)
    return SimpleNamespace(**base)


def _write(tmp_path, windows):
    for i, w in enumerate(windows):
        (tmp_path / f"case_{i}.manifest.json").write_text(json.dumps(_manifest(w)))
    return tmp_path


@pytest.mark.parametrize("nest", [False, True])
def test_manifest_reader_handles_both_nestings(nest):
    assert melt_window_from_manifest(_manifest(IN718, nest=nest)) == IN718


def test_manifest_reader_refuses_half_a_window():
    m = _manifest(IN718)
    del m["material"]["properties"]["solidus_temperature_k"]
    assert melt_window_from_manifest(m) is None, "half a window would mix two materials"


def test_unanimous_dataset_window_is_adopted_for_metrics(tmp_path, capsys):
    cfg = _cfg()
    adopt_metric_melt_window(cfg, _write(tmp_path, [IN718] * 3))
    assert cfg.metric_melt_window == IN718
    assert metric_melt_threshold_K(cfg) == pytest.approx(1565.65)
    assert "melt window is 1523.15-1608.15 K" in capsys.readouterr().out


def test_adoption_never_touches_material_params(tmp_path):
    """`fl` is a target and reads material_params; a measurement fix must not move it."""
    cfg = _cfg()
    adopt_metric_melt_window(cfg, _write(tmp_path, [IN718] * 3))
    assert cfg.material_params == {"T_s": TI64[0], "T_l": TI64[1]}


def test_multi_material_dataset_adopts_nothing(tmp_path, capsys):
    cfg = _cfg()
    adopt_metric_melt_window(cfg, _write(tmp_path, [IN718, TI64]))
    assert getattr(cfg, "metric_melt_window", None) is None
    assert "multi-material dataset" in capsys.readouterr().out


def test_opt_out_and_explicit_override(tmp_path):
    off = _cfg(melt_window_from_manifest=False)
    adopt_metric_melt_window(off, _write(tmp_path, [IN718] * 2))
    assert getattr(off, "metric_melt_window", None) is None
    explicit = _cfg(melt_window_k=[1000.0, 1200.0])
    adopt_metric_melt_window(explicit, _write(tmp_path, [IN718] * 2))
    assert explicit.metric_melt_window == (1000.0, 1200.0)


def test_channels_win_over_both_and_are_per_case():
    """A mixed batch has no single threshold; the channels give one per element."""
    cfg = _cfg(); cfg.metric_melt_window = IN718
    t = {"mat_Ts": np.array([TI64[0], IN718[0]]), "mat_Tl": np.array([TI64[1], IN718[1]])}
    assert metric_melt_threshold_K(cfg, t) == pytest.approx([1898.15, 1565.65])


def test_config_is_the_final_fallback():
    assert metric_melt_threshold_K(_cfg()) == pytest.approx(1898.15)


def test_single_material_margin_is_arithmetically_unchanged():
    """The invariance that keeps every existing campaign's number identical: with one
    material the threshold is constant, so max(peak) - thr == max(peak - thr)."""
    cfg = _cfg()
    thr = float(np.asarray(metric_melt_threshold_K(cfg)).min())
    peaks = [1700.0, 1950.0, 1880.0]
    assert max(peaks) - thr == pytest.approx(max(p - thr for p in peaks))
