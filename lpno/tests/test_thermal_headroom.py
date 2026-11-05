"""Thermal headroom: how close the predicted peak is to the melt threshold.

Measured 2026-08-04. Two models clip their temperature tail at a hard ceiling:

- varspeed model 29125: raw substrate peak 1631 K, i.e. 267 K BELOW the 1898 K
  melt threshold -> melt volume identically 0, while whole-domain T RMSE stayed a
  respectable 84 K and alpha IoU 0.998.
- "healthy" 1601 model 25218: ceiling 1944 K, only 46 K ABOVE the threshold. It
  produces melt, but is one small regression away from the same cliff.

Melt is ~1% of the substrate, so flattening the peak barely moves an L2
objective while any thresholded quantity collapses to exactly zero. No existing
metric showed this: melt volume reports the consequence after it happens, and
nothing reported the margin. `thermal_headroom_stats` reports it directly, so a
model that is about to fall off the cliff is visible before it does.

`fl > 0.5` corresponds to T above the MIDPOINT (T_s+T_l)/2, not T_l -- the
liquid fraction is a sigmoid centred there (custom_utils/utils.py; it was a tanh
until 2026-08-04, changed for gradient reasons with the forward values pinned
unchanged by tests/test_liquid_fraction_sigmoid.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import thermal_headroom_stats

T_S, T_L = 1873.0, 1923.0
THRESHOLD = 0.5 * (T_S + T_L)  # 1898 K
REF_T = 3000.0
NZ = 6


def _fields(peak_K, n=200):
    """Substrate temperatures with a controlled peak; z<=0 for the lower half."""
    zs = np.arange(NZ, dtype=np.float64) - (NZ - 2)
    z = np.broadcast_to(zs[None, None, None, None, :], (1, 1, 1, n, NZ)).copy()
    T = np.full((1, 1, 1, n, NZ), 300.0)
    T[0, 0, 0, -1, :] = peak_K       # one hot column
    T[0, 0, 0, -2, :] = peak_K * 0.9
    alpha = np.ones_like(T)          # all metal
    return T / REF_T, alpha, z


def test_reports_negative_headroom_when_the_peak_cannot_melt():
    """The varspeed case: peak below the threshold means melt is impossible."""
    T, alpha, z = _fields(1631.0)
    stats = thermal_headroom_stats(T, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T)
    assert stats["melt_threshold_K"] == pytest.approx(THRESHOLD)
    assert stats["pred_T_max_K"] == pytest.approx(1631.0)
    assert stats["thermal_headroom_K"] < 0
    assert stats["thermal_headroom_K"] == pytest.approx(1631.0 - THRESHOLD)
    assert stats["can_reach_melt"] == 0.0


def test_reports_small_positive_headroom_near_the_cliff():
    """The 1601 case: melt is produced, but only 46 K of margin."""
    T, alpha, z = _fields(1944.0)
    stats = thermal_headroom_stats(T, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T)
    assert stats["thermal_headroom_K"] == pytest.approx(1944.0 - THRESHOLD)
    assert 0 < stats["thermal_headroom_K"] < 60
    assert stats["can_reach_melt"] == 1.0


def test_gas_cells_do_not_set_the_peak():
    """z>0 is gas; with alpha masking it carries a boiling fill, not a prediction.

    Reading the peak from the whole domain is how a 3523 K "prediction" was
    reported for a model whose real substrate ceiling was 1631 K.
    """
    T, alpha, z = _fields(1631.0)
    T = T.copy()
    T[0, 0, 0, :, z[0, 0, 0, 0] > 0] = 3123.0 / REF_T  # boiling fill in the gas
    stats = thermal_headroom_stats(T, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T)
    assert stats["pred_T_max_K"] == pytest.approx(1631.0), "gas fill leaked into the peak"


def test_void_cells_do_not_set_the_peak():
    """`_apply_prediction_postprocessing` fills T=T_boil where alpha<0.5."""
    T, alpha, z = _fields(1631.0)
    T = T.copy()
    alpha = alpha.copy()
    T[0, 0, 0, 0, 0] = 3123.0 / REF_T
    alpha[0, 0, 0, 0, 0] = 0.0  # void -> that T is a fill, not a prediction
    stats = thermal_headroom_stats(T, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T)
    assert stats["pred_T_max_K"] == pytest.approx(1631.0)


def test_percentile_is_robust_to_a_single_spike():
    """p99.9 should track the bulk tail, not one outlier cell."""
    T, alpha, z = _fields(1700.0, n=2000)
    T = T.copy()
    T[0, 0, 0, 0, 0] = 5000.0 / REF_T
    stats = thermal_headroom_stats(T, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T)
    assert stats["pred_T_max_K"] == pytest.approx(5000.0)
    assert stats["pred_T_p999_K"] < 2000.0


def test_returns_empty_without_a_substrate():
    T, alpha, z = _fields(1631.0)
    assert thermal_headroom_stats(T, alpha, np.ones_like(z), T_s=T_S, T_l=T_L, ref_T=REF_T) == {}


def test_tail_deficit_measures_cold_running_where_headroom_does_not():
    """GAOT disp_deep_v2: +837 K headroom yet 151 K colder than truth at p99.9.

    Headroom asks "can it melt at all", which reads healthy for a model that
    clears the threshold and still runs cold. The tail deficit is the quantity
    that spans both regimes.
    """
    pred, alpha, z = _fields(2735.0, n=2000)
    true, _, _ = _fields(3130.0, n=2000)
    s = thermal_headroom_stats(
        pred, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T, true_T=true, true_alpha=alpha
    )
    assert s["thermal_headroom_K"] > 800, "headroom looks healthy"
    assert s["tail_deficit_K"] > 0, "yet the tail is measurably colder than truth"
    assert s["true_T_p999_K"] > s["pred_T_p999_K"]


def test_tail_deficit_absent_without_truth():
    pred, alpha, z = _fields(2735.0)
    s = thermal_headroom_stats(pred, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T)
    assert "tail_deficit_K" not in s


def test_can_reach_melt_is_numeric_so_it_survives_storage():
    """A Python bool is dropped by the AiiDA Dict round-trip.

    Measured on eval 30164: `can_reach_melt` vanished from the stored metrics
    entirely while every float beside it survived, which made the guard useless
    in production. 1.0 = can melt, 0.0 = cannot.
    """
    cold, alpha, z = _fields(1631.0)
    hot, _, _ = _fields(2200.0)
    assert thermal_headroom_stats(cold, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T)["can_reach_melt"] == 0.0
    assert thermal_headroom_stats(hot, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T)["can_reach_melt"] == 1.0


def test_both_tail_deficit_variants_are_emitted():
    """The streaming path can only report the peak-based one, so the
    non-streaming path must emit it too or the two are not comparable."""
    pred, alpha, z = _fields(2735.0, n=2000)
    true, _, _ = _fields(3130.0, n=2000)
    s = thermal_headroom_stats(pred, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T,
                               true_T=true, true_alpha=alpha)
    assert "tail_deficit_K" in s and "tail_deficit_max_K" in s
    assert s["tail_deficit_max_K"] == pytest.approx(3130.0 - 2735.0)


def test_tail_deficit_records_its_frame_count_and_normalisation():
    """A peak-based deficit grows as sqrt(2 ln N); comparing across N is invalid.

    Measured 2026-08-05: the cold peak is phase-averaging, and with sigma ~112.9 K
    the expected excess of max-over-N above trend is sigma*sqrt(2 ln N) --
    predicting 381 K at 301 frames (366.9 measured) and 471 K at 6016 frames
    (449.3 measured). So `tail_deficit_max_K` alone is not comparable between two
    evals unless they used the same number of frames.
    """
    import numpy as np

    pred, alpha, z = _fields(2735.0, n=2000)
    true, _, _ = _fields(3130.0, n=2000)
    s = thermal_headroom_stats(pred, alpha, z, T_s=T_S, T_l=T_L, ref_T=REF_T,
                               true_T=true, true_alpha=alpha)
    assert "tail_deficit_n_frames" in s
    assert "tail_deficit_per_sqrt2lnN_K" in s
    n = s["tail_deficit_n_frames"]
    expected = s["tail_deficit_max_K"] / np.sqrt(2 * np.log(max(n, 2)))
    assert s["tail_deficit_per_sqrt2lnN_K"] == pytest.approx(expected)


def test_the_eager_evaluation_reports_the_peak(monkeypatch, tmp_path):
    """`run_inference_and_metrics` (the eager path) never called `thermal_headroom_stats`.

    So an eager evaluation (lpno' own eval entry point on a folder of
    NPZ files, as in reproducibility/train_evaluate.ipynb) had no `pred_T_max_K` or
    `true_T_max_K`, which the lazy path reports and the paper's peak ratio is read
    from (reproducibility/lpno_repro.py `metrics`).
    """
    from types import SimpleNamespace

    import torch

    from lpno.models.pipeline import postprocessing

    T_true, alpha, z = _fields(2400.0)
    T_pred = T_true * (2100.0 / 2400.0)          # a model 300 K short at the peak

    class _Inferencer:
        def __init__(self, **kwargs):
            pass

        def run(self, ds_input, name, step, batch_size):
            return {"T": T_pred.astype(np.float32), "alpha": alpha.astype(np.float32)}

    monkeypatch.setattr(postprocessing, "FNOGridInferencer", _Inferencer)
    metrics = postprocessing.run_inference_and_metrics(
        model=torch.nn.Identity(),
        ds_input={"z": z.astype(np.float32)},
        ds_output={"T": T_true.astype(np.float32), "alpha": alpha.astype(np.float32)},
        run_cfg=SimpleNamespace(
            inference_batch_size=1,
            keys=SimpleNamespace(str_fno_input_keys=["z"], str_fno_output_keys=["T", "alpha"]),
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            alpha_masking=False,
            material_params={"T_s": T_S, "T_l": T_L},
            ref_scales={"T": REF_T},
        ),
        output_dir=tmp_path / "fno_eval",
    )
    T = metrics.errors["T"]
    assert T["pred_T_max_K"] == pytest.approx(2100.0, rel=1e-5)
    assert T["true_T_max_K"] == pytest.approx(2400.0, rel=1e-5)
    assert T["melt_threshold_K"] == pytest.approx(THRESHOLD)
    assert T["thermal_headroom_K"] == pytest.approx(2100.0 - THRESHOLD, rel=1e-5)
    assert "pred_T_max_K" in (tmp_path / "fno_eval" / "model_errors.txt").read_text()
