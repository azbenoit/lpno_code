from __future__ import annotations

import numpy as np
import pytest

from app.service import _same_grid_metrics, _steady_resampled_metrics


def test_same_grid_metrics_matches_numpy() -> None:
    pred = {"T": np.array([[[[[1.0, 2.0, 3.0]]]]])}
    gt = {"T": np.array([[[[[1.5, 2.0, 1.0]]]]])}
    mae, rmse = _same_grid_metrics(pred, gt, "T")
    d = np.array([0.5, 0.0, 2.0])
    assert mae == float(np.abs(d).mean())
    assert rmse == float(np.sqrt((d * d).mean()))


def test_same_grid_metrics_none_on_shape_mismatch() -> None:
    pred = {"T": np.zeros((1, 1, 4, 4, 4))}
    gt = {"T": np.zeros((3, 3, 3))}
    assert _same_grid_metrics(pred, gt, "T") == (None, None)


def _lin(x, y, z):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return 10.0 + 5.0 * X + 2.0 * Y - 3.0 * Z


def test_steady_resampled_metrics_zero_when_pred_equals_gt() -> None:
    # Coarse GT and a finer prediction of the SAME linear field over the same
    # extent -> trilinear interpolation reproduces GT exactly -> zero error.
    gx, gy, gz = np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3)
    px, py, pz = np.linspace(0, 1, 7), np.linspace(0, 1, 5), np.linspace(0, 1, 4)
    gt = _lin(gx, gy, gz)
    pred = _lin(px, py, pz)
    mae, rmse = _steady_resampled_metrics(pred, (px, py, pz), gt, (gx, gy, gz))
    assert mae < 1e-9 and rmse < 1e-9


def test_steady_resampled_metrics_handles_coarser_gt_yz() -> None:
    # GT y/z counts differ from the prediction (the native-10um-vs-super-res case
    # that used to return None); trilinear interp still compares and returns a
    # real number (constant offset of 100 -> MAE 100).
    gx, gy, gz = np.linspace(0, 1, 4), np.linspace(0, 1, 3), np.linspace(0, 1, 3)
    px, py, pz = np.linspace(0, 1, 8), np.linspace(0, 1, 6), np.linspace(0, 1, 5)
    gt = np.full((4, 3, 3), 500.0)
    pred = np.full((8, 6, 5), 600.0)
    mae, rmse = _steady_resampled_metrics(pred, (px, py, pz), gt, (gx, gy, gz))
    assert mae == pytest.approx(100.0) and rmse == pytest.approx(100.0)


def test_steady_resampled_metrics_none_when_no_overlap() -> None:
    # Prediction extent entirely outside the GT extent -> nothing to compare.
    gx, gy, gz = np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3)
    px, py, pz = np.linspace(5, 6, 4), np.linspace(5, 6, 4), np.linspace(5, 6, 4)
    assert _steady_resampled_metrics(
        np.zeros((4, 4, 4)), (px, py, pz), np.zeros((3, 3, 3)), (gx, gy, gz)
    ) == (None, None)


def test_same_grid_metrics_score_the_substrate_only() -> None:
    """The pages' error caption is the paper's substrate MAE: z <= 0 slices only.
    A gas-plane error (z > 0) must not reach it, and a model that serves fewer gas
    planes (the straight RIGNO's 30-plane crop) must score the same substrate."""
    import numpy as np

    from app.service import _same_grid_metrics

    z40 = np.linspace(-2.95e-4, 9.5e-5, 40)         # 30 substrate + 10 gas planes
    gt = {"T": np.full((2, 4, 3, 40), 1000.0)}
    pred = {"T": np.full((2, 4, 3, 40), 1000.0)}
    pred["T"][..., :30] += 5.0                        # 5 K everywhere in the substrate
    pred["T"][..., 30:] += 2000.0                     # a huge gas-plane error
    mae, rmse = _same_grid_metrics(pred, gt, "T", z_axis=z40)
    assert mae == pytest.approx(5.0) and rmse == pytest.approx(5.0)
    whole, _ = _same_grid_metrics(pred, gt, "T")      # no axis: the whole volume, gas included
    assert whole > 400
    # the 30-plane crop of the same field scores identically
    mae30, _ = _same_grid_metrics({"T": pred["T"][..., :30]}, {"T": gt["T"][..., :30]}, "T", z_axis=z40[:30])
    assert mae30 == pytest.approx(mae)


def test_steady_resampled_metrics_score_the_substrate_only() -> None:
    import numpy as np

    from app.service import _steady_resampled_metrics

    x, y = np.linspace(0, 1, 5), np.linspace(0, 1, 4)
    z = np.linspace(-2.95e-4, 9.5e-5, 40)
    gt = np.full((5, 4, 40), 1000.0)
    pred = gt.copy()
    pred[..., :30] += 3.0
    pred[..., 30:] += 3000.0
    mae, rmse = _steady_resampled_metrics(pred, (x, y, z), gt, (x, y, z))
    assert mae == pytest.approx(3.0, rel=1e-6) and rmse == pytest.approx(3.0, rel=1e-6)
