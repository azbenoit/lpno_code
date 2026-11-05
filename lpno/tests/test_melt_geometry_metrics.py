"""Melt-pool and keyhole geometry reported as default eval metrics.

The dataset side already computes them per simulation (aiida-pino
``console/viz.py``: cavity depth from Fraction Of Fluid < 0.5, melt pool from
melt fraction > 0.5, both inside the metal); the model side only reported
voxel-wise error, so nothing said whether a prediction gets the pool *shape*
right. ``melt_geometry_stats`` computes the same scalars from prediction and
truth with the same thresholds and reports both plus their per-sample gap.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lpno.models.pipeline.postprocessing import (
    melt_geometry_stats,
    run_lazy_dataset_metrics,
)

# z = -4..1 in normalized units; substrate is idx 0..4 (5 slices), idx 5 is gas.
Z1D = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0], dtype=np.float32)
# 1 normalized unit == 1 um, so cell counts and um are the same number here.
L_SCALE_M = 1e-6
SHAPE = (1, 1, 6, 6, 6)


def _z_field(shape=SHAPE) -> np.ndarray:
    return np.broadcast_to(Z1D, shape).astype(np.float32)


def test_melt_pool_depth_length_width_and_volume() -> None:
    truth_fl = np.zeros(SHAPE)
    truth_fl[0, 0, 1:5, 2:4, 3:5] = 1.0  # x span 4, y span 2, z idx 3-4 => depth 2
    pred_fl = np.zeros(SHAPE)
    pred_fl[0, 0, 1:5, 2:4, 2:5] = 1.0  # one cell deeper => depth 3

    stats = melt_geometry_stats({"fl": pred_fl}, {"fl": truth_fl}, _z_field(), L_SCALE_M)

    assert stats["melt_depth_um_true"] == pytest.approx(2.0)
    assert stats["melt_depth_um_pred"] == pytest.approx(3.0)
    assert stats["melt_depth_um_abs_err"] == pytest.approx(1.0)
    assert stats["melt_length_um_true"] == pytest.approx(4.0)
    assert stats["melt_width_um_true"] == pytest.approx(2.0)
    assert stats["melt_volume_um3_true"] == pytest.approx(16.0)
    assert stats["melt_volume_um3_pred"] == pytest.approx(24.0)
    assert stats["melt_volume_um3_abs_err"] == pytest.approx(8.0)


def test_keyhole_depth_from_alpha_cavity_ignores_boundary_columns() -> None:
    truth_alpha = np.ones(SHAPE)
    truth_alpha[0, 0, 2:4, 2:4, 3:5] = 0.0  # cavity, depth 2
    pred_alpha = np.ones(SHAPE)
    pred_alpha[0, 0, 2:4, 2:4, 1:5] = 0.0  # depth 4
    # A column of gas on the x boundary is a domain-edge artifact, not a
    # keyhole: aiida-pino drops boundary columns and so must this.
    pred_alpha[0, 0, 0, 2:4, 0:5] = 0.0

    stats = melt_geometry_stats({"alpha": pred_alpha}, {"alpha": truth_alpha}, _z_field(), L_SCALE_M)

    assert stats["keyhole_depth_um_true"] == pytest.approx(2.0)
    assert stats["keyhole_depth_um_pred"] == pytest.approx(4.0)
    assert stats["keyhole_depth_um_abs_err"] == pytest.approx(2.0)


def test_melt_above_the_surface_is_not_counted() -> None:
    """The gas layer (z > 0) is not substrate; melt there is a masking artifact."""
    fl = np.zeros(SHAPE)
    fl[0, 0, 1:5, 2:4, 5] = 1.0  # only the z > 0 slice

    stats = melt_geometry_stats({"fl": fl}, {"fl": fl}, _z_field(), L_SCALE_M)

    assert stats["melt_depth_um_true"] == pytest.approx(0.0)
    assert stats["melt_volume_um3_true"] == pytest.approx(0.0)


def test_geometry_crops_a_longer_z_field() -> None:
    """A prediction cropped to its truth overlap (A38) is shorter than z."""
    fl = np.zeros((1, 1, 6, 6, 4))
    # z center-crops to idx 1-4 (all substrate, surface last); melt in the two
    # slices below the surface is 2 um deep.
    fl[0, 0, 1:5, 2:4, 2:4] = 1.0

    stats = melt_geometry_stats({"fl": fl}, {"fl": fl}, _z_field(), L_SCALE_M)

    assert stats["melt_depth_um_true"] == pytest.approx(2.0)


class _CavityModel(torch.nn.Module):
    """Predicts a fixed alpha cavity two cells deeper than the truth's."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))  # gives the module a device

    def forward(self, invar):
        alpha = torch.ones_like(invar["z"])
        alpha[:, :, 2:4, 2:4, 1:5] = 0.0
        return {"alpha": alpha}


class _CavityDS(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.z = _z_field(SHAPE[1:])
        self.truth = np.ones(SHAPE[1:], dtype=np.float32)
        self.truth[0, 2:4, 2:4, 3:5] = 0.0  # depth 2

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i):
        invar = {"z": torch.from_numpy(self.z.copy())}
        return invar, {"alpha": torch.from_numpy(self.truth.copy())}, 0


def test_lazy_metrics_report_geometry(tmp_path) -> None:
    run_cfg = SimpleNamespace(
        inference_batch_size=1,
        material_params={"T_s": 1.0, "T_l": 2.0},
        ref_scales={"T": 1.0, "L": L_SCALE_M},
        alpha_masking=False,
    )

    errors = run_lazy_dataset_metrics(_CavityModel(), _CavityDS(), run_cfg, tmp_path, ["alpha"]).errors

    geometry = errors["geometry"]
    assert geometry["keyhole_depth_um_true"] == pytest.approx(2.0)
    assert geometry["keyhole_depth_um_pred"] == pytest.approx(4.0)
    assert geometry["keyhole_depth_um_abs_err"] == pytest.approx(2.0)
    assert "keyhole_depth_um_abs_err" in (tmp_path / "model_errors.txt").read_text()


def test_anisotropic_spacing_scales_each_axis():
    """A [2,2,1] stride leaves dz=10 um but dx=dy=20 um.

    Applying one z-derived cell size to all axes under-reported x/y lengths by 2x
    and volumes by 4x on every strided eval before 2026-08-04. Passing the x/y
    coordinate fields must recover the true scale.
    """
    import numpy as np
    from lpno.models.pipeline.postprocessing import melt_geometry_stats

    nx, ny, nz = 8, 8, 6
    zs = (np.arange(nz, dtype=np.float64) - (nz - 2)) * 1e-6      # dz = 1 um
    xs = np.arange(nx, dtype=np.float64) * 2e-6                    # dx = 2 um
    ys = np.arange(ny, dtype=np.float64) * 2e-6                    # dy = 2 um
    z = np.broadcast_to(zs[None, None, None, None, :], (1, 1, nx, ny, nz)).copy()
    x = np.broadcast_to(xs[None, None, :, None, None], (1, 1, nx, ny, nz)).copy()
    y = np.broadcast_to(ys[None, None, None, :, None], (1, 1, nx, ny, nz)).copy()

    fl = np.zeros((1, 1, nx, ny, nz))
    fl[0, 0, 2:6, 2:6, 0:3] = 1.0
    preds = {"fl": fl}
    truth = {"fl": fl}

    iso = melt_geometry_stats(preds, truth, z, 1.0)
    aniso = melt_geometry_stats(preds, truth, z, 1.0, x_field=x, y_field=y)

    assert aniso["melt_length_um_true"] == pytest.approx(2 * iso["melt_length_um_true"])
    assert aniso["melt_width_um_true"] == pytest.approx(2 * iso["melt_width_um_true"])
    assert aniso["melt_volume_um3_true"] == pytest.approx(4 * iso["melt_volume_um3_true"])
    assert aniso["melt_depth_um_true"] == pytest.approx(iso["melt_depth_um_true"]), "z unchanged"
    assert aniso["axes_isotropic"] is False
    assert aniso["cell_um_x"] == pytest.approx(2.0)
