"""A `beambox_*` stop metric that validation can never report falls back to `mae_T`.

The steady configs inherit `stop_criterion.metric: beambox_mae_T`. Validation reports
it only when a beam box is buildable (a `beam_dist` footprint, the beam centre, or a
beam-following frame), so a steady run on `[V_scan, P, H_norm, x, y, z]` with
`custom.moving_ref_frame=false` died at its first validation in physicsnemo's stop
criterion (found with reproducibility/train_evaluate.ipynb).
"""
from __future__ import annotations

from omegaconf import OmegaConf

from lpno.models.pipeline.config_loading import (
    fall_back_from_unbuildable_beambox_stop_metric as fall_back,
)

STEADY_KEYS = ["V_scan", "P", "H_norm", "x", "y", "z", "t_hat"]


def _cfg(metric="beambox_mae_T", **custom):
    return OmegaConf.create({"stop_criterion": {"metric": metric, "freq": 50},
                             "custom": dict(custom)})


def test_lab_frame_without_beam_channels_falls_back(capsys):
    cfg = _cfg()
    assert fall_back(cfg, STEADY_KEYS, moving_ref_frame=False) == "mae_T"
    assert cfg.stop_criterion.metric == "mae_T"
    assert "beambox_mae_T -> mae_T" in capsys.readouterr().out


def test_a_buildable_box_keeps_the_beambox_metric():
    for keys, frame in ((STEADY_KEYS, True), (STEADY_KEYS + ["beam_dist"], False),
                        (STEADY_KEYS + ["beam_x", "beam_y"], False)):
        cfg = _cfg()
        assert fall_back(cfg, keys, moving_ref_frame=frame) is None
        assert cfg.stop_criterion.metric == "beambox_mae_T"


def test_disabled_box_validation_falls_back_too():
    cfg = _cfg(validation_beambox=False)
    assert fall_back(cfg, STEADY_KEYS, moving_ref_frame=True) == "mae_T"


def test_other_metrics_and_eval_are_untouched():
    cfg = _cfg(metric="mae_T")
    assert fall_back(cfg, STEADY_KEYS, moving_ref_frame=False) is None
    cfg = _cfg(metric="beambox_rmse_T")
    assert fall_back(cfg, STEADY_KEYS, moving_ref_frame=False) == "rmse_T"
    cfg = _cfg()
    assert fall_back(cfg, STEADY_KEYS, moving_ref_frame=False, run_mode="eval") is None
    assert cfg.stop_criterion.metric == "beambox_mae_T"
