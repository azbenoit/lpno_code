"""An eval must not book less VRAM than a grid-bound model needs.

Origin (2026-08-21): `varspeed_rescore_m47850` (calc 61877, exit 500) booked gpu_vram_mb=4000
against a model that trained at 11218, was admitted onto a card another job held 9.5 GiB of,
ran, and then died inside RIGNO's edge MLP asking for 3.86 GiB in one allocation with 3.16 GiB
free. The 4000 came from generalising a GAOT model that was running fine at that number.

Under-declaring and over-declaring are not symmetric. Over-declaring wastes queue time and then
works; under-declaring gets the job ADMITTED and loses both the seat time and the result.
"""
import pytest

from aiida_pino.utils.model_registry import _check_graph_eval_vram_against_training


def test_rigno_eval_below_its_training_booking_is_refused():
    with pytest.raises(ValueError, match=r"grid-bound|edge set"):
        _check_graph_eval_vram_against_training(
            {"config_name": "config_flow3D_RIGNO_randompath_onecycle", "gpu_vram_mb": 11218},
            {"gpu_vram_mb": 4000},
        )


def test_gaot_is_covered_too():
    with pytest.raises(ValueError):
        _check_graph_eval_vram_against_training(
            {"config_name": "config_flow3D_GAOT_varspeed_ctrl", "gpu_vram_mb": 20000},
            {"gpu_vram_mb": 4000},
        )


def test_meeting_the_training_booking_passes():
    _check_graph_eval_vram_against_training(
        {"config_name": "config_flow3D_RIGNO_randompath_onecycle", "gpu_vram_mb": 11218},
        {"gpu_vram_mb": 18000},
    )


def test_fno_is_deliberately_exempt():
    """FNO's footprint scales with the batch, so a small eval booking is legitimate.

    Steady FNO evals measured a 164 MiB peak against a 3000 MiB booking. Refusing those would
    be the over-strict mistake in the other direction -- the one that made a finished model
    unevaluable earlier the same day.
    """
    _check_graph_eval_vram_against_training(
        {"config_name": "config_flow3D_FNO_pvlhs_steady", "gpu_vram_mb": 16000},
        {"gpu_vram_mb": 3000},
    )


def test_a_model_with_no_recorded_training_booking_is_not_blocked():
    _check_graph_eval_vram_against_training(
        {"config_name": "config_flow3D_RIGNO_randompath"}, {"gpu_vram_mb": 1000}
    )


def test_time_averaged_steady_eval_is_exempt():
    """The exemption that keeps the guard from refusing a config that demonstrably works.

    Model 59503 is a RIGNO steady model: it TRAINED at 12000 MiB and its eval ran cleanly
    declared at 3000 (calc 61787, measured peak 164 MiB), because time_avg_window=30 collapses
    thirty frames into one sample. The two evals that actually OOMed were both
    time_avg_window=1. Without this branch the guard refuses working steady evals -- the
    over-strict direction, which this same file already got wrong once on 2026-08-21.
    """
    _check_graph_eval_vram_against_training(
        {"config_name": "config_flow3D_RIGNO_pvlhs_steady", "gpu_vram_mb": 12000},
        {"gpu_vram_mb": 3000, "time_avg_window": 30},
    )


def test_per_frame_eval_of_the_same_architecture_is_still_checked():
    """taw=1 is the regime that OOMed, so the guard must still bite there."""
    with pytest.raises(ValueError, match="per-frame"):
        _check_graph_eval_vram_against_training(
            {"config_name": "config_flow3D_RIGNO_randompath_onecycle", "gpu_vram_mb": 11218},
            {"gpu_vram_mb": 4000, "time_avg_window": 1},
        )


def test_eval_does_not_inherit_the_training_gpu_pin():
    """A training's card choice must not pin its evals.

    Measured 2026-08-21: three models all froze cuda_device="0", so every eval of the
    trainfill3520 trio inherited a pin to GPU 0. one eval waited 24 min behind an 18000 MiB
    reservation on that card; another (from a model with cuda_device="1") waited 71 min on the
    mirror image. The training's occupancy is not the eval's occupancy.
    """
    from unittest.mock import Mock

    from aiida_pino.utils.model_registry import merge_model_training_with_evaluation_profile

    frozen = {
        "config_name": "config_flow3D_FNO_pvlhs_steady", "run_name": "x", "cuda_device": "0",
        "wandb_offline": True, "hydra_overrides": [], "power_weighted_dose": True,
    }
    model = Mock()
    model.get_training_profile.return_value = dict(frozen)
    merged = merge_model_training_with_evaluation_profile(model, None)
    assert merged["cuda_device"] == "auto", "the inherited pin must be dropped"

    model.get_training_profile.return_value = dict(frozen)
    explicit = merge_model_training_with_evaluation_profile(model, {"cuda_device": "1"})
    assert explicit["cuda_device"] == "1", "an explicit eval-side device must still win"
