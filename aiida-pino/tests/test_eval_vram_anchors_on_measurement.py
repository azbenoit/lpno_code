"""The grid-bound eval guard must anchor on what was MEASURED, not what was booked.

2026-08-24. `_check_graph_eval_vram_against_training` refused an eval
whose `gpu_vram_mb` was below the training profile's DECLARED booking, and never read
`training_summary.gpu_vram_peak_mb`.

Measured on two models identical in architecture, config (`config_flow3D_GAOT_pvlhs`),
dataset, channels and steps, with the SAME 5004 MiB training peak:

    66267  declared  6000, peak 5004  -> eval allowed at 8000, ran, actual peak 1514 MiB
    66067  declared 14000, peak 5004  -> eval refused at 8000, forced to 14000, stuck

66067's 14000 was hand-set to skip the whole-card auto-size smoke (its workchain label
says so), and hand-setting for that reason is documented practice -- so the guard was
enforcing a number that was never a measurement, against an eval whose real footprint on
the sibling model was ~9x smaller.

The guard's own docstring already argues for the fix: the quantity that OOMed calc 61877
is the edge-set/encoder graph footprint, and its 4000 was wrong *because it was
generalised rather than measured*.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.model_registry import _check_graph_eval_vram_against_training as check

GAOT = {"config_name": "config_flow3D_GAOT_pvlhs", "gpu_vram_mb": 14000}


def test_a_hand_set_booking_no_longer_blocks_an_eval_the_measurement_allows():
    """The reported case: declared 14000, measured 5004, eval at 8000."""
    check(GAOT, {"gpu_vram_mb": 8000, "time_avg_window": 1}, 5004)


def test_the_measured_peak_is_still_a_floor():
    """Anchoring on measurement must not mean anchoring on nothing.

    CHANGED 2026-09-06 (user-approved): the floor is now ENFORCED BY CORRECTION rather than
    by refusal. The assertion that matters is unchanged -- the eval may not proceed below the
    measured peak -- but the remedy is to raise the booking to it instead of costing a whole
    submission round to deliver a number the guard already quoted.
    """
    merged = {"gpu_vram_mb": 3000, "time_avg_window": 1}
    check(GAOT, merged, 5004)
    assert merged["gpu_vram_mb"] == 5004


def test_the_motivating_oom_cannot_run_under_booked():
    """calc 61877: a grid-bound model booked 4000 and needed 3.86 GiB in one allocation.

    This test's tripwire was "if this ever passes, the fix has been widened into the failure
    it was meant to keep out", written when the only alternative to refusing was ADMITTING the
    under-booked job. Correcting the booking is a third option the tripwire did not consider:
    the job still never runs at 4000, it runs at the measured 9000. So the invariant is
    asserted directly rather than through the exception -- what must never happen is the job
    proceeding with less VRAM than was measured, and that is what is checked here.
    """
    rigno = {"config_name": "config_flow3D_RIGNO_pvlhs", "gpu_vram_mb": 12000}
    merged = {"gpu_vram_mb": 4000, "time_avg_window": 1}
    check(rigno, merged, 9000)
    assert merged["gpu_vram_mb"] >= 9000, "the eval could still run under-booked"


def test_without_a_measured_peak_it_falls_back_to_the_booking():
    """Older models record no peak; they must keep the previous behaviour, not lose the guard."""
    with pytest.raises(ValueError) as excinfo:
        check(GAOT, {"gpu_vram_mb": 8000, "time_avg_window": 1}, None)
    assert "14000" in str(excinfo.value)
    assert "no measured peak recorded" in str(excinfo.value)


def test_a_junk_peak_does_not_disable_the_guard():
    """A non-numeric peak must fall back, never silently allow everything."""
    with pytest.raises(ValueError):
        check(GAOT, {"gpu_vram_mb": 8000, "time_avg_window": 1}, "not-a-number")


def test_the_time_averaged_exemption_still_applies():
    """Unchanged and load-bearing: a steady eval collapses frames and genuinely needs less."""
    check(GAOT, {"gpu_vram_mb": 3000, "time_avg_window": 30}, 5004)


def test_fno_is_still_not_covered():
    """FNO's footprint does scale with batch; refusing those is the other over-strict error."""
    fno = {"config_name": "config_flow3D_FNO_pvlhs_steady", "gpu_vram_mb": 12000}
    check(fno, {"gpu_vram_mb": 164, "time_avg_window": 1}, 9000)


def test_a_model_without_retrievable_metadata_falls_back_instead_of_raising():
    """Reading the peak must never be able to BLOCK a merge that used to work.

    Committed 2026-08-24 as a hard `model.get_model_metadata()` call, which broke six
    tests whose stub models implement only `get_training_profile` -- and would have done
    the same to any caller holding a lighter model object. The fallback to the declared
    booking is the documented behaviour when no measurement exists; an unreachable
    accessor is just another way of having no measurement.
    """
    from aiida_pino.utils.model_registry import merge_model_training_with_evaluation_profile

    class _NoMetadata:
        def get_training_profile(self):
            # Full required-key set: the frozen profile is validated before the guard
            # runs, so a minimal stub fails on that instead and tests nothing.
            return {
                "config_name": "config_flow3D_GAOT_pvlhs",
                "gpu_vram_mb": 14000,
                "run_name": "x",
                "hydra_overrides": [],
                "cuda_device": "auto",
                "wandb_offline": True,
                # Per-frame. This config DEFAULTS to time_avg_window=30, and a
                # time-averaged eval is exempt by design -- leave it and the guard exits
                # early, so the test would pass while exercising nothing.
                "time_avg_window": 1,
            }

    class _RaisingMetadata(_NoMetadata):
        def get_model_metadata(self):
            raise RuntimeError("metadata unavailable")

    for model in (_NoMetadata(), _RaisingMetadata()):
        with pytest.raises(ValueError) as excinfo:
            merge_model_training_with_evaluation_profile(model, {"gpu_vram_mb": 8000})
        # falls back to the booking, and says so, rather than raising AttributeError
        assert "no measured peak recorded" in str(excinfo.value)
