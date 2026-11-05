"""Startup guard for the recording/validation/stop-criterion cadence.

Every one of these mistakes was actually made in the 2026-08-11 PV-LHS launches, and
each one either destroyed work or wasted hours because it surfaced late or not at all:

  * rec_results_freq raised past max_steps -> the constraint/monitor/inference records
    it drives are never written. (Not the checkpoint interval, as this file once said:
    physicsnemo saves weights every training.save_network_freq steps, which
    cap_save_network_freq keeps within the run -- tests at the end.)
  * rec_validation_freq changed without stop_criterion.freq -> physicsnemo raises
    "Stop criterion frequency should be a multiple of the validation frequency", but
    only at the FIRST criterion evaluation, ~2 h into the run.
  * validation freq past max_steps would mean checkpoint selection has zero points.

The guard runs at config-load time, so all of them fail in seconds, before a seat is
taken or a graph is built.
"""
import pytest
from omegaconf import OmegaConf

from lpno.models.pipeline.config_loading import validate_recording_cadence


def _cfg(max_steps=32000, results=2000, validation=2000, stop_freq=2000):
    return OmegaConf.create({
        'training': {
            'max_steps': max_steps,
            'rec_results_freq': results,
            'rec_validation_freq': validation,
        },
        'stop_criterion': {'freq': stop_freq, 'patience': max_steps},
    })


def test_healthy_cadence_passes():
    validate_recording_cadence(_cfg())


def test_records_past_the_run_are_rejected():
    """rec_results_freq > max_steps means its records are never written."""
    with pytest.raises(ValueError, match='rec_results_freq'):
        validate_recording_cadence(_cfg(results=33000))


def test_no_validation_in_run_is_rejected():
    with pytest.raises(ValueError, match='validation'):
        validate_recording_cadence(_cfg(validation=40000))


def test_stop_criterion_mismatch_fails_at_startup_not_two_hours_in():
    """The exact wave-1 failure: stop freq 500, validation 2000."""
    with pytest.raises(ValueError, match='stop_criterion'):
        validate_recording_cadence(_cfg(stop_freq=500))


def test_stop_criterion_multiple_is_fine():
    validate_recording_cadence(_cfg(stop_freq=4000))


def test_absent_blocks_are_tolerated():
    """Configs without a stop_criterion (or the freq keys) must not crash the guard."""
    validate_recording_cadence(OmegaConf.create({'training': {'max_steps': 1000}}))


def test_eval_mode_is_not_subject_to_training_cadence():
    """An eval never trains, so it cannot violate any of these invariants.

    Reported 2026-08-25: eval 68075 (model 63797 `fno_rp2` on ds6675) exited 300
    on `stop_criterion.freq=1000 is not a multiple of rec_validation_freq=2000`,
    after the model had loaded and the geometry resolved. The check runs from
    `load_cfg_and_paths`, which is on EVERY entry to the pipeline including
    `NS_Energy_laser_PINO_eval`.

    Both fixed-length random-path models archive the offending pair
    (`rigno_rp3` 63795 and `fno_rp2` 63797), so every evaluation of either was
    refused over a setting that cannot affect a score -- and it presents as a
    broken model rather than a scope bug. The models' own earlier evals (63801,
    63852) predate the check.

    All three invariants are about a run that trains: a checkpoint it can be
    salvaged from, validation rows to select on, and a stop criterion it will
    reach. None applies in eval.
    """
    bad = _cfg(stop_freq=1000, validation=2000)
    with pytest.raises(ValueError, match='stop_criterion'):
        validate_recording_cadence(bad, run_mode='train')
    validate_recording_cadence(bad, run_mode='eval')  # must not raise


def test_eval_mode_skips_the_other_two_invariants_too():
    validate_recording_cadence(_cfg(results=33000), run_mode='eval')
    validate_recording_cadence(_cfg(validation=40000), run_mode='eval')
