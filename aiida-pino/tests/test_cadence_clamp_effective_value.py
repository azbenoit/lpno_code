"""The cadence clamp reads the value the run will use, and keeps the final weights.

Two defects found while writing reproducibility/train_evaluate.ipynb:

* The clamp compared the CONFIG's cadence with `max_steps` and appended `key=max_steps`
  after the profile's overrides, so a finer cadence the caller asked for
  (`training.rec_validation_freq=50` in a 200-step run) was replaced by 200. It now
  lowers only a value that exceeds the run.
* `training.save_network_freq` was not clamped at all. physicsnemo writes the weights
  only every save_network_freq steps (default 1000), so a 200-step training kept its
  step-0 weights and its evaluation scored an untrained model.
"""
from __future__ import annotations

import pytest

from aiida_pino.scripts.train_adapter import append_recording_cadence_clamp, build_train_command

CONFIG = "config_flow3D_RIGNO_pvlhs"          # declares the three validator keys at 2000
STEADY = "config_flow3D_RIGNO_pvlhs_steady"   # inherits them; no save_network_freq anywhere


def _last(command, key):
    """Hydra's last-wins value of `key` in `command`, or None."""
    value = None
    for item in command:
        k, sep, v = str(item).partition("=")
        if sep and k.lstrip("+") == key:
            value = v
    return value


def _clamped(config_name, max_steps, overrides=()):
    command = list(overrides)
    append_recording_cadence_clamp(command, {"config_name": config_name, "max_steps": max_steps})
    return command


@pytest.mark.requires_lpno
def test_a_finer_user_cadence_is_kept():
    command = _clamped(CONFIG, 200, ["training.rec_validation_freq=50",
                                     "training.rec_results_freq=50"])
    assert _last(command, "training.rec_validation_freq") == "50"
    assert _last(command, "training.rec_results_freq") == "50"
    # the config's 2000-step stop interval is still lowered, to a multiple of 50
    assert _last(command, "stop_criterion.freq") == "200"


@pytest.mark.requires_lpno
def test_a_coarse_user_cadence_is_still_lowered():
    command = _clamped(CONFIG, 200, ["training.rec_validation_freq=5000"])
    assert _last(command, "training.rec_validation_freq") == "200"


@pytest.mark.requires_lpno
def test_stop_interval_follows_a_validation_cadence_it_no_longer_divides():
    command = _clamped(CONFIG, 200, ["training.rec_validation_freq=60"])
    assert _last(command, "training.rec_validation_freq") == "60"
    assert int(_last(command, "stop_criterion.freq")) % 60 == 0


@pytest.mark.requires_lpno
def test_save_interval_is_capped_to_a_short_run():
    command = _clamped(STEADY, 200)
    assert _last(command, "training.save_network_freq") == "200"


@pytest.mark.requires_lpno
def test_save_interval_a_caller_set_within_the_run_is_kept():
    assert _last(_clamped(STEADY, 200, ["training.save_network_freq=50"]),
                 "training.save_network_freq") == "50"
    assert _last(_clamped(STEADY, 200, ["training.save_network_freq=5000"]),
                 "training.save_network_freq") == "200"
    assert _last(_clamped(STEADY, 32000), "training.save_network_freq") is None


def test_an_unknown_config_gets_no_save_clamp():
    assert _clamped("no_such_config_exists", 200) == []


@pytest.mark.requires_lpno
def test_the_built_command_keeps_the_callers_cadence_and_saves_the_final_step():
    """What hydra receives for the notebook's route-B profile: 200 steps, 50-step records."""
    from aiida_pino.utils.profiles import merge_training_profile
    from pathlib import Path

    profile = merge_training_profile({
        "config_name": STEADY, "run_name": "r", "cuda_device": "auto", "wandb_offline": True,
        "max_steps": 200,
        "hydra_overrides": ["training.rec_validation_freq=50", "stop_criterion.freq=50"],
    })
    command = build_train_command(profile, Path("train"), Path("val"), Path("out"))
    assert _last(command, "training.max_steps") == "200"
    assert _last(command, "training.rec_validation_freq") == "50"
    assert _last(command, "stop_criterion.freq") == "50"
    assert _last(command, "training.rec_results_freq") == "200"
    assert _last(command, "training.save_network_freq") == "200"
