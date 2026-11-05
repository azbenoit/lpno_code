"""Training-profile defaults are read from the config as hydra composes it.

`load_training_profile_defaults` read the named YAML alone. The `*_pvlhs_steady`
configs set neither the step budget nor the batch size themselves, so
`aiida-pino profile-keys training --config config_flow3D_RIGNO_pvlhs_steady` reported
15000 steps and batch 20 (this module's fallbacks) where the composed config has
32000 and 1. The train adapter emits these keys on every run, so a submission that
omitted them trained at 15000 steps, batch 20. Found writing
reproducibility/train_evaluate.ipynb. The learning rate was already chain-resolved
(`resolve_effective_lr`); the rest now are too.
"""
from __future__ import annotations

import pytest

from aiida_pino.utils.profiles import load_training_profile_defaults, merge_training_profile

STEADY = ("config_flow3D_FNO_pvlhs_steady", "config_flow3D_RIGNO_pvlhs_steady",
          "config_flow3D_GAOT_pvlhs_steady")


@pytest.mark.requires_lpno
@pytest.mark.parametrize("config_name", STEADY)
def test_steady_configs_report_their_inherited_budget_and_batch(config_name):
    defaults = load_training_profile_defaults(config_name)
    assert defaults["max_steps"] == 32000
    assert defaults["batch_size_grid"] == 1
    assert defaults["batch_size_validation"] == 1
    assert defaults["inference_batch_size"] == 1
    assert defaults["learning_rate"] == 1e-4


@pytest.mark.requires_lpno
def test_a_profile_that_omits_them_trains_at_the_configs_values():
    merged = merge_training_profile({
        "config_name": "config_flow3D_RIGNO_pvlhs_steady", "run_name": "r",
        "cuda_device": "auto", "wandb_offline": True, "hydra_overrides": [],
    })
    assert (merged["max_steps"], merged["batch_size_grid"]) == (32000, 1)


@pytest.mark.requires_lpno
def test_a_value_set_in_the_named_file_still_wins_over_its_parents():
    # config_flow3D_GAOT_pvlhs sets max_steps 32000 over its parent chain's 8000
    assert load_training_profile_defaults("config_flow3D_GAOT_pvlhs")["max_steps"] == 32000
    assert load_training_profile_defaults("config_flow3D_GAOT_randompath")["max_steps"] == 8000


@pytest.mark.requires_lpno
def test_the_layout_keys_are_not_inherited():
    """A steady config must not inherit its random-path parents' lab-frame layout.

    The composed `*_pvlhs_steady` configs carry `time_as_batch: true` and
    `moving_ref_frame: false` from the random-path chain. As defaults those would turn
    a steady submission that omits its layout into a lab-frame training that no layout
    guard notices, so the layout keys keep the named file's values (none here) over
    the packaged steady layout.
    """
    defaults = load_training_profile_defaults("config_flow3D_RIGNO_pvlhs_steady")
    assert defaults["time_as_batch"] is False
    assert defaults["moving_ref_frame"] is True
    assert (defaults["time_avg_window"], defaults["moving_frame_steps"]) == (30, 30)


@pytest.mark.requires_lpno
def test_the_identity_and_launch_keys_are_not_inherited():
    """A parent's `custom.name` and `cuda_device: 0` must not become this config's defaults.

    The FNO steady config inherits from a RIGNO config, whose `name` would label the run,
    and a blank `cuda_device` is what `resolve_cuda_device_for_target` reads as "auto".
    """
    defaults = load_training_profile_defaults("config_flow3D_FNO_pvlhs_steady")
    assert defaults["run_name"] == "unnamed"
    assert defaults["cuda_device"] == ""


def test_an_unknown_config_falls_back_to_the_packaged_defaults():
    defaults = load_training_profile_defaults("no_such_config_exists")
    assert (defaults["max_steps"], defaults["batch_size_grid"]) == (15000, 20)
