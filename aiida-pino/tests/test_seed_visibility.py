"""The resolved seed is explicit in the launched config and the summary.

Premise correction first (2026-08-10): an omitted seed was never "unseeded" --
`config_loading.py` resolves `custom.seed -> seed -> 42` and `seed_all` applies
it to python/numpy/torch/cuda, and `extras.training_seed` already identifies
replicates on all 223 models. What WAS true: the archived `.hydra/config.yaml`
of a default run carries no seed key, and a training that never becomes a model
node has no seed record anywhere. Given seed swings peak-T by ~1700 K, the seed
is part of the result.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from aiida_pino.scripts.train_adapter import build_train_command, resolve_training_seed


def _cmd(profile):
    return build_train_command(
        {"config_name": "no_such_config", "run_name": "seed_probe",
         "wandb_offline": True, "k_folds": 1, "fold_index": 0,
         "pde_loss": False, "vof_pde_loss": False,
         "diff_loss": False, "diff_weight": 1.0, "pde_weight": 1.0, "include_time": False,
         "time_as_batch": False, "time_as_dim": False, "use_2d": False,
         "time_avg_window": 1, "moving_ref_frame": False, "moving_frame_steps": 1,
         "fast_training": False, "detect_anomaly": False, "sharp_alpha": False,
         "alpha_masking": False, "max_steps": 10, "learning_rate": 1e-3,
         "weight_decay": 0.0, "batch_size_grid": 1, "batch_size_validation": 1,
         "inference_batch_size": 1, "infer_velocities": False, **profile},
        Path("/t"), Path("/v"), Path("/o"),
    )


@pytest.mark.requires_lpno
def test_an_implicit_seed_becomes_explicit_in_the_command():
    command = _cmd({})
    assert "++custom.seed=42" in command, "the default run must say which seed it used"


def test_an_explicit_seed_is_never_second_guessed():
    """The user's seed VALUE survives, exactly once, and no auto-injection fights it.

    Updated for the replay promotion (705fb24): a stored `+custom.seed` becomes
    `++custom.seed` at command build, because bare `+` errors against any config that
    already declares the key -- the promoted form carries the SAME value, so the
    contract this test pins (explicit seed wins, nothing second-guesses it) is
    unchanged; only the prefix moved.
    """
    command = _cmd({"hydra_overrides": ["+custom.seed=1003"]})
    seeds = [o for o in command if "custom.seed=" in o]
    assert seeds == ["++custom.seed=1003"], seeds


@pytest.mark.requires_lpno
def test_resolution_chain_and_sources():
    assert resolve_training_seed({"seed": 7}) == (7, "training_profile")
    assert resolve_training_seed({"hydra_overrides": ["~seed=9"]}) == (9, "hydra_override")
    assert resolve_training_seed({}) == (42, "lpno_default")
    # an unparsable override must resolve to unresolved, never to a guess
    assert resolve_training_seed({"hydra_overrides": ["custom.seed=oops"]}) == (None, "unresolved")
