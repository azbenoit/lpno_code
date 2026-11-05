"""The console must show the step budget Hydra will use, not the one asked for.

`train_adapter` emits `training.max_steps=<profile['max_steps']>` and THEN
appends `hydra_overrides` verbatim, so a profile that carries its own
`training.max_steps=` override puts the key on the command line twice. Hydra
applies overrides in order, so the LAST one wins and the structured field is
dead weight.

Measured on one run (`gaot_1601_fullres_35k`, 2026-08-07): the profile field
said 20000, override 49 said 35000, Hydra's resolved `config.yaml` said 35000,
and the trainer sized its one-cycle at `total_steps: 35200`. The console read
the structured field alone and drew `step 19900 / 20000` -- a bar about to
report 100% on a run with 15100 steps left.

The stakes are higher than a wrong bar. `_sync_one_cycle_total_steps` derives
the one-cycle length from whichever `max_steps` wins, so a requeue that trusts
the structured field would build a 20200-step cosine and load a step-19900
scheduler state into it: the anneal would be over before the run was, and
OneCycleLR raises "Tried to step N+1 times" past `total_steps`.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.profiles import effective_max_steps


def test_override_wins_over_the_structured_field():
    """Run 31934's exact shape: field 20000, override 35000, Hydra ran 35000."""
    profile = {"max_steps": 20000, "hydra_overrides": ["training.max_steps=35000"]}
    assert effective_max_steps(profile) == 35000


def test_structured_field_is_used_when_no_override_touches_it():
    profile = {"max_steps": 20000, "hydra_overrides": ["optimizer.lr=0.0003"]}
    assert effective_max_steps(profile) == 20000


def test_last_override_wins_when_the_key_repeats():
    """Hydra applies overrides in order; so must we."""
    profile = {
        "max_steps": 20000,
        "hydra_overrides": ["training.max_steps=30000", "training.max_steps=35000"],
    }
    assert effective_max_steps(profile) == 35000


@pytest.mark.parametrize("prefix", ["", "+", "++"])
def test_hydra_append_and_force_prefixes_are_honoured(prefix):
    """`+key=v` and `++key=v` set the same key and must not be missed."""
    profile = {"max_steps": 20000, "hydra_overrides": [f"{prefix}training.max_steps=35000"]}
    assert effective_max_steps(profile) == 35000


def test_a_similarly_named_key_is_not_mistaken_for_it():
    """Substring matching would catch these; the budget must not move."""
    profile = {
        "max_steps": 20000,
        "hydra_overrides": [
            "custom.autoregressive.max_steps=99",
            "training.max_steps_warmup=77",
            "stop_criterion.patience=16000",
        ],
    }
    assert effective_max_steps(profile) == 20000


def test_malformed_override_falls_back_to_the_field():
    """A guess is worse than the declared value; never crash the Runs list."""
    profile = {"max_steps": 20000, "hydra_overrides": ["training.max_steps=notanumber"]}
    assert effective_max_steps(profile) == 20000


def test_missing_pieces_do_not_raise():
    assert effective_max_steps({}) is None
    assert effective_max_steps({"max_steps": 15000}) == 15000
    assert effective_max_steps({"hydra_overrides": ["training.max_steps=500"]}) == 500
    assert effective_max_steps({"max_steps": None, "hydra_overrides": None}) is None


def test_non_list_overrides_are_tolerated():
    """Profiles are user data; a malformed field must not break the console."""
    assert effective_max_steps({"max_steps": 20000, "hydra_overrides": "training.max_steps=1"}) == 20000
