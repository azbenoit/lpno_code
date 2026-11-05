"""An eval override must ADD to the training's hydra overrides, not replace them.

2026-08-21. Six evals of an mref A/B test died at
`_assert_eval_checkpoint_loadable` with a state_dict shape mismatch on the FNO
lift layer: checkpoints built for 20 and 15 channels loaded into models built
with 19, the `config_flow3D_FNO_pvlhs` default.

Both models were trained with explicit `++custom.FNO_input_keys=[...]` lists,
and both frozen training profiles still carry them. The evals dropped them --
not because inheritance is missing, but because
`merge_model_training_with_evaluation_profile` did `merged[key] = value` for
`hydra_overrides`, replacing the whole list. Measured on the live graph:

    eval A     (exit 300)  evaluation_profile.hydra_overrides =
                             ["++custom.T_boil_k=3520.0"]        -> FAILED
    eval B     (exit 0)    evaluation_profile.hydra_overrides = None -> inherited

So ONE unrelated override -- a boiling temperature -- silently discarded all
eleven of the training's, the architecture spec among them. The reported
symptom was "the eval does not inherit FNO_input_keys"; the defect is that ANY
eval override discards EVERY training override.

The loud failure was the lucky case. A non-default key list that happens to
produce the SAME channel count as the default loads without complaint and
returns a plausible wrong number, which is the case these tests exist to make
impossible to reach by accident.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.model_registry import merge_model_training_with_evaluation_profile

_KEYS = "++custom.FNO_input_keys=[V_scan,P,H_norm,x,y,z]"

BASE = {
    "config_name": "config_flow3D_FNO_pvlhs",
    "run_name": "x",
    "cuda_device": "auto",
    "wandb_offline": True,
    "hydra_overrides": [_KEYS, "++custom.seed=42"],
}


class _Model:
    """Only `get_training_profile` is used by the merge."""

    def __init__(self, profile=None):
        self._profile = dict(profile or BASE)

    def get_training_profile(self):
        return dict(self._profile)


def test_an_unrelated_eval_override_does_not_discard_the_architecture():
    """The exact shape of the six failures."""
    merged = merge_model_training_with_evaluation_profile(
        _Model(), {"hydra_overrides": ["++custom.T_boil_k=3520.0"]}
    )
    overrides = merged["hydra_overrides"]

    assert _KEYS in overrides, (
        "the training's FNO_input_keys determines the ARCHITECTURE; an unrelated "
        "eval override must not drop it"
    )
    assert "++custom.seed=42" in overrides, "every training override survives, not just the famous one"
    assert "++custom.T_boil_k=3520.0" in overrides, "the eval's own override must still apply"


def test_an_eval_override_of_the_same_key_still_wins():
    """Inheriting must not become un-overridable: hydra takes the LAST value."""
    merged = merge_model_training_with_evaluation_profile(
        _Model(), {"hydra_overrides": ["++custom.seed=7"]}
    )
    overrides = merged["hydra_overrides"]

    assert overrides.index("++custom.seed=7") > overrides.index("++custom.seed=42"), (
        "the eval's value must come after the training's so hydra's last-wins "
        "resolution gives the eval precedence"
    )


def test_no_eval_overrides_still_inherits_everything():
    """The path that accidentally worked (eval B above) must keep working."""
    merged = merge_model_training_with_evaluation_profile(_Model(), None)
    assert merged["hydra_overrides"] == [_KEYS, "++custom.seed=42"]


def test_a_duplicate_override_is_not_repeated():
    """Re-stating the training's own value must not double the entry."""
    merged = merge_model_training_with_evaluation_profile(
        _Model(), {"hydra_overrides": [_KEYS]}
    )
    assert merged["hydra_overrides"].count(_KEYS) == 1


def test_a_training_with_no_overrides_is_unaffected():
    merged = merge_model_training_with_evaluation_profile(
        _Model({**BASE, "hydra_overrides": []}), {"hydra_overrides": ["++custom.a=1"]}
    )
    assert merged["hydra_overrides"] == ["++custom.a=1"]
