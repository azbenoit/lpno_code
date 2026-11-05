"""At eval time the dataset layout must follow the MODEL, not the dataset.

`EvaluateModelWorkChain.validate_inputs` defaulted the derivation profile to
`canonical_base_layout_profile()` (`time_as_dim=True`) whenever the caller gave
none. A model trained `time_as_batch=True` (the default for every FLOW-3D model
here) was then asked to evaluate against a `time_as_dim` variant of its test set:

- on a ragged dataset (cases with differing frame counts) that variant cannot
  exist at all, so the run died in `validate_inputs` with exit 303 -- observed
  live on `rigno_kh2579_w8` vs its test collection (2026-07-22);
- on a uniform dataset it *can* exist, which is worse: the eval rebuilds a
  different architecture than the checkpoint and fails at load (exit 302,
  two alsi_fno test evals).

The dataset variant should only decide which staged arrays to read. The layout
comes from the model's frozen training profile.
"""
from __future__ import annotations

from aiida_pino.utils import (
    build_derivation_profile_from_training_profile,
    canonical_base_layout_profile,
)
from aiida_pino.utils.profiles import DEFAULT_TRAINING_PROFILE


def _time_as_batch_profile():
    return {**DEFAULT_TRAINING_PROFILE, "time_as_batch": True, "time_as_dim": False,
            "include_time": True, "moving_ref_frame": False, "use_2d": False}


def test_time_as_batch_training_maps_to_its_own_layout():
    layout = build_derivation_profile_from_training_profile(_time_as_batch_profile())
    assert layout["time_as_batch"] is True
    assert layout["time_as_dim"] is False


def test_a_non_canonical_training_layout_is_not_replaced_by_canonical():
    """The invariant, independent of what canonical happens to be.

    Canonical is now time_as_batch, so the common case agrees with it by luck.
    A model trained some other way (time_as_dim, moving reference frame) must
    still be evaluated in ITS layout — that is what the workchain fix guarantees,
    and this test keeps holding if canonical ever moves again.
    """
    time_as_dim_profile = {**DEFAULT_TRAINING_PROFILE, "time_as_batch": False,
                           "time_as_dim": True, "include_time": True,
                           "moving_ref_frame": False}
    layout = build_derivation_profile_from_training_profile(time_as_dim_profile)
    assert layout["time_as_dim"] is True
    assert layout != canonical_base_layout_profile()


def test_moving_ref_frame_training_round_trips():
    """A non-default training layout is carried through unchanged too."""
    profile = {**DEFAULT_TRAINING_PROFILE, "time_as_batch": False, "time_as_dim": False,
               "include_time": True, "moving_ref_frame": True, "moving_frame_steps": 30,
               "time_avg_window": 30}
    layout = build_derivation_profile_from_training_profile(profile)
    assert layout["moving_ref_frame"] is True
    assert layout["moving_frame_steps"] == 30
    assert layout["time_avg_window"] == 30


def test_eval_workchain_defaults_the_layout_from_the_model():
    """The regression itself: no derivation_profile given -> use the model's."""
    import inspect

    from aiida_pino.workflows.evaluate_model import EvaluateModelWorkChain

    source = inspect.getsource(EvaluateModelWorkChain.validate_inputs)
    assert "build_derivation_profile_from_training_profile" in source, (
        "validate_inputs must derive the default layout from the model's training "
        "profile, not from canonical_base_layout_profile()"
    )


def test_env_is_an_allowed_evaluation_override():
    """Eval needs the same env switches as training (PINO_LAZY_STRIDE et al).

    Without this a model trained on a strided grid is evaluated on the full one:
    for RIGNO the graph is rebuilt from the coordinates, so it silently scores a
    different resolution than it trained on rather than raising.
    """
    from aiida_pino.utils.model_registry import (
        EVALUATION_OVERRIDE_KEYS,
        validate_evaluation_profile,
    )

    assert "env" in EVALUATION_OVERRIDE_KEYS
    profile = validate_evaluation_profile({"env": {"PINO_LAZY_STRIDE": "2,2,1"}})
    assert profile["env"] == {"PINO_LAZY_STRIDE": "2,2,1"}


def test_evaluation_env_must_be_a_mapping():
    import pytest

    from aiida_pino.utils.model_registry import validate_evaluation_profile

    with pytest.raises(ValueError):
        validate_evaluation_profile({"env": "PINO_LAZY_STRIDE=2,2,1"})
