"""power_weighted_dose is stamped per model, so the default can be ON safely.

Design (2026-08-14): dose channels should carry power by default on
every NEW training, but an already-trained model's evals must reproduce the
inputs it trained on. The profile key is tri-state:

- ``None`` (unset) in a NEW training submission -> stamped ``True`` by
  ``merge_training_profile`` and frozen into the model's training profile: the
  per-model boolean marker.
- ``None`` in a model's FROZEN profile (trained before the stamp existed) ->
  pinned ``False`` at eval merge: every pre-stamp model trained power-blind, and
  re-evals must not pick up the new default (or the edited pvlhs yaml) silently.
- an explicit value anywhere -> preserved.
"""

import pytest

from aiida_pino.utils.model_registry import merge_model_training_with_evaluation_profile
from aiida_pino.utils.profiles import merge_training_profile


def test_new_training_is_stamped_on_by_default():
    merged = merge_training_profile({"config_name": "config_flow3D_GAOT_pvlhs", "run_name": "r",
                                     "hydra_overrides": [], "cuda_device": "auto", "wandb_offline": True})
    assert merged["power_weighted_dose"] is True


def test_explicit_off_survives_the_merge():
    merged = merge_training_profile({"config_name": "config_flow3D_GAOT_pvlhs", "run_name": "r",
                                     "hydra_overrides": [], "cuda_device": "auto", "wandb_offline": True,
                                     "power_weighted_dose": False})
    assert merged["power_weighted_dose"] is False


def test_import_of_a_finished_run_is_not_stamped():
    # Importing an existing run records what happened; inventing True would be
    # fabricated provenance (same rule as IMPORT_PROFILE_REQUIRED_KEYS).
    merged = merge_training_profile(
        {"config_name": "config_flow3D_GAOT_pvlhs", "run_name": "r", "hydra_overrides": []},
        for_import=True,
    )
    assert merged["power_weighted_dose"] is None


@pytest.mark.usefixtures("aiida_profile_clean")
def test_eval_of_prestamp_model_pins_off(pino_dataset_factory, pino_model_factory):
    dataset = pino_dataset_factory("pwd-old")
    model = pino_model_factory("pwd-old", training_source=dataset)
    frozen = model.get_training_profile()
    assert frozen.get("power_weighted_dose") is None, "fixture must model a pre-stamp profile"

    merged = merge_model_training_with_evaluation_profile(
        model, {"run_name": "eval"}, dataset_metadata=dataset.get_dataset_metadata()
    )
    assert merged["power_weighted_dose"] is False


@pytest.mark.usefixtures("aiida_profile_clean")
def test_eval_of_stamped_model_inherits_its_marker(pino_dataset_factory, pino_model_factory):
    dataset = pino_dataset_factory("pwd-new")
    model = pino_model_factory("pwd-new", training_source=dataset)
    stamped = dict(model.get_training_profile())
    for value in (True, False):
        stamped["power_weighted_dose"] = value

        class _Stamped:  # the merge reads only get_training_profile
            def get_training_profile(self, _p=dict(stamped)):
                return dict(_p)

        merged = merge_model_training_with_evaluation_profile(
            _Stamped(), {"run_name": "eval"}, dataset_metadata=dataset.get_dataset_metadata()
        )
        assert merged["power_weighted_dose"] is value


def test_adapter_emits_override_only_when_stamped():
    from aiida_pino.scripts.train_adapter import append_runtime_overrides

    base = merge_training_profile({"config_name": "config_flow3D_GAOT_pvlhs", "run_name": "r",
                                   "hydra_overrides": [], "cuda_device": "auto", "wandb_offline": True})
    cmd: list[str] = []
    append_runtime_overrides(cmd, base, split_mode="standard")
    # MUST be the ++ force-add form: most configs do not declare the key in their
    # custom struct, and a bare custom.X= override crashes Hydra compose there
    # (a RIGNO onecycle eval died exactly this way on 2026-08-14).
    assert any(c == "++custom.power_weighted_dose=true" for c in cmd)

    cmd = []
    append_runtime_overrides(cmd, dict(base, power_weighted_dose=None), split_mode="standard")
    assert not any("power_weighted_dose" in c for c in cmd)
