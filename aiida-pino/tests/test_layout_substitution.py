"""The dataset you select must be the dataset you train on, or the run says so.

When a training profile omits a layout key, `load_training_profile_defaults`
fills it from the packaged Hydra config — a file in *another repository*. When
that disagrees with the dataset in front of the user, the workchain used to
build a different dataset and train on it, silently.

Measured on the authors' database 2026-08-06: **158 of 399** `PinoTrainCalculation`s trained
on a node other than the one their workchain was handed, and **68 of 158**
models record a training source the human did not select. Runs 31185 and 31186
picked a no-reference-frame split, got a moving-frame 30-frame-averaged
121x100x40 crop built 2 s after submit, and died 5 h 45 min later with
`KeyError: 'beam_x'` — the config asked for lab-frame beam channels a
moving-frame view never builds. 11 h 30 min of GPU on a layout nobody chose.

The whole log of the affected workchain was one line, naming the submitted job.
"""

from __future__ import annotations

import pytest
from aiida import orm
from aiida.engine import run_get_node

from aiida_pino.workflows.train_model import TrainModelWorkChain


def _reports(node) -> str:
    return "\n".join(log.message for log in orm.Log.collection.get_logs_for(node))


def _profile(**overrides) -> dict:
    """A training profile that names its layout explicitly."""
    profile = {
        "config_name": "config_NS_energy_PINO_flow3D",
        "run_name": "layout-guard",
        "cuda_device": "",
        "hydra_overrides": [],
        "wandb_offline": True,
        "max_steps": 10,
    }
    profile.update(overrides)
    return profile


def test_a_layout_the_user_did_not_choose_is_refused(
    aiida_profile_clean, fake_python_executable, local_code_factory, pino_dataset_factory
):
    """No explicit derivation_profile + a disagreeing config = refusal, not substitution."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-layout-1")
    dataset = pino_dataset_factory("layout-selected")

    # The config's layout differs from the dataset's own on `moving_ref_frame`,
    # which is exactly what the two GAOT runs hit.
    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=_profile(moving_ref_frame=True, moving_frame_steps=30, time_avg_window=30)),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )

    assert node.exit_status == TrainModelWorkChain.exit_codes.ERROR_UNDECLARED_LAYOUT_SUBSTITUTION.status
    reported = _reports(node)
    assert "undeclared layout substitution" in reported.lower()
    # The message must name the disagreement and both sides, or it is unactionable.
    assert "moving_ref_frame" in reported
    assert str(dataset.pk) in reported
    assert "derivation_profile" in reported, "must say how to proceed deliberately"


def test_a_matching_dataset_is_not_refused(
    aiida_profile_clean, fake_python_executable, local_code_factory, pino_dataset_factory
):
    """The guard must not fire when the selected dataset already matches."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-layout-2")
    dataset = pino_dataset_factory("layout-matching")
    layout = dataset.get_dataset_metadata().get("derivation_profile") or {}

    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=_profile(**{
            key: layout[key]
            for key in ("time_as_dim", "time_as_batch", "include_time", "moving_ref_frame")
            if key in layout
        })),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.exit_status != TrainModelWorkChain.exit_codes.ERROR_UNDECLARED_LAYOUT_SUBSTITUTION.status
    assert "undeclared layout substitution" not in _reports(node).lower()


def test_an_explicit_derivation_profile_states_the_intent(
    aiida_profile_clean, fake_python_executable, local_code_factory, pino_dataset_factory
):
    """Passing the layout explicitly is how a caller says the substitution is meant."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-layout-3")
    dataset = pino_dataset_factory("layout-explicit")
    profile = _profile(moving_ref_frame=True, moving_frame_steps=30, time_avg_window=30)

    from aiida_pino.utils import build_derivation_profile_from_training_profile
    from aiida_pino.utils.profiles import validate_training_profile

    explicit = build_derivation_profile_from_training_profile(
        validate_training_profile(profile, expected_split_mode="standard")
    )
    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=profile),
        derivation_profile=orm.Dict(dict=explicit),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.exit_status != TrainModelWorkChain.exit_codes.ERROR_UNDECLARED_LAYOUT_SUBSTITUTION.status


def test_a_substitution_that_does_happen_is_named_in_the_log(
    aiida_profile_clean, fake_python_executable, local_code_factory, pino_dataset_factory
):
    """Even when intended, the run must say which node it actually trained on."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-layout-4")
    dataset = pino_dataset_factory("layout-logged")
    profile = _profile(moving_ref_frame=True, moving_frame_steps=30, time_avg_window=30)

    from aiida_pino.utils import build_derivation_profile_from_training_profile
    from aiida_pino.utils.profiles import validate_training_profile

    explicit = build_derivation_profile_from_training_profile(
        validate_training_profile(profile, expected_split_mode="standard")
    )
    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=profile),
        derivation_profile=orm.Dict(dict=explicit),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    # No skip. The skip condition WAS the absence of the feature's output, so
    # deleting the `self.report` body flipped the guard and the test skipped
    # instead of failing — verified by shadowing HEAD's src and removing it:
    # "3 passed, 1 skipped, exit 0". The fixture cannot reach the skip anyway:
    # `tests/conftest.py` stamps `derivation_profile moving_ref_frame=False`
    # while the profile asks True, so `_resolve_source` always substitutes.
    reported = _reports(node)
    assert "Dataset substitution" in reported, (
        "a run that trains on a different node than it was given must say so"
    )
    assert str(dataset.pk) in reported, "the log must name the node that was asked for"
