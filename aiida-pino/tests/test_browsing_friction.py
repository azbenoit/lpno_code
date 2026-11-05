"""The six frictions reported from browsing models, each pinned by a test.

Reported 2026-08-05 after a 60-model census took ~7 minutes: model identity was
not queryable, workchain labels were empty, profiles validated at workchain
runtime rather than at submit, the required-key list was learnable only by
failing, and cloning a run's inputs had no command.
"""

from __future__ import annotations

import pytest

from aiida_pino.console.config import Settings
from aiida_pino.console.errors import InvalidRequestError
from aiida_pino.console.service import AiidaConsoleService
from aiida_pino.data.pino_model import queryable_model_extras


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


def test_model_identity_is_lifted_into_queryable_extras():
    """config / seed / dataset were reachable only by three link traversals."""
    metadata = {
        "config_name": "config_flow3D_RIGNO_randompath_onecycle",
        "training_source_pk": 24799,
        "training_source_label": "runs_train_split",
        "training_profile": {
            "config_name": "config_flow3D_RIGNO_randompath_onecycle",
            "max_steps": 35000,
            # The seed lives in a Hydra override, with the append prefix, which
            # is exactly why it was unqueryable: 93 models carry `+seed=`.
            "hydra_overrides": ["+seed=1003", "+h_norm_denom=216.386"],
        },
    }
    extras = queryable_model_extras(metadata)
    assert extras["config_name"] == "config_flow3D_RIGNO_randompath_onecycle"
    assert extras["training_seed"] == 1003
    assert extras["training_source_pk"] == 24799
    assert extras["max_steps"] == 35000


def test_a_stale_recorded_pk_is_replaced_by_the_uuid_it_sits_next_to(aiida_profile_clean):
    """The pk in `model_metadata` predates the migration that renumbered everything.

    Measured on the authors' database: 10 of 158 models stamp a `training_source_pk` that
    disagrees with the node their own `training_source_uuid` names, and every one
    of the ten loads successfully — as a `PinoSimulationData`, a single simulation
    standing in for the dataset collection the model actually trained on. A query
    by that pk gets a plausible wrong answer, which is worse than no answer.
    """
    from aiida import orm

    real = orm.Dict(dict={"i am": "the training source"}).store()
    extras = queryable_model_extras({
        "training_source_uuid": real.uuid,
        "training_source_pk": real.pk + 100000,  # what the pre-migration record says
    })
    assert extras["training_source_pk"] == real.pk

    # A uuid that names nothing stamps nothing: the stale number is not a fallback.
    assert "training_source_pk" not in queryable_model_extras({
        "training_source_uuid": "00000000-0000-0000-0000-000000000000",
        "training_source_pk": 4311,
    })


def test_seed_from_a_plain_profile_key_and_absent_seed():
    """Changed by round 6's F9, deliberately.

    This used to assert that an omitted seed records nothing. That read as
    "unseeded" when the value is perfectly determinate: `config_loading.py:784`
    resolves it to the literal 42, and 65 of 166 models were silent about it.
    The extra now always exists, and `training_seed_source` says where it came
    from — which is strictly more information, not less.
    """
    explicit = queryable_model_extras({"training_profile": {"seed": 7}})
    assert explicit["training_seed"] == 7
    assert explicit["training_seed_source"] == "training_profile.seed"

    absent = queryable_model_extras({"training_profile": {"hydra_overrides": []}})
    assert absent["training_seed"] == 42
    assert absent["training_seed_source"] == "lpno_default"


def test_a_stored_model_carries_the_extras(aiida_profile_clean, pino_model_factory):
    model = pino_model_factory()
    metadata = model.get_model_metadata()
    for key in queryable_model_extras(metadata):
        assert model.base.extras.get(key) is not None, f"{key} not stamped on the node"


def test_an_unknown_training_profile_key_is_refused_at_submit(svc, aiida_localhost):
    """A typo used to be stored, never read, and the run used the default.

    `wandb_offlien=true` satisfied the required-key check (defaults supply the
    real key), rode into the stored profile unread, and the run trained with
    `wandb_offline=False`.
    """
    with pytest.raises(InvalidRequestError, match="unknown training_profile key"):
        svc._validated_training_profile(
            {"config_name": "config_NS_energy_PINO", "run_name": "x", "wandb_offlien": True},
            split_mode="standard",
        )
    # The message must name the near miss, or it is just a different failure.
    try:
        svc._validated_training_profile(
            {"config_name": "config_NS_energy_PINO", "run_name": "x", "max_step": 500},
            split_mode="standard",
        )
    except InvalidRequestError as exc:
        assert "max_steps" in str(exc), "should suggest the key the user meant"
    else:  # pragma: no cover
        pytest.fail("a misspelt key was accepted")


def test_a_partial_profile_still_works(svc, aiida_localhost):
    """The leniency is deliberate and must survive the new strictness."""
    profile = svc._validated_training_profile(
        {"config_name": "config_NS_energy_PINO", "run_name": "x", "max_steps": 500},
        split_mode="standard",
    )
    assert profile["max_steps"] == 500
    assert profile["wandb_offline"] is not None  # filled from the config's defaults


def test_profile_keys_answers_what_can_i_set(svc, aiida_localhost):
    """The required list was learnable only by failing a submission."""
    training = svc.profile_keys("training")
    assert set(training["required"]) == {
        "config_name", "cuda_device", "hydra_overrides", "run_name", "wandb_offline"
    }
    assert "max_steps" in training["keys"]
    for which in ("evaluation", "derivation"):
        assert svc.profile_keys(which)["keys"], f"{which} lists no keys"
    with pytest.raises(InvalidRequestError, match="unknown profile"):
        svc.profile_keys("nonsense")


def test_submitted_processes_get_a_label(aiida_profile_clean):
    """4977 of 5718 processes had an empty label, so runs were unfindable."""
    from aiida import orm

    from aiida_pino.console.service import _label_submitted_process

    class _FakeNode:
        label = ""

    node = _FakeNode()
    _label_submitted_process(node, "pino.train_eval", {"training_profile": orm.Dict(dict={"run_name": "gaot_vs_01"})})
    assert node.label == "train_eval | gaot_vs_01"

    # A label a caller set deliberately is never overwritten.
    titled = _FakeNode()
    titled.label = "my careful name"
    _label_submitted_process(titled, "pino.train_eval", {"training_profile": orm.Dict(dict={"run_name": "x"})})
    assert titled.label == "my careful name"

    # An unnamed profile still yields something better than nothing.
    bare = _FakeNode()
    _label_submitted_process(bare, "pino.flow3d_generate", {"run_config": orm.Dict(dict={"case_name": "ti64_lhs_044"})})
    assert bare.label == "flow3d_generate | ti64_lhs_044"


def test_the_resolution_a_model_trained_at_is_recorded():
    """`env.PINO_LAZY_STRIDE` downsamples the grid and reached no record.

    18 of 158 models trained at 2x2x1 while `derivation_profile.stride` said
    [1,1,1] or nothing, including the model that holds the best T MAE badge
    (13.138 K) on a half-resolution grid. An error measured on a coarser grid is
    not comparable with one measured on the full grid, and nothing on the model
    said which it was.
    """
    from aiida_pino.data.pino_model import queryable_model_extras

    # The env var wins: the loader applies it after and independently of the
    # dataset's derivation, so it is what the model actually saw.
    strided = queryable_model_extras({
        "derivation_profile": {"stride": [1, 1, 1]},
        "training_profile": {"env": {"PINO_LAZY_STRIDE": "2,2,1"}},
    })
    assert strided["training_stride"] == [2, 2, 1]
    assert strided["full_resolution"] is False
    assert strided["training_stride_source"] == "env.PINO_LAZY_STRIDE"

    # With no env override the recorded derivation stands.
    recorded = queryable_model_extras({
        "derivation_profile": {"stride": [1, 1, 1]},
        "training_profile": {"env": {}},
    })
    assert recorded["training_stride"] == [1, 1, 1]
    assert recorded["full_resolution"] is True
    assert recorded["training_stride_source"] == "derivation_profile.stride"

    # Unknown stays unknown rather than defaulting to "full", which would be a
    # comparability claim nobody made.
    assert "training_stride" not in queryable_model_extras({"training_profile": {}})

    # A malformed env value must not be read as a stride.
    assert "training_stride" not in queryable_model_extras(
        {"training_profile": {"env": {"PINO_LAZY_STRIDE": "not,a,stride"}}}
    )
