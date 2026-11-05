"""Round 6's four minor findings: F8 val split, F9 seed, F10 labels, F11 config.

Measured on the authors' database 2026-08-06 before the fixes:

* F8 — `train_dataset_pk`/`val_dataset_pk` present on **0 of 166** models: the
  lookup read `training_source_summary["train_dataset_pk"]`, and the summary key
  is `pk`. The validation split — which decides the kept checkpoint — was on no
  model node in any form.
* F9 — **65 of 166** models recorded no seed, while an omitted seed resolves to
  the literal 42 at `config_loading.py:784`.
* F10 — **5406 of 6258** processes (86%) had no label; the console renders a
  blank label as the class name.
* F11 — `profile-keys training` printed `config_NS_energy_PINO_flow3D`'s
  defaults whatever config you were about to submit.
"""

from __future__ import annotations

import pytest

from aiida_pino.console.config import Settings
from aiida_pino.console.errors import InvalidRequestError
from aiida_pino.console.service import AiidaConsoleService
from aiida_pino.data.pino_model import PINO_DEFAULT_SEED, queryable_model_extras


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


# --- F8 -----------------------------------------------------------------------
def test_the_dataset_pks_come_from_the_key_the_summary_actually_has(aiida_profile_clean):
    from aiida import orm

    train = orm.Dict(dict={"role": "train"}).store()
    val = orm.Dict(dict={"role": "val"}).store()
    extras = queryable_model_extras({
        # `pk`, not `train_dataset_pk` — the key the lookup used to want.
        "training_source_summary": {"pk": train.pk, "uuid": train.uuid},
        "validation_source_summary": {"pk": val.pk, "uuid": val.uuid},
    })
    assert extras["train_dataset_pk"] == train.pk
    assert extras["val_dataset_pk"] == val.pk
    assert extras["train_dataset_uuid"] == train.uuid
    assert extras["val_dataset_uuid"] == val.uuid


def test_no_validation_summary_means_no_val_extra():
    extras = queryable_model_extras({"training_source_summary": {}})
    assert "val_dataset_pk" not in extras


# --- F9 -----------------------------------------------------------------------
def test_an_omitted_seed_is_recorded_as_the_value_that_will_be_used():
    """42 is not "no seed": `config_loading.py:784` resolves it deterministically."""
    extras = queryable_model_extras({"training_profile": {}})
    assert extras["training_seed"] == PINO_DEFAULT_SEED == 42
    assert extras["training_seed_source"] == "lpno_default"


def test_an_explicit_seed_keeps_its_provenance():
    override = queryable_model_extras({"training_profile": {"hydra_overrides": ["+seed=1003"]}})
    assert override["training_seed"] == 1003
    assert override["training_seed_source"] == "hydra_override"
    direct = queryable_model_extras({"training_profile": {"seed": 7}})
    assert direct["training_seed"] == 7
    assert direct["training_seed_source"] == "training_profile.seed"


# --- F10 ----------------------------------------------------------------------
def test_a_child_inherits_the_callers_label():
    from aiida_pino.workflows.labels import InheritsCallerLabel

    class _Base:
        def __init__(self):
            self.seen = None

        def submit(self, process, inputs=None, **kwargs):
            self.seen = (inputs, kwargs)
            return "submitted"

    class _Chain(InheritsCallerLabel, _Base):
        node = type("N", (), {"label": "gaot_sw_arm0_seed43"})()

    chain = _Chain()
    chain.submit(object(), training_profile="x")
    assert chain.seen[1]["metadata"]["label"] == "gaot_sw_arm0_seed43"

    # A label the caller set deliberately is never overwritten.
    chain.submit(object(), metadata={"label": "mine"})
    assert chain.seen[1]["metadata"]["label"] == "mine"

    # An unlabelled caller adds nothing rather than an empty label.
    class _Blank(InheritsCallerLabel, _Base):
        node = type("N", (), {"label": ""})()

    blank = _Blank()
    blank.submit(object(), training_profile="x")
    assert "metadata" not in blank.seen[1]


# --- F11 ----------------------------------------------------------------------
def test_profile_keys_reads_the_config_you_name(svc, aiida_localhost):
    default = svc.profile_keys("training")
    named = svc.profile_keys("training", config_name="config_flow3D_GAOT_varspeed_ctrl")
    assert default["keys"]["config_name"] == "config_NS_energy_PINO_flow3D"
    assert named["keys"]["config_name"] == "config_flow3D_GAOT_varspeed_ctrl"
    assert default["keys"] != named["keys"], "naming a config must change the answer"
    assert set(default["required"]) == set(named["required"])
