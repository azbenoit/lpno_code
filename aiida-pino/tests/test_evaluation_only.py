"""Evaluation-only data must not be trained on.

Marking a dataset evaluation-only means every member carries `split="test"`.
Without a guard the training path reads "no train members" as "no split
information at all" and falls back to the whole dataset — so a set reserved for
evaluation would be trained on, and its held-out number would be a training
number.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from aiida_pino.console.config import Settings
from aiida_pino.console.errors import InvalidRequestError
from aiida_pino.console.service import AiidaConsoleService


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


def _dataset(factory, name, n=3):
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    sims = [factory(name, group_name=f"{name}-{i}") for i in range(n)]
    meta = build_collection_metadata_from_simulations(sims, collection_name=name)
    node, _ = get_or_create_dataset_collection(sims, metadata=meta)
    return node


def test_marking_a_dataset_evaluation_only_sets_every_member_to_test(svc, pino_simulation_factory):
    node = _dataset(pino_simulation_factory, "evalset")
    out = svc.mark_evaluation_only(node.pk)

    assert out["changed"] == out["members"] == 3
    assert {str(s.base.extras.get("split")) for s in node.iter_member_simulations()} == {"test"}
    assert node.base.extras.get("evaluation_only") is True
    # idempotent: running it again changes nothing
    assert svc.mark_evaluation_only(node.pk)["changed"] == 0


def test_training_refuses_an_evaluation_only_dataset(svc, pino_simulation_factory, local_code_factory):
    """The whole point. Before the guard this fell through to "no split extras"
    and trained on the held-out data."""
    from aiida_pino.console.schemas import TrainEvalRequest

    node = _dataset(pino_simulation_factory, "reserved")
    svc.mark_evaluation_only(node.pk)
    local_code_factory("/bin/bash", label="pino-python-evalonly")

    with pytest.raises(InvalidRequestError, match="evaluation-only"):
        svc.submit_train_eval(
            TrainEvalRequest(
                train_dataset_pk=node.pk,
                python_code="pino-python-evalonly",
                training_profile={},
            )
        )


def test_a_mixed_dataset_is_still_trainable(svc, pino_simulation_factory):
    """The guard fires only when *every* member is test — a normal dataset with
    a held-out test split must stay usable."""
    node = _dataset(pino_simulation_factory, "mixed", n=4)
    members = node.iter_member_simulations()
    for sim, split in zip(members, ["train", "train", "val", "test"]):
        sim.base.extras.set("split", split)

    svc._refuse_if_evaluation_only(node, "training data")  # must not raise
