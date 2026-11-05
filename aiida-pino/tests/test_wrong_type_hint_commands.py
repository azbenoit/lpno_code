"""`show`/`logs` on a data node must point at a command that exists.

The hint suggested `aiida-pino evaluation <pk>` and `aiida-pino simulation <pk>`, and
neither subcommand exists (found writing reproducibility/train_evaluate.ipynb).
"""
from __future__ import annotations

import argparse
import re

import pytest

from aiida_pino import cli


@pytest.fixture
def svc():
    from aiida_pino.console.config import Settings
    from aiida_pino.console.service import AiidaConsoleService

    return AiidaConsoleService(Settings())


def _subcommands() -> set[str]:
    for action in cli.build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("aiida-pino has no subcommands")


def _suggested(hint: str) -> set[str]:
    return set(re.findall(r"`aiida-pino ([a-z][a-z0-9-]*)", hint)) - {"show"}


def test_every_suggested_command_exists(svc, pino_simulation_factory, pino_dataset_factory,
                                        pino_model_factory, pino_model_evaluation_factory):
    commands = _subcommands()
    dataset = pino_dataset_factory("hint-ds")
    model = pino_model_factory("hint-model", training_source=dataset)
    evaluation = pino_model_evaluation_factory(model=model, dataset_source=dataset)
    nodes = [pino_simulation_factory("hint-sim"), dataset, model, evaluation]
    for node in nodes:
        hint = svc._wrong_type_hint(node.pk, "show")
        suggested = _suggested(hint)
        assert suggested, hint
        assert suggested <= commands, f"{type(node).__name__}: {sorted(suggested - commands)} in {hint!r}"


def test_an_evaluation_points_at_its_model(svc, pino_dataset_factory, pino_model_factory,
                                           pino_model_evaluation_factory):
    from aiida_pino.console.service import NodeNotFoundError

    dataset = pino_dataset_factory("hint-ds-2")
    model = pino_model_factory("hint-model-2", training_source=dataset)
    evaluation = pino_model_evaluation_factory(model=model, dataset_source=dataset)
    with pytest.raises(NodeNotFoundError) as caught:
        svc.process_detail(evaluation.pk)
    message = str(caught.value)
    assert f"aiida-pino compare {evaluation.pk}" in message
    assert f"aiida-pino model {model.pk}" in message
