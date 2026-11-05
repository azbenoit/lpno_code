"""`import-model --training-source` must say what it wants when given the wrong node.

2026-08-23. Passing a training CalcJobNode or
its WorkChainNode failed with

    TypeError: Cannot convert value of type
    <class 'aiida.orm.nodes.process.calculation.calcjob.CalcJobNode'> to AiiDA type

and nothing launched. The message names no expected type, the CLI's `--training-source`
carried no help text, and the caller could not tell a wrong argument from a bug.

`training_source` is the DATASET the model was trained on -- a `PinoDatasetCollectionData`
-- not the training job. The workchain port is typed `orm.Data`, so a process node dies in
AiiDA's own input coercion BEFORE reaching `normalize_dataset_collection_node`, which would
have said so clearly. The service now checks at submit time instead.
"""

from __future__ import annotations

import pytest

from web.app.config import Settings
from web.app.aiida_service import AiidaConsoleService
from web.app.errors import InvalidRequestError


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


@pytest.fixture
def pino_code(aiida_profile_clean, local_code_factory):
    """The service resolves the code before the source, so the test needs a real one."""
    code = local_code_factory("/bin/true", label="pino")
    return f"{code.label}@{code.computer.label}"


def _request(pk, code_label="pino@localhost", **kw):
    from web.app.schemas import ImportModelRequest

    return ImportModelRequest(
        python_code=code_label,
        training_source_pk=pk,
        artifacts_path="/tmp/does-not-matter",
        training_profile={},
        model_name=None,
        **kw,
    )


def test_a_process_node_is_refused_with_a_message_naming_the_expected_type(svc, pino_code):
    """The reported case. Must not be a bare TypeError from AiiDA's coercion."""
    from aiida import orm

    job = orm.CalcJobNode()
    job.store()

    with pytest.raises(InvalidRequestError) as excinfo:
        svc.submit_import_model(_request(job.pk, pino_code))
    message = str(excinfo.value)
    assert "PinoDatasetCollectionData" in message, "must name what it wants"
    assert str(job.pk) in message, "must name the node the caller passed"


def test_a_data_node_of_the_wrong_type_is_refused_too(svc, pino_code):
    """Not just process nodes: any non-collection must fail the same clear way."""
    from aiida import orm

    node = orm.Int(3).store()
    with pytest.raises(InvalidRequestError) as excinfo:
        svc.submit_import_model(_request(node.pk, pino_code))
    assert "PinoDatasetCollectionData" in str(excinfo.value)


def test_the_error_points_at_the_dataset_the_training_job_actually_used(svc, pino_code):
    """A process node is a near-miss, not a random wrong answer.

    The caller who passes a training job wants the dataset that job trained on, and that
    dataset is one link away. Naming its pk turns the recovery into one step instead of a
    hunt through the provenance graph.
    """
    from aiida import orm
    from aiida.common.links import LinkType

    from aiida_pino.data import PinoDatasetCollectionData

    dataset = PinoDatasetCollectionData()
    dataset.store()
    job = orm.CalcJobNode()
    job.base.links.add_incoming(dataset, link_type=LinkType.INPUT_CALC, link_label="train_dataset")
    job.store()

    with pytest.raises(InvalidRequestError) as excinfo:
        svc.submit_import_model(_request(job.pk, pino_code))
    assert str(dataset.pk) in str(excinfo.value), "should name the dataset the job used"
