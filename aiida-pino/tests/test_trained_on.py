"""`models --trained-on`: the dataset->models reverse lookup (2026-08-10 friction).

The train input is usually a split or layout variant reached only through the
`base_collection_uuid` / `parent_collection_uuid` ATTRIBUTES -- splits mint
their own family_identifier and no graph edge points back to the parent, so
provenance traversal from the base finds nothing. Measured live before landing:
one dataset -> closure of 6 collections, 0 models, 12 top-level training
chains (8 failed exit 300, 2 killed, 2 in flight) -- the in-flight and
never-materialized runs are exactly what the cheap-extras answer can never see.
"""

from __future__ import annotations

from aiida import orm

from aiida_pino.console.config import Settings
from aiida_pino.console.service import AiidaConsoleService
from aiida_pino.data import PinoDatasetCollectionData, PinoModelData
from aiida_pino.data.pino_model_evaluation import PinoModelEvaluationData


def _collection(label, *, base_uuid=None):
    node = PinoDatasetCollectionData()
    node.label = label
    if base_uuid:
        node.base.attributes.set("base_collection_uuid", base_uuid)
    return node.store()


def test_trained_on_follows_the_pointer_attributes(aiida_profile_clean):
    base = _collection("base")
    split = _collection("base_train_split", base_uuid=base.uuid)
    other = _collection("unrelated")

    model = PinoModelData()
    model.base.extras.set("model_name", "m1")
    model.base.extras.set("training_source_uuid", split.uuid)
    model.store()
    bystander = PinoModelData()
    bystander.base.extras.set("model_name", "m2")
    bystander.base.extras.set("training_source_uuid", other.uuid)
    bystander.store()

    svc = AiidaConsoleService(Settings())
    # Asked about the BASE, the lookup must reach the model trained on the split.
    result = svc.models_trained_on(base.pk)
    assert {m["pk"] for m in result["closure"]} == {base.pk, split.pk}
    assert [m["model_name"] for m in result["models"]] == ["m1"]

    # Asked about the SPLIT, the closure walks up to the base and back down.
    result_up = svc.models_trained_on(split.pk)
    assert {m["pk"] for m in result_up["closure"]} == {base.pk, split.pk}

    assert svc.models_trained_on(other.pk)["models"] == [{
        "pk": bystander.pk, "model_name": "m2", "trained_on_uuid": other.uuid,
    }]


def test_evals_are_reported_separately_not_as_models(aiida_profile_clean):
    """An evaluation named like a model (e4_*) must not read as a trained control."""
    ds = _collection("scored-set")
    ev = PinoModelEvaluationData.from_payload(metadata={
        "evaluation_name": "e4_rigno_sp_test",
        "dataset_source_uuid": ds.uuid,
        "train_overlap_n": 0,
    }).store()

    svc = AiidaConsoleService(Settings())
    result = svc.models_trained_on(ds.pk)
    assert result["models"] == []
    assert [e["pk"] for e in result["evaluations_scored_on_it"]] == [ev.pk]
    assert result["evaluations_scored_on_it"][0]["held_out"] is True


def test_evaluations_mirror_the_training_source(aiida_profile_clean):
    """Querying training_source_uuid on an eval returned a None that looked real."""
    node = PinoModelEvaluationData.from_payload(metadata={
        "evaluation_name": "x", "training_source_uuid": "abc-123",
    }).store()
    assert node.base.extras.get("training_source_uuid") == "abc-123"

    silent = PinoModelEvaluationData.from_payload(metadata={"evaluation_name": "y"}).store()
    assert silent.base.extras.get("training_source_uuid", None) is None
