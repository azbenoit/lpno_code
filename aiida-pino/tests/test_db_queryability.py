"""Round-9 queryability fixes: held-out scores, payload twins, the dead seed reader.

Every number below was measured on the authors' database on 2026-08-09 before the fix landed.
"""

from __future__ import annotations

from aiida import orm

from aiida_pino.data.pino_model_evaluation import PinoModelEvaluationData
from aiida_pino.utils.model_registry import train_eval_overlap


def _summary(*uuids):
    return {"members": [{"uuid": u, "pk": i, "label": u, "simulation_identifier": u}
                        for i, u in enumerate(uuids)]}


def test_a_score_on_training_data_is_not_held_out():
    """4 of 208 evaluations were scored on their own training data.

    The worst evaluation scored 83 cases of which 65 were its training set --
    a 78% contaminated number sitting beside 202 genuine ones since 2026-07-23,
    with nothing in the DB to tell them apart.
    """
    model = {"training_source_summary": _summary("a", "b", "c")}
    assert train_eval_overlap(model, _summary("d", "e")) == 0
    assert train_eval_overlap(model, _summary("b", "c", "d")) == 2


def test_no_evidence_never_reads_as_no_overlap():
    """The trap: a missing training set intersects to the empty set.

    2 of 208 evals have neither a resolvable training source nor a frozen member
    list. Scoring those as overlap 0 would stamp `held_out=True` on exactly the
    numbers nobody can verify. Unknown must stay unknown.
    """
    assert train_eval_overlap({}, _summary("a")) is None
    assert train_eval_overlap({"training_source_summary": _summary("a")}, {"members": []}) is None
    assert train_eval_overlap({"training_source_summary": {}}, {}) is None


def test_held_out_is_one_filter(aiida_profile_clean):
    clean = PinoModelEvaluationData.from_payload(
        metadata={"evaluation_name": "clean", "train_overlap_n": 0}).store()
    dirty = PinoModelEvaluationData.from_payload(
        metadata={"evaluation_name": "dirty", "train_overlap_n": 65}).store()
    unknown = PinoModelEvaluationData.from_payload(
        metadata={"evaluation_name": "unknown"}).store()

    assert clean.base.extras.get("held_out") is True
    assert dirty.base.extras.get("held_out") is False
    assert unknown.base.extras.get("held_out", None) is None

    held = orm.QueryBuilder().append(
        PinoModelEvaluationData, filters={"extras.held_out": True}, project=["label"],
    ).all(flat=True)
    assert held == ["clean"], "the leaderboard filter must exclude both dirty and unknown"


def test_membership_follows_the_payload_not_the_node(aiida_profile_clean):
    """68 of 70 duplicated payloads answered differently depending on the twin.

    One twin reported 0 datasets, its twin 3 -- a shorter list, no error. The
    reverse lookup must key on content_hash, not on the uuid in your hand.
    """
    from aiida_pino.data import PinoDatasetCollectionData
    from aiida_pino.utils import datasets_containing, payload_twin_uuids

    from aiida_pino.data import PinoSimulationData

    twin_a, twin_b, lone = PinoSimulationData(), PinoSimulationData(), PinoSimulationData()
    for node, digest in ((twin_a, "deadbeef"), (twin_b, "deadbeef"), (lone, "0ther")):
        node.base.extras.set("content_hash", digest)
        node.store()

    # Only twin_a is referenced; asking about twin_b must still find the dataset.
    coll = PinoDatasetCollectionData()
    coll.base.attributes.set("member_simulation_uuids", [twin_a.uuid])
    coll.store()

    assert set(payload_twin_uuids(twin_b.uuid)) == {twin_a.uuid, twin_b.uuid}
    assert datasets_containing(twin_b.uuid) == [coll.pk]
    assert datasets_containing(twin_a.uuid) == [coll.pk]
    assert datasets_containing(lone.uuid) == []
