"""Tests for the shared annotation helpers (split/keywords/description extras).

Covers the constants contract other repos import, the core
`update_node_annotations` semantics on real stored nodes, and the
`AnnotationsRequest` schema envelope.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from aiida_pino.utils.annotations import (
    SPLIT_VALUES,
    SUGGESTED_KEYWORDS,
    normalize_keywords,
    HELD_OUT_SPLIT_VALUES,
    update_node_annotations,
)


def test_constants_contract():
    # Widened 2026-08-17 to sanction values that were ALREADY
    # live in production, written by ad-hoc `extras.set` scripts precisely
    # because this tuple rejected them: `eval` on 4 simulations (two
    # datasets) and `duplicate_excluded` on one dataset's 5 centre-point replicas.
    # `excluded` is what new work should use; `duplicate_excluded` stays
    # writable so the existing five can be restated rather than migrated.
    # Checked before widening: no sibling repo imports SPLIT_VALUES, so the
    # "other repos import" note in this module's docstring is not load-bearing.
    assert SPLIT_VALUES == (
        "train",
        "val",
        "test",
        "eval",
        "excluded",
        "duplicate_excluded",
        "unassigned",
    )
    # The original four must never silently disappear from the vocabulary, and
    # `unassigned` must remain the last entry -- it is the default a simulation
    # is created with.
    for legacy in ("train", "val", "test", "unassigned"):
        assert legacy in SPLIT_VALUES
    assert SPLIT_VALUES[-1] == "unassigned"
    assert SUGGESTED_KEYWORDS == (
        "keyhole",
        "conduction",
        "random path",
        "variable material",
        "variable path length",
        "straight path",
        "even timesteps",
        "steady",
        "downsampled",
        "smoke",
    )


def test_normalize_keywords_strips_drops_empties_dedupes_in_order():
    assert normalize_keywords([" keyhole ", "", "keyhole", "smoke", "   ", "steady"]) == [
        "keyhole",
        "smoke",
        "steady",
    ]
    assert normalize_keywords(None) == []


def test_defaults_and_result_shape(pino_simulation_factory):
    sim = pino_simulation_factory("ann-defaults")
    out = update_node_annotations(sim)
    assert out["pk"] == sim.pk
    assert out["kind"] == "simulation"
    assert out["label"]
    assert out["split"] == "unassigned"
    assert out["keywords"] == []


def test_set_and_change_split(pino_simulation_factory):
    sim = pino_simulation_factory("ann-split")
    out = update_node_annotations(sim, split="train")
    assert out["split"] == "train"
    assert sim.base.extras.get("split") == "train"
    out = update_node_annotations(sim, split="val")
    assert out["split"] == "val"
    assert sim.base.extras.get("split") == "val"


def test_invalid_split_value_rejected(pino_simulation_factory):
    # New sims are stamped split="unassigned" at creation; a rejected value
    # must leave that untouched.
    sim = pino_simulation_factory("ann-split-bad")
    before = sim.base.extras.get("split", "unassigned")
    with pytest.raises(ValueError):
        update_node_annotations(sim, split="training")
    assert sim.base.extras.get("split", "unassigned") == before


def test_split_on_non_simulation_rejected(pino_model_factory):
    model = pino_model_factory("ann-split-model")
    with pytest.raises(ValueError):
        update_node_annotations(model, split="train")
    out = update_node_annotations(model)
    assert out["kind"] == "model"
    assert "split" not in out


def test_keywords_set_add_remove_semantics(pino_simulation_factory):
    sim = pino_simulation_factory("ann-kw")
    out = update_node_annotations(sim, keywords_set=["keyhole", " smoke ", "keyhole"])
    assert out["keywords"] == ["keyhole", "smoke"]
    out = update_node_annotations(sim, keywords_add=["steady", "smoke"])
    assert out["keywords"] == ["keyhole", "smoke", "steady"]
    out = update_node_annotations(sim, keywords_remove=["smoke", "not-there"])
    assert out["keywords"] == ["keyhole", "steady"]
    # set replaces, and an explicit empty set clears
    out = update_node_annotations(sim, keywords_set=["conduction"])
    assert out["keywords"] == ["conduction"]
    out = update_node_annotations(sim, keywords_set=[])
    assert out["keywords"] == []
    assert sim.base.extras.get("keywords") == []


def test_description_native_field_and_simulation_extras_mirror(pino_simulation_factory):
    sim = pino_simulation_factory("ann-desc")
    out = update_node_annotations(sim, description="melt pool study")
    assert out["description"] == "melt pool study"
    assert sim.description == "melt pool study"
    assert sim.base.extras.get("description") == "melt pool study"
    assert sim.get_simulation_metadata()["description"] == "melt pool study"


def test_description_no_extras_mirror_on_non_simulation(pino_model_factory):
    model = pino_model_factory("ann-desc-model")
    out = update_node_annotations(model, description="model notes")
    assert out["description"] == "model notes"
    assert model.description == "model notes"
    assert model.base.extras.get("description", None) is None


def test_collection_kind_and_keywords(pino_dataset_factory):
    collection = pino_dataset_factory("ann-collection")
    out = update_node_annotations(collection, description="coll notes", keywords_add=["random path"])
    assert out["kind"] == "dataset_collection"
    assert "split" not in out
    assert out["keywords"] == ["random path"]
    assert collection.description == "coll notes"


def test_annotations_request_schema():
    import pydantic

    from aiida_pino.console.schemas import AnnotationsRequest

    with pytest.raises(pydantic.ValidationError):
        AnnotationsRequest()
    with pytest.raises(pydantic.ValidationError):
        AnnotationsRequest(split="train", bogus_field=1)
    req = AnnotationsRequest(split="train", keywords_add=["keyhole"])
    assert req.split == "train"
    assert req.keywords_add == ["keyhole"]
    assert req.description is None


def test_collection_create_request_schema_gains_annotation_fields():
    from aiida_pino.console.schemas import CollectionCreateRequest

    req = CollectionCreateRequest(simulation_pks=[1], collection_name="c")
    assert req.description is None and req.keywords is None
    req = CollectionCreateRequest(
        simulation_pks=[1], collection_name="c", description="d", keywords=["steady"]
    )
    assert req.description == "d" and req.keywords == ["steady"]


def test_split_change_records_an_audit_entry(pino_simulation_factory):
    """Every split write records its own history, whatever called it.

    `aiida-pino split` routes through `update_node_annotations` and used to
    write nothing: ten simulations were re-split on 2026-08-19 with no record of
    their previous values, and recovering them meant grepping a chat transcript.
    The audit lives here, at the one point every caller converges on, rather
    than in the console path that happened to have it.
    """
    sim = pino_simulation_factory("ann-split-audit")

    update_node_annotations(sim, split="train", split_reason="initial assignment")
    audit = sim.base.extras.get("split_audit")
    assert len(audit) == 1
    assert audit[0]["reason"] == "initial assignment"
    assert audit[0]["changed"] == 1
    assert audit[0]["actor"] and audit[0]["utc"]

    update_node_annotations(sim, split="eval", split_reason="held out")
    audit = sim.base.extras.get("split_audit")
    assert len(audit) == 2, "append-only: the first record survives"
    assert audit[1]["reason"] == "held out"

    # Re-writing the same value is not a change and must not pad the history.
    update_node_annotations(sim, split="eval")
    assert len(sim.base.extras.get("split_audit")) == 2


def test_eval_is_a_legal_held_out_split(pino_simulation_factory):
    """`eval` is legal and held out; a reader who cannot see that overwrites it.

    On 2026-08-19 a maintenance pass found `split='eval'`, judged it corrupt because the
    CLI help listed only train/val/test/unassigned, and rewrote ten nodes.
    """
    sim = pino_simulation_factory("ann-split-eval")
    out = update_node_annotations(sim, split="eval")
    assert out["split"] == "eval"
    assert "eval" in SPLIT_VALUES and "eval" in HELD_OUT_SPLIT_VALUES
