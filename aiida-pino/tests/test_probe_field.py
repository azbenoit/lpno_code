"""`--probe`: does this path exist, and if not, what did you mean?

QueryBuilder projects a path that does not exist as `None` rather than raising,
so a wrong field name and an empty database are indistinguishable. Four "X is
not stored" reports in one day were that mistake -- collection names (the name
is the `label` COLUMN), evaluation training sources, evals in `models --search`,
and grid shapes (`extras.mesh_cells`, projectable in 2.4 s against ~62 s spent
iterating). Each cost about an hour of believing the data was missing.
"""

from __future__ import annotations

from aiida_pino.utils import queryable_paths


def _report(paths):
    return {
        "kind": "simulation",
        "class": "PinoSimulationData",
        "total": len(paths),
        "scanned": 200,
        "paths": [
            {"path": p, "count": c, "coverage": 100.0 * c / 200, "type": "str", "example": None}
            for p, c in paths
        ],
    }


def test_a_hit_answers_with_the_count_and_a_pasteable_query(monkeypatch):
    monkeypatch.setattr(queryable_paths, "discover",
                        lambda kind, limit=None: _report([("extras.mesh_cells", 194)]))
    out = queryable_paths.probe("simulation", "extras.mesh_cells")
    assert out["exists"] is True and out["count"] == 194
    assert "194 of 200" in out["message"]
    assert "QueryBuilder" in out["query"]


def test_a_miss_says_zero_of_n_and_suggests(monkeypatch):
    monkeypatch.setattr(queryable_paths, "discover",
                        lambda kind, limit=None: _report([("extras.material_name", 200)]))
    out = queryable_paths.probe("simulation", "extras.material_nam")
    assert out["exists"] is False
    assert "exists on 0 of 200" in out["message"]
    assert out["suggestions"][0]["path"] == "extras.material_name"


def test_synonyms_bridge_what_fuzzy_matching_cannot(monkeypatch):
    """`grid_shape` and `mesh_cells` share no letters worth matching."""
    monkeypatch.setattr(queryable_paths, "discover", lambda kind, limit=None: _report([
        ("extras.mesh_cells", 194), ("extras.split", 200),
    ]))
    out = queryable_paths.probe("simulation", "extras.grid_shape")
    assert [s["path"] for s in out["suggestions"]][:1] == ["extras.mesh_cells"]


def test_a_name_question_is_answered_by_the_label_column(monkeypatch):
    """The motivating case: the name of a collection is a COLUMN, not an
    attribute, so the JSON walk can never surface it."""
    monkeypatch.setattr(queryable_paths, "discover", lambda kind, limit=None: _report([
        ("attributes.base_collection_pk", 200),
    ]))
    out = queryable_paths.probe("dataset", "attributes.collection_name")
    first = out["suggestions"][0]
    assert first["path"] == "label" and first["type"] == "column"
    assert "did you mean label?" in out["message"]


def test_a_hash_is_never_offered_as_an_answer(monkeypatch):
    monkeypatch.setattr(queryable_paths, "discover", lambda kind, limit=None: _report([
        ("extras._aiida_hash", 20),
    ]))
    out = queryable_paths.probe("simulation", "extras._aiida_hashish")
    assert all("_aiida_hash" not in s["path"] for s in out["suggestions"])


def test_nothing_similar_says_so_rather_than_reaching(monkeypatch):
    monkeypatch.setattr(queryable_paths, "discover",
                        lambda kind, limit=None: _report([("extras.split", 200)]))
    out = queryable_paths.probe("simulation", "extras.zzzz_unrelated_qqq")
    assert out["suggestions"] == []
    assert "nothing similar exists" in out["message"]
