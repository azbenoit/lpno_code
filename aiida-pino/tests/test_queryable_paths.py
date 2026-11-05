"""Discovering what is queryable, instead of writing a backfill per field.

Every "X isn't stored" report so far has been wrong the same way: X was stored,
nested in a metadata blob, and undiscoverable. The FLOW-3D solve wall clock was
reported missing while

    attributes.simulation_metadata.flow3d_case_manifest.run.elapsed_seconds

returned 1547 values through a plain QueryBuilder — it always did. The repo's
answer had been a module per field: eight `*_backfill.py` files by 2026-08-06.
"""

from __future__ import annotations

from aiida_pino.utils.queryable_paths import (
    LONG_LIST,
    MAX_DEPTH,
    _duplicates,
    _walk,
    discover,
    query_line,
)


def _paths(blob: dict) -> dict:
    seen: dict = {}
    _walk(blob, "attributes", seen)
    return seen


def test_a_nested_leaf_is_reported_with_its_full_path():
    seen = _paths({"simulation_metadata": {"flow3d_case_manifest": {"run": {"elapsed_seconds": 166.77}}}})
    path = "attributes.simulation_metadata.flow3d_case_manifest.run.elapsed_seconds"
    assert path in seen
    assert seen[path]["types"] == {"float"}
    assert seen[path]["example"] == "166.77"
    # The containers are reported too: that is how you find what is under them.
    assert "attributes.simulation_metadata.flow3d_case_manifest.run" in seen


def test_a_long_list_is_described_not_walked():
    """`laser_path_xy_m` holds thousands of points; it is data, not a field."""
    seen = _paths({"derived": {"laser_path_xy_m": [[0.0, 0.0]] * 4001}})
    entry = seen["attributes.derived.laser_path_xy_m"]
    assert entry["types"] == {"list[list[float] x2] x4001"}
    assert not any("laser_path_xy_m[" in path for path in seen), "must not expand a long list"


def test_a_short_list_of_objects_is_walked():
    seen = _paths({"members": [{"pk": 1}, {"pk": 2}]})
    assert "attributes.members[0].pk" in seen
    assert len(_paths({"members": [{"pk": i} for i in range(LONG_LIST + 1)]})) == 1


def test_walking_stops_before_it_runs_away():
    blob: dict = {"leaf": 1}
    for _ in range(MAX_DEPTH + 4):
        blob = {"deeper": blob}
    depths = [path.count(".") for path in _paths(blob)]
    assert max(depths) <= MAX_DEPTH + 1


def test_the_same_leaf_at_two_paths_is_flagged():
    """The F2 trap: `stride` under two profiles is two different numbers."""
    rows = [
        {"path": "attributes.model_metadata.derivation_profile.stride", "type": "list[int] x3"},
        {"path": "attributes.model_metadata.training_profile.env.PINO_LAZY_STRIDE", "type": "str"},
        {"path": "attributes.model_metadata.training_summary.profile.env.PINO_LAZY_STRIDE", "type": "str"},
        {"path": "attributes.model_metadata.training_profile", "type": "object"},
    ]
    clashes = _duplicates(rows)
    assert set(clashes) == {"PINO_LAZY_STRIDE"}
    assert len(clashes["PINO_LAZY_STRIDE"]) == 2
    # Containers are not leaves and must not be counted as clashes.
    assert "training_profile" not in clashes


def test_the_query_line_is_paste_ready():
    line = query_line("simulation", "extras.wall_seconds")
    assert "PinoSimulationData" in line
    assert "project=['extras.wall_seconds']" in line
    assert "'extras.wall_seconds': {'!==': None}" in line


def test_discover_finds_a_buried_value_on_a_real_node(aiida_profile_clean, pino_simulation_factory):
    sim = pino_simulation_factory("discover-case")
    metadata = dict(sim.base.attributes.get("simulation_metadata") or {})
    metadata["flow3d_case_manifest"] = {"run": {"elapsed_seconds": 1890.6}}
    sim.backend_entity.set_attribute("simulation_metadata", metadata)

    report = discover("simulation")
    paths = {row["path"]: row for row in report["paths"]}
    buried = "attributes.simulation_metadata.flow3d_case_manifest.run.elapsed_seconds"
    assert buried in paths
    assert paths[buried]["coverage"] == 100.0
    assert report["scanned"] == report["total"] == 1
