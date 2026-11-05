"""Direct parser unit tests against synthetic retrieved folders."""

from __future__ import annotations

import json
from uuid import uuid4

import numpy as np

from aiida import orm
from aiida.common.links import LinkType

from aiida_pino.data import PinoSimulationData
from aiida_pino.parsers import vtk_to_dataset
from aiida_pino.parsers.eval import PinoEvalParser
from aiida_pino.parsers.kfold import PinoKfoldParser
from aiida_pino.parsers.train import PinoTrainParser


def _make_calcjob_with_retrieved(computer, entry_point: str, files: dict[str, str]) -> orm.CalcJobNode:
    """Create a stored ``CalcJobNode`` for ``entry_point`` with a linked retrieved folder."""
    node = orm.CalcJobNode(computer=computer, process_type=f"aiida.calculations:{entry_point}")
    node.set_option("output_filename", "aiida.out")
    node.set_option("resources", {"num_machines": 1})
    node.store()
    retrieved = orm.FolderData()
    for name, content in files.items():
        retrieved.base.repository.put_object_from_bytes(content.encode("utf-8"), name)
    retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
    retrieved.store()
    return node


_EVAL_BASE_FILES = {
    "eval_output/predictions.npz": "prediction",
    "eval_summary.json": json.dumps({"output_dir": "eval_output", "metrics_file_present": False}),
}


# --- BUG-16: eval stdout metrics fallback ------------------------------------


def test_eval_parser_falls_back_to_stdout_metrics_when_metrics_json_missing(aiida_localhost) -> None:
    files = dict(_EVAL_BASE_FILES)
    files["aiida.out"] = "noise line\nT: {'rmse': 0.1, 'rel_rmse': 0.2}\nalpha: {'iou_mean': 0.9}\n"
    node = _make_calcjob_with_retrieved(aiida_localhost, "pino.eval", files)

    parser = PinoEvalParser(node)
    result = parser.parse()

    assert result is None
    assert parser.outputs["metrics"].get_dict() == {
        "T": {"rmse": 0.1, "rel_rmse": 0.2},
        "alpha": {"iou_mean": 0.9},
    }


# BUG-17 replaced the silent `{}` metrics degradation with exit code 502.
def test_eval_parser_returns_502_without_metrics_json_or_stdout(aiida_localhost) -> None:
    files = dict(_EVAL_BASE_FILES)
    files["aiida.out"] = "no metric dicts here\n"
    node = _make_calcjob_with_retrieved(aiida_localhost, "pino.eval", files)

    parser = PinoEvalParser(node)
    result = parser.parse()

    assert result == node.process_class.spec().exit_codes.ERROR_MISSING_METRICS
    assert result.status == 502


# --- BUG-22: kfold stdout metrics fallback ------------------------------------


_KFOLD_BASE_FILES = {
    "kfold_output/fold_0/model_errors.txt": "[T]\nrmse: 0.12\n",
    "kfold_summary.json": json.dumps({"output_dir": "kfold_output", "aggregated_errors_present": False}),
}


def test_kfold_parser_falls_back_to_stdout_metrics_when_metrics_json_missing(aiida_localhost) -> None:
    files = dict(_KFOLD_BASE_FILES)
    files["aiida.out"] = "noise line\nT: {'rmse': 0.1, 'rel_rmse': 0.2}\n"
    node = _make_calcjob_with_retrieved(aiida_localhost, "pino.kfold", files)

    parser = PinoKfoldParser(node)
    result = parser.parse()

    assert result is None
    assert parser.outputs["metrics"].get_dict() == {"T": {"rmse": 0.1, "rel_rmse": 0.2}}


# BUG-17 replaced the silent `{}` metrics degradation with exit code 502.
def test_kfold_parser_returns_502_without_metrics_json_or_stdout(aiida_localhost) -> None:
    files = dict(_KFOLD_BASE_FILES)
    files["aiida.out"] = "no metric dicts here\n"
    node = _make_calcjob_with_retrieved(aiida_localhost, "pino.kfold", files)

    parser = PinoKfoldParser(node)
    result = parser.parse()

    assert result == node.process_class.spec().exit_codes.ERROR_MISSING_METRICS
    assert result.status == 502


# --- BUG-17: structured exit codes for the ML CalcJob parsers -----------------


def test_train_parser_returns_500_when_summary_missing(aiida_localhost) -> None:
    files = {"training_output/model.ckpt": "checkpoint", "aiida.out": ""}
    node = _make_calcjob_with_retrieved(aiida_localhost, "pino.train", files)

    parser = PinoTrainParser(node)
    result = parser.parse()

    assert result == node.process_class.spec().exit_codes.ERROR_MISSING_SUMMARY
    assert result.status == 500


def test_train_parser_returns_501_when_artifacts_missing(aiida_localhost) -> None:
    files = {"train_summary.json": json.dumps({"output_dir": "training_output"}), "aiida.out": ""}
    node = _make_calcjob_with_retrieved(aiida_localhost, "pino.train", files)

    parser = PinoTrainParser(node)
    result = parser.parse()

    assert result == node.process_class.spec().exit_codes.ERROR_MISSING_ARTIFACTS
    assert result.status == 501


def test_eval_parser_returns_500_when_summary_missing(aiida_localhost) -> None:
    files = {"eval_output/predictions.npz": "prediction", "aiida.out": ""}
    node = _make_calcjob_with_retrieved(aiida_localhost, "pino.eval", files)

    parser = PinoEvalParser(node)
    result = parser.parse()

    assert result.status == 500


def test_kfold_parser_returns_500_when_summary_missing(aiida_localhost) -> None:
    files = {"kfold_output/fold_0/model_errors.txt": "[T]\nrmse: 0.12\n", "aiida.out": ""}
    node = _make_calcjob_with_retrieved(aiida_localhost, "pino.kfold", files)

    parser = PinoKfoldParser(node)
    result = parser.parse()

    assert result.status == 500


# --- BUG-20: check-then-store dedup race on simulation fingerprints ----------


def _make_dedup_candidate(token: str) -> PinoSimulationData:
    """Build an unstored candidate; the same token yields the same fingerprint.

    The array values depend on the token: distinct tokens must model distinct
    cases, and since imports now dedupe on payload bytes as well as on the
    source path, a shared constant array would make every fixture in this file
    the same simulation.
    """
    fill = float(int(token[:6], 16) % 1000)
    return PinoSimulationData.from_arrays(
        {"Temperature": np.full((2, 1, 4, 4, 4), fill, dtype=np.float32)},
        metadata={
            "source_folder_name": f"dedup_{token}",
            "source_group": f"dedup_{token}",
            "source_uri": f"file:///tmp/dedup_{token}",
        },
    )


def test_dedup_second_candidate_resolves_to_first_stored_node() -> None:
    token = uuid4().hex[:12]
    winner, created = vtk_to_dataset._resolve_deduplicated_simulation(_make_dedup_candidate(token))
    assert winner.is_stored
    assert created is True  # this import made it, so this import may claim it

    duplicate = _make_dedup_candidate(token)
    resolved, created_again = vtk_to_dataset._resolve_deduplicated_simulation(duplicate)

    assert resolved.uuid == winner.uuid
    assert not duplicate.is_stored  # fast path: duplicate is never stored
    # A Data node has exactly one CREATE link: a re-import must not claim it,
    # or the parser would raise when emitting a second creator.
    assert created_again is False


def test_dedup_race_loser_is_marked_superseded(monkeypatch) -> None:
    token = uuid4().hex[:12]
    winner, _ = vtk_to_dataset._resolve_deduplicated_simulation(_make_dedup_candidate(token))

    # Simulate the TOCTOU window: the fast-path query misses even though the
    # winner has been stored concurrently.
    real_query = vtk_to_dataset._query_simulation_by_fingerprint
    monkeypatch.setattr(
        vtk_to_dataset,
        "_query_simulation_by_fingerprint",
        lambda fingerprint, *, oldest_first=False: (
            real_query(fingerprint, oldest_first=True) if oldest_first else None
        ),
    )

    loser = _make_dedup_candidate(token)
    resolved, created = vtk_to_dataset._resolve_deduplicated_simulation(loser)

    assert resolved.uuid == winner.uuid  # oldest node wins deterministically
    assert created is False  # a race loser is claimed by nobody
    assert loser.is_stored
    assert loser.base.extras.get("superseded") is True
    assert loser.base.extras.get("duplicate_of_uuid") == winner.uuid


def test_dedup_catches_the_same_case_imported_from_two_directories() -> None:
    """The fingerprint hashes the source *path*, so the same case imported from
    two directories was stored twice under two fingerprints. Twelve such pairs
    exist in the live database (review criticism J): two nodes share
    payload key `d205e02e...` and both sit in one live collection, which
    therefore double-weights a case.
    """
    from aiida_pino.data.pino_simulation import payload_content_hash

    first, created = vtk_to_dataset._resolve_deduplicated_simulation(
        _make_dedup_candidate(uuid4().hex[:12])
    )
    assert created is True

    # identical arrays, different directory: different fingerprint, same bytes
    twin = PinoSimulationData.from_arrays(
        {"Temperature": np.asarray(first.get_array("Temperature"))},
        metadata={
            "source_folder_name": "elsewhere",
            "source_group": "elsewhere",
            "source_uri": "file:///tmp/somewhere_else",
        },
    )
    assert twin.get_simulation_fingerprint() != first.get_simulation_fingerprint()

    resolved, created_again = vtk_to_dataset._resolve_deduplicated_simulation(twin)
    assert resolved.uuid == first.uuid, "byte-identical import was stored as a new case"
    assert created_again is False
    assert twin.base.extras.get("superseded") is True
    assert twin.base.extras.get("duplicate_of_uuid") == first.uuid
    # the hash that made it detectable is content-derived and free to read
    assert payload_content_hash(first) == payload_content_hash(twin)
    assert first.base.extras.get("content_hash") == payload_content_hash(first)
