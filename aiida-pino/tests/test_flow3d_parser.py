"""Unit tests for `Flow3DToVtkParser` (BUG-06) and `Flow3DRunParser` failure exit codes."""

from __future__ import annotations

import json

from aiida import orm
from aiida.common.links import LinkType

from aiida_pino.parsers.flow3d_run import Flow3DRunParser
from aiida_pino.parsers.flow3d_to_vtk import Flow3DToVtkParser


def _make_calcjob_with_retrieved(
    computer, files: dict[str, str], process_type: str = "aiida.calculations:pino.flow3d_to_vtk"
) -> orm.CalcJobNode:
    """Create a stored ``CalcJobNode`` with a linked retrieved folder."""
    node = orm.CalcJobNode(computer=computer, process_type=process_type)
    node.set_option("output_filename", "aiida.out")
    node.set_option("resources", {"num_machines": 1})
    node.store()
    retrieved = orm.FolderData()
    for name, content in files.items():
        retrieved.base.repository.put_object_from_bytes(content.encode("utf-8"), name)
    retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
    retrieved.store()
    return node


def test_flow3d_parser_missing_summary_returns_310(aiida_localhost) -> None:
    node = _make_calcjob_with_retrieved(aiida_localhost, {"aiida.out": ""})

    parser = Flow3DToVtkParser(node)
    result = parser.parse()

    assert result == node.process_class.spec().exit_codes.ERROR_MISSING_SUMMARY
    assert result.status == 310


def test_flow3d_parser_zero_converted_cases_returns_312(aiida_localhost) -> None:
    summary = {"status": "error", "converted_cases": [], "skipped_cases": ["other"], "error_message": "zero cases"}
    node = _make_calcjob_with_retrieved(aiida_localhost, {"flow3d_summary.json": json.dumps(summary)})

    parser = Flow3DToVtkParser(node)
    result = parser.parse()

    assert result.status == 312
    assert parser.outputs["conversion_summary"].get_dict()["skipped_cases"] == ["other"]


def test_flow3d_parser_partial_failure_returns_311(aiida_localhost) -> None:
    summary = {
        "status": "error",
        "converted_cases": [{"case_name": "case_a"}],
        "failed_cases": [{"case_name": "case_b", "error": "boom"}],
        "error_message": "fatal after partial conversion",
    }
    node = _make_calcjob_with_retrieved(aiida_localhost, {"flow3d_summary.json": json.dumps(summary)})

    parser = Flow3DToVtkParser(node)
    result = parser.parse()

    assert result.status == 311


# --- Flow3DRunParser ----------------------------------------------------------

FLOW3D_RUN_TYPE = "aiida.calculations:pino.flow3d_run"


def test_flow3d_run_parser_missing_summary_returns_520(aiida_localhost) -> None:
    node = _make_calcjob_with_retrieved(aiida_localhost, {"aiida.out": ""}, process_type=FLOW3D_RUN_TYPE)

    result = Flow3DRunParser(node).parse()

    assert result == node.process_class.spec().exit_codes.ERROR_MISSING_SUMMARY
    assert result.status == 520


def test_flow3d_run_parser_error_summary_returns_521_with_summary_output(aiida_localhost) -> None:
    summary = {"status": "error", "case_name": "case_a", "returncode": 2, "error_message": "runner exited"}
    node = _make_calcjob_with_retrieved(
        aiida_localhost, {"flow3d_run_summary.json": json.dumps(summary)}, process_type=FLOW3D_RUN_TYPE
    )

    parser = Flow3DRunParser(node)
    result = parser.parse()

    assert result.status == 521
    # The failed run's summary is still recorded for provenance and reporting.
    assert parser.outputs["run_summary"].get_dict()["error_message"] == "runner exited"


def test_flow3d_run_parser_missing_artifacts_returns_522(aiida_localhost) -> None:
    summary = {"status": "ok", "case_name": "case_a", "returncode": 0}
    node = _make_calcjob_with_retrieved(
        aiida_localhost,
        {"flow3d_run_summary.json": json.dumps(summary), "aiida.out": ""},
        process_type=FLOW3D_RUN_TYPE,
    )

    result = Flow3DRunParser(node).parse()

    assert result.status == 522
