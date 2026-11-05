"""Recovering the completed cases of a KILLED batch.

`verdi process kill` skips `finalize`, which is where `Flow3DBatchWorkChain`
files its simulations into a named base dataset. So killing a sweep leaves every
case it had already finished ingested but UNFILED — invisible in the dataset and
folder views, not selectable by name, indistinguishable from cases that were
never run. Measured on one batch, killed at 20 of 40: 20 ingested, 0 filed.

That is the exact state the rule "batch results must land in a named dataset"
exists to prevent, and killing is the CORRECT response to
discovering a design flaw mid-sweep — so the current behaviour penalises the
right call.

Deliberately NOT fixed by an on-kill hook: after a kill you often want to split
valid from invalid before filing. That batch's recovery was two datasets (16
valid, 4 floor-invalid marked do-not-train), which an automatic hook would have
bundled into one.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.orphan_recovery import plan_orphan_filing


def _sim(uuid, case):
    return {"uuid": uuid, "case_name": case}


def test_plan_reports_what_would_be_filed_and_skips_owned_cases():
    plan = plan_orphan_filing(
        completed=[_sim("u1", "kh40_001"), _sim("u2", "kh40_002"), _sim("u3", "kh40_003")],
        owned_uuids={"u2"},
        dataset_name="ti64_keyhole40_partial",
    )
    assert [s["case_name"] for s in plan["to_file"]] == ["kh40_001", "kh40_003"]
    assert plan["already_filed"] == 1
    assert plan["dataset_name"] == "ti64_keyhole40_partial"


def test_a_case_filter_supports_splitting_valid_from_invalid():
    """The real recovery split 20 cases into two datasets; one call each."""
    completed = [_sim(f"u{i}", f"kh40_{i:03d}") for i in range(4)]
    plan = plan_orphan_filing(
        completed=completed, owned_uuids=set(),
        dataset_name="ti64_keyhole40_partial_floor_invalid",
        only_cases=["kh40_002", "kh40_003"],
    )
    assert [s["case_name"] for s in plan["to_file"]] == ["kh40_002", "kh40_003"]


def test_nothing_to_file_is_reported_not_raised():
    plan = plan_orphan_filing(completed=[_sim("u1", "a")], owned_uuids={"u1"}, dataset_name="x")
    assert plan["to_file"] == []
    assert plan["already_filed"] == 1


def test_an_unknown_case_in_the_filter_is_an_error_not_a_silent_drop():
    """A typo in a case name must not quietly file fewer cases than intended."""
    with pytest.raises(ValueError) as excinfo:
        plan_orphan_filing(
            completed=[_sim("u1", "kh40_001")], owned_uuids=set(),
            dataset_name="x", only_cases=["kh40_999"],
        )
    assert "kh40_999" in str(excinfo.value)
