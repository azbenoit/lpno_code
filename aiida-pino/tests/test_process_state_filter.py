"""The Runs state filter runs in the query, not over the fetched page.

Review criticism K: the front end fetched 100 rows and filtered those, so
selecting "failed / killed" showed 2 rows on a database with 395 killed
processes — and read as "nothing was ever killed".
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from aiida import orm

from aiida_pino.console.errors import InvalidRequestError
from aiida_pino.console.service import AiidaConsoleService, _process_category_filter, _process_state_filter
from aiida_pino.console.config import Settings


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


def test_failed_covers_every_way_a_run_can_fail():
    """`failed` is not an AiiDA process state: it is excepted, or killed, or
    finished with a non-zero exit. Missing any of the three is how a failure
    disappears from the list."""
    built = _process_state_filter("failed")
    dumped = str(built)
    assert "excepted" in dumped and "killed" in dumped
    assert "exit_status" in dumped  # the finished-but-nonzero arm


def test_finished_ok_accepts_a_missing_exit_status():
    """Workchains that finish cleanly have no `exit_status` attribute at all;
    treating "ok" as `exit_status == 0` alone would hide most of them. Written
    with has_key because a `== None` comparison raises TypeError on the SQLite
    backend while quietly working on PostgreSQL."""
    built = _process_state_filter("finished")
    assert built["attributes.process_state"] == "finished"
    assert {"attributes": {"!has_key": "exit_status"}} in built["or"]


def test_all_and_empty_mean_no_filter():
    assert _process_state_filter(None) == {}
    assert _process_state_filter("") == {}
    assert _process_state_filter("all") == {}


def test_an_unknown_state_is_rejected_not_ignored():
    """Silently ignoring it would show every run under a filter that claims
    otherwise — the failure mode this whole fix is about."""
    with pytest.raises(InvalidRequestError):
        _process_state_filter("borked")


def test_killed_runs_are_found_beyond_the_first_page(svc, aiida_localhost):
    """The regression itself: a killed run that is *older* than a page of
    finished ones must still be found."""
    killed = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:pino.train")
    killed.set_option("resources", {"num_machines": 1})
    killed.set_process_state("killed")
    killed.store()
    killed.seal()
    for _ in range(12):  # bury it under newer, finished runs
        node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:pino.train")
        node.set_option("resources", {"num_machines": 1})
        node.set_process_state("finished")
        node.set_exit_status(0)
        node.store()
        node.seal()

    assert killed.pk not in {r["pk"] for r in svc.list_processes(limit=5)}
    assert killed.pk in {r["pk"] for r in svc.list_processes(limit=5, state="failed")}
    assert killed.pk not in {r["pk"] for r in svc.list_processes(limit=5, state="finished")}


def test_superseded_datasets_are_findable_when_asked_for(svc, pino_simulation_factory):
    """Supersession retires about half the collections and hides them from every
    listing, so a replaced dataset was indistinguishable from one that never
    existed (review, section 5). It must be reachable on request, and labelled."""
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    sims = [pino_simulation_factory("retired", group_name=f"rt-{i}") for i in range(2)]
    meta = build_collection_metadata_from_simulations(sims, collection_name="retired")
    node, _ = get_or_create_dataset_collection(sims, metadata=meta)
    node.base.extras.set("superseded", True)
    node.base.extras.set("replaced_by_pk", 12345)

    assert node.pk not in {r["pk"] for r in svc.list_datasets(role="all", limit=0)}
    rows = {r["pk"]: r for r in svc.list_datasets(role="all", limit=0, include_superseded=True)}
    assert node.pk in rows
    assert rows[node.pk]["superseded"] is True
    assert rows[node.pk]["replaced_by_pk"] == 12345
    # and the detail pane says so too, for anything reached by a direct link
    assert svc.dataset_detail(node.pk)["superseded"] is True


def test_clamp_limit_returns_none_for_no_cap():
    """The sentinel is None, not 0.

    0 was a landmine: it is falsy but it is also a *valid* argument to
    `QueryBuilder.limit()` and to a slice, where it means "nothing". Every call
    site had to remember a guard and two of four forgot, so "Show all" answered
    zero models, zero processes and zero simulations. None is correct in all
    three positions with no guard at all.
    """
    settings = Settings()
    assert settings.clamp_limit(0) is None
    # -1 used to inherit the sentinel through `if limit <= 0`, so a `parseInt`
    # over an empty input fired the most expensive query in the store. Only 0
    # means "no cap"; anything below it is a bad request.
    with pytest.raises(InvalidRequestError):
        settings.clamp_limit(-1)
    assert settings.clamp_limit(None) == settings.list_default_limit
    assert settings.clamp_limit(20) == 20


def test_limit_zero_means_no_cap_on_every_listing(
    svc, pino_model_factory, pino_simulation_factory, pino_dataset_factory, aiida_localhost
):
    """Checked on all four listings, not just the one that was reported: this
    bug was found twice, in two different listings, for the same reason."""
    pino_simulation_factory("nocap", group_name="nc-1")
    pino_dataset_factory("nocap-ds")
    pino_model_factory()
    node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:pino.train")
    node.set_option("resources", {"num_machines": 1})
    node.set_process_state("finished")
    node.set_exit_status(0)
    node.store()
    node.seal()

    for name, listing in (
        ("models", svc.list_models),
        ("processes", svc.list_processes),
        ("simulations", svc.list_simulations),
    ):
        assert len(listing(limit=0)) >= 1, f"{name} returned nothing for the no-cap request"
        assert len(listing(limit=1)) == 1, f"{name} ignored a real cap"
    assert len(svc.list_datasets(limit=0, role="all")) >= 1


def test_flow3d_queue_reports_the_aiida_side_too(svc, aiida_localhost):
    """The spool is the manual fallback; AiiDA is the default entry point.

    Both take the same seat lock, but an AiiDA-launched solve writes no `.task`
    file — so a spool-only view said "0 queued" on 2026-08-02 while four
    Flow3DBatchWorkChains had been waiting since 07-31, three run calcjobs were
    blocked on the seat, and a fourth was solving.
    """
    node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:pino.flow3d_run")
    node.set_option("resources", {"num_machines": 1})
    node.base.attributes.set("process_label", "Flow3DRunCalculation")
    node.set_process_state("waiting")
    node.store()

    status = svc.flow3d_queue_status()
    assert "aiida" in status
    assert status["aiida"]["run_count"] >= 1
    assert node.pk in {r["pk"] for r in status["aiida"]["runs"]}
    # the spool half still reports independently, and the total spans both
    assert status["total_waiting"] >= status["queued_count"]


def test_health_reports_the_commit_this_process_is_serving(svc, aiida_localhost):
    """"Fixed" and "fixed for the user" drifted by three days and four fixes,
    because a long-lived uvicorn keeps the code it imported at startup and
    nothing exposed that. `served_sha` vs HEAD is the check that catches it."""
    code = svc.health()["code"]
    assert set(code) >= {"served_sha", "started_utc"}
    # a checkout has a sha; a tarball deployment says None rather than guessing
    assert code["served_sha"] is None or len(code["served_sha"]) >= 7


def test_tail_path_cannot_escape_the_job_directory(svc, aiida_localhost):
    """`path` is caller-supplied on a service that listens on 0.0.0.0 with no
    auth. `PurePosixPath(base) / "../../../etc/hostname"` escapes happily, and
    shlex.quote stops injection but not traversal:
    GET /processes/{pk}/tail?path=../../../../../etc/hostname returned the file.
    """
    node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:pino.train")
    node.set_option("resources", {"num_machines": 1})
    node.set_process_state("running")
    node.store()

    for bad in ("../../../../../etc/hostname", "/etc/hostname", "logs/../../../../etc/passwd"):
        with pytest.raises(InvalidRequestError, match=r"\.\.|relative"):
            svc.process_tail(node.pk, path=bad)


def test_category_all_means_no_filter_and_unknown_is_rejected():
    assert _process_category_filter(None) == {}
    assert _process_category_filter("all") == {}
    with pytest.raises(InvalidRequestError):
        _process_category_filter("borked")


def test_generation_runs_are_found_beyond_the_first_page(svc, aiida_localhost):
    """The Runs type filter used to apply to the fetched page, so 'data
    generation (FLOW-3D)' showed 'No rows.' whenever the 100 most recent runs
    were all evals. It is now a query filter, like `state`."""

    def store(label):
        node = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:pino.train")
        node.set_option("resources", {"num_machines": 1})
        node.set_process_label(label)
        node.set_process_state("finished")
        node.set_exit_status(0)
        node.store()
        node.seal()
        return node

    gen = store("Flow3DRunCalculation")
    importer = store("ImportModelCalculation")  # models, not conversion: same precedence as app.js
    for _ in range(12):
        store("PinoEvalCalculation")

    def pks(limit=5, **kw):
        return {r["pk"] for r in svc.list_processes(limit=limit, **kw)}

    assert gen.pk not in pks()
    assert gen.pk in pks(category="generation")
    assert gen.pk not in pks(category="models")
    assert importer.pk in pks(category="models", state="finished", limit=20)
    assert importer.pk not in pks(category="conversion")
    assert gen.pk not in pks(category="other") and importer.pk not in pks(category="other")
