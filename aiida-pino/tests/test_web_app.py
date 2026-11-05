"""Contract tests for the web console FastAPI app (no AiiDA profile needed).

Uses an injected stub service so create_app builds without loading a profile —
covers routing, the shared error envelope, CORS pinning, body-size limits, and
request-schema validation.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from web.app.config import Settings
from web.app.errors import BackendError, InvalidRequestError, NodeNotFoundError
from web.app.main import create_app


class StubService:
    def __init__(self):
        self.calls = {}
        self.models_error = True  # test_internal_error_is_redacted relies on GET /models raising

    def health(self):
        return {"service": "stub", "profile": "test", "daemon": {"available": False, "reason": "stub"}}

    def list_models(self, **kwargs):
        self.calls["list_models"] = kwargs
        if self.models_error:
            raise BackendError("secret /db/connection?password=hunter2")
        return []

    def list_simulations(self, **kwargs):
        self.calls["list_simulations"] = kwargs
        return []

    def update_annotations(self, pk, **kwargs):
        self.calls["update_annotations"] = {"pk": pk, **kwargs}
        if pk == 404:
            raise NodeNotFoundError(f"node {pk} not found")
        if kwargs.get("split") not in (None, "train", "val", "test", "unassigned"):
            raise InvalidRequestError(f"invalid split {kwargs['split']!r}")
        return {"pk": pk, "kind": "simulation", "label": "sim", "description": kwargs.get("description"),
                "keywords": kwargs.get("keywords_set") or [], "split": kwargs.get("split") or "unassigned"}

    def create_collection(self, simulation_pks, collection_name, *, description=None,
                          keywords=None, dataset_pks=None, skip_description=False,
                          field_policy=None):
        self.calls["create_collection"] = {
            "simulation_pks": simulation_pks, "collection_name": collection_name,
            "description": description, "keywords": keywords, "dataset_pks": dataset_pks,
            "skip_description": skip_description, "field_policy": field_policy,
        }
        return {"pk": 7, "uuid": "u-7", "created": True}

    def assign_splits(self, pk, *, train, val, test, only_unassigned=True, seed=0):
        self.calls["assign_splits"] = {
            "pk": pk, "train": train, "val": val, "test": test,
            "only_unassigned": only_unassigned, "seed": seed,
        }
        return {"pk": pk, "total": 10, "assigned": {"train": 8, "val": 1, "test": 1}, "skipped": 0}

    def env_status(self):
        return {"lpno_version": "9.9", "profile": "test", "daemon": {"available": False, "reason": "stub"},
                "codes": {"all": [], "by_role": {"python": [], "openfoam": [], "pvpython": []}}, "config_names": ["c1"]}

    def list_datasets(self, **kwargs):
        self.calls["list_datasets"] = kwargs
        return [{"pk": 1, "label": "ds1", "dataset_role": "base"}]

    def evict_dataset_data(self, pk, *, dry_run=True, reason=None):
        if dry_run:
            return {"pk": pk, "dry_run": True, "members": 3, "would_evict": 3, "already_evicted": 0, "reclaimable_bytes": 1_000_000}
        return {"pk": pk, "dry_run": False, "members": 3, "evicted": 3, "bytes_freed": 1_000_000, "reclaim_hint": "run maintenance"}

    def dataset_detail(self, pk):
        raise NodeNotFoundError(f"node {pk} not found")

    def simulation_stats(self, pk, recompute=False, compute=False):
        # Computing is opt-in: the real one reads every frame of the case.
        if not (compute or recompute):
            return {"pk": pk, "stats": None, "computed": False}
        return {"pk": pk, "stats": {"regime": "conduction", "melt_vol_um3": 1e6, "version": 1}, "computed": True}

    def simulation_plot(self, pk, *, field, view="long", frame=-1, contour=True):
        self.calls.setdefault("plot", []).append((pk, field, view, frame, contour))
        return b"\x89PNG\r\n\x1a\n_stub"

    def dataset_stats(self, pk):
        return {"pk": pk, "summary": {"total": 2, "computed": 1}, "cases": []}

    def dataset_stats_compute(self, pk, limit=25):
        return {"pk": pk, "computed_now": 1, "remaining": 0, "total": 2}

    def dataset_stats_plot(self, pk, *, kind="regime"):
        return b"\x89PNG\r\n\x1a\n_stub"

    def dataset_cases(self, pk):
        self.calls.setdefault("dataset_cases", []).append(pk)
        if pk == 404404:
            raise NodeNotFoundError(f"node {pk} not found")
        return {"pk": pk, "count": 1, "cases": [{"pk": 7, "case_name": "case_a", "split": "train", "keywords": []}]}

    def preview_collection(self, simulation_pks):
        raise InvalidRequestError("bad members")

    def flow3d_queue_status(self):
        return {"spool_dir": "/tmp/spool", "exists": True, "seat_held": True,
                "queued": [{"name": "case_a", "added": "t"}], "queued_count": 1, "running": []}

    def monitor_status(self):
        self.calls["monitor_status"] = True
        return {"gpus": [], "gpu_note": "nvidia-smi unavailable: stub", "cpu": {"load_1m": 1.0, "cores": 2},
                "active_processes": [], "gpu_spool": [], "flow3d": self.flow3d_queue_status()}


def _client(pages_origin: str = "", service: StubService | None = None) -> TestClient:
    settings = Settings(pages_origin=pages_origin, startup_load_service=False)
    return TestClient(create_app(settings=settings, service=service or StubService()))


def test_healthz_ok_with_stub():
    with _client() as c:
        b = c.get("/healthz").json()
        assert b["status"] == "ok"
        assert b["service_loaded"] is True
        assert b["service"] == "stub"


def test_healthz_service_none_when_not_loaded():
    settings = Settings(startup_load_service=False)
    with TestClient(create_app(settings=settings, service=None)) as c:
        b = c.get("/healthz").json()
        assert b["status"] == "ok" and b["service_loaded"] is False


def test_env_and_datasets_shapes():
    with _client() as c:
        env = c.get("/env").json()
        assert env["codes"]["by_role"]["python"] == []
        rows = c.get("/datasets").json()["datasets"]
        assert rows[0]["pk"] == 1


def test_node_not_found_maps_to_404_envelope():
    with _client() as c:
        r = c.get("/datasets/123")
        assert r.status_code == 404
        body = r.json()
        assert body["status"] == "error"
        assert body["error"]["code"] == "not_found"
        assert "server_request_id" in body


def test_invalid_request_maps_to_422():
    with _client() as c:
        r = c.post("/preview/collection", json={"simulation_pks": [1]})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "validation_error"


def test_body_size_limit_413():
    with _client() as c:
        big = {"simulation_pks": list(range(30000))}  # serializes well past the 64 KB default cap
        r = c.post("/preview/collection", json=big)
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "payload_too_large"


def test_cors_header_present_when_pages_origin_set():
    origin = "http://10.0.0.5:8180"
    with _client(pages_origin=origin) as c:
        r = c.options(
            "/env",
            headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
        )
        assert r.headers.get("access-control-allow-origin") == origin


def test_cors_absent_by_default():
    with _client() as c:
        r = c.get("/env", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in {k.lower(): v for k, v in r.headers.items()}


def test_request_id_echoed():
    with _client() as c:
        r = c.get("/healthz", headers={"X-Request-ID": "abc123"})
        assert r.headers.get("X-Request-ID") == "abc123"


def test_annotations_happy_path():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.post("/nodes/5/annotations", json={"split": "train", "keywords_add": ["keyhole"]})
        assert r.status_code == 200
        body = r.json()
        assert body["pk"] == 5 and body["split"] == "train"
        recorded = stub.calls["update_annotations"]
        assert recorded["pk"] == 5
        assert recorded["split"] == "train"
        assert recorded["keywords_add"] == ["keyhole"]
        assert recorded["description"] is None and recorded["keywords_set"] is None and recorded["keywords_remove"] is None


def test_annotations_title_passthrough():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.post("/nodes/6/annotations", json={"title": "AL round 3 retrain"})
        assert r.status_code == 200
        assert stub.calls["update_annotations"]["title"] == "AL round 3 retrain"


def test_evict_endpoint_dry_run_and_apply() -> None:
    with _client() as client:
        dry = client.post("/datasets/5/evict", json={"dry_run": True}).json()
        assert dry["would_evict"] == 3 and dry["reclaimable_bytes"] == 1_000_000
        done = client.post("/datasets/5/evict", json={"dry_run": False, "reason": "x"}).json()
        assert done["evicted"] == 3


def test_monitor_endpoint():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.get("/monitor")
        assert r.status_code == 200
        body = r.json()
        assert stub.calls.get("monitor_status") is True
        assert body["flow3d"]["queued_count"] == 1
        assert "gpus" in body and "cpu" in body and "active_processes" in body


def test_annotations_unknown_pk_404():
    with _client() as c:
        r = c.post("/nodes/404/annotations", json={"split": "train"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"


def test_annotations_bad_split_422():
    with _client() as c:
        r = c.post("/nodes/5/annotations", json={"split": "bogus"})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "validation_error"


def test_annotations_empty_body_422():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.post("/nodes/5/annotations", json={})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "validation_error"
        assert "update_annotations" not in stub.calls  # rejected by the schema, service never called


def test_simulations_filter_params_forwarded():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.get("/simulations?source=flow3d&split=train&keyword=a&keyword=b")
        assert r.status_code == 200 and r.json()["simulations"] == []
        kwargs = stub.calls["list_simulations"]
        assert kwargs["source"] == "flow3d"
        assert kwargs["split"] == "train"
        assert kwargs["keywords"] == ["a", "b"]


def test_dataset_and_model_keyword_params_forwarded():
    stub = StubService()
    stub.models_error = False
    with _client(service=stub) as c:
        assert c.get("/datasets?keyword=k1&keyword=k2").status_code == 200
        assert stub.calls["list_datasets"]["keywords"] == ["k1", "k2"]
        assert c.get("/models?keyword=m1").status_code == 200
        assert stub.calls["list_models"]["keywords"] == ["m1"]


def test_collection_create_forwards_annotations():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.post("/collections", json={
            "simulation_pks": [1, 2], "collection_name": "coll",
            "description": "filtered set", "keywords": ["keyhole", "smoke"],
        })
        assert r.status_code == 200 and r.json()["pk"] == 7
        recorded = stub.calls["create_collection"]
        assert recorded["simulation_pks"] == [1, 2]
        assert recorded["collection_name"] == "coll"
        assert recorded["description"] == "filtered set"
        assert recorded["keywords"] == ["keyhole", "smoke"]


def test_collection_create_accepts_dataset_pks():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.post("/collections", json={"collection_name": "combo", "dataset_pks": [11, 12]})
        assert r.status_code == 200
        recorded = stub.calls["create_collection"]
        assert recorded["dataset_pks"] == [11, 12]
        assert recorded["simulation_pks"] == []


def test_assign_splits_route():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.post("/datasets/5/assign_splits", json={"train": 0.8, "val": 0.1, "test": 0.1})
        assert r.status_code == 200
        assert r.json()["assigned"]["train"] == 8
        recorded = stub.calls["assign_splits"]
        assert recorded["pk"] == 5 and recorded["only_unassigned"] is True and recorded["seed"] == 0


def test_internal_error_is_redacted():
    # BackendError (and any bare ConsoleError) must not echo its raw text to the
    # client — only the exception class name, so paths/connection strings stay
    # server-side.
    with _client() as c:
        r = c.get("/models")
        assert r.status_code == 500
        body = r.json()
        assert body["error"]["code"] == "internal"
        assert body["error"]["message"] == "Internal error: BackendError"
        assert "secret" not in json.dumps(body) and "hunter2" not in json.dumps(body)


def test_dataset_cases_route():
    stub = StubService()
    with _client(service=stub) as c:
        body = c.get("/datasets/5/cases").json()
        assert body["count"] == 1 and body["cases"][0]["case_name"] == "case_a"
        assert stub.calls["dataset_cases"] == [5]
        r = c.get("/datasets/404404/cases")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"


def test_viz_and_stats_routes():
    stub = StubService()
    with _client(service=stub) as c:
        r = c.get("/simulations/9/plot?field=Temperature&view=surface&frame=3&contour=false")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert stub.calls["plot"] == [(9, "Temperature", "surface", 3, False)]
        # Cheap by default, and the caller asks for the work explicitly.
        assert c.get("/simulations/9/stats").json() == {"pk": 9, "stats": None, "computed": False}
        assert c.get("/simulations/9/stats?compute=1").json()["stats"]["regime"] == "conduction"
        assert c.get("/datasets/5/stats").json()["summary"]["computed"] == 1
        assert c.post("/datasets/5/stats/compute?limit=10").json()["remaining"] == 0
        r2 = c.get("/datasets/5/stats/plot?kind=regime")
        assert r2.status_code == 200 and r2.headers["content-type"] == "image/png"


def test_locked_database_maps_to_retryable_503():
    # A locked SQLite profile is transient. It must surface as a 503 handled
    # inside the CORS middleware (so the browser reads the message) rather than
    # a header-less 500 that the client can only report as "Failed to fetch".
    from sqlalchemy.exc import OperationalError

    stub = StubService()

    def locked(**kwargs):
        raise OperationalError("SELECT 1", {}, Exception("database is locked"))

    stub.list_models = locked
    with _client(service=stub) as c:
        r = c.get("/models")
        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "database_locked"
        assert body["error"]["retryable"] is True
        assert "locked" in body["error"]["message"].lower()


def test_non_lock_operational_error_stays_internal():
    from sqlalchemy.exc import OperationalError

    stub = StubService()

    def broken(**kwargs):
        raise OperationalError("SELECT 1", {}, Exception("no such table: db_dbnode"))

    stub.list_models = broken
    with _client(service=stub) as c:
        r = c.get("/models")
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "internal"


def test_cors_allows_the_loopback_alias_of_the_page_origin():
    """The README's own quick-start URL served a page that could not fetch.

    In LAN mode config.local.json pins the backend to the workstation IP and
    CORS allowed only that origin, so http://127.0.0.1:8180/index.html loaded
    fine and then failed every request with "Failed to fetch" — no mention of
    CORS, the config file, or the URL that works.
    """
    from aiida_pino.console.config import Settings

    origins = Settings(pages_origin="http://10.0.0.5:8180").cors_allow_origins
    assert origins[0] == "http://10.0.0.5:8180"
    assert "http://127.0.0.1:8180" in origins and "http://localhost:8180" in origins
    # no origin configured stays no origin: CORS off, not CORS open
    assert Settings(pages_origin="").cors_allow_origins == []


def test_code_role_classifies_shell_executable():
    from aiida_pino.console.service import _code_role

    assert _code_role("/bin/bash", "bash-box2@box2") == "shell"
    assert _code_role("/opt/miniconda3/envs/pino/bin/python", "python@localhost") == "python"
    assert _code_role("/opt/paraview/bin/pvpython", "pv@localhost") == "pvpython"
