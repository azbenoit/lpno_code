from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeService:
    def health(self):
        return {"runtime_loaded": True, "busy": False, "adapter": "Fake", "runtime": {"device": "cpu"}}

    async def infer(self, payload, *, server_request_id: str):  # pragma: no cover
        raise NotImplementedError


def test_healthz_returns_runtime_summary() -> None:
    app = create_app(settings=Settings(startup_load_runtime=False, use_stub_runtime=True), service=FakeService())
    # `with` runs the app lifespan, which wires the injected service into
    # app.state; a bare TestClient skips it and /healthz would see no service.
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["runtime_loaded"] is True
    assert data["runtime"]["device"] == "cpu"
