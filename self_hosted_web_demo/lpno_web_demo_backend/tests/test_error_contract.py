from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.service import BackendInferenceError, BusyError, InferenceTimeoutError


class RaisingService:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def health(self):
        return {"runtime_loaded": True, "busy": False, "adapter": "Raising", "runtime": {}}

    async def infer(self, payload, *, server_request_id: str):
        raise self.exc


def _payload() -> dict:
    return {
        "client_request_id": "abc",
        "client_seq": 1,
        "mode": "preview",
        "inputs": {"p_watts": 50, "v_scan_mps": 0.26},
        "view": {"field": "T", "y_index": -1, "show_t_contour": False, "t_contour_k": 1898.0},
        "render": {"width_px": 640, "height_px": 360},
    }


def _client(exc: Exception) -> TestClient:
    # Use as `with _client(exc) as client:` — the injected service is wired in
    # the app lifespan, which TestClient only runs inside its context manager.
    settings = Settings(startup_load_runtime=False, use_stub_runtime=True)
    app = create_app(settings=settings, service=RaisingService(exc))
    return TestClient(app)


def test_busy_error_contract() -> None:
    with _client(BusyError("Inference runtime is busy")) as client:
        resp = client.post("/infer", json=_payload())
    assert resp.status_code == 429
    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "busy"
    assert data["error"]["retryable"] is True


def test_timeout_error_contract() -> None:
    with _client(InferenceTimeoutError("timed out")) as client:
        resp = client.post("/infer", json=_payload())
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "timeout"


def test_internal_error_contract() -> None:
    with _client(BackendInferenceError("boom")) as client:
        resp = client.post("/infer", json=_payload())
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal"
