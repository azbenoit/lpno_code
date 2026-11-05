from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import (
    ClampedInputs,
    DerivedValues,
    InferResponse,
    PlotPayload,
    ResultMeta,
    TimingMs,
)


class FakeService:
    def health(self):
        return {"runtime_loaded": True, "busy": False, "adapter": "Fake", "runtime": {"mode": "fake"}}

    async def infer(self, payload, *, server_request_id: str) -> InferResponse:
        return InferResponse(
            client_request_id=payload.client_request_id,
            client_seq=payload.client_seq,
            mode=payload.mode,
            server_request_id=server_request_id,
            clamped_inputs=ClampedInputs(p_watts=payload.inputs.p_watts, v_scan_mps=payload.inputs.v_scan_mps),
            derived=DerivedValues(h_model=1.0, h_star=2.0, h_norm_model=3.0),
            result_meta=ResultMeta(
                field=payload.view.field,
                plot_mode_used=payload.view.plot_mode,
                grid_shape_xyz=(10, 10, 10),
                y_index_used=0,
            ),
            plot=PlotPayload(width_px=payload.render.width_px, height_px=payload.render.height_px, png_base64=base64.b64encode(b"PNG").decode("ascii")),
            timing_ms=TimingMs(infer=1.0, render=2.0, total=3.0),
        )


def _client() -> TestClient:
    # Use as `with _client() as client:` — the injected FakeService is wired in
    # the app lifespan, which TestClient only runs inside its context manager.
    settings = Settings(startup_load_runtime=False, use_stub_runtime=True)
    app = create_app(settings=settings, service=FakeService())
    return TestClient(app)


def test_infer_success_response_schema() -> None:
    payload = {
        "client_request_id": "abc",
        "client_seq": 1,
        "mode": "final",
        "inputs": {"p_watts": 50, "v_scan_mps": 0.26},
        "view": {"field": "T", "plot_mode": "3d", "y_index": -1, "show_t_contour": True, "t_contour_k": 1898.0},
        "render": {"width_px": 640, "height_px": 360},
    }
    with _client() as client:
        resp = client.post("/infer", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["client_seq"] == 1
    assert data["plot"]["mime_type"] == "image/png"
    assert data["result_meta"]["plot_mode_used"] == "3d"


def test_infer_rejects_unknown_fields() -> None:
    payload = {
        "client_request_id": "abc",
        "client_seq": 1,
        "mode": "preview",
        "inputs": {"p_watts": 50, "v_scan_mps": 0.26, "extra": 1},
        "view": {"field": "T", "plot_mode": "2d", "y_index": -1, "show_t_contour": False, "t_contour_k": 1800},
        "render": {"width_px": 640, "height_px": 360},
    }
    with _client() as client:
        resp = client.post("/infer", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "validation_error"


def test_infer_rejects_unknown_plot_mode() -> None:
    payload = {
        "client_request_id": "abc",
        "client_seq": 1,
        "mode": "preview",
        "inputs": {"p_watts": 50, "v_scan_mps": 0.26},
        "view": {"field": "T", "plot_mode": "iso", "y_index": -1, "show_t_contour": False, "t_contour_k": 1800},
        "render": {"width_px": 640, "height_px": 360},
    }
    with _client() as client:
        resp = client.post("/infer", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
