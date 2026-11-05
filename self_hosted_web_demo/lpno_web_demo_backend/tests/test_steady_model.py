from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

# A steady test case (P=110 W, V=0.340 m/s) -> GT available for gt_compare.
GT_P, GT_V = 110.0, 0.340


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Real InferenceService in stub mode so _validate_steady_model runs against
    # the discovered steady models (data/steady-state/models -> "fno").
    settings = Settings(
        startup_load_runtime=True,
        use_stub_runtime=True,
        steady_gt_data_dir="data/steady-state/ground_truth",
    )
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


def _infer_payload(**overrides) -> dict:
    payload = {
        "client_request_id": "m",
        "client_seq": 1,
        "mode": "final",
        "inputs": {"p_watts": 80.0, "v_scan_mps": 0.30},
        "view": {"field": "T", "plot_mode": "2d", "y_index": -1},
        "render": {"width_px": 320, "height_px": 240},
    }
    payload.update(overrides)
    return payload


def test_infer_accepts_discovered_model(client: TestClient) -> None:
    resp = client.post("/infer", json=_infer_payload(model="fno_cal"))
    assert resp.status_code == 200


def test_infer_accepts_empty_model_backward_compat(client: TestClient) -> None:
    # Older clients omit `model` entirely -> served by the loaded runtime.
    resp = client.post("/infer", json=_infer_payload())
    assert resp.status_code == 200


def test_infer_rejects_unknown_model(client: TestClient) -> None:
    resp = client.post("/infer", json=_infer_payload(model="deeponet"))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_gt_compare_rejects_unknown_model(client: TestClient) -> None:
    payload = {
        "model": "deeponet",
        "inputs": {"p_watts": GT_P, "v_scan_mps": GT_V},
        "field": "T",
        "render": {"width_px": 320, "height_px": 240},
    }
    resp = client.post("/infer/gt_compare", json=payload)
    assert resp.status_code == 422


def test_models_endpoint_lists_steady(client: TestClient) -> None:
    resp = client.get("/models?page=steady")
    assert resp.status_code == 200
    models = resp.json()["models"]
    ids = [m["id"] for m in models]
    assert "fno_cal" in ids
    # The RIGNO counterpart (rigno_M) is provisioned on the steady page too.
    assert "rigno_cal" in ids
    # Exactly one model is the dropdown default (which one is a meta.json choice).
    assert sum(1 for m in models if m["default"]) == 1


def test_infer_accepts_rigno_model(client: TestClient) -> None:
    # Stub mode serves the stub runtime for any valid id; this checks the id is
    # accepted end-to-end (real RIGNO load-by-id is validated live, not in CI).
    resp = client.post("/infer", json=_infer_payload(model="rigno_cal"))
    assert resp.status_code == 200
