from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import skip_without_served_gt

# The served calibrated steady set (the Settings default): 123 cases, 96/14/13.
STEADY_GT_DATA_DIR = Settings().steady_gt_data_dir
# A calibrated test-split case (test_70W_0298mps): P=70 W, V=0.298 m/s -> GT available.
GT_P, GT_V = 70.0, 0.298


@pytest.fixture(scope="module")
def client() -> TestClient:
    skip_without_served_gt(STEADY_GT_DATA_DIR)
    settings = Settings(
        startup_load_runtime=True,
        use_stub_runtime=True,
        steady_gt_data_dir=STEADY_GT_DATA_DIR,
    )
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


def _payload(p: float, v: float, **overrides) -> dict:
    payload = {
        "inputs": {"p_watts": p, "v_scan_mps": v},
        "field": "T",
        "y_index": -1,
        "render": {"width_px": 480, "height_px": 360},
    }
    payload.update(overrides)
    return payload


def _assert_png(b64: str) -> None:
    assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_gt_compare_matched_case_two_panel(client: TestClient) -> None:
    resp = client.post("/infer/gt_compare", json=_payload(GT_P, GT_V))
    assert resp.status_code == 200
    data = resp.json()
    rm = data["result_meta"]
    assert rm["panel"] == "gt_compare"
    assert rm["gt_available"] is True
    assert rm["gt_p_watts"] == pytest.approx(GT_P, abs=0.5)
    assert rm["gt_v_scan_mps"] == pytest.approx(GT_V, abs=1e-3)
    _assert_png(data["plot"]["png_base64"])
    assert set(data["timing_ms"]) == {"infer", "render", "total"}


def test_gt_compare_free_pv_single_panel(client: TestClient) -> None:
    # A P/V with no steady test case: prediction only, no GT panel.
    resp = client.post("/infer/gt_compare", json=_payload(123.0, 0.72))
    assert resp.status_code == 200
    rm = resp.json()["result_meta"]
    assert rm["panel"] == "single"
    assert rm["gt_available"] is False
    assert rm["gt_p_watts"] is None


def test_gt_compare_field_alpha(client: TestClient) -> None:
    resp = client.post("/infer/gt_compare", json=_payload(GT_P, GT_V, field="alpha"))
    assert resp.status_code == 200
    assert resp.json()["result_meta"]["field"] == "alpha"


def test_gt_compare_unknown_field_422(client: TestClient) -> None:
    resp = client.post("/infer/gt_compare", json=_payload(GT_P, GT_V, field="vorticity"))
    assert resp.status_code == 422


def test_gt_compare_body_too_large_413(client: TestClient) -> None:
    resp = client.post(
        "/infer/gt_compare", content=b"x" * (32 * 1024), headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 413


def test_gt_compare_reports_plot_mode_2d(client: TestClient) -> None:
    # The default (no plot_mode) stays the 2D stacked panels and now echoes it.
    resp = client.post("/infer/gt_compare", json=_payload(GT_P, GT_V))
    assert resp.status_code == 200
    assert resp.json()["result_meta"]["plot_mode_used"] == "2d"


def test_gt_compare_rejects_bad_plot_mode(client: TestClient) -> None:
    resp = client.post("/infer/gt_compare", json=_payload(GT_P, GT_V, plot_mode="iso"))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
