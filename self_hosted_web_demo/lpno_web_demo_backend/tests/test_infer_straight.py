from __future__ import annotations

import base64
import struct

import pytest
from fastapi.testclient import TestClient

from app.adapter_straightpath import STRAIGHT_LENGTH_M
from app.config import Settings
from app.main import create_app
from tests.conftest import skip_without_served_gt

# The straight page's own served ground truth: ti64_straight_path_calibrated_with_keyholes
# (123 cases, 96/14/13), built by scripts/build_calibrated_straight_gt.py. Repo-relative
# so the suite exercises what the demo actually serves.
STRAIGHT_DATA_DIR = "data/straight_path_time/ground_truth_calibrated"
# A real calibrated test-split case (test_70W_0298mps) -> GT available.
GT_P, GT_V = 70.0, 0.298


@pytest.fixture(scope="module")
def client() -> TestClient:
    skip_without_served_gt(STRAIGHT_DATA_DIR, Settings().steady_gt_data_dir)
    settings = Settings(
        startup_load_runtime=True,
        use_stub_runtime=True,
        straight_data_dir=STRAIGHT_DATA_DIR,
    )
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


def _payload(p: float, v: float, **overrides) -> dict:
    payload = {
        "client_request_id": "s1",
        "model": "fno_cal",
        "inputs": {"p_watts": p, "v_scan_mps": v},
        "render": {"width_px": 480, "height_px": 360},
    }
    payload.update(overrides)
    return payload


def _assert_png(b64: str) -> None:
    assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"


def _png_size(b64: str) -> tuple[int, int]:
    raw = base64.b64decode(b64)
    w, h = struct.unpack(">II", raw[16:24])
    return int(w), int(h)


# --------------------------------------------------------------------------- #
# /gt_catalog
# --------------------------------------------------------------------------- #

def test_gt_catalog_straight_lists_the_installed_splits(client: TestClient) -> None:
    resp = client.get("/gt_catalog", params={"dataset": "straight"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["dataset"] == "straight"
    splits = {e["split"] for e in data["entries"]}
    assert {"val", "test"} <= splits <= {"train", "val", "test"}
    # The published bundle holds the held-out cases (14 val + 13 test); a full build adds 96 train.
    assert len(data["entries"]) == (123 if "train" in splits else 27)
    for e in data["entries"]:
        assert e["p_watts"] > 0 and e["v_scan_mps"] > 0
    # The GT test point this suite uses must be present.
    assert any(
        abs(e["p_watts"] - GT_P) < 1 and abs(e["v_scan_mps"] - GT_V) < 1e-3 for e in data["entries"]
    )


def test_gt_catalog_steady_has_points(client: TestClient) -> None:
    resp = client.get("/gt_catalog", params={"dataset": "steady"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset"] == "steady"
    splits = {e["split"] for e in data["entries"]}
    assert {"val", "test"} <= splits <= {"train", "val", "test"}
    # The published bundle holds the held-out cases (14 val + 13 test); a full build adds 96 train.
    assert len(data["entries"]) == (123 if "train" in splits else 27)


def test_gt_catalog_unknown_dataset_422(client: TestClient) -> None:
    resp = client.get("/gt_catalog", params={"dataset": "nope"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --------------------------------------------------------------------------- #
# /infer_straight
# --------------------------------------------------------------------------- #

def test_infer_straight_gt_case_triple_panel(client: TestClient) -> None:
    resp = client.post("/infer_straight", json=_payload(GT_P, GT_V, model="rigno_cal"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "rigno_cal"
    assert len(data["frames"]) == 20
    for b64 in data["frames"]:
        _assert_png(b64)
    sm = data["straight_meta"]
    assert sm["gt_available"] is True
    assert sm["split"] == "test"
    assert sm["gt_name"] == "test_70W_0298mps"
    assert sm["p_watts"] == pytest.approx(GT_P, abs=0.5)
    assert sm["v_scan_mps"] == pytest.approx(GT_V, abs=1e-3)
    assert data["result_meta"]["panel"] == "triple"
    # rigno_cal (uni_rigno_lab_b8_32k_s42) trains on the full domain height, so it
    # ships no grid.npz and serves the dataset grid; a folder that pins a cropped
    # grid.npz still gets its GT cropped to match (model_grid / ground_truth).
    assert data["result_meta"]["grid_shape_xyz"] == [90, 40, 40]
    # A matched GT case uses the case's own 20 frame times from its NPZ.
    assert len(data["times_s"]) == 20
    assert data["times_s"][0] == 0.0


def test_infer_straight_free_pv_single_panel(client: TestClient) -> None:
    # A P/V far from any calibrated case: prediction only, no GT.
    resp = client.post("/infer_straight", json=_payload(123.0, 0.455))
    assert resp.status_code == 200
    data = resp.json()
    sm = data["straight_meta"]
    assert sm["gt_available"] is False
    assert sm["gt_name"] is None
    assert sm["split"] is None
    assert data["result_meta"]["panel"] == "single"
    # Free path: fixed geometry, total time = length / V (laser on throughout).
    assert sm["path_length_m"] == pytest.approx(STRAIGHT_LENGTH_M, rel=1e-9)
    assert sm["laser_off_time_s"] == pytest.approx(STRAIGHT_LENGTH_M / 0.455, rel=1e-6)
    assert len(data["frames"]) == 20


def test_infer_straight_cross_section_plane(client: TestClient) -> None:
    # xz cross-section (like the steady page) instead of the xy top view.
    resp = client.post(
        "/infer_straight",
        json=_payload(GT_P, GT_V, model="rigno_cal", view={"field": "T", "plane": "xz"}),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result_meta"]["panel"] == "triple"  # GT case -> GT/pred/diff
    assert len(data["frames"]) == 20
    _assert_png(data["frames"][10])


def test_infer_straight_field_selection(client: TestClient) -> None:
    resp = client.post("/infer_straight", json=_payload(GT_P, GT_V, view={"field": "fl", "z_index": 5}))
    assert resp.status_code == 200
    data = resp.json()
    assert data["result_meta"]["field"] == "fl"
    assert data["result_meta"]["z_index_used"] == 5


def test_infer_straight_zero_speed_422(client: TestClient) -> None:
    resp = client.post("/infer_straight", json=_payload(120.0, 0.0))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_infer_straight_unknown_model_422(client: TestClient) -> None:
    resp = client.post("/infer_straight", json=_payload(120.0, 0.4, model="deeponet"))
    assert resp.status_code == 422


def test_infer_straight_unknown_field_422(client: TestClient) -> None:
    resp = client.post(
        "/infer_straight", json=_payload(120.0, 0.4, view={"field": "vorticity", "z_index": -1})
    )
    assert resp.status_code == 422


def test_infer_straight_body_too_large_413(client: TestClient) -> None:
    resp = client.post(
        "/infer_straight", content=b"x" * (256 * 1024), headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 413


# --------------------------------------------------------------------------- #
# /infer_straight/render3d
# --------------------------------------------------------------------------- #

def test_infer_straight_render3d_gt_stack(client: TestClient) -> None:
    resp = client.post(
        "/infer_straight/render3d",
        json={
            "model": "rigno_cal",
            "inputs": {"p_watts": GT_P, "v_scan_mps": GT_V},
            "render": {"width_px": 320, "height_px": 240},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["frames"]) == 20
    for b64 in data["frames"]:
        _assert_png(b64)
    assert data["straight_meta"]["gt_available"] is True
    assert data["result_meta"]["panel"] == "3d"
    # GT stack: prediction + ground truth, one layout, ~2 screenshots tall.
    w, h = _png_size(data["frames"][0])
    assert w == 320
    assert h > 1.8 * 240


def test_infer_straight_prediction_cache_reused(client: TestClient) -> None:
    # Two identical straight requests must reuse the single-entry cache.
    body = _payload(GT_P, GT_V, model="fno_cal")
    client.post("/infer_straight", json=body)
    svc = client.app.state.inference_service
    entry = svc._straight_prediction_cache
    assert entry is not None
    client.post("/infer_straight", json=body)
    assert svc._straight_prediction_cache is entry
