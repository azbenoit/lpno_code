from __future__ import annotations

import base64
import math
import struct

import numpy as np
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapter_randompath import (
    KEEPOUT_X,
    KEEPOUT_Y,
    MAX_FRAMES,
    MAX_PATH_LENGTH_M,
    MIN_FRAMES,
    prepare_path,
)
from app.config import Settings
from app.main import create_app
from tests.conftest import skip_without_served_gt

# The repo's own served ground truth, not an absolute path into a sibling
# checkout: those files are instance-specific and a test that cites one fails on
# any other machine.
PATH_DATA_DIR = str(Path(__file__).resolve().parents[1] / "data" / "random_path" / "ground_truth")
# A calibrated (rand150_newphys) GT case -- the legacy tial64_* set is no
# longer served, and its models were archived with it.
PRESET = "r150np_010"


@pytest.fixture(scope="module")
def client() -> TestClient:
    skip_without_served_gt(PATH_DATA_DIR)
    settings = Settings(
        startup_load_runtime=True,
        use_stub_runtime=True,
        path_data_dir=PATH_DATA_DIR,
    )
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


def _drawn_payload(points, **overrides) -> dict:
    payload = {
        "client_request_id": "t1",
        "model": "fno_newphys",
        "path": {"points_xy_m": points},
        "render": {"width_px": 480, "height_px": 360},
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# prepare_path unit tests (clamp / trim / resample with exact expectations)
# --------------------------------------------------------------------------- #

def test_prepare_path_trims_3mm_zigzag_to_1p5mm() -> None:
    # 3 mm of arc inside the keep-out box: 5 vertical legs of 0.6 mm each.
    ymin, ymax = KEEPOUT_Y
    xs = np.linspace(-4e-4, 4e-4, 6)
    points = [[float(x), float(ymin if i % 2 == 0 else ymax)] for i, x in enumerate(xs)]
    # sanity: raw arc length > 3 mm
    raw = np.asarray(points)
    raw_len = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum())
    assert raw_len > 3.0e-3

    out, length, clamped = prepare_path(points)
    assert length == pytest.approx(MAX_PATH_LENGTH_M, abs=1e-12)
    assert clamped is True  # trimming reports as clamped
    # ~25 um arc-length-uniform spacing -> 61 points for 1.5 mm
    assert out.shape == (61, 2)
    # Uniform in arc length along the polyline: straight-leg chords equal the
    # 25 um spacing exactly; chords spanning a corner are shorter.
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.all(seg <= length / 60.0 + 1e-12)
    assert np.isclose(np.median(seg), length / 60.0, rtol=1e-9)


def test_prepare_path_clamps_off_domain_straight_line() -> None:
    # Straight 3 mm line along y=0, far beyond the box in x: clamps to the box
    # x-extent, so the usable path is exactly the keep-out x-span (1.1 mm).
    out, length, clamped = prepare_path([[-1.5e-3, 0.0], [1.5e-3, 0.0]])
    assert clamped is True
    assert length == pytest.approx(KEEPOUT_X[1] - KEEPOUT_X[0], rel=1e-9)
    assert out[:, 0].min() >= KEEPOUT_X[0] and out[:, 0].max() <= KEEPOUT_X[1]
    assert np.all(out[:, 1] == 0.0)
    assert out.shape == (int(round(length / 2.5e-5)) + 1, 2)


def test_prepare_path_clamps_into_keepout_box() -> None:
    out, _length, clamped = prepare_path([[-9.9e-3, 9.9e-3], [9.9e-3, -9.9e-3]])
    assert clamped is True
    assert out[:, 0].min() >= KEEPOUT_X[0] and out[:, 0].max() <= KEEPOUT_X[1]
    assert out[:, 1].min() >= KEEPOUT_Y[0] and out[:, 1].max() <= KEEPOUT_Y[1]


def test_prepare_path_inside_short_line_not_clamped() -> None:
    out, length, clamped = prepare_path([[0.0, 0.0], [5.0e-4, 0.0]])
    assert clamped is False
    assert length == pytest.approx(5.0e-4, rel=1e-12)
    assert out.shape == (21, 2)  # 0.5 mm / 25 um + 1


def test_prepare_path_degenerate_single_location() -> None:
    out, length, clamped = prepare_path([[1e-4, 1e-4], [1e-4, 1e-4]])
    assert length == 0.0
    assert clamped is False
    assert out.shape == (2, 2)


# --------------------------------------------------------------------------- #
# Schema validation (422)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "path",
    [
        {"points_xy_m": [[0.0, 0.0]]},  # < 2 points
        {"points_xy_m": [[0.0, 0.0]] * 4097},  # > 4096 points
        {"points_xy_m": [[0.0, 0.0], [0.0]]},  # not an [x, y] pair
        {"preset": ""},  # empty preset
        {"preset": PRESET, "points_xy_m": [[0.0, 0.0], [1e-4, 0.0]]},  # both
        {},  # neither
    ],
)
def test_infer_path_invalid_path_422(client: TestClient, path: dict) -> None:
    resp = client.post("/infer_path", json=_drawn_payload([[0.0, 0.0], [1e-4, 0.0]], path=path))
    assert resp.status_code == 422
    assert resp.json()["status"] == "error"
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_infer_path_non_finite_point_422(client: TestClient, literal: str) -> None:
    # Python's json parser accepts these literals; the boundary must reject them.
    body = (
        '{"model":"fno_newphys","path":{"points_xy_m":[[0.0,0.0],[%s,0.0]]},'
        '"render":{"width_px":480,"height_px":360}}' % literal
    )
    resp = client.post("/infer_path", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_infer_path_unknown_model_422(client: TestClient) -> None:
    resp = client.post("/infer_path", json=_drawn_payload([[0.0, 0.0], [1e-4, 0.0]], model="deeponet"))
    assert resp.status_code == 422


def test_infer_path_unknown_field_422(client: TestClient) -> None:
    resp = client.post(
        "/infer_path",
        json=_drawn_payload([[0.0, 0.0], [1e-4, 0.0]], view={"field": "vorticity", "z_index": -1}),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_infer_path_unknown_preset_422(client: TestClient) -> None:
    resp = client.post("/infer_path", json=_drawn_payload([[0.0, 0.0], [1e-4, 0.0]], path={"preset": "nope"}))
    assert resp.status_code == 422
    assert "Unknown preset" in resp.json()["error"]["message"]


def test_infer_path_body_too_large_413(client: TestClient) -> None:
    resp = client.post(
        "/infer_path",
        content=b"x" * (256 * 1024),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


def test_infer_path_chunked_body_requires_content_length_411(client: TestClient) -> None:
    # A chunked body has no Content-Length, which would bypass the size limit.
    resp = client.post(
        "/infer_path",
        content=iter([b"{}"]),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 411
    assert resp.json()["error"]["code"] == "length_required"


# --------------------------------------------------------------------------- #
# Stub end-to-end
# --------------------------------------------------------------------------- #

def _assert_png(b64: str) -> None:
    assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_infer_path_stub_drawn_zigzag(client: TestClient) -> None:
    points = [[-5e-4, -2e-4], [0.0, 2e-4], [5e-4, -2e-4]]
    resp = client.post("/infer_path", json=_drawn_payload(points))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "fno_newphys"
    # Lazy per-field rendering: one field per response (default "T").
    assert data["result_meta"]["field"] == "T"
    assert MIN_FRAMES <= len(data["frames"]) <= MAX_FRAMES
    assert len(data["frames"]) == len(data["times_s"])
    for png_b64 in data["frames"]:
        _assert_png(png_b64)
    meta = data["path_meta"]
    expected_len = 2.0 * math.hypot(5e-4, 4e-4)
    # Derived from the speed the MODEL reports, not a constant: the calibrated
    # models default to 0.6 m/s where the retired fixed-physics ones were 0.8,
    # and laser_off_time is length/speed.
    expected_off = expected_len / meta["v_scan_mps"]
    assert meta["path_length_m"] == pytest.approx(expected_len, rel=1e-6)
    assert meta["laser_off_time_s"] == pytest.approx(expected_off, rel=1e-6)

    # Drawn-path frame times. The calibrated models are duration-scaled, so the
    # SPAN is length/speed rather than a fixed 2 ms window and the count follows
    # from the frame step -- assert the properties that hold either way: starts
    # at 0, uniformly spaced, and spans the weld.
    times = np.asarray(data["times_s"])
    assert MIN_FRAMES <= len(times) <= MAX_FRAMES
    assert times[0] == 0.0
    steps = np.diff(times)
    # Not exactly uniform on purpose: frames are snapped onto the fine dose grid
    # (training dt 1e-5) rather than rounded onto it, so adjacent steps differ by
    # at most one fine step.
    assert steps.min() > 0
    assert float(steps.max() - steps.min()) <= 1.05e-5
    assert times[-1] == pytest.approx(expected_off, rel=0.05)
    assert meta["clamped"] is False
    assert meta["preset"] is None
    assert meta["gt_available"] is False
    assert meta["n_points_used"] == len(meta["points_xy_m"])
    assert meta["n_points_used"] == int(round(expected_len / 2.5e-5)) + 1

    rm = data["result_meta"]
    assert rm["grid_shape_xyz"] == [75, 50, 40]
    assert rm["panel"] == "single"
    assert 0 <= rm["z_index_used"] < 40
    assert set(data["timing_ms"]) == {"infer", "render", "total"}


def test_infer_path_stub_preset_triple_panel(client: TestClient) -> None:
    resp = client.post(
        "/infer_path",
        json={
            "model": "fno_newphys",
            "path": {"preset": PRESET},
            "view": {"field": "fl", "z_index": 5},
            "render": {"width_px": 480, "height_px": 720},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "fno_newphys"
    assert data["result_meta"]["field"] == "fl"
    assert MIN_FRAMES <= len(data["frames"]) <= MAX_FRAMES
    assert len(data["frames"]) == len(data["times_s"])
    _assert_png(data["frames"][len(data["frames"]) // 2])
    meta = data["path_meta"]
    assert meta["preset"] == PRESET
    assert meta["gt_available"] is True
    assert meta["clamped"] is False
    # A property of the preset's own polyline (this calibrated case resamples to
    # 25 points; the retired tial64 case was 61), so assert the invariant.
    assert meta["n_points_used"] >= 2
    # Also the preset's own geometry -- these are variable-LENGTH cases, which is
    # the point of the dataset, so assert it is a real weld inside the domain
    # rather than pinning one case's arc length.
    assert 1e-4 < meta["path_length_m"] < 6e-3
    # length / speed, both of which are the preset's own -- the relationship is
    # the invariant, the numbers are the case.
    assert meta["laser_off_time_s"] == pytest.approx(
        meta["path_length_m"] / meta["v_scan_mps"], rel=1e-3
    )
    rm = data["result_meta"]
    assert rm["panel"] == "triple"
    assert rm["z_index_used"] == 5
    # preset frame times come from the case NPZ, not the drawn-path linspace
    assert MIN_FRAMES <= len(data["times_s"]) <= MAX_FRAMES
    # First step is the frame cadence for THIS case (span/frames, snapped to the
    # fine dose grid), not a constant across cases.
    assert 0.0 < data["times_s"][1] <= meta["laser_off_time_s"]


def test_infer_path_presets_listing(client: TestClient) -> None:
    # No model -> the default model's own held-out set (the legacy shared root
    # is not part of the served product). PRESET is asked for by model below.
    resp = client.get("/infer_path/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    names = [p["name"] for p in data["presets"]]
    assert names and names == sorted(names)
    assert all(n.startswith("r150np_") for n in names)

    scoped = client.get("/infer_path/presets", params={"model": "fno_newphys"}).json()
    scoped_names = [p["name"] for p in scoped["presets"]]
    assert PRESET in scoped_names
    assert all(n.startswith("r150np_") for n in scoped_names)
    data = scoped
    for p in data["presets"]:
        assert len(p["points_xy_m"]) >= 2
        assert all(len(pt) == 2 for pt in p["points_xy_m"])


def test_old_infer_route_still_has_16k_limit(client: TestClient) -> None:
    resp = client.post("/infer", content=b"x" * (32 * 1024), headers={"Content-Type": "application/json"})
    assert resp.status_code == 413


# --------------------------------------------------------------------------- #
# /infer_path/render3d (3D block PNG frames)
# --------------------------------------------------------------------------- #

def _render3d_payload(**overrides) -> dict:
    payload = {
        "client_request_id": "r3d",
        "model": "fno_newphys",
        "path": {"preset": PRESET},
        "render": {"width_px": 320, "height_px": 240},
    }
    payload.update(overrides)
    return payload


def _assert_png_frames(frames: list[str]) -> None:
    assert MIN_FRAMES <= len(frames) <= MAX_FRAMES
    for b64 in frames:
        raw = base64.b64decode(b64)
        assert raw.startswith(b"\x89PNG\r\n\x1a\n"), f"not a PNG: {raw[:8]!r}"


def _png_size(b64: str) -> tuple[int, int]:
    raw = base64.b64decode(b64)
    w, h = struct.unpack(">II", raw[16:24])  # IHDR width/height
    return int(w), int(h)


def test_render3d_stub_preset_returns_png_frames(client: TestClient) -> None:
    resp = client.post("/infer_path/render3d", json=_render3d_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "fno_newphys"
    assert MIN_FRAMES <= len(data["times_s"]) <= MAX_FRAMES
    assert "video" not in data
    _assert_png_frames(data["frames"])
    assert data["path_meta"]["preset"] == PRESET
    assert data["path_meta"]["gt_available"] is True
    # Preset frames stack ground truth above the prediction: one layout for
    # the whole run, request width, roughly two screenshots + titles tall
    # (vs. the exact 320x240 single panel of a drawn path).
    sizes = {_png_size(b64) for b64 in data["frames"]}
    assert len(sizes) == 1
    w, h = sizes.pop()
    assert w == 320
    assert h > 1.8 * 240
    rm = data["result_meta"]
    assert rm["grid_shape_xyz"] == [75, 50, 40]
    assert rm["panel"] == "3d"
    assert set(data["timing_ms"]) == {"infer", "render", "total"}


def test_render3d_drawn_path_and_shared_cache(client: TestClient) -> None:
    points = [[-4e-4, -1e-4], [4e-4, 1e-4]]
    payload = _render3d_payload(model="fno_newphys", path={"points_xy_m": points})
    first = client.post("/infer_path/render3d", json=payload)
    assert first.status_code == 200
    _assert_png_frames(first.json()["frames"])
    assert first.json()["path_meta"]["gt_available"] is False
    # Drawn paths have no ground truth: single panel at the exact request size.
    assert _png_size(first.json()["frames"][0]) == (320, 240)
    # render3d populated the single-entry prediction cache; the same path via
    # /infer_path must reuse that exact entry (no re-inference).
    svc = client.app.state.inference_service
    entry = svc._path_prediction_cache
    assert entry is not None
    resp = client.post("/infer_path", json=_drawn_payload(points))
    assert resp.status_code == 200
    assert svc._path_prediction_cache is entry


def test_render3d_view_block_422(client: TestClient) -> None:
    resp = client.post("/infer_path/render3d", json=_render3d_payload(view={"z_index": 5}))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_render3d_unknown_preset_422(client: TestClient) -> None:
    resp = client.post("/infer_path/render3d", json=_render3d_payload(path={"preset": "nope"}))
    assert resp.status_code == 422
    assert "Unknown preset" in resp.json()["error"]["message"]


def test_render3d_unknown_model_422(client: TestClient) -> None:
    resp = client.post("/infer_path/render3d", json=_render3d_payload(model="deeponet"))
    assert resp.status_code == 422


def test_render3d_body_too_large_413(client: TestClient) -> None:
    resp = client.post(
        "/infer_path/render3d",
        content=b"x" * (256 * 1024),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


def test_render3d_chunked_body_requires_content_length_411(client: TestClient) -> None:
    resp = client.post(
        "/infer_path/render3d",
        content=iter([b"{}"]),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 411
    assert resp.json()["error"]["code"] == "length_required"


def test_path_infer_batch_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # Frames-per-forward-pass for /infer_path; default 1 (measured: RIGNO
    # batch 2 gains ~6% but doubles peak GPU memory), env-overridable.
    assert Settings(_env_file=None).path_infer_batch == 1
    monkeypatch.setenv("LPFNO_PATH_INFER_BATCH", "8")
    assert Settings(_env_file=None).path_infer_batch == 8
