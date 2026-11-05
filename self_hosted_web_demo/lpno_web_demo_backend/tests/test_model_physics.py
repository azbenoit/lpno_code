"""Model-driven laser physics (meta.json "physics"): capability discovery,
per-request P/V on /infer_path, duration-scaled time grids, and the
model-driven input-channel set (incl. path_dose_recent). Stub-mode, but the
path channels come from the real vendored builder (pure numpy)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.adapter_randompath import (
    MAX_FRAMES,
    ModelPhysics,
    RandomPathAdapter,
    PathInputError,
    _physics_from_meta,
)
from app.config import Settings
from app.main import create_app
from tests.conftest import needs_served_gt, skip_without_served_gt

needs_path_gt = needs_served_gt(Settings().path_data_dir)

# The repo's own served GT dir (data/random_path/ground_truth): the legacy cases at
# the root, each variable-physics model's split in its own subdir. Settings default.

VARPATH_PHYSICS = ModelPhysics(
    p_default=80.0,
    v_default=0.6,
    p_range=(31.0, 130.0),
    v_range=(0.21, 0.99),
    h_norm_denom=132.6826965854477,
    power_weighted_dose=True,
    duration_scaled_times=True,
    dt_s=1.0e-5,
    max_path_length_m=2.9e-3,
    start_anywhere=True,
)

# Mirrors the served rigno_1601 / gaot_1601 meta.json (the 1601
# keyhole+conduction split). VARPATH_PHYSICS above is no longer a served model;
# it stays as a standalone variable-physics fixture for the drawn-path maths.
KH1601_PHYSICS = ModelPhysics(
    p_default=150.0,
    v_default=0.6,
    p_range=(68.0, 237.0),
    v_range=(0.21, 0.99),
    h_norm_denom=250.3785065465,
    power_weighted_dose=True,
    duration_scaled_times=True,
    dt_s=1.0e-5,
    max_path_length_m=None,
    start_anywhere=True,
    presets_dir="rand150_newphys_test",
)

# A ~1 mm drawn diagonal inside the keep-out box.
DRAWN = [[-4.0e-4, -1.0e-4], [4.0e-4, 1.0e-4]]


@pytest.fixture(scope="module")
def client() -> TestClient:
    skip_without_served_gt(Settings().path_data_dir)
    settings = Settings(startup_load_runtime=True, use_stub_runtime=True)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def adapter() -> RandomPathAdapter:
    return RandomPathAdapter(Settings(use_stub_runtime=True), stub=True)


def _payload(model: str, **overrides) -> dict:
    payload = {
        "client_request_id": "phys1",
        "model": model,
        "path": {"points_xy_m": DRAWN},
        "render": {"width_px": 480, "height_px": 360},
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# /models capability discovery
# --------------------------------------------------------------------------- #

def test_models_expose_physics_capabilities(client: TestClient) -> None:
    body = client.get("/models", params={"page": "path"}).json()
    by_id = {m["id"]: m for m in body["models"]}
    # Every calibrated model varies P/V, so this page no longer serves a
    # fixed-physics model at all (the legacy `fno` was the last one).
    assert all(m["physics"] is not None for m in body["models"])
    ph = by_id["rigno_newphys"]["physics"]
    assert ph is not None
    # the slider must expose the TRAIN domain of the 1601 keyhole split
    # (P up to 194 W, V 0.22-0.99), rounded inward — never val/test extremes
    assert 67.0 <= ph["p_watts"]["min"] and ph["p_watts"]["max"] <= 238.0
    assert 0.20 <= ph["v_scan_mps"]["min"] and ph["v_scan_mps"]["max"] <= 0.995
    assert ph["p_watts"]["min"] < ph["p_watts"]["default"] < ph["p_watts"]["max"]
    assert ph["duration_scaled_times"] is True


# --------------------------------------------------------------------------- #
# Request-level P/V behaviour
# --------------------------------------------------------------------------- #

def test_pv_rejected_for_fixed_physics_model(client: TestClient) -> None:
    # The fixed-physics REFUSAL is still live code, but this page has no model to
    # trigger it any more: every calibrated model varies P/V. The fixed-window
    # branch stays covered by test_path_frame_dt.py.
    body = client.get("/models", params={"page": "path"}).json()
    assert body["models"]
    assert all(m["physics"] is not None for m in body["models"])


def test_pv_rejected_for_preset(client: TestClient) -> None:
    resp = client.post(
        "/infer_path",
        json=_payload("rigno_newphys", path={"preset": "r150np_010"}, p_watts=100.0),
    )
    assert resp.status_code == 422


def test_pv_applied_and_reported_for_variable_model(client: TestClient) -> None:
    resp = client.post("/infer_path", json=_payload("rigno_newphys", p_watts=100.0, v_scan_mps=0.5))
    assert resp.status_code == 200
    meta = resp.json()["path_meta"]
    assert meta["p_watts"] == 100.0 and meta["v_scan_mps"] == 0.5
    assert meta["physics_clamped"] is False


def test_pv_out_of_range_is_clamped_and_flagged(client: TestClient) -> None:
    models = client.get("/models", params={"page": "path"}).json()["models"]
    p_max = next(m["physics"]["p_watts"]["max"] for m in models if m["id"] == "rigno_newphys")
    resp = client.post("/infer_path", json=_payload("rigno_newphys", p_watts=999.0))
    assert resp.status_code == 200
    meta = resp.json()["path_meta"]
    assert meta["p_watts"] == p_max  # clamped to the model's trained max
    assert meta["physics_clamped"] is True


def test_fixed_model_meta_has_no_pv(client: TestClient) -> None:
    # Mirror of the retired fixed case: a variable model must REPORT the P/V it
    # ran at, so the page can say what produced the picture.
    resp = client.post("/infer_path", json=_payload("rigno_newphys"))
    assert resp.status_code == 200
    meta = resp.json()["path_meta"]
    assert meta["p_watts"] is not None and meta["v_scan_mps"] is not None


# --------------------------------------------------------------------------- #
# Duration-scaled time grid (unit level)
# --------------------------------------------------------------------------- #

def test_duration_scaled_times_span_the_real_weld(adapter: RandomPathAdapter) -> None:
    case = adapter.resolve(
        points_xy_m=DRAWN, preset=None, physics=VARPATH_PHYSICS, v_scan_mps=0.5
    )
    duration = case.path_length_m / 0.5
    assert case.times.size <= 20
    assert case.times[-1] == pytest.approx(duration, rel=0.02)
    # dose integrals run on the fine (training-dt) grid, render frames index it
    assert case.times_fine is not None
    assert np.allclose(np.diff(case.times_fine), VARPATH_PHYSICS.dt_s)
    assert np.array_equal(case.times, case.times_fine[case.frame_idx])


def test_short_fast_weld_gets_few_frames(adapter: RandomPathAdapter) -> None:
    short = [[-1.0e-4, 0.0], [1.0e-4, 0.0]]  # 0.2 mm
    case = adapter.resolve(points_xy_m=short, preset=None, physics=VARPATH_PHYSICS, v_scan_mps=0.99)
    # ~0.202 ms weld at dt 1e-5 -> ~21 fine steps -> <= 20 unique render frames
    assert case.times[-1] == pytest.approx(case.path_length_m / 0.99, rel=0.05)
    assert case.times.size <= 20


def test_fixed_physics_keeps_legacy_grid(adapter: RandomPathAdapter) -> None:
    case = adapter.resolve(points_xy_m=DRAWN, preset=None, physics=ModelPhysics())
    assert case.times.size == 20
    assert case.times[-1] == pytest.approx(2.0e-3)


# --------------------------------------------------------------------------- #
# Model-driven invar (channel set, H_norm, power-weighted dose)
# --------------------------------------------------------------------------- #

@needs_path_gt
def test_build_invar_recency_channel_and_h_norm(adapter: RandomPathAdapter) -> None:
    case = adapter.resolve(
        points_xy_m=DRAWN, preset=None, physics=VARPATH_PHYSICS, p_watts=100.0, v_scan_mps=0.5
    )
    keys = ["V_scan", "P", "H_norm", "x", "y", "z", "beam_x", "beam_y",
            "beam_dist", "path_dose", "path_dose_recent", "t_hat"]
    invar = adapter._build_invar(case, input_keys=keys)
    assert set(invar) == set(keys)
    nt = case.times.size
    assert invar["path_dose_recent"].shape[0] == nt
    # H_norm follows training: raw P/sqrt(V) over the train-split max
    expected_h = (100.0 / np.sqrt(0.5)) / VARPATH_PHYSICS.h_norm_denom
    assert float(invar["H_norm"].ravel()[0]) == pytest.approx(expected_h, rel=1e-5)
    assert float(invar["P"].ravel()[0]) == pytest.approx(100.0 / 10.0)
    assert float(invar["V_scan"].ravel()[0]) == pytest.approx(0.5 / 0.1)
    # recency dose decays after the beam leaves the start; cumulative never does
    rec = invar["path_dose_recent"][:, 0].max(axis=(1, 2, 3))
    cum = invar["path_dose"][:, 0].max(axis=(1, 2, 3))
    assert cum[-1] >= cum[0] - 1e-6


@needs_path_gt
def test_power_weighted_dose_scales_with_power(adapter: RandomPathAdapter) -> None:
    lo = adapter.resolve(points_xy_m=DRAWN, preset=None, physics=VARPATH_PHYSICS, p_watts=50.0, v_scan_mps=0.5)
    hi = adapter.resolve(points_xy_m=DRAWN, preset=None, physics=VARPATH_PHYSICS, p_watts=100.0, v_scan_mps=0.5)
    d_lo = adapter._path_channels(lo, keys=["path_dose"])["path_dose"]
    d_hi = adapter._path_channels(hi, keys=["path_dose"])["path_dose"]
    m = d_lo > 1e-9
    assert np.allclose(d_hi[m] / d_lo[m], 2.0, atol=1e-4)  # dose scales as P/P_ref


def test_effective_pv_fixed_model_rejects_and_defaults() -> None:
    fixed = ModelPhysics()
    p, v, clamped = RandomPathAdapter._effective_pv(fixed, None, None)
    assert (p, v, clamped) == (150.0, 0.8, False)
    with pytest.raises(PathInputError):
        RandomPathAdapter._effective_pv(fixed, 100.0, None)


def test_physics_from_meta_defaults_when_absent() -> None:
    assert _physics_from_meta(None) == ModelPhysics()
    assert _physics_from_meta({"label": "x"}) == ModelPhysics()


# --------------------------------------------------------------------------- #
# Per-model preset (ground-truth) sets
# --------------------------------------------------------------------------- #

def test_presets_are_per_model(client: TestClient) -> None:
    """A model is only offered the GT cases it was trained for: every served
    free-form model declares its own presets_dir, and the three share one
    training set, so they share one preset list (per training set, not per id)."""
    sets = {}
    for mid in ("fno_newphys", "gaot_newphys", "rigno_newphys"):
        got = client.get("/infer_path/presets", params={"model": mid}).json()["presets"]
        names = {p["name"] for p in got}
        assert names and all(n.startswith("r150np_") for n in names)
        sets[mid] = names
    assert sets["fno_newphys"] == sets["gaot_newphys"] == sets["rigno_newphys"]
    assert client.get("/infer_path/presets", params={"model": "no_such_model"}).status_code in (404, 422)


def test_kh1601_presets_carry_their_own_physics(client: TestClient) -> None:
    varp = client.get("/infer_path/presets", params={"model": "rigno_newphys"}).json()["presets"]
    # the variable test split varies P/V per case (the legacy set is all 150/0.8)
    assert len({p["p_watts"] for p in varp}) > 1
    assert len({p["v_scan_mps"] for p in varp}) > 1
    # These are the model's TEST split, whose physics can fall slightly outside the
    # train-derived slider range (e.g. V 0.206 < train min 0.235) — that is the
    # dataset, not a bug. Assert the dataset's own bounds instead.
    for p in varp:
        assert 60.0 <= p["p_watts"] <= 250.0
        assert 0.20 <= p["v_scan_mps"] <= 1.0


def test_foreign_preset_is_rejected(client: TestClient) -> None:
    """The legacy fno must not be able to run the 1601 model's GT case."""
    resp = client.post("/infer_path", json=_payload("rigno_newphys", path={"preset": "tial64_rand_batch_06"}))
    assert resp.status_code == 422
    assert "Unknown preset" in resp.json()["error"]["message"]


def test_presets_without_model_are_the_default_models(client: TestClient) -> None:
    """No model param -> the page's default model's set. The legacy shared root
    (the fixed-physics tial64_* cases) is not part of the served product: a fresh
    install ships only the tabled models' held-out cases, so this route must not
    depend on files the asset bundle does not carry."""
    names = {p["name"] for p in client.get("/infer_path/presets").json()["presets"]}
    default = {p["name"] for p in client.get("/infer_path/presets", params={"model": "fno_newphys"}).json()["presets"]}
    assert names and names == default


@needs_path_gt
def test_variable_length_preset_subsamples_frames(adapter: RandomPathAdapter) -> None:
    """The variable test cases store 22..396 frames; the old adapter hard-required
    exactly 20. They must be subsampled while keeping the stored grid as the fine
    (dose-integral) grid.

    The rendered count is now set by the frame time step rather than pinned at
    20 (a long preset earns more frames at the same Δt), so this asserts the
    invariants that still hold: strictly fewer frames than stored, within the
    cap, and every frame on the stored grid."""
    presets = adapter.presets("rigno_newphys")
    long_case = max(presets, key=lambda p: np.load(p["npz_path"])["time_s"].size)
    stored_nt = int(np.load(long_case["npz_path"])["time_s"].size)
    assert stored_nt > 20  # otherwise this test proves nothing
    case = adapter.resolve(
        points_xy_m=None, preset=long_case["name"], physics=KH1601_PHYSICS, model="rigno_newphys"
    )
    assert case.times.size < stored_nt
    assert case.times.size <= MAX_FRAMES
    assert case.times_fine is not None and case.times_fine.size == stored_nt
    assert np.array_equal(case.times, case.times_fine[case.frame_idx])
    # the preset's own stored physics is used, not the model defaults
    assert case.p_watts != KH1601_PHYSICS.p_default or case.v_scan_mps != KH1601_PHYSICS.v_default


# --------------------------------------------------------------------------- #
# No-trim for variable-length models + default model
# --------------------------------------------------------------------------- #

def test_variable_length_model_does_not_trim(adapter: RandomPathAdapter) -> None:
    """Path length is a free variable this model trained over, so it must not be
    trimmed — unlike the fixed-physics models' 1.5 mm cap."""
    no_trim = ModelPhysics(p_range=(31.0, 130.0), v_range=(0.21, 0.99), max_path_length_m=None)
    xs = np.linspace(-5.0e-4, 5.0e-4, 9)
    zig = [[float(x), float(-2.8e-4 if i % 2 == 0 else 2.8e-4)] for i, x in enumerate(xs)]
    raw = float(np.linalg.norm(np.diff(np.asarray(zig), axis=0), axis=1).sum())
    assert raw > 4.0e-3  # well past the legacy cap
    case = adapter.resolve(points_xy_m=zig, preset=None, physics=no_trim)
    assert case.path_length_m == pytest.approx(raw, rel=1e-3)
    assert case.clamped is False
    # and the legacy cap still applies to fixed-physics models
    legacy = adapter.resolve(points_xy_m=zig, preset=None, physics=ModelPhysics())
    assert legacy.path_length_m == pytest.approx(1.5e-3)
    assert legacy.clamped is True


def test_prepare_path_none_max_length_disables_trim() -> None:
    from app.adapter_randompath import prepare_path

    xs = np.linspace(-5.0e-4, 5.0e-4, 9)
    zig = [[float(x), float(-2.8e-4 if i % 2 == 0 else 2.8e-4)] for i, x in enumerate(xs)]
    _, length_none, trimmed_none = prepare_path(zig, max_length_m=None)
    _, length_cap, trimmed_cap = prepare_path(zig, max_length_m=1.5e-3)
    assert trimmed_none is False and length_none > 4.0e-3
    assert trimmed_cap is True and length_cap == pytest.approx(1.5e-3)


def test_keyhole_model_is_the_page_default(client: TestClient) -> None:
    # rigno_1601 (pk 23710, rigno_rp_film_35k_v6) is the best model on the 1601
    # keyhole+conduction split by beam-box MAE: 70.5 K on ds12968 (2026-08-31).
    body = client.get("/models", params={"page": "path"}).json()
    defaults = [m["id"] for m in body["models"] if m["default"]]
    assert defaults == ["fno_newphys"]  # exactly one default; best BB MAE (114.8 K)


def test_1601_models_carry_their_own_dataset_envelope(client: TestClient) -> None:
    """The envelope describes the DATASET, so both 1601 models carry the same one
    and it must not be inherited from the models they replaced.

    (This previously compared rigno_kh2579 against rigno_varpath; both were
    retired on 2026-08-31 when the page was cut to one model per architecture.)
    """
    body = client.get("/models", params={"page": "path"}).json()
    kh = next(m["physics"] for m in body["models"] if m["id"] == "rigno_newphys")
    gaot = next(m["physics"] for m in body["models"] if m["id"] == "gaot_newphys")
    # Same dataset -> same envelope. How the conduction input was built is the model's own
    # training choice: the RIGNO predates the re-run settings that FNO and GAOT use.
    per_model = {"conduction_source_sigma_factor", "power_weighted_conduction", "h_norm_denom"}
    assert {k: v for k, v in kh.items() if k not in per_model} == {k: v for k, v in gaot.items() if k not in per_model}
    assert kh["h_norm_denom"] == pytest.approx(gaot["h_norm_denom"])
    # Keyhole powers the retired varpath model never saw (its max was 116.97 W),
    # and h_norm_denom is max(P/sqrt(V)) over this split, not varpath's 132.68.
    assert kh["p_watts"]["max"] > 117.0  # keyhole powers the retired varpath model never saw
    assert kh["h_norm_denom"] != pytest.approx(132.6826965854477)
    assert kh["presets_dir"] == "rand150_newphys_test"


def test_1601_meta_declares_no_trim(client: TestClient) -> None:
    body = client.get("/models", params={"page": "path"}).json()
    ph = next(m["physics"] for m in body["models"] if m["id"] == "rigno_newphys")
    assert ph["max_path_length_m"] is None  # null => the frontend shows no trim text
