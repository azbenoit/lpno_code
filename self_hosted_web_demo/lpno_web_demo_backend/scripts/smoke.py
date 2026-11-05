#!/usr/bin/env python
"""Exercise every route of a running backend, every served model, 2D and 3D.

    python scripts/smoke.py http://127.0.0.1:8001

Reports per request: HTTP status, panel type, ground-truth match, MAE and the
server's own timing. Exit status is non-zero if any request failed, so it
doubles as the "does this instance work" check after a fresh setup. Needs only
the standard library.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001").rstrip("/")
RENDER = {"width_px": 480, "height_px": 360}
failures: list[str] = []


def call(method: str, path: str, body: dict | None = None, timeout: float = 240.0) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.load(exc)
        except Exception:  # noqa: BLE001
            return exc.code, {"error": exc.reason}


def check(label: str, status: int, body: dict, expect: int = 200) -> bool:
    ok = status == expect
    if not ok:
        failures.append(f"{label}: HTTP {status} {json.dumps(body)[:160]}")
    return ok


def models(page: str) -> list[dict]:
    status, body = call("GET", f"/models?page={page}")
    check(f"/models?page={page}", status, body)
    return body.get("models", [])


def timing(body: dict) -> str:
    t = body.get("timing_ms") or {}
    return f"infer {t.get('infer', 0):6.0f} ms render {t.get('render', 0):6.0f} ms"


print(f"backend {BASE}")
status, body = call("GET", "/healthz")
check("/healthz", status, body)
print(f"  healthz {body.get('status')} runtime_loaded={body.get('runtime_loaded')} device={(body.get('runtime') or {}).get('device')}")

# ---- steady ----------------------------------------------------------------
status, cat = call("GET", "/gt_catalog?dataset=steady")
check("/gt_catalog?dataset=steady", status, cat)
entries = cat.get("entries", [])
test_case = next((e for e in entries if e["split"] == "test"), entries[0] if entries else None)
print(f"\nsteady: {len(entries)} catalog cases; GT case {test_case['name'] if test_case else '-'}")
for m in models("steady"):
    for mode in ("2d", "3d"):
        # free (P, V)
        st, b = call("POST", "/infer", {"client_request_id": "smoke", "client_seq": 1, "model": m["id"], "mode": "final",
                                        "inputs": {"p_watts": 150.0, "v_scan_mps": 0.6},
                                        "view": {"field": "T", "plot_mode": mode}, "render": RENDER})
        ok = check(f"/infer {m['id']} {mode}", st, b)
        print(f"  /infer            {m['id']:11s} {mode}  {'ok' if ok else 'FAIL'}  {timing(b)}")
        if test_case:
            st, b = call("POST", "/infer/gt_compare", {"client_request_id": "smoke", "model": m["id"],
                         "inputs": {"p_watts": test_case["p_watts"], "v_scan_mps": test_case["v_scan_mps"]},
                         "field": "T", "plot_mode": mode, "render": RENDER})
            ok = check(f"/infer/gt_compare {m['id']} {mode}", st, b)
            rm = b.get("result_meta") or {}
            print(f"  /infer/gt_compare {m['id']:11s} {mode}  {'ok' if ok else 'FAIL'}  gt={rm.get('gt_available')} "
                  f"MAE {rm.get('gt_mae', float('nan')):6.2f} K  {timing(b)}")

# ---- straight --------------------------------------------------------------
status, cat = call("GET", "/gt_catalog?dataset=straight")
check("/gt_catalog?dataset=straight", status, cat)
entries = cat.get("entries", [])
test_case = next((e for e in entries if e["split"] == "test"), entries[0] if entries else None)
print(f"\nstraight: {len(entries)} catalog cases; GT case {test_case['name'] if test_case else '-'}")
for m in models("straight"):
    for label, pv in (("free", (150.0, 0.6)), ("gt", (test_case["p_watts"], test_case["v_scan_mps"]) if test_case else None)):
        if pv is None:
            continue
        st, b = call("POST", "/infer_straight", {"client_request_id": "smoke", "model": m["id"],
                     "inputs": {"p_watts": pv[0], "v_scan_mps": pv[1]}, "render": RENDER})
        ok = check(f"/infer_straight {m['id']} {label}", st, b)
        sm = b.get("straight_meta") or {}
        print(f"  /infer_straight   {m['id']:11s} {label:4s} {'ok' if ok else 'FAIL'}  frames={len(b.get('frames', []))} "
              f"grid={(b.get('result_meta') or {}).get('grid_shape_xyz')} gt={sm.get('gt_available')} "
              f"MAE {sm.get('gt_mae') if sm.get('gt_mae') is not None else float('nan'):6.2f} K  {timing(b)}")
    p3d, v3d = (test_case["p_watts"], test_case["v_scan_mps"]) if test_case else (150.0, 0.6)  # no GT installed
    st, b = call("POST", "/infer_straight/render3d", {"client_request_id": "smoke", "model": m["id"],
                 "inputs": {"p_watts": p3d, "v_scan_mps": v3d}, "render": RENDER})
    ok = check(f"/infer_straight/render3d {m['id']}", st, b)
    print(f"  /infer_straight/render3d {m['id']:5s}   {'ok' if ok else 'FAIL'}  frames={len(b.get('frames', []))}  {timing(b)}")

# ---- free-form path ----------------------------------------------------------
print("\npath:")
for m in models("path"):
    st, pre = call("GET", f"/infer_path/presets?model={m['id']}")
    check(f"/infer_path/presets {m['id']}", st, pre)
    presets = pre.get("presets", [])
    preset = presets[0]["name"] if presets else None
    st, b = call("POST", "/infer_path", {"client_request_id": "smoke", "model": m["id"],
                 "path": {"points_xy_m": [[-4.0e-4, -1.0e-4], [0.0, 1.0e-4], [4.0e-4, -1.0e-4]]},
                 "p_watts": 150.0, "v_scan_mps": 0.6, "view": {"field": "T"}, "render": RENDER})
    ok = check(f"/infer_path drawn {m['id']}", st, b)
    pm = b.get("path_meta") or {}
    print(f"  /infer_path drawn  {m['id']:13s} {'ok' if ok else 'FAIL'}  frames={pm.get('n_frames')} "
          f"grid={(b.get('result_meta') or {}).get('grid_shape_xyz')}  {timing(b)}")
    if preset:
        st, b = call("POST", "/infer_path", {"client_request_id": "smoke", "model": m["id"],
                     "path": {"preset": preset}, "view": {"field": "T"}, "render": RENDER})
        ok = check(f"/infer_path preset {m['id']}", st, b)
        pm = b.get("path_meta") or {}
        print(f"  /infer_path preset {m['id']:13s} {'ok' if ok else 'FAIL'}  {preset} frames={pm.get('n_frames')} "
              f"gt={pm.get('gt_available')} MAE {pm.get('gt_mae') if pm.get('gt_mae') is not None else float('nan'):6.2f} K  {timing(b)}")
        st, b = call("POST", "/infer_path/render3d", {"client_request_id": "smoke", "model": m["id"],
                     "path": {"preset": preset}, "render": RENDER})
        ok = check(f"/infer_path/render3d {m['id']}", st, b)
        print(f"  /infer_path/render3d {m['id']:11s} {'ok' if ok else 'FAIL'}  frames={len(b.get('frames', []))}  {timing(b)}")

st, b = call("POST", "/cache/clear", {})
check("/cache/clear", st, b)
print(f"\n/cache/clear: {b}")

if failures:
    print(f"\n{len(failures)} FAILED:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nall routes ok")
