"""app/gt.py: the shared GT masking / matching / axes helpers.

The oracle for ``mask_fields`` is the formula the three GT loaders carried
inline before 2026-09-14 (steady_gt, adapter_randompath.ground_truth,
adapter_straightpath.ground_truth), reproduced here verbatim so
the shared helper is pinned to what the loaders used to compute.
"""
from __future__ import annotations

import numpy as np

from app.gt import (
    MATCH_DP_W,
    MATCH_DV_MPS,
    T_FILL_K,
    T_LIQUIDUS_K,
    T_SOLIDUS_K,
    load_axes,
    mask_fields,
    nearest_pv,
)


def _legacy_mask(t_raw, a_raw):
    def fill(arr, mask, d):
        gate = 0.5 * (np.tanh(20.0 * (mask - 0.5)) + 1.0)
        return d + gate * (arr - d)

    phi = (t_raw - T_SOLIDUS_K) / (T_LIQUIDUS_K - T_SOLIDUS_K)
    fl_raw = 0.5 * (np.tanh(2.5 * (phi - 0.5)) + 1.0)
    t_gt = fill(t_raw, a_raw, T_FILL_K)
    a_gt = fill(a_raw, a_raw, 0.0)
    fl_gt = fill(fl_raw, a_gt, 0.0)
    return {"T": t_gt, "alpha": a_gt, "fl": fl_gt}


def test_mask_fields_matches_previous_inline_formula():
    rng = np.random.default_rng(0)
    t = rng.uniform(300.0, 3500.0, size=(3, 1, 6, 5, 4)).astype(np.float32)
    a = rng.uniform(0.0, 1.0, size=t.shape).astype(np.float32)
    got = mask_fields(t, a)
    want = _legacy_mask(t, a)
    for k in ("T", "alpha", "fl"):
        assert got[k].shape == t.shape
        np.testing.assert_array_equal(got[k], want[k]), k


def test_mask_fields_semantics():
    t = np.array([1000.0, 2000.0, 2000.0], dtype=np.float32)
    a = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    out = mask_fields(t, a)
    assert out["T"][0] == 1000.0 and out["T"][1] == 2000.0
    assert out["T"][2] == T_FILL_K            # gas fills to the boiling constant
    assert out["alpha"][2] == 0.0 and out["alpha"][0] == 1.0
    assert out["fl"][0] < 0.05                # well below solidus
    assert out["fl"][1] > 0.95                # well above liquidus
    assert out["fl"][2] == 0.0                # gated off in the gas


def test_nearest_pv_tolerance_and_tiebreak():
    entries = [
        {"name": "a", "p_watts": 100.0, "v_scan_mps": 0.500},
        {"name": "b", "p_watts": 100.9, "v_scan_mps": 0.500},   # 0.9 W off, in box
        {"name": "c", "p_watts": 100.0, "v_scan_mps": 0.504},   # 4 mm/s off, in box
    ]
    assert nearest_pv(entries, 100.0, 0.5)["name"] == "a"
    # 0.9 W (d=0.81) beats 4 mm/s (d=(0.004*200)^2=0.64)? No: 0.64 < 0.81 -> c wins
    assert nearest_pv(entries[1:], 100.0, 0.5)["name"] == "c"
    # box edge is inclusive (exactly representable on the P axis) ...
    assert nearest_pv([entries[0]], 100.0 + MATCH_DP_W, 0.5)["name"] == "a"
    assert nearest_pv([entries[0]], 100.0, 0.5 + 0.9 * MATCH_DV_MPS)["name"] == "a"
    # ... and a miss just past it on EITHER axis alone is a miss
    assert nearest_pv([entries[0]], 100.0 + MATCH_DP_W + 1e-6, 0.5) is None
    assert nearest_pv([entries[0]], 100.0, 0.5 + 1.1 * MATCH_DV_MPS) is None
    assert nearest_pv([], 100.0, 0.5) is None


def test_load_axes_float32(tmp_path):
    p = tmp_path / "g.npz"
    np.savez(p, x=np.arange(3, dtype=np.float64), y=[0.0, 1.0], z=np.array([5.0]))
    with np.load(p) as d:
        x, y, z = load_axes(d)
    assert x.dtype == np.float32 and y.dtype == np.float32 and z.dtype == np.float32
    assert x.tolist() == [0.0, 1.0, 2.0] and y.tolist() == [0.0, 1.0] and z.tolist() == [5.0]


def _steady_split(root, split, cases):
    """Write a tiny var_1 steady cache: cases = [(P, V)], grid 3x2x2."""
    n = len(cases)
    d = root / split
    d.mkdir(parents=True)
    x = np.linspace(0, 2e-5, 3, dtype=np.float32)
    shape = (n, 1, 3, 2, 2)
    np.savez(
        d / "var_1_time_steps30_avg30.npz",
        P=np.array([[[[[p]]]] for p, _ in cases], dtype=np.float32),
        V_scan=np.array([[[[[v]]]] for _, v in cases], dtype=np.float32),
        Temperature=np.full(shape, 2000.0, dtype=np.float32),
        **{"Fraction Of Fluid": np.ones(shape, dtype=np.float32)},
        x=np.broadcast_to(x.reshape(1, 1, 3, 1, 1), shape).copy(),
        y=np.zeros(shape, dtype=np.float32),
        z=np.zeros(shape, dtype=np.float32),
    )


def test_steady_catalog_is_cached_and_load_prefers_the_test_split(tmp_path):
    import os

    from app.steady_gt import catalog, load_steady_gt

    _steady_split(tmp_path, "train", [(100.0, 0.5), (120.0, 0.6)])
    _steady_split(tmp_path, "test", [(100.0, 0.5)])   # same P/V as a train case
    cat = catalog(tmp_path)
    assert [(e["split"], e["name"]) for e in cat] == [  # listed train, val, test (pre-dedup order)
        ("train", "100W_500mps"), ("train", "120W_600mps"), ("test", "100W_500mps"),
    ]
    assert catalog(tmp_path) is cat                          # cached
    got = load_steady_gt(tmp_path, 100.0, 0.5)
    assert got is not None and got[3] == "test"              # test split wins the tie
    assert got[1][0].shape == (3,) and got[0]["T"].shape == (3, 2, 2)
    assert load_steady_gt(tmp_path, 120.0, 0.6)[3] == "train"
    assert load_steady_gt(tmp_path, 130.0, 0.6) is None       # 10 W off: no match

    # A rebuilt split file (mtime bumped) invalidates the catalog.
    f = tmp_path / "train" / "var_1_time_steps30_avg30.npz"
    os.utime(f, ns=(1, 10**18))
    assert catalog(tmp_path) is not cat
