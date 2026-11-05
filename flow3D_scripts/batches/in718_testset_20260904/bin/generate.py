#!/usr/bin/env python3
"""Frozen IN718 TEST set for the two-alloy self-improvement loop.

Purpose: the held-out set the loop is scored on. Solved and frozen BEFORE round 0,
never trained on, and excluded from the acquisition lattice by construction.

Design
------
* **Domain is copied from ds86529, not chosen.** 900 x 600 x 450 um + 100 um gas at
  10 um cells (90 x 60 x 55). The base model (86551, model 86651) trained on that grid,
  and a DOMAIN-EXTENT mismatch silently scores
  gas the model never saw. The IN718 validation campaign used 900 x 400 x 300 + 100
  (90 x 40 x 40), so its existing solves are NOT evaluable by this model -- which is why
  these conditions are re-solved here rather than reused.

* **The 7 AM-Bench mds2-2718 anchors are included verbatim** so the model can be scored
  directly against NIST measurements. Their key is (power_W, speed_mm_s, spot_um) and
  `spot_um` is a DIAMETER (as tabulated for AMB2022-03), so r0 = spot/2 = 24.5-41.0 um.

* **The box is EXPANDED past the Ti64 anchor box, because the anchors sit outside it.**
  Anchor box: P 90-235 W, r0 42-65 um. Experimental: P 245-325 W, r0 24.5-41.0 um --
  disjoint on both axes. Built on anchor ranges, the model could never be validated
  against the only experimental IN718 data that exists.

* **LHS over (H, V, r0), not (P, V, r0)**, then P = K1(r0) * H * sqrt(V). Sampling power
  directly would let the enthalpy range drift with radius, and H is the axis the surrogate
  is conditioned on. K1 ~ r0^1.5, so a fixed-r0 K1 would misplace the whole set.

* **Two regimes, scored separately** (see plan section 5b): the OVERLAP regime H<=21.7,
  where Ti64 training data exists and alloy is the only thing changing, and the
  EXTRAPOLATION regime above it, where the base model has no Ti64 neighbours and a failure
  cannot be attributed to the alloy. Every case carries its regime in the design CSV.

* Recipe follows the standing post-calibration one: per-material Fresnel, recoil_model 2,
  void_vapor_pressure_pa 101325, IN718's own vapor_pressure_b (13.958762, not Ti64's
  16.077291), NO plume (retired), clean timing (turn_off_at_end true, finish = track/V).
"""
import csv, json, math, os, shutil
from pathlib import Path

BATCH = Path(__file__).resolve().parent.parent
from _repo import REPO
MAT_SRC = REPO / "validation" / "materials" / "in718_full" / "in718_full.json"

# --- domain: copied from ds86529 (edge75/anchorbox). Do not "improve" these. ---
LENGTH, WIDTH, THICK, GAS, CELL = 9.0e-4, 6.0e-4, 4.5e-4, 1.0e-4, 1.0e-5
HALF_TRACK = 2.9862e-4          # edge75 track, 597.24 um
VAPOR_B_IN718 = 13.958762       # per-material tvexp; Ti64's is 16.077291

# --- expanded design box ---
R_LO, R_HI = 22.0e-6, 65.0e-6   # covers experimental 24.5-41.0 and anchor 42-65
P_LO, P_HI = 90.0, 350.0        # covers experimental 245-325
V_LO, V_HI = 0.15, 1.25
H_LO, H_HI = 4.0, 48.0          # covers overlap (<=21.7) and experimental (21.6-46.8)
H_OVERLAP_MAX = 21.7            # ds86529's own H ceiling

# AM-Bench mds2-2718: (P_W, V_mm_s, spot_um_DIAMETER)
AMBENCH = [(245.0, 960.0, 67.0), (285.0, 800.0, 67.0), (285.0, 960.0, 49.0),
           (285.0, 960.0, 67.0), (285.0, 960.0, 82.0), (285.0, 1200.0, 67.0),
           (325.0, 960.0, 67.0)]
N_LHS = 18
SEED = 4904


def k1(mat, r0):
    """King constant; P = K1 * H * sqrt(V). K1 ~ r0^1.5, so it is per-case."""
    p, o = mat["properties"], mat["optical"]
    rho, cp = p["density_kg_m3"], p["specific_heat_j_kg_k"]
    D = p["thermal_conductivity_w_m_k"] / (rho * cp)
    return (rho * (cp * (p["liquidus_temperature_k"] - 293.15) + p["latent_heat_j_kg"])
            * math.sqrt(math.pi * D * r0 ** 3) / o["normal_absorptivity"])


def lhs(n, d, seed):
    import random
    rng = random.Random(seed)
    cols = []
    for _ in range(d):
        c = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(c)
        cols.append(c)
    return [[cols[j][i] for j in range(d)] for i in range(n)]


def maximin_order(pts, lo, hi):
    """Greedy farthest-point ordering so every prefix is space-filling -- if the
    sweep is interrupted, what solved is still a usable design rather than a
    contiguous corner of the box."""
    import statistics
    a = [[(v - l) / (h - l) for v, l, h in zip(p, lo, hi)] for p in pts]
    n = len(a)
    d2 = lambda u, v: sum((x - y) ** 2 for x, y in zip(u, v))
    cen = [0.5] * len(lo)
    order = [min(range(n), key=lambda i: d2(a[i], cen))]
    dmin = [d2(a[i], a[order[0]]) for i in range(n)]
    while len(order) < n:
        nxt = max(range(n), key=lambda i: dmin[i] if i not in order else -1)
        order.append(nxt)
        dmin = [min(dmin[i], d2(a[i], a[nxt])) for i in range(n)]
    return order


def main():
    mat = json.loads(MAT_SRC.read_text())
    cases = []

    for P, Vmm, spot in AMBENCH:                      # experimental anchors, verbatim
        V, r0 = Vmm / 1000.0, spot / 2 * 1e-6
        cases.append({"P": P, "V": V, "r0": r0, "H": P / (k1(mat, r0) * math.sqrt(V)),
                      "kind": "ambench"})

    for u in lhs(N_LHS, 3, SEED):                     # LHS fill over (H, V, r0)
        H = H_LO + u[0] * (H_HI - H_LO)
        V = V_LO + u[1] * (V_HI - V_LO)
        r0 = R_LO + u[2] * (R_HI - R_LO)
        P = k1(mat, r0) * H * math.sqrt(V)
        if not (P_LO <= P <= P_HI):                   # respect the power box
            P = min(max(P, P_LO), P_HI)
            H = P / (k1(mat, r0) * math.sqrt(V))
        cases.append({"P": P, "V": V, "r0": r0, "H": H, "kind": "lhs"})

    order = maximin_order([(c["H"], c["V"], c["r0"]) for c in cases],
                          (H_LO, V_LO, R_LO), (H_HI, V_HI, R_HI))
    cases = [cases[i] for i in order]

    cfg_dir = BATCH / "configs"; cfg_dir.mkdir(parents=True, exist_ok=True)
    mat_dir = BATCH / "materials" / "in718_full"; mat_dir.mkdir(parents=True, exist_ok=True)
    # Copy the WHOLE material dir: the renderer reads temperature-table CSVs
    # (density, cp, k, viscosity, surface tension, resistivity) beside the JSON,
    # and copying only the JSON fails at render with "material table CSV not found".
    for f in MAT_SRC.parent.iterdir():
        if f.is_file():
            shutil.copy2(f, mat_dir / f.name)

    rows = []
    for i, c in enumerate(cases):
        name = f"in718t_{i:03d}"
        # Clean timing: finish = track/V, rounded UP onto a 1 ns grid. The raw
        # quotient lands ~1e-12 s SHORT of "travel + stop event" and the renderer
        # refuses it; 1 ns is ~400-2000x that margin, and the +2 guarantees a
        # strictly positive excess even when the quotient sits just under a
        # grid line (ceil alone can leave an arbitrarily small excess).
        T = (math.floor(2 * HALF_TRACK / c["V"] / 1e-9) + 2) * 1e-9
        cfg = {
            "schema_version": 1,
            "case_name": name,
            "geometry": {"length_m": LENGTH, "width_m": WIDTH, "thickness_m": THICK,
                         "gas_clearance_m": GAS, "placement": "centered"},
            "mesh": {"cell_size_m": CELL},
            "material": {"file": "../materials/in718_full/in718_full.json"},
            "boundary_conditions": {"x_min": "wall", "x_max": "wall", "y_min": "wall",
                                    "y_max": "wall", "z_min": "wall", "z_max": "pressure"},
            "laser": {
                "power_w": c["P"], "scan_speed_m_per_s": c["V"],
                "beam_radius_m": c["r0"], "lens_radius_m": c["r0"],
                "start_point_m": [-HALF_TRACK, 0.0, 0.0],
                "end_point_m": [HALF_TRACK, 0.0, 0.0],
                "turn_off_at_end": True,
                "heat_distribution": "gaussian", "focal_distance_m": 0.001,
                "absorptivity": 1.0, "absorptivity_flag": 0,
                "fresnel_from_material": True,
                "radiation_distribution_flag": 2, "reflection_flag": 3,
                "reflection_limit": 0.01, "reflectivity": 0.35, "max_reflections": 20,
                "vapor_pressure_a": 101300.0, "vapor_pressure_b": VAPOR_B_IN718,
            },
            "parallel": {"cores": 10, "threads_per_rank": 10},
            "simulation": {
                "finish_time_s": T, "initial_temperature_k": 273.15,
                "boundary_temperature_k": 300.0, "max_iterations": 2000000000,
                "recoil_model": 2, "solver_profile": "legacy_case_fine_1",
                "subprocess_id": 1, "void_vapor_pressure_pa": 101325.0,
            },
        }
        (cfg_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        rows.append({"order": i, "case_name": name, "kind": c["kind"],
                     "H": round(c["H"], 4), "V": round(c["V"], 6), "P": round(c["P"], 3),
                     "r0_um": round(c["r0"] * 1e6, 2), "finish_s": round(T, 9),
                     "regime": "overlap" if c["H"] <= H_OVERLAP_MAX else "extrapolation",
                     "cells_per_radius": round(c["r0"] / CELL, 2)})

    with open(BATCH / "design.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (BATCH / "CASE_ORDER.txt").write_text("\n".join(r["case_name"] for r in rows) + "\n")

    ov = sum(1 for r in rows if r["regime"] == "overlap")
    print(f"  {len(rows)} cases -> {cfg_dir}")
    print(f"    ambench {sum(1 for r in rows if r['kind']=='ambench')} | lhs {N_LHS}")
    print(f"    regime: overlap {ov} | extrapolation {len(rows)-ov}")
    print(f"    H {min(r['H'] for r in rows):.1f}-{max(r['H'] for r in rows):.1f} | "
          f"P {min(r['P'] for r in rows):.0f}-{max(r['P'] for r in rows):.0f} W | "
          f"V {min(r['V'] for r in rows):.2f}-{max(r['V'] for r in rows):.2f} | "
          f"r0 {min(r['r0_um'] for r in rows):.1f}-{max(r['r0_um'] for r in rows):.1f} um")
    print(f"    cells/radius {min(r['cells_per_radius'] for r in rows):.2f}-"
          f"{max(r['cells_per_radius'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
