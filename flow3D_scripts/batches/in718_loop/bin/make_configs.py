#!/usr/bin/env python3
"""Render acquisition picks into FLOW-3D configs for one round.

Every field except (P, V, r0) is copied from the frozen IN718 test-set generator,
because the round's cases must be solvable and evaluable in the SAME domain as
the test set and as ds86529 -- 900 x 600 x 450 um + 100 um gas at 10 um cells. A
domain-extent mismatch silently scores gas the model never saw
so this is not a place to improvise.
"""
import argparse, csv, json, math, os, shutil
from pathlib import Path

# Prefix of this run's case names (see round.py RUN_TAG): a new run's cases must not share
# names with an earlier run's simulations or run directories.
RUN_TAG = os.environ.get("IN718_LOOP_TAG", "").strip().strip("_")

LOOP = Path(__file__).resolve().parent.parent
TESTSET = LOOP.parent / "in718_testset_20260904"
MAT_SRC = TESTSET / "materials" / "in718_full"

LENGTH, WIDTH, THICK, GAS, CELL = 9.0e-4, 6.0e-4, 4.5e-4, 1.0e-4, 1.0e-5
HALF_TRACK = 2.9862e-4
VAPOR_B_IN718 = 13.958762


def render(pick, name, mat_json):
    # finish = track/V rounded UP onto a 1 ns grid; the raw quotient lands ~1e-12 s
    # short of "travel + stop event" and the renderer refuses it.
    T = (math.floor(2 * HALF_TRACK / pick["V"] / 1e-9) + 2) * 1e-9
    return {
        "schema_version": 1, "case_name": name,
        "geometry": {"length_m": LENGTH, "width_m": WIDTH, "thickness_m": THICK,
                     "gas_clearance_m": GAS, "placement": "centered"},
        "mesh": {"cell_size_m": CELL},
        "material": {"file": str(mat_json)},
        "boundary_conditions": {"x_min": "wall", "x_max": "wall", "y_min": "wall",
                                "y_max": "wall", "z_min": "wall", "z_max": "pressure"},
        "laser": {"power_w": pick["P"], "scan_speed_m_per_s": pick["V"],
                  "beam_radius_m": pick["r0"], "lens_radius_m": pick["r0"],
                  "start_point_m": [-HALF_TRACK, 0.0, 0.0],
                  "end_point_m": [HALF_TRACK, 0.0, 0.0],
                  "turn_off_at_end": True, "heat_distribution": "gaussian",
                  "focal_distance_m": 0.001, "absorptivity": 1.0, "absorptivity_flag": 0,
                  "fresnel_from_material": True, "radiation_distribution_flag": 2,
                  "reflection_flag": 3, "reflection_limit": 0.01, "reflectivity": 0.35,
                  "max_reflections": 20, "vapor_pressure_a": 101300.0,
                  "vapor_pressure_b": VAPOR_B_IN718},
        "parallel": {"cores": 10, "threads_per_rank": 10},
        "simulation": {"finish_time_s": T, "initial_temperature_k": 273.15,
                       "boundary_temperature_k": 300.0, "max_iterations": 2000000000,
                       "recoil_model": 2, "solver_profile": "legacy_case_fine_1",
                       "subprocess_id": 1, "void_vapor_pressure_pa": 101325.0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--round", type=int, required=True)
    # A replacement for a round that is already collected belongs to THAT round, in its
    # own family, not to the next one. Without this, asking for a replacement wrote a
    # round0(N+1) directory holding a single case and consumed the next round's slot.
    ap.add_argument("--suffix", default="", help="family suffix, e.g. 'repl' or 'top'")
    a = ap.parse_args()

    rd = LOOP / f"round{a.round:02d}_{a.arm}{a.suffix}"
    cfg = rd / "configs"; cfg.mkdir(parents=True, exist_ok=True)
    mat = rd / "materials" / "in718_full"; mat.mkdir(parents=True, exist_ok=True)
    for f in MAT_SRC.iterdir():            # whole dir: the renderer reads the CSV tables
        if f.is_file():
            shutil.copy2(f, mat / f.name)

    picks = json.loads(a.picks.read_text())
    rows = []
    for i, p in enumerate(picks):
        # Full arm tag, not arm[:3]. Truncating collapsed maximintop/maximinrepl -> "max"
        # and probe2 -> "pro", so every top-up, replacement and probe-v2 case was named
        # identically to an existing round-1/round-2/probe run. run_rectangular_case.py
        # correctly refuses to overwrite an existing case dir and exited 2 -- 17 solves
        # failed this way before the pattern was visible.
        tag = a.arm.replace("_", "")
        run = f"{RUN_TAG}_" if RUN_TAG else ""
        name = f"in718_{run}{tag}_r{a.round:02d}{a.suffix}_{i:03d}"
        (cfg / f"{name}.json").write_text(
            json.dumps(render(p, name, "../materials/in718_full/in718_full.json"),
                       indent=2, sort_keys=True) + "\n")
        rows.append({"case_name": name, "arm": a.arm, "round": a.round,
                     "H": round(p["H"], 4), "V": round(p["V"], 6), "P": round(p["P"], 3),
                     "r0_um": round(p["r0"] * 1e6, 2), "region": p.get("region", ""),
                     "regime": p.get("regime", "")})
    with open(rd / "design.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (rd / "CASE_ORDER.txt").write_text("\n".join(r["case_name"] for r in rows) + "\n")
    print(f"  {len(rows)} configs -> {cfg}")
    from collections import Counter
    print("  regions:", dict(Counter(r['region'] for r in rows)))


if __name__ == "__main__":
    main()
