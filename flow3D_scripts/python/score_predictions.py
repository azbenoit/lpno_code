#!/usr/bin/env python3
"""Score a surrogate's predicted fields against the experimental anchors,
through exactly the same functions the FLOW-3D campaign is scored with.

One row per case, two measured columns on it: the solved NPZ (the FLOW-3D
column) and the model's prediction pulled out of an eval frame dump (the model
column). Both go through `melt_metrics` and `score_anchors.score_row`, so the
+-30 % band, the depth-kind guard and the anchor matching are the campaign's,
not a second implementation.

    python3 python/score_predictions.py \
        --runs runs/batches/ti64_anchorbox_lhs150_20260830/runs \
        --case abx_015_gan129 --case abx_013_gal150 \
        --dump runs/anchorbox_model_validation/eval_out_lab/eval_frames_dump.npz \
        --kelvin 3000 --out runs/anchorbox_model_validation/rows_lab.json

Without `--dump` it produces the FLOW-3D column alone, which is how the
like-for-like solver numbers are recomputed from the SAME solves the model was
evaluated on (never copied from earlier summary tables, which come from different runs).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import melt_metrics as mm                      # noqa: E402
import score_anchors as sa                     # noqa: E402
from predicted_fields import predicted_case    # noqa: E402

SOLIDUS_K = 1898.0     # T_s 1873 / T_l 1923 midpoint; the models' own metrics.json
                       # records melt_threshold_K = 1898.0. Never guessed per alloy.


def measure(data, *, laser_off_s, tail_fraction=mm.STEADY_TAIL_FRACTION):
    f = mm.fusion_geometry_auto(data, solidus_k=SOLIDUS_K)
    d = mm.depression(data, laser_off_s=laser_off_s, tail_fraction=tail_fraction)
    v = mm.domain_validity(data, solidus_k=SOLIDUS_K)
    cell = mm.cell_size_um(data)
    return {"fusion_depth_um": f.depth_um, "fusion_width_um": f.width_um,
            "depression_depth_um": d.depth_um, "depression_width_um": d.width_um,
            "depression_sd_um": d.depth_sd_um,
            "fusion_depth_cells": f.depth_um / cell, "cell_um": cell,
            "valid": v.ok, "floor_k": v.floor_k, "melt_span": v.melt_span_fraction}


def anchors_from_plan(plan_path: pathlib.Path):
    """[(sim_kind, points)] for every anchor the campaign plan scores."""
    plan = json.loads(plan_path.read_text())
    out = []
    for entry in plan["anchors"]:
        path = plan_path.parent / entry["anchor"]
        pts = sa.normalize_points(sa.load_anchor(path), series=entry.get("series"))
        out.append((entry["sim_kind"], entry.get("label"), pts))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path, required=True)
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--dump", type=pathlib.Path)
    ap.add_argument("--kelvin", type=float, default=None,
                    help="the model's RECORDED ref_scales.T "
                         "(recorded_kelvin_factor); required with --dump")
    ap.add_argument("--plan", type=pathlib.Path,
                    default=HERE.parent / "validation/scoring_plan.json")
    ap.add_argument("--tail-fraction", type=float, default=mm.STEADY_TAIL_FRACTION)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.dump and a.kelvin is None:
        ap.error("--kelvin is required with --dump; pass the number "
                 "recorded_kelvin_factor() returns, do not derive it")

    names = sorted(a.case)          # the dump is in ALPHABETICAL case order
    plan = anchors_from_plan(a.plan)

    npz = {n: a.runs / n / "npz" / f"{n}.npz" for n in names}
    missing = [str(p) for p in npz.values() if not p.exists()]
    if missing:
        raise SystemExit("missing exports:\n  " + "\n  ".join(missing))

    counts = {}
    for n in names:
        with np.load(npz[n]) as z:
            counts[n] = len(z["time_s"])

    segments = None
    if a.dump:
        with np.load(a.dump) as z:
            total = len(z["true_T"])
            if total == sum(counts.values()):
                # `render_case_video` is a script, not a package member, so it is
                # found by path: the sibling checkout via the same resolver
                # local_inference uses (env override, then a worktree-safe walk-up).
                from local_inference import lpno_root
                sys.path.insert(0, str(lpno_root() / "scripts"))
                from render_case_video import check_segments
                check_segments(z["true_T"], [counts[n] for n in names])
                per = [counts[n] for n in names]
            elif total == len(names):
                per = [1] * len(names)          # time-averaged (steady) eval
            else:
                raise SystemExit(
                    f"dump holds {total} frames; the {len(names)} cases hold "
                    f"{sum(counts.values())} and a per-case average would hold "
                    f"{len(names)}. A strided eval cannot be split by source frame.")
            start, segments = 0, {}
            for n, k in zip(names, per):
                segments[n] = (np.array(z["pred_T"][start:start + k]),
                               np.array(z["pred_alpha"][start:start + k]))
                start += k

    rows = []
    for n in names:
        case_dir = a.runs / n
        cond = sa.run_conditions(case_dir)
        if cond is None:
            raise SystemExit(f"{n}: no run conditions in its manifest")
        point = kind = label = None
        for sim_kind, lab, pts in plan:
            got = sa.match_point(cond, pts)     # raises AmbiguousAnchor, by design
            if got is not None:
                point, kind, label = got, sim_kind, lab
                break
        row = {"case": n, **{k: cond.get(k) for k in ("power_w", "speed_m_s", "r0_m")},
               "anchor": label, "depth_kind": kind,
               "measured_depth_um": point["depth_um"] if point else None,
               "measured_width_um": point.get("width_um") if point else None}

        with np.load(npz[n]) as z:
            data = dict(z)
            off = sa.laser_off_s(case_dir) or float(data["time_s"][-1])
            row["laser_off_s"] = off
            row["flow3d"] = measure(data, laser_off_s=off, tail_fraction=a.tail_fraction)
            src = {k: data[k] for k in ("x", "y", "z", "time_s")}

        if segments is not None:
            pT, pa = segments[n]
            pred = predicted_case(pT, pa, src, kelvin=a.kelvin)
            row["model"] = measure(pred, laser_off_s=float(pred["time_s"][-1]),
                                   tail_fraction=a.tail_fraction)
            row["model"]["grid"] = list(pT.shape)

        if point is not None:
            for col in ("flow3d", "model"):
                if col not in row:
                    continue
                m = row[col]
                depth, width = ((m["depression_depth_um"], m["depression_width_um"])
                                if kind == "depression"
                                else (m["fusion_depth_um"], m["fusion_width_um"]))
                m["scored"] = sa.score_row(sim_depth_um=depth, sim_width_um=width,
                                           point=point, sim_kind=kind)
        rows.append(row)

    for r in rows:
        head = (f"{r['case']:<22} {r['power_w']:6.1f} W {r['speed_m_s']:5.2f} m/s "
                f"r0 {r['r0_m'] * 1e6:4.1f} um  {r['depth_kind'] or '-'}")
        print(head)
        for col in ("flow3d", "model"):
            m = r.get(col)
            if not m:
                continue
            s = m.get("scored")
            verdict = ("no anchor" if s is None else
                       f"{s['depth_err_pct']:+.1f} %, "
                       f"{'inside' if s['depth_in_band'] else 'OUTSIDE'} "
                       f"+-{mm.DEFAULT_BAND * 100:.0f} %")
            print(f"    {col:<7} depth {m['fusion_depth_um']:6.1f} um fusion / "
                  f"{m['depression_depth_um']:6.1f} um depression "
                  f"[{m['fusion_depth_cells']:.1f} cells, floor "
                  f"{'ok' if m['valid'] else 'CLIPPED'}]  {verdict}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1, default=float) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
