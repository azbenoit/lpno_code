#!/usr/bin/env python3
"""Pre-average a per-case transient split into a steady laser-frame split.

Reads ``material_lhs_300_split/{train,val,val_quick,test}/<case>.npz`` (raw
FLOW-3D transient series, 100 frames on a 90x40x40 grid) and writes
``material_lhs_300_steady_split/{...}/<case>.npz`` + ``<case>.manifest.json``,
each holding ONE time-averaged temperature/flow field in the laser's co-moving
reference frame -- a single 61x40x40 frame with the laser pinned to a fixed
column, the same target representation the prior single-material P-V baselines
used.

Beam-centered co-moving average (why not the shared ``_crop_shift_moving_frame``):
that helper shifts frame ``t`` by an integer ``t`` cells, i.e. it assumes the
laser advances exactly one grid cell per frame. That holds for the original
straight-path data (resampled to even *distance* steps) but NOT for
``material_lhs_300``, whose frames are even in *time*: the laser advances only
~0.603 cells/frame, so the integer shift over-corrects 1.66x and smears the melt
pool over ~11 cells with a drifting hot spot. That smeared, position-variable
blob is what stalls RIGNO (its region graph is fixed on node positions) while
FNO's spectral conv tolerates it. Here each frame is instead re-centered on the
*actual* laser cell (from ``_beam_centres_over_time``), so the pool is sharp and
pinned at ``LASER_COL`` for every case regardless of scan speed -- and, being a
fixed width, material and original-straight-path cases share one grid and merge.

Why pre-average to disk instead of at load time: the material sweep needs lazy
per-case loading (the full split does not fit eagerly -- the eager
``load_consolidated_npz`` concatenates every case at full time resolution before
averaging), but the lazy store rejects ``moving_ref_frame=true``. Collapsing the
time axis once, offline, makes each case a single frame the existing per-frame
pipeline reads with ``moving_ref_frame: false`` -- no lazy store surgery, ~100x
smaller.

The steady configs (config_flow3D_{FNO,RIGNO}_material_steady) point at the
output split.

Usage:
  python scripts/build_steady_split.py                 # default src/dst, all splits
  python scripts/build_steady_split.py --limit 2       # smoke: 2 cases per split
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from lpno.custom_utils.npz_datasource import (
    _beam_centres_over_time, _build_path_channels, _load_manifest,
)

SRC_DEFAULT = Path("data/flow3D/material_lhs_300_split")
DST_DEFAULT = Path("data/flow3D/material_lhs_300_steady_split")
SUBDIRS = ["train", "val", "val_quick", "test"]
FRAME_STEPS = 30   # number of trailing frames averaged into the steady field.
OUT_WIDTH = 61     # x-width of the co-moving window (fixed, so all cases share a grid).
LASER_COL = 45     # column the laser is pinned to (LASER_COL behind it, OUT_WIDTH-LASER_COL ahead).


def collapse_case_to_steady(
    npz_path: Path,
    frame_steps: int = FRAME_STEPS,
    out_width: int = OUT_WIDTH,
    laser_col: int = LASER_COL,
) -> dict:
    """Collapse one transient case NPZ to a single steady beam-centered frame.

    Averages the last ``frame_steps`` frames after re-centering each on the laser
    cell (from the manifest's beam kinematics), so the melt pool is pinned at
    ``laser_col`` in an ``out_width``-wide window. Returns an on-disk dict (source
    schema, but ``time_s`` length 1 and x = ``out_width``), ready for
    ``np.savez_compressed``. x is stored laser-relative so the frame is identical
    across scan speeds.
    """
    manifest = _load_manifest(npz_path)
    if manifest is None:
        raise ValueError(f"[steady] no manifest beside {npz_path.name}; the co-moving frame needs laser kinematics.")
    with np.load(npz_path) as d:
        x = d["x"].astype(np.float64)
        y = d["y"].astype(np.float32)
        z = d["z"].astype(np.float32)
        nx = x.size
        t = np.asarray(d["time_s"])
        nt = t.size
        dx = float(np.mean(np.diff(x)))
        fields = [k for k in d.files if getattr(d[k], "ndim", 0) >= 4]
        raw = {k: d[k] for k in fields}
        out = {  # scalars copied through unchanged
            "time_s": np.asarray([t[-1]], dtype=t.dtype),
            "power_w": d["power_w"],
            "scan_speed_m_per_s": d["scan_speed_m_per_s"],
        }
        if "case_name" in d.files:
            out["case_name"] = d["case_name"]

    if not (1 <= frame_steps <= nt):
        raise ValueError(f"[steady] frame_steps must be in [1, {nt}]; got {frame_steps}.")
    if out_width > nx:
        raise ValueError(f"[steady] out_width {out_width} exceeds grid width {nx}.")

    # laser cell per frame from beam kinematics; window start pins it to laser_col.
    centres = _beam_centres_over_time(manifest, t)         # (nt, 2) metres
    cx = centres[:, 0]
    cell = np.round((cx - x[0]) / dx).astype(int)          # laser cell per frame
    starts = np.clip(cell[nt - frame_steps:] - laser_col, 0, nx - out_width)  # (frame_steps,)

    xr = ((np.arange(out_width) - laser_col) * dx)          # laser-relative x (beam at 0)
    out["x"] = xr.astype(np.float32)
    out["y"] = y
    out["z"] = z

    # Co-moving beam footprint: exp(-((x-0)^2+(y-cy)^2)/2sigma^2), broadcast over z.
    # This is the spatially-varying laser anchor RIGNO needs -- in the steady frame
    # every other input channel is constant, so without it a GNN collapses to a
    # constant output (FNO synthesizes structure from coords, RIGNO does not). Matches
    # _build_path_channels' beam_dist so the input key semantics are identical.
    sigma = float(manifest["input"]["laser"]["beam_radius_m"])
    cy = float(centres[-1, 1])                              # beam y (co-moving y unchanged)
    r2 = xr[:, None] ** 2 + (y.astype(np.float64) - cy)[None, :] ** 2  # (W, ny)
    footprint = np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)  # peak 1.0 at beam
    out["beam_dist"] = np.broadcast_to(footprint[:, :, None], (out_width, y.size, z.size)).astype(np.float32)[None]

    def wavg(a):  # (nt, X, Y, Z[, 3]) -> (1, W, Y, Z[, 3]): beam-centered window average
        acc = None
        for j, s in enumerate(starts):
            w = a[nt - frame_steps + j, s:s + out_width]
            acc = w.astype(np.float64) if acc is None else acc + w
        return (acc / frame_steps).astype(np.float32)[None]

    for k, a in raw.items():
        out[k] = wavg(a)

    # Co-moving history channels: cumulative path_dose (scan direction) and the
    # exp-decay path_dose_recent (the trailing thermal tail). Built lab-frame by the
    # tested _build_path_channels, then run through the same window average, so they
    # carry the ahead/behind asymmetry the symmetric beam_dist cannot.
    pc = _build_path_channels(["path_dose", "path_dose_recent"], manifest, x, y, z, t)
    for k in ("path_dose", "path_dose_recent"):
        out[k] = wavg(pc[k][0, :, 0])  # (1,nt,1,X,Y,Z) -> (nt,X,Y,Z) -> (1,W,Y,Z)
    return out


def _manifest_beside(npz_path: Path) -> Path | None:
    cand = npz_path.with_suffix(".manifest.json")
    if cand.exists():
        return cand
    cand = npz_path.parent / "case_manifest.json"
    return cand if cand.exists() else None


def build_split(src: Path, dst: Path, frame_steps: int, limit: int | None) -> None:
    total = 0
    for sub in SUBDIRS:
        src_sub = src / sub
        if not src_sub.is_dir():
            continue
        dst_sub = dst / sub
        dst_sub.mkdir(parents=True, exist_ok=True)
        cases = sorted(p for p in src_sub.glob("*.npz") if not p.name.startswith("var_"))
        if limit is not None:
            cases = cases[:limit]
        for i, case in enumerate(cases):
            out_npz = dst_sub / case.name
            if out_npz.exists():  # resumable
                continue
            steady = collapse_case_to_steady(case, frame_steps=frame_steps)
            np.savez_compressed(out_npz, **steady)
            mf = _manifest_beside(case)
            if mf is not None:
                shutil.copy(mf, dst_sub / (case.stem + ".manifest.json"))
            total += 1
            if (i + 1) % 20 == 0 or i + 1 == len(cases):
                print(f"[{sub}] {i + 1}/{len(cases)} ({case.name})", flush=True)
    print(f"done: wrote {total} steady cases to {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--dst", type=Path, default=DST_DEFAULT)
    ap.add_argument("--frame-steps", type=int, default=FRAME_STEPS)
    ap.add_argument("--limit", type=int, default=None, help="cases per split (smoke)")
    args = ap.parse_args()
    build_split(args.src, args.dst, args.frame_steps, args.limit)


if __name__ == "__main__":
    main()
