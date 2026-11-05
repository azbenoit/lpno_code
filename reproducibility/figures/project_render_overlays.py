#!/usr/bin/env python3
"""Printed length scale of the half-block 3-D renders (Figs 10 and 12), for their scale bars.

Figs 10 and 12 are PyVista views of the metal region (alpha >= 0.5) cut at the beam plane, in a three-quarter view.
This script rebuilds each panel's camera from the volume it drew (the renderer of preheat_prediction_panels.ipynb
and make_twoalloy_fig.py) and records the printed length (pt per um) of a segment along x at the middle of the cut
face. Figs 7-9 carry their own scales (make_path_renders.py).

Output: data/render_overlays.json, read by make_render_keys.py. Needs pyvista; Fig 10's volumes are the frozen
source of preheat_prediction_panels.ipynb:

    LPNO_PREHEAT_CASES=<three_cases.pkl> python3 reproducibility/figures/project_render_overlays.py
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

os.nice(19)
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True
HERE = Path(__file__).resolve().parent
TEXTWIDTH_PT = 372.0
SEG_UM = 100.0                        # probe segment for the scale


def to_px(pl, pts, H):
    """World points -> window pixels (column, row from the top)."""
    ren, out = pl.renderer, []
    for p in pts:
        ren.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
        ren.WorldToDisplay()
        dx, dy, _ = ren.GetDisplayPoint()
        out.append((dx, H - dy))
    return np.asarray(out)


def half_block_scene(T, a, cam_offset, zoom=1.25):
    grid = pv.ImageData(dimensions=T.shape, spacing=(10.0,) * 3)
    grid.point_data["alpha"] = np.asarray(a, np.float32).ravel(order="F")
    grid.point_data["T"] = np.asarray(T, np.float32).ravel(order="F")
    thr = grid.threshold(0.5, scalars="alpha", all_scalars=True)
    thr = thr.clip(normal=(0.0, 1.0, 0.0), origin=grid.center, invert=False)
    surf = thr.extract_surface()
    pl = pv.Plotter(off_screen=True, window_size=(1100, 825))
    pl.set_background("white")
    pl.add_mesh(surf, scalars="T", cmap="inferno", clim=[0.0, 3400.0], smooth_shading=True,
                ambient=0.25, diffuse=0.75, show_scalar_bar=False)
    cx, cy, cz = grid.center
    L = max(T.shape) * 10.0
    dx, dy, dz = cam_offset
    pl.camera_position = [(cx + dx * L, cy + dy * L, cz + dz * L), (cx, cy, cz), (0.0, 0.0, 1.0)]
    pl.reset_camera()
    pl.camera.zoom(zoom)
    pl.reset_camera_clipping_range()
    pl.screenshot(return_img=True)
    return pl, grid.center, surf.bounds


def half_block_scale(T, a, cam_offset, frac, trim_lr_bp):
    """pt per um along x in the cut face, for a panel placed at frac*textwidth with that left/right trim."""
    pl, c, b = half_block_scene(T, a, cam_offset)
    pt_per_px = (374.4 / 1100.0) * frac * TEXTWIDTH_PT / (374.4 - sum(trim_lr_bp))
    q = to_px(pl, [(c[0] - SEG_UM / 2, c[1], b[5]), (c[0] + SEG_UM / 2, c[1], b[5])], 825)
    pl.close()
    return float(np.hypot(*(q[1] - q[0])) * pt_per_px / SEG_UM)


out = {}
if os.environ.get("LPNO_PREHEAT_CASES"):
    store = pickle.load(open(os.environ["LPNO_PREHEAT_CASES"], "rb"))
    # the panels draw ground truth from `fl` (Fraction Of Fluid; `alpha` in this pickle is the
    # Cell Volume Fraction) and the prediction from its alpha, as make_fig_npz.py maps them
    out["preheat3d"] = {k: {tag: half_block_scale(store[k][tag]["T"], store[k][tag][key], (-0.06, -1.30, 1.50),
                                                  0.32, (25, 22)) for tag, key in (("gt", "fl"), ("pred", "alpha"))}
                        for k in ("high preheat", "small beam", "big beam")}
with np.load(HERE / "data" / "twoalloy_figure_frames.npz") as z:
    out["twoalloy3d"] = {f"{n}_{tag}": half_block_scale(z[f"{n}_{k}_T"], z[f"{n}_{k}_a"], (-0.06, -1.75, 0.70),
                                                        0.24, (39, 36))
                         for n in ("in718", "ti64") for tag, k in (("gt", "true"), ("pred", "pred"))}

(HERE / "data" / "render_overlays.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:12s} 100 um prints " + ", ".join(
        f"{n} {100 * s:.2f}" if not isinstance(s, dict) else f"{n} " + "/".join(f"{100 * x:.2f}" for x in s.values())
        for n, s in v.items()) + " pt")
