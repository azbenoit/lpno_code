#!/usr/bin/env python3
"""Figs 7-9 (fig:rp-artifact, fig:varspeed-pred, fig:multibeam-pred): 3-D renders of the metal region at physical
scale, with the laser path scanned so far, and each figure's length-scale strip.

Every panel is the metal region (alpha >= 0.5) of one full-resolution frame, drawn at the solver's 10 um cell size
in all three directions, colored by temperature, seen from above at 30 degrees (the demo website's top view: 12
smoothing passes, zoom 1.42). The panels of one simulation share the camera of its ground truth, so ground truth
and predictions are framed identically and the path projects onto all of them through that one camera. Frames
and paths come from data/path_frames/ (prepare_path_frames.py, which rebuilds the ground-truth frames from the
restricted record). The path is drawn with figstyle.draw_track:
scanned part only, stopping short of the beam with a chevron ahead; a finished path ends in a dot.
Writes each figure's panels, scale strip, path key and (Figs 7, 9) colour-bar strip.

    LPNO_REPLACE_FIGURES=1 python3 reproducibility/figures/make_path_renders.py
"""
import json
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle  # noqa: E402

figstyle.apply()
SRC = HERE / "data" / "path_frames"
TRACKS = json.loads((SRC / "tracks.json").read_text())
PANEL_FRAC = 0.24
WINDOW, SMOOTH, ZOOM, MARGIN = (1100, 825), 12, 1.42, 6
# figure -> scale-bar length (um) and its panels by simulation (ground truth first); each figure's colour limit is
# the highest metal temperature among its panels, rounded up to the kelvin
FIGS = {
    "rp_artifact": (200.0, {"r150np_136": ["rp_artifact_gt", "rp_artifact_rigno", "rp_artifact_fno",
                                                   "rp_artifact_gaot"]}),
    "varspeed_fno": (200.0, {"vsn_066": ["varspeed_fno_gt_fast", "varspeed_fno_pred_fast"],
                                     "vsn_091": ["varspeed_fno_gt_dwell", "varspeed_fno_pred_dwell"]}),
    "multibeam": (200.0, {"mbc40_026": ["multibeam_gt_3beam", "multibeam_pred_3beam"],
                                  "mbc40_031": ["multibeam_gt_4beam", "multibeam_pred_4beam"]}),
}


def scene(T, a, clim, camera=None):
    grid = pv.ImageData(dimensions=T.shape, spacing=(10.0, 10.0, 10.0))
    grid.point_data["alpha"] = a.ravel(order="F")
    grid.point_data["T"] = T.ravel(order="F")
    surf = grid.threshold(0.5, scalars="alpha", all_scalars=True).extract_surface().smooth(n_iter=SMOOTH)
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    pl.set_background("white")
    pl.add_mesh(surf, scalars="T", cmap=figstyle.FIELD_TEMPERATURE, clim=list(clim), smooth_shading=True,
                ambient=0.25, diffuse=0.75, show_scalar_bar=False)
    if camera is None:
        cx, cy, cz = grid.center
        tilt = np.deg2rad(30.0)
        pl.camera_position = [(cx, cy - np.sin(tilt), cz + np.cos(tilt)), (cx, cy, cz),
                              (0.0, np.cos(tilt), np.sin(tilt))]
        pl.reset_camera()
        pl.camera.zoom(ZOOM)
    else:
        pl.camera_position = camera["position"]
        pl.camera.view_angle = camera["angle"]
    pl.reset_camera_clipping_range()
    img = np.asarray(pl.screenshot(return_img=True))[..., :3]
    return pl, img, surf.bounds


def to_px(pl, pts):
    ren, out = pl.renderer, []
    for x, y, z in pts:
        ren.SetWorldPoint(float(x), float(y), float(z), 1.0)
        ren.WorldToDisplay()
        dx, dy, _ = ren.GetDisplayPoint()
        out.append((dx, WINDOW[1] - dy))
    return np.asarray(out)


def tracks_px(pl, case, z_top):
    """Scanned part, head and finished flag of every beam, in window pixels."""
    c = TRACKS[case]
    rx = lambda u: (np.asarray(u) - c["x0_um"]) / c["dx_um"] * 10.0      # noqa: E731
    ry = lambda u: (np.asarray(u) - c["y0_um"]) / c["dx_um"] * 10.0      # noqa: E731
    out = []
    for b in c["beams"]:
        xy, tt, t = np.asarray(b["xy_um"]), np.asarray(b["t_s"]), c["t_s"]
        on = t <= tt[-1] and (b["off_s"] is None or t <= b["off_s"])
        k = int(np.searchsorted(tt, t, side="right"))
        done = xy if k >= len(tt) else np.vstack([xy[:k], [np.interp(t, tt, xy[:, 0]), np.interp(t, tt, xy[:, 1])]])
        head = None
        if on:
            kk = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
            while kk > 1 and np.hypot(*(xy[kk] - xy[kk - 1])) < 1e-9:
                kk -= 1                                 # at rest: the leg it arrived on
            d = (xy[kk] - xy[kk - 1]) / np.linalg.norm(xy[kk] - xy[kk - 1])
            pos = done[-1]
            head = to_px(pl, [(rx(pos[0] - 30 * d[0]), ry(pos[1] - 30 * d[1]), z_top), (rx(pos[0]), ry(pos[1]), z_top)])
        out.append((to_px(pl, [(rx(px), ry(py), z_top) for px, py in done]), head))
    return out


CLIMS = {}
for fig_key, (bar_um, cases) in FIGS.items():
    renders, scales = {}, []
    hi = 0.0
    for names in cases.values():
        for name in names:
            with np.load(SRC / f"{name}.npz") as z:
                hi = max(hi, float(np.asarray(z["T"], np.float32)[np.asarray(z["alpha"], np.float32) >= 0.5].max()))
    hi = CLIMS[fig_key] = float(np.ceil(hi))
    for case, names in cases.items():
        camera, overlay = None, None
        for name in names:
            with np.load(SRC / f"{name}.npz") as z:
                T, a = np.asarray(z["T"], np.float32), np.asarray(z["alpha"], np.float32)
            pl, img, b = scene(T, a, (0.0, hi), camera)
            if camera is None:                  # the ground truth fixes the camera, the path and the scale
                camera = {"position": pl.camera_position, "angle": pl.camera.view_angle}
                z_top = b[5]
                cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
                seg = to_px(pl, [(cx - 50.0, cy, z_top), (cx + 50.0, cy, z_top)])
                px_per_um = float(np.hypot(*(seg[1] - seg[0]))) / 100.0
                overlay = tracks_px(pl, case, z_top)
                r_px = TRACKS[case]["r0_um"] * px_per_um
            renders[name] = (img, overlay, r_px, px_per_um)
            pl.close()
    # one crop box for the whole figure: the union of every panel's content, plus a margin
    box = None
    for img, *_ in renders.values():
        m = (img < 250).any(axis=-1)
        rows, cols = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
        b = [cols[0], rows[0], cols[-1] + 1, rows[-1] + 1]
        box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
    c0, r0 = max(box[0] - MARGIN, 0), max(box[1] - MARGIN, 0)
    c1, r1 = min(box[2] + MARGIN, WINDOW[0]), min(box[3] + MARGIN, WINDOW[1])
    w = figstyle.width_in(PANEL_FRAC)
    for name, (img, overlay, r_px, px_per_um) in renders.items():
        crop = img[r0:r1, c0:c1]
        fig = plt.figure(figsize=(w, w * crop.shape[0] / crop.shape[1]), constrained_layout=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(crop, interpolation="nearest")
        ax.set_xlim(-0.5, crop.shape[1] - 0.5); ax.set_ylim(crop.shape[0] - 0.5, -0.5)
        shift = np.array([c0, r0], float)
        for done, head in overlay:
            figstyle.draw_track(ax, done - shift, None if head is None else head - shift, radius=r_px)
        ax.axis("off")
        print(figstyle.save(fig, name))
        scales.append(px_per_um * PANEL_FRAC * figstyle.TEXTWIDTH_PT / (c1 - c0))     # printed pt per um
    pt_per_um = float(np.mean(scales))
    print(f"{fig_key}: crop {c1 - c0}x{r1 - r0} px, {bar_um:g} um prints {bar_um * pt_per_um:.1f} pt")
    frac = {"multibeam": 0.16}.get(fig_key, 0.2)
    print(figstyle.scale_bar_strip(f"{fig_key}_scale", bar_um, pt_per_um, frac=frac))
    print(f"{fig_key}: colour limit 0-{hi:.0f} K")
    if fig_key != "varspeed_fno":          # Fig 8's bar stands vertically beside its speed panel (make_varspeed_speed_profiles.py)
        print(figstyle.colorbar_strip(f"{fig_key}_cbar", 0.0, hi, force=True))
    # the path key; Fig 9 has a finished path, so its key has a second row
    print(figstyle.track_key_strip(f"{fig_key}_track_key", **({"frac": 0.26, "finished": True}
                                                              if fig_key == "multibeam" else {}), force=True))
# Fig 8's vertical bar is drawn by make_varspeed_speed_profiles.py, which reads these limits
(SRC / "clims.json").write_text(json.dumps(CLIMS, indent=1))
