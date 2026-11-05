#!/usr/bin/env python3
"""fig:gaot-smoothing: the GAOT panel of Fig 7 (fig:rp-artifact) before and after Gaussian post-processing of its temperature.

The frame carries the evaluation's fill: T was blended toward 3520 K and alpha toward 0 with a tanh gate of the network's alpha. The
fill is undone on the predicted metal, T is smoothed there (normalized over the metal, so the gas never enters) with a standard
deviation of a quarter of the latent token spacing along each axis, and the smoothed T is used in full below 1200 K, tapering to the
network's T at the 1898 K melt threshold. The fill then goes back on. Rendered as Fig 7 (make_path_renders.py), on Fig 7's colour
scale and camera, so its colour bar and path key are reused; only the scale strip differs, for the wider panels.

    LPNO_REPLACE_FIGURES=1 python3 reproducibility/figures/make_gaot_smoothing.py

With `--metrics <eval_frames_dump.npz>` it also prints the caption's numbers for the frame, from the case's evaluation dump (frames 0,
k, 2k, ..., position 1 is the frame shown), with the evaluation's definitions.
"""
import json
import os
import pathlib
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle  # noqa: E402
from frame_metrics import T_FILL, gate, score, substrate_and_box, undo_fill  # noqa: E402

figstyle.apply()
SRC = HERE / "data" / "path_frames"
TRACKS = json.loads((SRC / "tracks.json").read_text())
CASE = "r150np_136"
HI = json.loads((SRC / "clims.json").read_text())["rp_artifact"]     # Fig 7's colour limit
PANEL_FRAC = 0.49
WINDOW, SMOOTH, ZOOM, MARGIN = (2200, 1650), 12, 1.42, 12
T_FULL, T_NONE = 1200.0, 1898.0                                       # smoothed T in full below, network T from here up
CELL_UM = 10.0
LATTICE_SPACING_UM = np.array([1490.0 / 43.0, 990.0 / 19.0, 390.0 / 23.0])    # the 44 x 20 x 24 token lattice over the box
SIGMA_CELLS = 0.25 * LATTICE_SPACING_UM / CELL_UM
def smooth_network(T, a):
    """The network's T after smoothing below melting; the gas is left as it is."""
    metal = a >= 0.5
    num = gaussian_filter(np.where(metal, T, 0.0), SIGMA_CELLS, mode="nearest", truncate=3.0)
    den = gaussian_filter(metal.astype(np.float64), SIGMA_CELLS, mode="nearest", truncate=3.0)
    smooth = np.where(metal, num / np.maximum(den, 1e-3), T)
    w = np.clip((T_NONE - smooth) / (T_NONE - T_FULL), 0.0, 1.0)
    return np.where(metal, w * smooth + (1.0 - w) * T, T)


def smoothed_frame(T_filled, a_filled):
    """(stored T in K, stored alpha) -> stored T after smoothing, with the fill back on the metal; gas and alpha do not change."""
    T, a = undo_fill(T_filled, a_filled)
    return np.where(a >= 0.5, T_FILL + gate(a) * (smooth_network(T, a) - T_FILL), T_filled)


def metrics(dump_path):
    """The caption's numbers: the frame shown, before and after, as frame_metrics.py scores the panels of Figs 7-9."""
    with np.load(dump_path) as z:
        pT, pa = z["pred_T"][1, 0].astype(np.float64) * 3000.0, z["pred_alpha"][1, 0].astype(np.float64)
        tT, ta = z["true_T"][1, 0].astype(np.float64) * 3000.0, z["true_alpha"][1, 0].astype(np.float64)
    sub, box = substrate_and_box(TRACKS[CASE], pT.shape)
    raw, ra = undo_fill(pT, pa)
    for name, Q, T_net in (("before", pT, None), ("after", smoothed_frame(pT, pa), smooth_network(raw, ra))):
        r = score(Q, pa, tT, ta, sub, box, T_net=T_net)
        print(f"{name}: rel. err. {r['rel_err_pct']:.2f} %, T MAE {r['t_mae_K']:.2f} K, T RMSE {r['t_rmse_K']:.2f} K, "
              f"BB MAE {r['bb_mae_K']:.1f} K, melt-pool IoU {r['melt_pool_iou']:.3f}, box {r['box_cells']} cells")


def scene(T, a, camera=None):
    grid = pv.ImageData(dimensions=T.shape, spacing=(CELL_UM, CELL_UM, CELL_UM))
    grid.point_data["alpha"] = a.ravel(order="F")
    grid.point_data["T"] = T.ravel(order="F")
    surf = grid.threshold(0.5, scalars="alpha", all_scalars=True).extract_surface().smooth(n_iter=SMOOTH)
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    pl.set_background("white")
    pl.add_mesh(surf, scalars="T", cmap=figstyle.FIELD_TEMPERATURE, clim=[0.0, HI], smooth_shading=True,
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


def track_px(pl, z_top):
    """Scanned part of the path and its head, in window pixels."""
    c = TRACKS[CASE]
    rx = lambda u: (np.asarray(u) - c["x0_um"]) / c["dx_um"] * CELL_UM      # noqa: E731
    ry = lambda u: (np.asarray(u) - c["y0_um"]) / c["dx_um"] * CELL_UM      # noqa: E731
    b = c["beams"][0]
    xy, tt, t = np.asarray(b["xy_um"]), np.asarray(b["t_s"]), c["t_s"]
    k = int(np.searchsorted(tt, t, side="right"))
    done = xy if k >= len(tt) else np.vstack([xy[:k], [np.interp(t, tt, xy[:, 0]), np.interp(t, tt, xy[:, 1])]])
    kk = int(np.clip(np.searchsorted(tt, t), 1, len(tt) - 1))
    while kk > 1 and np.hypot(*(xy[kk] - xy[kk - 1])) < 1e-9:
        kk -= 1
    d = (xy[kk] - xy[kk - 1]) / np.linalg.norm(xy[kk] - xy[kk - 1])
    pos = done[-1]
    head = to_px(pl, [(rx(pos[0] - 30 * d[0]), ry(pos[1] - 30 * d[1]), z_top), (rx(pos[0]), ry(pos[1]), z_top)])
    return to_px(pl, [(rx(px), ry(py), z_top) for px, py in done]), head


def main():
    with np.load(SRC / "rp_artifact_gt.npz") as z:
        tT, ta = np.asarray(z["T"], np.float32), np.asarray(z["alpha"], np.float32)
    with np.load(SRC / "rp_artifact_gaot.npz") as z:
        pT, pa = np.asarray(z["T"], np.float64), np.asarray(z["alpha"], np.float64)
    # the ground truth fixes the camera, the path and the scale, as in Fig 7
    pl, _, b = scene(tT, ta)
    camera = {"position": pl.camera_position, "angle": pl.camera.view_angle}
    z_top = b[5]
    cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
    seg = to_px(pl, [(cx - 50.0, cy, z_top), (cx + 50.0, cy, z_top)])
    px_per_um = float(np.hypot(*(seg[1] - seg[0]))) / 100.0
    done, head = track_px(pl, z_top)
    r_px = TRACKS[CASE]["r0_um"] * px_per_um
    pl.close()
    renders = {}
    for name, field in (("gaot_smoothing_before", pT), ("gaot_smoothing_after", smoothed_frame(pT, pa))):
        pl, img, _ = scene(field.astype(np.float32), pa.astype(np.float32), camera)
        renders[name] = img
        pl.close()
    box = None
    for img in renders.values():
        m = (img < 250).any(axis=-1)
        rows, cols = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
        q = [cols[0], rows[0], cols[-1] + 1, rows[-1] + 1]
        box = q if box is None else [min(box[0], q[0]), min(box[1], q[1]), max(box[2], q[2]), max(box[3], q[3])]
    c0, r0 = max(box[0] - MARGIN, 0), max(box[1] - MARGIN, 0)
    c1, r1 = min(box[2] + MARGIN, WINDOW[0]), min(box[3] + MARGIN, WINDOW[1])
    w = figstyle.width_in(PANEL_FRAC)
    for name, img in renders.items():
        crop = img[r0:r1, c0:c1]
        fig = plt.figure(figsize=(w, w * crop.shape[0] / crop.shape[1]), constrained_layout=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(crop, interpolation="nearest")
        ax.set_xlim(-0.5, crop.shape[1] - 0.5)
        ax.set_ylim(crop.shape[0] - 0.5, -0.5)
        shift = np.array([c0, r0], float)
        figstyle.draw_track(ax, done - shift, head - shift, radius=r_px)
        ax.axis("off")
        print(figstyle.save(fig, name))
    pt_per_um = px_per_um * PANEL_FRAC * figstyle.TEXTWIDTH_PT / (c1 - c0)
    print(figstyle.scale_bar_strip("gaot_smoothing_scale", 200.0, pt_per_um, frac=0.2))
    print(f"standard deviations {0.25 * LATTICE_SPACING_UM.round(2)} um, {SIGMA_CELLS.round(3)} cells")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--metrics":
        metrics(sys.argv[2])
    main()
