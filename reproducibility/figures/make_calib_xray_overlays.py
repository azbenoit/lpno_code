#!/usr/bin/env python3
"""Validation radiograph figure (fig:calib-xray): simulated cavity on the X-ray frames.

Four panels, one LaTeX figure. Top row: the scanned Ti-6Al-4V track at 10 and 5 um cells at
127 us (drilling transient) and 627 us (settled keyhole), panels calib_xray_transient and
calib_xray_settled. Bottom row: the stationary 101.9 W spot at 847 us and 1407 us, panels
calib_spot_overlay_early and calib_spot_overlay_late, where the dashed line is the simulated
melt-pool boundary. Radiographs: NIST AM Bench record (flat-fielded frames, gray colour map,
2nd-99.5th percentile window), stored as 8-bit images with their extent in um in
data/figure_data/<name>.json and <name>_radiograph.png, with the simulated contours.
calibration_figures.ipynb (cells 24, 28) computes the contours from the validation simulations.

    LPNO_REPLACE_FIGURES=1 python3 make_calib_xray_overlays.py
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

REC = HERE / "data" / "figure_data"
FRAC = 0.49
XRAY = ("calib_xray_transient", "calib_xray_settled")
SPOT = ("calib_spot_overlay_early", "calib_spot_overlay_late")
# Figure aspect per row: what fit_layout() returns when the equal-aspect image
# fills the panel width with no empty band (the two panels of a row are twins).
# draw() asserts it, so a change of text or limits that moves it fails loudly.
ASPECT = {"xray": 0.6278, "spot": 0.9239}

# Okabe-Ito blue/orange, as in the notebook. The 10 um contour is drawn wider and
# underneath, so where the two meshes agree both stay visible.
C10, C5 = figstyle.PALETTE[0], figstyle.PALETTE[1]
LW10, LW5 = 3.0, 1.6    # cavity lines; the dashed fusion boundary is 0.6x these,
                         # recorded per contour in the JSON from the original paths
MESH = {C10.lower(): "10 µm", C5.lower(): "5 µm"}
NOTEBOOK_LIMITS = {"xray": ((-305, 175), (-170, 80)), "spot": ((-190, 190), (-250, 80))}


def scale_bar(ax, length_um=100, pad=0.055, colour="w"):
    """Notebook scale_bar, at the one figure text size."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xe = x1 - pad * (x1 - x0); xs = xe - length_um
    y = y0 + pad * (y1 - y0)
    ax.plot([xs, xe], [y, y], color=colour, lw=3.0, solid_capstyle="butt", zorder=5)
    ax.text((xs + xe) / 2, y + 0.035 * (y1 - y0), f"{length_um:g} µm",
            color=colour, ha="center", va="bottom", zorder=5)
    return xs, xe, y


def fit_layout(fig, ax, pad_pt=1.5):
    """Place the equal-aspect axes to fill the panel width, and size the height to fit.

    Constrained layout (matplotlib 3.3) does not converge on an equal-aspect axes
    with a legend above it: it clipped the x label and the legend. Here the
    decorations (ticks, labels, legend) are measured and the axes box is set so
    that its data aspect is exactly 1 and every glyph sits inside the canvas.
    Returns the resulting figure aspect (height / width).
    """
    fig.set_constrained_layout(False)
    W = fig.get_size_inches()[0] * 72
    span = np.ptp(ax.get_ylim()) / np.ptp(ax.get_xlim())
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        k = 72 / fig.dpi
        tb, ab = ax.get_tightbbox(r), ax.get_window_extent(r)
        left, right = (ab.x0 - tb.x0) * k, (tb.x1 - ab.x1) * k
        bottom, top = (ab.y0 - tb.y0) * k, (tb.y1 - ab.y1) * k
        aw = W - left - right - 2 * pad_pt
        ah = aw * span
        H = ah + bottom + top + 2 * pad_pt
        fig.set_size_inches(W / 72, H / 72)
        ax.set_position([(left + pad_pt) / W, (bottom + pad_pt) / H, aw / W, ah / H])
    fig.canvas.draw()
    tb = ax.get_tightbbox(fig.canvas.get_renderer())
    assert tb.x0 >= -0.5 and tb.y0 >= -0.5 and tb.x1 <= fig.bbox.x1 + 0.5 \
        and tb.y1 <= fig.bbox.y1 + 0.5, "decorations outside the canvas"
    return H / W


def draw(name):
    row = "spot" if name in SPOT else "xray"
    d = json.loads((REC / f"{name}.json").read_text())
    img = np.asarray(Image.open(REC / d["radiograph"]["file"]))
    fig, ax = figstyle.panel(frac=FRAC, aspect=ASPECT[row])
    # The stored raster is already the 600 dpi nearest-neighbour rendering of the
    # detector pixels (the notebook's default interpolation resolved to nearest at
    # that upsampling); nearest keeps those pixel blocks sharp instead of filtering them.
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=d["radiograph"]["extent_um"],
              origin="upper", aspect="equal", interpolation="nearest")
    for c in d["contours"]:
        x = np.concatenate([np.r_[s[0], np.nan] for s in c["segments"]])
        z = np.concatenate([np.r_[s[1], np.nan] for s in c["segments"]])
        ax.plot(x, z, color=c["colour"], lw=c["lw"],
                ls="--" if c["kind"] == "fusion boundary" else "-")
    ax.axhline(d["zero_line_z_um"], color="0.3", lw=0.6, ls=":")
    ax.set_xlim(*d["xlim"]); ax.set_ylim(*d["ylim"])
    ax.set_xticks(d["xticks"]); ax.set_yticks(d["yticks"])
    ax.set_xlabel(r"$x$ from beam (µm)" if row == "xray" else r"$x$ (µm)")
    ax.set_ylabel(r"$z$ (µm)")
    xs, xe, y = scale_bar(ax)
    assert np.allclose([xs, xe, y], d["scale_bar_um"]["x"] + [d["scale_bar_um"]["y"]], atol=1e-2)
    ax.text(0.02, 0.05, f"$t$ = {d['t_us']:.0f} µs", transform=ax.transAxes,
            color="w", va="bottom")
    # Key above the axes: colour = mesh. The spot row also needs the line style, and
    # its three entries do not fit one row at the figure text size, so it is a 2x2
    # key: mesh in the first column, boundary type (solid cavity, dashed melt pool) in
    # the second.
    handles = [plt.Line2D([], [], color=C10, lw=LW10, label="10 µm"),
               plt.Line2D([], [], color=C5, lw=LW5, label="5 µm")]
    if row == "spot":
        handles += [plt.Line2D([], [], color="0.35", lw=LW5, label="Cavity boundary"),
                    plt.Line2D([], [], color="0.35", lw=LW5, ls="--", label="Melt-pool boundary")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2 if row == "spot" else len(handles), handlelength=1.2,
              handletextpad=0.4, columnspacing=0.8, borderpad=0.0, borderaxespad=0.3)
    aspect = fit_layout(fig, ax)
    assert abs(aspect - ASPECT[row]) < 2e-3, f"{name}: fitted aspect {aspect:.4f}, update ASPECT"
    return figstyle.save(fig, name)


if __name__ == "__main__":
    figstyle.apply()
    # every stored vertex is drawn (no path simplification)
    with matplotlib.rc_context({"path.simplify": False}):
        for n in XRAY + SPOT:
            print(draw(n))
