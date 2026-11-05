#!/usr/bin/env python3
"""The validation melt cross-sections: three pairs of panels.

    calib_metric_defs, calib_metric_defs_along   fig:calib-metrics
    calib_spot_size_r60, calib_spot_size_r130     fig:calib-spot
    calib_mesh_c10, calib_mesh_c5                 fig:calib-mesh

Each panel draws one simulation cell per mask entry (melted metal, vapor cavity) on the
simulation's own 10 um (5 um for the refined run) lattice, the substrate surface, and the
measured melt-pool depth and width where the panel compares against a measurement
(Galiegue et al. 2025). The masks, lines and arrow positions are in data/figure_data/<name>.json;
calibration_figures.ipynb (cells 9, 10, 15, 19) computes them from the validation simulations.

    LPNO_REPLACE_FIGURES=1 python3 make_calib_cross_sections.py
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

REC = HERE / "data" / "figure_data"
NAMES = ["calib_metric_defs", "calib_metric_defs_along",
         "calib_spot_size_r60", "calib_spot_size_r130",
         "calib_mesh_c10", "calib_mesh_c5"]
CELL_UM = {"calib_mesh_c5": 5.0}          # every other run is on the 10 um mesh

# Mask colours, as in the notebook: MELT_C = Blues(0.65), CAV_C = Okabe-Ito yellow.
MELT_C = "#3b8bc2"
CAV_C = "#F0E442"
MEAS_C = figstyle.PALETTE[5]              # measured fusion depth / width (vermillion)
GREY = "0.25"                             # annotations in the along-track panel


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
FRAC = 0.49
W_PT = figstyle.width_in(FRAC) * 72.0
# Fixed margins in points, so the two panels of a pair put their axes at the
# same place on the page: tick labels (-300) and the rotated axis label on the
# left, tick labels and the axis label below, the legend above.
LEFT, BOTTOM, GAP = 84.0, 58.0, 6.0
METRIC_TOP = {"calib_metric_defs": 90.0, "calib_metric_defs_along": 90.0}
LEGEND = dict(loc="upper center", handlelength=1.2, handletextpad=0.4,
              columnspacing=0.8, borderpad=0.2)


def load(name):
    d = json.loads((REC / f"{name}.json").read_text())
    if name in METRIC_TOP:
        # The originals stopped 45 um (across) and 78 um (along) above the
        # surface. At body size that leaves no room for "fusion width" anywhere
        # but on top of the pool, nor for the along-track labels above the bead,
        # so both panels now run to the same top: one z range, one scale.
        d["ylim_original"] = list(d["ylim"])
        d["ylim"][1] = METRIC_TOP[name]
    return d


def _handles(kind):
    import matplotlib.patches as mpatches
    melt = mpatches.Patch(facecolor=MELT_C, label="Melted metal")
    if kind == "metric":
        return [melt, mpatches.Patch(facecolor=CAV_C, label="Vapor cavity")], 2
    blank = plt.Line2D([], [], ls="none", label=" ")
    depth = plt.Line2D([], [], color=MEAS_C, lw=2.2, ls="--", label="Measured depth")
    width = plt.Line2D([], [], color=MEAS_C, lw=1.2, ls=":", label="Measured width")
    # two columns filled top to bottom: the mask on its own, the two measured
    # lengths together (three entries on one row are wider than the panel)
    return [melt, blank, depth, width], 2


def _legend_size(kind):
    handles, ncol = _handles(kind)
    fig = plt.figure(figsize=(6, 2), dpi=72)
    leg = fig.legend(handles=handles, ncol=ncol, **LEGEND)
    fig.canvas.draw()
    bb = leg.get_window_extent()
    plt.close(fig)
    return bb.height, bb.width


def pair_layout(names, kind, right):
    """One axes height for both panels of a pair (equal aspect, so each width
    follows from its own limits) and the figure height that holds them."""
    lh, lw = _legend_size(kind)
    assert lw < W_PT - 4, f"legend {lw:.0f} pt wider than the panel"
    spans = []
    for n in names:
        d = load(n)
        spans.append((d["xlim"][1] - d["xlim"][0], d["ylim"][1] - d["ylim"][0]))
    h = min((W_PT - LEFT - right) * zs / xs for xs, zs in spans)
    return h, BOTTOM + h + lh + GAP + 1.0


def new_panel(d, h, H):
    fig, ax = figstyle.panel(frac=FRAC, aspect=H / W_PT)
    # placed by hand: constrained layout would fit each equal-aspect axes to its
    # own labels, and the twins of a pair would land at different places
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine("none")
    else:
        fig.set_constrained_layout(False)
    xs, zs = d["xlim"][1] - d["xlim"][0], d["ylim"][1] - d["ylim"][0]
    ax.set_position([LEFT / W_PT, BOTTOM / H, h * xs / zs / W_PT, h / H])
    ax.set_xlim(*d["xlim"]); ax.set_ylim(*d["ylim"])
    ax.set_aspect("equal", anchor="SW")
    ax.set_xticks(d["xticks"]); ax.set_yticks(d["yticks"])
    draw_cells(ax, d["cells"])
    ax.axhline(d["surface"]["z"], color="k", lw=d["surface"]["lw"])
    return fig, ax


def draw_cells(ax, c):
    from matplotlib.colors import ListedColormap
    C = np.ma.masked_equal(np.array([[int(ch) for ch in r] for r in c["rows"]], float), 0.0)
    ax.pcolormesh(c["x_edges"], c["z_edges"], C, cmap=ListedColormap([MELT_C, CAV_C]),
                  vmin=0.5, vmax=2.5, shading="flat", rasterized=True)


def finish(fig, kind, H, name):
    handles, ncol = _handles(kind)
    fig.legend(handles=handles, ncol=ncol, bbox_to_anchor=(0.5, 1.0 - 1.0 / H),
               bbox_transform=fig.transFigure, **LEGEND)
    return figstyle.save(fig, name)


def plot_measured(name, h, H):
    """Spot-size and mesh panels: melt union, surface, measured depth and width."""
    d = load(name)
    fig, ax = new_panel(d, h, H)
    ax.axhline(d["measured_depth"]["z"], color=MEAS_C, lw=d["measured_depth"]["lw"], ls="--")
    for x in d["measured_width"]["x"]:
        ax.axvline(x, color=MEAS_C, lw=d["measured_width"]["lw"], ls=":")
    if "depth_note" in d:
        # simulated fusion depth, at the height the original used. At body size
        # it is wider than the gap left of the measured-width line, so it is
        # centred between the two lines instead of crossing one.
        ax.text(0.5, d["depth_note"]["axes_frac"][1], d["depth_note"]["text"],
                transform=ax.transAxes, ha="center", color=GREY)
    ax.set_xlabel("$y$ (µm)")
    ax.set_ylabel("$z$ (µm)")
    return finish(fig, "measured", H, name)


def plot_metric_across(h, H):
    d = load("calib_metric_defs")
    fig, ax = new_panel(d, h, H)
    ann = dict(arrowprops=dict(arrowstyle="<->", color="k", lw=d["arrow_lw"]))
    a = d["arrows"]
    ax.annotate("", xy=tuple(a["width"][0]), xytext=tuple(a["width"][1]), **ann)
    ax.annotate("", xy=tuple(a["depth"][1]), xytext=tuple(a["depth"][0]), **ann)
    ax.annotate("", xy=tuple(a["depression"][1]), xytext=tuple(a["depression"][0]), **ann)
    xd, xp = a["depth"][0][0], a["depression"][0][0]
    # above the pool, on white, centred over the width arrow
    ax.text(0, 30, "Melt-pool width", ha="center", va="baseline")
    ax.text(xd - 10, -d["fusion_depth"] / 2, "Melt-pool depth", ha="right", va="center", rotation=90)
    # three horizontal lines are wider than the space right of the arrow at body
    # size, so the label turns like its fusion-depth twin, clear of the rim
    # (melted cells reach y = 130 um just below the surface)
    ax.text(xp + 18, -d["depression_depth"] / 2, "Vapor depression\ndepth", ha="left",
            va="center", rotation=90, multialignment="center")
    ax.set_xlabel("$y$ from track center (µm)")
    ax.set_ylabel("$z$ (µm)")
    return finish(fig, "metric", H, "calib_metric_defs")


def plot_metric_along(h, H):
    d = load("calib_metric_defs_along")
    fig, ax = new_panel(d, h, H)
    px = d["plane"]["x"]
    ax.axvline(px, color=GREY, lw=d["plane"]["lw"], ls="--")
    # Both labels stay in the clear band above the rim: grey text on the blue
    # mask is unreadable. At body size "scan direction" is wider than the band
    # right of the plane, where the original put it, so it and its arrow move
    # left of the plane and the plane label stands against its own line.
    ax.text(px + 8, 48, "Plane of (a)", ha="left", va="baseline", color=GREY)
    t = ax.text(-325, 48, "Scan direction", ha="left", va="baseline", color=GREY)
    fig.canvas.draw()
    x1 = t.get_window_extent().transformed(ax.transData.inverted()).x1
    sa = d["scan_arrow"]
    ax.annotate("", xy=(px - 14, 59), xytext=(x1 + 6, 59),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=sa["lw"]))
    ax.set_xlabel("$x$ from the plane of (a) (µm)")
    ax.set_ylabel("$z$ (µm)")
    return finish(fig, "metric", H, "calib_metric_defs_along")


def plot():
    figstyle.apply()
    out = []
    # right margin: the along-track panel's last tick label (200) sits 20 um
    # from the axis end and overhangs it
    h, H = pair_layout(["calib_metric_defs", "calib_metric_defs_along"], "metric", right=12.0)
    out += [plot_metric_across(h, H), plot_metric_along(h, H)]
    print(f"metric pair: aspect {H / W_PT:.3f}, axes height {h:.1f} pt")
    for pair in (["calib_spot_size_r60", "calib_spot_size_r130"], ["calib_mesh_c10", "calib_mesh_c5"]):
        h, H = pair_layout(pair, "measured", right=4.0)
        out += [plot_measured(n, h, H) for n in pair]
        print(f"{pair}: aspect {H / W_PT:.3f}, axes height {h:.1f} pt")
    return out


if __name__ == "__main__":
    for p in plot():
        print(p)
