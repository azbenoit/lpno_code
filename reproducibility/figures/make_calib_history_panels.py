#!/usr/bin/env python3
"""Fig calib-history (a-d): depression depth and absorbed fraction against time.

Four panels, laid out 2x2 in one LaTeX figure: the 195 W / 0.7 m/s Ti-6Al-4V scan
(a: calib_depth_history, b: calib_absorptance_history) and the 101.9 W stationary spot
(c: calib_spot_depth, d: calib_spot_abs) of the NIST AM Bench record. Each shows the measured
trace and the simulated ones at 10 and 5 um cells, every spiky trace as a faint raw line under
its running median. The traces are in data/figure_data/<name>.json (polylines clipped to the axes
box: a point on the box edge marks where a line leaves the frame); calibration_figures.ipynb
(cells 22, 23, 27) computes them from the validation simulations and the NIST measurements.

    LPNO_REPLACE_FIGURES=1 python3 make_calib_history_panels.py
"""
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

FIGDATA = HERE / "data" / "figure_data"
NAMES = ["calib_depth_history", "calib_absorptance_history", "calib_spot_depth", "calib_spot_abs"]
FRAC = 0.49
ASPECT = 0.80

# Mesh identity shared with the radiograph overlays (notebook cell 22): colour and
# weight say which cell size a trace belongs to, across the whole section.
C10, C5, CMEAS = figstyle.PALETTE[0], figstyle.PALETTE[1], "#000000"
LW10, LW5 = 3.0, 1.6
RAW_LW, RAW_ALPHA = 0.7, 0.30
SERIES = [("measured", CMEAS), ("10um", C10), ("5um", C5)]
LABELS = {"measured": "Measured", "10um": "10 µm", "5um": "5 µm"}
# Measured-trace weight per panel, as the notebook drew it.
MEAS_LW = {"calib_depth_history": 1.0, "calib_absorptance_history": 1.6,
           "calib_spot_depth": 1.5, "calib_spot_abs": 1.5}
MEDIAN_LW = {"10um": LW10, "5um": LW5}
AXES = {"calib_depth_history": ("time (us)", "depression depth (um)"),
        "calib_absorptance_history": ("time (us)", "absorbed fraction (%)"),
        "calib_spot_depth": ("time (us)", "depression depth (um)"),
        "calib_spot_abs": ("time (us)", "absorbed fraction (%)")}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _xy(pieces):
    """Pieces -> one x, y pair of lists with NaN breaks, drawn as one line."""
    xs, ys = [], []
    for piece in pieces:
        if xs:
            xs.append(float("nan"))
            ys.append(float("nan"))
        xs += [p[0] for p in piece]
        ys += [p[1] for p in piece]
    return xs, ys


def plot(name):
    figstyle.apply()
    # The traces are stored as drawn polylines; a second simplification would drop up
    # to a third of the measured trace's vertices.
    plt.rcParams["path.simplify"] = False
    d = json.loads((FIGDATA / f"{name}.json").read_text())
    fig, ax = figstyle.panel(frac=FRAC, aspect=ASPECT)
    if d["span"]:
        ax.axvspan(*d["span"], color="0.85", alpha=0.5, lw=0)
    for key, colour in SERIES:
        s = d["series"][key]
        lw = MEAS_LW[name] if key == "measured" else MEDIAN_LW[key]
        if "raw" in s:
            ax.plot(*_xy(s["raw"]), "-", color=colour, lw=RAW_LW, alpha=RAW_ALPHA)
        if "points" in s:     # (a): the eight segmented radiograph depths, joined
            ax.plot([p[0] for p in s["points"]], [p[1] for p in s["points"]], "o-",
                    color=colour, ms=5, lw=lw, label=LABELS[key])
        else:
            ax.plot(*_xy(s["median"]), "-", color=colour, lw=lw, label=LABELS[key])
    ax.set_xlim(*d["xlim"])
    ax.set_ylim(*d["ylim"])
    ax.set_xticks(d["xticks"])
    ax.set_yticks(d["yticks"])
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Depression depth (µm)" if "depth" in name else "Absorbed fraction (%)")
    if d["span"]:
        # Named in place, at the foot of the band where both scan panels are empty.
        ax.text(LABEL_X, 0.03, "Drilling\ntransient", ha="center", va="bottom", color="0.45",
                transform=ax.get_xaxis_transform())
    # One row above the frame, centred on the panel, so it cannot cover a trace.
    leg = ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
                    bbox_transform=blended_transform_factory(fig.transFigure, ax.transAxes),
                    handlelength=1.3, handletextpad=0.5, columnspacing=1.0,
                    borderpad=0.0, borderaxespad=LEG_GAP)
    _reserve_legend_row(fig, ax, leg)
    return figstyle.save(fig, name)


def _reserve_legend_row(fig, ax, leg):
    """Constrained layout for the frame, then lower its top to free the legend row.

    matplotlib 3.3's constrained layout charges a legend placed above the frame
    with its horizontal overhang as well, which narrowed the frame by 19 pt. So
    the frame is laid out with the legend excluded, the layout is frozen, and the
    frame top is set to leave exactly the legend's height plus its gap.
    """
    leg.set_in_layout(False)
    fig.canvas.draw()
    pos = ax.get_position()
    fig.set_constrained_layout(False)
    r = fig.canvas.get_renderer()
    pad = plt.rcParams["figure.constrained_layout.h_pad"] * fig.dpi
    gap = LEG_GAP * figstyle.FONT * fig.dpi / 72
    top = 1 - (leg.get_window_extent(r).height + gap + pad) / fig.bbox.height
    ax.set_position([pos.x0, pos.y0, pos.width, top - pos.y0])


LABEL_X = 275.0     # centre of the band label, us: the foot of 140-400 us is empty
LEG_GAP = 0.45      # legend-to-frame gap, in font sizes: clears the "80" tick label


if __name__ == "__main__":
    for n in NAMES:
        print(plot(n))
