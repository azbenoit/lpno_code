#!/usr/bin/env python3
"""Panel (a) of fig:calib-materials: simulated against measured length for every scored
validation condition (35 across four materials), log-log.

Filled markers are melt-pool depth, open markers melt-pool width and crosses vapor-depression
depth, with the diagonal and the +-30 % band. The plotted lengths are in
data/figure_data/calib_summary_parity.json; calibration_figures.ipynb computes them from the
validation simulations and the measurement files in flow3D_scripts/validation/anchors/. The
sibling panel (b) is make_calib_regime_panel.py; the two share frac and aspect.

    LPNO_REPLACE_FIGURES=1 python3 make_calib_summary_parity.py
"""
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

DATA = HERE / "data" / "figure_data" / "calib_summary_parity.json"
REGIME = HERE / "data" / "calib_regime_points.json"
FRAC, ASPECT = 0.49, 0.92            # as calib_summary_regime, its twin
BAND = 0.30                           # melt_metrics.DEFAULT_BAND
LO, HI = 20.0, 750.0                  # axis limits, um. Was 620 in notebook cell 31, which hid the 304L 456 W width (717 um measured)
MARK = {"Ti-6Al-4V": "o", "CoCr": "s", "304L": "D", "IN718": "^"}
MCOL = {"Ti-6Al-4V": figstyle.PALETTE[0], "CoCr": figstyle.PALETTE[1],
        "304L": figstyle.PALETTE[2], "IN718": figstyle.PALETTE[3]}
TIC = MCOL["Ti-6Al-4V"]               # depression depth exists for Ti-6Al-4V only
NEUTRAL = "0.35"


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot():
    figstyle.apply()
    dj = json.loads(DATA.read_text())
    FUS = [r for r in dj["rows"] if r["kind"] == "fusion"]
    DEP = [r for r in dj["rows"] if r["kind"] == "depression"]
    fig, ax = figstyle.panel(frac=FRAC, aspect=ASPECT)
    grid = np.array([LO, HI])
    ax.fill_between(grid, grid * (1 - BAND), grid * (1 + BAND), color="0.6", alpha=0.18, lw=0)
    ax.plot(grid, grid, "k-", lw=1.0)
    for r in FUS:
        c, mk = MCOL[r["mat"]], MARK[r["mat"]]
        ax.plot(r["meas_d"], r["sim_d"], mk, color=c, ms=6.5, zorder=3)
        if r["meas_w"] is not None:
            ax.plot(r["meas_w"], r["sim_w"], mk, color=c, ms=6.5, mfc="none", zorder=3)
    for r in DEP:
        ax.plot(r["meas_d"], r["sim_d"], "+", color=TIC, ms=8, mew=1.8, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
    # One labelled decade, as before, written as 100 (a 10^2 superscript would
    # print below the body size).
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Measured length (µm)")
    ax.set_ylabel("Simulated length (µm)")
    mats = [plt.Line2D([], [], ls="", marker=MARK[m], color=MCOL[m], ms=6.5, label=m) for m in MARK]
    kinds = [plt.Line2D([], [], ls="", marker="o", color=NEUTRAL, ms=6.5, label="Melt-pool depth"),
             plt.Line2D([], [], ls="", marker="o", color=NEUTRAL, ms=6.5, mfc="none",
                        label="Melt-pool width"),
             plt.Line2D([], [], ls="", marker="+", color=TIC, ms=8, mew=1.8,
                        label="Depression depth")]
    # Two keys at 20 pt. Materials fit the empty corner above the band at upper
    # left; the three kinds do not fit inside anywhere without covering the
    # sub-keyhole Ti-6Al-4V points below the band, so they go above the axes.
    leg = ax.legend(handles=mats, loc="upper left", handlelength=0.8, handletextpad=0.4,
                    labelspacing=0.3, borderaxespad=0.3, borderpad=0.2)
    ax.add_artist(leg)
    ax.legend(handles=kinds, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              handlelength=0.8, handletextpad=0.3, columnspacing=0.6,
              labelspacing=0.3, borderaxespad=0.2, borderpad=0.2)
    return [figstyle.save(fig, "calib_summary_parity")]


if __name__ == "__main__":
    print(plot())
