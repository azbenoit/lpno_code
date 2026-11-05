#!/usr/bin/env python3
"""fig:calib-keyhole: vapor-depression depth against laser power (a) and scan speed (b).

Simulated depression depth of the validation simulations (FLOW-3D WELD, 10 um cells, Ti-6Al-4V,
beam radius 47.5 um) against the in-situ X-ray measurements of Gan et al. (2021), with the
+-30 % band and the number of cells across each simulated cavity. The plotted values are in
data/figure_data/calib_keyhole_ladder.json and calib_speed_ladder.json; calibration_figures.ipynb
computes them from the validation simulations (melt_metrics.depression) and the measurement file
flow3D_scripts/validation/anchors/gan_points.json.

    LPNO_REPLACE_FIGURES=1 python3 make_calib_keyhole_panels.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

DATA = HERE / "data" / "figure_data"
FRAC = 0.49
ASPECT = 0.80          # shared by both panels of the figure
BAND = 0.30            # melt_metrics.DEFAULT_BAND, the +-30 % agreement band
GREY = "0.45"          # cell-count annotations
CELL_KEY = "Cells deep"
# (nominal power, colour, marker) of the two speed series, as in notebook cell 7
SPEED_SERIES = ((208.0, figstyle.PALETTE[0], "s"), (156.0, figstyle.PALETTE[3], "D"))
SPEED_POWER = {208.0: "208\u2013209 W", 156.0: "156\u2013158 W"}   # the powers of each series


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
# Where each cell count sits relative to its simulated marker. Above, as in the
# notebook, except where the measured marker or the other series' line is in the
# way at 20 pt type; keyed by (nominal power, scan speed).
SPEED_LABEL_PLACE = {(208.0, 0.9): "left", (208.0, 1.1): "right",
                     (156.0, 0.9): "below", (156.0, 1.2): "above_marker"}
_PLACE = {"above": dict(xytext=(0, 7), ha="center", va="bottom"),
          "above_marker": dict(xytext=(0, 12), ha="center", va="bottom"),  # clears a measured dot just above
          "below": dict(xytext=(0, -7), ha="center", va="top"),
          "right": dict(xytext=(7, 0), ha="left", va="center"),
          "left": dict(xytext=(-7, 0), ha="right", va="center")}


class _Numeral(HandlerBase):
    """Legend handle for the grey cell counts: a grey numeral in the annotation's
    own type, so the key reads '9 Cells deep'."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        # on the label's baseline (y = -ydescent), so it lines up with the label text
        return [Text(width / 2 - xdescent, -ydescent, "9", color=GREY,
                     ha="center", va="baseline", transform=trans)]


def _cells_key():
    return plt.Line2D([], [], ls="", label=CELL_KEY)


def _annotate(ax, x, y, cells, place="above"):
    ax.annotate(f"{cells}", (x, y), textcoords="offset points", color=GREY, **_PLACE[place])


def _legend_above(ax, handles):
    """Two rows above the axes, flush with its right edge so the wider legend
    runs out over the y label. At 20 pt the four keys cover data anywhere inside
    these axes. Anchored to the axes, so constrained layout makes room for it."""
    key = next(h for h in handles if h.get_label() == CELL_KEY)
    ax.legend(handles=handles, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              borderaxespad=0.1, borderpad=0.0,
              handlelength=0.9, handletextpad=0.2, columnspacing=0.5, labelspacing=0.3,
              handler_map={key: _Numeral()})


def plot_keyhole():
    d = json.loads((DATA / "calib_keyhole_ladder.json").read_text())
    fig, ax = figstyle.panel(frac=FRAC, aspect=ASPECT)
    pts = d["points"]
    P = [p["P"] for p in pts]; meas = [p["meas"] for p in pts]; sim = [p["sim"] for p in pts]
    band = ax.fill_between(d["band"]["P"], d["band"]["lo"], d["band"]["hi"],
                           color=figstyle.PALETTE[0], alpha=0.13, lw=0, label="\u00b130 % band")
    hm, = ax.plot(P, meas, "o-", color="k", ms=4.8, lw=1.0, label="Measured")
    hs, = ax.plot(P, sim, "s", color=figstyle.PALETTE[0], ms=6, label="Simulated")
    for p in pts:
        _annotate(ax, p["P"], p["sim"], p["cells"])
    ax.set_xlabel("Laser power (W)")
    ax.set_ylabel("Depression depth (\u00b5m)")
    ax.set_xlim(*d["xlim"]); ax.set_ylim(*d["ylim"])
    ax.set_xticks(d["xticks"]); ax.set_yticks(d["yticks"])
    _legend_above(ax, [hm, _cells_key(), hs, band])
    return figstyle.save(fig, "calib_keyhole_ladder")


def plot_speed():
    d = json.loads((DATA / "calib_speed_ladder.json").read_text())
    fig, ax = figstyle.panel(frac=FRAC, aspect=ASPECT)
    hs = []
    for s, (P_nom, colour, mark) in zip(d["series"], SPEED_SERIES):
        V, meas, sim = s["V"], np.array(s["meas"]), s["sim"]
        ax.fill_between(V, meas * (1 - BAND), meas * (1 + BAND), color=colour, alpha=0.13, lw=0)
        hm, = ax.plot(V, meas, "o-", color="k", ms=4.8, lw=1.0, label="Measured")
        h, = ax.plot(V, sim, mark, color=colour, ms=6, label=f"Simulated, {SPEED_POWER[P_nom]}")
        hs.append(h)
        for v, y, c in zip(V, sim, s["cells"]):
            _annotate(ax, v, y, c, SPEED_LABEL_PLACE.get((P_nom, round(v, 2)), "above"))
    ax.set_xlabel("Scan speed (m/s)")
    ax.set_ylabel("Depression depth (\u00b5m)")
    ax.set_xlim(*d["xlim"]); ax.set_ylim(*d["ylim"])
    ax.set_xticks(d["xticks"]); ax.set_yticks(d["yticks"])
    _legend_above(ax, [hm, _cells_key()] + hs)
    return figstyle.save(fig, "calib_speed_ladder")


def plot():
    figstyle.apply()
    return [plot_keyhole(), plot_speed()]


if __name__ == "__main__":
    print(plot())
