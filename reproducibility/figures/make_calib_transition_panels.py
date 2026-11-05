#!/usr/bin/env python3
"""fig:calib-transition: melt-pool error (a) and aspect ratio (b) for Ti-6Al-4V.

Simulated melt-pool depth and width of the validation simulations (melt_metrics.fusion_geometry_auto)
against the metallographic sections of Galiegue et al. (2025), at 0.20 and 0.40 m/s and
r0 = 60 um: one colour per quantity, solid lines and filled markers for 0.20 m/s, dashed lines and
open markers for 0.40 m/s. The plotted values are in data/figure_data/calib_transition_ladder.json
and calib_aspect_ratio.json; calibration_figures.ipynb computes them from the validation simulations
and flow3D_scripts/validation/anchors/galiegue2025_ti64_targets.json.

    LPNO_REPLACE_FIGURES=1 python3 make_calib_transition_panels.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

DATA = HERE / "data" / "figure_data"
FRAC = 0.49
ASPECT = 0.80          # shared by both panels of the figure
BAND = 30.0            # melt_metrics.DEFAULT_BAND, in percent
NEUTRAL = "0.35"       # legend handles that key a line style, not a series
# Speed series as the notebook draws them: (speed, line style, marker face)
SPEEDS = ((0.20, "-", None), (0.40, "--", "none"))
# (a) one colour and marker per quantity, notebook cell 13
ERR = (("depth", figstyle.PALETTE[5], "o"), ("width", figstyle.PALETTE[2], "s"))
# (b) measured and simulated aspect, notebook cell 17
ASP = (("measured", "#000000", "o"), ("simulated", figstyle.PALETTE[0], "s"))


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _speed_keys():
    """Line style and marker fill key the scan speed; colour keys the series."""
    return [plt.Line2D([], [], color=NEUTRAL, ls=ls, marker="o", mfc=mfc,
                       label=f"{speed:.2f} m/s") for speed, ls, mfc in SPEEDS]


def _kind_keys(kinds):
    return [plt.Line2D([], [], color=c, ls="", marker=m, label=k.capitalize()) for k, c, m in kinds]


def _legend_above(ax, handles, ncol):
    """Two rows above the axes, flush with its right edge. Four series at 20 pt
    do not fit inside without covering data, so the key is factored: colour
    and marker name the quantity, line style and fill name the speed."""
    ax.legend(handles=handles, ncol=ncol, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              borderaxespad=0.2, borderpad=0.2, handlelength=1.4, handletextpad=0.4,
              columnspacing=0.8, labelspacing=0.3)


def _panel(name, field, kinds):
    dj = json.loads((DATA / f"{name}.json").read_text())
    fig, ax = figstyle.panel(frac=FRAC, aspect=ASPECT)
    for s in dj["series"]:
        colour, mark = {k: (c, m) for k, c, m in kinds}[s["kind"]]
        ls, mfc = {sp: (l, f) for sp, l, f in SPEEDS}[s["speed_m_s"]]
        ax.plot(s["P"], s[field], marker=mark, ls=ls, mfc=mfc, color=colour)
    ax.set_xlabel("Laser power (W)")
    ax.set_xlim(*dj["xlim"]); ax.set_ylim(*dj["ylim"])
    ax.set_xticks(dj["xticks"]); ax.set_yticks(dj["yticks"])
    return fig, ax


def plot_transition():
    fig, ax = _panel("calib_transition_ladder", "err_pct", ERR)
    ax.axhspan(-BAND, BAND, color=figstyle.PALETTE[0], alpha=0.13, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Melt-pool error (%)")
    band = Patch(facecolor=figstyle.PALETTE[0], alpha=0.13, lw=0, label=f"±{BAND:.0f} %")
    _legend_above(ax, _kind_keys(ERR) + _speed_keys() + [band], ncol=3)
    return figstyle.save(fig, "calib_transition_ladder")


def plot_aspect():
    fig, ax = _panel("calib_aspect_ratio", "aspect", ASP)
    ax.set_ylabel("Melt pool aspect ratio")
    _legend_above(ax, _kind_keys(ASP) + _speed_keys(), ncol=2)
    return figstyle.save(fig, "calib_aspect_ratio")


def plot():
    figstyle.apply()
    return [plot_transition(), plot_aspect()]


if __name__ == "__main__":
    print(plot())
