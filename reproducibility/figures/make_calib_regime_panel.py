#!/usr/bin/env python3
"""Panel (b) of fig:calib-materials: depth error against simulated cavity aspect ratio, 35 scored
validation conditions.

The plotted points are in data/calib_regime_points.json; calibration_figures.ipynb (cell 31)
computes them from the validation simulations. The plotting code is the notebook's.

    LPNO_REPLACE_FIGURES=1 python3 make_calib_regime_panel.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle

figstyle.apply()

BAND = 0.30                     # mm.DEFAULT_BAND, the +-30 % agreement band
MARK = {"Ti-6Al-4V": "o", "CoCr": "s", "304L": "D", "IN718": "^"}
MCOL = {"Ti-6Al-4V": figstyle.PALETTE[0], "CoCr": figstyle.PALETTE[1],
        "304L": figstyle.PALETTE[2], "IN718": figstyle.PALETTE[3]}
TIC = MCOL["Ti-6Al-4V"]

rows = json.loads((pathlib.Path(__file__).resolve().parent
                   / "data" / "calib_regime_points.json").read_text())["rows"]
FUS = [r for r in rows if r["kind"] == "fusion"]
DEP = [r for r in rows if r["kind"] == "depression"]

fig, ax = figstyle.panel(frac=0.49, aspect=0.92)
ax.axhspan(-BAND * 100, BAND * 100, color=figstyle.PALETTE[0], alpha=.13, lw=0)
ax.axhline(0, color="k", lw=.6)
for r in FUS:
    ax.plot(r["aspect"], r["err"], MARK[r["mat"]], color=MCOL[r["mat"]],
            ms=6.5, zorder=3)
for r in DEP:
    ax.plot(r["aspect"], r["err"], "+", color=TIC, ms=8, mew=1.8, zorder=4)
ax.set_xlabel("Simulated cavity aspect ratio")
ax.set_ylabel("Depth error (%)")
ax.set_xlim(0, 1.75)
# Two arrows, not a dividing line. Conduction gives way to keyholing gradually
# and nothing in this data marks the crossing.
y0, y1 = ax.get_ylim()
ax.set_ylim(y0, y1 + 0.16 * (y1 - y0))
ya = ax.get_ylim()[1] - 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])
_arr = dict(arrowstyle="-|>", color="0.45", lw=1.1, shrinkA=2, shrinkB=0)
_sty = dict(fontsize=figstyle.FONT, color="0.45", va="center", annotation_clip=False,
            arrowprops=_arr)
ax.annotate("Conduction", xy=(0.02, ya), xytext=(0.87, ya), ha="right", **_sty)
ax.annotate("Deep keyhole", xy=(1.73, ya), xytext=(0.93, ya), ha="left", **_sty)
print("y limits", ax.get_ylim())
figstyle.save(fig, "calib_summary_regime")
