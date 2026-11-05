#!/usr/bin/env python3
"""fig:dataset-regimes (Supplementary Note A): the simulations of each dataset, by melt regime.

One bar per dataset of tab:datasets, its height the number of simulations the models were trained, validated and tested
on, split by the vapor-cavity class aiida-pino assigns each simulation (no cavity, shallow cavity, keyhole;
aiida_pino.console.viz.classify_regime). Data: data/dataset_regimes.json, with the label of every simulation.

    LPNO_REPLACE_FIGURES=1 python3 make_dataset_regimes_fig.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

DATA = HERE / "data" / "dataset_regimes.json"
FRAC = 1.0
# aiida-pino's labels, named for what its rule measures: "LoF" is a vapor cavity under two cells (no melt-depth test),
# "conduction" a cavity that is not a keyhole
REGIMES = [("LoF", "No cavity", figstyle.PALETTE[4]), ("conduction", "Shallow cavity", figstyle.PALETTE[0]),
           ("keyhole", "Keyhole", figstyle.PALETTE[1])]
LABELS = {"Power and scan speed grid": "Power and scan speed grid", "Variable paths": "Variable paths",
          "Variable speed": "Variable speed", "Preheat sweep": "Preheat sweep", "Beam-radius sweep": "Beam-radius sweep",
          "Two-alloy, IN718": "Active-learning two-alloy (IN718)", "Multi-beam": "Multi-beam"}


def main():
    figstyle.apply()
    rows = json.loads(DATA.read_text())["datasets"]
    fig, ax = figstyle.panel(frac=FRAC, aspect=0.42)
    y = np.arange(len(rows))[::-1]          # the table's order, top to bottom
    left = np.zeros(len(rows))
    for key, label, colour in REGIMES:
        w = np.array([r[key] for r in rows], float)
        ax.barh(y, w, left=left, height=0.62, color=colour, label=label, lw=0)
        left += w
    for yi, total in zip(y, left):
        ax.text(total + 4, yi, f"{int(total)}", ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[r["dataset"]] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Simulations")
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.2,
              columnspacing=1.4, borderaxespad=0.2)
    return figstyle.save(fig, "dataset_regimes")


if __name__ == "__main__":
    print(main())
