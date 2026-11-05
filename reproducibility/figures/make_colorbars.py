#!/usr/bin/env python3
"""The colour-bar strips of the half-block 3-D render figures (Figs 10, 12), re-typeset without data.

Their render scripts (make_twoalloy_fig.py and preheat_prediction_panels.ipynb)
compute each clim from the frames they render and draw the strip with
figstyle.colorbar_strip. This script redraws only the strips, at the clims those
renders used, so the strips can follow a style change on a machine without the
frames, at the clims those renders used. Figs 7-9 draw their strips in make_path_renders.py.

    LPNO_REPLACE_FIGURES=1 python3 reproducibility/figures/make_colorbars.py
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle

figstyle.apply()

# name, clim upper bound (K). Width and size come from figstyle.colorbar_strip,
# so every strip looks identical.
STRIPS = [
    ("preheat3d_cbar", 3348.0),
    ("twoalloy3d_cbar", 3380.0),
]

for name, hi in STRIPS:
    print(figstyle.colorbar_strip(name, 0.0, hi, force=True))
