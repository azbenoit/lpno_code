#!/usr/bin/env python3
"""Length scales for the half-block 3-D render figures (Figs 10 and 12), one strip per figure.

Each strip stands beside the figure's colour bar. Its bar length comes from the
camera projections in data/render_overlays.json (project_render_overlays.py): the
printed length of a segment along x at the middle of the beam-plane cut face. Figs 7-9
draw their strips in make_path_renders.py. Perspective changes that length by about
10 % between the front and back of a block, so the bar holds at that reference
point only. Where the panels of one figure differ slightly in framing, the bar uses
their mean.

    LPNO_REPLACE_FIGURES=1 python3 reproducibility/figures/make_render_keys.py
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle  # noqa: E402

figstyle.apply()
ov = json.loads((HERE / "data" / "render_overlays.json").read_text())

scales = {
    "preheat3d_scale": (100.0, np.mean([s for case in ov["preheat3d"].values() for s in case.values()])),
    "twoalloy3d_scale": (100.0, np.mean(list(ov["twoalloy3d"].values()))),
}
FRAC = {"twoalloy3d_scale": 0.16}   # beside the vertical bar in the empty slot of Fig 12
for name, (length_um, pt_per_um) in scales.items():
    print(f"{name}: {length_um:g} um bar prints {length_um * pt_per_um:.1f} pt")
    print(figstyle.scale_bar_strip(name, length_um, pt_per_um, frac=FRAC.get(name, 0.2)))
