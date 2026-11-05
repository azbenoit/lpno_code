#!/usr/bin/env python3
"""Fig 8(e): beam speed along the two variable-speed tracks shown in Fig 8.

The speed traces of the two cases (vsn_066 fast, vsn_091 dwell) come from their laser-path
manifests (in the open record's simulation nodes); they are stored in
data/figure_data/varspeed_speed_profiles.json. This script draws from that file.

    LPNO_REPLACE_FIGURES=1 python3 make_varspeed_speed_profiles.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

DATA = HERE / "data" / "figure_data" / "varspeed_speed_profiles.json"
FRAC = 0.42
TRACKS = [("fast", "#0072b2"), ("dwell", "#e69f00")]


# Fixed plot box, in authored inches, so the temperature bar beside it (drawn here
# too, from the same numbers) spans exactly the same height on the page.
L_IN, R_IN, B_IN, T_IN = 1.05, 0.15, 0.85, 0.12
# the clim of Fig 8's four renders, written by make_path_renders.py
CLIM = (0.0, json.loads((HERE / "data" / "path_frames" / "clims.json").read_text())["varspeed_fno"])


def plot():
    figstyle.apply()
    d = json.loads(DATA.read_text())
    w = figstyle.width_in(FRAC)
    ax_w = w - L_IN - R_IN
    ax_h = ax_w * 0.42
    h = B_IN + ax_h + T_IN
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    ax = fig.add_axes([L_IN / w, B_IN / h, ax_w / w, ax_h / h])
    for name, colour in TRACKS:
        tr = d[name]
        ax.plot(tr["t"], tr["speed"], color=colour, lw=2.0, label=name.capitalize())
        ax.plot(*tr["instant"], "o", color=colour, ms=9, mec="white", mew=1.0, zorder=3)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(-0.05, 1.65)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.legend(loc="upper right", ncol=1, frameon=False, handlelength=1.2)
    out = figstyle.save(fig, "varspeed_fno_speed_profiles")
    bar = figstyle.colorbar_column("varspeed_fno_cbar_vertical", *CLIM, height_in=h,
                                   bottom_in=B_IN, bar_h_in=ax_h, frac=0.15,
                                   label="$T$ (K)", force=True)
    return out, bar


if __name__ == "__main__":
    print(plot())
