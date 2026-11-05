#!/usr/bin/env python3
"""Plot a laser scan path to a PNG for quick visual inspection of a run.

Imported lazily by ``run_rectangular_case.py`` (which is otherwise stdlib only),
so matplotlib is only needed when a path plot is actually written.  The path
points, mesh bounds, and beam radius come from ``case_manifest.json``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MM = 1000.0  # metres -> millimetres for the axes


def plot_path(
    path_xy_m,
    x_bounds_m,
    y_bounds_m,
    beam_radius_m,
    out_png,
    *,
    case_name: str = "",
    path_type: str = "",
    length_m: float | None = None,
):
    """Write a PNG of the in-plane laser path.

    ``path_xy_m`` is an ordered list of ``[x, y]`` points in metres (the sampled
    scan path).  The domain rectangle and the 4-beam-radius keep-out are drawn
    for context; the path is coloured by progress (green start -> red end).
    """
    xs = np.array([p[0] for p in path_xy_m]) * MM
    ys = np.array([p[1] for p in path_xy_m]) * MM
    x0, x1 = x_bounds_m[0] * MM, x_bounds_m[1] * MM
    y0, y1 = y_bounds_m[0] * MM, y_bounds_m[1] * MM
    m = 4.0 * beam_radius_m * MM

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="0.4", lw=1.4, label="domain"))
    if (x1 - x0) > 2 * m and (y1 - y0) > 2 * m:
        ax.add_patch(
            Rectangle((x0 + m, y0 + m), (x1 - x0) - 2 * m, (y1 - y0) - 2 * m,
                      fill=False, ec="crimson", ls="--", lw=1.0, label="4·beam-radius keep-out")
        )
    if len(xs) >= 2:
        pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", array=np.linspace(0, 1, len(segs)), lw=1.6)
        ax.add_collection(lc)
    ax.plot(xs[0], ys[0], "o", ms=7, mfc="lime", mec="k", zorder=5, label="start")
    ax.plot(xs[-1], ys[-1], "s", ms=7, mfc="red", mec="k", zorder=5, label="end")

    ax.set_aspect("equal")
    pad = 0.05 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    subtitle = ", ".join(
        s for s in (path_type, f"length {length_m * MM:.3g} mm" if length_m else "", f"{len(xs)} pts") if s
    )
    ax.set_title(f"Laser scan path — {case_name}" + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png
