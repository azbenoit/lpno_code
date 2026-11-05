#!/usr/bin/env python3
"""Score the NIST-spot 10/5 um mesh pair once the 5 um arm exports.

Answers one question: does halving the cell produce the keyhole that 10 um
never forms? Reports each arm against the measured radiographs, so the verdict
does not depend on remembering what the 10 um number was.

    python validation/score_spot_mesh_pair.py
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

def _find_sibling(name):
    """Nearest ancestor holding a `name` sibling.

    Walking up rather than taking ROOT.parent because ROOT is
    nested inside the repository when this runs from a git worktree, where the sibling
    repos are three levels further up.
    """
    for base in [ROOT, *ROOT.parents]:
        if (base.parent / name).is_dir():
            return base.parent / name
    return ROOT.parent / name


# The radiograph reader and the spot frames live in the lpno
# sibling. Overridable, and never an absolute path in tracked code.
SIBLING = pathlib.Path(os.environ.get("PINO_SIMULATIONS_ROOT",
                                      _find_sibling("lpno")))
sys.path.insert(0, str(SIBLING / "src" / "notebooks"))
import melt_metrics as mm  # noqa: E402
import spot_radiographs as sr  # noqa: E402

# `runs/` sits in the main checkout, so ROOT/runs is empty when this is invoked
# from a git worktree. Honour the same env var the notebook uses.
RUNS = pathlib.Path(os.environ.get("F3D_RUNS_ROOT", ROOT / "runs")) / "validation"
SPOT = SIBLING / "data" / "experimental" / "spot"
ARMS = [("full 600x600x700, 10 um", "spot_nist102_2ms"),
        ("small 500x500x400, 10 um", "spot_nist102_2ms_sm_10um"),
        ("small 500x500x400,  5 um", "spot_nist102_2ms_sm_5um")]
# Measured: peak cavity 161 um, keyhole onset 867 us, bore 24-42 um.
KEYHOLE_UM = 60.0     # unambiguously past the 2-3 cell dimple the 10 um arms hold


def load(case):
    hits = sorted(glob.glob(str(RUNS / case / "npz" / "*.npz")))
    return np.load(hits[0]) if hits else None


def main():
    meas = np.array([(t, d) for _, t, d in sr.depth_series(SPOT)])
    meas = meas[(meas[:, 0] >= 0) & (meas[:, 0] <= sr.PULSE_US)]
    print(f"measured: peak {meas[:, 1].max():.0f} um, "
          f"onset(>20 um) {meas[meas[:, 1] > 20][0, 0]:.0f} us\n")
    print(f"{'arm':28s} {'cell':>5s} {'peak':>7s} {'cells':>6s} {'onset':>7s} "
          f"{'Tmax':>6s}  keyhole?")
    for label, case in ARMS:
        z = load(case)
        if z is None:
            print(f"{label:28s} {'-':>5s} not exported yet")
            continue
        cell = mm.cell_size_um(z)
        d = mm.depression_series(z)
        t = z["time_s"] * 1e6
        on = t <= sr.PULSE_US
        peak = float(d[on].max())
        hit = t[on][d[on] > KEYHOLE_UM]
        onset = f"{hit[0]:.0f} us" if hit.size else "never"
        print(f"{label:28s} {cell:5.0f} {peak:6.0f}u {peak / cell:5.1f} {onset:>7s} "
              f"{z['Temperature'].max():6.0f}  {'YES' if hit.size else 'no'}")
        v = mm.domain_validity(z)
        if not v.ok or v.depth_truncated:
            print(f"{'':28s} WARNING domain_validity {v}")
    print("\nA 'YES' on 5 um with 'no' on both 10 um arms = resolution artifact.\n"
          "'no' everywhere eliminates the mesh and implicates recoil/reflection.\n"
          "The small-box 10 um arm is the control for the domain change; if it\n"
          "disagrees with the full-domain arm, the box moved and the pair is void.")


if __name__ == "__main__":
    main()
