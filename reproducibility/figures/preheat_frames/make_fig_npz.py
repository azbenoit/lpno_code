"""Frozen figure source for the manuscript preheat 3D panels.
Volumes from the verified local forward of m65230 (see
preheat_frames/local_forward.py): channels bit-exact vs eval
calc 65233's preview, forward reproduces its pred_T to 0.2 K mean in metal.
GT alpha is `Fraction Of Fluid` (== loader alpha, verified); temperatures in
Kelvin (pred scaled by ref_scales.T = 3000, recorded not derived)."""
import pickle
import sys
from pathlib import Path

import numpy as np

OUT = str(Path(__file__).resolve().parent / "figure_frames.npz")
# `three_cases.pkl` is an intermediate of local_forward.py that holds FLOW-3D ground-truth fields, so it is
# never tracked (restricted record only); pass its path as the first argument. The predictions alone are kept
# in reproducibility/figures/data/preheat_prediction_frames.npz.
store = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "three_cases.pkl", "rb"))
KF = 3000.0
tags = {"high preheat": "hot", "small beam": "small", "big beam": "big"}
payload = {}
for tag, short in tags.items():
    e = store[tag]
    m = e["meta"]
    payload[f"{short}_true_T_K"] = e["gt"]["T"].astype(np.float32)
    payload[f"{short}_pred_T_K"] = (e["pred"]["T"] * KF).astype(np.float32)
    payload[f"{short}_true_alpha"] = e["gt"]["fl"].astype(np.float32)   # Fraction Of Fluid
    payload[f"{short}_pred_alpha"] = e["pred"]["alpha"].astype(np.float32)
    payload[f"{short}_meta"] = np.array([m["P"], m["V"], m["T0"], m["r_um"]], dtype=np.float64)
    payload[f"{short}_name"] = np.array(m["name"])
np.savez_compressed(OUT, **payload)
import os
print("wrote", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB")
for k in sorted(payload):
    if k.endswith("meta"): print(k, payload[k])
