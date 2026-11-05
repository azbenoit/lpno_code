#!/usr/bin/env python3
"""Ground-truth frames of the Fig 10 and Fig 12 renders, rebuilt from the deposit.

FLOW-3D WELD output is distributed only in the restricted record, so these frames are kept out of version
control (.gitignore) and written here: data/preheat_truth_frames.npz (fig:preheat-pred) and
data/twoalloy_truth_frames.npz (fig:twoalloy-pred). The predictions beside them are committed
(data/preheat_prediction_frames.npz, data/twoalloy_prediction_frames.npz). The ground-truth frames of
Figs 7-9 come from prepare_path_frames.py.

  * Fig 12: the joint RIGNO's evaluation previews in restricted_files_main.tar (lazy_preview_batch0.npz of
    evaluations 3a3492a6, IN718 extrapolation slice, and 6c883ef6, Ti-6Al-4V test split), each case picked
    by laser power. Needs only the tar.
  * Fig 10: aiida-pino's own derivation (derive_simulation_arrays) of the three preheat simulations with the
    derivation profile their evaluations used (time average over the last 30 frames in the beam frame). Needs
    an AiiDA profile holding the open archives and the three payloads, restored with
        aiida-pino restore-payloads --deposit <restricted record> --yes \\
            --for c82581e6-33b6-46e3-8e6f-03b3163936e4 --for 6c188437-1f82-4895-9a90-1b799e3383c1 \\
            --for e6ca3821-9589-41a4-b398-57e1628aac85

    LPNO_RESTRICTED=<restricted record dir> python3 prepare_truth_frames.py [fig10] [fig12]
"""
import io
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TSCALE = 3000.0
WHICH = set(sys.argv[1:]) or {"fig10", "fig12"}

# Fig 12: (artifact folder of the evaluation, its preview's object sha256, laser power of the shown case in W)
TWOALLOY = {"in718": ("c9a1f952-9ba3-49d3-908c-d5c1e473aa6d",
                      "34645208bdd79e1442b362caf4766e015f37e510b9202e9fe58c3481dc353d6a", 346.8),
            "ti64": ("909cc245-a0b9-4935-9a15-42d62296589a",
                     "fb867861debda7c6414ce7d45e82de3c517b28d54f25cb406c47b5dc83524a72", 225.0)}
# Fig 10: panel tag -> deposited simulation; the derivation profile of their evaluations
PREHEAT = {"hot": "665ca22b-0349-4ffa-8c4e-2deb01d1c0a6",      # ti64_ph_113, 754 K preheat
           "small": "425acbad-a7b4-4842-bffb-d5b7879cf18d",    # ti64_ph_107, 35 um beam
           "big": "9a109f4d-72ea-454c-8e90-71b1970da7aa"}      # ti64_ph_082, 73 um beam
PREHEAT_PROFILE = {"crop_z_top": 0, "derived_fields": ["conduction_field"], "include_time": True,
                   "materialize": False, "moving_frame_steps": 30, "moving_ref_frame": True, "stride": [1, 1, 1],
                   "super_res_factor": 1, "time_as_batch": False, "time_as_dim": False, "time_avg_window": 30,
                   "time_frames": None}


def fig12(restricted):
    tar = restricted / "restricted_files_main.tar"
    out = {}
    with tarfile.open(tar) as tf:
        names = tf.getnames()
        for n, (folder, sha, P_w) in TWOALLOY.items():
            member = [m for m in names if m.endswith(sha)]
            assert len(member) == 1, f"{sha[:12]} not in {tar.name}"
            z = np.load(io.BytesIO(tf.extractfile(member[0]).read()))
            P = [float(z["in_P"][i].flat[0]) * 10 for i in range(z["in_P"].shape[0])]
            i = int(np.argmin([abs(p - P_w) for p in P]))
            assert abs(P[i] - P_w) < 1.0, f"{n}: no case at {P_w} W in the preview, have {P}"
            out[f"{n}_true_T"] = (z["true_T"][i, 0] * TSCALE).astype(np.float32)
            out[f"{n}_true_a"] = z["true_alpha"][i, 0].astype(np.float32)
    np.savez_compressed(DATA / "twoalloy_truth_frames.npz", **out)
    print("wrote", DATA / "twoalloy_truth_frames.npz")


def fig10():
    from aiida import load_profile, orm
    load_profile()
    from aiida_pino.data.pino_dataset_collection import derive_simulation_arrays
    out = {}
    for short, uuid in PREHEAT.items():
        arrays, _ = derive_simulation_arrays(orm.load_node(uuid), PREHEAT_PROFILE)
        A = lambda k: np.asarray(arrays[k], np.float32).reshape(np.asarray(arrays[k]).shape[-3:])
        out[f"{short}_true_T_K"] = A("Temperature")
        out[f"{short}_true_alpha"] = A("Fraction Of Fluid")
    np.savez_compressed(DATA / "preheat_truth_frames.npz", **out)
    print("wrote", DATA / "preheat_truth_frames.npz")


if __name__ == "__main__":
    if "fig12" in WHICH:
        fig12(Path(os.environ["LPNO_RESTRICTED"]))
    if "fig10" in WHICH:
        fig10()
