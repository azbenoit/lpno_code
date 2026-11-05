#!/usr/bin/env python3
"""Source data of the Fig 7-9 renders (make_path_renders.py): one full-resolution frame per panel and the laser
paths of each simulation, written to data/path_frames/.

Frames come from the per-case evaluations of the tabled tab:paths-accuracy models (reproducibility/rerun/eval.sbatch on one-case
collections of the deposit, settings of each model's published evaluation, PINO_DUMP_FRAMES=k with k the shown
frame's index, so the dump holds frames 0, k, 2k, ...): `pred_*` is the model's prediction. Ground-truth panels
are the deposited FLOW-3D WELD frame k (Temperature, Fraction Of Fluid), checked against the dump's `true_*` on
metal cells when a dump is given. Temperatures are stored in kelvin and fractions as float16, which is ample for
rendering. The ground-truth frames are solver output, so they are kept out of version control (.gitignore) and
rebuilt from the restricted record; the prediction frames and tracks.json are committed. Paths are read from the deposited case manifests (single beams) and rebuilt with flow3D_scripts' own
multi-beam table generator (each beam checked against the recorded sample counts and path length).

    LPNO_RESTRICTED=<restricted record dir> LPNO_FLOW3D_SCRIPTS=<flow3D_scripts/python> \\
        python3 reproducibility/figures/prepare_path_frames.py [<dir holding the dumps>]

Without a dump directory only the ground-truth frames and tracks.json are written.
"""
import importlib.util
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "path_frames"
DUMPS = Path(sys.argv[1]) if len(sys.argv) > 1 else None
DEP = Path(os.environ["LPNO_RESTRICTED"])
TSCALE = 3000.0
# case -> deposit tar, simulation uuid, shown frame index, beam radius (um)
CASES = {
    "r150np_136": ("payloads_ti64_rand150_newphys.tar", "245463e3-c8f7-4153-bc3e-99df9d5e4344", 130, 50.0),
    "vsn_066": ("payloads_ti64_varspeed_newphys.tar", "24fc48a2-8932-4768-a1a2-1e20256b37b8", 78, 50.0),
    "vsn_091": ("payloads_ti64_varspeed_newphys.tar", "8de044f6-a7a5-47fa-8aeb-806f69f98a79", 174, 50.0),
    "mbc40_026": ("payloads_ti64_multibeam40_capped.tar", "2cabfd59-f577-496d-977b-cc83f7848b37", 185, 47.5),
    "mbc40_031": ("payloads_ti64_multibeam40_capped.tar", "df38e155-e7ae-49c6-af6b-0dcf8c95a112", 205, 47.5),
}
# panel -> (case, model whose dump it reads, which field). FNO and GAOT: the uniform-drawing re-run models of
# tab:paths-accuracy (2026-11-02); the RIGNO panel keeps the variable-path RIGNO of that table (chosen on validation
# BB MAE, 2026-11-03).
PANELS = {
    "rp_artifact_gt": ("r150np_136", "uni_fno_rand150_b2_12ch_35k_s43", "true"),
    "rp_artifact_fno": ("r150np_136", "uni_fno_rand150_b2_12ch_35k_s43", "pred"),
    "rp_artifact_rigno": ("r150np_136", "rigno_rand150_newphys_35k_e8_12ch", "pred"),
    "rp_artifact_gaot": ("r150np_136", "uni_gaot_rand150_b2_12ch_35k_s43", "pred"),
    "varspeed_fno_gt_fast": ("vsn_066", "uni_fno_varspeed_b2_12ch_35k_s43", "true"),
    "varspeed_fno_pred_fast": ("vsn_066", "uni_fno_varspeed_b2_12ch_35k_s43", "pred"),
    "varspeed_fno_gt_dwell": ("vsn_091", "uni_fno_varspeed_b2_12ch_35k_s43", "true"),
    "varspeed_fno_pred_dwell": ("vsn_091", "uni_fno_varspeed_b2_12ch_35k_s43", "pred"),
    "multibeam_gt_3beam": ("mbc40_026", "uni_fno_mb_ft6k_b2_12ch_s43", "true"),
    "multibeam_pred_3beam": ("mbc40_026", "uni_fno_mb_ft6k_b2_12ch_s43", "pred"),
    "multibeam_gt_4beam": ("mbc40_031", "uni_fno_mb_ft6k_b2_12ch_s43", "true"),
    "multibeam_pred_4beam": ("mbc40_031", "uni_fno_mb_ft6k_b2_12ch_s43", "pred"),
}


def manifest_and_axes(tar, uuid, k):
    """Manifest, axes, times, frame 0 (to check a dump's orientation) and frame k of one deposited simulation."""
    with tarfile.open(DEP / tar) as tf:
        man = json.load(tf.extractfile(tf.getmember(uuid + ".manifest.json")))
        with np.load(tf.extractfile(tf.getmember(uuid + ".npz"))) as z:
            axes = {k: np.asarray(z[k]) * 1e6 for k in ("x", "y", "z")}
            t = np.asarray(z["time_s"])
            TT, AA = z["Temperature"], z["Fraction Of Fluid"]
            frames = {i: (np.asarray(TT[i], np.float32), np.asarray(AA[i], np.float32)) for i in (0, k)}
    return man, axes, t, frames


def single_paths(man):
    d = man["derived"]
    xy = np.asarray(d["laser_path_xy_m"]) * 1e6
    if d.get("laser_path_times_s"):
        tt = np.asarray(d["laser_path_times_s"])
    else:
        tt = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) / (
            man["input"]["laser"]["scan_speed_m_per_s"] * 1e6)
    return [(xy, tt, None)]


def multi_paths(man):
    py = os.environ["LPNO_FLOW3D_SCRIPTS"]
    sys.path.insert(0, py)
    spec = importlib.util.spec_from_file_location("rrc", py + "/run_rectangular_case.py")
    rrc = importlib.util.module_from_spec(spec)
    sys.modules["rrc"] = rrc
    spec.loader.exec_module(rrc)
    geo, laser = man["input"]["geometry"], man["input"]["laser"]
    L, W = geo["length_m"], geo["width_m"]
    _, _, _, meta = rrc.multi_beam_tables(laser["beams"], radius=laser["beam_radius_m"],
                                          bounds=(-L / 2, L / 2, -W / 2, W / 2), default_power=0.0,
                                          default_speed=0.1, sample_spacing=None)
    out = []
    for rec, b in zip(man["derived"]["beams"], meta):
        assert rec["samples_used"] == b["samples_used"] and abs(rec["path_length_m"] - b["path_length_m"]) < 1e-12
        built = [tuple(p) for p in b["path_xy_m"]]
        if b["samples_used"] < len(built):
            step = (len(built) - 1) / (b["samples_used"] - 1)
            picks = sorted({int(round(i * step)) for i in range(b["samples_used"])} | {0, len(built) - 1})
            built = [built[i] for i in picks]
        xy = np.asarray(built) * 1e6
        tt = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) / (b["scan_speed_m_per_s"] * 1e6)
        out.append((xy, tt, b["turn_off_time_s"]))
    return out


OUT.mkdir(parents=True, exist_ok=True)
tracks = {}
for case, (tar, uuid, k, r0) in CASES.items():
    man, axes, t, frames = manifest_and_axes(tar, uuid, k)
    paths = multi_paths(man) if man["input"]["laser"].get("beams") else single_paths(man)
    t_now = float(t[k])
    tracks[case] = {"t_s": t_now, "frame": k, "r0_um": r0, "x0_um": float(axes["x"][0]), "y0_um": float(axes["y"][0]),
                    "z0_um": float(axes["z"][0]), "dx_um": float(np.diff(axes["x"]).mean()),
                    "beams": [{"xy_um": xy.round(3).tolist(), "t_s": tt.tolist(),
                               "off_s": off} for xy, tt, off in paths]}
    tracks[case]["_frames"] = frames
for name, (case, model, kind) in PANELS.items():
    frames = tracks[case]["_frames"]
    k = tracks[case]["frame"]
    (T0, A0), (Tk, Ak) = frames[0], frames[k]
    hits = [] if DUMPS is None else (sorted(DUMPS.glob(f"{model}_on_*__{case}*dump*/**/eval_frames_dump.npz")) or
                                     sorted(DUMPS.glob(f"{model}_on_*{case}*dump*/**/eval_frames_dump.npz")))
    if kind == "pred" and not hits:
        continue                                                       # committed; needs the evaluation's dump
    if hits:
        assert len(hits) == 1, (name, hits)
        with np.load(hits[0]) as z:
            T = np.asarray(z[f"{kind}_T"][1, 0], np.float32) * TSCALE  # dumped frames 0, k, ...: position 1 is k
            a = np.asarray(z[f"{kind}_alpha"][1, 0], np.float32)
            true0 = np.asarray(z["true_T"][0, 0], np.float32) * TSCALE
            truek = np.asarray(z["true_T"][1, 0], np.float32) * TSCALE
        assert T.shape == T0.shape, (name, T.shape, T0.shape)
        # The evaluation's truth departs from the deposited frame in a few hundred interface cells (fraction of fluid
        # 0.5-0.7, its loader's preprocessing), so the check asks for agreement on 99 % of metal cells. That still
        # separates frame k from its neighbours, which differ by more than 1 K in 20-30 % of them (r150np_136).
        for Td, Tp, Ap, i in ((true0, T0, A0, 0), (truek, Tk, Ak, k)):
            metal = Ap >= 0.5                                          # the gas carries the eval's fill value
            off = float(np.mean(np.abs(Td - Tp)[metal] > 1.0))
            assert off < 0.01, f"{name}: dump frame {i} does not match the deposited one ({off:.1%} of metal cells)"
    if kind == "true":
        T, a = Tk, Ak                                                  # the deposited frame itself
    np.savez_compressed(OUT / f"{name}.npz", T=T.astype(np.float16), alpha=a.astype(np.float16), case=case)
    print(f"{name}: {case} frame {tracks[case]['frame']}, metal max {T[a >= 0.5].max():.0f} K")
for c in tracks.values():
    c.pop("_frames")
(OUT / "tracks.json").write_text(json.dumps(tracks, indent=1))
print("wrote", OUT)
