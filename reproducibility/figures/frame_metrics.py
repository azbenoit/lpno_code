#!/usr/bin/env python3
"""The metrics printed under the prediction panels of Figs 7-9 (fig:rp-artifact, fig:varspeed-pred, fig:multibeam-pred), scored on the
instant shown, with the definitions of the evaluation: relative error is the substrate (z <= 0) MAE over the mean true temperature
there, BB MAE the MAE in the beam box, IoU that of the melt pool (fl > 0.5).

The beam box is the union, over the beams that are on, of the disc within 2 beam radii and the layers down to 2 sigma below the surface,
sigma being half the x extent of the grid cells within 1 beam radius. For one beam this is the evaluation's box, cell for cell (2817,
2808 and 2817 cells on the Fig 7-8 frames). On multi-beam cases the evaluation's box is built from a footprint scaled by each beam's
power fraction, which usually loses its depth bound, so Fig 9 uses this one to match Figs 7-8; tab:paths-accuracy keeps the evaluation's.

Predictions and truth come from the per-case evaluation dumps (prepare_path_frames.py; frames 0, k, 2k, ..., position 1 is the frame
shown), named as in `PANELS`. The Fig 7 RIGNO has its committed frame instead. Writes data/frame_metrics_figs7_9.json.

    python3 reproducibility/figures/frame_metrics.py <dir holding the dumps>
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "data" / "path_frames"
T_FILL, K_GATE = 3520.0, 20.0                                         # the evaluation's fill and gate sharpness
T_S, T_L = 1873.0, 1923.0                                             # Ti-6Al-4V solidus and liquidus; fl = 0.5 at 1898 K
BETA = 2.5
# panel -> case, dump holding its prediction (None: the committed frame), dump holding the truth
PANELS = {
    "rp_artifact_rigno": ("r150np_136", None, "fig7_fno"),
    "rp_artifact_fno": ("r150np_136", "fig7_fno", "fig7_fno"),
    "rp_artifact_gaot": ("r150np_136", "fig7_gaot", "fig7_gaot"),
    "varspeed_fno_pred_fast": ("vsn_066", "fig8_fast", "fig8_fast"),
    "varspeed_fno_pred_dwell": ("vsn_091", "fig8_dwell", "fig8_dwell"),
    "multibeam_pred_3beam": ("mbc40_026", "fig9_3beam", "fig9_3beam"),
    "multibeam_pred_4beam": ("mbc40_031", "fig9_4beam", "fig9_4beam"),
}
_A = np.linspace(0.0, 1.5, 15001)


def gate(a):
    return 0.5 * (np.tanh(K_GATE * (a - 0.5)) + 1.0)


_S = gate(_A) * _A


def undo_fill(T_filled, a_filled):
    """The network's T (K, valid on its metal) and alpha, from the fields the evaluation stored."""
    a = np.interp(np.asarray(a_filled, np.float64), _S, _A)
    T = np.asarray(T_filled, np.float64).copy()
    metal = a >= 0.5
    T[metal] = T_FILL + (T[metal] - T_FILL) / gate(a[metal])
    return T, a


def liquid(T):
    return 1.0 / (1.0 + np.exp(-2.0 * BETA * ((T - T_S) / (T_L - T_S) - 0.5)))


def substrate_and_box(track, shape):
    """Masks of the substrate and the beam box on one frame's grid."""
    nx, ny, nz = shape
    x = track["x0_um"] + np.arange(nx) * track["dx_um"]
    y = track["y0_um"] + np.arange(ny) * track["dx_um"]
    z = track["z0_um"] + np.arange(nz) * track["dx_um"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    sub = np.broadcast_to((z <= 1e-6)[None, None, :], shape)
    box = np.zeros(shape, bool)
    t, r0 = track["t_s"], track["r0_um"]
    for b in track["beams"]:
        xy, tt = np.asarray(b["xy_um"]), np.asarray(b["t_s"])
        if t > tt[-1] or (b["off_s"] is not None and t > b["off_s"]):
            continue
        r = np.hypot(X - np.interp(t, tt, xy[:, 0]), Y - np.interp(t, tt, xy[:, 1]))
        sigma = (X[r <= r0].max() - X[r <= r0].min()) / 2.0
        box |= (r <= 2.0 * r0)[:, :, None] & sub & (z >= -2.0 * sigma - 1e-3)[None, None, :]
    return sub, box


def score(pT, pa, tT, ta, sub, box, T_net=None):
    """Metrics of a stored prediction (T in K, stored alpha) against the evaluation's truth; `T_net` replaces the network's T in fl."""
    raw, ra = undo_fill(pT, pa)
    g = gate(ra)
    melt = gate(g * ra) * g * liquid(raw if T_net is None else T_net) > 0.5      # fl is gated twice by the evaluation's masking
    true_melt = liquid(tT) * gate(ta) > 0.5
    e = pT - tT
    mae = float(np.abs(e[sub]).mean())
    return {"rel_err_pct": 100.0 * mae / float(tT[sub].mean()), "t_mae_K": mae, "t_rmse_K": float(np.sqrt((e[sub] ** 2).mean())),
            "bb_mae_K": float(np.abs(e[box]).mean()), "melt_pool_iou": float((melt & true_melt).sum() / (melt | true_melt).sum()),
            "box_cells": int(box.sum())}


def main(dumps):
    tracks = json.loads((SRC / "tracks.json").read_text())
    rows = {}
    for panel, (case, pred, truth) in PANELS.items():
        with np.load(dumps / f"{truth}.npz") as z:
            tT, ta = z["true_T"][1, 0].astype(np.float64) * 3000.0, z["true_alpha"][1, 0].astype(np.float64)
            if pred is not None:
                pT, pa = z["pred_T"][1, 0].astype(np.float64) * 3000.0, z["pred_alpha"][1, 0].astype(np.float64)
        if pred is None:
            with np.load(SRC / f"{panel}.npz") as z:
                pT, pa = np.asarray(z["T"], np.float64), np.asarray(z["alpha"], np.float64)
        sub, box = substrate_and_box(tracks[case], pT.shape)
        rows[panel] = {"case": case, "frame": tracks[case]["frame"], "t_ms": round(tracks[case]["t_s"] * 1e3, 3),
                       **score(pT, pa, tT, ta, sub, box)}
        r = rows[panel]
        print(f"{panel:26s} frame {r['frame']:3d}: rel. err. {r['rel_err_pct']:.2f} %, BB MAE {r['bb_mae_K']:.1f} K, IoU {r['melt_pool_iou']:.3f}, "
              f"T MAE {r['t_mae_K']:.2f} K, T RMSE {r['t_rmse_K']:.2f} K, box {r['box_cells']} cells", flush=True)
    out = {"what": "Figs 7-9 panel metrics over the instant shown, written by notebooks/frame_metrics.py.", "rows": rows}
    (HERE / "data" / "frame_metrics_figs7_9.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
