"""fig:twoalloy-pred: the joint two-alloy model on the deep-keyhole IN718 test case (P = 346.8 W, the deepest melt
of the 24 held-out IN718 simulations) and on a Ti-6Al-4V test case, truth beside prediction, in the half-block view.

The model is the alloy-label joint RIGNO (rigno_joint_r05_alloyflag_s42). Its fields are taken from evaluation
previews (lazy_preview_batch0.npz), which hold the predictions the tabled metrics were scored on; cases are selected
by laser power, never by preview index. By default the script renders from two files:
data/twoalloy_prediction_frames.npz (committed: the predicted fields and the case parameters, and panel (e), the
Ti-6Al-4V-only FNO on the same Ti-6Al-4V case, written by forward_ti64only_fno.py) and data/twoalloy_truth_frames.npz
(FLOW-3D ground truth, restricted record only, written by prepare_truth_frames.py from the deposited previews of
the joint RIGNO's evaluations 3a3492a6 and 6c883ef6).

LPNO_TWOALLOY_PREVIEWS="in718_extrap=<npz>,ti64=<npz>" reads two preview files instead and rewrites both frame files.
The colour bar stands vertically in the empty slot of the IN718 row."""
import os, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NB = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(NB)); os.chdir(NB)
import figstyle; figstyle.apply()
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv; pv.OFF_SCREEN = True

MESH_UM, MELT_ALPHA, TSCALE = 10.0, 0.5, 3000.0
CAM_OFFSET, CAM_ZOOM = (-0.06, -1.75, 0.70), 1.25  # 2026-09-14: lowered from 49 to 22 deg elevation so the keyhole profile is seen head-on
PRED = pathlib.Path("data") / "twoalloy_prediction_frames.npz"
TRUTH = pathlib.Path("data") / "twoalloy_truth_frames.npz"
KEYS = ("true_T", "pred_T", "true_a", "pred_a", "P", "V", "r", "H")

def pick(path, want_P_w):
    """The case at want_P_w from one evaluation preview file."""
    z = np.load(path)
    P = [float(z["in_P"][i].flat[0]) * 10 for i in range(z["in_P"].shape[0])]
    i = int(np.argmin([abs(p - want_P_w) for p in P]))
    assert abs(P[i] - want_P_w) < 1.0, f"{path}: no case at {want_P_w} W, have {P}"
    return dict(true_T=z["true_T"][i, 0] * TSCALE, pred_T=z["pred_T"][i, 0] * TSCALE,
                true_a=z["true_alpha"][i, 0], pred_a=z["pred_alpha"][i, 0],
                P=P[i], V=float(z["in_V_scan"][i].flat[0]) / 10,
                r=float(z["in_beam_radius"][i].flat[0]) * 100, H=float(z["in_H_norm"][i].flat[0]))

previews = os.environ.get("LPNO_TWOALLOY_PREVIEWS")
if previews:
    files = dict(kv.split("=", 1) for kv in previews.split(","))
    cases = {"in718": pick(files["in718_extrap"], 346.8), "ti64": pick(files["ti64"], 225.0)}
    with np.load(PRED) as zz:
        keep = {k: zz[k] for k in zz.files if k.startswith("ti64only_")}     # panel (e): forward_ti64only_fno.py
    np.savez_compressed(PRED, **keep, **{f"{n}_{k}": v for n, c in cases.items() for k, v in c.items()
                                         if not k.startswith("true_")})
    np.savez_compressed(TRUTH, **{f"{n}_{k}": v for n, c in cases.items() for k, v in c.items() if k.startswith("true_")})
else:
    assert TRUTH.exists(), f"{TRUTH} missing: write it with prepare_truth_frames.py (restricted record)"
    with np.load(PRED) as zp, np.load(TRUTH) as zt:
        both = {**{k: zp[k] for k in zp.files}, **{k: zt[k] for k in zt.files}}
    cases = {n: {k: (both[f"{n}_{k}"] if both[f"{n}_{k}"].ndim else float(both[f"{n}_{k}"])) for k in KEYS}
             for n in ("in718", "ti64")}
hi = max(float(c[k][c[a] >= MELT_ALPHA].max()) for c in cases.values()
         for k, a in (("true_T", "true_a"), ("pred_T", "pred_a")))
CLIM = [0.0, hi]
for n, c in cases.items():
    print(f"{n}: P={c['P']:.1f} W V={c['V']:.2f} m/s r0={c['r']:.0f} um H_norm={c['H']:.3f}  "
          f"GT metal peak {c['true_T'][c['true_a']>=MELT_ALPHA].max():.0f} K  pred {c['pred_T'][c['pred_a']>=MELT_ALPHA].max():.0f} K")
print(f"shared clim 0-{hi:.0f} K")

def half_block(T, a):
    grid = pv.ImageData(dimensions=T.shape, spacing=(MESH_UM,) * 3)
    grid.point_data["alpha"] = np.asarray(a, np.float32).ravel(order="F")
    grid.point_data["T"] = np.asarray(T, np.float32).ravel(order="F")
    thr = grid.threshold(MELT_ALPHA, scalars="alpha", all_scalars=True)
    thr = thr.clip(normal=(0, 1, 0), origin=grid.center, invert=False)
    surf = thr.extract_surface()
    pl = pv.Plotter(off_screen=True, window_size=(1100, 825))
    try:
        pl.set_background("white")
        pl.add_mesh(surf, scalars="T", cmap=figstyle.FIELD_TEMPERATURE, clim=CLIM,
                    smooth_shading=True, ambient=0.25, diffuse=0.75, show_scalar_bar=False)
        cx, cy, cz = grid.center; L = max(T.shape) * MESH_UM; dx, dy, dz = CAM_OFFSET
        pl.camera_position = [(cx + dx*L, cy + dy*L, cz + dz*L), (cx, cy, cz), (0, 0, 1)]
        pl.reset_camera(); pl.camera.zoom(CAM_ZOOM); pl.reset_camera_clipping_range()
        return pl.screenshot(return_img=True)
    finally:
        pl.close()

for n, c in cases.items():
    for tag, (Tk, ak) in (("gt", ("true_T", "true_a")), ("pred", ("pred_T", "pred_a"))):
        img = half_block(c[Tk], c[ak])
        fig = plt.figure(figsize=(5.2, 5.2 * img.shape[0] / img.shape[1])); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
        print(figstyle.save(fig, f"twoalloy3d_{n}_{tag}"))
print(figstyle.colorbar_strip("twoalloy3d_cbar", *CLIM))

# (e) the Ti-6Al-4V-only FNO on the same Ti-6Al-4V case, on the same colour scale
with np.load(PRED) as zz:
    assert abs(float(zz["ti64only_P"]) - cases["ti64"]["P"]) < 1.0 and \
        abs(float(zz["ti64only_V"]) - cases["ti64"]["V"]) < 1e-3, "not the Fig 12 Ti-6Al-4V case"
    T_only, a_only = zz["ti64only_pred_T"], zz["ti64only_pred_a"]
assert T_only[a_only >= MELT_ALPHA].max() <= hi, "Ti-6Al-4V-only peak outside the shared clim"
img = half_block(T_only, a_only)
fig = plt.figure(figsize=(5.2, 5.2 * img.shape[0] / img.shape[1])); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
print(figstyle.save(fig, "twoalloy3d_ti64_ti64only"))

# Vertical bar for the empty slot of the IN718 row: the height of a panel placed at 0.32\textwidth
# with trim=39 19 36 34 (bp of the 374.4 x 280.8 bp page), authored at the 0.5 reduction.
PANEL_FRAC, TRIM_LR, TRIM_TB = 0.32, 39 + 36, 19 + 34
h_in = (280.8 - TRIM_TB) * PANEL_FRAC * figstyle.TEXTWIDTH_PT / (374.4 - TRIM_LR) / figstyle.REDUCTION / 72.0
print(figstyle.colorbar_column("twoalloy3d_cbar_vertical", *CLIM, height_in=h_in, bottom_in=0.18,
                               bar_h_in=h_in - 0.36, frac=0.14))
