#!/usr/bin/env python3
"""Zero-shot pred vs GT fields, round 0 (Ti64 material-blind model on IN718).

Colour follows the job, not taste:
  * temperature fields -> SEQUENTIAL, perceptually uniform, monotonic in lightness
    (never a rainbow/jet, which invents edges that are not in the data);
  * pred - true        -> DIVERGING about a NEUTRAL midpoint, symmetric limits, so
    zero error reads as no colour and the sign is legible.
Melt boundaries are drawn as contours rather than a third fill, so the geometry
comparison does not depend on reading two colour scales at once.
"""
import io, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KELVIN = 3000.0          # recorded_kelvin_factor: ref_scales.T=3000, archived config
TL_IN718 = 1608.15       # IN718 liquidus -- the correct melt contour for this alloy
TL_TI64 = 1923.15        # what the Ti64-trained model was taught

def load(pk=87439):
    from aiida import load_profile, orm; load_profile()
    a = orm.load_node(pk).outputs.eval_artifacts
    return np.load(io.BytesIO(a.get_object_content("lazy_preview_batch0.npz", mode="rb")))


def fields(named_evals, out, title, cases=None):
    """Actual T fields, truth beside each model's PREDICTION -- no difference panels.

    Same conventions as main(): sequential inferno for temperature, one shared scale per
    ROW so the models are comparable by eye, depth-down orientation, and both melt
    boundaries drawn as contours. `named_evals` is [(label, eval_pk), ...]; all of them
    must be evals of the SAME dataset or the rows are not the same cases.
    """
    zs = [(name, load(pk)) for name, pk in named_evals]
    tT = zs[0][1]["true_T"][:, 0] * KELVIN
    n = tT.shape[0] if cases is None else len(cases)
    idx = range(tT.shape[0]) if cases is None else cases
    ymid = tT.shape[2] // 2
    ncol = 1 + len(zs)

    fig, axes = plt.subplots(n, ncol, figsize=(3.7 * ncol, 3.5 * n), squeeze=False)
    for r, i in enumerate(idx):
        t = tT[i, :, ymid, :].T[::-1]
        preds = [(nm, z["pred_T"][i, 0, :, ymid, :].T[::-1] * KELVIN) for nm, z in zs]
        vmax = max([t.max()] + [p.max() for _, p in preds])
        for c, (nm, arr) in enumerate([("ground truth", t)] + preds):
            ax = axes[r][c]
            im = ax.imshow(arr, aspect="auto", cmap="inferno", vmin=273, vmax=vmax)
            ax.contour(t, levels=[TL_IN718], colors="#00e5ff", linewidths=1.5)
            if c:                        # predicted boundary only on the model panels
                ax.contour(arr, levels=[TL_IN718], colors="#ffffff",
                           linewidths=1.5, linestyles="--")
                mae = float(np.abs(arr - t).mean())
                ax.set_title(f"{nm}   MAE {mae:.0f} K", fontsize=9)
            else:
                ax.set_title(f"case {i} — ground truth   peak {t.max():.0f} K", fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("K", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_alpha(0.25)
    fig.suptitle(title + "\ncyan = true IN718 melt boundary (1608 K), white dashed = predicted",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"  wrote {out}")

def main():
    z = load()
    pT, tT = z["pred_T"] * KELVIN, z["true_T"] * KELVIN
    n = pT.shape[0]
    # squeeze the singleton time axis: (case, 1, x, y, z) -> (case, x, y, z)
    pT, tT = pT[:, 0], tT[:, 0]
    ny, nz = pT.shape[2], pT.shape[3]
    ymid = ny // 2

    fig, axes = plt.subplots(n, 3, figsize=(15, 4.4 * n), squeeze=False)
    for i in range(n):
        # longitudinal slice through the beam centreline, oriented depth-down
        t = tT[i, :, ymid, :].T[::-1]
        p = pT[i, :, ymid, :].T[::-1]
        d = p - t
        vmax = max(t.max(), p.max())
        lim = float(np.abs(d).max())

        for j, (arr, ttl, cmap, kw) in enumerate((
            (t, "ground truth", "inferno", dict(vmin=273, vmax=vmax)),
            (p, "prediction",   "inferno", dict(vmin=273, vmax=vmax)),
            (d, "prediction − truth", "RdBu_r", dict(vmin=-lim, vmax=lim)),
        )):
            ax = axes[i][j]
            im = ax.imshow(arr, aspect="auto", cmap=cmap, **kw)
            # melt boundary of BOTH alloys: the model was taught Ti64's, the data is IN718's
            for src, ls, c in ((t, "-", "#00e5ff"), (p, "--", "#ffffff")):
                ax.contour(src, levels=[TL_IN718], colors=c, linewidths=1.6, linestyles=ls)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("K" if j < 2 else "ΔK", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            ax.set_title(f"case {i} — {ttl}", fontsize=10)
            ax.set_xlabel("x (cells, beam-centred window)", fontsize=8)
            ax.set_ylabel("z (cells, depth ↓)", fontsize=8)
            ax.tick_params(labelsize=7)
            for s in ax.spines.values():
                s.set_alpha(0.25)

    fig.suptitle("Round 0 zero-shot — Ti64-trained material-blind FNO on IN718\n"
                 "cyan = true IN718 melt boundary (1608 K), white dashed = predicted",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = str(Path(__file__).resolve().parent.parent / "figs" / "zeroshot_fields.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  wrote {out}")

    # quantitative companion: error vs depth, which the slices show only qualitatively
    fig2, ax = plt.subplots(figsize=(7.5, 4.2))
    for i in range(n):
        err = np.abs(pT[i] - tT[i]).mean(axis=(0, 1))
        ax.plot(np.arange(len(err)), err, lw=2, label=f"case {i}")
    ax.set_xlabel("z cell (0 = plate bottom)", fontsize=9)
    ax.set_ylabel("mean |prediction − truth| (K)", fontsize=9)
    ax.set_title("Zero-shot error by depth", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=9, frameon=False)
    for s in ax.spines.values(): s.set_alpha(0.25)
    out2 = out.replace("zeroshot_fields", "zeroshot_error_by_depth")
    fig2.tight_layout(); fig2.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"  wrote {out2}")

if __name__ == "__main__":
    sys.exit(main())
