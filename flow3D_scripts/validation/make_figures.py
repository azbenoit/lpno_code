#!/usr/bin/env python3
"""Generate the calibration figures for the manuscript, from tracked sources.

Every panel is built from the anchor files in ``validation/anchors/`` and the
exported run NPZs, using the metric definitions in ``python/melt_metrics.py``.
Nothing here re-derives a metric or hard-codes a measured value: the numbers in
the figures and the numbers in the text come from the same code, so they cannot
drift apart.

    python3 validation/make_figures.py --runs runs --out validation/figures

Figures skip themselves with a message when their cases are not on disk yet,
so this is safe to run mid-campaign.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import melt_metrics as mm            # noqa: E402
import score_anchors as sa           # noqa: E402


def _find_sibling(name: str) -> Path:
    """Nearest ancestor holding a `name` sibling checkout.

    Copied from `score_spot_mesh_pair.py` rather than imported: that module has
    `sys.path` and `spot_radiographs` import side effects. Walking up rather than
    taking `ROOT.parent` because from a worktree ROOT is nested inside the repository
    and the sibling repos are three levels further up.
    """
    for base in [ROOT, *ROOT.parents]:
        if (base.parent / name).is_dir():
            return base.parent / name
    return ROOT.parent / name

import matplotlib                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402

SIM, EXP, ACC = "#4c72b0", "#c44e52", "#55a868"
BAND = mm.DEFAULT_BAND


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 220, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    })


def load_case(runs: Path, case: str):
    """Exported NPZ plus its manifest laser-off time, or (None, None)."""
    for pat in (runs / case / "npz" / f"{case}.npz",
                runs / "batches" / "*" / "runs" / case / "npz" / f"{case}.npz"):
        hits = sorted(glob.glob(str(pat)))
        if hits:
            d = Path(hits[0]).parent.parent
            off = None
            man = d / "case_manifest.json"
            if man.is_file():
                try:
                    off = json.loads(man.read_text())["derived"].get("laser_off_time_s")
                except Exception:  # noqa: BLE001
                    off = None
            return np.load(hits[0]), off
    return None, None


def anchor_points(name: str, **kw):
    return sa.normalize_points(sa.load_anchor(ROOT / "validation" / "anchors" / name), **kw)


def measured_at(points, P, V, r0_um, tol=2.0):
    for p in points:
        if abs(p["power_w"] - P) > 0.01 * max(P, 1):
            continue
        if abs(p["speed_m_s"] - V) > 5e-4:
            continue
        if p["r0_m"] is not None and abs(p["r0_m"] * 1e6 - r0_um) > tol:
            continue
        return p
    return None


def band_fill(ax, x, y, label=True):
    y = np.asarray(y, dtype=float)
    ax.fill_between(x, y * (1 - BAND), y * (1 + BAND), color=ACC, alpha=0.16, lw=0,
                    label=f"measurement $\\pm${BAND:.0%}" if label else None)


# --------------------------------------------------------------------- figures --
def fig_keyhole_ladder(runs: Path, out: Path) -> str | None:
    """Gan Ti64: measured vs simulated vapour depression against power."""
    cases = [("t2_103w", 103.198), ("t1_129w_tv3520_noplume", 129.36),
             ("t2_156w", 155.523), ("t2_206w", 206.395)]
    pts = anchor_points("gan_points.json")
    P, sim, meas = [], [], []
    for case, pw in cases:
        z, off = load_case(runs, case)
        m = measured_at(pts, pw, 0.4, 47.5)
        if z is None or m is None:
            return None
        P.append(pw)
        sim.append(mm.depression(z, laser_off_s=off).depth_um)
        meas.append(m["depth_um"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    band_fill(ax, P, meas)
    ax.plot(P, meas, "o-", color=EXP, label="measured (Gan, in-situ X-ray)")
    ax.plot(P, sim, "s--", color=SIM, label="this work (calibrated)")
    errs = [100 * (s - m) / m for s, m in zip(sim, meas)]
    L = mm.ladder_summary(errs)
    ax.set_xlabel("laser power (W)"); ax.set_ylabel("vapour-depression depth (µm)")
    ax.set_title(f"Ti-6Al-4V keyhole ladder at 0.4 m/s\n"
                 f"{L.n_within_band}/{L.n} within band, mean signed {L.mean_signed_error_pct:+.0f}%")
    ax.legend(frameon=False)
    fig.tight_layout(); f = out / "fig_keyhole_ladder.png"; fig.savefig(f); plt.close(fig)
    return str(f)


def fig_transition_ladder(runs: Path, out: Path) -> str | None:
    """Galiegue: depth AND width vs power at fixed spot. Width tracks, depth does not."""
    cases = [("spot60_p80_v020_std", 80.0), ("spot60_p90_v020_std", 90.0),
             ("spot60_p100_v020_std", 100.0), ("spot60_p110_v020_std", 110.0)]
    pts = anchor_points("galiegue2025_ti64_targets.json")
    P, sd, sw, md, mw = [], [], [], [], []
    for case, pw in cases:
        z, _ = load_case(runs, case)
        m = measured_at(pts, pw, 0.20, 60.0)
        if z is None or m is None:
            return None
        g = mm.fusion_geometry_auto(z)
        P.append(pw); sd.append(g.depth_um); sw.append(g.width_um)
        md.append(m["depth_um"]); mw.append(m["width_um"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharex=True)
    for ax, s, m, lab in ((axes[0], sd, md, "fusion depth"),
                          (axes[1], sw, mw, "fusion width")):
        band_fill(ax, P, m)
        ax.plot(P, m, "o-", color=EXP, label="measured (Galiègue)")
        ax.plot(P, s, "s--", color=SIM, label="this work")
        e = [100 * (a - b) / b for a, b in zip(s, m)]
        L = mm.ladder_summary(e)
        ax.set_xlabel("laser power (W)"); ax.set_ylabel(f"{lab} (µm)")
        ax.set_title(f"{lab}: {L.n_within_band}/{L.n} in band, "
                     f"mean {L.mean_signed_error_pct:+.0f}%")
    axes[0].legend(frameon=False)
    fig.suptitle("Transition regime, 0.20 m/s, $r_0=60$ µm — width tracks, depth does not",
                 fontsize=10)
    fig.tight_layout(); f = out / "fig_transition_ladder.png"; fig.savefig(f); plt.close(fig)
    return str(f)


def fig_aspect(runs: Path, out: Path) -> str | None:
    """Aspect ratio across the ladder plus the larger spot: the pool grows sideways."""
    entries = [("spot60_p80_v020_std", 80.0, 60.0), ("spot60_p90_v020_std", 90.0, 60.0),
               ("spot60_p100_v020_std", 100.0, 60.0), ("spot60_p110_v020_std", 110.0, 60.0),
               ("spot130_p100_v020_std", 100.0, 130.0)]
    pts = anchor_points("galiegue2025_ti64_targets.json")
    rows = []
    for case, pw, r0 in entries:
        z, _ = load_case(runs, case)
        m = measured_at(pts, pw, 0.20, r0)
        if z is None or m is None:
            continue
        g = mm.fusion_geometry_auto(z)
        if not (g.width_um and m["width_um"]):
            continue
        rows.append((pw, r0, g.aspect, m["depth_um"] / m["width_um"]))
    if len(rows) < 3:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    lad = [r for r in rows if abs(r[1] - 60) < 1]
    big = [r for r in rows if abs(r[1] - 130) < 1]
    ax.plot([r[0] for r in lad], [r[3] for r in lad], "o-", color=EXP, label="measured, $r_0$=60 µm")
    ax.plot([r[0] for r in lad], [r[2] for r in lad], "s--", color=SIM, label="this work, $r_0$=60 µm")
    for r in big:
        ax.plot(r[0], r[3], "o", mfc="none", color=EXP, ms=9, label="measured, $r_0$=130 µm")
        ax.plot(r[0], r[2], "s", mfc="none", color=SIM, ms=9, label="this work, $r_0$=130 µm")
    ax.set_xlabel("laser power (W)"); ax.set_ylabel("aspect ratio (depth / width)")
    ax.set_title("Measured aspect climbs with power; ours stays flat")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); f = out / "fig_aspect.png"; fig.savefig(f); plt.close(fig)
    return str(f)


def fig_cross_sections(runs: Path, out: Path) -> str | None:
    """What the cavity deficit looks like: transition vs keyhole, physical orientation."""
    picks = [("spot60_p100_v020_std", "transition, 100 W, $r_0$=60 µm"),
             ("t2_156w", "keyhole, 156 W, $r_0$=47.5 µm")]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.2))
    made = 0
    for row, (case, title) in enumerate(picks):
        z, _ = load_case(runs, case)
        if z is None:
            continue
        F = np.asarray(z["Fraction Of Fluid"]); M = np.asarray(z["Melt Region"])
        x = np.asarray(z["x"]) * 1e6; zc = np.asarray(z["z"]) * 1e6
        n = F.shape[0]; fr = int(0.8 * n); j = F.shape[2] // 2
        for col, (arr, name, cmap) in enumerate(((F, "gas fraction", "gray_r"),
                                                 (M, "melt region", "magma"))):
            ax = axes[row][col]
            ax.pcolormesh(x, zc, arr[fr, :, j, :].T, cmap=cmap, shading="nearest")
            ax.axhline(0, color="cyan", lw=0.9, ls=":")
            ax.set_ylim(zc.min(), zc.max())
            ax.set_title(f"{title} — {name}", fontsize=9)
            ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
        made += 1
    if made < 2:
        plt.close(fig); return None
    fig.tight_layout(); f = out / "fig_cross_sections.png"; fig.savefig(f); plt.close(fig)
    return str(f)


def fig_old_vs_new(runs: Path, out: Path) -> str | None:
    """Fusion depth at identical conditions: previous physics vs calibrated."""
    # The pre-calibration NPZs live in the sibling ML repo's data tree. A reader
    # without that checkout sets PINO_SIMULATIONS_ROOT to wherever the
    # `original_straight_path_data` archive was unpacked (see README, Reproducibility).
    orig_dir = (Path(os.environ.get("PINO_SIMULATIONS_ROOT", _find_sibling("lpno")))
                / "data" / "flow3D" / "original_straight_path_data")
    pairs = [("train_190W_0870mps", "190W_0870mps"), ("train_140W_0472mps", "140W_0472mps"),
             ("val_140W_0551mps", "140W_0551mps"), ("train_140W_0954mps", "140W_0954mps"),
             ("train_90W_0195mps", "90W_0195mps"), ("test_90W_0632mps", "90W_0632mps")]
    lab, old, new = [], [], []
    for v2case, origcase in pairs:
        hits = glob.glob(str(orig_dir / "*" / f"{origcase}_line_track_vtk.npz"))
        z2, _ = load_case(runs, v2case)
        if not hits or z2 is None:
            continue
        z1 = np.load(hits[0])
        old.append(mm.fusion_geometry_auto(z1).depth_um)
        new.append(mm.fusion_geometry_auto(z2).depth_um)
        lab.append(origcase.replace("_", " ").replace("mps", " m/s").replace("W ", " W / "))
    if len(lab) < 4:
        return None
    idx = np.arange(len(lab)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(idx - w/2, old, w, color="#999999", label="previous physics")
    ax.bar(idx + w/2, new, w, color=SIM, label="calibrated")
    for i, (a, b) in enumerate(zip(old, new)):
        ax.text(i + w/2, b, f"{100*(b-a)/a:+.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(idx); ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("fusion depth (µm)")
    ax.set_title("Identical conditions under previous and calibrated physics")
    ax.legend(frameon=False)
    fig.tight_layout(); f = out / "fig_old_vs_new.png"; fig.savefig(f); plt.close(fig)
    return str(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "validation" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    for name, fn in (("keyhole ladder", fig_keyhole_ladder),
                     ("transition ladder", fig_transition_ladder),
                     ("aspect ratio", fig_aspect),
                     ("cross-sections", fig_cross_sections),
                     ("old vs new", fig_old_vs_new)):
        try:
            got = fn(args.runs, args.out)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not block the rest
            print(f"  {name:20s} ERROR {type(exc).__name__}: {exc}")
            continue
        print(f"  {name:20s} {got if got else 'skipped (cases not on disk yet)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
