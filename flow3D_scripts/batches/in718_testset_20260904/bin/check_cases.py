#!/usr/bin/env python3
"""Health + anchor check for the IN718 test set, run in bulk as cases land.

Two lessons are baked in, both hit by hand on the first two cases:
  * melt must be masked on `Fraction Of Fluid > 0.5` -- masking on temperature
    alone counts hot GAS and puts "melt" above the substrate surface;
  * width must be BEAM-SECTIONED at one mid-track x plane. A melt-span union over
    all frames and x is a different quantity (span/width reaches 1.286) and read
    +60.1 % against a NIST cross-section that is really -5.8 %.
"""
import ast, csv, json, math, sys
from pathlib import Path
import numpy as np

BATCH = Path(__file__).resolve().parent.parent
from _repo import REPO, RUNS
TL = 1608.15                     # IN718 liquidus
AMB = json.loads((REPO / "validation/anchors/amb718_targets.json").read_text())
AMB = {ast.literal_eval(k): v for k, v in AMB.items() if k.startswith("(")}


def geom(npz):
    T = np.squeeze(npz["Temperature"]); F = np.squeeze(npz["Fraction Of Fluid"])
    metal = F > 0.5
    melt = (T > TL) & metal
    nf, nx, ny, nz = T.shape
    surf = int(np.where(metal[0].any(axis=(0, 1)))[0].max())
    if not melt.any():
        return dict(melted=False, peakT=float(T.max()))
    zi = np.where(melt.any(axis=(0, 1, 2)))[0]
    yi = np.where(melt.any(axis=(0, 1, 3)))[0]
    xi = np.where(melt.any(axis=(0, 2, 3)))[0]
    best = (0, 0, -1)
    for f in range(nf // 3, nf):                       # past the start transient
        for x in range(nx // 3, 2 * nx // 3):          # mid-track only
            sl = melt[f, x]
            if sl.any():
                zz = np.where(sl.any(axis=0))[0]; yy = np.where(sl.any(axis=1))[0]
                d = (surf - int(zz.min())) * 10
                if d > best[0]:
                    best = (d, (int(yy.max()) - int(yy.min()) + 1) * 10, f)
    return dict(melted=True, peakT=float(T.max()), frames=nf,
                depth=best[0], width=best[1],
                floor_margin=int(zi.min()) * 10,
                span_frac=(int(yi.max()) - int(yi.min()) + 1) / ny,
                x_margin=min(int(xi.min()), nx - 1 - int(xi.max())) * 10)


def main():
    design = {r["case_name"]: r for r in csv.DictReader(open(BATCH / "design.csv"))}
    print(f"  {'case':<12}{'regime':<15}{'H':>6}{'peakT':>7}{'depth':>7}{'width':>7}"
          f"{'floor':>7}{'xmarg':>7}{'span':>6}  anchor")
    bad = []
    for name, r in design.items():
        f = RUNS / name / "npz" / f"{name}.npz"
        if not f.is_file():
            continue
        log = (RUNS / name / "runhyd.log")
        ok = log.is_file() and "normal completion" in log.read_text(errors="ignore")
        g = geom(np.load(f))
        note = ""
        key = (float(r["P"]), round(float(r["V"]) * 1000), round(float(r["r0_um"]) * 2))
        tgt = AMB.get(key)
        if tgt and g.get("melted"):
            de = 100 * (g["depth"] - tgt["depth_um"]) / tgt["depth_um"]
            we = 100 * (g["width"] - tgt["width_um"]) / tgt["width_um"]
            # Verdict must cover BOTH quantities. An earlier version tested depth
            # only and printed "[in +-30%]" beside a width of -35.9% -- the exact
            # habit to avoid: a verdict word with an unstated scope.
            vd = "d-in" if abs(de) <= 30 else "d-OUT"
            vw = "w-in" if abs(we) <= 30 else "w-OUT"
            note = (f"NIST d{tgt['depth_um']:.0f}/w{tgt['width_um']:.0f} "
                    f"-> {de:+.1f}%/{we:+.1f}% [{vd},{vw} vs +-30%]")
            if abs(de) > 30 or abs(we) > 30:
                bad.append(name)
        if not g.get("melted"):
            note = "NO MELT"; bad.append(name)
        if not ok:
            note += " NO-NORMAL-COMPLETION"; bad.append(name)
        if g.get("melted"):
            # 50 um, not 20. in718t_010 melted 410 um into a 450 um plate -- a
            # 30 um (3-cell) margin, which a 20 um threshold let through. The
            # thermal boundary layer is thicker than three cells, and the deep
            # melt sits UNDER the beam so it is inside the 61-cell steady window
            # (unlike an x-wall artifact at cell 1, which the window crops out).
            if g["floor_margin"] <= 50: note += " FLOOR!"; bad.append(name)
            if g["x_margin"] <= 10: note += " XWALL!"; bad.append(name)
            if g["span_frac"] >= 0.8: note += " SPAN!"; bad.append(name)
            print(f"  {name:<12}{r['regime']:<15}{float(r['H']):>6.1f}{g['peakT']:>7.0f}"
                  f"{g['depth']:>7}{g['width']:>7}{g['floor_margin']:>7}{g['x_margin']:>7}"
                  f"{g['span_frac']:>6.2f}  {note}")
        else:
            print(f"  {name:<12}{r['regime']:<15}{float(r['H']):>6.1f}{g['peakT']:>7.0f}"
                  f"{'-':>7}{'-':>7}{'-':>7}{'-':>7}{'-':>6}  {note}")
    print(f"\n  flagged: {sorted(set(bad)) if bad else 'none'}")


if __name__ == "__main__":
    sys.exit(main())
