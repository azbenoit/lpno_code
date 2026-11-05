#!/usr/bin/env python3
"""Health check for a loop round -- same rules as the test-set checker.

Both arms bought heavily in keyhole:fine, which is where the test set produced its
one plate-bounded case (in718t_010, 410 um melt in a 450 um plate). Catching that
here matters more than it did there: a boundary-affected case in a TRAINING set
teaches the artifact instead of being merely excluded from a score.
"""
import argparse, copy, csv, json, sys, zipfile
from pathlib import Path
import numpy as np

LOOP = Path(__file__).resolve().parent.parent
from _repo import RUNS
TL = 1608.15


def geom(f, cache=None, key=None):
    """Per-case geometry, memoised on (case, npz mtime).

    Rescanning every NPZ went past a 120 s timeout at 36 cases and would only get
    worse -- each round doubles the corpus. The mtime in the key means a re-exported
    case is recomputed rather than served stale.
    """
    if cache is not None and key in cache:
        return cache[key]
    out = _geom_uncached(f)
    if cache is not None:
        cache[key] = out
    return out


def _geom_uncached(f):
    z = np.load(f)
    T = np.squeeze(z["Temperature"]); F = np.squeeze(z["Fraction Of Fluid"])
    metal = F > 0.5
    melt = (T > TL) & metal                     # molten METAL: masking on T alone counts hot gas
    if not melt.any():
        return dict(melted=False, peakT=float(T.max()))
    nf, nx, ny, nz = T.shape
    surf = int(np.where(metal[0].any(axis=(0, 1)))[0].max())
    zi = np.where(melt.any(axis=(0, 1, 2)))[0]
    yi = np.where(melt.any(axis=(0, 1, 3)))[0]
    xi = np.where(melt.any(axis=(0, 2, 3)))[0]
    return dict(melted=True, peakT=float(T.max()),
                depth=(surf - int(zi.min())) * 10, floor=int(zi.min()) * 10,
                span=(int(yi.max()) - int(yi.min()) + 1) / ny,
                xmarg=min(int(xi.min()), nx - 1 - int(xi.max())) * 10)


def boundary_reasons(g):
    """Why a solved case is truncated by the domain, one reason per flag; [] if it is not.

    These are the flags the experiment dropped cases on (rounds 3 and 5 each lost an
    x-wall-bounded point; round 1's plate-floor cases were replaced). A case with no melt
    is NOT among them: it is a lack-of-fusion sample, the regime the bandit's first region
    covers, so it is reported in the table and kept.
    """
    if not g.get("melted"):
        return []
    out = []
    if g["floor"] <= 50:
        out.append(f"FLOOR: melt reaches {g['floor']} um above the plate floor (flag <= 50 um), "
                   "so the depth is truncated by the domain")
    if g["xmarg"] <= 10:
        out.append(f"XWALL: melt within {g['xmarg']} um of an x-domain wall (flag <= 10 um), "
                   "so the width is truncated by the domain")
    if g["span"] >= 0.8:
        out.append(f"SPAN: melt spans {g['span']:.2f} of the domain width (flag >= 0.80)")
    return out


def merge_drops(old, arm, reasons):
    """`old` (a roundNN_drops.json payload) with this check's drops for `arm` set.

    The check writes its own key, `<arm>_health_check`, and replaces it on every run, so a
    rerun cannot duplicate entries and a person's own entries under other keys are kept.
    `round.py collect` reads every value of the file, list or dict, so the key name is free.
    """
    new = copy.deepcopy(old)
    new[f"{arm}_health_check"] = dict(sorted(reasons.items()))
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--write-drops", action="store_true",
                    help="record the boundary-truncated cases in roundNN_drops.json, which "
                         "`round.py collect` then marks split=excluded (autonomous mode)")
    a = ap.parse_args()
    # Read the top/repl families too, exactly as cmd_collect and tick do. Reading only
    # round0N_<arm> meant replacement cases were ingested and TRAINED ON without ever
    # being health-checked -- the check whose whole job is catching plate-floor and
    # wall-bounded solves silently skipped them.
    # Same predicate as round.py and tick.py: a family is a rendered directory (CASE_ORDER.txt).
    fams = [d for suf in ("", "top", "repl")
            if (d := LOOP / f"round{a.round:02d}_{a.arm}{suf}" / "CASE_ORDER.txt").is_file()]
    fams = [d.parent for d in fams]
    if not fams:
        print(f"  no rendered family for {a.arm} round {a.round}")
        return 1
    design = {}
    for rd in fams:
        design.update({r["case_name"]: r
                       for r in csv.DictReader(open(rd / "design.csv"))})
    cpath = fams[0] / ".geom_cache.json"
    cache = json.loads(cpath.read_text()) if cpath.is_file() else {}
    print(f"  {'case':<24}{'region':<18}{'H':>6}{'peakT':>7}{'depth':>7}{'floor':>7}{'xmarg':>7}{'span':>6}")
    bad, drops = [], {}
    for name, r in design.items():
        f = RUNS / name / "npz" / f"{name}.npz"
        if not f.is_file():
            continue
        if not zipfile.is_zipfile(f):        # still being written, or the export died
            print(f"  {name:<24}{r['region']:<18} export incomplete, not checked")
            continue
        g = geom(f, cache, f"{name}:{int(f.stat().st_mtime)}")
        if not g["melted"]:
            bad.append(name)
            print(f"  {name:<24}{r['region']:<18}{float(r['H']):>6.1f}{g['peakT']:>7.0f}"
                  f"{'-':>7}{'-':>7}{'-':>7}{'-':>6} NO MELT")
            continue
        reasons = boundary_reasons(g)
        if reasons:
            bad.append(name)
            drops[name] = "; ".join(reasons)
        note = "".join(f" {x.split(':')[0]}!" for x in reasons)
        print(f"  {name:<24}{r['region']:<18}{float(r['H']):>6.1f}{g['peakT']:>7.0f}"
              f"{g['depth']:>7}{g['floor']:>7}{g['xmarg']:>7}{g['span']:>6.2f}{note}")
    cpath.write_text(json.dumps(cache))
    print(f"\n  flagged: {sorted(set(bad)) if bad else 'none'}")
    if a.write_drops:
        dpath = LOOP / f"round{a.round:02d}_drops.json"
        old = json.loads(dpath.read_text()) if dpath.is_file() else {}
        dpath.write_text(json.dumps(merge_drops(old, a.arm, drops), indent=1) + "\n")
        print(f"  drops: {len(drops)} boundary-truncated case(s) recorded in {dpath.name} "
              f"(no-melt cases are lack of fusion and are kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
