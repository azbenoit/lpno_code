#!/usr/bin/env python3
"""Acquisition for the two-alloy IN718 loop: maximin baseline and the bandit.

Two arms. They are genuinely DIFFERENT allocations -- an earlier version of this
docstring claimed the bandit "degenerates exactly to maximin" when learning rates
are equal, and a cold-start comparison falsified that: the two share 4 of 25 picks.

What actually happens with no rate signal is that the UCB term dominates and the
bandit becomes a STRATIFIED round-robin over regions (2-3 picks each of 11), while
maximin concentrates geometrically -- 8 of its 25 cold-start picks land in
`keyhole:fine`, the large, far-from-centre corner of normalized space. Worth
noting on its own: the maximin BASELINE over-concentrates too, for geometric
rather than model reasons.

So the null result to look for is not "the bandit became maximin" but "the
bandit's regional allocation stayed flat", i.e. every b_r shrank to the pooled
value and nothing was learned about where returns differ.

  maximin  : farthest-point over the candidate lattice (user-specified baseline)
  bandit   : the bandit picks the REGION by expected error reduction; maximin
             picks the POINT inside it. Allocation is error-driven, placement is
             space-filling.

The bandit is a ROTTING bandit: reward decays as an arm is pulled, because
returns diminish and some regions carry irreducible error (chaotic keyhole
fluctuation, or transient content the steady average discards). The
explore/exploit trade-off is therefore over learning RATES, not over errors --
which is the correction to the P-V experiment, where max-variance chased high
error into a region whose error was not reducible: 26 of 70 picks in high-H
bought a 0.44 K improvement there.

Guards, each for a failure that has already happened or is predicted:
  * b_r is SHRUNK toward a pooled exponent and floored. At the measured +-3.19 K
    selection noise, a raw dE/dn is mostly noise and one unlucky round could
    otherwise kill an arm permanently.
  * The probe is out-of-fold on TRAINING data, never the frozen test set.
  * Credit is kernel-weighted and heuristic: the model is global, so a region's
    error responds to data everywhere. Not a causal estimate, and not claimed as one.
  * Cold start is maximin by construction (no rates yet), so the bandit can only
    diverge once it has evidence.
"""
import csv, json, math, sys
from pathlib import Path
from _repo import MATERIALS
import numpy as np

# design box -- must match the test set's expanded box (bin/generate.py there)
#
# H_HI = 48 is CORRECT and must not be lowered to ~28. That bound was tried on
# 2026-09-10 and reverted the same day: it was argued from an anchor table computed with
# a CONSTANT K1, and King H has K1 proportional to r0^1.5, so dropping the radius term
# across a 3x r0 range moved every anchor. Recomputed with the per-case k1(mat, r0) that
# `lattice()` already uses:
#
#   Ti64 anchor train (ds86529, n=171)   H  6.9 - 20.8   (p95 18.7)
#   IN718 frozen test (ds87436, n=24)    H  6.3 - 46.8
#   experimental IN718 anchors (n=16)    H 21.6 - 46.8   <- ALL above the Ti64 ceiling
#
# The experimental conditions this study validates against sit at the TOP of the box, not
# inside the Ti64 overlap region. H_HI = 28 cut off every validation target. Verified
# against Materials variations' independent measurement of 287 solves: their King H equals
# this one at ratio 1.00 (range 1.00-1.00), and it predicts melt aspect d/w at corr +0.889
# versus +0.576 for the flat-K1 version.
#
# The r0 floor is separately justified: the anchors reach down to 24.5 um and the test set
# to 23.0 um, against a Ti64 training floor of 42.5 um. That is why R_LO is 22.0e-6.
H_LO, H_HI = 4.0, 48.0
V_LO, V_HI = 0.15, 1.25
R_LO, R_HI = 22.0e-6, 65.0e-6
P_LO, P_HI = 90.0, 350.0
H_OVERLAP_MAX = 21.7            # ds86529's Ti64 H ceiling

REGIMES = [("lof", 4.0, 8.0), ("conduction", 8.0, 14.0),
           ("transition", 14.0, 21.7), ("keyhole", 21.7, 48.0)]
R_BANDS = [("fine", 22e-6, 36e-6), ("mid", 36e-6, 50e-6), ("coarse", 50e-6, 65e-6)]
EXCLUDE_RADIUS = 0.06           # normalized; the P-V audit raised 0.03 -> 0.06 after
                                # finding picks a median 0.034 apart, i.e. re-buying
                                # adjacent lattice cells


def k1(mat, r0):
    p, o = mat["properties"], mat["optical"]
    rho, cp = p["density_kg_m3"], p["specific_heat_j_kg_k"]
    D = p["thermal_conductivity_w_m_k"] / (rho * cp)
    return (rho * (cp * (p["liquidus_temperature_k"] - 293.15) + p["latent_heat_j_kg"])
            * math.sqrt(math.pi * D * r0 ** 3) / o["normal_absorptivity"])


#: Linear energy density per spot width, P/(V*2*r0). Depth correlates 0.895 with this
#: and only 0.748 with H, so it is the right containment variable.
#: MEASURED (36 solved cases, 2026-09-05): every floor-affected case sits above 1.01e7
#: and every contained case below 1.49e7 -- the bands overlap, so this threshold is
#: deliberately set at the LOW end of the overlap. It costs some containable candidates
#: to guarantee no uncontainable ones.
#: Why it matters more here than for the test set: a plate-bounded case in a TRAINING
#: set teaches the artifact, where in a test set it can merely be excluded.
E_MAX = 1.0e7


def energy_density(c):
    return c["P"] / (c["V"] * 2.0 * c["r0"])


def lattice(mat, nh=22, nv=14, nr=9):
    """Candidate points on a regular (H, V, r0) grid, keeping only those whose
    implied power lies in the box. Power is DERIVED, never sampled: K1 ~ r0^1.5,
    so a fixed-r0 K1 would misplace the whole set."""
    out = []
    for H in np.linspace(H_LO, H_HI, nh):
        for V in np.linspace(V_LO, V_HI, nv):
            for r0 in np.linspace(R_LO, R_HI, nr):
                P = k1(mat, r0) * H * math.sqrt(V)
                if not (P_LO <= P <= P_HI):
                    continue
                c = dict(H=float(H), V=float(V), r0=float(r0), P=float(P))
                # Containment: the ds86529 domain is a 450 um plate, and melt that
                # reaches the floor is bounded by the boundary rather than by physics.
                # Enlarging the domain is NOT an option -- the steady window width is
                # derived from nx (61 = 90 - 29) and a deeper plate changes the z extent,
                # either of which breaks compatibility with base model 86651.
                if energy_density(c) > E_MAX:
                    continue
                out.append(c)
    return out


def norm(c):
    return np.array([(c["H"] - H_LO) / (H_HI - H_LO),
                     (c["V"] - V_LO) / (V_HI - V_LO),
                     (c["r0"] - R_LO) / (R_HI - R_LO)])


def region_of(c):
    """Bandit arm = physical REGIME only.

    The r0 crossing was dropped 2026-09-06, measured: round 1 spread 20 usable cases
    over 9 regions (maximin) and 7 (bandit), leaving most with 1-3 cases -- far too
    thin to fit a learning-rate exponent, which is the whole point of the arm. Regime
    alone gives 3-10 cases each, and regime is the axis that actually carried the P-V
    failure (transition starvation). Radius still varies within a regime; it is simply
    not a separate arm.
    """
    return next((n for n, lo, hi in REGIMES if lo <= c["H"] < hi), REGIMES[-1][0])


def drop_near(cands, taken, radius=EXCLUDE_RADIUS):
    """Remove candidates within `radius` of anything already owned -- the frozen
    TEST points included, so the test set is excluded by CONSTRUCTION rather than
    checked for overlap afterwards as the P-V experiment did."""
    if not taken:
        return list(cands)
    T = np.stack([norm(t) for t in taken])
    return [c for c in cands
            if np.min(np.linalg.norm(T - norm(c), axis=1)) > radius]


def maximin_pick(cands, taken, n):
    """Farthest-point: each pick maximises distance to everything already owned."""
    picks = []
    owned = [norm(t) for t in taken]
    pool = list(cands)
    for _ in range(n):
        if not pool:
            break
        if owned:
            O = np.stack(owned)
            d = [float(np.min(np.linalg.norm(O - norm(c), axis=1))) for c in pool]
        else:
            d = [-float(np.linalg.norm(norm(c) - 0.5)) for c in pool]
        i = int(np.argmax(d))
        picks.append(pool[i]); owned.append(norm(pool[i])); pool.pop(i)
    return picks


def learning_rates(history, shrink=4.0, floor=0.05):
    """Per-region learning-rate exponent b_r from E_r(n) ~ a_r (n+1)^(-b_r),
    shrunk toward the pooled exponent.

    `history[region]` is a list of (n_r, E_r) observed at successive rounds.
    Shrinkage weight `shrink` is in units of observations: a region needs about
    `shrink` rounds of its own evidence before its estimate outweighs the pool.
    The floor keeps a region alive -- without it one unlucky round zeroes an arm
    forever, which at the measured noise is a coin flip, not a decision.
    """
    fits = {}
    for r, obs in history.items():
        # entries are [round, n_r, E_r]; tolerate the older [n_r, E_r] shape
        pts = [(e[-2], e[-1]) for e in obs if e[-2] > 0 and e[-1] > 0]
        # a fit needs at least two DISTINCT n_r -- two points at the same n_r define a
        # vertical line and polyfit returns a degenerate slope with a RankWarning
        if len({n for n, _ in pts}) < 2:
            continue
        if len(pts) >= 2:
            x = np.log([n + 1 for n, _ in pts]); y = np.log([E for _, E in pts])
            b = -float(np.polyfit(x, y, 1)[0])
            fits[r] = (b, len(pts))
    if not fits:
        return {}
    pooled = float(np.mean([b for b, _ in fits.values()]))
    return {r: max(floor, (b * k + pooled * shrink) / (k + shrink))
            for r, (b, k) in fits.items()}


def region_mass(cands):
    """Each region's share of the candidate lattice.

    Regions are VERY unequal here -- 34.4 % (keyhole:fine) down to 0.7 % (lof:mid) --
    because the P<=350 W ceiling admits high H only at small r0. An unweighted
    bandit therefore misallocates badly: measured at cold start it gave
    conduction:fine 3 picks against an expected 0.3 while giving keyhole:fine 2
    against an expected 8.6. Weighting by mass fixes both that and the fact that a
    0.7 %-mass region can never accumulate enough pulls for a trustworthy b_r.
    """
    from collections import Counter
    c = Counter(region_of(x) for x in cands)
    tot = sum(c.values()) or 1
    return {r: k / tot for r, k in c.items()}


def bandit_pick(cands, taken, n, region_err, region_n, rates, c_ucb=0.5, rng=None,
                mass=None):
    """Bandit chooses the region, maximin chooses the point inside it.

    value_r = E_r * b_r / (n_r + 1)   expected reduction from one more pull
    score_r = value_r + c * sqrt(log t / n_r)      UCB exploration bonus

    With every b_r equal this ranks regions by E_r/(n_r+1) and places by
    farthest-point within the winner. That is a STRATIFIED maximin, not global
    maximin: region-first selection cannot reproduce a global farthest-point
    sequence, and measured cold-start agreement is 4 of 25. Do not describe the
    arms as nested.
    """
    t = max(1, sum(region_n.values()))
    picks = []
    owned = list(taken)
    for _ in range(n):
        pool = drop_near(cands, owned)
        if not pool:
            break
        avail = {region_of(c) for c in pool}
        if not avail:
            break
        # E_r arrives in KELVIN (measured 124-323 K on the round-1 probe) while c_ucb is
        # O(1). Left raw, the exploration term is ~0.2% of the score and the "bandit" is
        # pure greedy on error -- the exact over-exploitation this design exists to avoid.
        # Normalise E_r to its own max across regions so c_ucb is a real trade-off knob.
        _emax = max([v for v in region_err.values() if v] or [1.0])
        scored = {}
        for r in avail:
            E = region_err.get(r)
            E = (E / _emax) if E else E
            nr = region_n.get(r, 0)
            b = rates.get(r, 1.0)
            # Weight by the region's share of the space we are SCORED on: a large
            # region with the same error holds more test points, so a pull there
            # is worth more. Without this the cold start is a flat round-robin.
            w = (mass or {}).get(r, 1.0 / max(len(avail), 1))
            # Divide the WHOLE bracket by (n_r + 1), exploration included. Scaling
            # only the value term by mass makes the largest region win forever
            # (measured: 19 of 25 picks in keyhole:fine against an expected 8.6);
            # scaling nothing by mass gives a flat round-robin that over-samples a
            # 0.7 %-mass region 10x. This form is proportional apportionment: with
            # no error signal it allocates w_r * N picks to region r, matching what
            # maximin does naturally, and deviates from proportional only as E_r
            # and b_r accumulate evidence.
            scored[r] = w * ((0.0 if E is None else E * b) + c_ucb) / (nr + 1)
        best = max(scored, key=scored.get)
        sub = [c for c in pool if region_of(c) == best]
        pick = maximin_pick(sub, owned, 1)
        if not pick:
            break
        picks.append(dict(pick[0], region=best, region_score=scored[best]))
        owned.append(pick[0])
        region_n[best] = region_n.get(best, 0) + 1
    return picks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("maximin", "bandit"), required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--state", type=Path, help="JSON: taken points, region errors/counts, history")
    ap.add_argument("--out", type=Path, required=True)
    # Confine ONE round without redefining the study domain. Round 5 is confined to the
    # Ti64 general training region (2026-09-10: "you can do one more round confined
    # to the Ti 64 general training region"), i.e. H <= H_OVERLAP_MAX = 21.7, where the
    # Ti64 anchor bulk sits -- p95 21.0, only 4 of 171 above it. The study's evaluable
    # domain stays H <= 28 because the frozen test set reaches 27.5; this flag samples a
    # sub-box of it, it does not shrink it.
    ap.add_argument("--h-max", type=float, default=None,
                    help="confine this acquisition to H <= this (default: the H_HI box)")
    ap.add_argument("--material", type=Path,
                    default=MATERIALS / "inconel718" / "inconel718.json")
    a = ap.parse_args()

    mat = json.loads(a.material.read_text())
    cands = lattice(mat)
    if a.h_max is not None:
        before = len(cands)
        cands = [c for c in cands if c["H"] <= a.h_max + 1e-9]
        print(f"  confined to H <= {a.h_max}: {len(cands)} of {before} lattice points")
        if not cands:
            print("  refusing: no candidates left inside the confinement"); return 1
    st = json.loads(a.state.read_text()) if a.state and a.state.is_file() else {}
    taken = st.get("taken", [])
    pool = drop_near(cands, taken)

    if a.arm == "maximin":
        picks = maximin_pick(pool, taken, a.n)
    else:
        rates = learning_rates(st.get("history", {}))
        picks = bandit_pick(cands, taken, a.n, st.get("region_err", {}),
                            dict(st.get("region_n", {})), rates,
                            mass=region_mass(cands))

    for p in picks:
        p["region"] = p.get("region", region_of(p))
        p["regime"] = "overlap" if p["H"] <= H_OVERLAP_MAX else "extrapolation"
    a.out.write_text(json.dumps(picks, indent=2) + "\n")
    print(f"  arm={a.arm}  lattice={len(cands)}  after-exclusion={len(pool)}  picked={len(picks)}")
    from collections import Counter
    print("  by region:", dict(Counter(p["region"] for p in picks)))
    print("  by regime:", dict(Counter(p["regime"] for p in picks)))
    print(f"  H {min(p['H'] for p in picks):.1f}-{max(p['H'] for p in picks):.1f} | "
          f"P {min(p['P'] for p in picks):.0f}-{max(p['P'] for p in picks):.0f} W | "
          f"r0 {min(p['r0'] for p in picks)*1e6:.1f}-{max(p['r0'] for p in picks)*1e6:.1f} um")


if __name__ == "__main__":
    sys.exit(main())
