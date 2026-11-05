"""Beam-frame fluctuation floor (Reynolds decomposition) for a straight-track case.

On a straight track at constant speed the solution is quasi-steady, so the field is
stationary in a frame travelling with the beam and every settled frame is a repeat
observation of one condition.  Fit, per voxel in the beam frame, a mean plus a
polynomial-in-time drift over the settled frames; what is left is the irreducible
per-frame fluctuation.

    T(xi, t) = Tbar(xi) + a(xi)*(t - tbar) + ... + T''(xi, t)

order 0 = beam-frame mean only, 1 = + linear drift (the published estimator),
2 = + quadratic (convergence check).  The linear term is removed because a settled
traverse still warms (substrate preheat, pool deepening); that drift is smooth and
encoded in the exposure-history channels, so a model can follow it.

Reimplementation of the manuscript estimator (LP-NO Supplementary Note
"The fluctuation floor of a per-frame formulation", sec:fluctuation-floor).

`half` is pure bookkeeping for the reported beam-box floor: the metric region M lies
inside the r_box disc, and holding edge=40 fixed, half=12 reproduces half=40 to the last
digit on all five published cases.  `edge` is NOT: it is the settled criterion, and the
window it selects is the whole estimate.  Relaxing it to 30 or 20 raises the frame count
but the mean/linear/quadratic sequence stops converging (quad/lin 0.70-0.88 against 0.995
at edge=40), which is the estimator failing its own check, so do not trade edge for frames.

Import:   from fluctuation_floor import fluctuation_floor;  r = fluctuation_floor(24163)
Run:      python fluctuation_floor.py 24163 26185 ...
          python fluctuation_floor.py --check     # reproduce the published calibrated numbers
"""
from __future__ import annotations

import io
import math
import statistics as st
import sys

import numpy as np

# Defaults are the published settings: co-moving half-window 40 cells (+-400 um at
# dx = 10 um), metric region = metal voxels within two beam radii (r0 = 50 um -> 10
# cells) of the beam, beam kept 12 cells clear of both x ends.
HALF = 40
R_BOX = 10.0
EDGE = 40           # binds jointly with HALF (keep = max(edge, half)); see note below
MIN_SETTLED = 25
ORDERS = (0, 1, 2)


def _load(pk, field_names):
    """Return (case_name, {field: ndarray (nt,nx,ny,nz)}) for one PinoSimulationData pk."""
    from aiida import orm

    node = orm.load_node(pk)
    with node.base.repository.open("payload.npz", "rb") as fh:
        buf = io.BytesIO(fh.read())
    out = {}
    with np.load(buf) as z:
        for name in field_names:
            a = np.asarray(z[name], np.float32)
            out[name] = a.reshape((a.shape[0],) + a.shape[-3:])  # drop a channel axis if present
    try:
        case = node.base.attributes.all["simulation_metadata"]["case_name"]
    except KeyError:
        case = str(pk)
    return case, out


def fluctuation_floor(pk, orders=ORDERS, half=HALF, r_box=R_BOX, edge=EDGE,
                      min_settled=MIN_SETTLED, thin="none", load_profile_name=None):
    """Beam-frame fluctuation floor for one simulation node.

    Returns a dict; ``ok`` is False when the traverse never settles for long enough
    (fewer than ``min_settled`` frames), in which case the floor fields are None.
    """
    from aiida import load_profile

    load_profile(load_profile_name)
    case, arr = _load(pk, ("Heat Flux Spatial Distribution", "Temperature", "Fraction Of Fluid"))
    T = arr["Temperature"]
    n, nx, ny, nz = T.shape

    # 1. Beam position from the deposited heat flux, never from the temperature field.
    hf = arr.pop("Heat Flux Spatial Distribution")
    ix, iy, _ = np.unravel_index(hf.reshape(n, -1).argmax(axis=1), hf.shape[1:])
    del hf
    metal = arr.pop("Fraction Of Fluid") > 0.5
    tmax = np.where(metal, T, -np.inf).reshape(n, -1).max(axis=1)

    # 2. Settled window: beam clear of both x ends and past the thermal ramp.
    keep = max(edge, half)
    ok = (ix >= keep) & (ix < nx - keep)
    plateau = tmax >= 0.98 * np.nanmax(tmax)
    ok &= np.arange(n) >= (int(np.argmax(plateau)) if plateau.any() else n)
    idx = np.where(ok)[0]
    n_ix = int(np.unique(ix[idx]).size) if idx.size else 0
    if thin.startswith("rand:") and idx.size:
        # Control for per_cell: same number of frames, but drawn at RANDOM times from the
        # same settled window, so the sub-cell beam phase is NOT held fixed.
        _, k, seed = thin.split(":")
        rng = np.random.default_rng(int(seed))
        k = min(int(k), idx.size)
        idx = np.sort(rng.choice(idx, size=k, replace=False))
    if thin.startswith("stride:") and idx.size:
        idx = idx[::int(thin.split(":")[1])]
    if thin.startswith("per_cell") and idx.size:
        _off = int(thin.split(":")[1]) if ":" in thin else 0
        # Pre-calibration export cadence: the beam advances one cell per frame, so keep
        # the FIRST settled frame at each distinct beam x cell.  Same estimator, same
        # window, same physics -- only the temporal sampling changes.
        _run, _k, _prev = 0, [], None
        for j in idx:
            _run = 0 if ix[j] != _prev else _run + 1
            _prev = ix[j]
            if _run == _off:
                _k.append(j)
        idx = np.array(_k, dtype=int)

    res = dict(pk=int(pk), case_name=case, frames=int(n), settled=int(idx.size),
               n_ix=n_ix, thin=thin,
               ok=False, drift_k=None, sigma_peak={}, peak_bound={}, box_floor={},
               box_floor_dof={}, sub_floor={}, sub_voxels=0, sub_entries=0,
               box_voxels=0, box_entries=0, orders=tuple(orders),
               half=half, r_box=r_box, edge=edge)
    if idx.size < min_settled:
        return res
    res["ok"] = True

    # 3. Co-moving sub-volume, +- half cells around the beam in x.
    sub = np.stack([T[j, ix[j] - half:ix[j] + half] for j in idx])
    msk = np.stack([metal[j, ix[j] - half:ix[j] + half] for j in idx])
    del T, metal, arr

    # Metric region M: metal voxels within two beam radii of the beam.
    gx = np.arange(2 * half)[:, None]
    gy = np.arange(ny)[None, :]
    disc = ((gx - half) ** 2 + (gy - iy[idx].mean()) ** 2) <= r_box ** 2
    ever = msk.any(axis=0)                              # every voxel that is metal at some frame
    cols = ever                                         # fit over the whole co-moving substrate
    inbox = disc[:, :, None] & ever                     # beam-box subset of those columns
    res["box_voxels"] = int(inbox.sum())
    res["sub_voxels"] = int(cols.sum())
    sub_c = sub[:, cols].astype(np.float64)             # (nf, ncols); lstsq is per column
    all_c = msk[:, cols]
    box_c = all_c & inbox[cols][None, :]
    res["box_entries"] = int(box_c.sum())
    res["sub_entries"] = int(all_c.sum())
    del sub, msk

    # Total drift of the peak metal temperature across the settled window, in K.
    t = np.arange(idx.size, dtype=np.float64)
    pk_t = tmax[idx].astype(np.float64)
    res["drift_k"] = float(np.polyfit(t, pk_t, 1)[0] * idx.size)
    t -= t.mean()

    # 4-6. Detrend per voxel, mean |residual| over M; and the same on the peak trace.
    for order in orders:
        V = np.vander(t, order + 1)
        coef, *_ = np.linalg.lstsq(V, sub_c, rcond=None)
        resid = np.abs(sub_c - V @ coef)
        f = float(resid[box_c].mean()) if box_c.any() else float("nan")
        res["box_floor"][order] = f
        res["sub_floor"][order] = float(resid[all_c].mean()) if all_c.any() else float("nan")
        # A fit with p = order+1 parameters absorbs p of the N observations, so the raw
        # residual is biased low by sqrt((N-p)/N).  Matters when comparing across N.
        _N, _p = idx.size, order + 1
        res["box_floor_dof"][order] = f * math.sqrt(_N / (_N - _p)) if _N > _p else float("nan")
        del resid, coef
        c, *_ = np.linalg.lstsq(V, pk_t, rcond=None)
        sigma = float((pk_t - V @ c).std())
        res["sigma_peak"][order] = sigma
        res["peak_bound"][order] = sigma * math.sqrt(2.0 * math.log(idx.size))
    return res


def _fmt(r):
    if not r["ok"]:
        return (f"  {r['case_name']:20s} pk {r['pk']:6d}  {r['settled']:4d}/{r['frames']:4d} settled"
                f"  -- too few settled frames")
    b, s = r["box_floor"], r["sigma_peak"]
    d = r["box_floor_dof"]
    return (f"  {r['case_name']:20s} pk {r['pk']:6d}  {r['settled']:4d}/{r['frames']:4d} settled"
            f"  ncell {r['n_ix']:3d}  drift {r['drift_k']:+8.1f} K  box "
            + " / ".join(f"{b[o]:6.2f}" for o in r["orders"]) + " K  dof "
            + " / ".join(f"{d[o]:6.2f}" for o in r["orders"]) + " K  sub "
            + " / ".join(f"{r['sub_floor'][o]:6.2f}" for o in r["orders"]) + " K  sigma "
            + " / ".join(f"{s[o]:6.1f}" for o in r["orders"]) + f" K  vox {r['box_voxels']}")


def main(pks, **kw):
    rows = [fluctuation_floor(int(p), **kw) for p in pks]
    for r in rows:
        print(_fmt(r), flush=True)
    good = [r for r in rows if r["ok"]]
    if good:
        print(f"\n=== MEDIAN over {len(good)} settled cases (order: box floor K / peak sigma K / peak bound K)")
        for o in ORDERS:
            sig = st.median(r["sigma_peak"][o] for r in good)
            nmed = st.median(r["settled"] for r in good)
            # The paper combines the MEDIAN sigma with the MEDIAN settled count, which is
            # not the median of the per-case bounds; both are printed.
            print(f"  order {o}: {st.median(r['box_floor'][o] for r in good):6.2f} "
                  f"(dof {st.median(r['box_floor_dof'][o] for r in good):6.2f}) / {sig:6.1f} / "
                  f"{st.median(r['peak_bound'][o] for r in good):6.1f}"
                  f"   [paper-style bound {sig * math.sqrt(2 * math.log(nmed)):6.1f} at N={nmed:.0f}]")
    return rows


# Published calibrated straight-track values (result_details.md, sec:fluctuation-floor):
# pk -> (settled, drift_K, box floor at order 0 / 1 / 2).
PUBLISHED = {
    24163: (67, 74.4, (8.60, 7.67, 7.63)),
    26185: (32, 7.6, (15.09, 14.73, 14.46)),
    27374: (168, 54.9, (12.29, 8.99, 8.90)),
    27539: (114, 10.3, (10.69, 8.55, 8.51)),
    27691: (54, 11.4, (6.30, 5.90, 5.89)),
}


def check(tol=0.02):
    """Assert the estimator reproduces the published numbers (needs the authors' database)."""
    got = []
    for pk, (settled, drift, box) in PUBLISHED.items():
        r = fluctuation_floor(pk)
        got.append(r)
        assert r["settled"] == settled, (pk, r["settled"], settled)
        assert abs(r["drift_k"] - drift) < 0.1, (pk, r["drift_k"], drift)
        for o, want in enumerate(box):
            assert abs(r["box_floor"][o] - want) <= tol * want, (pk, o, r["box_floor"][o], want)
        print(f"  ok  {r['case_name']:20s} {r['box_floor'][0]:.2f}/{r['box_floor'][1]:.2f}/{r['box_floor'][2]:.2f}")
    med = st.median(r["box_floor"][1] for r in got)
    assert abs(med - 8.55) <= 0.02 * 8.55, med
    print(f"  ok  median linear-detrended beam-box floor {med:.2f} K (published 8.55 K)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--check":
        check()
    else:
        kw, pks = {}, []
        for a in sys.argv[1:]:
            if a.startswith("--min="):
                kw["min_settled"] = int(a.split("=", 1)[1])
            elif a.startswith("--thin="):
                kw["thin"] = a.split("=", 1)[1]
            else:
                pks.append(a)
        main(pks, **kw)
