"""Per-case material conditioning scalars for the ck300 sweeps.

Emits the material-derived quantities a neural-operator surrogate needs but that
the AiiDA-stored manifest cannot supply. The manifest carries the 15 material
scalars and ``optical.normal_absorptivity``, but for the property CURVES it stores
only row COUNTS (``table_rows``), so k(T) at the liquidus is unreachable from a
stored simulation and must be read from the batch's own material CSVs.

Six scalars per case:

``k_liquidus_w_m_k``
    k(T) interpolated at this material's liquidus. Measured on a
    gradient-boosting proxy for log fusion depth, adding this ONE channel to the
    six shipped ones moves CV R^2 from 0.356 to 0.669, while adding all six
    remaining hidden scalars moves it only to 0.385.

``h_king``
    The generator's own normalized enthalpy, ``H = P / (K1 * sqrt(V))`` with
    ``K1 = rho (c dTm + L) sqrt(pi D a^3) / A``. Re-derived here with the SAME
    closed form as ``runs/batches/*/bin/generate.py:k1`` so there is one
    definition of H in the project, not a competing one.

    This is NOT what the training pipeline's ``H_norm`` channel contains.
    ``custom_utils/utils.py:h_norm_raw`` computes ``P / sqrt(|V| r0^3)`` and lets
    a split-max denominator absorb the material constants -- exact only while a
    split shares one material, as its own docstring says. On a 300-material split
    with a fixed 50 um beam that channel is exactly ``P/sqrt(V)`` rescaled
    (measured r = +1.000) and carries no material information at all.

``x_keyhole``
    ``H / (Tv/Tl)``, the vaporisation-corrected group the sweep's depth cap was
    fitted on. H normalises on MELTING enthalpy while keyhole depth is set by
    vaporisation, so dividing by Tv/Tl corrects for how readily the material
    boils.

    ``h_king`` and ``x_keyhole`` are NOT redundant despite correlating at
    r = +0.904. Pooled over all cases, adding H on top of X buys +0.003 -- but
    that is a pooling artifact of two regimes preferring opposite groups. Split
    by measured regime, on top of the shipped six: in CONDUCTION H adds +0.067
    and X adds +0.004; in KEYHOLE X adds +0.203 and H adds +0.048.

    Feeding the raw axes ``H`` and ``tv_ratio`` instead of the quotient is NOT
    equivalent -- keyhole proxy R^2 is 0.469 that way against 0.601 for H + X.
    The composite has to be handed over explicitly.

``tv_ratio``
    Tv/Tl. Kept as its own channel so H and X stay algebraically connected for
    the network (X = H / tv_ratio) rather than only through a ratio it must infer.

``theta_v``
    Quasi-steady Eagar-Tsai peak surface temperature (``eagar_tsai_peak_k``,
    sigma0 = r0/2) divided by Tv.

``theta_m``
    The same peak divided by Tl. Both are conditioning FEATURES, not
    predictions of anything, and ``theta_m / theta_v == tv_ratio`` by construction.

Usage::

    python python/material_conditioning.py runs/batches/<batch>/ -o <out.json>
    python python/material_conditioning.py --self-check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

#: Ambient temperature the generator measures melting enthalpy from, in kelvin.
#: Matches ``bin/generate.py``; melting enthalpy is measured FROM here, which is
#: why H is not invariant to a preheat.
T0_K = 293.15

#: Beam radius for the ck300 sweeps, in metres. Constant across all 300 cases and
#: absent from ``design.csv``.
# ponytail: a literal because this sweep fixes it; read it from the case config
# the day a sweep varies r0, which is the change that makes H_norm's r0^3 term matter.
CK300_BEAM_RADIUS_M = 5e-5


def _read_curve(path: Path) -> list[tuple[float, float]]:
    """Read a two-column ``T,value`` property table, sorted by temperature."""
    rows: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except ValueError:
                continue  # header line
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    return sorted(rows)


def interp_curve(rows: list[tuple[float, float]], t_k: float) -> float:
    """Linear interpolation with FLAT extrapolation past either end.

    Flat rather than linear because the solver itself clamps a property table
    outside its range, so extending the trend would feed the network a value the
    solve never used. Every ck300 k(T) table ends below its material's
    vaporisation temperature, so the hot end is always clamped in practice.
    """
    if t_k <= rows[0][0]:
        return rows[0][1]
    if t_k >= rows[-1][0]:
        return rows[-1][1]
    for (t_lo, v_lo), (t_hi, v_hi) in zip(rows, rows[1:]):
        if t_lo <= t_k <= t_hi:
            if t_hi == t_lo:
                return v_hi
            return v_lo + (v_hi - v_lo) * (t_k - t_lo) / (t_hi - t_lo)
    raise AssertionError("unreachable: bracket search failed on a sorted table")


def k1_king(absorptivity: float, rho: float, c_p: float, d_tm: float,
            latent: float, k_th: float, beam_radius: float) -> float:
    """King's H denominator: ``P = H * K1 * sqrt(V)``.

    Same closed form as the batch generator's ``k1``; kept identical so that H
    here and H in ``design.csv`` are the same quantity rather than two conventions.
    """
    diffusivity = k_th / (rho * c_p)
    return (rho * (c_p * d_tm + latent)
            * math.sqrt(math.pi * diffusivity * beam_radius ** 3) / absorptivity)


def eagar_tsai_peak_k(absorptivity: float, power_w: float, rho: float, c_p: float,
                      latent: float, t_l: float, k_th: float, speed_m_s: float,
                      sigma0_m: float) -> float:
    """Quasi-steady Eagar-Tsai peak surface temperature, in kelvin.

    Moving Gaussian surface source on a semi-infinite body with an insulated free
    surface (image source, hence the 1/2pi prefactor), integrated over tau with
    xi = x - V tau::

        dT(xi) = (A P / rho_c) * INT_0^inf dtau
                 exp[ -(xi + V tau)^2 / (2 (2 alpha tau + sigma0^2)) ]
                 / ( 2 pi sqrt(pi alpha tau) (2 alpha tau + sigma0^2) )

    ``sigma0 = r0/2``, NOT r0: FLOW-3D's ``rflbm`` is the 1/e^2 radius, so
    ``I ~ exp(-2 r^2 / a^2) = exp(-r^2 / 2 sigma0^2)`` gives sigma0 = a/2. Using
    r0 directly doubles the spot and manufactures a clean-looking onset threshold
    at theta_v ~ 1 that is purely an artifact of the error.

    Integrated on a log-spaced trapezoid rather than adaptive quadrature: the
    integrand has an integrable 1/sqrt(tau) singularity at the origin that makes
    `scipy.integrate.quad` report roundoff non-convergence on this parameter
    range, and a fixed log grid is both stable and dependency-free.

    Valid only where the semi-infinite assumption holds. It fails in roughly a
    third of the ck300 sweep (the diffusion length exceeds the 300 um plate) and
    it runs 2-3x above the simulated peak in keyhole cases, because the solver has
    multiple reflections and angle-dependent Fresnel on while evaporative cooling
    caps the real peak. It is used here as a monotone conditioning FEATURE, never
    as a physical prediction.
    """
    alpha = k_th / (rho * c_p)
    # Lumped volumetric heat capacity including the latent load, so the ratio
    # tracks the enthalpy the material actually needs rather than c_p alone.
    d_tm = max(t_l - T0_K, 1.0)
    rho_c = rho * (c_p * d_tm + latent) / d_tm

    tau = _logspace(1.0e-9, 5.0e-2, 400)
    width = [2.0 * alpha * t + sigma0_m ** 2 for t in tau]
    kernel = [1.0 / (2.0 * math.pi * math.sqrt(math.pi * alpha * t) * w)
              for t, w in zip(tau, width)]

    best = 0.0
    for offset in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        xi = -offset * sigma0_m
        vals = [k * math.exp(-((xi + speed_m_s * t) ** 2) / (2.0 * w))
                for k, t, w in zip(kernel, tau, width)]
        best = max(best, absorptivity * power_w / rho_c * _trapz(vals, tau))
    return T0_K + best


def _logspace(lo: float, hi: float, n: int) -> list[float]:
    step = (math.log(hi) - math.log(lo)) / (n - 1)
    return [math.exp(math.log(lo) + i * step) for i in range(n)]


def _trapz(y: list[float], x: list[float]) -> float:
    return sum(0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]) for i in range(len(x) - 1))


def conditioning_for_case(material_json: Path, power_w: float, speed_m_s: float,
                          beam_radius_m: float) -> dict[str, float]:
    """The six conditioning scalars for one case."""
    mat = json.loads(material_json.read_text(encoding="utf-8"))
    props = mat["properties"]
    t_l = float(props["liquidus_temperature_k"])
    t_v = float(props["vaporization_temperature_k"])
    rho = float(props["density_kg_m3"])
    c_p = float(props["specific_heat_j_kg_k"])
    latent = float(props["latent_heat_j_kg"])
    k_scalar = float(props["thermal_conductivity_w_m_k"])
    absorptivity = float(mat["optical"]["normal_absorptivity"])

    curve_path = material_json.parent / "thermal_conductivity_w_m_k.csv"
    k_liq = interp_curve(_read_curve(curve_path), t_l) if curve_path.is_file() else k_scalar

    k1 = k1_king(absorptivity, rho, c_p, t_l - T0_K, latent, k_scalar, beam_radius_m)
    h_king = power_w / (k1 * math.sqrt(speed_m_s))
    tv_ratio = t_v / t_l
    t_et = eagar_tsai_peak_k(absorptivity, power_w, rho, c_p, latent, t_l,
                             k_scalar, speed_m_s, beam_radius_m / 2.0)
    return {
        "k_liquidus_w_m_k": k_liq,
        "h_king": h_king,
        "x_keyhole": h_king / tv_ratio,
        "tv_ratio": tv_ratio,
        "theta_v": t_et / t_v,
        "theta_m": t_et / t_l,
    }


def build(batch_dir: Path,
          beam_radius_m: float = CK300_BEAM_RADIUS_M) -> dict[str, dict[str, float]]:
    """Conditioning scalars for every case in a batch's ``design.csv``."""
    out: dict[str, dict[str, float]] = {}
    with (batch_dir / "design.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            case = row["case"]
            material_json = batch_dir / "materials" / case / f"{case}.json"
            if not material_json.is_file():
                raise FileNotFoundError(f"{case}: missing {material_json}")
            out[case] = conditioning_for_case(
                material_json,
                power_w=float(row["P_w"]),
                speed_m_s=float(row["V_mps"]),
                beam_radius_m=beam_radius_m,
            )
    return out


def self_check() -> None:
    """Assertions that fail if the maths or the table reading breaks."""
    rows = [(300.0, 10.0), (1000.0, 20.0), (2000.0, 25.0)]
    assert interp_curve(rows, 300.0) == 10.0
    assert interp_curve(rows, 650.0) == 15.0, interp_curve(rows, 650.0)
    assert interp_curve(rows, 100.0) == 10.0, "below-range must clamp flat"
    assert interp_curve(rows, 9.0e9) == 25.0, "above-range must clamp flat"

    # K1 is linear in 1/A, so halving absorptivity must exactly double it.
    a = k1_king(0.2, 4000.0, 600.0, 1500.0, 3.0e5, 20.0, 5e-5)
    b = k1_king(0.4, 4000.0, 600.0, 1500.0, 3.0e5, 20.0, 5e-5)
    assert math.isclose(a, 2.0 * b, rel_tol=1e-12), (a, b)

    batch = (Path(__file__).resolve().parents[1] / "runs" / "batches"
             / "material_ck300_anchored_v3_20260814")
    if not (batch / "design.csv").is_file():
        print("self-check: batch dir absent, formula checks only -- OK")
        return

    table = build(batch)
    assert len(table) == 300, len(table)

    with (batch / "design.csv").open(newline="", encoding="utf-8") as fh:
        design = {r["case"]: r for r in csv.DictReader(fh)}

    # H must reproduce design.csv: that is the point of reusing k1's closed form.
    worst_h = max(abs(table[c]["h_king"] - float(design[c]["H"])) / float(design[c]["H"])
                  for c in table)
    assert worst_h < 1.0e-3, f"H disagrees with design.csv by {worst_h:.2%}"

    worst_x = 0.0
    for case, row in design.items():
        want = float(row["H"]) / float(row["tv_ratio"])
        worst_x = max(worst_x, abs(table[case]["x_keyhole"] - want) / want)
    assert worst_x < 2.0e-3, f"X disagrees with design.csv by {worst_x:.2%}"

    k_liq = [t["k_liquidus_w_m_k"] for t in table.values()]
    assert min(k_liq) > 0.0, "k(T_l) must be positive"
    spread = max(k_liq) / min(k_liq)
    assert spread > 5.0, f"k(T_l) spread {spread:.1f}x is too flat to be a real axis"

    # theta_m / theta_v is Tv/Tl exactly, by construction. If that identity ever
    # breaks, the two ratios were built from different peak temperatures.
    worst_id = max(abs(t["theta_m"] / t["theta_v"] - t["tv_ratio"]) / t["tv_ratio"]
                   for t in table.values())
    assert worst_id < 1.0e-9, f"theta_m/theta_v != tv_ratio by {worst_id:.2e}"

    theta_v = [t["theta_v"] for t in table.values()]
    assert min(theta_v) > 0.0
    # Every channel must actually vary, or it is a constant plane wasting capacity.
    assert max(theta_v) / min(theta_v) > 3.0, "theta_v is too flat to condition on"
    print(f"self-check OK: 300 cases, H max rel err {worst_h:.2e}, "
          f"X max rel err {worst_x:.2e}, k(T_l) spread {spread:.1f}x, "
          f"theta_v {min(theta_v):.2f}..{max(theta_v):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-case material conditioning scalars for the ck300 sweeps.")
    ap.add_argument("batch_dir", nargs="?", type=Path, help="runs/batches/<batch>/")
    ap.add_argument("-o", "--output", type=Path, help="destination JSON")
    ap.add_argument("--self-check", action="store_true", help="run assertions and exit")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return
    if args.batch_dir is None:
        ap.error("batch_dir is required unless --self-check is given")

    table = build(args.batch_dir)
    text = json.dumps(table, indent=1, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {len(table)} cases to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
