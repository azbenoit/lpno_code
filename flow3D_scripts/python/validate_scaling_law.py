#!/usr/bin/env python3
"""Keyhole scaling-law consistency check against Gan et al. 2021 (Nat Commun 12:2379).

A per-case validation surrogate for materials without experimental anchors:
the case's keyhole number Ke is computed from its config and material
properties, the steady-window keyhole depth is measured from the exported NPZ,
and the normalized depth e/r0 is compared with a curve fitted to the 87
experiment points of the paper's Supplementary Data 1 (4 alloy/spot groups).
Deviation is reported as a z-score in units of the reference residual sd:
|z| <= 2 PASS, <= 3 WARN, > 3 FAIL.

Keyhole number (the form with the tightest collapse on the SD1 data):

    Ke = eta * P / ((Tv - T0) * pi * rho * cp * sqrt(alpha * V * r0**3))

with rho, cp and alpha = K/(rho*cp) evaluated at the liquidus temperature when
the material ships temperature curves (Gan's own property convention: their
Ti64 K=33.4, cp=830 are this repo's Mills curves at 1923.15 K), and the scalar
properties otherwise (a constant-property solve really used the scalar).

eta is the net absorbed fraction. The reference eta is Gan's measured,
Ke-dependent absorptance; a simulation-side equivalent is not derived here, so
pass --eta when you have one (e.g. from Heat Absorption Rate integration) or
accept the documented flat default of 0.35, which is warned about and biases
Ke low at high Ke where real keyhole absorptance reaches ~0.6-0.7.

Modes:
    check <run_dir>                          measure depth from runs/<case>/npz
    check --power-w .. --depth-um ..         explicit parameters, no NPZ
    sweep <batch_runs_dir>                   one line per case + summary
    build-reference <sd1.xlsx>               regenerate the derived reference JSON

The shipped reference (python/data/gan_scaling_reference.json) contains only
derived dimensionless pairs (Ke, e/r0); the SD1 spreadsheet itself is not
redistributed.

Scope guard -- aluminium-family materials
-----------------------------------------
The universal curve does NOT describe aluminium alloys. Gan's own Al6061 group
sits ~+2.2 sigma above the fitted curve, and a direct check on 2026-07-29
scored *their measured* 332 um depth at 520 W / 0.45 m/s as z = +3.02 (FAIL)
while our simulation of the same condition -- within +29% of that measurement,
i.e. inside its experimental band -- scored z = +5.27.

So for Al/Mg-family materials a FAIL here is not evidence of a model error:
score them against the experimental deviation (or against an experimental
anchor directly), not against the curve. The gate is meaningful for the
Fe/Ni/Ti/Co families the curve was fitted on, where a Ti64 control scores
z = -0.49.

Known power limitation (found 2026-07-30 by sweeping the original dataset)
-------------------------------------------------------------------------
The acceptance band is ABSOLUTE in e/r0 units (the reference scatter), while
e/r0 itself spans ~0.1-10 across the design space. In the shallow/conduction
regime a 100% depth error therefore scores |z| < 0.5 and passes trivially: a
sweep of the pre-campaign straight-path dataset returned 56/56 PASS with every
z negative and measured depths 50-80% BELOW the curve. Per-case verdicts are
consequently near-useless below e/r0 ~ 1; read the summary's mean z, which is
what exposes a one-sided set. A proper fix is a relative (log-space) or
heteroscedastic residual, which would require refitting the reference.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = Path(__file__).resolve().parent / "data" / "gan_scaling_reference.json"
DEFAULT_ETA = 0.35
DEFAULT_T0_K = 293.15
# steady-window depth metric: laser-on = first 85% of frames, steady = tail 30%
LASER_ON_FRACTION = 0.85
STEADY_TAIL_FRACTION = 0.30
FOF_GAS_THRESHOLD = 0.5
Z_PASS, Z_WARN = 2.0, 3.0

SOURCE_CITATION = (
    "Gan, Z., Kafka, O.L., Parab, N. et al. Universal scaling laws of keyhole "
    "stability and porosity in 3D printing of metals. Nat Commun 12, 2379 (2021). "
    "Derived from Supplementary Data 1 (87 points, 4 alloy/spot groups)."
)


# --------------------------------------------------------------------------
# Keyhole number and reference curve
# --------------------------------------------------------------------------

def keyhole_number(
    *,
    eta: float,
    power_w: float,
    speed_m_per_s: float,
    r0_m: float,
    rho_kg_m3: float,
    cp_j_kg_k: float,
    alpha_m2_s: float,
    dtv_k: float,
) -> float:
    """Gan 2021 keyhole number Ke = eta*P / ((Tv-T0)*pi*rho*cp*sqrt(alpha*V*r0^3))."""
    denominator = (
        dtv_k * math.pi * rho_kg_m3 * cp_j_kg_k
        * math.sqrt(alpha_m2_s * speed_m_per_s * r0_m**3)
    )
    return eta * power_w / denominator


def load_reference(path: Path | str = DEFAULT_REFERENCE) -> dict:
    with open(path, encoding="utf-8") as handle:
        reference = json.load(handle)
    if reference.get("schema_version") != 1:
        raise SystemExit(f"Unsupported reference schema in {path}")
    return reference


def eval_curve(reference: dict, ke: float) -> float:
    """Predicted e/r0 at Ke from the fitted polynomial in sqrt(Ke)."""
    coeffs = reference["fit"]["coeffs_sqrt_ke"]
    return float(np.polyval(coeffs, math.sqrt(ke)))


# --------------------------------------------------------------------------
# Depth metric (steady-window deepest interior gas cell)
# --------------------------------------------------------------------------

def measure_depth_npz(npz_path: Path | str, surface_z_m: float = 0.0) -> dict:
    """Steady-window keyhole depth from an exported all-steps NPZ.

    Per frame: the deepest interior gas cell (Fraction Of Fluid < 0.5) with a
    cell-center below the substrate surface; x/y boundary columns are excluded
    as wall artifacts. Laser-on is the first 85% of frames and the metric is
    the mean over the last 30% of those (the steady tail).
    """
    with np.load(npz_path) as data:
        if "Fraction Of Fluid" not in data:
            raise SystemExit(f"NPZ has no 'Fraction Of Fluid' field: {npz_path}")
        fof = data["Fraction Of Fluid"]
        z = np.asarray(data["z"], dtype=np.float64)
    if fof.ndim != 4:
        raise SystemExit(f"Expected (t, x, y, z) fluid-fraction array in {npz_path}")
    n_frames = fof.shape[0]
    n_on = max(1, int(round(LASER_ON_FRACTION * n_frames)))
    n_steady = max(1, int(round(STEADY_TAIL_FRACTION * n_on)))
    start = n_on - n_steady

    gas = np.nan_to_num(fof[start:n_on], nan=1.0) < FOF_GAS_THRESHOLD
    if gas.shape[1] > 2:
        gas[:, [0, -1], :, :] = False  # x wall columns are artifacts, not keyholes
    if gas.shape[2] > 2:
        gas[:, :, [0, -1], :] = False
    gas &= z[None, None, None, :] < surface_z_m

    depths = np.zeros(gas.shape[0])
    hit = gas.any(axis=(1, 2))  # (frames, z)
    for index in range(gas.shape[0]):
        if hit[index].any():
            depths[index] = surface_z_m - z[hit[index]].min()
    return {
        "depth_m": float(depths.mean()),
        "depth_sd_m": float(depths.std()),
        "n_frames_total": n_frames,
        "steady_frames": (start, n_on - 1),
    }


# --------------------------------------------------------------------------
# Material properties (Gan convention: values at liquidus when curves exist)
# --------------------------------------------------------------------------

_PROPERTY_CSVS = {
    "rho_kg_m3": "density_kg_m3",
    "cp_j_kg_k": "specific_heat_j_kg_k",
    "k_w_m_k": "thermal_conductivity_w_m_k",
}


def _props_from_scalars(properties: dict) -> dict:
    try:
        return {
            "rho_kg_m3": float(properties["density_kg_m3"]),
            "cp_j_kg_k": float(properties["specific_heat_j_kg_k"]),
            "k_w_m_k": float(properties["thermal_conductivity_w_m_k"]),
            "tv_k": float(properties["vaporization_temperature_k"]),
            "tl_k": float(properties["liquidus_temperature_k"]),
        }
    except KeyError as error:
        raise SystemExit(f"material properties missing {error} for the Ke computation") from error


def load_material_props(material_json: Path | str) -> dict:
    """rho/cp/K for Ke, interpolated at liquidus from sibling CSVs when present."""
    material_json = Path(material_json)
    raw = json.loads(material_json.read_text(encoding="utf-8"))
    props = _props_from_scalars(raw.get("properties", {}))
    used_csv = False
    for key, stem in _PROPERTY_CSVS.items():
        table = material_json.parent / f"{stem}.csv"
        if not table.is_file():
            continue
        with open(table, newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if row]
        try:
            pairs = sorted((float(t), float(v)) for t, v in rows[1:])
        except ValueError:
            continue  # malformed table: keep the scalar
        if pairs:
            temps, values = zip(*pairs)
            props[key] = float(np.interp(props["tl_k"], temps, values))
            used_csv = True
    props["source"] = "csv@liquidus" if used_csv else "scalar"
    return props


def _material_props_for_manifest(manifest: dict, run_dir: Path) -> dict:
    """Resolve the manifest's material file if reachable, else use its scalars."""
    material = manifest.get("material") or {}
    relative = material.get("file")
    if relative:
        candidates = [Path(relative)] if Path(relative).is_absolute() else [
            parent / relative for parent in [run_dir, *run_dir.parents][:8]
        ] + [ROOT / relative]
        for candidate in candidates:
            if candidate.is_file():
                return load_material_props(candidate)
    props = _props_from_scalars(material.get("properties", {}))
    props["source"] = "manifest-scalar"
    return props


# --------------------------------------------------------------------------
# Check / sweep
# --------------------------------------------------------------------------

def _classify(z: float) -> str:
    if abs(z) <= Z_PASS:
        return "PASS"
    if abs(z) <= Z_WARN:
        return "WARN"
    return "FAIL"


def _assemble(
    *,
    case: str,
    eta: float | None,
    power_w: float,
    speed_m_per_s: float,
    r0_m: float,
    props: dict,
    t0_k: float,
    depth_m: float,
    reference: dict,
) -> dict:
    eta_source = "explicit"
    if eta is None:
        eta, eta_source = DEFAULT_ETA, "default"
        print(
            f"WARNING: {case}: no --eta given; using flat eta={DEFAULT_ETA} "
            "(Gan's absorptance is Ke-dependent, ~0.3-0.7)",
            file=sys.stderr,
        )
    alpha = props["k_w_m_k"] / (props["rho_kg_m3"] * props["cp_j_kg_k"])
    ke = keyhole_number(
        eta=eta,
        power_w=power_w,
        speed_m_per_s=speed_m_per_s,
        r0_m=r0_m,
        rho_kg_m3=props["rho_kg_m3"],
        cp_j_kg_k=props["cp_j_kg_k"],
        alpha_m2_s=alpha,
        dtv_k=props["tv_k"] - t0_k,
    )
    measured = depth_m / r0_m
    predicted = eval_curve(reference, ke)
    z = (measured - predicted) / reference["fit"]["residual_sd"]
    status = _classify(z)
    ke_min, ke_max = reference["fit"]["ke_range"]
    notes = []
    if not ke_min <= ke <= ke_max:
        notes.append(f"Ke outside reference range [{ke_min:.2f}, {ke_max:.2f}]")
        if status == "PASS":
            status = "WARN"
    return {
        "case": case,
        "ke": ke,
        "depth_m": depth_m,
        "e_r0_measured": measured,
        "e_r0_predicted": predicted,
        "z": z,
        "status": status,
        "eta": eta,
        "eta_source": eta_source,
        "props_source": props["source"],
        "notes": notes,
    }


def check_run_dir(run_dir: Path | str, eta: float | None = None, reference: dict | None = None) -> dict:
    run_dir = Path(run_dir).resolve()  # symlinked case dirs must still find material CSVs
    reference = reference or load_reference()
    manifest = json.loads((run_dir / "case_manifest.json").read_text(encoding="utf-8"))
    laser = manifest["input"]["laser"]
    surface_z = float(
        manifest.get("derived", {}).get("substrate_bounds_m", {}).get("z", [0.0, 0.0])[1]
    )
    npz_files = sorted((run_dir / "npz").glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No NPZ export found under {run_dir}/npz")
    depth = measure_depth_npz(npz_files[0], surface_z_m=surface_z)
    result = _assemble(
        case=manifest.get("case_name", run_dir.name),
        eta=eta,
        power_w=float(laser["power_w"]),
        speed_m_per_s=float(laser["scan_speed_m_per_s"]),
        r0_m=float(laser["beam_radius_m"]),
        props=_material_props_for_manifest(manifest, run_dir),
        t0_k=float(manifest["input"].get("simulation", {}).get("initial_temperature_k", DEFAULT_T0_K)),
        depth_m=depth["depth_m"],
        reference=reference,
    )
    result["steady_frames"] = depth["steady_frames"]
    return result


def _format_line(result: dict) -> str:
    notes = f" [{'; '.join(result['notes'])}]" if result["notes"] else ""
    return (
        f"{result['case']}: Ke={result['ke']:.3f} "
        f"e/r0 meas={result['e_r0_measured']:.3f} pred={result['e_r0_predicted']:.3f} "
        f"z={result['z']:+.2f} {result['status']} "
        f"(eta={result['eta']:.3f} {result['eta_source']}, props={result['props_source']})"
        f"{notes}"
    )


def run_check(args: argparse.Namespace) -> int:
    reference = load_reference(args.reference)
    if args.run_dir is not None:
        result = check_run_dir(args.run_dir, eta=args.eta, reference=reference)
    else:
        missing = [
            name for name, value in [
                ("--power-w", args.power_w), ("--speed-m-per-s", args.speed_m_per_s),
                ("--radius-m", args.radius_m), ("--material-json", args.material_json),
                ("--depth-um", args.depth_um),
            ] if value is None
        ]
        if missing:
            raise SystemExit(f"check needs a run directory or all of: {', '.join(missing)}")
        result = _assemble(
            case="explicit",
            eta=args.eta,
            power_w=args.power_w,
            speed_m_per_s=args.speed_m_per_s,
            r0_m=args.radius_m,
            props=load_material_props(args.material_json),
            t0_k=args.t0_k,
            depth_m=args.depth_um * 1e-6,
            reference=reference,
        )
    print(_format_line(result))
    return 1 if result["status"] == "FAIL" else 0


def summarize(zs, n_skip: int) -> str:
    """Sweep summary line, including the mean z.

    Per-case verdicts are blind to systematic bias: the acceptance band is
    absolute in e/r0 units while e/r0 spans ~0.1-10, so in the shallow regime
    even a 100% error scores |z| < 0.5 and passes. Sweeping the original
    straight-path dataset on 2026-07-30 returned 56/56 PASS with every z
    negative and measured depths 50-80% below the curve. The mean z is what
    exposes that, so it is reported and flagged when the whole set leans one way.
    """
    if not zs:
        return f"summary: n={n_skip} scored=0 skip={n_skip}"
    mean_z = sum(zs) / len(zs)
    same_sign = all(z < 0 for z in zs) or all(z > 0 for z in zs)
    flag = "  <-- SYSTEMATIC OFFSET (every case leans the same way)" if (
        same_sign and abs(mean_z) > 0.4) else ""
    return (f"summary: n={len(zs) + n_skip} scored={len(zs)} skip={n_skip} "
            f"mean z={mean_z:+.2f}{flag}")


def run_sweep(args: argparse.Namespace) -> int:
    reference = load_reference(args.reference)
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    zs: list[float] = []
    for case_dir in sorted(path for path in Path(args.batch_runs_dir).iterdir() if path.is_dir()):
        if not (case_dir / "case_manifest.json").is_file() or not list((case_dir / "npz").glob("*.npz")):
            counts["SKIP"] += 1
            print(f"{case_dir.name}: SKIP (no manifest or NPZ export)")
            continue
        try:
            result = check_run_dir(case_dir, eta=args.eta, reference=reference)
        except SystemExit as error:
            counts["SKIP"] += 1
            print(f"{case_dir.name}: SKIP ({error})")
            continue
        counts[result["status"]] += 1
        if result.get("z") is not None:
            zs.append(float(result["z"]))
        print(_format_line(result))
    total = sum(counts.values())
    print(
        f"summary: n={total} pass={counts['PASS']} warn={counts['WARN']} "
        f"fail={counts['FAIL']} skip={counts['SKIP']}"
    )
    print(summarize(zs, counts["SKIP"]))
    return 1 if counts["FAIL"] else 0


# --------------------------------------------------------------------------
# Reference builder (derived data only; the SD1 xlsx is not redistributed)
# --------------------------------------------------------------------------

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _sheet_rows(blob: bytes) -> list[dict]:
    """Sheet 1 of an xlsx as {column_letter: text} rows, stdlib only."""
    archive = zipfile.ZipFile(io.BytesIO(blob))
    shared = []
    if "xl/sharedStrings.xml" in archive.namelist():
        shared = [
            "".join(node.text or "" for node in item.iter(_XLSX_NS + "t"))
            for item in ElementTree.fromstring(archive.read("xl/sharedStrings.xml")).iter(_XLSX_NS + "si")
        ]
    sheet = sorted(name for name in archive.namelist() if name.startswith("xl/worksheets/sheet"))[0]
    rows = []
    for row in ElementTree.fromstring(archive.read(sheet)).iter(_XLSX_NS + "row"):
        values = {}
        for cell in row.iter(_XLSX_NS + "c"):
            value = cell.find(_XLSX_NS + "v")
            column = "".join(ch for ch in cell.get("r") if ch.isalpha())
            values[column] = "" if value is None else (
                shared[int(value.text)] if cell.get("t") == "s" else value.text
            )
        rows.append(values)
    return rows


def _parse_sd1(xlsx_path: Path) -> list[dict]:
    """SD1 layout: A group header, B=P, C=eta, D=V, F=r0, G..N properties, O=e."""
    rows = _sheet_rows(xlsx_path.read_bytes())
    points, group = [], None
    for row in rows[1:]:  # row 0 is the column-name header
        if row.get("A"):
            group = row["A"]
        if not (row.get("B") and row.get("O")):
            continue
        value = lambda letter: float(row[letter])  # noqa: E731
        points.append({
            "group": group,
            "eta": value("C"),
            "power_w": value("B"),
            "speed_m_per_s": value("D"),
            "r0_m": value("F"),
            "rho_kg_m3": value("H"),
            "cp_j_kg_k": value("I"),
            "alpha_m2_s": value("J"),
            "dtv_k": value("K"),
            "lv_j_kg": value("L"),
            "dtl_k": value("M"),
            "e_m": value("O"),
        })
    return points


def _ke_simple(point: dict) -> float:
    return keyhole_number(
        eta=point["eta"], power_w=point["power_w"], speed_m_per_s=point["speed_m_per_s"],
        r0_m=point["r0_m"], rho_kg_m3=point["rho_kg_m3"], cp_j_kg_k=point["cp_j_kg_k"],
        alpha_m2_s=point["alpha_m2_s"], dtv_k=point["dtv_k"],
    )


def _ke_enthalpy(point: dict) -> float:
    # variant with an effective enthalpy cp*(Tl-T0) + Lv replacing cp*(Tv-T0)
    enthalpy = point["cp_j_kg_k"] * point["dtl_k"] + point["lv_j_kg"]
    return point["eta"] * point["power_w"] / (
        enthalpy * math.pi * point["rho_kg_m3"]
        * math.sqrt(point["alpha_m2_s"] * point["speed_m_per_s"] * point["r0_m"] ** 3)
    )


def _fit(ke: np.ndarray, e_star: np.ndarray) -> tuple[list[float], float]:
    """Monotone fit of e/r0 vs Ke: quadratic in sqrt(Ke), linear fallback."""
    root = np.sqrt(ke)
    for degree in (2, 1):
        coeffs = np.polyfit(root, e_star, degree)
        grid = np.linspace(root.min(), root.max(), 512)
        if np.all(np.diff(np.polyval(coeffs, grid)) > 0):
            residual_sd = float((e_star - np.polyval(coeffs, root)).std(ddof=degree + 1))
            return [float(c) for c in coeffs], residual_sd
    raise SystemExit("No monotone fit found; inspect the reference points")


def run_build_reference(args: argparse.Namespace) -> int:
    points = _parse_sd1(Path(args.xlsx))
    e_star = np.array([point["e_m"] / point["r0_m"] for point in points])
    forms = {"simple_Tv": _ke_simple, "enthalpy_TlLv": _ke_enthalpy}
    fits = {}
    for name, formula in forms.items():
        ke = np.array([formula(point) for point in points])
        coeffs, residual_sd = _fit(ke, e_star)
        fits[name] = (ke, coeffs, residual_sd)
        print(f"form {name}: residual sd of e/r0 = {residual_sd:.4f}")
    chosen = min(fits, key=lambda name: fits[name][2])
    ke, coeffs, residual_sd = fits[chosen]
    print(f"chosen form: {chosen}")

    groups = sorted({point["group"] for point in points})
    residuals = e_star - np.polyval(coeffs, np.sqrt(ke))
    per_group = {}
    for group in groups:
        mask = np.array([point["group"] == group for point in points])
        per_group[group] = {
            "n": int(mask.sum()),
            "residual_sd": float(residuals[mask].std()),
            "mean_residual": float(residuals[mask].mean()),
            "mean_z": float(residuals[mask].mean() / residual_sd),
        }
        print(
            f"  {group}: n={per_group[group]['n']} sd={per_group[group]['residual_sd']:.3f} "
            f"mean z={per_group[group]['mean_z']:+.2f}"
        )

    reference = {
        "schema_version": 1,
        "source": SOURCE_CITATION,
        "provenance": (
            "Derived dimensionless pairs (Ke, e/r0) computed from Supplementary "
            "Data 1 by validate_scaling_law.py build-reference; the spreadsheet "
            "itself is not redistributed here."
        ),
        "ke_definition": {
            "simple_Tv": "eta*P / ((Tv-T0)*pi*rho*cp*sqrt(alpha*V*r0^3))",
            "enthalpy_TlLv": "eta*P / ((cp*(Tl-T0)+Lv)*pi*rho*sqrt(alpha*V*r0^3))",
        }[chosen],
        "ke_form": chosen,
        "e_star_definition": "keyhole depth e divided by beam spot radius r0",
        "fit": {
            "form": f"poly{len(coeffs) - 1}_in_sqrt_ke",
            "coeffs_sqrt_ke": coeffs,
            "residual_sd": residual_sd,
            "ke_range": [float(ke.min()), float(ke.max())],
        },
        "per_group": per_group,
        "points": [
            {"group": point["group"], "ke": float(k), "e_over_r0": float(es)}
            for point, k, es in zip(points, ke, e_star)
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reference, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(points)} reference points to {out_path}")

    # sanity anchors from the paper's own Ti64 group
    for target_power, expected_depth_um in ((129.36, 85.3), (206.395, 226.8)):
        for point in points:
            if (
                point["group"].startswith("Ti64")
                and abs(point["power_w"] - target_power) < 0.01
                and point["speed_m_per_s"] == 0.4
            ):
                k = forms[chosen](point)
                predicted_um = np.polyval(coeffs, math.sqrt(k)) * point["r0_m"] * 1e6
                print(
                    f"sanity Ti64 {target_power:.2f} W / 0.4 m/s: Ke={k:.3f}, "
                    f"measured e={point['e_m'] * 1e6:.1f} um (expected ~{expected_depth_um}), "
                    f"curve predicts {predicted_um:.1f} um"
                )
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    check = sub.add_parser("check", help="check one case against the scaling-law curve")
    check.add_argument("run_dir", nargs="?", type=Path, help="runs/<case> directory with npz/ + case_manifest.json")
    check.add_argument("--power-w", type=float, help="laser power [W] (explicit mode)")
    check.add_argument("--speed-m-per-s", type=float, help="scan speed [m/s] (explicit mode)")
    check.add_argument("--radius-m", type=float, help="beam spot radius r0 [m] (explicit mode)")
    check.add_argument("--material-json", type=Path, help="material JSON (CSV curves read from its directory)")
    check.add_argument("--depth-um", type=float, help="measured keyhole depth [um] (explicit mode)")
    check.add_argument("--eta", type=float, help="net absorbed fraction; defaults to a flat 0.35 with a warning")
    check.add_argument("--t0-k", type=float, default=DEFAULT_T0_K, help="ambient temperature [K] (explicit mode)")
    check.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    check.set_defaults(func=run_check)

    sweep = sub.add_parser("sweep", help="check every case directory of a batch runs/ directory")
    sweep.add_argument("batch_runs_dir", type=Path)
    sweep.add_argument("--eta", type=float, help="net absorbed fraction; defaults to a flat 0.35 with a warning")
    sweep.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    sweep.set_defaults(func=run_sweep)

    build = sub.add_parser("build-reference", help="regenerate the derived reference JSON from the SD1 xlsx")
    build.add_argument("xlsx", type=Path, help="Gan 2021 Supplementary Data 1 spreadsheet (not committed)")
    build.add_argument("--out", type=Path, default=DEFAULT_REFERENCE)
    build.set_defaults(func=run_build_reference)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
