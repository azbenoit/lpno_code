#!/usr/bin/env python3
"""Check an exported FLOW-3D NPZ series for heating and melting.

This is the NPZ counterpart of validate_vtk_series.py: it reads the single
``<case>.npz`` written by export_all_timesteps_to_npz.py and confirms the run
heated above ambient and (optionally) produced a melt region.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


class ValidationError(RuntimeError):
    """The exported fields do not satisfy the requested check."""


def summarize(npz_path: Path) -> dict[str, float | int]:
    with np.load(npz_path, allow_pickle=True) as data:
        for field in ("Temperature", "Melt Region", "Solid Fraction"):
            if field not in data:
                raise ValidationError(f"{npz_path} is missing the '{field}' field")
        temperature = data["Temperature"]
        out = {
            "frames": int(temperature.shape[0]),
            "max_temperature_k": float(np.nanmax(temperature)),
            "max_melt_region": float(np.nanmax(data["Melt Region"])),
            "min_solid_fraction": float(np.nanmin(data["Solid Fraction"])),
        }
        # Median METAL temperature of the first stored frame. This is the
        # export-level witness of the configured initial temperature: the
        # 2026-08-10 preheat sweep ran 230 cases whose frame-0 metal median was
        # 273.1 K against configured values of 293-773 K, and nothing in the
        # pipeline compared the two. The median is robust to the laser's small
        # hot spot in an early frame; z < 0 selects the metal when the axis is
        # present, else the whole domain still catches a 100+ K miss.
        t0 = np.asarray(temperature[0], dtype=float)
        z = data.get("z")
        if z is not None and np.asarray(z).ndim == 1 and t0.ndim == 3 \
                and np.asarray(z).shape[0] == t0.shape[-1]:
            metal = np.asarray(z) < 0.0
            if metal.any():
                t0 = t0[..., metal]
        out["frame0_median_metal_k"] = float(np.nanmedian(t0))
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="NPZ from export_all_timesteps_to_npz.py")
    parser.add_argument("--initial-temperature-k", type=float, default=293.15)
    parser.add_argument("--min-temperature-rise-k", type=float, default=1.0)
    parser.add_argument("--require-melt", action="store_true", help="Require max Melt Region to be greater than zero")
    parser.add_argument(
        "--check-initial", action="store_true",
        help="Fail unless the first frame's median metal temperature is within "
             "--initial-tolerance-k of --initial-temperature-k. This is the "
             "check the 230-case preheat sweep did not have: its configs asked "
             "for 293-773 K and every frame-0 median was 273.1 K.")
    parser.add_argument("--initial-tolerance-k", type=float, default=25.0,
                        help="Tolerance for --check-initial; generous because "
                             "the first stored frame is slightly after t=0")
    args = parser.parse_args()

    try:
        if not args.npz.is_file():
            raise ValidationError(f"NPZ file does not exist: {args.npz}")
        summary = summarize(args.npz)
        required = args.initial_temperature_k + args.min_temperature_rise_k
        if summary["max_temperature_k"] <= required:
            raise ValidationError(
                f"max temperature {summary['max_temperature_k']:.6g} K did not exceed {required:.6g} K"
            )
        if args.require_melt and summary["max_melt_region"] <= 0.0:
            raise ValidationError("Melt Region never became positive")
        if args.check_initial:
            got = summary["frame0_median_metal_k"]
            want = args.initial_temperature_k
            if abs(got - want) > args.initial_tolerance_k:
                raise ValidationError(
                    f"frame-0 median metal temperature {got:.1f} K does not "
                    f"match the configured initial {want:.1f} K "
                    f"(tolerance {args.initial_tolerance_k:.0f} K). The "
                    "configured value did not reach the solver -- the preheat "
                    "bug signature.")
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        "NPZ validation passed: "
        f"frames={summary['frames']}, "
        f"max_temperature_k={summary['max_temperature_k']:.6g}, "
        f"max_melt_region={summary['max_melt_region']:.6g}, "
        f"min_solid_fraction={summary['min_solid_fraction']:.6g}, "
        f"frame0_median_metal_k={summary['frame0_median_metal_k']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
