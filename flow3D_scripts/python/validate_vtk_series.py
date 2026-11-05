#!/usr/bin/env python3
"""Check exported FLOW-3D VTK/PVD fields for heating and melting."""

from __future__ import annotations

import argparse
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


FIELD_NAMES = {
    "Temperature": b"Temperature",
    "Melt Region": b"Melt%20Region",
    "Solid Fraction": b"Solid%20Fraction",
}


class ValidationError(RuntimeError):
    """The exported fields do not satisfy the requested check."""


def pvd_frames(path: Path) -> list[tuple[float, Path]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValidationError(f"Invalid PVD XML: {path}: {error}") from error
    frames: list[tuple[float, Path]] = []
    for dataset in root.findall(".//DataSet"):
        filename = dataset.attrib.get("file")
        timestep = dataset.attrib.get("timestep", "0")
        if not filename:
            continue
        try:
            time = float(timestep)
        except ValueError as error:
            raise ValidationError(f"Invalid PVD timestep {timestep!r}") from error
        frames.append((time, (path.parent / filename).resolve()))
    if not frames:
        raise ValidationError(f"No DataSet entries found in {path}")
    return frames


def vtk_float_array(path: Path, field_name: str) -> list[float]:
    marker = FIELD_NAMES[field_name]
    data = path.read_bytes()
    match = re.search(re.escape(marker) + rb"\s+1\s+([0-9]+)\s+float\s*\n", data)
    if not match:
        raise ValidationError(f"{field_name} was not found in {path}")
    count = int(match.group(1))
    offset = match.end()
    needed = count * 4
    payload = data[offset : offset + needed]
    if len(payload) != needed:
        raise ValidationError(f"{field_name} in {path} is shorter than its VTK header says")
    return [value[0] for value in struct.iter_unpack(">f", payload)]


def summarize_series(pvd: Path) -> dict[str, float | int]:
    frames = pvd_frames(pvd)
    max_temperature = float("-inf")
    max_melt = float("-inf")
    min_solid_fraction = float("inf")
    for _, frame in frames:
        if not frame.is_file():
            raise ValidationError(f"PVD references missing VTK frame: {frame}")
        temperatures = vtk_float_array(frame, "Temperature")
        melt = vtk_float_array(frame, "Melt Region")
        solid = vtk_float_array(frame, "Solid Fraction")
        max_temperature = max(max_temperature, max(temperatures))
        max_melt = max(max_melt, max(melt))
        min_solid_fraction = min(min_solid_fraction, min(solid))
    return {
        "frames": len(frames),
        "max_temperature_k": max_temperature,
        "max_melt_region": max_melt,
        "min_solid_fraction": min_solid_fraction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pvd", type=Path, help="ParaView PVD collection from export_all_timesteps_to_vtk.py")
    parser.add_argument("--initial-temperature-k", type=float, default=293.15)
    parser.add_argument("--min-temperature-rise-k", type=float, default=1.0)
    parser.add_argument("--require-melt", action="store_true", help="Require max Melt Region to be greater than zero")
    args = parser.parse_args()

    try:
        summary = summarize_series(args.pvd)
        required_temperature = args.initial_temperature_k + args.min_temperature_rise_k
        if summary["max_temperature_k"] <= required_temperature:
            raise ValidationError(
                f"max temperature {summary['max_temperature_k']:.6g} K did not exceed "
                f"{required_temperature:.6g} K"
            )
        if args.require_melt and summary["max_melt_region"] <= 0.0:
            raise ValidationError("Melt Region never became positive")
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        "VTK validation passed: "
        f"frames={summary['frames']}, "
        f"max_temperature_k={summary['max_temperature_k']:.6g}, "
        f"max_melt_region={summary['max_melt_region']:.6g}, "
        f"min_solid_fraction={summary['min_solid_fraction']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
