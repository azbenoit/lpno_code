#!/usr/bin/env python3
"""Compare two legacy FLOW-3D VTK export directories by time suffix and arrays."""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from pathlib import Path


DEFAULT_ARRAYS = [
    "Temperature",
    "Melt Region",
    "Solid Fraction",
    "Pressure",
    "Heat Absorption Rate",
    "Heat Flux Spatial Distribution",
    "X-velocity",
    "Y-velocity",
    "Z-velocity",
]


class CompareError(RuntimeError):
    """The two VTK directories are not functionally identical."""


def suffix_map(directory: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in directory.glob("*.vtk"):
        match = re.search(r"_(\d+)\.vtk$", path.name)
        if not match:
            continue
        suffix = int(match.group(1))
        if suffix in mapping:
            raise CompareError(f"Duplicate time suffix {suffix} in {directory}")
        mapping[suffix] = path
    if not mapping:
        raise CompareError(f"No legacy VTK frames found in {directory}")
    return mapping


def escaped(name: str) -> bytes:
    return name.replace(" ", "%20").encode("ascii")


def vtk_count(data: bytes, keyword: bytes) -> int:
    match = re.search(keyword + rb"\s+([0-9]+)", data)
    if not match:
        raise CompareError(f"Could not find {keyword.decode('ascii')} count")
    return int(match.group(1))


def vtk_float_array(data: bytes, field_name: str, cell_count: int) -> tuple[float, float, float]:
    marker = escaped(field_name)
    scalar = re.search(re.escape(marker) + rb"\s+1\s+([0-9]+)\s+float\s*\n", data)
    if scalar:
        count = int(scalar.group(1))
        components = 1
        offset = scalar.end()
    else:
        vector = re.search(rb"VECTORS\s+" + re.escape(marker) + rb"\s+float\s*\n", data)
        if not vector:
            raise CompareError(f"{field_name} was not found")
        count = cell_count
        components = 3
        offset = vector.end()
    needed = count * components * 4
    payload = data[offset : offset + needed]
    if len(payload) != needed:
        raise CompareError(f"{field_name} payload is shorter than its VTK header says")
    values = [item[0] for item in struct.iter_unpack(">f", payload)]
    return min(values), max(values), sum(values) / len(values)


def compare_stats(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    *,
    atol: float,
    rtol: float,
    stats: tuple[int, ...],
) -> float:
    max_error = 0.0
    labels = ("min", "max", "mean")
    for index in stats:
        av = a[index]
        bv = b[index]
        error = abs(av - bv)
        allowed = atol + rtol * max(abs(av), abs(bv))
        if error > allowed:
            raise CompareError(
                f"array {labels[index]} differs: left={a}, right={b}, error={error:g}, allowed={allowed:g}"
            )
        max_error = max(max_error, error)
    return max_error


def frame_pairs(reference: dict[int, Path], candidate: dict[int, Path], suffix_tolerance: int) -> list[tuple[int, int]]:
    if suffix_tolerance == 0:
        if set(reference) != set(candidate):
            missing = sorted(set(reference) - set(candidate))
            extra = sorted(set(candidate) - set(reference))
            raise CompareError(f"Frame suffix mismatch. Missing={missing[:8]} Extra={extra[:8]}")
        return [(suffix, suffix) for suffix in sorted(reference)]

    unused = set(candidate)
    pairs: list[tuple[int, int]] = []
    for suffix in sorted(reference):
        nearest = min(unused, key=lambda item: (abs(item - suffix), item), default=None)
        if nearest is None or abs(nearest - suffix) > suffix_tolerance:
            raise CompareError(f"No candidate frame within {suffix_tolerance} us of reference suffix {suffix}")
        pairs.append((suffix, nearest))
        unused.remove(nearest)
    if unused:
        raise CompareError(f"Unmatched candidate frames remain: {sorted(unused)[:8]}")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--array", action="append", dest="arrays", help="Array to compare. May be repeated.")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument(
        "--suffix-tolerance-us",
        type=int,
        default=0,
        help="Allow candidate frame suffixes to differ from reference suffixes by this many microseconds.",
    )
    parser.add_argument(
        "--stat",
        action="append",
        choices=["min", "max", "mean"],
        help="Summary statistic to compare. May be repeated. Defaults to min, max, and mean.",
    )
    args = parser.parse_args()

    arrays = args.arrays or DEFAULT_ARRAYS
    stat_indices = {"min": 0, "max": 1, "mean": 2}
    stats = tuple(stat_indices[name] for name in (args.stat or ["min", "max", "mean"]))
    try:
        reference = suffix_map(args.reference_dir)
        candidate = suffix_map(args.candidate_dir)
        pairs = frame_pairs(reference, candidate, args.suffix_tolerance_us)
        max_error = 0.0
        max_suffix_delta = 0
        for reference_suffix, candidate_suffix in pairs:
            max_suffix_delta = max(max_suffix_delta, abs(reference_suffix - candidate_suffix))
            left = reference[reference_suffix].read_bytes()
            right = candidate[candidate_suffix].read_bytes()
            left_points = vtk_count(left, b"POINTS")
            right_points = vtk_count(right, b"POINTS")
            left_cells = vtk_count(left, b"CELL_DATA")
            right_cells = vtk_count(right, b"CELL_DATA")
            if left_points != right_points or left_cells != right_cells:
                raise CompareError(
                    f"Frame {reference_suffix}/{candidate_suffix}: mesh size differs: "
                    f"points {left_points} vs {right_points}, cells {left_cells} vs {right_cells}"
                )
            for array in arrays:
                left_stats = vtk_float_array(left, array, left_cells)
                right_stats = vtk_float_array(right, array, right_cells)
                if any(math.isnan(value) for value in (*left_stats, *right_stats)):
                    raise CompareError(f"Frame {reference_suffix}/{candidate_suffix}: {array} contains NaN in summary")
                max_error = max(
                    max_error,
                    compare_stats(left_stats, right_stats, atol=args.atol, rtol=args.rtol, stats=stats),
                )
    except CompareError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        "Legacy VTK directory comparison passed: "
        f"frames={len(pairs)}, arrays={len(arrays)}, stats={','.join(args.stat or ['min', 'max', 'mean'])}, "
        f"max_suffix_delta_us={max_suffix_delta}, max_summary_error={max_error:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
