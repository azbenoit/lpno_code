#!/usr/bin/env python3
"""Compare a FLOW-3D NPZ export against a reference NPZ (e.g. one produced by the
aiida-pino openFoam datasource) and confirm they are almost identical.

The two files may use different layouts: this script reduces every shared field
to per-series global min/max/mean, which is orientation- and frame-count
independent, then checks those summaries agree within tolerance.  Per-field
numbers are printed so a human can see how close the runs are.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# Coordinate axes, provenance, and mesh-internal arrays are not physics fields.
NON_FIELD_KEYS = {
    "x", "y", "z", "time_s", "power_w", "scan_speed_m_per_s", "case_name",
    "P", "V_scan", "vtkGhostType", "Cell Type", "Component Number",
}


class CompareError(RuntimeError):
    """The two NPZ files are not almost identical."""


def field_stats(array: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(array, dtype=np.float64)
    return float(np.nanmin(values)), float(np.nanmax(values)), float(np.nanmean(values))


def frame_times(data) -> np.ndarray:
    """Per-frame scalar times, whether time_s is 1D or a broadcast grid."""
    times = np.asarray(data["time_s"], dtype=np.float64)
    return times.reshape(times.shape[0], -1)[:, 0]


def grid_shape(data) -> dict[str, int]:
    """Per-axis cell count, whether the axis is 1D or a broadcast grid.

    Cell count, not extent: the FLSGRF point axes sit half a cell off the VTK
    cell-centre axes (potential_bugs.md P13), so coordinate extents differ ~µm even on the same
    grid; the cell count does not.
    """
    return {
        axis: int(np.unique(np.asarray(data[axis])).size)
        for axis in ("x", "y", "z")
        if axis in data.files
    }


def match_frames(cand_times: np.ndarray, ref_times: np.ndarray) -> np.ndarray:
    """Candidate frame index nearest each reference time, so the two series are
    aggregated over the same sampled times even when their step counts differ."""
    return np.array([int(np.argmin(np.abs(cand_times - t))) for t in ref_times])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="NPZ from export_all_timesteps_to_npz.py")
    parser.add_argument("reference", type=Path, help="Reference NPZ (openFoam datasource output)")
    parser.add_argument("--field", action="append", dest="fields", help="Field to compare. May be repeated; default is every shared field.")
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol-mean", type=float, default=0.05, help="Mean tolerance as a fraction of the field dynamic range (default 0.05)")
    parser.add_argument(
        "--rtol-extrema",
        type=float,
        default=0.25,
        help=(
            "min/max tolerance as a fraction of the field dynamic range (default 0.25). "
            "Instantaneous extrema of transient keyhole fields legitimately differ run-to-run."
        ),
    )
    args = parser.parse_args()

    try:
        if not args.candidate.is_file():
            raise CompareError(f"Candidate NPZ does not exist: {args.candidate}")
        if not args.reference.is_file():
            raise CompareError(f"Reference NPZ does not exist: {args.reference}")
        cand = np.load(args.candidate, allow_pickle=True)
        ref = np.load(args.reference, allow_pickle=True)

        # Rank, not name, decides what is comparable. Every per-frame field is
        # (frames, nx, ny, nz); a 0-d entry is scalar provenance -- case_name,
        # power_w, scan_speed_m_per_s, frame_stride -- and indexing it by frame
        # raises IndexError. NON_FIELD_KEYS listed three of those four and missed
        # `frame_stride`, which the exporter writes on EVERY export
        # (export_all_timesteps_to_npz.py:298), so comparing two real exports
        # crashed at :128. Naming keys is what drifted; the rank test cannot.
        #
        # It only ever bit the SUCCESSFUL path: the loop exits at the first field
        # outside tolerance, and `frame_stride` sorts after the capitalised field
        # names, so a comparison that was going to fail bailed before reaching it.
        # A rerun-vs-rerun determinism check -- the one thing the tool exists for
        # (potential_bugs.md P12) -- is exactly the case that got far enough.
        candidates = (
            (set(cand.files) & set(ref.files)) - NON_FIELD_KEYS
            if not args.fields
            else set(args.fields)
        )
        # Missing-field check FIRST. `cand[k]` on an absent name raises numpy's
        # KeyError (a traceback, exit 1), so filtering by ndim before this check
        # made `--field Nope` crash past the guard that was written for it.
        missing = sorted(f for f in candidates if f not in cand.files or f not in ref.files)
        if missing:
            raise CompareError(f"Requested fields missing from one file: {missing}")
        shared = sorted(k for k in candidates if np.asarray(cand[k]).ndim > 0)
        if not shared:
            raise CompareError("No shared physics fields to compare")

        # Global min/max/mean is only meaningful when both files cover the same
        # grid.  A gas-gap-stripped reference (fewer z-cells) would otherwise be
        # reduced over a different cell population and silently "agree".
        cgrid, rgrid = grid_shape(cand), grid_shape(ref)
        mismatched = {
            axis: (cgrid[axis], rgrid[axis])
            for axis in cgrid.keys() & rgrid.keys()
            if cgrid[axis] != rgrid[axis]
        }
        if mismatched:
            detail = ", ".join(
                f"{axis}: candidate={c} reference={r}" for axis, (c, r) in sorted(mismatched.items())
            )
            raise CompareError(
                f"candidate and reference cover different grids (per-axis cell counts differ: {detail}). "
                "Global-stat comparison needs the same domain; crop the candidate to the reference's "
                "z-extent (e.g. drop the gas-gap cells) before comparing."
            )

        # Aggregate both series over the same sampled times.  Different finish
        # times or step counts otherwise bias the mean of spiky transient fields.
        cand_idx = match_frames(frame_times(cand), frame_times(ref))
        print(f"Matched {cand_idx.size} reference frames to candidate frames by nearest time.")

        labels = ("min", "max", "mean")
        rtols = (args.rtol_extrema, args.rtol_extrema, args.rtol_mean)
        max_error = 0.0
        print(f"{'field':32s} {'stat':4s} {'candidate':>13s} {'reference':>13s} {'err/range':>9s}")
        for field in shared:
            cstats = field_stats(cand[field][cand_idx])
            rstats = field_stats(ref[field])
            # Judge every statistic against the field's own dynamic range, so a
            # near-zero mean of a field that peaks high is not a false alarm.
            dyn_range = max(abs(rstats[0]), abs(rstats[1]), abs(cstats[0]), abs(cstats[1]))
            for index, label in enumerate(labels):
                cv, rv = cstats[index], rstats[index]
                error = abs(cv - rv)
                rel = error / dyn_range if dyn_range > 0 else 0.0
                allowed = args.atol + rtols[index] * dyn_range
                flag = "" if error <= allowed else "  <-- EXCEEDS"
                print(f"{field:32s} {label:4s} {cv:13.5g} {rv:13.5g} {rel:9.2%}{flag}")
                if error > allowed:
                    raise CompareError(
                        f"{field} {label} differs: candidate={cv:g}, reference={rv:g}, "
                        f"error={error:g} ({rel:.1%} of range), allowed={allowed:g}"
                    )
                max_error = max(max_error, rel)
    except CompareError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        f"\nNPZ comparison passed: fields={len(shared)}, "
        f"rtol_mean={args.rtol_mean:g}, rtol_extrema={args.rtol_extrema:g}, "
        f"max_error_vs_range={max_error:.4%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
