#!/usr/bin/env python3
"""Subsample a run NPZ to at most N evenly-spaced time steps (keeps first + last).

Only arrays whose first axis is the time axis are thinned; the ``x``/``y``/``z``
coordinate axes and per-run scalars are copied unchanged.  Useful for shrinking
stored results (e.g. after the FLSGRF file is deleted) without a re-solve.

    python3 python/subsample_npz.py runs/<case>/npz/<case>.npz --max-frames 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Same directory; the runner puts python/ on sys.path and so does every caller.
from verify_npz_checksums import sidecar_path, write_sidecar


def subsample(npz_path: Path, max_frames: int) -> int:
    """Thin ``npz_path`` in place to <= ``max_frames`` frames; return the new count."""
    data = dict(np.load(npz_path, allow_pickle=True))
    n = int(np.asarray(data["time_s"]).shape[0])
    if n <= max_frames:
        return n
    idx = np.unique(np.linspace(0, n - 1, max_frames).round().astype(int))
    out = {}
    for key, value in data.items():
        value = np.asarray(value)
        if key not in ("x", "y", "z") and value.ndim >= 1 and value.shape[0] == n:
            out[key] = value[idx]
        else:
            out[key] = value
    np.savez_compressed(npz_path, **out)
    # The rewrite changes the bytes, so a sha256 sidecar written for the old file
    # now describes nothing on disk, and `check_file` -- which prefers the sidecar
    # -- would report a healthy thinned file as corrupt. Refresh it only if one
    # exists: pre-sidecar data keeps its CRC fallback rather than gaining a claim.
    if sidecar_path(npz_path).exists():
        write_sidecar(npz_path)
    return int(len(idx))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="Run NPZ to thin in place")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum time steps to keep (default 100)")
    args = parser.parse_args()
    if args.max_frames < 2:
        raise SystemExit("--max-frames must be at least 2")
    before = int(np.asarray(np.load(args.npz, allow_pickle=True)["time_s"]).shape[0])
    after = subsample(args.npz, args.max_frames)
    print(f"{args.npz.name}: {before} -> {after} frames")


if __name__ == "__main__":
    main()
