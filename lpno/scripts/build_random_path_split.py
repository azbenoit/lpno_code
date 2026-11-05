#!/usr/bin/env python3
"""Build a train/val/test split of FLOW-3D random/sinusoidal weld-path runs.

Reads consolidated-NPZ runs (plus their ``case_manifest.json``) produced by the
sibling ``flow3D_scripts`` batches and writes a stratified 60:20:20 split into a
destination directory, one ``<case>.npz`` + ``<case>.manifest.json`` per case.

Frames are optionally subsampled (evenly in time) to keep the eager NPZ loader's
memory bounded when training with ``time_as_batch`` (each frame becomes a sample).

Source batch directories are passed on the command line -- this script hard-codes
no external absolute paths. Each source is a batch root whose runs live under
``<root>/runs/<case>/`` with ``npz/<case>.npz`` and ``case_manifest.json``.

Example::

    python scripts/build_random_path_split.py \
        --source /abs/tial64_random50 --source /abs/tial64_sine10 \
        --dest data/flow3D/random_paths --frames 20 --seed 0
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
from pathlib import Path

import numpy as np

# Field arrays carry a leading time axis (nt, nx, ny, nz[, 3]); coords are 1-D and
# scalars are 0-D, so "ndim >= 4 or the time vector" cleanly selects time-indexed data
# without confusing a spatial axis that happens to equal nt (e.g. ny == nt == 100).
_TIME_VECTOR_KEY = "time_s"


def _find_cases_flat(source: Path) -> list[tuple[str, Path, Path]]:
    """Flat pool: ``<source>/<case>.npz`` + ``<source>/<case>.manifest.json``.

    This is the layout a prior split (or a staging move) already produced, i.e.
    the same shape the loader consumes -- so a staged flat pool can be re-split
    without the original ``runs/<case>/npz/`` tree."""
    cases: list[tuple[str, Path, Path]] = []
    for npz in sorted(source.glob("*.npz")):
        name = npz.stem
        manifest = npz.with_name(f"{name}.manifest.json")
        if not manifest.is_file():
            print(f"[split] skip {name}: missing {manifest.name}")
            continue
        cases.append((name, npz, manifest))
    return cases


def _find_cases(source: Path) -> list[tuple[str, Path, Path]]:
    """Return (case_name, npz_path, manifest_path) for every run under a batch root."""
    runs_dir = source / "runs"
    if not runs_dir.is_dir():
        # No runs/ tree: treat the source as a flat pool of <case>.npz + sidecar.
        return _find_cases_flat(source)
    cases: list[tuple[str, Path, Path]] = []
    for case_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        name = case_dir.name
        npz = case_dir / "npz" / f"{name}.npz"
        manifest = case_dir / "case_manifest.json"
        if not npz.is_file():
            print(f"[split] skip {name}: missing {npz.relative_to(source)}")
            continue
        if not manifest.is_file():
            print(f"[split] skip {name}: missing case_manifest.json")
            continue
        cases.append((name, npz, manifest))
    return cases


def _subsample_indices(nt: int, frames: int) -> np.ndarray:
    if frames <= 0 or frames >= nt:
        return np.arange(nt)
    # Evenly spaced, inclusive of first and last frame.
    return np.unique(np.linspace(0, nt - 1, frames).round().astype(int))


def _link_case(npz: Path, manifest: Path, dest_dir: Path) -> int:
    """Symlink a case (npz + manifest) into a split dir -- no copy, all frames kept.

    This is the layout the lazy loader wants: full-frame NPZ referenced in place,
    so the whole dataset stays on disk once and every timestep is available."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = npz.stem
    for src, link in ((npz, dest_dir / f"{name}.npz"), (manifest, dest_dir / f"{name}.manifest.json")):
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.realpath(src), link)
    with np.load(dest_dir / f"{name}.npz") as d:
        return int(np.asarray(d[_TIME_VECTOR_KEY]).size)


def _write_case(npz: Path, manifest: Path, dest_dir: Path, frames: int,
                stride=(1, 1, 1), skip_existing: bool = True) -> int:
    out_npz = dest_dir / f"{npz.stem}.npz"
    if skip_existing and out_npz.is_file() and (dest_dir / f"{npz.stem}.manifest.json").is_file():
        with np.load(out_npz) as d:
            return int(np.asarray(d[_TIME_VECTOR_KEY]).size)
    sx, sy, sz = stride
    with np.load(npz, allow_pickle=True) as data:
        keys = list(data.files)
        nt = int(np.asarray(data[_TIME_VECTOR_KEY]).size)
        idx = _subsample_indices(nt, frames)
        nx = int(np.asarray(data["x"]).size)
        ny = int(np.asarray(data["y"]).size)
        nz = int(np.asarray(data["z"]).size)
        out: dict[str, np.ndarray] = {}
        for k in keys:
            arr = np.asarray(data[k])
            if k == _TIME_VECTOR_KEY:
                out[k] = arr[idx]
            elif k == "x":
                out[k] = arr[::sx]
            elif k == "y":
                out[k] = arr[::sy]
            elif k == "z":
                out[k] = arr[::sz]
            elif arr.ndim >= 4:  # field (nt,nx,ny,nz[,3]) -> stride time + space
                out[k] = arr[idx][:, ::sx, ::sy, ::sz]
            else:
                out[k] = arr
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = npz.stem
    np.savez_compressed(dest_dir / f"{name}.npz", **out)
    shutil.copyfile(manifest, dest_dir / f"{name}.manifest.json")
    return len(idx)


def _split_counts(n: int, fracs=(0.6, 0.2, 0.2)) -> tuple[int, int, int]:
    n_train = round(n * fracs[0])
    n_val = round(n * fracs[1])
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", default=[], help="batch root (repeatable)")
    ap.add_argument("--from-split", type=Path, default=None, dest="from_split",
                    help="copy an existing split's membership verbatim from DIR/{train,val,test} "
                         "instead of assigning a new one -- use when a dataset already has a "
                         "recorded split (e.g. an AiiDA collection) and only --stride/--frames "
                         "should change")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=20, help="frames kept per case (0 = all)")
    ap.add_argument("--stride", type=int, nargs=3, default=[1, 1, 1],
                    metavar=("SX", "SY", "SZ"), help="spatial downsample stride per axis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", action="append", default=[],
                    help="drop cases whose name matches this glob (repeatable), e.g. 'abx_*'")
    ap.add_argument("--force-test", action="append", default=[], dest="force_test",
                    help="route cases matching this glob into the test split only, held out "
                         "of train/val (repeatable), e.g. 'abx_*' for a real-alloy probe")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total cases per source to this many (random subset), 0 = all; "
                         "bounds the eager-loader memory footprint of large pools")
    ap.add_argument("--fracs", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    metavar=("TRAIN", "VAL", "TEST"), help="train/val/test split fractions")
    ap.add_argument("--link", action="store_true",
                    help="symlink full-frame cases instead of copying subsampled frames "
                         "(for the lazy loader; ignores --frames/--stride)")
    ap.add_argument("--clean", action="store_true", help="wipe dest split dirs first")
    args = ap.parse_args()
    if not args.source and args.from_split is None:
        ap.error("pass --source (assign a new split) or --from-split (reuse an existing one)")

    rng = np.random.default_rng(args.seed)
    dest = args.dest
    if args.clean:
        for sub in ("train", "val", "test"):
            shutil.rmtree(dest / sub, ignore_errors=True)

    assign: dict[str, list[tuple[str, Path, Path]]] = {"train": [], "val": [], "test": []}
    if args.from_split is not None:
        for split in ("train", "val", "test"):
            assign[split] = _find_cases_flat(args.from_split / split)
            print(f"[split] from-split {split}: {len(assign[split])} cases")
    # Stratify per source so each path family (random, sine) is represented in every split.
    for source in args.source:
        cases = _find_cases(Path(source))
        if args.exclude:
            kept = [c for c in cases if not any(fnmatch.fnmatch(c[0], pat) for pat in args.exclude)]
            print(f"[split] {Path(source).name}: excluded {len(cases) - len(kept)} by {args.exclude}")
            cases = kept
        forced_test = []
        if args.force_test:
            forced_test = [c for c in cases if any(fnmatch.fnmatch(c[0], pat) for pat in args.force_test)]
            cases = [c for c in cases if c not in forced_test]
            print(f"[split] {Path(source).name}: {len(forced_test)} cases forced to test by {args.force_test}")
        if args.limit and len(cases) > args.limit:
            pick = rng.permutation(len(cases))[: args.limit]
            cases = [cases[i] for i in sorted(pick)]
            print(f"[split] {Path(source).name}: capped to {len(cases)} cases (--limit {args.limit})")
        fracs = tuple(args.fracs)
        order = rng.permutation(len(cases))
        n_train, n_val, _ = _split_counts(len(cases), fracs)
        for pos, ci in enumerate(order):
            split = "train" if pos < n_train else ("val" if pos < n_train + n_val else "test")
            assign[split].append(cases[ci])
        for c in forced_test:
            assign["test"].append(c)
        print(f"[split] {Path(source).name}: {len(cases)} split cases -> {_split_counts(len(cases), fracs)} "
              f"+ {len(forced_test)} forced-test (train,val,test)")

    total_frames = {}
    for split, items in assign.items():
        kept = 0
        for name, npz, manifest in items:
            if args.link:
                kept = _link_case(npz, manifest, dest / split)
            else:
                kept = _write_case(npz, manifest, dest / split, args.frames, stride=tuple(args.stride))
        total_frames[split] = (len(items), kept)
        print(f"[split] {split}: {len(items)} cases, {kept} frames/case -> {dest/split}")
    print(f"[split] done. dest={dest}")


if __name__ == "__main__":
    main()
