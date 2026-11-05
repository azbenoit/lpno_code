#!/usr/bin/env python3
"""Apply the flsgrf-cleanup policy to runs that finished before it was automatic.

`run_rectangular_case.py` now deletes the multi-GB `flsgrf*` result database once
the NPZ export completes.  Runs made before that still carry theirs -- 77 GB over
52 cases in `al_selfimprove_pv` alone, which is what filled the root filesystem on
2026-07-20 and refused a whole 300-case sweep.

The decision is delegated to `run_rectangular_case.discard_solver_result`, so this
script cannot drift from the live path: a case keeps its result unless
`npz/<case>.npz` opens as a complete archive.

    python3 python/reclaim_flsgrf.py runs             # report only
    python3 python/reclaim_flsgrf.py runs --apply     # delete
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_rectangular_case as rect  # noqa: E402


def case_dirs(root: Path) -> list[Path]:
    """Run directories under `root`, identified by the manifest the runner writes."""
    return sorted(path.parent for path in root.rglob("case_manifest.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Directory to walk (e.g. runs/)")
    parser.add_argument("--apply", action="store_true", help="Actually delete; without it, only report")
    args = parser.parse_args()
    if not args.root.is_dir():
        print(f"ERROR: not a directory: {args.root}", file=sys.stderr)
        return 2

    reclaimable = kept = 0
    freed_total = 0
    for case_dir in case_dirs(args.root):
        case_name = case_dir.name
        results = [path for path in case_dir.glob(f"flsgrf*.{case_name}*") if path.is_file()]
        if not results:
            continue
        size = sum(path.stat().st_size for path in results)
        if not rect.npz_export_is_complete(case_dir / "npz" / f"{case_name}.npz"):
            kept += 1
            print(f"KEEP  {case_name}: {size / 1024**3:.2f} GB (no complete NPZ export)")
            continue
        reclaimable += 1
        if args.apply:
            freed_total += rect.discard_solver_result(case_dir, case_name)
        else:
            freed_total += size
        print(f"{'FREED' if args.apply else 'WOULD'} {case_name}: {size / 1024**3:.2f} GB")

    verb = "reclaimed" if args.apply else "reclaimable"
    print(f"\n{reclaimable} cases {verb}, {freed_total / 1024**3:.1f} GB; {kept} kept (export missing or partial)")
    if not args.apply and reclaimable:
        print("Re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
