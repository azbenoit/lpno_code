#!/usr/bin/env python3
"""Reproduce the experimental validation campaign.

Every config under ``validation/configs/`` is one solve of the campaign
inventoried in ``MANIFEST.md``. Two routes:

- **Without a FLOW-3D license** (default): render every case with
  ``--prepare-only`` and verify the generated ``prepin`` inputs. This checks
  the full config -> physics rendering pipeline.
- **With a license** (``--solve``): solve the cases sequentially (the shared
  lock serializes them) and export the NPZ per case. Then score the tree
  against the published anchors with
  ``python3 python/score_anchors.py --plan validation/scoring_plan.json --runs <out-root>``;
  the manuscript's supplementary notebook (``experimental_validation.ipynb`` in
  the ``lpno`` sibling repository) presents the same rows. An
  archive of already-solved NPZs, once deposited, scores the same way (see
  ``MANIFEST.md``).

Cases run into ``<out-root>/<case_name>/`` (default ``runs/validation/``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = REPO / "python" / "run_rectangular_case.py"


def configs(only: list[str] | None = None) -> list[Path]:
    found = sorted((HERE / "configs").glob("*.json"))
    if only:
        wanted = set(only)
        found = [p for p in found if p.stem in wanted]
        missing = wanted - {p.stem for p in found}
        if missing:
            raise SystemExit(f"unknown case(s): {', '.join(sorted(missing))}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve", action="store_true",
                    help="full solves (needs a configured FLOW-3D license); default is --prepare-only rendering")
    ap.add_argument("--only", nargs="+", metavar="CASE",
                    help="restrict to these case names (config file stems)")
    ap.add_argument("--out-root", type=Path, default=REPO / "runs" / "validation")
    ap.add_argument("--cores", type=int, default=4)
    args = ap.parse_args()

    todo = configs(args.only)
    failed: list[str] = []
    for i, cfg in enumerate(todo, 1):
        cmd = [sys.executable, str(RUNNER), str(cfg),
               "--case-name", cfg.stem, "--out-root", str(args.out_root)]
        if args.solve:
            cmd += ["--cores", str(args.cores), "--threads-per-rank", str(args.cores)]
        else:
            cmd += ["--prepare-only"]
        print(f"[{i}/{len(todo)}] {cfg.stem}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(cfg.stem)
            print(f"  FAILED: {cfg.stem}", flush=True)
    print(f"\n{len(todo) - len(failed)}/{len(todo)} ok" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
