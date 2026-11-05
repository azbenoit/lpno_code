#!/usr/bin/env python
"""Build the demo's ground truth from the restricted Zenodo record's payload tars.

The three pages compare against cases that are in the restricted record
(doi:10.5281/zenodo.23325088) as `payloads_<family>.tar`, one `<uuid>.npz` and
`<uuid>.manifest.json` per simulation, with `payload_manifest.csv` mapping each uuid
to its family, split and case name:

  steady and straight pages: the held-out straight-track cases (14 val, 13 test) of the
      families original_straight_path_data_calibrated_20260731, ti64_keyhole40 and
      ti64_keyhole40_capped; `--splits train val test` adds their 96 training cases;
  free-form page: the 22 test cases of ti64_rand150_newphys.

Each case goes through the same per-case code as the AiiDA builders
(`build_calibrated_steady_gt.py`, `build_calibrated_straight_gt.py`,
`build_newphys_assets.py`), with the derivation the served models were scored on:

  steady   moving frame of 30 steps, 30-frame average, super-resolution 1, full z
           (one 61x40x40 window per case), one consolidated file per split;
  straight the 20 stored frames nearest an even sweep of each case, full 90x40x40 grid;
  free-form every stored frame, strided 2x2x1 to the models' 75x50x40 grid.

    python scripts/build_ground_truth_from_deposit.py --deposit <dir with the tars and payload_manifest.csv>
    python scripts/build_ground_truth_from_deposit.py --deposit <dir> --pages straight --limit 1
    python scripts/assets.py bundle --part ground_truth      # then the bundle

The steady page needs the `lpno` package (its moving-frame derivation); the other two
need numpy only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STRAIGHT_FAMILIES = ("original_straight_path_data_calibrated_20260731", "ti64_keyhole40",
                     "ti64_keyhole40_capped")
PAGES = {
    "steady": {"families": STRAIGHT_FAMILIES, "splits": ("train", "val", "test"),
               "out": REPO / "data" / "steady-state" / "ground_truth_calibrated"},
    "straight": {"families": STRAIGHT_FAMILIES, "splits": ("train", "val", "test"),
                 "out": REPO / "data" / "straight_path_time" / "ground_truth_calibrated"},
    "path": {"families": ("ti64_rand150_newphys",), "splits": ("test",),
             "out": REPO / "data" / "random_path" / "ground_truth" / "rand150_newphys_test"},
}
STEADY_PROFILE = {"moving_ref_frame": True, "moving_frame_steps": 30, "time_avg_window": 30,
                  "super_res_factor": 1, "include_time": True, "crop_z_top": 0}
STRAIGHT_PROFILE = {"stride": [1, 1, 1], "crop_z_top": 0}
EXPECTED = {("steady", "train"): 96, ("steady", "val"): 14, ("steady", "test"): 13,
            ("straight", "train"): 96, ("straight", "val"): 14, ("straight", "test"): 13,
            ("path", "test"): 22}


def _builder(file_name: str):
    spec = importlib.util.spec_from_file_location(file_name[:-3], REPO / "scripts" / file_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cases(deposit: Path, families, split: str) -> list[dict]:
    """The rows of payload_manifest.csv for these families and this split, by case name."""
    rows = [r for r in csv.DictReader(open(deposit / "payload_manifest.csv"))
            if r["family"] in families and r["split"] == split]
    return sorted(rows, key=lambda r: (r["family"], r["case_name"]))


class Payloads:
    """Random access to `<uuid>.npz` / `<uuid>.manifest.json` in the family tars, extracted one
    case at a time into a scratch directory (a family tar holds up to 19 GB)."""

    def __init__(self, deposit: Path):
        self.deposit, self.open_tars = deposit, {}

    def extract(self, row: dict, work: Path) -> tuple[Path, dict]:
        fam = row["family"]
        if fam not in self.open_tars:
            self.open_tars[fam] = tarfile.open(self.deposit / f"payloads_{fam}.tar")
        tar = self.open_tars[fam]
        npz = work / f"{row['case_name']}.npz"
        with tar.extractfile(f"{row['uuid']}.npz") as src, open(npz, "wb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
        with tar.extractfile(f"{row['uuid']}.manifest.json") as src:
            manifest = json.load(src)
        return npz, manifest


def build_straight(pay: Payloads, deposit: Path, splits, limit: int) -> None:
    mod = _builder("build_calibrated_straight_gt.py")
    for split in splits:
        rows = cases(deposit, PAGES["straight"]["families"], split)[: limit or None]
        dest = PAGES["straight"]["out"] / split
        total = 0
        for row in rows:
            with tempfile.TemporaryDirectory() as td:
                npz, manifest = pay.extract(row, Path(td))
                total += mod.write_case(npz, manifest, row["case_name"], STRAIGHT_PROFILE, dest)
        print(f"[straight/{split}] {len(rows)} cases, {total / 1e6:.0f} MB -> {dest}")


def build_path(pay: Payloads, deposit: Path, splits, limit: int) -> None:
    mod = _builder("build_newphys_assets.py")
    rows = cases(deposit, PAGES["path"]["families"], "test")[: limit or None]
    dest, total = PAGES["path"]["out"], 0
    for row in rows:
        with tempfile.TemporaryDirectory() as td:
            npz, manifest = pay.extract(row, Path(td))
            total += mod.write_case(npz, manifest, row["case_name"], dest)
    print(f"[path/test] {len(rows)} cases, {total / 1e6:.0f} MB -> {dest}")


def build_steady(pay: Payloads, deposit: Path, splits, limit: int) -> None:
    mod = _builder("build_calibrated_steady_gt.py")
    for split in splits:
        rows = cases(deposit, PAGES["steady"]["families"], split)[: limit or None]
        stack: dict[str, list[np.ndarray]] = {}
        for row in rows:
            with tempfile.TemporaryDirectory() as td:
                case_dir = Path(td) / row["case_name"]
                case_dir.mkdir()
                npz, manifest = pay.extract(row, case_dir)
                (case_dir / f"{row['case_name']}.manifest.json").write_text(json.dumps(manifest))
                sr = STEADY_PROFILE["super_res_factor"]
                arrays = mod._crop_gas(mod._derive_one(case_dir, sr, STEADY_PROFILE),
                                       STEADY_PROFILE["crop_z_top"], sr)
            for k, v in arrays.items():
                stack.setdefault(k, []).append(v)
            print(f"    {row['case_name']}")
        dest = PAGES["steady"]["out"] / split
        dest.mkdir(parents=True, exist_ok=True)
        payload = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in stack.items()}
        path = dest / "var_1_time_steps30_avg30.npz"
        np.savez_compressed(path, **payload)
        print(f"[steady/{split}] {len(rows)} cases, Temperature{payload['Temperature'].shape}, "
              f"{path.stat().st_size / 1e6:.0f} MB -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deposit", type=Path, required=True,
                    help="directory holding the payload tars and payload_manifest.csv")
    ap.add_argument("--pages", nargs="*", default=["straight", "path", "steady"], choices=list(PAGES))
    ap.add_argument("--splits", nargs="*", default=["val", "test"],
                    help="splits of the straight-track pages (the free-form page has test only)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases per split (a check)")
    a = ap.parse_args()
    missing = [f for p in a.pages for f in PAGES[p]["families"]
               if not (a.deposit / f"payloads_{f}.tar").is_file()]
    if missing or not (a.deposit / "payload_manifest.csv").is_file():
        print(f"missing in {a.deposit}: {sorted(set(missing)) or 'payload_manifest.csv'}")
        return 1
    if not a.limit:
        for (page, split), n in EXPECTED.items():
            if page in a.pages and (page == "path" or split in a.splits):
                got = len(cases(a.deposit, PAGES[page]["families"], split))
                if got != n:
                    print(f"refusing: {page}/{split} has {got} cases in payload_manifest.csv, expected {n}")
                    return 1
    pay = Payloads(a.deposit)
    build = {"straight": build_straight, "path": build_path, "steady": build_steady}
    for page in a.pages:
        build[page](pay, a.deposit, a.splits, a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
