#!/usr/bin/env python3
"""Expand a reference config into a batch of run configs plus a task file to queue with the spooler (bash/spool.sh).

Pick exactly one way to vary parameters:

  --zip KEY=SPEC ...    parallel (paired) lists, e.g. the matched (power, speed)
                        pairs of a dataset. All SPECs must share one length.
  --grid KEY=SPEC ...   cross product of lists (a sweep). A SPEC of the form
                        lo:hi:n is evenly spaced (the even-spacing shortcut).
  --params-tsv FILE     one case per row; the header holds dotted config keys
                        (e.g. laser.power_w) and an optional case_name column.

SPEC is a comma list (200,300,400) or, only for grids/sweeps, lo:hi:n (200:400:5).

Examples:
  # Evenly spaced power sweep
  python3 python/build_batch.py configs/rectangular_moving_laser_smoke.json batches/psweep \\
    --grid laser.power_w=200:400:5

  # Matched (power, speed) cases, with finish time derived from the fixed track
  python3 python/build_batch.py configs/case_fine_1_10um_reproduction.json batches/pino \\
    --params-tsv params.tsv --derive-finish-time-from-track
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import batch


def _coerce(text: str):
    text = text.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def parse_spec(spec: str) -> list:
    if ":" in spec and "," not in spec:
        lo, hi, num = spec.split(":")
        return batch.linspace(float(lo), float(hi), int(num))
    return [_coerce(value) for value in spec.split(",")]


def parse_kv(items: list[str]) -> dict[str, list]:
    parsed: dict[str, list] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected KEY=SPEC, got {item!r}")
        key, spec = item.split("=", 1)
        parsed[key.strip()] = parse_spec(spec)
    return parsed


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key.strip(): _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]
    if not rows:
        raise SystemExit(f"{path} needs a header row and at least one data row")
    return rows


def track_length(config: dict) -> float:
    start = config["laser"]["start_point_m"]
    end = config["laser"]["end_point_m"]
    return math.hypot(end[0] - start[0], end[1] - start[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", type=Path, help="Reference JSON config providing defaults")
    parser.add_argument("out_dir", type=Path, help="Batch directory for configs/ and tasks.txt")
    parser.add_argument("--zip", action="append", metavar="KEY=SPEC", default=[])
    parser.add_argument("--grid", action="append", metavar="KEY=SPEC", default=[])
    parser.add_argument("--params-tsv", type=Path)
    parser.add_argument(
        "--derive-finish-time-from-track",
        action="store_true",
        help="Set simulation.finish_time_s = track_length / scan_speed per case "
        "(for turn_off_at_end=false line tracks of fixed length).",
    )
    parser.add_argument("--name-prefix", default="", help="Prefix for every case_name (e.g. train_)")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--threads-per-rank", type=int, default=1)
    parser.add_argument("--export", default="npz", choices=["npz", "vtk", "vtk-legacy60", "both", "none"])
    args = parser.parse_args()

    if sum(bool(x) for x in (args.params_tsv, args.zip, args.grid)) != 1:
        raise SystemExit("Choose exactly one of --params-tsv, --zip, or --grid")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if args.params_tsv:
        overrides = read_tsv(args.params_tsv)
    elif args.zip:
        overrides = batch.zip_overrides(**parse_kv(args.zip))
    else:
        overrides = batch.product_overrides(**parse_kv(args.grid))
    if not overrides:
        raise SystemExit("No cases produced")

    def name_fn(override: dict, config: dict) -> str:
        base = override.get("case_name") or batch.pv_name(override, config)
        return f"{args.name_prefix}{base}"

    configs = batch.build_configs(reference, overrides, name_fn)
    if args.derive_finish_time_from_track:
        for config in configs:
            # Deriving finish_time_s = scan_time leaves no room for the turn-off
            # event, which requires finish_time_s > scan_time; the flag is only
            # for turn_off_at_end=false fixed-length line tracks.
            if config.get("laser", {}).get("turn_off_at_end", True):
                raise SystemExit(
                    f"--derive-finish-time-from-track needs laser.turn_off_at_end=false "
                    f"(case {config.get('case_name', '?')!r} keeps it on); the derived finish time "
                    "equals the scan time, which is too short for the turn-off event"
                )
            speed = config["laser"]["scan_speed_m_per_s"]
            batch.set_dotted(config, "simulation.finish_time_s", track_length(config) / speed)

    task_file = batch.write_jobs(
        configs,
        args.out_dir,
        cores=args.cores,
        threads=args.threads_per_rank,
        export=args.export,
    )
    print(f"Wrote {len(configs)} configs under {args.out_dir / 'configs'}")
    print(f"Task file: {task_file}")
    print("Queue with the spooler (runs go one at a time):")
    print(f'  while read -r c; do bash bash/spool.sh add --cmd "$c"; done < {task_file}')
    print("  bash bash/spool.sh start --drain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
