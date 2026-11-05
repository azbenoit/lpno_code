#!/usr/bin/env python3
"""Build batches of rectangular-case runs from one reference JSON config.

A batch is just a list of config dicts: each is a deep copy of a reference
config with a few keys overridden. This module builds that list (paired lists,
a grid, or evenly spaced sweeps), writes one JSON per case, and writes a task
file of run commands to queue with the spooler (``bash/spool.sh``). A lock
serialises solver runs that share a licence, so the cases run one at a time.

Override keys are dotted paths into the config, e.g. ``laser.power_w`` or
``simulation.finish_time_s``. The CLI front end is ``build_batch.py``.
"""

from __future__ import annotations

import copy
import json
import shlex
from itertools import product
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNNER = ROOT_DIR / "python" / "run_rectangular_case.py"


def linspace(lo: float, hi: float, num: int) -> list[float]:
    """Evenly spaced values from lo to hi inclusive (the even-spacing shortcut)."""
    if num < 1:
        raise ValueError("linspace num must be >= 1")
    if num == 1:
        return [float(lo)]
    step = (hi - lo) / (num - 1)
    return [lo + step * i for i in range(num)]


def set_dotted(target: dict, dotted_key: str, value: Any) -> None:
    """Set ``target['a']['b'] = value`` for dotted_key 'a.b', creating dicts."""
    keys = dotted_key.split(".")
    node = target
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def zip_overrides(**lists: list) -> list[dict]:
    """Parallel lists -> one override dict per index. All lists share a length.

    zip_overrides(power_w=[70, 90], scan_speed_m_per_s=[0.7, 0.3])
      -> [{'power_w': 70, 'scan_speed_m_per_s': 0.7},
          {'power_w': 90, 'scan_speed_m_per_s': 0.3}]
    """
    if not lists:
        return []
    lengths = {key: len(values) for key, values in lists.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"zip_overrides lists must share one length, got {lengths}")
    count = next(iter(lengths.values()))
    return [{key: lists[key][i] for key in lists} for i in range(count)]


def product_overrides(**lists: list) -> list[dict]:
    """Cross product of lists -> one override dict per combination (a sweep grid)."""
    keys = list(lists)
    return [dict(zip(keys, combo)) for combo in product(*(lists[key] for key in keys))]


def build_configs(
    reference: dict, overrides: list[dict], name_fn: Callable[[dict, dict], str]
) -> list[dict]:
    """Deep-copy reference per override, apply dotted keys, set a unique case_name."""
    configs: list[dict] = []
    seen: set[str] = set()
    for override in overrides:
        config = copy.deepcopy(reference)
        for key, value in override.items():
            set_dotted(config, key, value)
        name = name_fn(override, config)
        if name in seen:
            raise ValueError(f"Duplicate case_name {name!r}; make the names unique")
        seen.add(name)
        config["case_name"] = name
        configs.append(config)
    return configs


def pv_name(_override: dict, config: dict) -> str:
    """Default name from power and scan speed, e.g. 70W_0711mps (matches PINO dirs)."""
    laser = config["laser"]
    power = laser["power_w"]
    speed = laser["scan_speed_m_per_s"]
    power_text = str(int(power)) if float(power).is_integer() else f"{power:g}".replace(".", "p")
    return f"{power_text}W_{round(speed * 1000):04d}mps"


def write_jobs(
    configs: list[dict],
    out_dir: Path,
    *,
    cores: int = 1,
    threads: int = 1,
    export: str = "npz",
) -> Path:
    """Write one JSON per case and a task file of run commands for the spooler.

    Each task line is a complete run command; queue the file with the spooler
    (``spool.sh add --cmd`` per line) and drain it one run at a time. Returns
    the task-file path.
    """
    out_dir = Path(out_dir)
    config_dir = out_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    # Each case's run folder lands under the batch dir (out_dir/<case_name>), so a
    # batch keeps its configs, tasks.txt, pool.log, and run outputs together
    # instead of crowding runs/.
    out_root = out_dir.resolve()
    lines: list[str] = []
    for config in configs:
        name = config["case_name"]
        config_path = config_dir / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        command = (
            f"python3 {shlex.quote(str(RUNNER))} {shlex.quote(str(config_path))} "
            f"--cores {cores} --threads-per-rank {threads} --export {export} "
            f"--out-root {shlex.quote(str(out_root))}"
        )
        lines.append(command)
    task_file = out_dir / "tasks.txt"
    task_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return task_file
