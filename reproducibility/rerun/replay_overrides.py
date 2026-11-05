#!/usr/bin/env python3
"""Print the Hydra command-line overrides of an archived model, ready to replay elsewhere.

A model's `.hydra/overrides.yaml` is the exact override list its training ran with, in order.
Replaying it reproduces that training (or, against the model's own `.hydra/config.yaml`, its
evaluation), once the entries that name the original machine are dropped: the run name, the
data and output directories, and the GPU index (on a Slurm node the scheduler's
CUDA_VISIBLE_DEVICES must decide, and `custom.cuda_device` would override it). A single
leading `+` becomes `++`, as aiida-pino's eval adapter does, because the archived config
already declares every key the original run appended.

    replay_overrides.py MODEL_DIR/.hydra                      # one override per line
    replay_overrides.py MODEL_DIR/.hydra --seed 43 --max-steps 200
    replay_overrides.py MODEL_DIR/.hydra --config-name        # the config the model trained with
"""
import argparse
import sys
from pathlib import Path

import yaml

DROP = {"custom.name", "custom.data_dir", "custom.val_data_dir", "network_dir", "initialization_network_dir",
        "custom.cuda_device", "custom.cuda_visible_devices", "custom.cuda_local_device", "hydra.run.dir"}


def key_of(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+~")


def shortened_cadence(hydra_dir: Path, steps: int) -> dict[str, str]:
    """Overrides for a run cut to `steps`: every recording interval longer than the run is
    capped to it, since the pipeline refuses a run that would never checkpoint or validate,
    and the stop-criterion interval stays a multiple of the validation interval."""
    cfg = yaml.safe_load((hydra_dir / "config.yaml").read_text())
    training, stop = cfg.get("training", {}), cfg.get("stop_criterion", {})
    out = {"training.max_steps": f"training.max_steps={steps}"}
    cadence = {f"training.{k}": int(training[k]) for k in ("rec_results_freq", "rec_validation_freq",
                                                           "save_network_freq") if k in training}
    if "freq" in stop:
        cadence["stop_criterion.freq"] = int(stop["freq"])
    capped = {k: min(v, steps) for k, v in cadence.items()}
    validation = capped.get("training.rec_validation_freq")
    if validation and capped.get("stop_criterion.freq", validation) % validation:
        capped["stop_criterion.freq"] = validation
    out.update({k: f"{k}={v}" for k, v in capped.items() if v != cadence[k]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("hydra_dir", type=Path, help="a model's .hydra directory")
    parser.add_argument("--seed", type=int, help="replace the training seed")
    parser.add_argument("--max-steps", type=int, help="replace training.max_steps (smoke tests)")
    parser.add_argument("--config-name", action="store_true", help="print the config name and stop")
    args = parser.parse_args()
    if args.config_name:
        hydra = yaml.safe_load((args.hydra_dir / "hydra.yaml").read_text())["hydra"]
        print(hydra["job"]["config_name"])
        return
    overrides = yaml.safe_load((args.hydra_dir / "overrides.yaml").read_text()) or []
    replace = {}
    if args.seed is not None:
        replace["custom.seed"] = f"++custom.seed={args.seed}"
    if args.max_steps is not None:
        replace.update(shortened_cadence(args.hydra_dir, args.max_steps))
    out = []
    for item in map(str, overrides):
        key = key_of(item)
        if key in DROP:
            continue
        if key in replace:
            item = replace.pop(key)
        elif item.startswith("+") and not item.startswith("++"):
            item = "+" + item
        out.append(item)
    out.extend(replace.values())   # keys the original run never set
    sys.stdout.write("".join(f"{item}\n" for item in out))


if __name__ == "__main__":
    main()
