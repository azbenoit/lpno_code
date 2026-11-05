"""Lightweight CUDA environment bootstrap helpers for model entrypoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf" / "hydra"
_DEFAULT_CONFIG_NAME = "config_NS_energy_PINO"
_NULL_STRINGS = {"", "none", "null", "~"}

# The GPU ids this workstation's launchers support. Lived as a bare {0, 1} in
# three places -- here, config_loading.configure_device, and nowhere at all in
# the tuner's --gpus parsing, which is how the tuner came to accept ids that
# every trial then died on. Kept in this module because it is import-light
# (stdlib only) and both consumers can reach it without pulling in torch.
SUPPORTED_CUDA_DEVICES = (0, 1)
_SUPPORTED_CUDA_DEVICE_STRINGS = {str(d) for d in SUPPORTED_CUDA_DEVICES}


def ensure_single_process_launch_env(environ=None) -> bool:
    """Run as one plain process when the job sits in a Slurm allocation without a Slurm task launch.

    PhysicsNeMo's DistributedManager tries `RANK` (torchrun) first; when that is absent it falls back to the Slurm
    variables if `SLURM_PROCID` is set, else to Open MPI, else to a single non-distributed process on whatever device
    is there. A plain `python -m lpno.models...` inside an `sbatch` script without `srun` carries `SLURM_PROCID` but
    none of the variables only `srun` sets, so the Slurm path fails on `int(None)` (`SLURM_NPROCS` or
    `SLURM_LOCALID`, depending on the sbatch options). `SLURM_LAUNCH_NODE_IPADDR` marks an `srun` launch. Without it
    this removes `SLURM_PROCID`, so PhysicsNeMo takes the single-process path a run outside Slurm takes, on GPU or CPU
    alike (its env and Slurm paths both call `torch.cuda.device`, which fails on a CPU-only node).
    `aiida_pino.scripts.live_tail` applies the same rule to the jobs it launches. A launch by `srun` or `torchrun`
    and a run outside Slurm are left as they are. Returns True when it removed the variable.
    """
    env = os.environ if environ is None else environ
    if "RANK" in env or "SLURM_PROCID" not in env or env.get("SLURM_LAUNCH_NODE_IPADDR"):
        return False
    env.pop("SLURM_PROCID")
    return True


def _strip_hydra_value(value: str) -> str:
    return value.strip().strip("'\"")


def _cli_key_value(arg: str) -> tuple[str, str] | None:
    if not arg or arg.startswith("--") or "=" not in arg:
        return None
    key, value = arg.split("=", 1)
    key = key.strip().lstrip("+")
    if not key:
        return None
    return key, _strip_hydra_value(value)


def _config_name_from_argv(args: Sequence[str]) -> str:
    for index, arg in enumerate(args):
        if arg.startswith("--config-name="):
            return _strip_hydra_value(arg.split("=", 1)[1])
        if arg == "--config-name" and index + 1 < len(args):
            return _strip_hydra_value(args[index + 1])
    return _DEFAULT_CONFIG_NAME


def _read_custom_cuda_defaults(config_name: str) -> tuple[str | None, str | None]:
    """Read custom.cuda_visible_devices/cuda_device from a Hydra YAML profile.

    This intentionally avoids importing OmegaConf/Hydra/PhysicsNeMo so that CUDA
    visibility is decided before those packages can import torch or initialize CUDA.
    """

    config_path = _CONFIG_DIR / f"{Path(config_name).stem}.yaml"
    if not config_path.exists():
        return None, None

    visible_devices: str | None = None
    cuda_device: str | None = None
    in_custom = False
    custom_indent: int | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue

        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        stripped = line_without_comment.strip()

        if stripped == "custom:":
            in_custom = True
            custom_indent = indent
            continue
        if in_custom and custom_indent is not None and indent <= custom_indent:
            break
        if not in_custom or ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        value = _strip_hydra_value(value)
        if key == "cuda_visible_devices":
            visible_devices = None if value.lower() in _NULL_STRINGS else value
        elif key == "cuda_device":
            cuda_device = value

    return visible_devices, cuda_device


def bootstrap_cuda_visible_devices_from_argv(argv: Sequence[str] | None = None) -> None:
    """Set ``CUDA_VISIBLE_DEVICES`` from CLI overrides before torch is imported."""

    args = list(sys.argv[1:] if argv is None else argv)
    cli_visible_devices: str | None = None
    cli_cuda_device: str | None = None

    for arg in args:
        parsed = _cli_key_value(arg)
        if parsed is None:
            continue
        key, value = parsed
        if key in {"cuda_visible_devices", "custom.cuda_visible_devices"}:
            cli_visible_devices = None if value.lower() in _NULL_STRINGS else value
        elif key in {"cuda_device", "custom.cuda_device"}:
            cli_cuda_device = value

    if cli_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli_visible_devices
        return
    if cli_cuda_device is not None:
        if cli_cuda_device in _SUPPORTED_CUDA_DEVICE_STRINGS:
            os.environ["CUDA_VISIBLE_DEVICES"] = cli_cuda_device
        return

    # Batch schedulers such as Slurm set CUDA_VISIBLE_DEVICES to the GPU(s)
    # allocated for the job. Preserve that assignment unless the user passed an
    # explicit CLI override above.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    config_visible_devices, config_cuda_device = _read_custom_cuda_defaults(_config_name_from_argv(args))
    if config_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config_visible_devices
        return
    if config_cuda_device in _SUPPORTED_CUDA_DEVICE_STRINGS:
        os.environ["CUDA_VISIBLE_DEVICES"] = config_cuda_device
