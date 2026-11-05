"""GPU selection: put each loaded model on whichever CUDA device has the most
free memory, so a full GPU0 (e.g. a training job) is skipped automatically.

Free memory is read from ``nvidia-smi`` rather than ``torch.cuda.mem_get_info``:
the latter needs a CUDA context on the queried device and would OOM on the very
GPU we are trying to avoid. Torch's device indices are pinned to nvidia-smi's PCI
order (``CUDA_DEVICE_ORDER=PCI_BUS_ID``), set here before torch initialises CUDA,
so the free-memory index maps to the right ``cuda:N``.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

logger = logging.getLogger(__name__)


def _free_mem_per_gpu() -> list[int] | None:
    """Free MiB per GPU in nvidia-smi (PCI) order, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("nvidia-smi free-memory query failed (%s); using default device", type(exc).__name__)
        return None
    free = [int(x.strip()) for x in out.splitlines() if x.strip()]
    return free or None


def bootstrap_cuda_visible_devices(cuda_device: str | None = None) -> str | None:
    """Confine the process to ONE GPU by setting ``CUDA_VISIBLE_DEVICES`` *before*
    torch imports, so the CUDA context and every allocation land on the chosen GPU
    — never a stray one on a full GPU0. Mirrors lpno' cuda_bootstrap
    (deciding visibility before torch initialises CUDA). Must be called before the
    first torch import.

    Precedence: an existing ``CUDA_VISIBLE_DEVICES`` (a caller / Slurm pin) is left
    untouched; otherwise ``cuda_device`` (or ``LPFNO_CUDA_DEVICE``): ``cpu`` hides
    all GPUs, an index / ``cuda:N`` pins that GPU, and ``auto`` pins the GPU with
    the most free memory (nvidia-smi — measured before we allocate anything, so
    it reflects only *other* processes). Returns the value set, or None.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return None
    spec = (cuda_device if cuda_device is not None else os.environ.get("LPFNO_CUDA_DEVICE", "auto")).strip().lower()
    if spec == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return ""
    if spec.startswith("cuda:"):
        spec = spec.split(":", 1)[1]
    if spec.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = spec
        logger.info("CUDA_VISIBLE_DEVICES=%s (pinned via cuda_device)", spec)
        return spec
    free = _free_mem_per_gpu()
    if free is not None and len(free) > 1:
        idx = max(range(len(free)), key=lambda i: free[i])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        logger.info("CUDA_VISIBLE_DEVICES=%d (freest; free MiB per GPU: %s)", idx, free)
        return str(idx)
    return None


def empty_cache() -> None:
    """Return the CUDA caching allocator's reserved-but-unused blocks to the GPU.

    Each model's forward has a different working-set shape, so without freeing,
    the allocator's reserved pool grows to the *sum* of every model's peak — the
    backend held 17 GB across a handful of models whose resident weights are under
    2 GB. Inference here is infrequent (seconds per request), so calling this after
    each forward keeps the footprint at ~resident weights/graphs plus one transient
    working set, for a negligible re-malloc cost. No-op without CUDA (stub/CPU).
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 — freeing cache must never break a response
        logger.debug("empty_cache skipped (%s)", type(exc).__name__)


def _torch_cuda_mem_get_info():
    """(free, total) bytes on the current device, or None without CUDA.

    Split out so tests can stub it: the headroom guard must degrade to a no-op
    off-GPU rather than raise inside a request.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        logger.debug("mem_get_info unavailable (%s)", type(exc).__name__)
        return None


def free_vram_bytes() -> int | None:
    """Free VRAM on the device this process is pinned to, or None if unknown.

    This is the DEVICE's free memory, not the allocator's -- what matters is
    whether the next forward can be satisfied at all, including memory held by
    other processes sharing the card (a training, typically).
    """
    info = _torch_cuda_mem_get_info()
    return int(info[0]) if info else None


def pick_device(torch: Any, cuda_device: str = "auto") -> Any:
    """Resolve the torch device to load a model on.

    ``cuda_device``:
      - ``"auto"`` (default): the GPU with the most free memory (skips a full GPU0);
      - ``"cpu"``: force CPU;
      - ``"1"`` / ``"cuda:1"``: force that specific device.
    """
    spec = (cuda_device or "auto").strip().lower()
    if spec == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if spec.startswith("cuda:"):
        spec = spec.split(":", 1)[1]
    if spec.isdigit():
        # An index here is the PHYSICAL gpu -- the same value
        # bootstrap_cuda_visible_devices already pinned CUDA_VISIBLE_DEVICES to.
        # Once pinned, that gpu is the only one torch can see and it is numbered
        # 0, so reusing the physical index raises "invalid device ordinal" and
        # kills startup. Keep physical and logical apart the way lpno
        # does with cuda_device / cuda_local_device (config_loading.py BUG-10).
        idx = int(spec)
        return torch.device(f"cuda:{idx}" if idx < int(torch.cuda.device_count()) else "cuda:0")

    n = int(torch.cuda.device_count())
    free = _free_mem_per_gpu()
    # Only trust the auto pick when nvidia-smi's GPU count matches torch's (no
    # CUDA_VISIBLE_DEVICES remapping) and there is a choice to make.
    if free is not None and len(free) == n and n > 1:
        idx = max(range(n), key=lambda i: free[i])
        logger.info("Auto-selected cuda:%d (free MiB per GPU: %s)", idx, free)
        return torch.device(f"cuda:{idx}")
    return torch.device("cuda")
