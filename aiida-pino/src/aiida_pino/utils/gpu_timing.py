"""Per-query GPU cost that a busy box cannot move.

`utils/timing.py` handles whole operations, where the only honest defences are
normalising by work done and recording the contention. Per-query inference cost
needs something stronger, and can have it:

* **peak memory is deterministic.** `reset_peak_memory_stats` →
  `max_memory_allocated` measures this process's own allocator high-water mark.
  Another process's kernels cannot move it. Half the answer is contention-free
  by construction.
* **CUDA events measure kernel time on the device**, not wall time on a loaded
  host. On a `gpu_spool` lease the card is exclusive (`CUDA_VISIBLE_DEVICES` is
  pinned), so no other process's kernels interleave, and a median over many
  repeats absorbs launch-gap jitter from CPU contention.

What is still not contention-proof, and is therefore reported rather than hidden:
host→device copies (PCIe is shared) are timed separately, and the SM clock is
recorded at both ends so thermal throttling shows up as a number instead of as
unexplained slowness.

    from aiida_pino.utils.gpu_timing import timed_call
    stats = timed_call(lambda: model(x), warmup=10, repeats=100)
    # {'median_ms': 12.4, 'iqr_ms': 0.3, 'peak_mb': 8123, 'sm_mhz': [1635, 1590], ...}

Falls back to a wall-clock timer with `device='cpu'` when torch or CUDA is
absent, and says so in the payload rather than quietly reporting a different
quantity.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

#: Repeats below this make a median meaningless; the caller is told, not fixed.
MIN_USEFUL_REPEATS = 5


def _sm_clock_mhz() -> int | None:
    from aiida_pino.scripts.live_tail import _first_line

    line = _first_line(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
    try:
        return int(str(line).split("\n")[0].strip())
    except (TypeError, ValueError):
        return None


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    if len(ordered) >= 4:
        lower = statistics.median(ordered[: len(ordered) // 2])
        upper = statistics.median(ordered[(len(ordered) + 1) // 2 :])
    else:
        lower = upper = median
    return {
        "median_ms": round(median, 4),
        "p25_ms": round(lower, 4),
        "p75_ms": round(upper, 4),
        "iqr_ms": round(upper - lower, 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def timed_call(
    fn: Callable[[], Any],
    *,
    warmup: int = 10,
    repeats: int = 100,
    sync_each: bool = True,
) -> dict[str, Any]:
    """Median CUDA-event time and peak allocator memory for one repeated call.

    `warmup` runs first and is discarded — it reaches thermal and cache steady
    state, and absorbs lazy CUDA context and autotune costs that would otherwise
    land entirely in the first sample and drag a small-`repeats` median.

    Peak memory is measured over the *whole* run including warmup, because the
    high-water mark is what a caller needs to size a card, and it is the one
    figure contention cannot touch.
    """
    payload: dict[str, Any] = {"warmup": warmup, "repeats": repeats}
    if repeats < MIN_USEFUL_REPEATS:
        payload["warning"] = (
            f"{repeats} repeats is too few for a median to mean much; "
            f"{MIN_USEFUL_REPEATS} is the floor and ~100 is the intended use"
        )

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    cuda = bool(torch is not None and torch.cuda.is_available())
    payload["device"] = "cuda" if cuda else "cpu"
    if not cuda:
        # Honest fallback: this is wall time, and wall time on a shared host is
        # not the same quantity. Say which was measured.
        payload["timer"] = "perf_counter (no CUDA; wall time, NOT contention-free)"
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1000.0)
        payload.update(_summarize(samples))
        return payload

    payload["timer"] = "cuda events"
    payload["sm_mhz_start"] = _sm_clock_mhz()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start_event.record()
        fn()
        end_event.record()
        if sync_each:
            torch.cuda.synchronize()
            samples.append(start_event.elapsed_time(end_event))
    if not sync_each:
        torch.cuda.synchronize()
        samples.append(start_event.elapsed_time(end_event) / max(repeats, 1))

    payload.update(_summarize(samples))
    # Deterministic: another process cannot move this process's own high-water mark.
    payload["peak_mb"] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    payload["peak_reserved_mb"] = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    payload["sm_mhz_end"] = _sm_clock_mhz()
    start_clock, end_clock = payload["sm_mhz_start"], payload["sm_mhz_end"]
    if start_clock and end_clock and end_clock < start_clock * 0.9:
        payload["throttled"] = True
        payload["throttle_note"] = (
            f"SM clock fell {start_clock} -> {end_clock} MHz during the run; "
            "the median is of a throttled device"
        )
    return payload


def timed_h2d(tensor_bytes: int, *, repeats: int = 50) -> dict[str, Any]:
    """Host→device copy cost, timed separately because PCIe is shared.

    Kernel time on a leased GPU is insensitive to what else the box is doing;
    the transfer over a shared bus is not. Folding the two together would hide
    exactly the term that contention moves.
    """
    try:
        import torch
    except ImportError:
        return {"skipped": "torch not importable"}
    if not torch.cuda.is_available():
        return {"skipped": "no CUDA device"}

    host = torch.empty(max(tensor_bytes // 4, 1), dtype=torch.float32).pin_memory()
    device = torch.device("cuda")
    stats = timed_call(lambda: host.to(device, non_blocking=True), warmup=5, repeats=repeats)
    stats["bytes"] = tensor_bytes
    if stats.get("median_ms"):
        stats["gb_per_s"] = round(tensor_bytes / (stats["median_ms"] / 1000.0) / 1e9, 2)
    return stats
