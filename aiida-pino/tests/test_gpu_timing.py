"""Per-query cost that a busy box cannot move.

Measured on a two-GPU machine with both TITAN RTXs already at 100%/78%, timing a
1024² matmul quiet, then under 16 CPU burners, then quiet again:

    baseline             median 0.1782 ms   IQR 0.0078   max 0.1992   peak 16 MB
    16 CPU burners       median 0.1802 ms   IQR 0.0136   max 3.3708   peak 16 MB
    after, quiet again   median 0.1782 ms   IQR 0.0118   max 0.1966   peak 16 MB

    median spread across all three: 1.1%
    peak_mb identical in all three: True

So the median is contention-independent to about a percent, peak allocator
memory is exactly invariant, and the **max blows up 17×** under load — which is
what a mean would have absorbed and misreported as "the model got slower".
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.gpu_timing import MIN_USEFUL_REPEATS, _summarize, timed_call


def test_the_median_ignores_a_single_huge_outlier():
    """The statistic has to be the one that survives launch-gap jitter."""
    stats = _summarize([1.0] * 99 + [3000.0])
    assert stats["median_ms"] == 1.0
    assert stats["max_ms"] == 3000.0, "the outlier is reported, not hidden"
    assert stats["iqr_ms"] == 0.0


def test_too_few_repeats_is_said_out_loud():
    stats = timed_call(lambda: None, warmup=0, repeats=2)
    assert "warning" in stats and str(MIN_USEFUL_REPEATS) in stats["warning"]


def test_the_fallback_admits_it_is_measuring_something_else():
    """Without CUDA this is wall time, which is not the same quantity."""
    import sys

    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        pytest.skip("CUDA present; the fallback path is exercised on CPU-only boxes")
    stats = timed_call(lambda: sum(range(100)), warmup=1, repeats=MIN_USEFUL_REPEATS)
    assert stats["device"] == "cpu"
    assert "NOT contention-free" in stats["timer"]


@pytest.mark.parametrize("repeats", [MIN_USEFUL_REPEATS, 20])
def test_it_returns_a_usable_shape_on_whatever_device_exists(repeats):
    stats = timed_call(lambda: sum(range(1000)), warmup=2, repeats=repeats)
    for key in ("median_ms", "p25_ms", "p75_ms", "iqr_ms", "min_ms", "max_ms", "device", "timer"):
        assert key in stats, f"{key} missing"
    assert stats["min_ms"] <= stats["median_ms"] <= stats["max_ms"]


def test_peak_memory_is_reported_when_cuda_is_there():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    tensor = torch.randn(256, 256, device="cuda")
    stats = timed_call(lambda: tensor @ tensor, warmup=3, repeats=MIN_USEFUL_REPEATS)
    assert stats["device"] == "cuda" and stats["timer"] == "cuda events"
    # Deterministic: this process's own high-water mark, which no other process
    # on the box can move.
    assert stats["peak_mb"] >= 0
    assert stats["peak_reserved_mb"] >= stats["peak_mb"]
