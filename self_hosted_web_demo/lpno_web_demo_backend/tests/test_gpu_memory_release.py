"""GPU memory must not grow with use.

Two independent retainers, both measured on the live backend 2026-08-31 after it
reached 11.4 GiB from an 886 MiB baseline and starved a training sharing the card:

1. A FAILED forward never freed the allocator's reserve, because the free ran on
   the success path only. One failed request took the process 1180 -> 9364 MiB,
   and the retained reserve then made the next OOM likelier.
2. Loaded models were held in unbounded dicts, so every model a viewer selected
   pinned its weights AND its lazily built neighbour graphs for the life of the
   process. empty_cache() cannot touch live objects.
"""

from __future__ import annotations

import gc
import weakref
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.model_registry import evict_to_cap
from app.service import BackendInferenceError, InferenceService


class _BoomAdapter:
    """Steady adapter whose forward always raises, standing in for a CUDA OOM."""

    def infer(self, *args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def _service(adapter, **overrides):
    overrides.setdefault("use_stub_runtime", True)
    settings = Settings(startup_load_runtime=False, **overrides)
    return InferenceService(
        settings=settings,
        adapter=adapter,
        runtime=SimpleNamespace(h_norm_denom=1.0, ctx=None),
    )


# --------------------------------------------------------------------------- #
# 1. free on failure
# --------------------------------------------------------------------------- #

def test_failed_steady_inference_still_frees_gpu_cache(monkeypatch) -> None:
    service = _service(_BoomAdapter())
    freed: list[int] = []
    monkeypatch.setattr(InferenceService, "_free_gpu_cache", lambda self: freed.append(1))

    with pytest.raises(BackendInferenceError):
        service._predict_steady_cached("m", 100.0, 0.5)

    assert freed, "a failed forward must still return the allocator reserve"


def test_successful_steady_inference_frees_gpu_cache_once(monkeypatch) -> None:
    service = _service(SimpleNamespace(infer=lambda *a, **k: {"T": [[[0.0]]]}))
    freed: list[int] = []
    monkeypatch.setattr(InferenceService, "_free_gpu_cache", lambda self: freed.append(1))

    service._predict_steady_cached("m", 100.0, 0.5)
    assert len(freed) == 1

    # A cache HIT does no GPU work, so it must not pay for a free.
    service._predict_steady_cached("m", 100.0, 0.5)
    assert len(freed) == 1


# --------------------------------------------------------------------------- #
# 2. bounded model caches
# --------------------------------------------------------------------------- #

def test_evict_to_cap_drops_least_recently_used() -> None:
    cache: "OrderedDict[str, int]" = OrderedDict(a=1, b=2, c=3)
    cache.move_to_end("a")  # a is now the most recently used
    assert evict_to_cap(cache, 2) == ["b"]
    assert list(cache) == ["c", "a"]


def test_evict_to_cap_keeps_at_least_one() -> None:
    cache: "OrderedDict[str, int]" = OrderedDict(a=1, b=2)
    evict_to_cap(cache, 0)
    assert len(cache) == 1, "a cap of 0 would evict the entry being served"


def test_steady_runtimes_are_capped(tmp_path) -> None:
    for name in ("m0", "m1", "m2"):
        (tmp_path / name).mkdir()

    loaded: list[str] = []

    class _LoadAdapter:
        def build_runtime(self, settings, *, run_dir, checkpoint_path, sim_params_path):
            loaded.append(run_dir.name)
            return SimpleNamespace(name=run_dir.name, ctx=None)

    service = _service(
        _LoadAdapter(),
        use_stub_runtime=False,
        steady_models_dir=str(tmp_path),
        model_cache_entries=2,
    )
    for name in ("m0", "m1", "m2"):
        service._steady_runtime_for(name)

    assert loaded == ["m0", "m1", "m2"]
    assert list(service._steady_runtimes) == ["m1", "m2"], "m0 should have been evicted"


def test_evicted_steady_runtime_is_actually_collectable(tmp_path) -> None:
    """The cap is worthless if something still references the evicted runtime.

    The steady prediction cache used to store (preds, runtime), so up to
    `prediction_cache_entries` runtimes stayed reachable no matter how small the
    model cap was -- the dict shrank and not one byte of GPU memory came back.
    """
    for name in ("m0", "m1", "m2"):
        (tmp_path / name).mkdir()

    class _Runtime:  # SimpleNamespace does not support weak references
        def __init__(self, name: str) -> None:
            self.name = name
            self.ctx = None

    class _LoadAdapter:
        def build_runtime(self, settings, *, run_dir, checkpoint_path, sim_params_path):
            return _Runtime(run_dir.name)

        def infer(self, *args, **kwargs):
            return {"T": [[[0.0]]]}

    service = _service(
        _LoadAdapter(),
        use_stub_runtime=False,
        steady_models_dir=str(tmp_path),
        model_cache_entries=2,
    )
    # Run a prediction through m0 so it lands in the prediction cache too.
    service._predict_steady_cached("m0", 100.0, 0.5)
    dead = weakref.ref(service._steady_runtimes["m0"])

    for name in ("m1", "m2"):
        service._predict_steady_cached(name, 100.0, 0.5)

    assert "m0" not in service._steady_runtimes
    gc.collect()
    assert dead() is None, "evicted runtime is still referenced; no GPU memory is freed"


def test_failed_inference_does_not_retain_the_forward_traceback(monkeypatch) -> None:
    """Freeing on failure is not enough on its own.

    The exception's traceback holds every frame of the failed forward, so every
    intermediate CUDA tensor stays reachable and empty_cache() -- which only
    returns allocator slack, never live objects -- frees nothing. Measured on the
    live backend 2026-08-31: with the traceback retained, five failed requests
    walked the process 886 -> 10204 MiB and kept climbing; with it dropped, the
    same five sat flat at ~6000 MiB.
    """
    service = _service(_BoomAdapter())

    with pytest.raises(BackendInferenceError) as caught:
        service._predict_steady_cached("m", 100.0, 0.5)

    # The cause is still reachable for debugging, but stripped of its frames.
    original = caught.value.__context__
    assert original is not None, "the underlying error should still be reachable"
    assert original.__traceback__ is None, "forward frames still held; GPU memory stays pinned"
    # ...and the message must still name what actually failed.
    assert "CUDA out of memory" in str(caught.value)


# --------------------------------------------------------------------------- #
# 3. proactive headroom: make room BEFORE a big forward
# --------------------------------------------------------------------------- #

def test_headroom_guard_evicts_when_free_vram_is_low(tmp_path, monkeypatch) -> None:
    """Capping the caches bounds growth; it does not guarantee room to RUN.

    Measured 2026-09-02 on the live demo: the backend sat at a bounded ~9 GiB of
    cached models on a card also holding an 11.9 GiB training, leaving 2.28 GiB
    free -- and the steady forward needs to allocate 4.64 GiB, so every /infer
    returned 500. Cleaning up after a forward is too late; room has to be made
    before it.
    """
    for name in ("m0", "m1", "m2"):
        (tmp_path / name).mkdir()

    class _LoadAdapter:
        def build_runtime(self, settings, *, run_dir, checkpoint_path, sim_params_path):
            return SimpleNamespace(name=run_dir.name, ctx=None)

        def infer(self, *args, **kwargs):
            return {"T": [[[0.0]]]}

    service = _service(
        _LoadAdapter(), use_stub_runtime=False,
        steady_models_dir=str(tmp_path), model_cache_entries=8, min_free_vram_gib=6.0,
    )
    for name in ("m0", "m1", "m2"):
        service._steady_runtime_for(name)
    assert len(service._steady_runtimes) == 3  # cap is deliberately not the limit here

    # Plenty free -> guard must not disturb a warm cache.
    monkeypatch.setattr("app.gpu.free_vram_bytes", lambda: 20 * 1024**3)
    service._ensure_vram_headroom("m2")
    assert len(service._steady_runtimes) == 3

    # Under the floor -> evict everything except the model about to run.
    monkeypatch.setattr("app.gpu.free_vram_bytes", lambda: 2 * 1024**3)
    service._ensure_vram_headroom("m2")
    assert list(service._steady_runtimes) == ["m2"], "the in-use model must survive"


def test_headroom_guard_is_disabled_at_zero(tmp_path, monkeypatch) -> None:
    for name in ("m0", "m1"):
        (tmp_path / name).mkdir()

    class _LoadAdapter:
        def build_runtime(self, settings, *, run_dir, checkpoint_path, sim_params_path):
            return SimpleNamespace(name=run_dir.name, ctx=None)

    service = _service(
        _LoadAdapter(), use_stub_runtime=False,
        steady_models_dir=str(tmp_path), model_cache_entries=8, min_free_vram_gib=0.0,
    )
    for name in ("m0", "m1"):
        service._steady_runtime_for(name)
    monkeypatch.setattr("app.gpu.free_vram_bytes", lambda: 0)
    service._ensure_vram_headroom("m1")
    assert len(service._steady_runtimes) == 2  # 0 disables the guard entirely


def test_free_vram_bytes_is_none_without_cuda(monkeypatch) -> None:
    """The guard must degrade to a no-op, never crash a request, off-GPU."""
    from app import gpu
    monkeypatch.setattr(gpu, "_torch_cuda_mem_get_info", lambda: None)
    assert gpu.free_vram_bytes() is None


# --------------------------------------------------------------------------- #
# 4. idle release: hand the GPU back to trainings when nobody is using the demo
# --------------------------------------------------------------------------- #

def _loaded_service(tmp_path, **overrides):
    for name in ("m0", "m1"):
        (tmp_path / name).mkdir()

    class _LoadAdapter:
        def build_runtime(self, settings, *, run_dir, checkpoint_path, sim_params_path):
            return SimpleNamespace(name=run_dir.name, ctx=None)

        def infer(self, *args, **kwargs):
            return {"T": [[[0.0]]]}

    service = _service(
        _LoadAdapter(), use_stub_runtime=False, steady_models_dir=str(tmp_path),
        model_cache_entries=8, **overrides,
    )
    for name in ("m0", "m1"):
        service._steady_runtime_for(name)
    return service


def test_idle_release_frees_everything_after_the_ttl(tmp_path) -> None:
    """An idle demo must not sit on GPU a training could use.

    The headroom guard only fires when a request arrives, so an idle backend
    holds its cached models indefinitely -- fine for the demo, not fine for a
    card shared with trainings.
    """
    service = _loaded_service(tmp_path, model_idle_ttl_s=600.0)
    service.mark_activity(now=1000.0)
    assert len(service._steady_runtimes) == 2

    # Well inside the TTL -> keep the cache warm.
    assert service.idle_release_if_stale(now=1000.0 + 599.0) == []
    assert len(service._steady_runtimes) == 2

    # Past it -> drop everything (no model is "in use" when idle).
    dropped = service.idle_release_if_stale(now=1000.0 + 601.0)
    assert set(dropped) == {"m0", "m1"}
    assert not service._steady_runtimes
    # ...and it must not keep re-releasing an already-empty cache.
    assert service.idle_release_if_stale(now=1000.0 + 5000.0) == []


def test_idle_release_disabled_at_zero(tmp_path) -> None:
    service = _loaded_service(tmp_path, model_idle_ttl_s=0.0)
    service.mark_activity(now=1000.0)
    assert service.idle_release_if_stale(now=1e9) == []
    assert len(service._steady_runtimes) == 2


def test_idle_release_skips_while_a_forward_holds_the_gpu(tmp_path) -> None:
    """Never yank a model out from under a running forward."""
    service = _loaded_service(tmp_path, model_idle_ttl_s=1.0)
    service.mark_activity(now=1000.0)
    import asyncio

    async def _hold():
        await service._gpu_lock.acquire()
        try:
            return service.idle_release_if_stale(now=1e9)
        finally:
            service._gpu_lock.release()

    assert asyncio.new_event_loop().run_until_complete(_hold()) == []
    assert len(service._steady_runtimes) == 2


def test_clear_cache_endpoint_releases_and_reports() -> None:
    """The manual lever behind the UI button / a training that needs the card."""
    from fastapi.testclient import TestClient
    from app.main import create_app

    app = create_app(settings=Settings(startup_load_runtime=True, use_stub_runtime=True))
    with TestClient(app) as client:
        body = client.post("/cache/clear")
        assert body.status_code == 200
        data = body.json()
        assert data["status"] == "ok"
        assert isinstance(data["released"], list)
        assert "free_vram_gib_before" in data and "free_vram_gib_after" in data


# --------------------------------------------------------------------------- #
# 5. last resort: an OOM releases everything and retries once
# --------------------------------------------------------------------------- #

class _OOMOnceAdapter:
    """Fails the first forward with an OOM, succeeds on the retry."""

    def __init__(self) -> None:
        self.calls = 0

    def build_runtime(self, settings, *, run_dir, checkpoint_path, sim_params_path):
        return SimpleNamespace(name=run_dir.name, ctx=None)

    def infer(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 4.64 GiB")
        return {"T": [[[0.0]]]}


def test_oom_releases_everything_and_retries_once(tmp_path) -> None:
    """The headroom guard fires on a threshold; an OOM is the ground truth.

    When a forward OOMs anyway, dropping every cached model and retrying is free
    when the cache is small and decisive when it is not -- and it needs no
    estimate of how much the forward will want.
    """
    for name in ("m0", "m1"):
        (tmp_path / name).mkdir()
    adapter = _OOMOnceAdapter()
    service = _service(
        adapter, use_stub_runtime=False, steady_models_dir=str(tmp_path),
        model_cache_entries=8, min_free_vram_gib=0.0,
    )
    service._steady_runtime_for("m0")
    service._steady_runtime_for("m1")

    preds, _axes, _h = service._predict_steady_cached("m1", 100.0, 0.5)
    assert preds == {"T": [[[0.0]]]}
    assert adapter.calls == 2, "the forward should have been retried exactly once"


def test_non_oom_failure_is_not_retried(tmp_path) -> None:
    """Only OOM justifies the retry; a genuine bug must fail fast and loudly."""
    class _Boom:
        def __init__(self): self.calls = 0
        def build_runtime(self, settings, **kw): return SimpleNamespace(ctx=None)
        def infer(self, *a, **k):
            self.calls += 1
            raise ValueError("bad channel")

    adapter = _Boom()
    service = _service(adapter, use_stub_runtime=True, min_free_vram_gib=0.0)
    with pytest.raises(BackendInferenceError):
        service._predict_steady_cached("m", 100.0, 0.5)
    assert adapter.calls == 1
