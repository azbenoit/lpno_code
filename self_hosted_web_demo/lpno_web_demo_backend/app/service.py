from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from .adapter_randompath import PathCase, PathInputError, RandomPathAdapter
from .adapter_straightpath import StraightCase, StraightPathAdapter
from .config import Settings
from .logging_utils import emit_json_log
from .model_registry import discover_models, evict_to_cap
from .render import (
    MELT_CONTOUR_K,
    PathFrameRenderer,
    RenderError,
    render_3d_block_frames,
    render_3d_surface_png,
    render_steady_gt_compare_3d_png,
    render_steady_gt_compare_png,
    render_xz_cross_section_png,
)
from . import steady_gt
from .steady_gt import load_steady_gt
from .schemas import (
    ClampedInputs,
    DerivedValues,
    GtCatalogEntry,
    GtCompareResultMeta,
    InferGtCompareRequest,
    InferGtCompareResponse,
    InferPathRender3DRequest,
    InferPathRender3DResponse,
    InferPathRequest,
    InferPathResponse,
    InferRequest,
    InferResponse,
    InferStraightRender3DRequest,
    InferStraightRender3DResponse,
    InferStraightRequest,
    InferStraightResponse,
    PathMeta,
    PathRender3DResultMeta,
    PathResultMeta,
    PathSpec,
    PlotPayload,
    PresetEntry,
    ResultMeta,
    StraightMeta,
    TimingMs,
)

logger = logging.getLogger(__name__)


def _substrate_mask(z_axis: Any, shape: tuple[int, ...]) -> Any:
    """Boolean mask selecting the SUBSTRATE slices -- every z <= 0 plane -- of a
    volume whose last axis is z. This is the paper's substrate region
    (`lpno` `z_region_indices`: z1d <= eps), so the number the
    pages show is the paper's substrate MAE and is comparable across models that
    serve different z extents (the straight RIGNO's 30-plane crop against the
    40-plane FNO/GAOT: the gas planes it drops are exactly the ones this mask
    drops for everyone)."""
    import numpy as np

    z = np.asarray(z_axis, dtype=np.float64)
    sel = z <= 1e-9
    if sel.size != shape[-1] or not sel.any():
        return np.ones(shape, dtype=bool)          # no usable z axis: whole volume
    return np.broadcast_to(sel, shape)


def _same_grid_metrics(
    pred: dict[str, Any], gt: dict[str, Any], field: str, z_axis: Any = None
) -> tuple[float, float] | tuple[None, None]:
    """Substrate MAE and RMSE of ``field`` between pred and GT on the same grid
    (z <= 0 slices when ``z_axis`` is given, else the whole volume).
    Returns (None, None) if the shapes disagree (e.g. stub GT)."""
    import numpy as np

    p = np.asarray(pred[field], dtype=np.float64)
    g = np.asarray(gt[field], dtype=np.float64)
    if p.shape != g.shape:
        return None, None
    d = (p - g)[_substrate_mask(z_axis, p.shape)] if z_axis is not None else (p - g)
    return float(np.abs(d).mean()), float(np.sqrt(np.mean(d * d)))


def _steady_resampled_metrics(
    pred_vol: Any,
    pred_axes: Any,
    gt_vol: Any,
    gt_axes: Any,
) -> tuple[float, float] | tuple[None, None]:
    """Substrate MAE/RMSE for the steady field: the GT is trilinearly interpolated
    from its own (x, y, z) axes onto the prediction grid and compared over the
    substrate slices (z <= 0) the GT covers (prediction points outside the GT
    extent are excluded). Works for a
    GT grid coarser than the prediction, e.g. native 10 µm GT vs a super-res
    prediction — no per-axis shape match required."""
    import numpy as np

    from .render import _interp_gt_to_grid

    pred_vol = np.asarray(pred_vol, dtype=np.float64)
    if any(np.asarray(a).size < 2 for a in gt_axes):
        return None, None
    gt_on_pred = _interp_gt_to_grid(gt_vol, gt_axes, pred_axes, fill=np.nan, dtype=np.float64)
    if gt_on_pred.shape != pred_vol.shape:
        return None, None

    mask = ~np.isnan(gt_on_pred)  # keep only prediction points inside the GT extent
    mask &= _substrate_mask(pred_axes[2], pred_vol.shape)   # and only the substrate (z <= 0)
    if not mask.any():
        return None, None
    d = pred_vol[mask] - gt_on_pred[mask]
    return float(np.abs(d).mean()), float(np.sqrt(np.mean(d * d)))


class BusyError(RuntimeError):
    pass


class InferenceTimeoutError(RuntimeError):
    pass


class ServiceNotReadyError(RuntimeError):
    pass


class BackendInferenceError(RuntimeError):
    pass


@dataclass(slots=True)
class PathPredictionCacheEntry:
    preds: dict[str, Any]
    centres: Any


class RuntimeAdapterProtocol(Protocol):
    def build_runtime(self, settings: Settings, **kwargs: Any) -> Any: ...
    def infer(
        self, runtime: Any, p_watts: float, v_scan_mps: float, x_shift_m: float = 0.0,
    ) -> dict[str, Any]: ...
    def runtime_summary(self, runtime: Any) -> dict[str, Any]: ...


@dataclass(slots=True)
class InferenceService:
    settings: Settings
    adapter: RuntimeAdapterProtocol
    runtime: Any | None = None
    path_adapter: RandomPathAdapter | None = None
    straight_adapter: StraightPathAdapter | None = None
    _admission_lock: asyncio.Lock = field(init=False, repr=False)
    _gpu_lock: asyncio.Lock = field(init=False, repr=False)
    # Bounded LRU caches keyed to include the model id, so comparing several
    # models on one input keeps every model's prediction and switching back
    # re-renders without re-inferring (see prediction_cache_entries). Steady is
    # keyed by (model, p, v); path/straight by (model, input-hash).
    _steady_prediction_cache: "OrderedDict[tuple[str, float, float, float], tuple[dict[str, Any], tuple[Any, Any, Any], float | None]]" = field(
        init=False, repr=False, default_factory=OrderedDict
    )
    _path_prediction_cache: "OrderedDict[tuple[str, tuple[float, float], bytes], PathPredictionCacheEntry]" = field(
        init=False, repr=False, default_factory=OrderedDict
    )
    _straight_prediction_cache: "OrderedDict[tuple[str, bytes], PathPredictionCacheEntry]" = field(
        init=False, repr=False, default_factory=OrderedDict
    )
    # Lazily-loaded non-default steady runtimes, keyed by model id (the default
    # model stays on self.runtime). Guarded by a load lock: the GPU lock is
    # released on a timeout while the worker thread keeps running.
    _steady_runtimes: "OrderedDict[str, Any]" = field(init=False, repr=False, default_factory=OrderedDict)
    _steady_load_lock: "threading.Lock" = field(init=False, repr=False)
    _last_activity: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self._admission_lock = asyncio.Lock()
        self._gpu_lock = asyncio.Lock()
        self._steady_load_lock = threading.Lock()
        self._last_activity = time.monotonic()

    def _cache_get_lru(
        self, cache: "OrderedDict[Any, Any]", key: Any
    ) -> Any | None:
        entry = cache.get(key)
        if entry is not None:
            cache.move_to_end(key)
        return entry

    def _cache_put_lru(self, cache: "OrderedDict[Any, Any]", key: Any, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max(1, int(self.settings.prediction_cache_entries)):
            cache.popitem(last=False)

    def _cached_forward(self, cache: "OrderedDict[Any, Any]", key: Any, keep: str | None, run: Any) -> Any:
        """Serve ``key`` from ``cache`` or compute it with ``run()`` and store it.

        The one place the miss path lives for the steady, path and straight
        routes: record activity (idle TTL), make VRAM room before the forward
        (dropping every cached model but ``keep``), and return the allocator's
        reserved-but-unused pool afterwards -- on failure too, since a forward
        that RAISES has already allocated and an OOM leaves the largest working
        set of all. Errors propagate unchanged; ``run`` owns its error contract
        (the steady route converts to BackendInferenceError and drops the
        traceback, the path routes re-raise so PathInputError stays a 422).
        A hit does no GPU work and so frees nothing.
        """
        hit = self._cache_get_lru(cache, key)
        if hit is not None:
            return hit
        self.mark_activity()
        self._ensure_vram_headroom(keep)
        try:
            value = run()
        except BaseException:
            self._free_gpu_cache()
            raise
        self._free_gpu_cache()
        self._cache_put_lru(cache, key, value)
        return value

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"

    def _infer_retrying_on_oom(self, run):
        """Run a forward; on OOM drop every cached model and retry exactly once.

        The headroom guard fires on a threshold, which is a guess. An OOM is the
        ground truth, and by then the only lever left is to give back everything
        that is not the forward itself. Free when the cache is small, decisive
        when it is not, and it needs no estimate of the forward's peak. Retried
        once only: a second OOM means the forward genuinely does not fit, and
        looping would just burn the request's timeout.
        """
        try:
            return run()
        except Exception as exc:  # noqa: BLE001
            if not self._is_oom(exc):
                raise
            exc.__traceback__ = None  # frames hold the failed forward's tensors
            dropped = self.release_all_models()
            logger.warning("Forward OOMed; released %s and retrying once", dropped or "nothing")
            return run()

    def mark_activity(self, now: float | None = None) -> None:
        """Record that inference happened, restarting the idle clock."""
        self._last_activity = time.monotonic() if now is None else float(now)

    def release_all_models(self) -> list[str]:
        """Drop every cached model everywhere and return the allocator's pool."""
        dropped = list(self._steady_runtimes)
        self._steady_runtimes.clear()
        for adapter in (self.path_adapter, self.straight_adapter):
            if adapter is not None and hasattr(adapter, "release_models"):
                dropped += adapter.release_models(None)
        if dropped:
            self._free_gpu_cache()
        return dropped

    def idle_release_if_stale(self, now: float | None = None) -> list[str]:
        """Release cached models once the demo has been idle past the TTL.

        Returns the ids dropped (empty when the TTL is disabled, not yet
        reached, nothing is cached, or a forward is in flight). Called by the
        background sweeper; safe to call from anywhere.
        """
        ttl = float(self.settings.model_idle_ttl_s)
        if ttl <= 0:
            return []
        if self._gpu_lock.locked():
            return []  # a forward is running -- never pull a model out from under it
        now = time.monotonic() if now is None else float(now)
        if now - self._last_activity <= ttl:
            return []
        dropped = self.release_all_models()
        if dropped:
            logger.info("Idle %.0fs > TTL %.0fs: released %s", now - self._last_activity, ttl, dropped)
        return dropped

    def _ensure_vram_headroom(self, keep: str | None = None) -> None:
        """Make room BEFORE a forward when the device is nearly full.

        Capping the caches bounds growth; it does not guarantee room to RUN.
        Measured 2026-09-02: a bounded ~9 GiB of cached models shared a card with
        an 11.9 GiB training, leaving 2.28 GiB free against a steady forward that
        allocates 4.64 GiB -- so every /infer returned 500 while the caches were
        behaving exactly as designed. Note the 8.95 GiB was ALLOCATED, not merely
        reserved, so empty_cache() could not have reclaimed any of it; only
        dropping the live models can.

        Below the floor, everything except the model about to run is dropped.
        Deliberately blunt: under pressure, serving the current request matters
        more than a warm cache, and a reload is seconds.
        """
        floor_gib = float(self.settings.min_free_vram_gib)
        if floor_gib <= 0 or self.settings.use_stub_runtime:
            return
        from .gpu import empty_cache, free_vram_bytes

        free = free_vram_bytes()
        if free is None or free >= floor_gib * 1024**3:
            return  # unknown (no CUDA) or comfortable -> leave the cache warm
        dropped = [m for m in list(self._steady_runtimes) if m != keep]
        for m in dropped:
            self._steady_runtimes.pop(m, None)
        for adapter in (self.path_adapter, self.straight_adapter):
            if adapter is not None and hasattr(adapter, "release_models"):
                dropped += adapter.release_models(keep)
        empty_cache()
        after = free_vram_bytes()
        logger.info(
            "VRAM headroom guard: %.2f GiB free < %.2f floor; dropped %s -> %.2f GiB free",
            free / 1024**3, floor_gib, dropped or "nothing",
            (after or 0) / 1024**3,
        )

    def _free_gpu_cache(self) -> None:
        """Return the CUDA allocator's reserved-but-unused memory after a forward,
        so the footprint stays ~resident weights + one working set instead of the
        summed peak of every model that has run (see gpu.empty_cache). Called only
        on a prediction-cache miss (a hit does no GPU work). No-op in stub mode."""
        if self.settings.use_stub_runtime:
            return
        from .gpu import empty_cache

        empty_cache()

    @classmethod
    def build_default(cls, settings: Settings) -> "InferenceService":
        path_adapter = RandomPathAdapter(settings, stub=settings.use_stub_runtime)
        straight_adapter = StraightPathAdapter(settings, stub=settings.use_stub_runtime)
        if settings.use_stub_runtime:
            runtime = SimpleNamespace(h_norm_denom=1.0, device="stub", output_keys=["T", "alpha", "fl"], input_keys=[])
            return cls(
                settings=settings,
                adapter=_StubAdapter(),
                runtime=runtime,
                path_adapter=path_adapter,
                straight_adapter=straight_adapter,
            )

        from .adapter_pino import PinoRuntimeAdapter

        adapter = PinoRuntimeAdapter()
        runtime = adapter.build_runtime(settings)
        return cls(
            settings=settings,
            adapter=adapter,
            runtime=runtime,
            path_adapter=path_adapter,
            straight_adapter=straight_adapter,
        )

    def health(self) -> dict[str, Any]:
        return {
            "runtime_loaded": self.runtime is not None,
            "busy": self._gpu_lock.locked(),
            "worker_policy": "single-worker-required",
            "adapter": type(self.adapter).__name__,
            "runtime": self.adapter.runtime_summary(self.runtime) if self.runtime is not None else None,
        }

    async def infer(self, request_model: InferRequest, *, server_request_id: str) -> InferResponse:
        if self.runtime is None:
            raise ServiceNotReadyError("Inference runtime has not been initialized")

        async with self._admission_lock:
            if self._gpu_lock.locked():
                raise BusyError("Inference runtime is busy")
            await self._gpu_lock.acquire()

        total_start = time.perf_counter()
        try:
            timeout_s = self.settings.infer_timeout_seconds(request_model.mode)
            response = await asyncio.wait_for(
                asyncio.to_thread(self._infer_sync, request_model, server_request_id, total_start),
                timeout=timeout_s,
            )
            return response
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(
                f"Inference exceeded timeout of {timeout_s:.3f}s (note: in-process GPU work is not preempted in v1)"
            ) from exc
        finally:
            self._gpu_lock.release()

    def _infer_sync(self, request_model: InferRequest, server_request_id: str, total_start: float) -> InferResponse:
        self._validate_steady_model(request_model.model)
        p_clamped, v_clamped = self.settings.clamp_inputs(
            request_model.inputs.p_watts,
            request_model.inputs.v_scan_mps,
        )
        if abs(v_clamped) < 1e-12:
            raise BackendInferenceError("Vscan must be non-zero after clamping")

        width_px, height_px = self.settings.clamp_render_size(
            request_model.render.width_px,
            request_model.render.height_px,
        )

        infer_start = time.perf_counter()
        preds, pred_axes, h_norm_denom = self._predict_steady_cached(
            request_model.model, p_clamped, v_clamped
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        render_start = time.perf_counter()
        try:
            if request_model.view.plot_mode == "3d":
                rendered = render_3d_surface_png(
                    preds=preds,
                    field=request_model.view.field,
                    mesh_size_um=self.settings.mesh_size_um,
                    width_px=width_px,
                    height_px=height_px,
                    smoothing_iterations=0 if request_model.mode == "preview" else 20,
                    plane=request_model.view.plane,
                )
            else:
                rendered = render_xz_cross_section_png(
                    preds=preds,
                    field=request_model.view.field,
                    y_index=request_model.view.y_index,
                    mesh_size_um=self.settings.mesh_size_um,
                    show_t_contour=request_model.view.show_t_contour,
                    t_contour_k=request_model.view.t_contour_k,
                    width_px=width_px,
                    height_px=height_px,
                    plane=request_model.view.plane,
                )
        except RenderError as exc:
            raise BackendInferenceError(f"Render failed: {exc}") from exc
        render_ms = (time.perf_counter() - render_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        h_model = float(p_clamped) / (abs(float(v_clamped)) ** 0.5)
        h_star = self.settings.compute_hstar(p_clamped, v_clamped)
        h_norm_model = None
        if h_norm_denom is not None:
            try:
                denom = float(h_norm_denom)
                if abs(denom) > 1e-30:
                    h_norm_model = h_model / denom
            except Exception:  # noqa: BLE001
                h_norm_model = None

        png_b64 = base64.b64encode(rendered.png_bytes).decode("ascii")
        response = InferResponse(
            client_request_id=request_model.client_request_id,
            client_seq=request_model.client_seq,
            mode=request_model.mode,
            server_request_id=server_request_id,
            clamped_inputs=ClampedInputs(p_watts=p_clamped, v_scan_mps=v_clamped),
            derived=DerivedValues(h_model=h_model, h_star=h_star, h_norm_model=h_norm_model),
            result_meta=ResultMeta(
                field=request_model.view.field,
                plot_mode_used=request_model.view.plot_mode,
                grid_shape_xyz=rendered.grid_shape_xyz,
                y_index_used=rendered.y_index_used,
            ),
            plot=PlotPayload(
                width_px=rendered.width_px,
                height_px=rendered.height_px,
                png_base64=png_b64,
            ),
            timing_ms=TimingMs(infer=round(infer_ms, 3), render=round(render_ms, 3), total=round(total_ms, 3)),
        )

        emit_json_log(
            logger,
            logging.INFO,
            "infer_ok",
            server_request_id=server_request_id,
            client_request_id=request_model.client_request_id,
            client_seq=request_model.client_seq,
            mode=request_model.mode,
            plot_mode=request_model.view.plot_mode,
            p_watts=p_clamped,
            v_scan_mps=v_clamped,
            timing_ms=response.timing_ms.model_dump(),
        )
        return response

    def _validate_steady_model(self, model: str) -> None:
        """Reject an unknown steady model id with a 422 (PathInputError), matching
        the path/straight adapters. Empty means the startup default runtime; the
        default model id is always accepted even if discovery is momentarily
        empty."""
        if not model or model == Path(self.settings.runtime_run_dir).name:
            return
        ids = {m["id"] for m in discover_models(self.settings.steady_models_dir)}
        if model not in ids:
            raise PathInputError(
                f"Unknown steady model '{model}'. Available: {', '.join(sorted(ids)) or '(none)'}"
            )

    def _steady_runtime_for(self, model: str) -> Any:
        """The runtime that serves steady model ``model``. Empty id, stub mode,
        or the startup default model use the runtime loaded at build time; any
        other discovered model is loaded lazily from its folder and cached (FNO
        or RIGNO — the adapter reads the architecture from the folder's config)."""
        if not model or self.settings.use_stub_runtime:
            return self.runtime
        if model == Path(self.settings.runtime_run_dir).name:
            return self.runtime  # the model already loaded as self.runtime
        cached = self._steady_runtimes.get(model)
        if cached is not None:
            self._steady_runtimes.move_to_end(model)
            return cached
        with self._steady_load_lock:
            cached = self._steady_runtimes.get(model)
            if cached is not None:
                self._steady_runtimes.move_to_end(model)
                return cached
            from .model_registry import resolve_model_dir

            model_dir = Path(resolve_model_dir(self.settings.steady_models_dir, model))
            rt = self.adapter.build_runtime(
                self.settings,
                run_dir=model_dir,
                checkpoint_path=model_dir / "u_network.0.pth",
                sim_params_path=model_dir / "flow3D_params.yaml",
            )
            self._steady_runtimes[model] = rt
            self._steady_runtimes.move_to_end(model)
            logger.info("Loaded steady model '%s' from %s", model, model_dir)
            evicted = evict_to_cap(self._steady_runtimes, self.settings.model_cache_entries)
            if evicted:
                logger.info("Evicted steady model(s) %s to cap resident GPU models", evicted)
                self._free_gpu_cache()
            return rt

    def _predict_steady_cached(
        self,
        model: str,
        p_clamped: float,
        v_clamped: float,
        x_shift_m: float = 0.0,
    ) -> tuple[dict[str, Any], Any]:
        """Steady-state prediction for ``model`` at (P, V), via a bounded LRU
        keyed by (model, P, V, x_shift) so comparing models on one point keeps
        each prediction. Returns (preds, axes, h_norm_denom): the grid
        axes used to place/resample the prediction, plus the runtime's H
        normalisation. Both are extracted here so the cache never holds a
        reference to the runtime object that produced them."""
        key = (
            model or "",
            round(float(p_clamped), 6),
            round(float(v_clamped), 9),
            round(float(x_shift_m), 12),   # a shifted window is a different prediction
        )

        def run() -> tuple[dict[str, Any], tuple[Any, Any, Any], float | None]:
            try:
                runtime = self._steady_runtime_for(model)  # inside try: a bad/incomplete
                preds = self._infer_retrying_on_oom(  # folder -> clean 500
                    lambda: self.adapter.infer(runtime, p_clamped, v_clamped, x_shift_m=x_shift_m)
                )
            except Exception as exc:  # noqa: BLE001
                # Dropping the TRACEBACK is what actually returns the memory
                # (_cached_forward frees the allocator pool next) -- it holds
                # every frame of the failed forward, hence every intermediate
                # CUDA tensor, and empty_cache() cannot free live objects. With
                # the traceback retained, five failed requests still walked the
                # process 886 -> 10204 MiB (measured 2026-08-31).
                message = f"Inference failed: {type(exc).__name__}: {exc}"
                exc.__traceback__ = None
                raise BackendInferenceError(message) from None
            # Cache the axes, never the runtime: holding the runtime object here
            # kept up to prediction_cache_entries models reachable, so capping
            # the model cache freed nothing.
            axes = self._steady_pred_axes(runtime, preds)
            if x_shift_m:
                import numpy as np

                axes = (np.asarray(axes[0]) + float(x_shift_m), axes[1], axes[2])
            h_norm_denom = getattr(getattr(runtime, "ctx", runtime), "h_norm_denom", None)
            return preds, axes, h_norm_denom

        return self._cached_forward(self._steady_prediction_cache, key, model or None, run)

    async def infer_gt_compare(
        self, request_model: InferGtCompareRequest, *, server_request_id: str
    ) -> InferGtCompareResponse:
        if self.runtime is None:
            raise ServiceNotReadyError("Inference runtime has not been initialized")
        async with self._admission_lock:
            if self._gpu_lock.locked():
                raise BusyError("Inference runtime is busy")
            await self._gpu_lock.acquire()
        total_start = time.perf_counter()
        try:
            timeout_s = self.settings.infer_timeout_seconds("final")
            return await asyncio.wait_for(
                asyncio.to_thread(self._infer_gt_compare_sync, request_model, server_request_id, total_start),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(
                f"GT compare exceeded timeout of {timeout_s:.3f}s (note: in-process GPU work is not preempted in v1)"
            ) from exc
        finally:
            self._gpu_lock.release()

    def _steady_pred_axes(self, runtime: Any, preds: dict[str, Any]) -> tuple[Any, Any, Any]:
        """Physical (x, y, z) axes of the steady prediction, in metres, from the
        runtime that produced it. Real runtimes carry them on ctx; the stub has
        none, so synthesize uniform axes from the configured mesh size (only used
        to place stub panels)."""
        import numpy as np

        ctx = getattr(runtime, "ctx", None)
        x_phys = getattr(ctx, "x_phys", None) if ctx is not None else None
        if x_phys is not None:
            return ctx.x_phys, ctx.y_phys, ctx.z_phys
        nx, ny, nz = np.asarray(preds["T"]).shape[-3:]
        step = float(self.settings.mesh_size_um) * 1e-6
        return np.arange(nx) * step, np.arange(ny) * step, np.arange(nz) * step

    def _steady_window_shift(self, model: str, gt_loaded: Any) -> float:
        """Metres to translate the served x window so it matches the GT case's.

        The steady moving-frame window is a 61-cell slice of the ABSOLUTE domain
        following the beam, and its position is not the same for every case:
        across the 13 calibrated test cases x[0] runs -1.877e-4..-1.550e-4 while
        the runtime is built on exactly one window. Comparing a case that used a
        different one measures that offset -- 6.23 K substrate on the nine cases
        whose window matches, 21.00 K on the four offset by 1-3 cells (measured
        2026-09-03). Returns 0.0 whenever there is nothing to match, which is
        also the right answer for free (P, V).
        """
        if gt_loaded is None:
            return 0.0
        try:
            import numpy as np

            gt_axes = gt_loaded[1]
            runtime = self._steady_runtime_for(model)
            x_phys = getattr(getattr(runtime, "ctx", None), "x_phys", None)
            if x_phys is None:
                return 0.0
            gx = np.asarray(gt_axes[0], dtype=float)
            px = np.asarray(x_phys, dtype=float)
            if gx.size != px.size:
                # Different cell counts is a geometry mismatch, not an offset;
                # shifting would paper over it. Leave it to the interpolator.
                logger.warning(
                    "steady GT has %d x cells but the runtime serves %d; not shifting",
                    gx.size, px.size,
                )
                return 0.0
            return float(gx[0] - px[0])
        except Exception:  # noqa: BLE001 - a shift is an optimisation, never a hard failure
            logger.warning("could not compute the steady window shift; serving unshifted",
                           exc_info=True)
            return 0.0

    def _infer_gt_compare_sync(
        self, request_model: InferGtCompareRequest, server_request_id: str, total_start: float
    ) -> InferGtCompareResponse:
        self._validate_steady_model(request_model.model)
        p_clamped, v_clamped = self.settings.clamp_inputs(
            request_model.inputs.p_watts, request_model.inputs.v_scan_mps
        )
        if abs(v_clamped) < 1e-12:
            raise BackendInferenceError("Vscan must be non-zero after clamping")
        width_px, height_px = self.settings.clamp_render_size(
            request_model.render.width_px, request_model.render.height_px
        )
        field = request_model.field

        # The ground truth is resolved BEFORE the forward, because the matched
        # case decides which x window to serve on. The steady models are trained
        # on a beam-following 61-cell window of the absolute domain whose
        # position varies case to case; serving every case on the runtime's one
        # window measured the offset instead of the model (63-192 K against the
        # 7.51 K of the model's own recorded eval, 2026-09-03).
        gt_loaded = load_steady_gt(self.settings.steady_gt_data_dir, p_clamped, v_clamped)
        x_shift_m = self._steady_window_shift(request_model.model, gt_loaded)

        infer_start = time.perf_counter()
        preds, pred_axes, h_norm_denom = self._predict_steady_cached(
            request_model.model, p_clamped, v_clamped, x_shift_m=x_shift_m
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        is_3d = request_model.plot_mode == "3d"
        gt_mae = gt_rmse = None
        render_start = time.perf_counter()
        try:
            if gt_loaded is not None:
                gt_dict, gt_axes, (gt_p, gt_v), gt_split = gt_loaded
                import numpy as np

                gt_mae, gt_rmse = _steady_resampled_metrics(
                    np.asarray(preds[field])[0, 0], pred_axes, gt_dict[field], gt_axes
                )
                if is_3d:
                    rendered = render_steady_gt_compare_3d_png(
                        pred=preds,
                        gt=gt_dict,
                        pred_axes=pred_axes,
                        gt_axes=gt_axes,
                        field=field,
                        mesh_size_um=self.settings.mesh_size_um,
                        width_px=width_px,
                        height_px=height_px,
                        smoothing_iterations=20,
                        plane=request_model.plane,
                        gt_split=gt_split,
                    )
                    panel = "gt_compare_3d"
                else:
                    rendered = render_steady_gt_compare_png(
                        pred=preds,
                        gt=gt_dict,
                        pred_axes=pred_axes,
                        gt_axes=gt_axes,
                        field=field,
                        y_index=request_model.y_index,
                        width_px=width_px,
                        height_px=height_px,
                        gt_split=gt_split,
                    )
                    panel = "gt_compare"
            else:
                gt_p = gt_v = gt_split = None
                if is_3d:
                    rendered = render_3d_surface_png(
                        preds=preds,
                        field=field,
                        mesh_size_um=self.settings.mesh_size_um,
                        width_px=width_px,
                        height_px=height_px,
                        smoothing_iterations=20,
                        plane=request_model.plane,
                    )
                    panel = "single_3d"
                else:
                    rendered = render_xz_cross_section_png(
                        preds=preds,
                        field=field,
                        y_index=request_model.y_index,
                        mesh_size_um=self.settings.mesh_size_um,
                        show_t_contour=False,
                        t_contour_k=1898.0,
                        width_px=width_px,
                        height_px=height_px,
                    )
                    panel = "single"
        except RenderError as exc:
            raise BackendInferenceError(f"GT compare render failed: {exc}") from exc
        render_ms = (time.perf_counter() - render_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        response = InferGtCompareResponse(
            server_request_id=server_request_id,
            clamped_inputs=ClampedInputs(p_watts=p_clamped, v_scan_mps=v_clamped),
            result_meta=GtCompareResultMeta(
                field=field,
                gt_available=gt_loaded is not None,
                gt_p_watts=gt_p,
                gt_v_scan_mps=gt_v,
                gt_split=gt_split,
                grid_shape_xyz=rendered.grid_shape_xyz,
                y_index_used=rendered.y_index_used,
                panel=panel,
                plot_mode_used="3d" if is_3d else "2d",
                gt_mae=gt_mae,
                gt_rmse=gt_rmse,
            ),
            plot=PlotPayload(
                width_px=rendered.width_px,
                height_px=rendered.height_px,
                png_base64=base64.b64encode(rendered.png_bytes).decode("ascii"),
            ),
            timing_ms=TimingMs(infer=round(infer_ms, 3), render=round(render_ms, 3), total=round(total_ms, 3)),
        )
        emit_json_log(
            logger,
            logging.INFO,
            "infer_gt_compare_ok",
            server_request_id=server_request_id,
            client_request_id=request_model.client_request_id,
            gt_available=gt_loaded is not None,
            p_watts=p_clamped,
            v_scan_mps=v_clamped,
            timing_ms=response.timing_ms.model_dump(),
        )
        return response

    # ------------------------------------------------------------------ #
    # /infer_path (time-dependent random-path inference)
    # ------------------------------------------------------------------ #
    def path_presets(self, model: str | None = None) -> list[PresetEntry]:
        """Preset GT cases for ``model`` — its own test split when its meta.json
        declares one, else the legacy shared set. A model must only be offered the
        cases it was trained for (the variable-path model's GT is not rp3's)."""
        if self.path_adapter is None:
            raise ServiceNotReadyError("Random-path adapter is not initialized")
        return [
            PresetEntry(
                name=p["name"],
                points_xy_m=p["points_xy_m"],
                p_watts=p.get("p_watts"),
                v_scan_mps=p.get("v_scan_mps"),
            )
            for p in self.path_adapter.presets(model)
        ]

    async def infer_path(self, request_model: InferPathRequest, *, server_request_id: str) -> InferPathResponse:
        if self.path_adapter is None:
            raise ServiceNotReadyError("Random-path adapter is not initialized")
        return await self._run_path_locked(self._infer_path_sync, request_model, server_request_id)

    async def infer_path_render3d(
        self, request_model: InferPathRender3DRequest, *, server_request_id: str
    ) -> InferPathRender3DResponse:
        if self.path_adapter is None:
            raise ServiceNotReadyError("Random-path adapter is not initialized")
        return await self._run_path_locked(self._infer_path_render3d_sync, request_model, server_request_id)

    async def infer_straight(
        self, request_model: InferStraightRequest, *, server_request_id: str
    ) -> InferStraightResponse:
        if self.straight_adapter is None:
            raise ServiceNotReadyError("Straight-line adapter is not initialized")
        return await self._run_path_locked(self._infer_straight_sync, request_model, server_request_id)

    async def infer_straight_render3d(
        self, request_model: InferStraightRender3DRequest, *, server_request_id: str
    ) -> InferStraightRender3DResponse:
        if self.straight_adapter is None:
            raise ServiceNotReadyError("Straight-line adapter is not initialized")
        return await self._run_path_locked(self._infer_straight_render3d_sync, request_model, server_request_id)

    async def _run_path_locked(self, sync_fn: Any, request_model: Any, server_request_id: str) -> Any:
        """Shared admission/GPU-lock + timeout wrapper for the time-dependent routes."""
        async with self._admission_lock:
            if self._gpu_lock.locked():
                raise BusyError("Inference runtime is busy")
            await self._gpu_lock.acquire()

        total_start = time.perf_counter()
        timeout_s = self.settings.path_timeout_ms / 1000.0
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(sync_fn, request_model, server_request_id, total_start),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(
                f"Path inference exceeded timeout of {timeout_s:.3f}s (note: in-process GPU work is not preempted in v1)"
            ) from exc
        finally:
            self._gpu_lock.release()

    def _predict_path_cached(
        self,
        model: str,
        path: PathSpec,
        p_watts: float | None = None,
        v_scan_mps: float | None = None,
        frame_dt_s: float | None = None,
    ) -> tuple[PathCase, dict[str, Any], Any]:
        """Resolve the case and return (case, preds, centres), via the per-route bounded LRU (prediction_cache_entries, default 8)
        prediction cache: same model + same resolved path/times/physics skips inference."""
        import hashlib

        adapter = self.path_adapter
        assert adapter is not None
        case = adapter.resolve(
            points_xy_m=path.points_xy_m,
            preset=path.preset,
            physics=adapter.physics_for(model),
            p_watts=p_watts,
            v_scan_mps=v_scan_mps,
            model=model,
            frame_dt_s=frame_dt_s,
        )
        cache_key = (
            model,
            # effective P/V are part of the prediction identity for variable models
            (float(case.p_watts), float(case.v_scan_mps)),
            hashlib.sha256(case.points.tobytes() + case.times.tobytes()).digest(),
        )
        entry = self._cached_forward(
            self._path_prediction_cache, cache_key, model,
            lambda: PathPredictionCacheEntry(preds=adapter.infer(model, case), centres=adapter.beam_centres(case)),
        )
        return case, entry.preds, entry.centres

    @staticmethod
    def _path_meta(case: PathCase, mae: float | None = None, rmse: float | None = None) -> PathMeta:
        variable = case.physics.variable
        return PathMeta(
            n_points_used=int(case.points.shape[0]),
            path_length_m=float(case.path_length_m),
            laser_off_time_s=float(case.laser_off_time_s),
            clamped=case.clamped,
            preset=case.preset,
            gt_available=case.gt_available,
            points_xy_m=[[float(px), float(py)] for px, py in case.points],
            gt_mae=mae,
            gt_rmse=rmse,
            # applied physics is only meaningful for variable-physics models
            p_watts=float(case.p_watts) if variable else None,
            v_scan_mps=float(case.v_scan_mps) if variable else None,
            physics_clamped=bool(case.physics_clamped),
            n_frames=int(case.times.size),
            # Effective spacing: frames snap to the training-dt dose grid, so
            # this is what was actually rendered, not what was requested.
            frame_dt_s=(
                float((case.times[-1] - case.times[0]) / max(1, case.times.size - 1))
                if case.times.size > 1 else None
            ),
        )

    @staticmethod
    def _build_2d_frames(
        *,
        field: str,
        preds: dict[str, Any],
        gt: dict[str, Any] | None,
        centres: Any,
        points: Any,
        x_axis: Any,
        y_axis: Any,
        z_axis: Any,
        z_index: int,
        width_px: int,
        times: Any,
        plane: str = "xy",
        y_index: int = -1,
        gt_split: str | None = None,
    ) -> list[str]:
        """Encode the requested field's 20 2D-slice PNGs (single panel, or the
        GT/prediction/diff triple when ``gt`` is present). Shared by /infer_path
        and /infer_straight.

        ``plane`` selects the slice: "xy" is the top view at ``z_index`` (the
        vertical axis is y, the drawn path is overlaid); "xz" is a cross-section
        at ``y_index`` (vertical axis z, no path — the section is edge-on).

        Serial on purpose (Agg holds a class-level draw lock; a thread
        pool measured ~1.7x slower — see assumptions A11). One symmetric diff
        range for the whole run so the "Prediction - GT" colorbar is constant
        across every frame, not rescaled per frame.
        """
        import numpy as np

        nt = int(preds["T"].shape[0])
        # Raw served T volumes for the melt-pool contour (no fill inversion).
        t_pred_vol = np.asarray(preds["T"])
        t_gt_vol = np.asarray(gt["T"]) if gt is not None else None
        if plane == "xz":
            ny = int(preds[field].shape[2])
            yi = ny // 2 if int(y_index) < 0 else max(0, min(int(y_index), ny - 1))
            pred_slices = [preds[field][i, :, yi, :] for i in range(nt)]
            gt_slices = [gt[field][i, :, yi, :] for i in range(nt)] if gt is not None else None
            t_pred_slices = [t_pred_vol[i, :, yi, :] for i in range(nt)]
            t_gt_slices = [t_gt_vol[i, :, yi, :] for i in range(nt)] if t_gt_vol is not None else None
            vertical_axis = z_axis
            # Beam marker at (beam_x, surface z=0); the path is not drawn edge-on.
            frame_centres = [np.array([centres[i][0], 0.0]) for i in range(nt)]
            path_overlay = None
        else:
            pred_slices = [preds[field][i, :, :, z_index] for i in range(nt)]
            gt_slices = [gt[field][i, :, :, z_index] for i in range(nt)] if gt is not None else None
            t_pred_slices = [t_pred_vol[i, :, :, z_index] for i in range(nt)]
            t_gt_slices = [t_gt_vol[i, :, :, z_index] for i in range(nt)] if t_gt_vol is not None else None
            vertical_axis = y_axis
            frame_centres = [centres[i] for i in range(nt)]
            path_overlay = points

        # Floor the symmetric diff range so tiny all-zero diffs don't blow up the
        # colorbar: ~1 K for T, but ~0.01 for the 0..1 fields (alpha/fl) — a 1.0
        # floor there pins the range to +/-1 and the diff panel renders blank.
        diff_floor = 1.0 if field == "T" else 0.01
        diff_dmax = diff_floor
        if gt is not None:
            diff_dmax = max(diff_floor, float(np.abs(np.stack(pred_slices) - np.stack(gt_slices)).max()))

        renderer = PathFrameRenderer(
            field=field,
            has_gt=gt is not None,
            x_axis=x_axis,
            y_axis=vertical_axis,
            path_xy=path_overlay,
            width_px=width_px,
            diff_dmax=diff_dmax,
            gt_split=gt_split,
            plane=plane,
            contour_k=MELT_CONTOUR_K,
        )
        return [
            base64.b64encode(
                renderer.render(
                    pred_xy=pred_slices[i],
                    gt_xy=gt_slices[i] if gt_slices is not None else None,
                    centre_xy=frame_centres[i],
                    time_s=float(times[i]),
                    t_pred_xy=t_pred_slices[i],
                    t_gt_xy=t_gt_slices[i] if t_gt_slices is not None else None,
                )
            ).decode("ascii")
            for i in range(nt)
        ]

    @staticmethod
    def _guard_time_dependent(label: str, fn: Any) -> Any:
        """Run predict + ground-truth under the time-dependent error contract:
        PathInputError/FileNotFoundError pass through (422/404), anything else
        is logged server-side and the client gets only the exception type, so
        runtime filesystem paths and state-dict keys never leave the host."""
        try:
            return fn()
        except (PathInputError, FileNotFoundError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s inference failed", label)
            raise BackendInferenceError(f"{label} inference failed: {type(exc).__name__}") from exc

    def _time_dependent_2d(
        self,
        *,
        label: str,
        view: Any,
        preds: dict[str, Any],
        gt: dict[str, Any] | None,
        centres: Any,
        points: Any,
        times: Any,
        grid: tuple[Any, Any, Any],
        width_px: int,
        gt_split: str | None,
    ) -> tuple[list[str], PathResultMeta, tuple[float | None, float | None], float]:
        """2D frame series shared by /infer_path and /infer_straight: pick the z
        plane, render the requested field's frames (single panel, or the
        GT/prediction/diff triple), score it. Returns (frames, result_meta,
        (mae, rmse), render_ms)."""
        import numpy as np

        _nt, nx, ny, nz = preds["T"].shape
        x_axis, y_axis, z_axis = grid
        if view.z_index < 0:
            z_index = int(np.argmin(np.abs(z_axis)))  # grid plane nearest z=0
        else:
            z_index = min(int(view.z_index), nz - 1)
        render_start = time.perf_counter()
        try:
            # Lazy per-field rendering: only the requested field's frames. The
            # other fields are fetched on demand (the prediction cache makes
            # those render-only), so a run costs nt panels, not 3 nt.
            frames = self._build_2d_frames(
                field=view.field,
                preds=preds,
                gt=gt,
                centres=centres,
                points=points,
                times=times,
                x_axis=x_axis,
                y_axis=y_axis,
                z_axis=z_axis,
                z_index=z_index,
                width_px=width_px,
                plane=view.plane,
                y_index=view.y_index,
                gt_split=gt_split,
            )
        except Exception as exc:  # noqa: BLE001
            raise BackendInferenceError(f"{label} frame render failed: {type(exc).__name__}: {exc}") from exc
        render_ms = (time.perf_counter() - render_start) * 1000.0
        meta = PathResultMeta(
            grid_shape_xyz=(nx, ny, nz),
            z_index_used=z_index,
            field=view.field,
            panel="triple" if gt is not None else "single",
        )
        scores = _same_grid_metrics(preds, gt, view.field, z_axis=grid[2]) if gt is not None else (None, None)
        return frames, meta, scores, render_ms

    def _time_dependent_3d(
        self,
        *,
        label: str,
        preds: dict[str, Any],
        gt: dict[str, Any] | None,
        times: Any,
        grid: tuple[Any, Any, Any],
        width_px: int,
        height_px: int,
        plane: str,
        gt_split: str | None,
    ) -> tuple[list[str], PathRender3DResultMeta, tuple[float | None, float | None], float]:
        """3D block frame series shared by the two render3d routes. Returns
        (base64 frames, result_meta, (mae, rmse) on T, render_ms)."""
        _nt, nx, ny, nz = preds["T"].shape
        x_axis, y_axis, z_axis = grid
        render_start = time.perf_counter()
        try:
            frames = render_3d_block_frames(
                preds=preds,
                gt=gt,
                x_axis=x_axis,
                y_axis=y_axis,
                z_axis=z_axis,
                times_s=times,
                width_px=width_px,
                height_px=height_px,
                gt_split=gt_split,
                plane=plane,
            )
        except Exception as exc:  # noqa: BLE001
            raise BackendInferenceError(f"{label} 3D block render failed: {type(exc).__name__}: {exc}") from exc
        render_ms = (time.perf_counter() - render_start) * 1000.0
        scores = _same_grid_metrics(preds, gt, "T", z_axis=grid[2]) if gt is not None else (None, None)
        return (
            [base64.b64encode(png).decode("ascii") for png in frames],
            PathRender3DResultMeta(grid_shape_xyz=(nx, ny, nz)),
            scores,
            render_ms,
        )

    def _predict_path_with_gt(self, request_model: Any) -> tuple[PathCase, dict[str, Any], Any, dict[str, Any] | None]:
        """(case, preds, centres, gt) for either /infer_path route."""
        adapter = self.path_adapter
        assert adapter is not None

        def go():
            case, preds, centres = self._predict_path_cached(
                request_model.model, request_model.path,
                p_watts=request_model.p_watts, v_scan_mps=request_model.v_scan_mps,
                frame_dt_s=request_model.frame_dt_s,
            )
            return case, preds, centres, adapter.ground_truth(case, request_model.model)

        return self._guard_time_dependent("Path", go)

    def _infer_path_sync(
        self, request_model: InferPathRequest, server_request_id: str, total_start: float
    ) -> InferPathResponse:
        adapter = self.path_adapter
        assert adapter is not None
        # render.height_px is ignored for this route: the figure height is
        # derived from the physical domain aspect; width_px controls the size.
        width_px, _ = self.settings.clamp_render_size(
            request_model.render.width_px,
            request_model.render.height_px,
        )

        infer_start = time.perf_counter()
        case, preds, centres, gt = self._predict_path_with_gt(request_model)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        frames, result_meta, (mae, rmse), render_ms = self._time_dependent_2d(
            label="Path",
            view=request_model.view,  # honors the xz cross-section like the straight route
            preds=preds,
            gt=gt,
            centres=centres,
            points=case.points,
            times=case.times,
            grid=adapter.grid(),
            width_px=width_px,
            gt_split="test" if gt is not None else None,  # presets come from the test split
        )
        total_ms = (time.perf_counter() - total_start) * 1000.0

        response = InferPathResponse(
            model=request_model.model,
            times_s=[float(t) for t in case.times],
            frames=frames,
            path_meta=self._path_meta(case, mae, rmse),
            result_meta=result_meta,
            timing_ms=TimingMs(infer=round(infer_ms, 3), render=round(render_ms, 3), total=round(total_ms, 3)),
        )
        emit_json_log(
            logger,
            logging.INFO,
            "infer_path_ok",
            server_request_id=server_request_id,
            client_request_id=request_model.client_request_id,
            model=request_model.model,
            preset=case.preset,
            n_points=int(case.points.shape[0]),
            timing_ms=response.timing_ms.model_dump(),
        )
        return response

    def _infer_path_render3d_sync(
        self, request_model: InferPathRender3DRequest, server_request_id: str, total_start: float
    ) -> InferPathRender3DResponse:
        adapter = self.path_adapter
        assert adapter is not None
        width_px, height_px = self.settings.clamp_render_size(
            request_model.render.width_px,
            request_model.render.height_px,
        )

        infer_start = time.perf_counter()
        case, preds, _centres, gt = self._predict_path_with_gt(request_model)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        frames, result_meta, (mae, rmse), render_ms = self._time_dependent_3d(
            label="Path",
            preds=preds,
            gt=gt,
            times=case.times,
            grid=adapter.grid(),
            width_px=width_px,
            height_px=height_px,
            plane=request_model.plane,
            gt_split="test" if gt is not None else None,
        )
        total_ms = (time.perf_counter() - total_start) * 1000.0

        response = InferPathRender3DResponse(
            model=request_model.model,
            times_s=[float(t) for t in case.times],
            frames=frames,
            path_meta=self._path_meta(case, mae, rmse),
            result_meta=result_meta,
            timing_ms=TimingMs(infer=round(infer_ms, 3), render=round(render_ms, 3), total=round(total_ms, 3)),
        )
        emit_json_log(
            logger,
            logging.INFO,
            "infer_path_render3d_ok",
            server_request_id=server_request_id,
            client_request_id=request_model.client_request_id,
            model=request_model.model,
            preset=case.preset,
            n_frames=int(case.times.size),
            timing_ms=response.timing_ms.model_dump(),
        )
        return response

    # ------------------------------------------------------------------ #
    # /infer_straight (time-dependent straight-line inference, free P/V)
    # ------------------------------------------------------------------ #
    def gt_catalog(self, dataset: str) -> list[GtCatalogEntry]:
        if dataset == "straight":
            if self.straight_adapter is None:
                raise ServiceNotReadyError("Straight-line adapter is not initialized")
            entries = self.straight_adapter.gt_catalog()
        elif dataset == "steady":
            entries = steady_gt.catalog(self.settings.steady_gt_data_dir)
        else:
            raise PathInputError(f"Unknown dataset '{dataset}'. Available: steady, straight")
        return [
            GtCatalogEntry(
                name=e["name"],
                p_watts=float(e["p_watts"]),
                v_scan_mps=float(e["v_scan_mps"]),
                split=e["split"],
            )
            for e in entries
        ]

    def _predict_straight_cached(
        self,
        model: str,
        p_watts: float,
        v_scan_mps: float,
    ) -> tuple[StraightCase, dict[str, Any], Any]:
        import hashlib

        import numpy as np

        adapter = self.straight_adapter
        assert adapter is not None
        physics = adapter.physics_for(model)
        case = adapter.resolve(
            p_watts=p_watts,
            v_scan_mps=v_scan_mps,
            h_ref=physics.h_norm_denom,
            physics=physics,
            grid_xyz=adapter.model_grid(model),
        )
        cache_key = (
            model,
            hashlib.sha256(
                case.points.tobytes()
                + case.times.tobytes()
                + np.array([case.p_watts, case.v_scan_mps], dtype=np.float64).tobytes()
            ).digest(),
        )
        entry = self._cached_forward(
            self._straight_prediction_cache, cache_key, model,
            lambda: PathPredictionCacheEntry(preds=adapter.infer(model, case), centres=adapter.beam_centres(case)),
        )
        return case, entry.preds, entry.centres

    @staticmethod
    def _straight_meta(
        case: StraightCase,
        mae: float | None = None,
        rmse: float | None = None,
    ) -> StraightMeta:
        return StraightMeta(
            p_watts=float(case.p_watts),
            v_scan_mps=float(case.v_scan_mps),
            path_length_m=float(case.path_length_m),
            laser_off_time_s=float(case.laser_off_time_s),
            gt_available=case.gt_available,
            gt_name=case.gt_name,
            split=case.split,
            points_xy_m=[[float(px), float(py)] for px, py in case.points],
            gt_mae=mae,
            gt_rmse=rmse,
        )

    def _predict_straight_with_gt(
        self, request_model: Any
    ) -> tuple[StraightCase, dict[str, Any], Any, dict[str, Any] | None]:
        """(case, preds, centres, gt) for either /infer_straight route."""
        adapter = self.straight_adapter
        assert adapter is not None

        def go():
            case, preds, centres = self._predict_straight_cached(
                request_model.model, request_model.inputs.p_watts, request_model.inputs.v_scan_mps,
            )
            return case, preds, centres, adapter.ground_truth(case)

        return self._guard_time_dependent("Straight", go)

    @staticmethod
    def _straight_grid(adapter: StraightPathAdapter, case: StraightCase) -> tuple[Any, Any, Any]:
        return case.grid_xyz if case.grid_xyz is not None else adapter.grid()

    def _infer_straight_sync(
        self, request_model: InferStraightRequest, server_request_id: str, total_start: float
    ) -> InferStraightResponse:
        adapter = self.straight_adapter
        assert adapter is not None
        width_px, _ = self.settings.clamp_render_size(
            request_model.render.width_px, request_model.render.height_px
        )

        infer_start = time.perf_counter()
        case, preds, centres, gt = self._predict_straight_with_gt(request_model)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        frames, result_meta, (mae, rmse), render_ms = self._time_dependent_2d(
            label="Straight",
            view=request_model.view,
            preds=preds,
            gt=gt,
            centres=centres,
            points=case.points,
            times=case.times,
            grid=self._straight_grid(adapter, case),
            width_px=width_px,
            gt_split=case.split,
        )
        total_ms = (time.perf_counter() - total_start) * 1000.0

        response = InferStraightResponse(
            model=request_model.model,
            times_s=[float(t) for t in case.times],
            frames=frames,
            straight_meta=self._straight_meta(case, mae, rmse),
            result_meta=result_meta,
            timing_ms=TimingMs(infer=round(infer_ms, 3), render=round(render_ms, 3), total=round(total_ms, 3)),
        )
        emit_json_log(
            logger,
            logging.INFO,
            "infer_straight_ok",
            server_request_id=server_request_id,
            client_request_id=request_model.client_request_id,
            model=request_model.model,
            gt_name=case.gt_name,
            p_watts=case.p_watts,
            v_scan_mps=case.v_scan_mps,
            timing_ms=response.timing_ms.model_dump(),
        )
        return response

    def _infer_straight_render3d_sync(
        self, request_model: InferStraightRender3DRequest, server_request_id: str, total_start: float
    ) -> InferStraightRender3DResponse:
        adapter = self.straight_adapter
        assert adapter is not None
        width_px, height_px = self.settings.clamp_render_size(
            request_model.render.width_px, request_model.render.height_px
        )

        infer_start = time.perf_counter()
        case, preds, _centres, gt = self._predict_straight_with_gt(request_model)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        frames, result_meta, (mae, rmse), render_ms = self._time_dependent_3d(
            label="Straight",
            preds=preds,
            gt=gt,
            times=case.times,
            grid=self._straight_grid(adapter, case),
            width_px=width_px,
            height_px=height_px,
            plane=request_model.plane,
            gt_split=case.split,
        )
        total_ms = (time.perf_counter() - total_start) * 1000.0

        response = InferStraightRender3DResponse(
            model=request_model.model,
            times_s=[float(t) for t in case.times],
            frames=frames,
            straight_meta=self._straight_meta(case, mae, rmse),
            result_meta=result_meta,
            timing_ms=TimingMs(infer=round(infer_ms, 3), render=round(render_ms, 3), total=round(total_ms, 3)),
        )
        emit_json_log(
            logger,
            logging.INFO,
            "infer_straight_render3d_ok",
            server_request_id=server_request_id,
            client_request_id=request_model.client_request_id,
            model=request_model.model,
            gt_name=case.gt_name,
            n_frames=int(case.times.size),
            timing_ms=response.timing_ms.model_dump(),
        )
        return response


class _StubAdapter:
    def build_runtime(self, settings: Settings) -> Any:  # pragma: no cover
        return SimpleNamespace(h_norm_denom=1.0)

    def infer(
        self, runtime: Any, p_watts: float, v_scan_mps: float, x_shift_m: float = 0.0,
    ) -> dict[str, Any]:
        import numpy as np

        nx, ny, nz = 64, 32, 48
        x = np.linspace(0, 1, nx, dtype=np.float32)
        y = np.linspace(0, 1, ny, dtype=np.float32)
        z = np.linspace(0, 1, nz, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        base = 300 + 2800 * np.exp(-((X - 0.55) ** 2 / 0.03 + (Z - 0.45) ** 2 / 0.04)) * (0.4 + 0.6 * (p_watts / 200.0))
        speed_factor = np.clip(1.2 - v_scan_mps, 0.3, 1.3)
        t = (base * speed_factor).astype(np.float32)
        alpha = np.clip((t - 1400.0) / 900.0, 0.0, 1.0).astype(np.float32)
        fl = np.clip((t - 1600.0) / 400.0, 0.0, 1.0).astype(np.float32)
        return {
            "T": t[None, None, ...],
            "alpha": alpha[None, None, ...],
            "fl": fl[None, None, ...],
        }

    def runtime_summary(self, runtime: Any) -> dict[str, Any]:
        return {"mode": "stub"}
