from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .logging_utils import configure_logging, emit_json_log
from .adapter_randompath import PathInputError
from .schemas import (
    ErrorResponse,
    GtCatalogResponse,
    InferPathRender3DRequest,
    InferPathRender3DResponse,
    InferPathRequest,
    InferPathResponse,
    InferGtCompareRequest,
    InferGtCompareResponse,
    InferRequest,
    InferResponse,
    InferStraightRender3DRequest,
    InferStraightRender3DResponse,
    InferStraightRequest,
    InferStraightResponse,
    ModelInfo,
    ModelsResponse,
    PresetsResponse,
    make_error_response,
)
from .model_registry import discover_models
from .security import install_http_middlewares
from .service import (
    BackendInferenceError,
    BusyError,
    InferenceService,
    InferenceTimeoutError,
    ServiceNotReadyError,
)

logger = logging.getLogger(__name__)


def _error_json(status_code: int, err: ErrorResponse) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=err.model_dump())


def _request_id_from_request(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def create_app(*, settings: Settings | None = None, service: InferenceService | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    # Confine the process to one GPU before torch is ever imported (the runtime is
    # built lazily in the lifespan below), so the CUDA context + all allocations
    # land on the chosen device — never a stray one on a full GPU0. Skip in stub
    # mode (no CUDA) and when a service is injected (tests).
    if service is None and not settings.use_stub_runtime:
        from .gpu import bootstrap_cuda_visible_devices

        bootstrap_cuda_visible_devices(settings.cuda_device)

    injected_service = service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if injected_service is not None:
            app.state.inference_service = injected_service
        elif settings.startup_load_runtime:
            emit_json_log(logger, logging.INFO, "startup_runtime_build_begin")
            app.state.inference_service = InferenceService.build_default(settings)
            emit_json_log(logger, logging.INFO, "startup_runtime_build_done")
        else:
            app.state.inference_service = None
            emit_json_log(logger, logging.WARNING, "startup_runtime_build_skipped")

        # Background sweeper: an idle demo hands its GPU back to trainings
        # sharing the card. The headroom guard only fires on a request, so
        # without this an idle backend sits on several GiB indefinitely.
        sweeper = None
        svc = app.state.inference_service
        if svc is not None and settings.model_idle_ttl_s > 0:
            async def _sweep() -> None:
                while True:
                    await asyncio.sleep(max(1.0, float(settings.cache_sweep_interval_s)))
                    try:
                        dropped = svc.idle_release_if_stale()
                    except Exception as exc:  # noqa: BLE001 -- a sweeper must never kill the app
                        logger.debug("idle sweep skipped (%s)", type(exc).__name__)
                        continue
                    if dropped:
                        emit_json_log(logger, logging.INFO, "idle_cache_released", models=dropped)

            sweeper = asyncio.create_task(_sweep())
        try:
            yield
        finally:
            if sweeper is not None:
                sweeper.cancel()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings

    # CORS must be added last so it is outermost: short-circuit responses from
    # the security middlewares (411/413) must still carry CORS headers for
    # cross-origin dev clients to read the error envelope.
    install_http_middlewares(app, settings)
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
            max_age=600,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = _request_id_from_request(request)
        err = make_error_response(
            server_request_id=request_id,
            code="validation_error",
            message=str(exc),
            retryable=False,
        )
        return _error_json(422, err)

    @app.exception_handler(PathInputError)
    async def path_input_error_handler(request: Request, exc: PathInputError):
        request_id = _request_id_from_request(request)
        err = make_error_response(
            server_request_id=request_id,
            code="validation_error",
            message=str(exc),
            retryable=False,
        )
        return _error_json(422, err)

    @app.exception_handler(BusyError)
    async def busy_exception_handler(request: Request, exc: BusyError):
        request_id = _request_id_from_request(request)
        err = make_error_response(
            server_request_id=request_id,
            code="busy",
            message=str(exc),
            retryable=True,
            retry_after_ms=100,
        )
        return _error_json(429, err)

    @app.exception_handler(InferenceTimeoutError)
    async def timeout_exception_handler(request: Request, exc: InferenceTimeoutError):
        request_id = _request_id_from_request(request)
        err = make_error_response(
            server_request_id=request_id,
            code="timeout",
            message=str(exc),
            retryable=True,
            retry_after_ms=250,
        )
        return _error_json(504, err)

    @app.exception_handler(ServiceNotReadyError)
    async def service_not_ready_handler(request: Request, exc: ServiceNotReadyError):
        request_id = _request_id_from_request(request)
        err = make_error_response(
            server_request_id=request_id,
            code="service_unavailable",
            message=str(exc),
            retryable=True,
            retry_after_ms=1000,
        )
        return _error_json(503, err)

    @app.exception_handler(BackendInferenceError)
    async def backend_inference_error_handler(request: Request, exc: BackendInferenceError):
        request_id = _request_id_from_request(request)
        err = make_error_response(
            server_request_id=request_id,
            code="internal",
            message=str(exc),
            retryable=False,
        )
        return _error_json(500, err)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = _request_id_from_request(request)
        emit_json_log(logger, logging.ERROR, "unhandled_exception", server_request_id=request_id, error=repr(exc))
        err = make_error_response(
            server_request_id=request_id,
            code="internal",
            message=f"Unhandled error: {type(exc).__name__}",
            retryable=False,
        )
        return _error_json(500, err)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        service_obj = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            return {"status": "ok", "runtime_loaded": False, "service": None}
        return {"status": "ok", **service_obj.health()}

    @app.post("/infer", response_model=InferResponse)
    async def infer(request: Request, payload: InferRequest) -> InferResponse:
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        request_id = _request_id_from_request(request)
        return await service_obj.infer(payload, server_request_id=request_id)

    @app.post("/infer/gt_compare", response_model=InferGtCompareResponse)
    async def infer_gt_compare(request: Request, payload: InferGtCompareRequest) -> InferGtCompareResponse:
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        request_id = _request_id_from_request(request)
        return await service_obj.infer_gt_compare(payload, server_request_id=request_id)

    @app.post("/infer_path", response_model=InferPathResponse)
    async def infer_path(request: Request, payload: InferPathRequest) -> InferPathResponse:
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        request_id = _request_id_from_request(request)
        return await service_obj.infer_path(payload, server_request_id=request_id)

    @app.post("/infer_path/render3d", response_model=InferPathRender3DResponse)
    async def infer_path_render3d(request: Request, payload: InferPathRender3DRequest) -> InferPathRender3DResponse:
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        request_id = _request_id_from_request(request)
        return await service_obj.infer_path_render3d(payload, server_request_id=request_id)

    @app.get("/infer_path/presets", response_model=PresetsResponse)
    async def infer_path_presets(request: Request, model: str | None = None) -> PresetsResponse:
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        # Per-model: a model is only offered the GT cases it was trained for.
        return PresetsResponse(presets=service_obj.path_presets(model))

    @app.post("/infer_straight", response_model=InferStraightResponse)
    async def infer_straight(request: Request, payload: InferStraightRequest) -> InferStraightResponse:
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        request_id = _request_id_from_request(request)
        return await service_obj.infer_straight(payload, server_request_id=request_id)

    @app.post("/infer_straight/render3d", response_model=InferStraightRender3DResponse)
    async def infer_straight_render3d(
        request: Request, payload: InferStraightRender3DRequest
    ) -> InferStraightRender3DResponse:
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        request_id = _request_id_from_request(request)
        return await service_obj.infer_straight_render3d(payload, server_request_id=request_id)

    @app.get("/models", response_model=ModelsResponse)
    async def models(page: Literal["steady", "straight", "path"]) -> ModelsResponse:
        # Folder-based discovery (see model_registry.py); independent of the
        # runtime so the dropdown populates even before a model is loaded.
        dirs = {
            "steady": settings.steady_models_dir,
            "straight": settings.straight_models_dir,
            "path": settings.path_models_dir,
        }
        found = discover_models(dirs[page])  # sorted; `order` is internal, dropped here
        return ModelsResponse(
            page=page,
            models=[
                ModelInfo(
                    id=m["id"], label=m["label"], type=m["type"], default=m["default"],
                    physics=m.get("physics"),
                )
                for m in found
            ],
        )

    @app.post("/cache/clear")
    async def clear_cache(request: Request) -> dict:
        """Drop every cached model now and return the GPU memory.

        Operator lever (no page calls it): the way to hand
        the card back to a training without waiting out the idle TTL. Safe at any
        time: models reload on next use.
        """
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        from .gpu import free_vram_bytes

        before = free_vram_bytes()
        dropped = service_obj.release_all_models()
        after = free_vram_bytes()
        return {
            "status": "ok",
            "released": dropped,
            "free_vram_gib_before": round(before / 1024**3, 2) if before else None,
            "free_vram_gib_after": round(after / 1024**3, 2) if after else None,
        }

    @app.get("/gt_catalog", response_model=GtCatalogResponse)
    async def gt_catalog(request: Request, dataset: str) -> GtCatalogResponse:
        service_obj: InferenceService | None = getattr(request.app.state, "inference_service", None)
        if service_obj is None:
            raise ServiceNotReadyError("Inference service is not initialized")
        return GtCatalogResponse(dataset=dataset, entries=service_obj.gt_catalog(dataset))

    return app


def run() -> None:  # pragma: no cover
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=settings.uvicorn_workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
