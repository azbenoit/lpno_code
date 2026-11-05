from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .aiida_service import AiidaConsoleService
from .config import Settings
from .errors import (
    BackendError,
    ConsoleError,
    DaemonUnavailableError,
    FeatureDeprecatedError,
    InvalidRequestError,
    NodeNotFoundError,
    OperationFailedError,
    ServiceNotReadyError,
)
from .logging_utils import configure_logging, emit_json_log
from .schemas import (
    AnnotationsRequest,
    EvictDataRequest,
    FolderEditRequest,
    FolderRequest,
    AssignSplitsRequest,
    CollectionCreateRequest,
    CollectionPreviewRequest,
    DeleteRequest,
    DeriveDatasetRequest,
    ErrorResponse,
    make_error_response,
)
from .security import install_http_middlewares

logger = logging.getLogger(__name__)


def _error_json(status_code: int, err: ErrorResponse) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=err.model_dump())


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _cap_threadpool_to_db_pool() -> int | None:
    """Hold the request threadpool under the database connection pool.

    Every endpoint here is a sync ``def``, so Starlette runs it on anyio's
    threadpool — 40 threads by default — and aiida-core hands each thread its
    own scoped session, which pins one connection for as long as that thread is
    alive. The profile's pool is ``pool_size 5 + max_overflow 10``, so thread 16
    waited the full 30 s and then failed with ``QueuePool limit ... timed out``
    on whichever endpoint it happened to be serving: 73 pool timeouts and 46
    HTTP 500s in one 17-minute window, surfaced to the user as "the AiiDA
    daemon/broker is not reachable" while the daemon was up the whole time.

    Capping the threadpool below the pool makes request 16 *wait for a thread*
    instead of racing for a connection it cannot get. Slower under burst,
    never a spurious 500.
    """
    try:
        import anyio.to_thread
        from aiida.manage import get_manager

        engine = get_manager().get_profile_storage().get_session().get_bind()
        total = int(engine.pool.size()) + int(getattr(engine.pool, "_max_overflow", 10))
        cap = max(4, total - 3)  # headroom for the connections aiida itself holds
        limiter = anyio.to_thread.current_default_thread_limiter()
        if limiter.total_tokens > cap:
            limiter.total_tokens = cap
        return int(limiter.total_tokens)
    except Exception:  # noqa: BLE001 - a cap is an optimisation, never a startup failure
        return None


def create_app(*, settings: Settings | None = None, service: AiidaConsoleService | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    injected = service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if injected is not None:
            app.state.service = injected
        elif settings.startup_load_service:
            emit_json_log(logger, logging.INFO, "startup_profile_load_begin", profile=settings.aiida_profile or "<default>")
            app.state.service = AiidaConsoleService.build_default(settings)
            emit_json_log(logger, logging.INFO, "startup_profile_load_done", threadpool=_cap_threadpool_to_db_pool())
        else:
            app.state.service = None
            emit_json_log(logger, logging.WARNING, "startup_profile_load_skipped")
        yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings

    # CORS added last => outermost, so 411/413 short-circuits from the security
    # middlewares still carry CORS headers for cross-origin dev clients.
    install_http_middlewares(app, settings)

    # Added before CORS => inside it. Starlette runs its `Exception` handler in
    # ServerErrorMiddleware, which sits *outside* CORS, so an unhandled error
    # reaches a cross-origin browser header-less and shows up as the useless
    # "Failed to fetch". Converting it here keeps the real message visible.
    @app.middleware("http")
    async def _errors_inside_cors(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            return await _internal_error(request, exc)

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

    # --- typed error handlers -> fixed status + shared envelope --------------
    def _handler(code: str, status: int, retryable: bool, retry_after_ms: int | None = None):
        async def h(request: Request, exc: Exception):
            err = make_error_response(
                server_request_id=_request_id(request),
                code=code,
                message=str(exc) or code,
                retryable=retryable,
                retry_after_ms=retry_after_ms,
            )
            return _error_json(status, err)

        return h

    async def _validation_error(request: Request, exc: Exception):
        """Name the fields, not the payload.

        `str(exc)` on a pydantic ValidationError embeds every rejected value:
        `POST /submit/flow3d_batch` with 1200 run_configs and one missing field
        answered 422 with a 293,091-character message that was mostly the
        caller's own body echoed back. The console renders it pre-wrap with no
        max-height, and the CLI and scripted clients -- which is how most
        submissions arrive -- got half a megabyte of their own request.
        """
        details = getattr(exc, "errors", None)
        lines: list[str] = []
        for error in (details() if callable(details) else []):
            where = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            lines.append(f"{where or 'body'}: {error.get('msg', 'invalid')}")
        message = "; ".join(lines[:12]) or str(exc)[:2000]
        if len(lines) > 12:
            message += f" (+{len(lines) - 12} more)"
        err = make_error_response(
            server_request_id=_request_id(request),
            code="validation_error",
            message=message[:4000],
            retryable=False,
        )
        return _error_json(422, err)

    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(InvalidRequestError, _handler("validation_error", 422, False))
    app.add_exception_handler(NodeNotFoundError, _handler("not_found", 404, False))
    app.add_exception_handler(FeatureDeprecatedError, _handler("feature_deprecated", 410, False))
    app.add_exception_handler(DaemonUnavailableError, _handler("daemon_unavailable", 503, True, 2000))
    app.add_exception_handler(ServiceNotReadyError, _handler("service_unavailable", 503, True, 1000))

    async def _internal_error(request: Request, exc: Exception):
        # Redact: log the full exception server-side, return only the class name
        # so internal paths / connection strings never reach the client.
        emit_json_log(logger, logging.ERROR, "internal_error", server_request_id=_request_id(request), error=repr(exc))
        err = make_error_response(
            server_request_id=_request_id(request),
            code="internal",
            message=f"Internal error: {type(exc).__name__}",
            retryable=False,
        )
        return _error_json(500, err)

    app.add_exception_handler(BackendError, _internal_error)
    app.add_exception_handler(ConsoleError, _internal_error)
    # Registered after the base class so it wins: its message is authored for
    # the user (BackendError's is redacted to a class name, deliberately).
    app.add_exception_handler(OperationFailedError, _handler("operation_failed", 500, False))

    async def _db_busy(request: Request, exc: Exception):
        # A locked SQLite profile is transient, not an internal fault. Handled
        # here (inside the CORS middleware) so the browser sees the message
        # instead of an opaque "Failed to fetch" on a header-less 500.
        from aiida_pino.utils.sqlite_tuning import is_locked_error

        if not is_locked_error(exc):
            return await _internal_error(request, exc)
        emit_json_log(logger, logging.WARNING, "database_locked", server_request_id=_request_id(request))
        err = make_error_response(
            server_request_id=_request_id(request),
            code="database_locked",
            message="AiiDA database is locked by another process (daemon worker or ingest); retry shortly.",
            retryable=True,
            retry_after_ms=3000,
        )
        return _error_json(503, err)

    try:  # sqlalchemy ships with aiida; guard keeps import errors from breaking the app
        from sqlalchemy.exc import OperationalError, TimeoutError as SATimeoutError

        app.add_exception_handler(OperationalError, _db_busy)
        # A saturated pool is a queue, not a fault. TimeoutError is not an
        # OperationalError, so it used to fall through to a 500 with
        # retryable:false — which blanked the environment strip and disabled
        # every submit button until the page was reloaded.
        app.add_exception_handler(
            SATimeoutError,
            _handler("backend_busy", 503, True, 3000),
        )
    except ImportError:  # pragma: no cover
        pass

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        emit_json_log(logger, logging.ERROR, "unhandled_exception", server_request_id=_request_id(request), error=repr(exc))
        err = make_error_response(
            server_request_id=_request_id(request),
            code="internal",
            message=f"Unhandled error: {type(exc).__name__}",
            retryable=False,
        )
        return _error_json(500, err)

    def _service(request: Request) -> AiidaConsoleService:
        svc = getattr(request.app.state, "service", None)
        if svc is None:
            raise ServiceNotReadyError("AiiDA profile/service is not initialized")
        return svc

    # --- health -------------------------------------------------------------
    @app.get("/healthz")
    def healthz(request: Request) -> dict[str, Any]:
        svc = getattr(request.app.state, "service", None)
        if svc is None:
            return {"status": "ok", "service_loaded": False}
        return {"status": "ok", "service_loaded": True, **svc.health()}

    @app.get("/env")
    def env(request: Request) -> dict[str, Any]:
        return _service(request).env_status()

    @app.get("/codes")
    def codes(request: Request) -> dict[str, Any]:
        return _service(request).list_codes()

    @app.get("/datasets")
    def datasets(
        request: Request,
        search: str | None = None,
        role: str | None = None,
        source: str | None = None,
        keyword: list[str] | None = Query(None),
        limit: int | None = None,
        offset: int | None = None,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        rows = _service(request).list_datasets(
            search=search, role=role, source=source, keywords=keyword, limit=limit,
            offset=offset, include_superseded=include_superseded,
        )
        # Truncation is reported at the payload level, never silently applied:
        # a listing that drops rows must say so and say how many exist.
        # Union-correct per-folder totals (summing the rows double-counts what
        # two datasets in a folder share) and the whole-population keyword
        # facet. Both ride on the payload rather than on the service: parked on
        # the shared singleton they could be read back by a *different*
        # concurrent request, which handed 500 unfiltered rows the 4-row
        # `keyword=core` request's facets and re-created the double-counted
        # folder header M4 was filed for.
        total = rows[0].pop("_total", len(rows)) if rows else 0
        truncated = rows[0].pop("_truncated", False) if rows else False
        folders = rows[0].pop("_folders", {}) if rows else {}
        keywords = rows[0].pop("_keywords", {}) if rows else {}
        for row in rows:
            for sentinel in ("_total", "_truncated", "_folders", "_keywords"):
                row.pop(sentinel, None)
        return {
            "datasets": rows,
            "total": total,
            "truncated": bool(truncated),
            "folders": folders,
            "keywords": keywords,
            "offset": int(offset or 0),
            "has_more": bool(truncated),
        }

    @app.get("/datasets/options")
    def dataset_options(request: Request) -> dict[str, Any]:
        return {"datasets": _service(request).dataset_options()}

    @app.post("/datasets/{pk}/evict")
    def evict_dataset_data(request: Request, pk: int, payload: EvictDataRequest) -> dict[str, Any]:
        return _service(request).evict_dataset_data(pk, dry_run=payload.dry_run, reason=payload.reason)

    @app.get("/datasets/{pk}")
    def dataset_detail(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).dataset_detail(pk)

    @app.get("/datasets/{pk}/cases")
    def dataset_cases(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).dataset_cases(pk)

    @app.get("/simulations/{pk}/frames")
    def simulation_frames(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).simulation_frames(pk)

    @app.get("/simulations/{pk}/stats")
    def simulation_stats(request: Request, pk: int, recompute: bool = False, compute: bool = False) -> dict[str, Any]:
        # `compute` is opt-in: without it this answers from the cache in
        # milliseconds instead of reading every frame of the case for minutes.
        return _service(request).simulation_stats(pk, recompute=recompute, compute=compute)

    @app.get("/datasets/{pk}/stats")
    def dataset_stats(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).dataset_stats(pk)

    @app.post("/datasets/{pk}/stats/compute")
    def dataset_stats_compute(request: Request, pk: int, limit: int = 25) -> dict[str, Any]:
        return _service(request).dataset_stats_compute(pk, limit=limit)

    @app.get("/datasets/{pk}/stats/plot")
    def dataset_stats_plot(request: Request, pk: int, kind: str = "regime"):
        from fastapi.responses import Response

        png = _service(request).dataset_stats_plot(pk, kind=kind)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/models/{pk}/curves/plot")
    def model_curves_plot(request: Request, pk: int, kind: str = "validation"):
        from fastapi.responses import Response

        png = _service(request).model_curves_plot(pk, kind=kind)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/evaluations/{pk}/preview")
    def evaluation_preview(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).evaluation_preview(pk)

    @app.get("/evaluations/{pk}/preview/plot")
    def evaluation_preview_plot(request: Request, pk: int, field: str = "T", z: int = -1):
        from fastapi.responses import Response

        png = _service(request).evaluation_preview_plot(pk, field=field, z=z)
        # The URL fully determines the image, so it can be cached hard.
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.get("/simulations/{pk}/plot")
    def simulation_plot(
        request: Request, pk: int, field: str, view: str = "long", frame: int = -1, contour: bool = True
    ):
        from fastapi.responses import Response

        png = _service(request).simulation_plot(pk, field=field, view=view, frame=frame, contour=contour)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.get("/models")
    def models(
        request: Request,
        search: str | None = None,
        evaluated_only: bool = False,
        keyword: list[str] | None = Query(None),
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        rows = _service(request).list_models(
            search=search, evaluated_only=evaluated_only, keywords=keyword,
            limit=limit, offset=offset,
        )
        # Same contract as /datasets: a listing that drops rows says so and says
        # how many exist. `?limit=20&keyword=rigno` used to return 9 of 50 with
        # nothing to indicate the other 41 were never considered.
        total = rows[0].pop("_total", len(rows)) if rows else 0
        truncated = rows[0].pop("_truncated", False) if rows else False
        for row in rows:
            row.pop("_total", None)
            row.pop("_truncated", None)
        return {
            "models": rows, "total": total, "truncated": bool(truncated),
            # What the caller needs to ask for the next page without counting for itself.
            "offset": int(offset), "has_more": bool(truncated),
        }

    @app.get("/models/{pk}")
    def model_detail(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).model_detail(pk)

    @app.get("/models/{pk}/compare/{other}")
    def model_compare(request: Request, pk: int, other: int) -> dict[str, Any]:
        return _service(request).compare_models(pk, other)

    @app.get("/simulations")
    def simulations(
        request: Request,
        source: str | None = None,
        folder: str | None = None,
        fields: str | None = None,
        split: str | None = None,
        keyword: list[str] | None = Query(None),
        limit: int | None = None,
    ) -> dict[str, Any]:
        field_list = [f.strip() for f in fields.split(",") if f.strip()] if fields else None
        return {
            "simulations": _service(request).list_simulations(
                source=source, folder=folder, fields=field_list, split=split, keywords=keyword, limit=limit
            )
        }

    @app.get("/processes")
    def processes(
        request: Request,
        limit: int | None = None,
        label: str | None = None,
        only_pino: bool = True,
        include_smokes: bool = False,
        state: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        return {
            "processes": _service(request).list_processes(
                limit=limit,
                label_filter=label,
                only_pino=only_pino,
                include_smokes=include_smokes,
                state=state,
                category=category,
            )
        }

    @app.get("/processes/{pk}")
    def process_detail(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).process_detail(pk)

    @app.get("/processes/{pk}/logs")
    def process_logs(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).process_logs(pk)

    @app.get("/processes/{pk}/tail")
    def process_tail(request: Request, pk: int, lines: int = 200, path: str | None = None) -> dict[str, Any]:
        return _service(request).process_tail(pk, lines=lines, path=path)

    # --- previews (broker-free) --------------------------------------------
    @app.post("/preview/collection")
    def preview_collection(request: Request, payload: CollectionPreviewRequest) -> dict[str, Any]:
        return _service(request).preview_collection(payload.simulation_pks)

    @app.post("/collections")
    def create_collection(request: Request, payload: CollectionCreateRequest) -> dict[str, Any]:
        return _service(request).create_collection(
            payload.simulation_pks,
            payload.collection_name,
            description=payload.description,
            keywords=payload.keywords,
            dataset_pks=payload.dataset_pks,
            # was silently dropped before: the schema accepted it, the service
            # never saw it, and blank-description requests failed anyway
            skip_description=payload.skip_description,
            field_policy=payload.field_policy,
        )

    @app.post("/datasets/{pk}/evaluation_only")
    def evaluation_only(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).mark_evaluation_only(pk)

    @app.post("/datasets/{pk}/assign_splits")
    def assign_splits(request: Request, pk: int, payload: AssignSplitsRequest) -> dict[str, Any]:
        return _service(request).assign_splits(
            pk,
            train=payload.train,
            val=payload.val,
            test=payload.test,
            only_unassigned=payload.only_unassigned,
            seed=payload.seed,
        )

    # --- annotations (broker-free: works with the daemon down) --------------
    @app.post("/nodes/{pk}/annotations")
    def update_annotations(request: Request, pk: int, payload: AnnotationsRequest) -> dict[str, Any]:
        return _service(request).update_annotations(
            pk,
            title=payload.title,
            description=payload.description,
            split=payload.split,
            keywords_set=payload.keywords_set,
            keywords_add=payload.keywords_add,
            keywords_remove=payload.keywords_remove,
        )

    @app.get("/folders")
    def folders(request: Request) -> dict[str, Any]:
        return _service(request).list_folders()

    @app.post("/folders/move")
    def move_to_folder(request: Request, payload: FolderRequest) -> dict[str, Any]:
        return _service(request).set_folder(payload.pks, payload.path)

    @app.post("/folders/rename")
    def rename_folder(request: Request, payload: FolderEditRequest) -> dict[str, Any]:
        if not payload.new:
            raise InvalidRequestError("new folder name is required")
        return _service(request).rename_folder(payload.label, payload.new)

    @app.post("/folders/remove")
    def remove_folder(request: Request, payload: FolderEditRequest) -> dict[str, Any]:
        return _service(request).remove_folder(payload.label)

    # --- deletions / kill ---------------------------------------------------
    @app.post("/nodes/delete")
    def delete_nodes(request: Request, payload: DeleteRequest) -> dict[str, Any]:
        return _service(request).delete_nodes(payload.pks, dry_run=payload.dry_run)

    @app.post("/processes/{pk}/kill")
    def kill_process(request: Request, pk: int) -> dict[str, Any]:
        return _service(request).kill_process(pk)

    # --- submissions (gated on daemon) -------------------------------------
    @app.get("/flow3d/queue")
    def flow3d_queue(request: Request) -> dict[str, Any]:
        return _service(request).flow3d_queue_status()

    @app.get("/monitor")
    def monitor(request: Request) -> dict[str, Any]:
        return _service(request).monitor_status()

    @app.post("/submit/derive_dataset")
    def submit_derive(request: Request, payload: DeriveDatasetRequest) -> dict[str, Any]:
        return _service(request).submit_derive(payload)

    # The submission surface (Train / Eval, K-Fold, Import, FLOW-3D and their presets and
    # previews) lives in submit_routes.py. The public export ships without that module and
    # this console stays complete: browsing, monitoring and composing need none of it.
    try:
        from .submit_routes import register as _register_submit_routes
    except ImportError:  # public export: no submission surface
        _register_submit_routes = None
    if _register_submit_routes is not None:
        _register_submit_routes(app, _service)

    return app


def run() -> None:  # pragma: no cover
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "web.app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=settings.uvicorn_workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
