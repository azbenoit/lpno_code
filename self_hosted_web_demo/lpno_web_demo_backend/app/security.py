from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp

from .config import Settings
from .schemas import make_error_response

REQUEST_ID_HEADER = "X-Request-ID"


def _request_id_from_headers(request: Request) -> str:
    incoming = request.headers.get(REQUEST_ID_HEADER)
    if incoming and incoming.strip():
        return incoming.strip()
    return uuid.uuid4().hex


def install_http_middlewares(app: FastAPI, settings: Settings) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable]):
        request_id = _request_id_from_headers(request)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    # Per-route POST body limits; /infer_path carries drawn polylines (~128 KB).
    # /infer_straight carries only P/V + model + render, so the small default
    # request limit is plenty.
    body_limits = {
        "/infer": settings.request_max_bytes,
        "/infer/gt_compare": settings.request_max_bytes,
        "/infer_path": settings.path_request_max_bytes,
        "/infer_path/render3d": settings.path_request_max_bytes,
        "/infer_straight": settings.request_max_bytes,
        "/infer_straight/render3d": settings.request_max_bytes,
    }

    @app.middleware("http")
    async def request_size_limit_middleware(request: Request, call_next: Callable[[Request], Awaitable]):
        limit = body_limits.get(request.url.path)
        if limit is not None and request.method.upper() == "POST":
            content_length = request.headers.get("content-length")
            if content_length is None:
                # Chunked/no-length bodies would bypass the size check below, so
                # require a Content-Length on the limited inference routes.
                request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
                err = make_error_response(
                    server_request_id=request_id,
                    code="length_required",
                    message="Content-Length header is required on this route",
                    retryable=False,
                )
                response = JSONResponse(status_code=411, content=err.model_dump())
                response.headers[REQUEST_ID_HEADER] = request_id
                return response
            try:
                size = int(content_length)
            except ValueError:
                size = None
            if size is None or size > limit:
                request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
                err = make_error_response(
                    server_request_id=request_id,
                    code="payload_too_large",
                    message=f"Request body exceeds limit of {limit} bytes",
                    retryable=False,
                )
                response = JSONResponse(status_code=413, content=err.model_dump())
                response.headers[REQUEST_ID_HEADER] = request_id
                return response
        return await call_next(request)
