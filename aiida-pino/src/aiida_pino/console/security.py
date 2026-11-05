from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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

    # POST body cap. Submit routes carry profile dicts (and, for FLOW-3D
    # generate/batch, a full run_config JSON), so they get the larger cap;
    # everything else uses the small default.
    default_limit = settings.request_max_bytes
    submit_limit = settings.submit_max_bytes
    submit_prefixes = ("/submit/",)

    def _limit_for(path: str, method: str) -> int | None:
        if method.upper() != "POST":
            return None
        if any(path.startswith(p) for p in submit_prefixes):
            return submit_limit
        return default_limit

    @app.middleware("http")
    async def request_size_limit_middleware(request: Request, call_next: Callable[[Request], Awaitable]):
        limit = _limit_for(request.url.path, request.method)
        if limit is not None:
            content_length = request.headers.get("content-length")
            request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
            if content_length is None:
                # Require a length so a chunked body cannot bypass the cap.
                err = make_error_response(
                    server_request_id=request_id,
                    code="length_required",
                    message="Content-Length header is required on POST routes",
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
