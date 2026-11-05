from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any):
    # Lazy import so `web.app:create_app` resolves without importing FastAPI or
    # loading an AiiDA profile at package import time (tests import pieces).
    from .main import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
