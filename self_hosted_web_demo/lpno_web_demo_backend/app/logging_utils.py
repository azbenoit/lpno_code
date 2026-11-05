from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        stream=sys.stdout,
        format="%(message)s",
        force=True,
    )


def _default_json(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:  # pragma: no cover
        return repr(obj)


def emit_json_log(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    logger.log(level, json.dumps(payload, default=_default_json, sort_keys=True))
