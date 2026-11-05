"""Shim: moved to `aiida_pino.console.logging_utils`."""

from aiida_pino.console.logging_utils import configure_logging, emit_json_log  # noqa: F401

__all__ = ["configure_logging", "emit_json_log"]
