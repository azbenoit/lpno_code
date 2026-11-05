"""Shim: moved to `aiida_pino.console.errors`."""

from aiida_pino.console.errors import (  # noqa: F401
    BackendError,
    ConsoleError,
    DaemonUnavailableError,
    FeatureDeprecatedError,
    InvalidRequestError,
    NodeNotFoundError,
    OperationFailedError,
    ServiceNotReadyError,
)

__all__ = [
    "BackendError",
    "ConsoleError",
    "DaemonUnavailableError",
    "FeatureDeprecatedError",
    "InvalidRequestError",
    "NodeNotFoundError",
    "OperationFailedError",
    "ServiceNotReadyError",
]
