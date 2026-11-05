from __future__ import annotations


class ConsoleError(Exception):
    """Base for typed console errors mapped to HTTP status + error envelope."""


class ServiceNotReadyError(ConsoleError):
    """The AiiDA profile/service is not initialized (503)."""


class DaemonUnavailableError(ConsoleError):
    """A submission or kill was requested but no process controller / broker is
    reachable. Submissions require a RabbitMQ-backed profile with a running
    `verdi daemon`. Mapped to 503."""


class NodeNotFoundError(ConsoleError):
    """A pk/uuid was not found or is the wrong node type (404)."""


class InvalidRequestError(ConsoleError):
    """Input failed validation (a validate_* profile check, a bad path, etc). 422."""


class FeatureDeprecatedError(ConsoleError):
    """The feature is deliberately disabled pending a fix (410)."""


class BackendError(ConsoleError):
    """Unexpected backend failure (500)."""


class OperationFailedError(ConsoleError):
    """An operation failed for a reason the user must be told, verbatim (500).

    `BackendError`'s message is redacted to its class name, because it is raised
    from all over the codebase and its text can carry paths or connection
    strings. That redaction is right, and it is also why a failed submit could
    only say "Internal error" — leaving the user with no idea whether the run
    launched. This carries a message written for the user at the raise site;
    never put a caught exception's `str()` in it.
    """
