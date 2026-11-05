"""Shim: the console service moved into the installable package so the
`aiida-pino` CLI can import it. Kept so `web.app.aiida_service` imports keep
working; new code should import `aiida_pino.console.service`."""

from aiida_pino.console.service import AiidaConsoleService  # noqa: F401

__all__ = ["AiidaConsoleService"]
