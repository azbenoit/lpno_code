"""Make the SQLite-backed AiiDA profile usable while other processes write.

AiiDA's ``core.sqlite_dos`` storage opens the profile database in rollback-journal
mode with SQLite's default busy timeout (effectively zero), so any writer — a
daemon worker stamping extras, an ingest parser — makes every concurrent reader
fail immediately with ``database is locked``. In WAL mode readers do not block
behind a writer at all, and a multi-second busy timeout absorbs the remaining
collisions (checkpoints, the brief exclusive lock a WAL checkpoint takes).

``journal_mode = WAL`` is a persistent property of the database file: the first
connection that manages to set it converts the file for every process, including
the daemon. ``busy_timeout`` is per connection, so the listener re-applies it.

This does not rescue a writer that is genuinely wedged inside an open
transaction — nothing short of restarting that process does — it removes the
routine collisions that look identical from the outside.
"""

from __future__ import annotations

import os

DEFAULT_BUSY_TIMEOUT_MS = 30_000

_installed = False


def tune_sqlite_connections(busy_timeout_ms: int | None = None) -> int:
    """Install a SQLAlchemy connect hook setting WAL + busy timeout. Idempotent.

    Returns the busy timeout in ms that new connections will use.
    """
    global _installed
    timeout = int(
        busy_timeout_ms
        or os.environ.get("AIIDA_PINO_SQLITE_BUSY_TIMEOUT_MS")
        or DEFAULT_BUSY_TIMEOUT_MS
    )
    if _installed:
        return timeout
    import sqlite3

    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @event.listens_for(Engine, "connect")
    def _apply_pragmas(dbapi_connection, _record):  # noqa: ANN001
        if not isinstance(dbapi_connection, sqlite3.Connection):
            return
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA busy_timeout = {timeout}")
            # Can fail while another process holds the file; the next connection
            # retries, and a failure here must never break opening the profile.
            try:
                cursor.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
        finally:
            cursor.close()

    _installed = True
    return timeout


def is_locked_error(exc: BaseException) -> bool:
    """True for the "database is locked" family of SQLite errors."""
    return "database is locked" in str(exc).lower()
