"""The SQLite connect hook must put new connections in WAL with a busy timeout.

Without it, one writing daemon worker makes every console read fail with
"database is locked" — the failure mode that repeatedly took the web console
down while the active-learning loop was writing.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import create_engine, text

from aiida_pino.utils.sqlite_tuning import is_locked_error, tune_sqlite_connections


def test_connect_hook_sets_wal_and_busy_timeout(tmp_path):
    timeout = tune_sqlite_connections(7000)
    assert timeout == 7000  # first caller wins; later calls are no-ops
    engine = create_engine(f"sqlite:///{tmp_path / 'probe.sqlite'}")
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 7000


def test_wal_lets_a_reader_read_during_an_open_write(tmp_path):
    # The point of WAL: a writer holding a transaction no longer blocks readers.
    tune_sqlite_connections()
    url = f"sqlite:///{tmp_path / 'concurrent.sqlite'}"
    setup = create_engine(url)
    with setup.begin() as conn:
        conn.execute(text("CREATE TABLE t (v INTEGER)"))
        conn.execute(text("INSERT INTO t VALUES (1)"))
    writer = create_engine(url).connect()
    reader = create_engine(url).connect()
    try:
        trans = writer.begin()
        writer.execute(text("INSERT INTO t VALUES (2)"))
        assert reader.execute(text("SELECT count(*) FROM t")).scalar() == 1  # pre-commit snapshot
        trans.commit()
    finally:
        writer.close()
        reader.close()


def test_is_locked_error_recognises_sqlite_text():
    assert is_locked_error(Exception("(sqlite3.OperationalError) database is locked"))
    assert not is_locked_error(Exception("no such table: db_dbnode"))
