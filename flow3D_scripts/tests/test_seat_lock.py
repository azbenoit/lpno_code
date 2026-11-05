#!/usr/bin/env python3
"""Short, no-license test for the shared solver lock around solve().

The lock makes a direct run and a spooled run mutually exclusive, since both
reach the solver through ``solve()``. No solver or
license is involved here -- it exercises the lock context manager directly.
"""

from __future__ import annotations

import fcntl
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_rectangular_case as rectangular  # noqa: E402


class SeatLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ, {"F3D_SPOOL_DIR": str(self.dir)})
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def _seat_busy(self) -> bool:
        """Probe the shared lock from an independent fd (flock contends across
        separate open descriptions, even within one process)."""
        fd = os.open(self.dir / ".seat.lock", os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True
        finally:
            os.close(fd)

    def test_seat_lock_is_exclusive_while_held(self) -> None:
        self.assertFalse(self._seat_busy())  # free before
        with rectangular.seat_lock():
            # A second solve (direct or spooled) could not take the lock now.
            self.assertTrue(self._seat_busy())
        self.assertFalse(self._seat_busy())  # released after

    def test_seat_lock_uses_shared_spool_dir(self) -> None:
        # The lock lives under F3D_SPOOL_DIR, so sibling repos share one lock.
        with rectangular.seat_lock():
            pass
        self.assertTrue((self.dir / ".seat.lock").exists())


if __name__ == "__main__":
    unittest.main()


class SeatHandoffTests(unittest.TestCase):
    """`F3D_SEAT_HELD` lets a launcher take the lock and hand it down.

    The aiida-pino run adapter acquires the lock *before* it wraps the runner in
    the wall-clock `timeout`, so a batch submitted all at once serializes with no
    solve budget running. The child must then not re-acquire: `flock` is per open
    file description, so re-opening the same path would block against its own
    parent forever.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.spool = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"F3D_SPOOL_DIR": str(self.spool)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tmp.cleanup)
        self.fd = os.open(self.spool / ".seat.lock", os.O_CREAT | os.O_WRONLY, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)          # stand in for the launcher
        self.addCleanup(os.close, self.fd)

    def test_handed_down_seat_is_not_reacquired(self) -> None:
        with mock.patch.dict(os.environ, {"F3D_SEAT_HELD": "1"}):
            with rectangular.seat_lock():            # must not block
                pass
