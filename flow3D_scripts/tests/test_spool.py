#!/usr/bin/env python3
"""Short, no-license tests for the persistent task spooler.

Uses ``--cmd``-style tasks with harmless shell commands (touch / false), so no
FLOW-3D run or solver license is involved.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import spool  # noqa: E402


class SpoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _count(self, sub: str) -> int:
        return len(list((self.dir / sub).glob("*.task")))

    def test_enqueue_is_atomic_and_ordered(self) -> None:
        a = spool.enqueue(self.dir, "a", "true")
        b = spool.enqueue(self.dir, "b", "true")
        self.assertEqual(self._count("queue"), 2)
        # Time-ordered filenames: first enqueued sorts first.
        self.assertEqual(sorted([a.name, b.name])[0], a.name)
        self.assertEqual(json.loads(a.read_text())["name"], "a")

    def test_drain_runs_all_and_records_done(self) -> None:
        # The worker drains the queue sequentially (one solve at a time).
        markers = [self.dir / f"m{i}" for i in range(3)]
        for i, marker in enumerate(markers):
            spool.enqueue(self.dir, f"task{i}", f"touch {marker}")
        spool.run_worker(self.dir, drain=True, poll=0.01)
        for marker in markers:
            self.assertTrue(marker.exists(), f"{marker} was not run")
        self.assertEqual(self._count("done"), 3)
        self.assertEqual(self._count("queue"), 0)
        self.assertEqual(self._count("running"), 0)

    def test_failing_command_goes_to_failed(self) -> None:
        spool.enqueue(self.dir, "boom", "false")
        spool.run_worker(self.dir, drain=True, poll=0.01)
        self.assertEqual(self._count("failed"), 1)
        self.assertEqual(self._count("done"), 0)
        log = (self.dir / "pool.log").read_text()
        self.assertIn("failed", log)
        self.assertIn("boom", log)

    def test_recovers_tasks_left_in_running(self) -> None:
        # Simulate a worker killed mid-run: a task stranded in running/.
        spool.enqueue(self.dir, "stranded", f"touch {self.dir / 'recovered'}")
        spool.ensure_dirs(self.dir)
        task = next((self.dir / "queue").glob("*.task"))
        task.rename(self.dir / "running" / task.name)
        self.assertEqual(self._count("running"), 1)
        spool.run_worker(self.dir, drain=True, poll=0.01)
        self.assertTrue((self.dir / "recovered").exists())
        self.assertEqual(self._count("done"), 1)

    def test_vanished_running_task_is_not_fatal(self) -> None:
        # Regression for 2026-07-28: a running task file vanished mid-run and
        # the completion rename raised FileNotFoundError, killing the worker
        # and stranding 7 queued tasks for a day. The completion bookkeeping
        # must survive a missing task file and keep draining.
        spool.enqueue(self.dir, "vanish", f"rm -f {self.dir}/running/*.task")
        spool.enqueue(self.dir, "after", f"touch {self.dir / 'still_ran'}")
        spool.run_worker(self.dir, drain=True, poll=0.01)
        self.assertTrue((self.dir / "still_ran").exists(), "worker died before the next task")
        log = (self.dir / "pool.log").read_text()
        self.assertIn("vanish", log)

    def test_corrupt_task_is_shelved_not_fatal(self) -> None:
        spool.ensure_dirs(self.dir)
        bad = self.dir / "queue" / "00000000000000000000-1.task"
        bad.write_text("not json\n")
        spool.enqueue(self.dir, "good", f"touch {self.dir / 'ok'}")
        spool.run_worker(self.dir, drain=True, poll=0.01)
        self.assertTrue((self.dir / "ok").exists())
        self.assertEqual(self._count("failed"), 1)  # the corrupt one
        self.assertEqual(self._count("done"), 1)  # the good one

    def test_build_command_absolutizes_config_and_targets_runner(self) -> None:
        # Shared queue: the config path is absolutized at add time so whichever
        # worker drains the task opens the right file regardless of its cwd.
        cmd = spool.build_command("configs/with space.json", ["--cores", "2"])
        self.assertIn("run_rectangular_case.py", cmd)
        self.assertRegex(cmd, r"'/.*configs/with space\.json'")
        self.assertIn("--cores 2", cmd)

    def test_default_spool_dir_is_shared_per_user(self) -> None:
        # Both repos default to one per-user queue so their runs serialize on a
        # single worker and lock (one solve at a time).
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("F3D_SPOOL_DIR", None)
            self.assertEqual(spool.default_spool_dir(), Path.home() / ".flow3d" / "spool")
            # Not inside a repo checkout, so sibling repos resolve to the same place.
            self.assertFalse(str(spool.default_spool_dir()).startswith(str(ROOT)))

    def test_default_spool_dir_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"F3D_SPOOL_DIR": str(self.dir)}):
            self.assertEqual(spool.default_spool_dir(), self.dir)

    def test_start_has_no_jobs_option(self) -> None:
        # One solve at a time: the worker is sequential, so --jobs was removed and the
        # CLI must reject it instead of silently accepting concurrency.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                spool.main(["--dir", str(self.dir), "start", "--jobs", "2", "--drain"])

    def test_add_cli_enqueues_config_run(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            spool.main(["--dir", str(self.dir), "add", "configs/smoke.json", "--cores", "2"])
        self.assertEqual(self._count("queue"), 1)
        task = json.loads(next((self.dir / "queue").glob("*.task")).read_text())
        self.assertEqual(task["name"], "smoke")
        self.assertIn("run_rectangular_case.py", task["cmd"])
        self.assertIn("--cores 2", task["cmd"])


if __name__ == "__main__":
    unittest.main()
