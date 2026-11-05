#!/usr/bin/env python3
"""Memory/disk guards switch between enforce and warn through one literal.

Memory and disk limits are hardware-dependent, so the published copy ships the guards
as warnings. On a shared machine the 30 GB disk floor and the 25 GB MemAvailable gate
can be made binding; `python/resource_guards.py` carries the one `DEFAULT_MODE`
literal, and `F3D_RESOURCE_GUARDS` overrides it either way.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import resource_guards  # noqa: E402
import run_rectangular_case as rectangular  # noqa: E402


def _env(value):
    return unittest.mock.patch.dict("os.environ", {resource_guards.ENV: value})


class ResourceGuardSwitchTests(unittest.TestCase):
    def test_the_default_is_one_literal_the_assembly_can_rewrite(self) -> None:
        # "enforce" in this checkout, "warn" in the public copy (the assembly rewrites the
        # literal); this test runs in both, so it pins the shape and that the default is
        # what `mode()` returns with the variable unset -- not the value.
        source = (ROOT / "python" / "resource_guards.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("DEFAULT_MODE = "), 1)
        self.assertIn(resource_guards.DEFAULT_MODE, ("enforce", "warn"))
        with unittest.mock.patch.dict("os.environ"):
            os.environ.pop(resource_guards.ENV, None)
            self.assertEqual(resource_guards.mode(), resource_guards.DEFAULT_MODE)
        with _env("enforce"), self.assertRaisesRegex(RuntimeError, "no room"):
            resource_guards.resource_guard("no room")

    def test_warn_prints_and_returns(self) -> None:
        err = io.StringIO()
        with _env("warn"), contextlib.redirect_stderr(err):
            resource_guards.resource_guard("no room", exc=SystemExit)   # must not raise
        self.assertIn("[resource-guard] WARNING: no room", err.getvalue())
        self.assertIn("not enforced", err.getvalue())

    def test_an_unknown_mode_is_refused_not_guessed(self) -> None:
        with _env("maybe"), self.assertRaisesRegex(SystemExit, "enforce"):
            resource_guards.mode()

    def test_disk_floor_warns_once_and_does_not_wait(self) -> None:
        """Under warn the floor is a single reading and a message: no 30-minute wait
        on a reader's laptop, no refusal after it."""
        readings = iter([1.0, 1.0, 500.0])
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, _env("warn"), contextlib.redirect_stderr(err):
            with unittest.mock.patch.object(rectangular, "free_space_gb", lambda _p: next(readings)):
                rectangular.require_free_space(Path(tmp), 30.0, wait_s=600.0, poll_s=0.0)
        self.assertEqual(list(readings), [1.0, 500.0], "warn mode must read once and proceed")
        self.assertIn("free disk", err.getvalue())

    def test_headroom_gate_warns_once_and_does_not_wait(self) -> None:
        readings = iter([4.0, 4.0, 90.0])
        err = io.StringIO()
        with _env("warn"), contextlib.redirect_stderr(err):
            with unittest.mock.patch.object(rectangular, "available_memory_gb", lambda: next(readings)):
                rectangular.require_machine_headroom(min_mem_gb=25.0, max_load=0.0, wait_s=600.0, poll_s=0.0)
        self.assertEqual(list(readings), [4.0, 90.0])
        self.assertIn("available RAM", err.getvalue())

    def test_enforced_guards_still_refuse(self) -> None:
        """Under `enforce`, the guards still refuse exactly as before."""
        with tempfile.TemporaryDirectory() as tmp, _env("enforce"):
            with unittest.mock.patch.object(rectangular, "free_space_gb", lambda _p: 1.0):
                with self.assertRaisesRegex(RuntimeError, "free disk"):
                    rectangular.require_free_space(Path(tmp), 30.0, wait_s=0.0)
            with unittest.mock.patch.object(rectangular, "available_memory_gb", lambda: 4.0):
                with self.assertRaisesRegex(RuntimeError, "available RAM"):
                    rectangular.require_machine_headroom(min_mem_gb=25.0, max_load=0.0, wait_s=0.0)

    def test_exporter_scratch_check_routes_through_the_switch(self) -> None:
        """The NPZ exporter's scratch-space refusal is a disk guard too; it runs under
        pvpython, so this is checked statically."""
        source = (ROOT / "python" / "export_all_timesteps_to_npz.py").read_text(encoding="utf-8")
        self.assertIn("from resource_guards import resource_guard", source)
        self.assertIn("resource_guard(", source)
        self.assertNotIn('raise SystemExit(\n            f"Need ~', source)


if __name__ == "__main__":
    unittest.main()
