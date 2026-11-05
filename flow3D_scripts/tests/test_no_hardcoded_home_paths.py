#!/usr/bin/env python3
"""No tracked Python file may embed a `/home/<user>` path.

Three scripts carried the authors' home directory as a literal -- the sibling
`lpno` checkout, its `scripts/` dir, and a runtime-cache path. On any
other machine each one fails in a way that looks like something else: an
`ImportError` for a package that is installed, or a figure quietly "skipped
(cases not on disk yet)". The convention is a `PINO_SIMULATIONS_ROOT` environment
override with a walk-up to the sibling checkout as the fallback, because the
obvious `Path(__file__).parents[N]` form is wrong from a git worktree (the sibling
is three levels further up from a worktree nested inside the repository).

This test is the guard that keeps the literal from coming back.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: One-off scripts the public copy excludes; listed so the guard is explicit about
#: what it tolerates rather than silently narrower than "every file".
TOLERATED = {
    "python/backfill_manifest_runtime.py",
}

HOME_LITERAL = re.compile(r"[\"'][^\"'\n]*/home/[A-Za-z0-9_.-]+/")


class NoHardcodedHomePathTests(unittest.TestCase):
    def test_no_python_file_embeds_a_home_path(self) -> None:
        offenders = []
        for sub in ("python", "validation", "tests", "bash"):
            for f in sorted((ROOT / sub).rglob("*")):
                if f.suffix not in (".py", ".sh") or not f.is_file():
                    continue
                rel = f.relative_to(ROOT).as_posix()
                if rel in TOLERATED or rel == "tests/test_no_hardcoded_home_paths.py":
                    continue
                for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if HOME_LITERAL.search(line):
                        offenders.append(f"{rel}:{n}: {line.strip()[:80]}")
        self.assertEqual(offenders, [], "hardcoded home paths:\n" + "\n".join(offenders))

    def test_archived_batches_carry_no_home_path_in_any_text_file(self) -> None:
        """`batches/` holds the tracked copies of loop drivers and state that ran from
        untracked `runs/batches/`. Those were written with absolute paths (386 loop
        configs named their frozen material by the authors' home directory;
        30 resource-usage records carried the training command line). The tracked
        copy must be portable: no `/home/<user>/` anywhere, quoted or not, in any
        text file — configs, CSVs, JSON state, logs, drivers."""
        bare = re.compile(r"/home/[A-Za-z0-9_.-]+/")
        offenders = []
        for f in sorted((ROOT / "batches").rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue          # a binary (figure) is not a place a path hides
            for n, line in enumerate(text.splitlines(), 1):
                if bare.search(line):
                    offenders.append(f"{f.relative_to(ROOT).as_posix()}:{n}: {line.strip()[:80]}")
                    break         # one line per file is enough to name it
        self.assertEqual(offenders, [], "home paths in batches/:\n" + "\n".join(offenders))

    def test_sibling_lookup_survives_a_worktree(self) -> None:
        """The walk-up must find `../lpno` even when ROOT is a worktree."""
        import sys
        sys.path.insert(0, str(ROOT / "python"))
        import local_inference as li
        found = li.lpno_root()
        if not found.is_dir():
            self.skipTest(f"no lpno checkout beside this repository ({found}); "
                          "the walk-up has nothing to find in a lone clone")
        # Either the env override or the walk-up; both must yield a directory that
        # actually holds the package, not a path assembled from ROOT.parent.
        self.assertTrue((found / "src" / "lpno").is_dir()
                        or (found / "lpno").is_dir(),
                        f"resolved {found}, which holds no lpno package")


if __name__ == "__main__":
    unittest.main()
