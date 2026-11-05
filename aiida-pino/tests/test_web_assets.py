"""Static checks on the console's front end, which no Python test executes.

`web/public/app.js` is served straight to the browser with no build step, so
nothing in this repo's 800-odd tests ever runs a line of it. `node --check`
parses it, which is not enough: an assignment to an *undeclared* name is legal
syntax and a `ReferenceError` at run time under `"use strict"`.

That is not hypothetical. Commit 890865d assigned `folderTotals` and read it
back without declaring it, and every Datasets load threw
`ReferenceError: folderTotals is not defined` and rendered a red notice instead
of any of the 836 datasets — for four commits, through a full test suite that
could not see it, until a reviewer opened the tab.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PUBLIC = Path(__file__).resolve().parent.parent / "web" / "public"
SCRIPTS = sorted(PUBLIC.glob("*.js"))


def test_public_scripts_exist():
    assert SCRIPTS, f"no front-end scripts found under {PUBLIC}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script: Path):
    """`node --check`: catches a syntax error before the browser does."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr.strip()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_has_no_undeclared_names(script: Path):
    """`no-undef`: catches the class of bug that took the Datasets tab down.

    Skipped rather than failed when eslint is absent, because a missing linter
    is an environment gap, not a defect in the file — but it runs on this
    workstation and in any environment that has eslint on PATH.
    """
    eslint = shutil.which("eslint")
    if eslint is None:
        pytest.skip("eslint not installed")
    result = subprocess.run(
        [eslint, "--format", "unix", str(script)],
        capture_output=True,
        text=True,
        cwd=str(PUBLIC.parent.parent),
    )
    assert result.returncode == 0, f"eslint findings in {script.name}:\n{result.stdout.strip()}"
