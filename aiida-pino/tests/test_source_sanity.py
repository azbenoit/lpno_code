"""A name that is called but never defined must not reach a run.

The adapters under `src/aiida_pino/scripts/` are uploaded and executed in a
remote workdir, so an undefined name is not an import error here — it is a
`NameError` at the moment that line runs, which for a summary written at the
end is *after* the expensive part is over.

Commit ca0a9bd added two calls to `_report_declaration_overrun` and never wrote
the function. Every training reached the end, then died on the final line of
`train_adapter.py` writing `train_summary.json`. The workchain reported exit 300
and materialized no model node. Cost: 8 sweep arms x 20 000 steps, about 36
GPU-hours. Nothing in the suite could see it — no test imports the adapter's
`_main`, and the tests that do import the module never reach line 716.

`ruff --select F821` reads the whole module and flags a load of a name that no
scope defines. Run against `train_adapter.py` as of 3d897fe it reports both call
sites; against the current tree it reports nothing, so this gate costs no noise.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def _ruff() -> list[str]:
    if shutil.which("ruff"):
        return ["ruff"]
    return [sys.executable, "-m", "ruff"]


def _ruff_or_skip() -> list[str]:
    """The ruff command, or a skip when no working ruff executable is available.

    `python -m ruff` exits non-zero without raising when the package is absent, or
    when the package is installed without its binary, so availability is probed
    with `--version` before the real check runs.
    """
    command = _ruff()
    try:
        probe = subprocess.run([*command, "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("ruff is not installed")
    if probe.returncode != 0:
        pytest.skip("no working ruff executable (`ruff --version` failed): "
                    + (probe.stderr.strip().splitlines() or ["no output"])[-1])
    return command


def test_no_undefined_names_in_src():
    """F821 undefined name, F811 redefinition, F822 undefined `__all__` entry."""
    result = subprocess.run(
        [*_ruff_or_skip(), "check", "--select", "F821,F811,F822", "--no-cache", "--output-format", "concise", str(SRC)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "a name is called but never defined — this is the 36-GPU-hour class of bug:\n"
        + (result.stdout.strip() or result.stderr.strip())
    )
