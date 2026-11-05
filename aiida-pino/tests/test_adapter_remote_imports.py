"""A staged adapter must import with ONLY the files that are staged beside it.

`stage_script` uploads the adapter plus `live_tail.py` and nothing else. On this
box the bug is invisible: `aiida_pino` is editable-installed, so
`eval_adapter.py`'s `from aiida_pino.utils import parse_stdout_metric_dicts`
succeeds and the fallback below it never runs. On a compute box there is no
`aiida_pino` at all, the fallback runs, and it reaches for
`Path(__file__).resolve().parents[1] / "utils" / "profiles.py"` -- which is not
staged. That raises FileNotFoundError from `exec_module`, which the enclosing
`except ImportError` does not catch, so every remote eval dies at import.

These tests recreate the compute box: the staged files copied alone into a
directory, with `aiida_pino` and `aiida` blocked from import.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "aiida_pino" / "scripts"
#: Read from the source of truth, so adding a helper there cannot leave this
#: test staging a different set than a real job gets.
from aiida_pino.calculations.base import STAGED_HELPERS

ADAPTERS_STAGED_TO_COMPUTE_BOXES = ["eval_adapter.py", "train_adapter.py", "kfold_adapter.py"]


def _sandbox(tmp_path: Path, adapter: str) -> Path:
    """Exactly what `stage_script` uploads: the adapter and live_tail.py."""
    box = tmp_path / "sandbox"
    box.mkdir()
    shutil.copy(SCRIPTS / adapter, box / adapter)
    for helper in STAGED_HELPERS:
        if helper != adapter:
            shutil.copy(SCRIPTS / helper, box / helper)
    return box


def _import_in_sandbox(box: Path, adapter: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(
        f"""
        import importlib.util, sys

        blocked = ("aiida_pino", "aiida")

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if any(name == b or name.startswith(b + ".") for b in blocked):
                    raise ImportError("blocked import: " + name)
                return None

        sys.meta_path.insert(0, Blocker())
        spec = importlib.util.spec_from_file_location("staged_adapter", r"{box / adapter}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("IMPORTED")
        print("HAS_PARSER", hasattr(module, "parse_stdout_metric_dicts"))
        """
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.mark.parametrize("adapter", ADAPTERS_STAGED_TO_COMPUTE_BOXES)
def test_a_staged_adapter_imports_with_no_aiida_pino_installed(tmp_path, adapter):
    box = _sandbox(tmp_path, adapter)
    proc = _import_in_sandbox(box, adapter)
    assert "IMPORTED" in proc.stdout, (
        f"{adapter} cannot import on a compute box.\nstderr:\n{proc.stderr[-2000:]}"
    )


def test_the_eval_adapter_still_gets_its_metric_parser(tmp_path):
    """Importing is not enough: the parser must actually be bound."""
    proc = _import_in_sandbox(_sandbox(tmp_path, "eval_adapter.py"), "eval_adapter.py")
    assert "HAS_PARSER True" in proc.stdout, f"stderr:\n{proc.stderr[-2000:]}"


@pytest.mark.parametrize("helper", ["live_tail.py", "staged_dataset.py"])
def test_every_staged_helper_needs_nothing_but_the_standard_library(tmp_path, helper):
    """These are the files adapters are allowed to lean on, so each must be a leaf."""
    box = tmp_path / "alone"
    box.mkdir()
    shutil.copy(SCRIPTS / helper, box / helper)
    proc = _import_in_sandbox(box, helper)
    assert "IMPORTED" in proc.stdout, f"{helper} is not standalone.\nstderr:\n{proc.stderr[-2000:]}"


def test_the_metric_parser_is_in_a_staged_helper(tmp_path):
    box = tmp_path / "alone"
    box.mkdir()
    shutil.copy(SCRIPTS / "live_tail.py", box / "live_tail.py")
    proc = _import_in_sandbox(box, "live_tail.py")
    assert "HAS_PARSER True" in proc.stdout, "the metric parser lives here now"
