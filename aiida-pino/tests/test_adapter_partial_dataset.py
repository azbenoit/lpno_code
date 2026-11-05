"""A manifest is not a dataset: the members it names must be there.

`validate_dataset_input_dir` returned as soon as it saw `manifest.json`. A
directory holding only that file therefore passed, and training proceeded on a
dataset with no members — silently, producing a model rather than an error.

That is reachable: a staging failure clears the destination before copying and
can leave exactly the wreckage that keeps `manifest.json` (transfer order is
`os.listdir` order), and anything that removes files on the compute box by hand
does the same. Nothing repairs a remote cache, because there is no object store
there to rebuild from.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "aiida_pino" / "scripts"

#: Every adapter that validates a staged dataset. The check lived in ONE of these
#: and the other two went on accepting wreckage, so the test covers all three --
#: pinning shared semantics across every branch, not just the one that was fixed.
ADAPTERS = ["train_adapter.py", "eval_adapter.py", "kfold_adapter.py"]


def _adapter(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collection(root: Path, members: list[str], *, create: list[str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "source_kind": "dataset_collection",
                "collection_uuid": "abc",
                "simulation_count": len(members),
                "members": [{"relative_path": m, "manifest_file": f"{m}/manifest.json"} for m in members],
            }
        )
    )
    for name in create if create is not None else members:
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "manifest.json").write_text("{}")
    return root


@pytest.mark.parametrize("name", ADAPTERS)
def test_a_complete_collection_passes(tmp_path, name):
    adapter = _adapter(name)
    path = _collection(tmp_path / "ds", ["members/member_000", "members/member_001"])
    adapter.validate_dataset_input_dir(path, label="train_dataset")


@pytest.mark.parametrize("name", ADAPTERS)
def test_a_manifest_with_no_members_present_is_refused(tmp_path, name):
    """The wreckage case: manifest.json survived, everything else did not."""
    adapter = _adapter(name)
    path = _collection(tmp_path / "ds", ["members/member_000", "members/member_001"], create=[])
    with pytest.raises(SystemExit) as excinfo:
        adapter.validate_dataset_input_dir(path, label="train_dataset")
    message = str(excinfo.value)
    assert "member" in message.lower()
    assert "2" in message


@pytest.mark.parametrize("name", ADAPTERS)
def test_a_partially_transferred_collection_is_refused(tmp_path, name):
    """One member of two: a count check on the directory would pass this."""
    adapter = _adapter(name)
    path = _collection(
        tmp_path / "ds",
        ["members/member_000", "members/member_001"],
        create=["members/member_000"],
    )
    with pytest.raises(SystemExit) as excinfo:
        adapter.validate_dataset_input_dir(path, label="train_dataset")
    assert "member_001" in str(excinfo.value)


@pytest.mark.parametrize("name", ADAPTERS)
def test_a_manifest_without_a_member_list_is_left_alone(tmp_path, name):
    """Per-simulation manifests name no members; they must not be rejected."""
    adapter = _adapter(name)
    root = tmp_path / "sim"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"format": 1, "source_kind": "simulation"}))
    adapter.validate_dataset_input_dir(root, label="train_dataset")


def _python_without_aiida_pino(tmp_path: Path) -> Path:
    """An interpreter that cannot import `aiida_pino`, as on a compute machine.

    The test is about the machine where the package is absent. `/usr/bin/python3` is
    not such a machine whenever `PYTHONPATH` names the source tree or the package is
    installed into it, and then the probe succeeds and the announcement never prints.
    `-I -S` drops `PYTHONPATH`, the user site and site-packages, so only the standard
    library is importable.
    """
    import sys

    wrapper = tmp_path / "python-without-aiida-pino"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n')
    wrapper.chmod(0o755)
    return wrapper


@pytest.mark.parametrize("name", ADAPTERS)
def test_the_repair_says_so_when_it_cannot_run_and_stays_non_fatal(name, tmp_path):
    """Renamed 2026-08-25; was `..._is_quiet_when_it_cannot_run`.

    The original concern was real: unsilenced, this printed a raw "Error while finding
    module specification" as the FIRST error text in an otherwise healthy run's log and
    sent a reviewer looking in the wrong place.

    But being quiet ALSO hid that the repair is permanently inert on every remote computer
    -- `aiida_pino` is not installed on a compute box, so it has never once run on the
    machines where staged-cache corruption actually happens, which is what the repair path
    exists for. A one-line statement is not the noise the original test was defending
    against; a raw traceback was.

    Asserted by RUNNING the command rather than matching its tail: the property is
    "non-fatal", and `endswith("|| true")` was a check on shape that broke the moment the
    shape legitimately changed.
    """
    import os
    import subprocess

    adapter = _adapter(name)
    python = _python_without_aiida_pino(tmp_path)
    command = adapter._shared_runtime_repair_command(str(python))
    assert "2>/dev/null" in command, "the repair call's own stderr must still be dropped"

    probe = subprocess.run([str(python), "-c", "import aiida_pino"], capture_output=True, text=True)
    assert probe.returncode != 0, "sanity: the stand-in interpreter must not import aiida_pino"

    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env=env)
    assert result.returncode == 0, "the failure must stay non-fatal"
    assert "Error while finding module specification" not in result.stdout + result.stderr, (
        "the raw import traceback is the noise this has always guarded against"
    )
    assert "[runtime-repair] unavailable on this computer" in result.stdout, (
        "a permanent skip must announce itself, not read as 'repair ran, found nothing'"
    )


def test_every_adapter_builds_that_command_the_same_way():
    """Three copies of one line is how the dataset validator drifted."""
    commands = {name: _adapter(name)._shared_runtime_repair_command("/x/py") for name in ADAPTERS}
    assert len(set(commands.values())) == 1, commands
