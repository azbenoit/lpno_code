"""A workchain that refuses an input must say what was wrong with it.

The exit code names the input ("Invalid `training_profile`.") and stops there.
The ValueError that carries the actual reason was caught and dropped, so
`verdi process report` printed `No log messages recorded for this entry`.

Measured cost, 2026-08-05: four TrainModelWorkChains finished exit 300 within two
minutes of each other. All four declared `host_ram_mb=32768` against a guard that
then demanded 49152; `e8804c4` recalibrated it two hours later as having
"rejected a PROVEN configuration". Neither fact was visible from the run. The
response was to resubmit twice and then to inflate the declaration to 49152,
which was admitted and died for an unrelated reason.

22 sites across 9 workflows had the same shape, not the 4 the report named.
"""

from __future__ import annotations

import re
from pathlib import Path

from aiida import orm
from aiida.engine import run_get_node

from aiida_pino.workflows.train_model import TrainModelWorkChain

WORKFLOWS = Path(__file__).resolve().parent.parent / "src" / "aiida_pino" / "workflows"

#: `except ValueError:` whose whole handler is `return self.exit_codes.X` — the
#: reason is bound to nothing and cannot be reported.
SILENT = re.compile(r"^([ \t]*)except ValueError:[ \t]*\n[ \t]+return self\.exit_codes\.\w+[ \t]*$", re.M)


def _reports(node) -> str:
    return "\n".join(log.message for log in orm.Log.collection.get_logs_for(node))


def test_no_workflow_swallows_the_reason():
    """The whole class, not the two entry points that were reproduced."""
    offenders = {
        path.name: len(SILENT.findall(path.read_text()))
        for path in sorted(WORKFLOWS.glob("*.py"))
        if SILENT.search(path.read_text())
    }
    assert not offenders, f"validation failures with no reported reason: {offenders}"


def test_a_rejected_training_profile_reports_what_was_wrong(
    fake_python_executable, local_code_factory, pino_dataset_factory
):
    """Exit 300 is not enough: the run must carry the reason."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-reject-why")
    dataset = pino_dataset_factory("reject-why")

    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        # Required keys missing: the shape of every profile rejection.
        training_profile=orm.Dict(dict={"config_name": "config_NS_energy_PINO_flow3D"}),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )

    assert node.exit_status == TrainModelWorkChain.exit_codes.ERROR_INVALID_TRAINING_PROFILE.status
    reported = _reports(node)
    assert reported.strip(), "exit 300 with no log messages — the incident this test exists for"
    assert "missing required keys" in reported, reported
