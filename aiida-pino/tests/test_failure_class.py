"""`failure_class` and `live_log`: triage without opening a workdir.

`exit_status 300` is uniform across a config crash, an OOM and a stall-kill, so
diagnosis always meant reading the workdir's live-tail file -- deleted with the
workdir, and reachable only through node -> calcjob -> remote workdir -> log.
Both are fixed at the source: the class is decided at parse time from stdout
AiiDA retains, and the chain is one call.

Every pattern in FAILURE_PATTERNS came from a real failed run on this profile
(2026-08-10). The backfill over 652 failed calcjobs: 418 no_output, 139
waiting_for_resources, 42 unclassified, 21 oom_gpu, 18 config_error, 9
disk_full, 5 python_exception.
"""

from __future__ import annotations

import pytest

from aiida_pino.parsers.common import classify_failure


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("CUDA out of memory. Tried to allocate 2.00 GiB", "oom_gpu"),
        ("torch.cuda.OutOfMemoryError: CUDA out of memory", "oom_gpu"),
        ("OSError: [Errno 28] No space left on device", "disk_full"),
        ("[gpu-spool] waiting for 3000 MiB on GPU 1 + 8000 MiB host RAM", "waiting_for_resources"),
        ("Error executing job with overrides: [custom.name=x]", "config_error"),
        ("ModuleNotFoundError: No module named 'physicsnemo'", "import_error"),
        ("Traceback (most recent call last):\nValueError: bad", "python_exception"),
    ],
)
def test_each_class_comes_from_a_real_failure(stdout, expected):
    assert classify_failure(stdout) == expected


def test_no_output_and_unclassified_are_different_answers():
    """One is a retrieval problem, the other a gap in the table.

    Collapsing them would hide 418 jobs whose stdout was never retained behind
    the same label as 42 whose output simply matches nothing yet.
    """
    assert classify_failure("") == "no_output"
    assert classify_failure(None) == "no_output"
    assert classify_failure("   \n  ") == "no_output"
    assert classify_failure("training proceeded normally") == "unclassified"


def test_the_specific_beats_the_generic():
    """An OOM traceback is also a Traceback; order in the table decides."""
    oom = "Traceback (most recent call last):\ntorch.cuda.OutOfMemoryError: CUDA out of memory"
    assert classify_failure(oom) == "oom_gpu"
    disk = "Traceback (most recent call last):\nOSError: [Errno 28] No space left on device"
    assert classify_failure(disk) == "disk_full"


def test_ansi_is_stripped_before_matching():
    """wandb writes ANSI-wrapped lines; a raw substring test missed all of them
    once already (the failure_tail backfill, 2026-08-09)."""
    assert classify_failure("\x1b[31mCUDA out of memory\x1b[0m") == "oom_gpu"


def test_live_log_says_why_rather_than_returning_a_blank(aiida_profile_clean):
    """A missing path is an answer, not an absence.

    "no calcjob below this node", "never reached the scheduler" and "workdir
    cleaned up" are three different diagnoses, and a monitor that gets None for
    all three learns nothing. Verified against the live DB the same day: a
    running TrainModelWorkChain resolves through its calcjob to
    .aiida_pino_live_tail.log with a ready-to-paste tail command.
    """
    from aiida import orm

    from aiida_pino.utils.queryable_paths import live_log

    lonely = orm.Data().store()
    info = live_log(lonely.pk)
    assert info["path"] is None
    assert info["reason"] == "no calcjob below this node"
    assert info["calcjob_pk"] is None
