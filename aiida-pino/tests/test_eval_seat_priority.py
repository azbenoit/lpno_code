"""GPU seat priority is an explicit opt-in, and it buys the jump WITHOUT a deadline.

`gpu_spool seat --priority` does two things: it parks the queue-jump marker, and
it swaps the unbounded wait for `DEFAULT_PRIORITY_MAX_WAIT_S = 3600` "so CI
reports a cause". The second is right for CI and fatal for a queued job on a
saturated box -- two evals died at 3601 s on 2026-07-29, and training 99083 died
the same way at 3602 s on 2026-09-09 waiting for a seat that was ~33 h out.

Nothing here wants the CI deadline, so every adapter that asks for priority also
passes `--max-wait-s 0` (falsy at gpu_spool's deadline check = wait forever).
The queue-jump and the deadline were never one feature; only the flag conflated
them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aiida_pino.scripts import eval_adapter, kfold_adapter, train_adapter

CWD = Path("/tmp/job")


def _profile(**kw):
    return {"cuda_device": "0", **kw}


def _seat_flags(argv):
    return argv[: argv.index("--")]


def test_eval_seat_is_not_priority_by_default():
    argv = eval_adapter.wrap_with_gpu_seat(["cmd"], _profile(), cwd=CWD)
    assert "--priority" not in _seat_flags(argv)


def test_eval_seat_priority_is_opt_in():
    argv = eval_adapter.wrap_with_gpu_seat(["cmd"], _profile(gpu_seat_priority=True), cwd=CWD)
    assert "--priority" in _seat_flags(argv)


@pytest.mark.parametrize("adapter", [train_adapter, kfold_adapter], ids=lambda a: a.__name__.split(".")[-1])
def test_training_seats_are_not_priority(adapter):
    argv = adapter.wrap_with_gpu_seat(["cmd"], _profile(), cwd=CWD)
    assert "--priority" not in _seat_flags(argv)


def test_priority_coexists_with_declared_memory():
    argv = eval_adapter.wrap_with_gpu_seat(
        ["cmd"], _profile(gpu_seat_priority=True, gpu_vram_mb=6000), cwd=CWD
    )
    flags = _seat_flags(argv)
    assert "--priority" in flags
    assert flags[flags.index("--vram-mb") + 1] == "6000"


def test_priority_flag_precedes_the_command_separator():
    argv = eval_adapter.wrap_with_gpu_seat(["cmd"], _profile(gpu_seat_priority=True), cwd=CWD)
    assert argv.index("--priority") < argv.index("--")
    assert argv[-1] == "cmd"


@pytest.mark.parametrize("adapter", [eval_adapter, train_adapter], ids=lambda a: a.__name__.split(".")[-1])
def test_priority_does_not_buy_a_deadline(adapter):
    """The queue-jump must not smuggle in gpu_spool's 3600 s CI deadline.

    One training (exit 500) died at 3602 s without ever seating,
    while two sibling arms with identical profiles and no priority flag were still
    alive 12 h later -- the control that isolates the flag rather than the box.
    """
    flags = _seat_flags(adapter.wrap_with_gpu_seat(["cmd"], _profile(gpu_seat_priority=True), cwd=CWD))
    assert "--priority" in flags
    assert flags[flags.index("--max-wait-s") + 1] == "0", "priority must wait indefinitely"


@pytest.mark.parametrize("adapter", [eval_adapter, train_adapter, kfold_adapter],
                         ids=lambda a: a.__name__.split(".")[-1])
def test_an_ordinary_seat_needs_no_deadline_flag(adapter):
    """DEFAULT_SEAT_MAX_WAIT_S is already 0, so a non-priority seat says nothing."""
    assert "--max-wait-s" not in _seat_flags(adapter.wrap_with_gpu_seat(["cmd"], _profile(), cwd=CWD))
