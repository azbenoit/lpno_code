"""AiiDA jobs declare their GPU/host memory so the spooler can pack them.

`gpu_spool seat` admits runs by declared memory, but `wrap_with_gpu_seat` passed
neither `--vram-mb` nor `--ram-mb`, so every AiiDA-launched job booked the whole
card exclusively plus a blind 32 GB of host RAM. Measured live 2026-07-22: an
eval needing ~12 GB queued behind `alsi_rigno_r03_m1p`, which had declared 8 GB
non-exclusively on a card with 22.4 GB physically free.

Defaults stay absent -- omitting the flags keeps the old exclusive booking, so a
run that has not measured its peak is never packed on optimistic numbers. The
asymmetry matters: under-declaring VRAM OOMs whatever it was packed beside,
while over-declaring host RAM only starves the queue.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aiida_pino.scripts import eval_adapter, kfold_adapter, train_adapter

CWD = Path("/tmp/job")
ADAPTERS = (train_adapter, kfold_adapter, eval_adapter)


def _profile(**kw):
    return {"cuda_device": "1", **kw}


def _flag_value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.__name__.split(".")[-1])
def test_no_declaration_keeps_the_exclusive_booking(adapter):
    """Unmeasured runs must not be packed on a guess."""
    argv = adapter.wrap_with_gpu_seat(["cmd"], _profile(), cwd=CWD)
    assert "--vram-mb" not in argv
    assert "--ram-mb" not in argv


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.__name__.split(".")[-1])
def test_declared_vram_is_passed_to_seat(adapter):
    argv = adapter.wrap_with_gpu_seat(["cmd"], _profile(gpu_vram_mb=12000), cwd=CWD)
    assert _flag_value(argv, "--vram-mb") == "12000"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.__name__.split(".")[-1])
def test_declared_host_ram_is_passed_to_seat(adapter):
    argv = adapter.wrap_with_gpu_seat(["cmd"], _profile(host_ram_mb=16384), cwd=CWD)
    assert _flag_value(argv, "--ram-mb") == "16384"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.__name__.split(".")[-1])
def test_both_declarations_precede_the_command_separator(adapter):
    """Flags belong to `seat`, not to the wrapped command."""
    argv = adapter.wrap_with_gpu_seat(["cmd"], _profile(gpu_vram_mb=8000, host_ram_mb=12000), cwd=CWD)
    sep = argv.index("--")
    assert argv.index("--vram-mb") < sep
    assert argv.index("--ram-mb") < sep
    assert argv[-1] == "cmd"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.__name__.split(".")[-1])
@pytest.mark.parametrize("bad", [0, -1, "big", None, True])
def test_unusable_declarations_fall_back_to_exclusive(adapter, bad):
    """A typo must book the whole card, never a zero-size reservation.

    `--vram-mb 0` would reserve nothing and let unlimited runs pile onto one
    card; silently ignoring the bad value is the safe direction.
    """
    argv = adapter.wrap_with_gpu_seat(["cmd"], _profile(gpu_vram_mb=bad), cwd=CWD)
    assert "--vram-mb" not in argv


def test_profiles_accept_the_new_keys():
    from aiida_pino.utils.profiles import DEFAULT_TRAINING_PROFILE, merge_training_profile

    assert "gpu_vram_mb" in DEFAULT_TRAINING_PROFILE
    assert "host_ram_mb" in DEFAULT_TRAINING_PROFILE
    assert DEFAULT_TRAINING_PROFILE["gpu_vram_mb"] is None, "unmeasured by default"
    merged = merge_training_profile({
        "config_name": "config_flow3D_RIGNO_randompath", "run_name": "r",
        "hydra_overrides": [], "cuda_device": "0", "wandb_offline": True,
        "gpu_vram_mb": 12000, "host_ram_mb": 16384,
    })
    assert merged["gpu_vram_mb"] == 12000
    assert merged["host_ram_mb"] == 16384


def test_evaluation_profile_accepts_the_new_keys():
    from aiida_pino.utils.model_registry import (
        EVALUATION_OVERRIDE_KEYS,
        validate_evaluation_profile,
    )

    assert {"gpu_vram_mb", "host_ram_mb"} <= EVALUATION_OVERRIDE_KEYS
    profile = validate_evaluation_profile({"gpu_vram_mb": 12000, "host_ram_mb": 16384})
    assert profile["gpu_vram_mb"] == 12000
    assert profile["host_ram_mb"] == 16384


# --------------------------------------------------------------------------- #
# `gpu_seat_priority`: priority buys queue-jumping at the cost of a deadline.
#
# gpu_spool waits forever for a plain seat (DEFAULT_SEAT_MAX_WAIT_S = 0) but
# gives up after DEFAULT_PRIORITY_MAX_WAIT_S = 3600s with --priority, "so CI
# reports a cause". The eval adapter hardcoded --priority, so on 2026-07-29 two
# evals of a finished model died at 3601s while both cards stayed full -- an
# ordinary seat would simply have waited. Keep priority the default (evals are
# short and should go in ahead of queued trainings) but let a caller opt out
# when the machine is saturated and waiting beats failing.
# --------------------------------------------------------------------------- #
def test_eval_is_not_priority_by_default():
    """Waiting beats failing: a plain seat waits forever, priority dies at 3600s."""
    argv = eval_adapter.wrap_with_gpu_seat(["cmd"], _profile(), cwd=CWD)
    assert "--priority" not in argv


def test_eval_priority_can_be_re_enabled():
    argv = eval_adapter.wrap_with_gpu_seat(
        ["cmd"], _profile(gpu_seat_priority=True), cwd=CWD)
    assert "--priority" in argv


def test_eval_priority_can_be_disabled():
    argv = eval_adapter.wrap_with_gpu_seat(
        ["cmd"], _profile(gpu_seat_priority=False), cwd=CWD)
    assert "--priority" not in argv, "cannot opt out of the 3600s priority deadline"


def test_disabling_priority_keeps_the_memory_flags():
    argv = eval_adapter.wrap_with_gpu_seat(
        ["cmd"], _profile(gpu_seat_priority=False, gpu_vram_mb=8000), cwd=CWD)
    assert "--priority" not in argv
    assert argv[argv.index("--vram-mb") + 1] == "8000"
