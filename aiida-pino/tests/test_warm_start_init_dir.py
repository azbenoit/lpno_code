"""Warm start goes through physicsnemo's `initialization_network_dir`.

`initialization_network_dir` is the documented mechanism (physicsnemo's own
fpga examples use it): load weights from another run's directory, start a fresh
optimizer at step 0. That is exactly what a warm start with a *changed* LR
schedule wants, and it leaves `network_dir` clean so nothing is mistaken for a
resume.

Consequence for callers: `max_steps` counts steps in THIS run, from 0 -- not a
total across runs.

Scope note (2026-07-23): this file pins the weights-only rendering only.
Continuing a run with its optimizer and step counter intact takes the other
path -- seed `network_dir`, pass no init_dir -- chosen by `plan_warm_start` and
covered by `test_resume_optimizer.py`. This docstring used to claim seeding was
measured broken on one run (`l2_relative_error_T = 0.9055`); that run and a second
were later traced to a `host_ram_mb` cgroup SIGKILL (the weights were never the
variable), so that claim is withdrawn.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aiida_pino.scripts.train_adapter import build_train_command

from aiida_pino.utils.profiles import merge_training_profile


def _base():
    return merge_training_profile({
        "config_name": "config_flow3D_RIGNO_randompath",
        "run_name": "warm",
        "hydra_overrides": [],
        "cuda_device": "0",
        "wandb_offline": True,
    })


def _cmd(profile, init_dir=None):
    return build_train_command(
        {**_base(), **profile},
        Path("/tmp/train"),
        Path("/tmp/val"),
        Path("/tmp/out"),
        init_dir=init_dir,
    )


def _value(cmd, key):
    for token in cmd:
        if token.startswith(f"{key}="):
            return token.split("=", 1)[1]
    return None


def test_init_dir_sets_initialization_network_dir():
    cmd = _cmd({}, init_dir=Path("/tmp/init_model"))
    assert _value(cmd, "initialization_network_dir") == "/tmp/init_model"


def test_no_init_dir_leaves_the_key_unset():
    """A cold start must not mention the key at all."""
    cmd = _cmd({})
    assert _value(cmd, "initialization_network_dir") is None


def test_network_dir_still_points_at_the_output_dir():
    """Warm start must not redirect where checkpoints are written."""
    cmd = _cmd({}, init_dir=Path("/tmp/init_model"))
    assert _value(cmd, "network_dir") == "/tmp/out"


def test_absolute_path_is_used_verbatim():
    cmd = _cmd({}, init_dir=Path("/some/where/init"))
    assert _value(cmd, "initialization_network_dir") == "/some/where/init"


@pytest.mark.parametrize("steps", [1000, 79822])
def test_max_steps_is_unchanged_by_warm_start(steps):
    """max_steps counts this run's steps; the adapter must not adjust it."""
    cmd = _cmd({"max_steps": steps}, init_dir=Path("/tmp/init_model"))
    assert _value(cmd, "training.max_steps") == str(steps)
