"""Warm start continues the optimizer when physicsnemo can actually restore it.

`initialization_network_dir` loads weights and starts a *fresh* optimizer at
step 0: the LR schedule restarts at warmup and every step-indexed curriculum
replays from the beginning. Continuing a run wants the opposite -- the
optimizer, the scheduler and the step counter carried over -- which
physicsnemo only does for checkpoints it finds in `network_dir`.

That resume path is all-or-nothing. `Trainer._load_optimizer` runs five
`load_state_dict` calls inside one bare `except:` and falls back to `step = 0`,
so a single raiser silently costs the step counter *and* leaves the optimizer
half-loaded. The realistic raiser is an empty scaler state: torch's `GradScaler`
and physicsnemo's `DerivScalers` both raise `RuntimeError` on an empty dict when
enabled, and both save `{}` when disabled -- so a checkpoint written with
`amp.enabled=false` cannot be resumed by a run with AMP on (the config default).

Hence: resume when the checkpoint will load cleanly, warm-start otherwise.
"""
from __future__ import annotations

import pytest

from aiida_pino.scripts.train_adapter import amp_enabled_in_profile, resume_blockers


def _ckpt(step=6000, scaler=None, deriv=None):
    return {
        "step": step,
        "optimizer_state_dict": {"state": {0: {"exp_avg": 1}}},
        "aggregator_state_dict": {},
        "scheduler_state_dict": {"last_epoch": step},
        "scaler_state_dict": {} if scaler is None else scaler,
        "deriv_scalers_state_dict": {} if deriv is None else deriv,
    }


def test_amp_off_with_empty_scalers_is_resumable():
    """The disabled scalers no-op on an empty dict, so all five loads succeed."""
    assert resume_blockers(_ckpt(), amp_enabled=False) is None


def test_amp_on_with_empty_scalers_is_blocked():
    """An enabled GradScaler raises on {} -- bare except -> step silently 0."""
    blocker = resume_blockers(_ckpt(), amp_enabled=True)
    assert blocker is not None
    assert "scaler_state_dict" in blocker


def test_amp_on_with_populated_scalers_is_resumable():
    assert resume_blockers(
        _ckpt(scaler={"scale": 65536.0}, deriv={"T": {"scale": 1.0}}),
        amp_enabled=True,
    ) is None


def test_step_zero_is_not_worth_resuming():
    """Nothing to carry over; the weights-only path is equivalent and simpler."""
    assert resume_blockers(_ckpt(step=0), amp_enabled=False) is not None


def test_missing_step_key_is_blocked():
    ckpt = _ckpt()
    del ckpt["step"]
    assert resume_blockers(ckpt, amp_enabled=False) is not None


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ([], True),  # config default is amp.enabled=true
        (["amp.enabled=false"], False),
        (["amp.enabled=False"], False),
        (["amp.enabled=true"], True),
        (["amp.enabled=true", "amp.enabled=false"], False),  # last wins
        (["training.max_steps=10"], True),
    ],
)
def test_amp_enabled_is_read_from_hydra_overrides(overrides, expected):
    assert amp_enabled_in_profile({"hydra_overrides": overrides}) is expected


def test_gpu_seat_priority_is_opt_in():
    """Admission is a poll race; a restart needs to jump it, but only on ask."""
    from aiida_pino.scripts.train_adapter import wrap_with_gpu_seat

    base = {"cuda_device": "1", "hydra_overrides": []}
    assert "--priority" not in wrap_with_gpu_seat(["train"], base)
    assert "--priority" in wrap_with_gpu_seat(["train"], {**base, "gpu_seat_priority": True})


def test_scheduler_horizon_mismatch_forces_weights_only():
    """A donor checkpoint from a different run length must NOT be resumed.

    Measured 2026-08-14: a 20k-step donor's one-cycle scheduler
    (total_steps=20200) was restored into a 32k-step warm-started run, which
    then crashed at step 20201 with "Tried to step 20201 times. The specified
    number of total steps is 20200" -- ~200 wasted steps and a dead A/B arm.
    A horizon mismatch means this is a cross-run transfer, not a resume.
    """
    ckpt = _ckpt(step=20000)
    ckpt["scheduler_state_dict"]["total_steps"] = 20200
    blocker = resume_blockers(ckpt, amp_enabled=False, max_steps=32000)
    assert blocker is not None and "total_steps" in blocker

    # Same horizon (a genuinely killed 32k run at step 20000): resumable.
    ckpt["scheduler_state_dict"]["total_steps"] = 32200
    assert resume_blockers(ckpt, amp_enabled=False, max_steps=32000) is None


def test_checkpoint_at_or_past_max_steps_forces_weights_only():
    """A checkpoint at (or past) this run's max_steps has nothing to continue:
    the trainer would exit immediately and 'produce' the donor unchanged."""
    blocker = resume_blockers(_ckpt(step=32000), amp_enabled=False, max_steps=32000)
    assert blocker is not None

    # Plenty of steps left and no horizon recorded: still resumable.
    assert resume_blockers(_ckpt(step=6000), amp_enabled=False, max_steps=32000) is None


def test_longer_donor_horizon_also_forces_weights_only():
    """A 75k donor into a 32k run would not crash -- it silently anneals on the
    donor's horizon. Above-band horizons are transfers too."""
    ckpt = _ckpt(step=20000)
    ckpt["scheduler_state_dict"]["total_steps"] = 75200
    assert resume_blockers(ckpt, amp_enabled=False, max_steps=32000) is not None


def _cmd_for(config_name, lr=3.0e-4):
    from aiida_pino.scripts.train_adapter import append_runtime_overrides
    from aiida_pino.utils.profiles import merge_training_profile

    profile = merge_training_profile({
        "config_name": config_name, "run_name": "r", "hydra_overrides": [],
        "cuda_device": "auto", "wandb_offline": True, "learning_rate": lr,
    })
    cmd: list[str] = []
    append_runtime_overrides(cmd, profile, split_mode="standard")
    return cmd


@pytest.mark.requires_lpno
def test_one_cycle_config_gets_max_lr_from_learning_rate():
    """`learning_rate` must reach the LR that actually runs.

    Found 2026-08-15: the profile's learning_rate is emitted as optimizer.lr,
    but OneCycleLR overwrites each param group's lr from scheduler.max_lr, which
    nothing synced -- so four campaign arms recorded 3e-4 and every one of them
    trained at the config's 1e-4. The autopsy's "LR was the bug" fix was a no-op
    for two weeks.
    """
    cmd = _cmd_for("config_flow3D_GAOT_pvlhs")
    tok = [c for c in cmd if "scheduler.max_lr" in c]
    assert tok == ["scheduler.max_lr=0.0003"], tok
    # MUST NOT start with + : pino's remap_plus_args_to_custom rewrites any
    # +/++ arg into the custom. namespace, which turned the 2026-08-15 fix into
    # a dead `custom.scheduler.max_lr` while every arm silently trained at the
    # config default (measured 2026-08-17 across the whole LR sweep).
    assert not tok[0].startswith("+"), tok
    # and optimizer.lr stays consistent rather than contradicting the peak
    assert any(c == "optimizer.lr=0.0003" for c in cmd), cmd


def test_decay_scheduler_config_gets_no_max_lr():
    """An exponential-decay config has no max_lr key; emitting one bare would
    crash compose in struct mode (the same trap as bare custom.<key>)."""
    cmd = _cmd_for("config_NS_energy_PINO_flow3D")
    assert not any("max_lr" in c for c in cmd), cmd


@pytest.mark.requires_lpno
def test_default_learning_rate_is_the_lr_that_binds():
    """The packaged default must report the LR the config will actually run.

    `load_training_profile_defaults` read `optimizer.lr`, but on a one-cycle
    config OneCycleLR runs `scheduler.max_lr` -- so the default advertised the
    pvlhs chain's optimizer.lr while the run used max_lr. Other tools read this
    default to decide what a run "is"; it has to round-trip with what the
    adapter emits.
    """
    from aiida_pino.utils.profiles import load_training_profile_defaults

    one_cycle = load_training_profile_defaults("config_flow3D_GAOT_pvlhs")
    assert one_cycle["learning_rate"] == 1e-4, one_cycle["learning_rate"]

    # A decay-scheduler config has no max_lr: optimizer.lr still governs.
    decay = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    assert decay["learning_rate"] == 1e-4, decay["learning_rate"]
