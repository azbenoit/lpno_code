"""The 100-step VRAM smoke must be able to start, and say why when it cannot.

2026-08-22, after ~1 h of an exclusive card was lost to this. The
smoke overrides `max_steps` to 100 but inherited the config's recording cadence,
so `validate_recording_cadence` refused it before it ever reached the GPU:

    ValueError: training.rec_results_freq=8000 > max_steps=100: the run would
    never write a checkpoint, so a crash at any point loses everything.

Any config with a cadence coarser than 100 steps could therefore never be
VRAM-smoked -- which is exactly the set of long-running configs the auto-sizing
path exists for. And the smoke holds an EXCLUSIVE card while it waits, so a
config that can never smoke burns the scarcest resource on the box to produce a
failure that was knowable at submit time.

`validate_recording_cadence` checks three things against `max_steps`, so a
clamp has to satisfy all three or it just moves the error:
  * rec_results_freq <= max_steps
  * rec_validation_freq <= max_steps
  * stop_criterion.freq a multiple of rec_validation_freq
"""

from __future__ import annotations

import pytest
from aiida_pino.workflows.train_model import SMOKE_MAX_STEPS, smoke_training_profile


def _overrides(profile):
    return smoke_training_profile(profile)["hydra_overrides"]


def _clamp_for(config_name, max_steps=SMOKE_MAX_STEPS, overrides=None):
    """The cadence tokens the ADAPTER emits -- what hydra actually receives."""
    from aiida_pino.scripts.train_adapter import append_recording_cadence_clamp

    command: list[str] = []
    append_recording_cadence_clamp(
        command,
        {"config_name": config_name, "max_steps": max_steps, "hydra_overrides": overrides or []},
    )
    return command


# The cadence clamp moved out of `smoke_training_profile` on 2026-08-22. These four
# tests kept their INTENT and changed their mechanism, because the mechanism they
# asserted was proven dead in production: the profile-level `++` clamp was
# rewritten into the `custom.` namespace by pino's `remap_plus_args_to_custom` and never
# reached the run. Asserting on the profile's override list is what let that through --
# it is the function's output, not what hydra receives -- so these assert on the built
# command instead.
CADENCE_CONFIG = "config_flow3D_RIGNO_pvlhs"  # declares all three keys at 2000


@pytest.mark.requires_lpno
def test_the_recording_cadence_is_clamped_to_the_smoke_length():
    joined = " ".join(_clamp_for(CADENCE_CONFIG))
    for key in ("rec_results_freq", "rec_validation_freq"):
        assert f"training.{key}={SMOKE_MAX_STEPS}" in joined, (
            f"{key} must be clamped or the smoke cannot start at all"
        )
    assert smoke_training_profile({"run_name": "r", "hydra_overrides": []})["max_steps"] == SMOKE_MAX_STEPS


@pytest.mark.requires_lpno
def test_the_stop_criterion_stays_a_multiple_of_the_validation_cadence():
    """The third check in the guard; clamping only two of three moves the error."""
    assert f"stop_criterion.freq={SMOKE_MAX_STEPS}" in " ".join(_clamp_for(CADENCE_CONFIG))


def test_the_clamps_are_plain_overrides_because_plus_args_are_remapped_away():
    """Was `must start with ++`. That is the bug, not the fix.

    `remap_plus_args_to_custom` sends any `+`/`++` key that is not already under
    `custom.` into `custom.`, so `++training.rec_results_freq=100` sets
    `custom.training.rec_results_freq` -- read by nothing. A plain override is
    untouched by the remapper.
    """
    for item in _clamp_for(CADENCE_CONFIG):
        assert not item.startswith("+"), f"{item} would be remapped into custom. and die there"
        assert not item.startswith("custom."), f"{item} is not the key the validator reads"


def test_a_key_the_config_does_not_declare_is_never_emitted():
    """The reason `++` was chosen originally, and the reason plain must be conditional.

    A plain override of an absent key dies in hydra's struct mode, which is a worse
    failure than the refusal being fixed. `validate_recording_cadence` skips absent keys
    too, so an undeclared key needs no clamp. `config_NS_energy_PINO_flow3D` declares
    `training.rec_results_freq` but no `stop_criterion`.
    """
    joined = " ".join(_clamp_for("config_NS_energy_PINO_flow3D"))
    assert "stop_criterion" not in joined
    assert " ".join(_clamp_for("no_such_config_exists")) == ""


def test_a_cadence_already_fine_for_the_run_is_left_alone():
    """Only a cadence COARSER than the run is clamped; a real 32k run is untouched."""
    assert _clamp_for(CADENCE_CONFIG, max_steps=32000) == []


@pytest.mark.requires_lpno
def test_the_clamp_beats_an_inherited_plain_override():
    """A donor profile can carry a plain `training.rec_results_freq=2000`.

    The adapter appends the clamp AFTER the profile's hydra_overrides, so hydra's
    last-wins rule leaves the clamp in force. Asserted at the command level, since
    ordering is the whole point.
    """
    from aiida_pino.scripts.train_adapter import append_recording_cadence_clamp

    command = ["training.rec_results_freq=2000"]
    append_recording_cadence_clamp(
        command, {"config_name": CADENCE_CONFIG, "max_steps": SMOKE_MAX_STEPS}
    )
    surviving = [i for i in command if i.startswith("training.rec_results_freq=")][-1]
    assert int(surviving.split("=")[1]) <= SMOKE_MAX_STEPS


def test_no_plus_override_in_src_escapes_the_custom_namespace():
    """Repo-wide guard for the class of bug, not just this instance.

    Any emitted `++key=` whose key is not already `custom.`-prefixed is silently
    rewritten into `custom.` and read by nothing. That killed `++scheduler.max_lr`
    (2026-08-15) and the smoke cadence clamp (2026-08-22). Every current emission in
    `src/` is correctly namespaced; this keeps it that way.
    """
    import re
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent / "src" / "aiida_pino"
    offenders = []
    for path in root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # BOTH `+key=` and `++key=`: the remapper rewrites either form, so a
            # single-plus emission is remapped just as silently. None exist today;
            # matching only `++` would have let the first one through.
            for match in re.finditer(r'["\']\+{1,2}([A-Za-z_][\w.]*)=', line):
                if not match.group(1).startswith("custom."):
                    offenders.append(f"{path.relative_to(root)}:{lineno}: {match.group(0)}")
    assert not offenders, (
        "these ++ overrides are not custom.-namespaced, so remap_plus_args_to_custom "
        "will rewrite them into a namespace nothing reads:\n  " + "\n  ".join(offenders)
    )


def test_the_smoke_never_inherits_seat_priority_or_a_booking():
    out = smoke_training_profile({
        "run_name": "r",
        "hydra_overrides": [],
        "gpu_seat_priority": True,
        "gpu_vram_mb": 20000,
    })
    assert "gpu_seat_priority" not in out
    assert out["gpu_vram_mb"] is None
    assert out["run_name"].endswith("_vramsmoke")


# --- the failure must name the real cause, not a guess ---------------------
# The old message said "usually means the configuration does not fit this GPU at
# all". Measured on the real failure (smoke 64216): `exit_status=500`,
# `exit_message="Training did not produce train_summary.json"` -- truthful and
# useless, while the ACTUAL cause sat in the retrieved `_scheduler-stderr.txt`:
#
#     ValueError: training.rec_results_freq=8000 > max_steps=100: ...
#
# So reporting the exit message alone only downgrades a wrong answer to a blank
# one. The error line itself is what a reader needs.


class _Repo:
    def __init__(self, files):
        self._files = files

    def list_object_names(self):
        return list(self._files)

    def get_object_content(self, name):
        content = self._files[name]
        if content is None:
            raise IsADirectoryError(f"object with path `{name}` is not a file")
        return content


class _Retrieved:
    def __init__(self, files):
        self._repo = _Repo(files)

    @property
    def base(self):
        outer = self

        class _B:
            repository = outer._repo

        return _B()


def test_the_root_cause_is_reported_not_the_subprocess_wrapper():
    """The real shape of smoke 64216: cause first, wrapper last.

    A chained traceback prints the original cause first. The outer frame is
    always a `subprocess.CalledProcessError` naming the gpu_spool seat command,
    which says what ran and not why it failed -- taking the LAST error line
    reported exactly that and buried the reason.
    """
    from aiida_pino.workflows.train_model import smoke_failure_detail

    text = (
        "some setup noise\n"
        "Traceback (most recent call last):\n"
        "ValueError: training.rec_results_freq=8000 > max_steps=100: the run would\n"
        "Traceback (most recent call last):\n"
        "subprocess.CalledProcessError: Command '['python', '-m', 'gpu_spool', 'seat']'\n"
    )
    detail = smoke_failure_detail(_Retrieved({"_scheduler-stderr.txt": text}))
    assert "rec_results_freq=8000" in detail
    assert "CalledProcessError" not in detail


def test_a_directory_member_does_not_break_extraction():
    """`training_output` is a directory; reading it raises IsADirectoryError."""
    from aiida_pino.workflows.train_model import smoke_failure_detail

    detail = smoke_failure_detail(
        _Retrieved({"training_output": None, "aiida.out": "RuntimeError: CUDA out of memory\n"})
    )
    assert "CUDA out of memory" in detail


def test_no_error_line_yields_nothing_rather_than_noise():
    from aiida_pino.workflows.train_model import smoke_failure_detail

    assert smoke_failure_detail(_Retrieved({"aiida.out": "all fine\n"})) == ""
    assert smoke_failure_detail(None) == ""
