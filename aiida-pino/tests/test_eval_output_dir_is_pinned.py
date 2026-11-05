"""A completed eval must not be failed for missing metrics with its files on disk.

2026-08-25. Evals 67015 and 67034 ran to completion -- 2887 batches each, full phase
timing, wandb sync, no traceback -- and were reported ERROR_MISSING_METRICS with
`metrics.json` written as `{}` and `eval_output/` empty. The artifacts were present the
whole time under `outputs/2026-08-25/07-02-27/fno_eval/`.

Cause, and it is a regression from `6992416` the same evening: that commit made
archived-config evaluation actually happen (it had been inert since 2026-08-14 because
staging deleted the `.hydra` it needed). A model's archived `config.yaml` carries no
`hydra:` block -- hydra writes its own settings to a separate `hydra.yaml` -- so the eval
reverted to hydra's DEFAULT `outputs/<date>/<time>/`, two levels deep, while
`discover_raw_eval_dir` globbed `outputs/*/fno_eval`, one level. Every eval took that path
after the 19:12:36 restart, not just the two that were noticed.

The failure is indistinguishable from a genuine one in every view (exit 502,
`metrics_file_present: False`), so the natural response is to resubmit and pay the full
eval cost to land in the same place.
"""

from __future__ import annotations

from pathlib import Path

from aiida_pino.scripts.eval_adapter import EVAL_RUN_DIR, discover_raw_eval_dir


def _make(root: Path, relative: str) -> Path:
    target = root / relative
    target.mkdir(parents=True)
    (target / "model_errors.txt").write_text("x", encoding="utf-8")
    return target


def test_the_date_based_default_is_found(tmp_path):
    """The reported layout: hydra's default run.dir is TWO levels under outputs/."""
    expected = _make(tmp_path, "outputs/2026-08-25/07-02-27/fno_eval")
    assert discover_raw_eval_dir(tmp_path) == expected


def test_the_run_named_layout_still_works(tmp_path):
    """What the packaged configs produce; must not regress."""
    expected = _make(tmp_path, "outputs/gaot_x_eval_0824_1733/fno_eval")
    assert discover_raw_eval_dir(tmp_path) == expected


def test_the_legacy_cwd_layout_still_wins(tmp_path):
    """`cwd/fno_eval` is checked first and must keep priority."""
    legacy = _make(tmp_path, "fno_eval")
    _make(tmp_path, "outputs/2026-08-25/07-02-27/fno_eval")
    assert discover_raw_eval_dir(tmp_path) == legacy


def test_the_newest_wins_when_several_exist(tmp_path):
    """A rerun in the same sandbox must collect the latest, not an earlier attempt."""
    import os
    import time

    old = _make(tmp_path, "outputs/2026-08-25/07-02-27/fno_eval")
    time.sleep(0.01)
    new = _make(tmp_path, "outputs/2026-08-25/09-30-00/fno_eval")
    os.utime(old, (1, 1))
    assert discover_raw_eval_dir(tmp_path) == new


def test_nothing_found_returns_the_legacy_path_rather_than_raising(tmp_path):
    """Discovery must not itself throw; the caller reports the missing metrics."""
    assert discover_raw_eval_dir(tmp_path) == tmp_path / "fno_eval"


def test_the_eval_command_pins_hydra_run_dir(tmp_path):
    """The root fix: state the output location instead of inferring it afterwards.

    Discovery alone would still be guessing where hydra chose to write. Pinning makes it
    identical whether the archived or the packaged config wins.
    """
    from aiida_pino.scripts.eval_adapter import build_eval_command
    from aiida_pino.utils.profiles import merge_training_profile

    staged = tmp_path / "training_input"
    (staged / ".hydra").mkdir(parents=True)
    (staged / ".hydra" / "config.yaml").write_text("arch: {}\n", encoding="utf-8")
    profile = merge_training_profile({
        "config_name": "config_flow3D_RIGNO_pvlhs",
        "run_name": "eval_probe",
        "hydra_overrides": [],
        "cuda_device": "auto",
        "wandb_offline": True,
    })

    for use_packaged in (False, True):
        profile["use_packaged_config"] = use_packaged
        command = build_eval_command(profile, tmp_path / "dataset", staged)
        pins = [c for c in command if c.startswith("hydra.run.dir=")]
        assert pins == [f"hydra.run.dir={EVAL_RUN_DIR}"], (
            f"the output location must be pinned regardless of which config wins "
            f"(use_packaged_config={use_packaged})"
        )
        assert not any(c.startswith("+") and "hydra.run.dir" in c for c in command), (
            "a + prefix would be rewritten into the custom. namespace and set nothing"
        )


def test_a_model_without_a_grid_stamp_says_the_trained_grid_is_unverifiable(tmp_path, capsys):
    """Silence here reads as "the grids match"; it means "nobody looked".

    `lpno` stamps `trained_grid.json` during training and compares trained vs
    served at eval. A model that carries no stamp skips that comparison entirely -- and the
    archived config cannot substitute, because an NPZ-trained model's `custom.geometry`
    records what was ASKED for, not what ran (one model archives [90, 40, 30] and trained
    on 40 layers). One eval scored a 30-layer model on 40 and produced numbers.
    """
    from aiida_pino.scripts.eval_adapter import warn_if_trained_grid_is_unverifiable

    staged = tmp_path / "training_input"
    staged.mkdir()
    assert warn_if_trained_grid_is_unverifiable(staged) is not None
    assert "cannot be verified" in capsys.readouterr().out


def test_a_model_with_a_grid_stamp_is_quiet(tmp_path, capsys):
    """The stamped case is already handled downstream; do not add noise to it."""
    from aiida_pino.scripts.eval_adapter import GRID_STAMP_NAME, warn_if_trained_grid_is_unverifiable

    staged = tmp_path / "training_input"
    staged.mkdir()
    (staged / GRID_STAMP_NAME).write_text("{}", encoding="utf-8")
    assert warn_if_trained_grid_is_unverifiable(staged) is None
    assert capsys.readouterr().out == ""


def test_the_stamp_is_found_beside_the_checkpoint_too(tmp_path, capsys):
    """`training_dir` is moved to the checkpoint's parent, so search upward like the config does."""
    from aiida_pino.scripts.eval_adapter import GRID_STAMP_NAME, warn_if_trained_grid_is_unverifiable

    root = tmp_path / "training_input"
    deep = root / "run" / "outputs"
    deep.mkdir(parents=True)
    (root / GRID_STAMP_NAME).write_text("{}", encoding="utf-8")
    assert warn_if_trained_grid_is_unverifiable(deep) is None
