"""An old checkpoint is evaluated under the config it was TRAINED with.

`e4_gaot_sp_test12467` failed on its model with "Current model
architecture/config likely differs from the training run": the GAOT arch gained
fields on 2026-08-05, the eval rebuilt from today's packaged config, and an
intact `u_network.0.pth` became unloadable. Every stored model degrades that way
as the schema moves -- which is what a provenance store must not do.

Chosen 2026-08-14: option (a), evaluate against the model's archived
config, with option (c)'s record of which config was used.
"""

from __future__ import annotations

from pathlib import Path

from aiida_pino.scripts.eval_adapter import archived_config_dir, build_eval_command

PROFILE = {
    "config_name": "config_flow3D_RIGNO", "run_name": "r", "wandb_offline": True,
    "k_folds": 1, "fold_index": 0, "pde_loss": False, "vof_pde_loss": False,
    "diff_loss": False, "diff_weight": 1.0, "pde_weight": 1.0, "include_time": False,
    "time_as_batch": False, "time_as_dim": False, "use_2d": False, "time_avg_window": 1,
    "moving_ref_frame": False, "moving_frame_steps": 1, "fast_training": False,
    "detect_anomaly": False, "sharp_alpha": False, "alpha_masking": False,
    "max_steps": 10, "learning_rate": 1e-3, "weight_decay": 0.0, "batch_size_grid": 1,
    "batch_size_validation": 1, "inference_batch_size": 1, "infer_velocities": False,
}


def _training_dir(tmp_path: Path, *, archived: bool) -> Path:
    root = tmp_path / "training_output"
    root.mkdir(parents=True, exist_ok=True)
    if archived:
        hydra = root / ".hydra"
        hydra.mkdir()
        (hydra / "config.yaml").write_text("arch:\n  rigno:\n    latent: 64\n")
    return root


def test_an_archived_config_is_preferred(tmp_path):
    training = _training_dir(tmp_path, archived=True)
    command = build_eval_command(dict(PROFILE), tmp_path / "data", training)
    assert f"--config-dir={training / '.hydra'}" in command
    assert "--config-name=config" in command
    assert "--config-name=config_flow3D_RIGNO" not in command, \
        "the packaged config must not also be named"


def test_a_model_without_one_keeps_todays_behaviour(tmp_path):
    """29 of 243 models predate the archiving; they must still be evaluable."""
    training = _training_dir(tmp_path, archived=False)
    command = build_eval_command(dict(PROFILE), tmp_path / "data", training)
    assert "--config-name=config_flow3D_RIGNO" in command
    assert not any(c.startswith("--config-dir=") for c in command)


def test_the_packaged_config_can_be_demanded(tmp_path):
    """Scoring an old model under today's config is sometimes the POINT --
    reproducing the drift itself. It must be possible, just not the default."""
    training = _training_dir(tmp_path, archived=True)
    command = build_eval_command({**PROFILE, "use_packaged_config": True},
                                 tmp_path / "data", training)
    assert "--config-name=config_flow3D_RIGNO" in command
    assert not any(c.startswith("--config-dir=") for c in command)


def test_archived_config_dir_requires_the_file_not_just_the_directory(tmp_path):
    root = tmp_path / "training_output"
    (root / ".hydra").mkdir(parents=True)
    assert archived_config_dir(root) is None, "an empty .hydra is not an archived config"
    (root / ".hydra" / "config.yaml").write_text("a: 1\n")
    assert archived_config_dir(root) == root / ".hydra"


def test_the_evaluation_profile_accepts_the_escape_hatch():
    from aiida_pino.utils.model_registry import validate_evaluation_profile

    assert validate_evaluation_profile({"use_packaged_config": True}) == {
        "use_packaged_config": True
    }
