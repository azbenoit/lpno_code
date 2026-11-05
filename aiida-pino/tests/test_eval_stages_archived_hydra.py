"""An eval must stage the model's `.hydra`, or it is scored under TODAY's config.

2026-08-24. `eval_adapter.build_eval_command` prefers the model's ARCHIVED config over
the packaged one (decided 2026-08-14), so that an old checkpoint is
evaluated under the config it was trained with and stays evaluable as the packaged
schema moves.

`calculations/eval.py` deleted every `.hydra` directory while staging (2026-04-07,
b7b17ca, four months earlier). Nobody removed it, so `archived_config_dir` looked for a
directory that staging had just removed and the adapter silently fell back to the packaged
config with only a warning in the log. **The 08-14 fix was inert from the day it landed.**

Measured on one eval of a model trained on the same machine and scored cleanly
days before:

    Warning: Eval defaults not loaded from model .hydra config; no config.yaml found
             under '.../training_input', its parent, or its grandparent
    RuntimeError: Missing key(s) in state_dict: "proc_steps.8.edge_mlp.layers.0.weight"

The archived config records `arch.rigno.processor_steps: 8`; the eval skeleton was built
from a config whose default had drifted. A model that was evaluable yesterday was not
today, which is exactly what the adapter docstring says a provenance store must not do.
"""

from __future__ import annotations

from pathlib import Path


def test_the_staging_code_no_longer_deletes_dot_hydra():
    """Structural: the deletion is what made the archived-config feature inert.

    Asserted against the source because the alternative -- a full CalcJob sandbox with a
    stored PinoModelData -- tests the same one line at far greater cost. What matters is
    that nothing removes `.hydra` between the copy and the adapter reading it.
    """
    import aiida_pino

    source = (Path(aiida_pino.__file__).parent / "calculations" / "eval.py").read_text()
    assert "rmtree" not in source, (
        "eval staging must not delete anything under the copied model tree; deleting "
        "`.hydra` silently defeats archived-config evaluation (18b90a0)"
    )


def test_the_adapter_still_looks_for_it_where_staging_puts_it():
    """The two halves must agree on the path, which is the coupling that broke.

    `archived_config_dir` looks for `<training_dir>/.hydra/config.yaml`, and staging copies
    the model repository (which contains `.hydra/`) into `training_input`.
    """
    from aiida_pino.scripts.eval_adapter import archived_config_dir

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "training_input"
        (staged / ".hydra").mkdir(parents=True)
        assert archived_config_dir(staged) is None, "no config.yaml yet -> not usable"
        (staged / ".hydra" / "config.yaml").write_text("arch: {}\n", encoding="utf-8")
        assert archived_config_dir(staged) == staged / ".hydra"


def test_use_packaged_config_still_opts_out():
    """The escape hatch must survive: some evals deliberately want today's config."""
    from aiida_pino.scripts.eval_adapter import build_eval_command

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "training_input"
        (staged / ".hydra").mkdir(parents=True)
        (staged / ".hydra" / "config.yaml").write_text("arch: {}\n", encoding="utf-8")
        from aiida_pino.utils.profiles import merge_training_profile

        profile = merge_training_profile({
            "config_name": "config_flow3D_RIGNO_pvlhs",
            "run_name": "eval_probe",
            "hydra_overrides": [],
            "cuda_device": "auto",
            "wandb_offline": True,
        })
        profile["use_packaged_config"] = True
        command = build_eval_command(profile, Path(tmp) / "dataset", staged)
        assert any("--config-name=config_flow3D_RIGNO_pvlhs" in c for c in command)
        assert not any("--config-dir" in c for c in command)

        profile["use_packaged_config"] = False
        command = build_eval_command(profile, Path(tmp) / "dataset", staged)
        assert any("--config-dir" in c for c in command), "archived config must be preferred"
