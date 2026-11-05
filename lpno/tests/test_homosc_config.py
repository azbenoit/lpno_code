"""The homoscedastic-aggregator configs must compose and carry the right target.

The trap this guards: config M has an inline `loss:` block (relobralo
alpha/beta/tau). A bare `override /loss: homoscedastic` merges that block onto
the homoscedastic schema and dies in struct validation — the scheduler version
of this failure is documented on config_flow3D_RIGNO_onecycle. The dedicated
configs re-declare the loss node; this test is the cheap proof they compose,
so the failure mode is a red test instead of a dead queued arm.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONF_DIR = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"
HOMOSC = "physicsnemo.sym.loss.aggregator.HomoscedasticUncertainty"


def _register_all():
    import importlib

    for mod in ("loss", "optimizer", "scheduler", "arch", "training",
                "profiler", "metric", "callbacks", "graph", "amp", "hydra"):
        m = importlib.import_module(f"physicsnemo.sym.hydra.{mod}")
        getattr(m, f"register_{mod}_configs")()
    # the top-level schema has its own naming (physicsnemo_default lives here)
    importlib.import_module("physicsnemo.sym.hydra.config").register_physicsnemo_configs()
    # lion / one_cycle are app-registered groups, same call the trainer makes.
    from lpno.models import register_custom_hydra_inputs

    register_custom_hydra_inputs()


@pytest.fixture(scope="module")
def composer():
    _register_all()
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.1"):
        yield compose


@pytest.mark.parametrize("name", ["config_flow3D_RIGNO_homosc", "config_flow3D_GAOT_homosc"])
def test_composes_with_the_homoscedastic_target(composer, name):
    cfg = composer(config_name=name)
    assert cfg.loss._target_ == HOMOSC


@pytest.mark.parametrize("name", ["config_flow3D_RIGNO_homosc", "config_flow3D_GAOT_homosc"])
def test_velocity_overrides_apply_on_top(composer, name):
    """The vel arms are these configs + overrides; they must not need a third file."""
    cfg = composer(config_name=name, overrides=[
        "custom.infer_velocities=true",
        "custom.vel_warmup_steps=4000",
        "custom.vel_ramp_steps=1000",
    ])
    assert cfg.custom.infer_velocities is True
    assert cfg.custom.vel_warmup_steps == 4000
    assert cfg.loss._target_ == HOMOSC


def test_gaot_still_inherits_the_gaot_arch(composer):
    cfg = composer(config_name="config_flow3D_GAOT_homosc")
    assert cfg.custom.model_family == "gaot"


@pytest.mark.parametrize("name", ["config_flow3D_RIGNO", "config_flow3D_GAOT"])
def test_config_m_family_loss_is_byte_identical_after_the_group_move(composer, name):
    """The relobralo params moved from an inline block to loss/relobralo_tuned.

    Every existing recipe composes through config M; its resolved loss node
    must be exactly what the inline block produced or every baseline number
    silently changes meaning.
    """
    cfg = composer(config_name=name)
    assert cfg.loss._target_.endswith("Relobralo")
    assert cfg.loss.alpha == 0.95
    assert cfg.loss.beta == 0.99
    assert cfg.loss.tau == 3.0
