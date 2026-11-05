"""The three pvlhs arms must differ ONLY in architecture.

The grid these configs exist for -- FNO vs RIGNO vs GAOT on the straight-path family --
is only interpretable if every other knob is held equal. Two of the knobs that matter
are not obvious and are not inherited: `stop_criterion.freq` must divide evenly with
`rec_validation_freq` (physicsnemo checks it only at the FIRST criterion evaluation, so
the inherited 500 against 2000 killed a wave-1 arm about two hours in), and
`stop_criterion.patience` must not sit below `max_steps` (an inherited 4000 fired
mid-descent and produced an untrained model).

Both partners override them. config_flow3D_FNO_pvlhs was added on 2026-08-20 and did
not, so both FNO arms died at config load. Cheap to fix, cheap to miss again on the
fourth architecture -- so pin the agreement rather than the individual values.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

ARMS = ("config_flow3D_FNO_pvlhs", "config_flow3D_RIGNO_pvlhs", "config_flow3D_GAOT_pvlhs")

# Everything that must be identical across the three. Architecture, model_family and
# amp are deliberately NOT here -- those are what the grid varies.
SHARED = (
    "training.max_steps",
    "training.rec_validation_freq",
    "training.rec_results_freq",
    "scheduler._target_",
    "scheduler.max_lr",
    "scheduler.pct_start",
    "scheduler.div_factor",
    "stop_criterion.freq",
    "stop_criterion.patience",
    "custom.moving_ref_frame",
    "custom.power_weighted_dose",
    "custom.time_as_batch",
)


@pytest.fixture(scope="module")
def composed():
    import physicsnemo.sym.hydra.utils as U
    from lpno.models import register_custom_hydra_inputs
    for name in dir(U):
        if name.startswith("register_") and name.endswith("_configs"):
            try:
                getattr(U, name)()
            except Exception:
                pass
    register_custom_hydra_inputs()

    import pathlib
    from hydra import compose, initialize_config_dir

    conf_dir = str(
        (pathlib.Path(__file__).resolve().parents[1]
         / "src/lpno/conf/hydra").resolve()
    )
    out = {}
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        for arm in ARMS:
            out[arm] = compose(config_name=arm)
    return out


@pytest.mark.parametrize("key", SHARED)
def test_the_three_arms_agree(composed, key):
    values = {arm: OmegaConf.select(cfg, key) for arm, cfg in composed.items()}
    assert len(set(map(repr, values.values()))) == 1, f"{key} differs: {values}"


def test_stop_criterion_divides_the_validation_cadence(composed):
    """The specific rule physicsnemo enforces late and config_loading enforces early."""
    for arm, cfg in composed.items():
        freq = OmegaConf.select(cfg, "stop_criterion.freq")
        val = OmegaConf.select(cfg, "training.rec_validation_freq")
        assert freq % val == 0 or val % freq == 0, f"{arm}: {freq} vs {val}"


def test_patience_does_not_cut_the_anneal_short(composed):
    """One-cycle needs its full anneal; patience below max_steps produces a stub."""
    for arm, cfg in composed.items():
        assert (OmegaConf.select(cfg, "stop_criterion.patience")
                >= OmegaConf.select(cfg, "training.max_steps")), arm


def test_each_arm_is_a_different_architecture(composed):
    """The one thing that must NOT be shared."""
    families = {OmegaConf.select(cfg, "custom.model_family") for cfg in composed.values()}
    assert families == {"fno", "rigno", "gaot"}, families


def test_steady_film_keys_are_a_subset_of_the_steady_inputs(composed):
    """The six-key steady arms must not condition FiLM on channels they no longer receive.

    RIGNOArch asserts film_keys is a subset of the input keys; the parent's six FiLM keys
    include three (preheat_T0, focus_factor, beam_shape_uniform) that the steady input trim
    removed, and 57757 died on exactly that assert at model build.
    """
    import pathlib

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    conf_dir = str(
        (pathlib.Path(__file__).resolve().parents[1]
         / "src/lpno/conf/hydra").resolve()
    )
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        for name in ("config_flow3D_RIGNO_pvlhs_steady",
                     "config_flow3D_GAOT_pvlhs_steady",
                     "config_flow3D_FNO_pvlhs_steady"):
            cfg = compose(config_name=name)
            inputs = set(OmegaConf.select(cfg, "custom.FNO_input_keys") or [])
            film = list(OmegaConf.select(cfg, "arch.rigno.film_keys") or [])
            missing = [k for k in film if k not in inputs]
            assert not missing, f"{name}: film_keys not in inputs: {missing}"
