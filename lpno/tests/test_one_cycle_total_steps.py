"""`scheduler.total_steps` must be derived from `training.max_steps`, never hand-set.

One-cycle's whole benefit is the deep cosine anneal at the *end* of the run. If
`total_steps` exceeds `max_steps` the cycle never finishes and the run stops at a
learning rate higher than a plain exponential schedule would have reached -- worse
than not using one-cycle. If it is smaller, OneCycleLR raises. Either way the bug
is silent at config time and only shows up as a bad number hours later, so the two
are tied together in code.
"""
from omegaconf import OmegaConf

from lpno.models.pipeline.config_loading import (
    ONE_CYCLE_STEP_MARGIN,
    _sync_one_cycle_total_steps,
)

ONE_CYCLE = "torch.optim.lr_scheduler.OneCycleLR"
EXPONENTIAL = "custom.schedulers.ExponentialLR"


def _cfg(target, max_steps, total_steps=None):
    sched = {"_target_": target}
    if total_steps is not None:
        sched["total_steps"] = total_steps
    return OmegaConf.create({"scheduler": sched, "training": {"max_steps": max_steps}})


def test_total_steps_derived_from_max_steps():
    cfg = _cfg(ONE_CYCLE, 20000, total_steps=32200)
    assert _sync_one_cycle_total_steps(cfg) == 20000 + ONE_CYCLE_STEP_MARGIN
    assert cfg.scheduler.total_steps == 20000 + ONE_CYCLE_STEP_MARGIN


def test_stale_total_steps_is_overwritten_not_trusted():
    # The exact failure this guards: a config written for a 32k run reused at 20k.
    # 32200 would leave the cosine ~60% down, at a LR above what the exponential
    # schedule reaches by then.
    cfg = _cfg(ONE_CYCLE, 20000, total_steps=32200)
    _sync_one_cycle_total_steps(cfg)
    assert cfg.scheduler.total_steps < 32200


def test_total_steps_always_exceeds_max_steps():
    # OneCycleLR raises "Tried to step N+1 times" if stepped past total_steps.
    for steps in (100, 8000, 20000, 32000):
        cfg = _cfg(ONE_CYCLE, steps)
        _sync_one_cycle_total_steps(cfg)
        assert cfg.scheduler.total_steps > steps


def test_no_total_steps_key_added_for_other_schedulers():
    cfg = _cfg(EXPONENTIAL, 20000)
    assert _sync_one_cycle_total_steps(cfg) is None
    assert "total_steps" not in cfg.scheduler


def test_missing_max_steps_is_left_alone():
    cfg = OmegaConf.create({"scheduler": {"_target_": ONE_CYCLE, "total_steps": 999}})
    assert _sync_one_cycle_total_steps(cfg) is None
    assert cfg.scheduler.total_steps == 999


def test_shipped_configs_do_not_hand_set_total_steps():
    """A hand-set value in a YAML would be silently overwritten -- do not ship one."""
    import yaml
    from pathlib import Path

    conf = Path(__file__).resolve().parents[1] / "src" / "lpno" / "conf" / "hydra"
    offenders = []
    for path in sorted(conf.glob("config_*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        if "total_steps" in (cfg.get("scheduler") or {}):
            offenders.append(path.name)
    assert not offenders, f"these configs hand-set scheduler.total_steps: {offenders}"
