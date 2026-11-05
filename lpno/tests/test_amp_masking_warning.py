"""Guard: loud warning when fp16 autocast is enabled on a RIGNO-family training."""
from omegaconf import OmegaConf

from lpno.models.pipeline.config_loading import (
    _warn_if_amp_with_rigno_edge_masking,
)


def _cfg(amp: bool, p_mask, family: str = "rigno"):
    node = {"amp": {"enabled": amp}, "arch": {"rigno": {}}, "custom": {"model_family": family}}
    if p_mask is not None:
        node["arch"]["rigno"]["p_edge_masking"] = p_mask
    return OmegaConf.create(node)


def test_warns_on_amp_plus_masking(capsys):
    assert _warn_if_amp_with_rigno_edge_masking(_cfg(True, 0.10), "rigno", "train")
    out = capsys.readouterr().out
    assert "fp16 autocast" in out and "amp.enabled=false" in out


def test_warns_when_masking_unset_because_arch_default_is_half(capsys):
    # model_domain defaults p_edge_masking to 0.5 when the config omits it, so
    # an omitted value under amp still deserves the warning.
    assert _warn_if_amp_with_rigno_edge_masking(_cfg(True, None), "rigno", "train")


def test_warns_on_amp_even_without_masking(capsys):
    """fp16 gradient overflow on this data family occurs without edge masking too.

    A masking-only guard would stay silent on such a config, so the warning fires
    for any rigno-family training with amp on.
    """
    assert _warn_if_amp_with_rigno_edge_masking(_cfg(True, 0.0), "rigno", "train")
    out = capsys.readouterr().out
    assert "amp.enabled=false" in out


def test_silent_when_safe():
    assert not _warn_if_amp_with_rigno_edge_masking(_cfg(False, 0.10), "rigno", "train")
    assert not _warn_if_amp_with_rigno_edge_masking(_cfg(False, 0.0), "rigno", "train")
    # non-rigno families and non-training modes stay silent
    assert not _warn_if_amp_with_rigno_edge_masking(_cfg(True, 0.0), "fno", "train")
    assert not _warn_if_amp_with_rigno_edge_masking(_cfg(True, 0.0), "rigno", "eval")
    assert not _warn_if_amp_with_rigno_edge_masking(_cfg(True, 0.10), "fno", "train")
    assert not _warn_if_amp_with_rigno_edge_masking(_cfg(True, 0.10), "rigno", "eval")


def test_warns_when_one_cycle_can_be_truncated_by_early_stopping(capsys):
    """Patience-based early stopping can truncate a one-cycle schedule.

    One-cycle must finish inside the run or its anneal never completes, so the
    weights are left mid-schedule at an elevated LR. Exponential decay degrades
    gracefully and needs no warning.
    """
    from lpno.models.pipeline.config_loading import (
        _warn_if_one_cycle_can_be_truncated,
    )

    # the REAL composed config carries _target_, not _name_: a guard keyed only
    # on _name_ never fires in production, which is how this shipped once
    cfg = OmegaConf.create({
        "scheduler": {"_target_": "torch.optim.lr_scheduler.OneCycleLR"},
        "training": {"max_steps": 32000},
        "stop_criterion": {"patience": 4000},
    })
    assert _warn_if_one_cycle_can_be_truncated(cfg, "train")
    out = capsys.readouterr().out
    assert "one-cycle" in out.lower() and "patience" in out.lower()

    # patience beyond the run cannot truncate it
    cfg.stop_criterion.patience = 40000
    assert not _warn_if_one_cycle_can_be_truncated(cfg, "train")
    # no early stopping configured at all
    del cfg.stop_criterion
    assert not _warn_if_one_cycle_can_be_truncated(cfg, "train")
    # exponential decay degrades gracefully
    cfg2 = OmegaConf.create({
        "scheduler": {"_target_": "physicsnemo.sym.ExponentialLR"},
        "training": {"max_steps": 32000},
        "stop_criterion": {"patience": 4000},
    })
    assert not _warn_if_one_cycle_can_be_truncated(cfg2, "train")
