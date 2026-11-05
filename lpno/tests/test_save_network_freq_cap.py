"""A run shorter than its weights-saving interval must still save its final weights.

physicsnemo writes `u_network.0.pth` only when `step % training.save_network_freq == 0`
(default 1000). The 200-step runs of reproducibility/train_evaluate.ipynb therefore
kept their step-0 weights, and their evaluations scored untrained models with no
warning. `cap_save_network_freq` (run from `load_cfg_and_paths`) caps the interval to
`max_steps`, as `reproducibility/rerun/replay_overrides.py` does for a shortened replay.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from lpno.models.pipeline.config_loading import cap_save_network_freq


def _cfg(max_steps=200, save=1000):
    training = {"max_steps": max_steps}
    if save is not None:
        training["save_network_freq"] = save
    return OmegaConf.create({"training": training})


def _saved_steps(cfg):
    """The steps physicsnemo's loop (range(0, max_steps + 1)) saves weights on."""
    freq, last = int(cfg.training.save_network_freq), int(cfg.training.max_steps)
    return [step for step in range(last + 1) if step % freq == 0]


def test_short_run_saves_its_final_step():
    cfg = _cfg(max_steps=200, save=1000)
    assert _saved_steps(cfg) == [0]                      # the trap: step 0 only
    assert cap_save_network_freq(cfg) == 200
    assert cfg.training.save_network_freq == 200
    assert _saved_steps(cfg)[-1] == 200


def test_interval_within_the_run_is_kept(capsys):
    cfg = _cfg(max_steps=200, save=50)
    assert cap_save_network_freq(cfg) == 50
    assert cfg.training.save_network_freq == 50
    assert "WARNING" not in capsys.readouterr().out


def test_interval_that_does_not_divide_the_run_is_announced(capsys):
    cfg = _cfg(max_steps=250, save=100)
    assert cap_save_network_freq(cfg) == 100
    assert cfg.training.save_network_freq == 100
    out = capsys.readouterr().out
    assert "does not divide" in out and "step 200" in out


def test_eval_is_untouched():
    cfg = _cfg(max_steps=200, save=1000)
    assert cap_save_network_freq(cfg, run_mode="eval") is None
    assert cfg.training.save_network_freq == 1000


def test_absent_keys_are_tolerated():
    assert cap_save_network_freq(_cfg(save=None)) is None
    assert cap_save_network_freq(OmegaConf.create({"training": {}})) is None
