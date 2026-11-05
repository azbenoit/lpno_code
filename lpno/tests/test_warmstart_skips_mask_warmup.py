"""A warm-started model must not re-run the mask warmup.

`WarmupNode(alpha_mask_module, start_step=500)` exists to avoid hard masking
"too early when training from scratch" (model_domain.py). A run that loads
pre-trained weights via physicsnemo's `initialization_network_dir` is not
training from scratch, and re-applying the warmup is actively harmful:

- the model's sharp outputs go through an unmasked head for 500 steps, so the
  loss starts at 3.185e+04 instead of ~2e+02;
- under fp16 AMP those gradients overflow immediately -- measured on run 7151
  (2026-07-22), the grad scaler halved on every step from 32768 down to 8 in
  four seconds and the run died;
- and the step-0 validation reads 0.9055 against 0.0293 for the same weights at
  step 32000, which looks exactly like a failed checkpoint load.

So the warmup start step is 0 whenever weights are initialized from a previous
run, the same way `eval_only` already zeroes it.
"""
from __future__ import annotations

import pytest

from lpno.models.pipeline.model_domain import mask_warmup_start_step


@pytest.mark.parametrize("init_dir", ["", None])
def test_cold_start_keeps_the_warmup(init_dir):
    assert mask_warmup_start_step(eval_only=False, initialization_network_dir=init_dir) == 500


def test_warm_start_skips_the_warmup():
    assert mask_warmup_start_step(
        eval_only=False, initialization_network_dir="/tmp/init_model"
    ) == 0


def test_eval_only_still_skips_the_warmup():
    assert mask_warmup_start_step(eval_only=True, initialization_network_dir="") == 0


def test_eval_only_and_warm_start_together():
    assert mask_warmup_start_step(
        eval_only=True, initialization_network_dir="/tmp/init_model"
    ) == 0


def test_whitespace_only_dir_counts_as_cold():
    """physicsnemo's default is the empty string; do not treat blanks as a warm start."""
    assert mask_warmup_start_step(eval_only=False, initialization_network_dir="   ") == 500


# --------------------------------------------------------------------------- #
# `custom.mask_warmup_steps`: the warmup must outlast the LR warmup.
#
# The fixed 500 assumes a schedule already at speed by then. One-cycle is not:
# it starts at max_lr/div_factor and peaks at pct_start*total_steps (step 3510
# for pct_start=0.05, total_steps=70200). Run rigno_rp_all3_2ep_s221
# (2026-07-27) hit the mask at step 500 with lr=1.44e-05, 7x below peak; alpha
# was not learned, hard masking collapsed the outputs to the constant fill,
# grad_norm went to 0 and 30k steps trained nothing. Hence a configurable start.
# --------------------------------------------------------------------------- #
def test_configured_value_overrides_the_default():
    assert mask_warmup_start_step(
        eval_only=False, initialization_network_dir="", configured=4000) == 4000


def test_none_keeps_the_shipped_default():
    """Not configuring it must not change any existing run."""
    assert mask_warmup_start_step(
        eval_only=False, initialization_network_dir="", configured=None) == 500


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_config_falls_back_to_the_default(blank):
    """An unset hydra key arrives as an empty string, not None."""
    assert mask_warmup_start_step(
        eval_only=False, initialization_network_dir="", configured=blank) == 500


def test_string_value_is_accepted():
    """Hydra/OmegaConf overrides arrive as strings."""
    assert mask_warmup_start_step(
        eval_only=False, initialization_network_dir="", configured="4000") == 4000


@pytest.mark.parametrize("bad", [-1, -500])
def test_negative_is_clamped_to_zero(bad):
    assert mask_warmup_start_step(
        eval_only=False, initialization_network_dir="", configured=bad) == 0


def test_warm_start_still_wins_over_a_configured_value():
    """Pre-trained weights are already sharp; re-running any warmup is the BUG above."""
    assert mask_warmup_start_step(
        eval_only=False, initialization_network_dir="/tmp/init_model", configured=4000) == 0


def test_eval_only_still_wins_over_a_configured_value():
    assert mask_warmup_start_step(
        eval_only=True, initialization_network_dir="", configured=4000) == 0
