"""Regression tests: WarmupNode must be CUDA-graph-capture-safe.

The old forward branched on ``self.step.item()`` (a CPU sync) and returned ``{}``
before warmup; graph capture forbids both the sync and the dynamic structure.
The capture-safe version always runs the inner node and blends with the
pre-inner values via ``torch.where`` on the step buffer, which is equivalent
because the shipped inners (MaskNode) only overwrite existing keys.
"""

import inspect

import pytest
import torch

from lpno.equations import FNO_postprocess as _fp
from lpno.equations.FNO_postprocess import WarmupNode, set_train_step


@pytest.fixture(autouse=True)
def _reset_published_train_step():
    """`_TRAIN_STEP` is module-level, so leaking it across tests would change
    which branch (published step vs call count) every other test exercises."""
    _fp._TRAIN_STEP["value"] = None
    yield
    _fp._TRAIN_STEP["value"] = None


class _Doubler(torch.nn.Module):
    """Mutates the outvar dict in place like MaskNode does."""

    def forward(self, outvar):
        outvar["T"] = outvar["T"] * 2
        return outvar


def test_warmup_passthrough_then_inner():
    node = WarmupNode(_Doubler(), start_step=3)
    x = torch.full((2, 3), 5.0)

    # Steps 1 and 2 (< start_step): values pass through unchanged, and the keys
    # are present so the blended output is usable downstream under capture.
    for _ in range(2):
        out = node({"T": x.clone()})
        assert torch.equal(out["T"], x)

    # Step 3 (>= start_step): inner node takes effect.
    out = node({"T": x.clone()})
    assert torch.equal(out["T"], x * 2)


def test_warmup_start_step_zero_is_always_inner():
    # eval_only wiring uses start_step=0: the inner must apply from step 1.
    node = WarmupNode(_Doubler(), start_step=0)
    x = torch.ones(2, 2)
    out = node({"T": x.clone()})
    assert torch.equal(out["T"], x * 2)


def test_warmup_forward_has_no_cpu_sync():
    # .item() on the step buffer is the exact call that kills graph capture.
    source = inspect.getsource(WarmupNode.forward)
    assert ".item(" not in source


# --------------------------------------------------------------------------- #
# The counter must measure TRAINING steps.
#
# In-training validation calls this node too, so counting every forward made the
# warmup expire far earlier than configured: run rigno_rp_all3_35k_v3
# (2026-07-28) was set to `mask_warmup_steps=4000` and the mask engaged before
# training step 2000 -- val mae_T jumped 1076.4 K -> 2093.1 K, the exact
# constant-fill value, and l2_relative_error_fl hit 1.0009 (predicting zeros).
# The whole point of the knob is "mask only once the model has begun to learn",
# which is unachievable while the unit is forward calls.
# --------------------------------------------------------------------------- #
def test_eval_forwards_do_not_advance_the_warmup():
    node = WarmupNode(_Doubler(), start_step=3)
    x = torch.full((2, 3), 5.0)

    node.eval()
    for _ in range(10):  # a validation pass: many batches, zero training steps
        node({"T": x.clone()})

    node.train()
    for _ in range(2):  # training steps 1 and 2 -> still pre-warmup
        out = node({"T": x.clone()})
        assert torch.equal(out["T"], x), "validation forwards advanced the warmup"

    out = node({"T": x.clone()})  # training step 3 trips it
    assert torch.equal(out["T"], x * 2)


def test_gate_follows_the_published_train_step_not_call_count():
    """The node runs several times per optimizer step, so counting calls lies.

    With `mask_warmup_steps=4000`, run rigno_rp_all3_35k_v4 (2026-07-28) engaged
    the mask around training step 2000 -- val mae_T went 1217.5 K -> 2093.1 K,
    the constant-fill value -- because every call incremented the counter. When
    the trainer publishes its step, forwards within one step must not advance it.
    """
    node = WarmupNode(_Doubler(), start_step=100)
    x = torch.ones(2, 2)
    node.train()

    set_train_step(50)
    for _ in range(20):  # many forwards inside a single optimizer step
        out = node({"T": x.clone()})
        assert torch.equal(out["T"], x), "call count advanced the warmup past the step"

    set_train_step(100)
    out = node({"T": x.clone()})
    assert torch.equal(out["T"], x * 2)


def test_call_counting_still_applies_when_nobody_publishes():
    """Standalone use (no sym trainer) has no step source; keep the old behaviour."""
    node = WarmupNode(_Doubler(), start_step=2)
    x = torch.ones(2, 2)
    node.train()
    assert torch.equal(node({"T": x.clone()})["T"], x)
    assert torch.equal(node({"T": x.clone()})["T"], x * 2)


def test_eval_still_sees_the_mask_once_warmed():
    """Freezing the counter in eval must not freeze the *gate*."""
    node = WarmupNode(_Doubler(), start_step=1)
    x = torch.ones(2, 2)
    node.train()
    node({"T": x.clone()})  # training step 1 trips the warmup
    node.eval()
    out = node({"T": x.clone()})
    assert torch.equal(out["T"], x * 2)
