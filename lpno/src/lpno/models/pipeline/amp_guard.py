"""Fail a training run whose AMP grad scaler has collapsed.

The damage of an fp16 gradient overflow is its silence: with fp16 autocast the
grad scaler found infs on *every* step, so physicsnemo skipped every optimizer
update, the scale fell 32768 -> 0.0625 and kept falling, and the run still
completed and registered a model that was never trained. Flat loss curves and
init weights are easy to mistake for "the architecture did not learn".

This guard makes that failure loud. It wraps the scaler physicsnemo builds inside
`Trainer._train_loop` (so there is no instance to patch after the fact -- we swap
the class the trainer looks up) and raises as soon as the run is provably not
training:

- `max_consecutive_skips` steps in a row were skipped (the only check on by
  default), or
- the scale fell below `scale_floor`, which is OFF by default -- see below.

Consecutive skips is the sound signal and is unambiguous: a backed-off scale
means that step applied no update, so N in a row means N steps of nothing. It
does not fire on healthy AMP warm-up, which backs off a handful of times at the
initial scale and then recovers -- measured on a small RIGNO: 2 skips of 40,
scale settling at 16384.

The scale floor was the original second check and it was a mistake, corrected
2026-08-21: a low scale is not evidence of a no-op. It killed `rigno2579_e8_oc`
at step 600 of 32000 while that run was training normally (loss 3.725e+05 ->
1.835e+05, 29 inf events in 600 steps). Whether a run is progressing is a
question about *steps applied*, not about the magnitude the scaler settled on,
and a loss of ~3.7e5 legitimately needs a tiny scale. Left as an opt-in knob
(`custom.amp_scale_floor`) rather than deleted, since a scale ratcheting to
denormal with no recovery is still a plausible failure -- just not one that has
been observed here.

Deliberately a hard error, unlike `_warn_if_amp_with_rigno_edge_masking`'s banner:
by the time the scale has collapsed the run has no value left to preserve. Set
`custom.amp_collapse_guard=false` to opt out for an experiment that wants it.
"""

from __future__ import annotations

import contextlib

from omegaconf import OmegaConf

# A healthy scaler backs off a few times at the initial scale, then recovers.
# 50 in a row is far outside that and well short of wasting a run.
DEFAULT_MAX_CONSECUTIVE_SKIPS = 50
# Off by default. This was 2**-8 and it was WRONG: a low scale is not evidence
# that a run is not training. `rigno2579_e8_oc` (2026-08-08) was killed at step
# 600 of 32000 while its loss was falling 3.725e+05 -> 1.835e+05, with just 29
# inf events in 600 steps -- 4.8% skipped, 95% of steps applying updates. Its
# scale oscillated in a 0.0039-0.0625 band and kept *growing back*, which under
# physicsnemo's recover_growth_interval=100 takes 100 consecutive good steps each
# time. On a loss of ~3.7e5 a tiny scale is what the scaler is supposed to find,
# not a symptom. Consecutive skips is the signal that actually means "no updates
# are being applied"; set custom.amp_scale_floor > 0 to opt the backstop back in.
DEFAULT_SCALE_FLOOR = 0.0


class AmpScalerCollapse(RuntimeError):
    """Raised when the AMP grad scaler shows the run is applying no updates."""


def _message(reason: str) -> str:
    return (
        f"AMP grad scaler collapsed: {reason}. The run is applying no optimizer "
        "updates, so it would finish with the weights it started from (see "
        "potential_bugs.md, BUG-43a). Set "
        "amp.enabled=false to train in fp32 "
        "(verified finite), or custom.amp_collapse_guard=false to allow it."
    )


def make_guarded_scaler_class(
    base_cls,
    max_consecutive_skips: int = DEFAULT_MAX_CONSECUTIVE_SKIPS,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
):
    """Build a subclass of `base_cls` that raises once the scaler has collapsed.

    Takes the base class as an argument rather than importing physicsnemo's so the
    behaviour is unit-testable against a stub on CPU, with no CUDA and no license.
    """

    class _GuardedGradScaler(base_cls):  # type: ignore[misc,valid-type]
        # State is created lazily instead of in __init__ so we never have to
        # mirror the base class's constructor signature.
        def update(self, new_scale=None):
            enabled = self.is_enabled()
            before = self.get_scale() if enabled else None
            super().update(new_scale)
            if not enabled:
                return
            after = self.get_scale()
            # A backed-off scale is exactly how a skipped step shows up; an
            # unchanged or grown scale means the step was applied.
            if after < before:
                skips = getattr(self, "_pino_consecutive_skips", 0) + 1
            else:
                skips = 0
            self._pino_consecutive_skips = skips
            if skips >= max_consecutive_skips:
                raise AmpScalerCollapse(
                    _message(f"{skips} consecutive steps skipped on non-finite gradients")
                )
            if scale_floor and after < scale_floor:
                raise AmpScalerCollapse(
                    _message(f"scale fell to {after:g}, below the {scale_floor:g} floor")
                )

    _GuardedGradScaler.__name__ = f"Guarded{getattr(base_cls, '__name__', 'GradScaler')}"
    return _GuardedGradScaler


@contextlib.contextmanager
def amp_collapse_guard(cfg):
    """Swap physicsnemo's `GradScaler` for a guarded subclass for the solve.

    `Trainer._train_loop` constructs the scaler itself, so there is no instance to
    wrap beforehand; patching the name the trainer module resolves is the smallest
    hook that covers every path into it. Restored on exit.
    """
    if not bool(OmegaConf.select(cfg, "custom.amp_collapse_guard", default=True)):
        yield
        return
    if not bool(OmegaConf.select(cfg, "amp.enabled", default=False)):
        # No scaler is created without fp16 AMP, so there is nothing to guard.
        yield
        return

    from physicsnemo.sym import trainer as _trainer

    original = _trainer.GradScaler
    _trainer.GradScaler = make_guarded_scaler_class(
        original,
        int(OmegaConf.select(cfg, "custom.amp_max_consecutive_skips",
                             default=DEFAULT_MAX_CONSECUTIVE_SKIPS)),
        float(OmegaConf.select(cfg, "custom.amp_scale_floor",
                               default=DEFAULT_SCALE_FLOOR)),
    )
    try:
        yield
    finally:
        _trainer.GradScaler = original
