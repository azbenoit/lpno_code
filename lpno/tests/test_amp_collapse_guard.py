"""A collapsed AMP grad scaler must fail the run, not finish it silently.

CPU-only and license-free. BUG-43's cost was that a run whose every step was
skipped on non-finite gradients still completed and registered a model holding
its init weights. The guard turns that into an error. Tested against a stub
scaler so the logic is checked without CUDA, fp16, or a real training loop.
"""
import pytest

from lpno.models.pipeline.amp_guard import (
    DEFAULT_SCALE_FLOOR,
    AmpScalerCollapse,
    DEFAULT_MAX_CONSECUTIVE_SKIPS,
    make_guarded_scaler_class,
)


class StubScaler:
    """Minimal stand-in for physicsnemo's GradScaler.

    `update()` moves the scale the way a real scaler does: halve it when the last
    step was skipped on non-finite gradients, otherwise leave it (or grow it).
    """

    def __init__(self, scale=65536.0, enabled=True):
        self._scale = float(scale)
        self._enabled = bool(enabled)
        self.skip_next = False
        self.grow_next = False

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._scale

    def update(self, new_scale=None):
        if not self._enabled:
            return
        if self.skip_next:
            self._scale /= 2.0
        elif self.grow_next:
            self._scale *= 2.0


def _guarded(**kw):
    cls = make_guarded_scaler_class(StubScaler, **kw)
    return cls()


def test_healthy_warmup_does_not_raise():
    # Real measured pattern on a small RIGNO: a couple of backoffs at the initial
    # scale, then steps land. That must not be mistaken for a collapse.
    s = _guarded()
    for skip in [True, True] + [False] * 200:
        s.skip_next = skip
        s.update()
    assert s.get_scale() == 65536.0 / 4


def test_every_step_skipped_raises():
    s = _guarded(max_consecutive_skips=50, scale_floor=0.0)
    s.skip_next = True
    with pytest.raises(AmpScalerCollapse) as err:
        for _ in range(200):
            s.update()
    assert "consecutive steps skipped" in str(err.value)
    # actionable: names the flag that makes the run finite
    assert "amp.enabled=false" in str(err.value)


def test_skip_counter_resets_on_a_successful_step():
    # A run that skips a lot but still makes progress is not collapsed; the
    # counter has to measure *consecutive* skips, not a total.
    s = _guarded(max_consecutive_skips=10, scale_floor=0.0)
    for _ in range(30):
        for _ in range(9):
            s.skip_next = True
            s.update()
        s.skip_next = False
        s.update()  # one good step clears it


def test_scale_floor_catches_a_slow_collapse():
    # Skips just under the consecutive limit each time would evade that check,
    # so the floor is the backstop: the scale still ratchets to nothing.
    s = _guarded(max_consecutive_skips=1_000_000, scale_floor=2.0**-8)
    with pytest.raises(AmpScalerCollapse) as err:
        for _ in range(200):
            s.skip_next = True
            s.update()
    assert "floor" in str(err.value)


def test_disabled_scaler_is_left_alone():
    # amp off -> no scaler activity at all; the guard must not fire or crash.
    cls = make_guarded_scaler_class(StubScaler, max_consecutive_skips=1)
    s = cls(enabled=False)
    for _ in range(100):
        s.skip_next = True
        s.update()


def test_default_limit_is_not_hair_trigger():
    assert DEFAULT_MAX_CONSECUTIVE_SKIPS >= 20


def test_guard_patches_and_restores_physicsnemo(monkeypatch):
    from omegaconf import OmegaConf

    from lpno.models.pipeline.amp_guard import amp_collapse_guard

    trainer = pytest.importorskip("physicsnemo.sym.trainer")
    original = trainer.GradScaler

    cfg = OmegaConf.create({"amp": {"enabled": True}, "custom": {}})
    with amp_collapse_guard(cfg):
        assert trainer.GradScaler is not original, "guard did not install"
        assert issubclass(trainer.GradScaler, original)
    assert trainer.GradScaler is original, "guard did not restore"


def test_guard_is_a_no_op_without_amp_or_when_switched_off():
    from omegaconf import OmegaConf

    from lpno.models.pipeline.amp_guard import amp_collapse_guard

    trainer = pytest.importorskip("physicsnemo.sym.trainer")
    original = trainer.GradScaler

    for cfg in (
        OmegaConf.create({"amp": {"enabled": False}, "custom": {}}),
        OmegaConf.create({"amp": {"enabled": True}, "custom": {"amp_collapse_guard": False}}),
    ):
        with amp_collapse_guard(cfg):
            assert trainer.GradScaler is original


def test_low_scale_with_progress_is_not_a_collapse():
    """Regression: a run that trains at a very low scale must not be aborted.

    `rigno2579_e8_oc` (2026-08-08) was killed at step 600 of 32000 by the scale
    floor, and it was training: loss 3.725e+05 -> 1.835e+05, with only 29 inf
    events in 600 steps (4.8% of steps skipped). Its scale oscillated in a low
    band (0.0039-0.0625), repeatedly *growing back* -- which under physicsnemo's
    recover_growth_interval=100 takes 100 consecutive successful steps each time.

    A small scale is not evidence of a no-op; on a loss of ~3.7e5 it is exactly
    what the scaler is supposed to find. The signal that actually means "no
    updates are being applied" is consecutive skips, so the floor no longer
    fires by default.
    """
    s = _guarded()  # defaults
    # 29 skips spread over 600 steps, scale ratcheting down into the 1e-3 band
    for step in range(600):
        s.skip_next = (step % 20 == 0)
        s.update()
    assert s.get_scale() < 2.0**-8, "test should drive the scale below the old floor"


def test_scale_floor_is_opt_in_by_default():
    assert DEFAULT_SCALE_FLOOR == 0.0, (
        "a low scale alone is not a collapse -- it killed a healthy run on "
        "2026-08-08; consecutive skips is the sound signal")
