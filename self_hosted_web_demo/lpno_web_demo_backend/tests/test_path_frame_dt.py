"""Frame times on /infer_path are set by a time step, not a fixed frame count.

A fixed 20 frames meant Δt silently tracked path length and speed: a 1.5 mm
weld and a 6 mm weld both got 20 frames, so the same animation covered wildly
different physical durations and two runs were not comparable. Frames are now
spaced by `frame_dt_s`, and the count follows from the weld duration.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.adapter_randompath import (
    DEFAULT_FRAME_DT_S,
    MAX_FRAMES,
    MIN_FRAMES,
    ModelPhysics,
    RandomPathAdapter,
)
from app.config import Settings

# The calibration point: a 1.5 mm path at the variable-path model's default
# scan speed must come out at 20 frames on the default step.
REF_LENGTH_M = 1.5e-3
REF_V_MPS = 0.6


@pytest.fixture(scope="module")
def adapter() -> RandomPathAdapter:
    return RandomPathAdapter(Settings(use_stub_runtime=True), stub=True)


def _straight_path(length_m: float) -> list[list[float]]:
    """A straight run along x. The keep-out box is x in +-5.5e-4, so this tops
    out at 1.1 mm — the reference 1.5 mm path has to turn (see _ref_path)."""
    return [[-length_m / 2.0, 0.0], [length_m / 2.0, 0.0]]


def _ref_path() -> list[list[float]]:
    """An L of arc length exactly 1.5 mm inside the keep-out box:
    1.1 mm along x, then 0.4 mm along y."""
    return [[-5.5e-4, -2.0e-4], [5.5e-4, -2.0e-4], [5.5e-4, 2.0e-4]]


def _varpath_physics(**over) -> ModelPhysics:
    """Mirrors rigno_varpath's meta.json: variable P/V (so the fixture may set
    the speed), duration-scaled times, 10 µs training dt, no length trim."""
    base = dict(
        duration_scaled_times=True,
        dt_s=1.0e-5,
        start_anywhere=True,
        p_range=(31.0, 116.0),
        v_range=(0.24, 0.99),
        v_default=REF_V_MPS,
        max_path_length_m=None,
    )
    base.update(over)
    return ModelPhysics(**base)


def _resolve(adapter, points, v, frame_dt_s=None, physics=None):
    ph = physics if physics is not None else _varpath_physics()
    return adapter.resolve(
        points_xy_m=points,
        preset=None,
        physics=ph,
        p_watts=None,
        # Fixed-physics models reject an explicit speed (422), so only pass one
        # when the model actually has a speed to set.
        v_scan_mps=v if ph.variable else None,
        frame_dt_s=frame_dt_s,
    )


def test_default_step_gives_20_frames_at_the_reference_weld(adapter) -> None:
    case = _resolve(adapter, _ref_path(), REF_V_MPS)
    assert case.path_length_m == pytest.approx(REF_LENGTH_M, rel=0.02)
    assert case.times.size == 20


def test_frame_count_tracks_duration_at_a_fixed_step(adapter) -> None:
    """The whole point: double the weld duration, double the frames — instead
    of 20 frames covering twice the physical time."""
    short = _resolve(adapter, _straight_path(5.5e-4), REF_V_MPS)
    long = _resolve(adapter, _straight_path(1.1e-3), REF_V_MPS)
    assert long.times.size > short.times.size
    # 2x the path at one speed -> 2x the duration -> ~2x the intervals.
    assert (long.times.size - 1) == pytest.approx(2 * (short.times.size - 1), rel=0.15)
    # ...and halving the speed doubles the duration the same way.
    slow = _resolve(adapter, _straight_path(5.5e-4), REF_V_MPS / 2.0)
    assert (slow.times.size - 1) == pytest.approx(2 * (short.times.size - 1), rel=0.15)


def test_step_is_actually_the_frame_spacing(adapter) -> None:
    case = _resolve(adapter, _ref_path(), REF_V_MPS)
    d = np.diff(case.times)
    # Frames snap to the training-dt dose grid, so spacing is uniform to within
    # one fine step rather than exactly DEFAULT_FRAME_DT_S.
    assert d.max() - d.min() <= 1.0e-5 + 1e-12
    assert float(d.mean()) == pytest.approx(DEFAULT_FRAME_DT_S, rel=0.1)


def test_smaller_step_gives_more_frames(adapter) -> None:
    coarse = _resolve(adapter, _ref_path(), REF_V_MPS, frame_dt_s=DEFAULT_FRAME_DT_S)
    fine = _resolve(adapter, _ref_path(), REF_V_MPS, frame_dt_s=DEFAULT_FRAME_DT_S / 4.0)
    assert fine.times.size > coarse.times.size


def test_frames_never_exceed_the_cap(adapter) -> None:
    """A tiny step on a long slow weld would otherwise ask for thousands of
    renders (and a multi-minute request)."""
    case = _resolve(adapter, _ref_path(), 0.25, frame_dt_s=1.0e-7)
    assert case.times.size == MAX_FRAMES


def test_frames_never_drop_below_the_floor(adapter) -> None:
    case = _resolve(adapter, _ref_path(), REF_V_MPS, frame_dt_s=1.0)
    assert case.times.size == MIN_FRAMES


def test_frames_still_span_the_whole_weld(adapter) -> None:
    """Changing the step must not change WHICH window is shown: still 0..the
    real weld duration, endpoints included."""
    for dt in (DEFAULT_FRAME_DT_S, DEFAULT_FRAME_DT_S / 3.0, 1.0):
        case = _resolve(adapter, _ref_path(), REF_V_MPS, frame_dt_s=dt)
        assert case.times[0] == pytest.approx(0.0)
        assert case.times[-1] == pytest.approx(REF_LENGTH_M / REF_V_MPS, rel=0.02)
        assert np.all(np.diff(case.times) > 0.0)


def test_fixed_duration_models_keep_their_window_but_honour_the_step(adapter) -> None:
    """rp3 & friends were trained on one ~2 ms window, so the span must stay
    put — only how finely it is sampled follows the step."""
    fixed = ModelPhysics()  # duration_scaled_times False
    coarse = _resolve(adapter, _ref_path(), REF_V_MPS, physics=fixed)
    fine = _resolve(adapter, _ref_path(), REF_V_MPS, frame_dt_s=2.0e-4, physics=fixed)
    assert coarse.times[-1] == pytest.approx(2.0e-3)
    assert fine.times[-1] == pytest.approx(2.0e-3)
    assert fine.times.size < coarse.times.size


def test_render_frames_stay_on_the_dose_grid(adapter) -> None:
    """times_fine carries the dose integral at the training dt; every rendered
    frame must be one of its samples or the dose lookup misaligns."""
    case = _resolve(adapter, _ref_path(), REF_V_MPS, frame_dt_s=DEFAULT_FRAME_DT_S / 2.0)
    assert case.frame_idx is not None
    assert np.array_equal(case.times, case.times_fine[case.frame_idx])
    assert case.frame_idx.max() < case.times_fine.size
