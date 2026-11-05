"""The steady window must end where the beam leaves its DECLARED path.

`turn_off_at_end: false` keeps the laser MOVING after its path is done, so the export tail
is a stretch of frames the beam spends against the +x wall. `beam_following_x_slice` clamps
the window there, and the beam lands at window index 57-61 instead of the nominal ~41 -- a
placement no training case has, because their paths end mid-domain and never clamp.

Measured on the anchorbox anchor replicates
(2026-09-01): one model scored substrate MAE 96.5 K / fl IoU 0.370 against those clamped
frames, versus 18.4 K / 0.886 on the 30 laser-on frames before the path end. A ~10x penalty
that is entirely which frames the window lands on.

`frame_transfer.laser_on_frame_stop` (lpno 73e2be7) says where the tail ends.
This module has TWO writers that must both honour it -- the serial
`derive_simulation_arrays` and the parallel-worker `apply_derivation` -- which is the
duplicated-builders trap this file exists to pin shut.
"""
import inspect

import numpy as np
import pytest

from aiida_pino.data.pino_dataset_collection import _crop_shift_moving_frame_single
from aiida_pino.utils.runtime_export import apply_derivation, derive_time_values
frame_transfer = pytest.importorskip(
    "lpno.models.pipeline.frame_transfer",
    reason="needs the sibling lpno checkout installed (README: Companion repositories)",
)
beam_following_slices = frame_transfer.beam_following_slices

NX = 90
STEPS = 30
WIDTH = NX - (STEPS - 1)          # 61, the legacy window width
CLAMPED_START = NX - WIDTH        # 29, and it is the MAXIMUM start: a window pinned
                                  # against the +x wall can have no other.
BEHIND = 41                       # legacy_window_split(61) -> (41, 20); the beam's
                                  # nominal index inside an unclamped window.
PATH_END_IX = 65                  # declared path end: start 24, beam at index 41
STOP = 30                         # frames 0..29 are on the declared path


def _overrun_case(n_frames=40):
    """A beam reaching ix 65 at frame 29, then overrunning to the +x wall by frame 39.

    At ix 65 the window is comfortably unclamped (start 24, beam at the nominal 41). By
    ix 89 the desired start is 48, the clamp pins it at 29, and the beam lands at window
    index 60 -- the 57-61 placement measured on the anchorbox replicates.
    """
    on_path = np.linspace(40.0, float(PATH_END_IX), STOP)
    overrun = np.linspace(72.0, 89.0, n_frames - STOP + 1)[1:]
    beam = np.concatenate([on_path, overrun])
    data = np.zeros((n_frames, 1, NX, 2, 2), dtype=np.float32)
    for t, centre in enumerate(beam):
        data[t, 0, int(round(centre))] = 1.0
    centres = np.stack([beam, np.zeros_like(beam)], axis=1)
    return data, centres, np.arange(NX, dtype=np.float64)


def _hot_index(frame):
    return int(np.argmax(frame[0, :, 0, 0]))


def test_without_the_stop_the_window_is_pinned_against_the_wall():
    """The bug, stated as a measurement: this is what every overrun export currently holds."""
    data, centres, x = _overrun_case()
    slices = beam_following_slices(centres, x, steps=STEPS, super_res_factor=1)
    assert slices[-1].start == CLAMPED_START, "fixture does not reproduce the clamp"
    out = _crop_shift_moving_frame_single(data, STEPS, super_res_factor=1, slices=slices)
    assert _hot_index(out[-1]) >= 55, "fixture does not reproduce the 57-61 placement"


def test_the_stop_ends_the_window_at_the_declared_path_end():
    data, centres, x = _overrun_case()
    slices = beam_following_slices(centres, x, steps=STEPS, super_res_factor=1, stop=STOP)
    assert slices[-1].start < CLAMPED_START, "window is still pinned against the wall"
    assert slices[-1].start == PATH_END_IX - BEHIND
    out = _crop_shift_moving_frame_single(
        data, STEPS, super_res_factor=1, slices=slices, stop=STOP
    )
    assert out.shape == (STEPS, 1, WIDTH, 2, 2)
    hot = [_hot_index(out[t]) for t in range(STEPS)]
    assert max(hot) - min(hot) <= 1, f"beam drifted {max(hot) - min(hot)} cells in the window"
    assert max(hot) == BEHIND, f"beam sits at window index {max(hot)}, not the nominal {BEHIND}"


def test_the_array_cut_selects_the_frames_the_slices_were_built_for():
    """The slices describe frames [stop-steps, stop); the array must be cut to match.

    Cropping the LAST `steps` frames with slices built for an EARLIER window silently
    pairs each frame with another frame's x-offset -- worse than the clamp, because the
    result still has the right shape.
    """
    data, centres, x = _overrun_case()
    slices = beam_following_slices(centres, x, steps=STEPS, super_res_factor=1, stop=STOP)
    out = _crop_shift_moving_frame_single(
        data, STEPS, super_res_factor=1, slices=slices, stop=STOP
    )
    manual = _crop_shift_moving_frame_single(
        data[:STOP], STEPS, super_res_factor=1, slices=slices
    )
    assert np.array_equal(out, manual)


def test_a_stopping_laser_is_untouched():
    """`stop=None` and `stop=T` must both mean "the whole axis"; no behaviour change."""
    data, centres, x = _overrun_case()
    slices = beam_following_slices(centres, x, steps=STEPS, super_res_factor=1)
    plain = _crop_shift_moving_frame_single(data, STEPS, super_res_factor=1, slices=slices)
    explicit = _crop_shift_moving_frame_single(
        data, STEPS, super_res_factor=1, slices=slices, stop=data.shape[0]
    )
    assert np.array_equal(plain, explicit)


def test_the_worker_writer_honours_the_stop_too():
    """`apply_derivation` is the parallel-export copy; a fix in one writer is half a fix."""
    data, centres, x = _overrun_case()
    slices = beam_following_slices(centres, x, steps=STEPS, super_res_factor=1, stop=STOP)
    pairs = [(int(s.start), int(s.stop)) for s in slices]
    profile = {"moving_ref_frame": True, "moving_frame_steps": STEPS}
    out = apply_derivation(data, profile, pairs, stop=STOP)
    expected = _crop_shift_moving_frame_single(
        data, STEPS, super_res_factor=1, slices=slices, stop=STOP
    )
    assert np.array_equal(out, expected)


def test_the_serial_writer_passes_the_stop():
    """Source guard: the serial path builds the slices, so it must cut the array as well."""
    from aiida_pino.data.pino_dataset_collection import derive_simulation_arrays

    src = inspect.getsource(derive_simulation_arrays)
    assert "moving_frame_stop" in src, "serial writer does not carry the stop"
    assert "stop=moving_frame_stop" in src, "serial writer does not apply the stop"


def test_the_worker_task_ships_the_stop():
    """The worker has no AiiDA session, so the parent must send the stop with the slices."""
    from aiida_pino.data import pino_dataset_collection as mod

    src = inspect.getsource(mod)
    assert '"moving_frame_stop"' in src, "the export task does not carry the stop"


def test_time_values_are_cut_at_the_stop():
    """`time_s` must name the instants the kept frames are, or the axis lies about them."""
    times = np.arange(40, dtype=np.float32) * 1e-5
    out = derive_time_values(times, {"moving_ref_frame": True, "moving_frame_steps": STEPS},
                             stop=STOP)
    assert out.shape == (STEPS,)
    assert np.allclose(out, times[:STOP][-STEPS:])
    assert float(out[-1]) < float(times[-1]), "time axis still ends in the overrun"


def test_time_values_without_a_stop_are_unchanged():
    times = np.arange(40, dtype=np.float32) * 1e-5
    profile = {"moving_ref_frame": True, "moving_frame_steps": STEPS}
    assert np.array_equal(derive_time_values(times, profile),
                          derive_time_values(times, profile, stop=None))


def test_the_slice_builder_reports_the_stop():
    """`_beam_following_slices_for` owns the manifest, so it is what can compute the stop."""
    from aiida_pino.data.pino_dataset_collection import _beam_following_slices_for

    src = inspect.getsource(_beam_following_slices_for)
    assert "laser_on_frame_stop" in src
    assert "return " in src and "stop" in src.split("return")[-1], "stop is not returned"
