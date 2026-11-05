"""The steady/moving-frame tail must end where the beam leaves its declared path.

Every consumer takes "the last N frames" of the export. For a laser that keeps moving past
its declared path end (`turn_off_at_end: false`, scan shorter than the run) those frames
are the overrun: on the anchorbox anchor replicates the beam reaches ix 86-90 of 90, the
beam-following window clamps at start 29, and the beam sits at window index 57-61 instead
of the nominal 41 -- a placement no training case has. Model 83645 scored substrate MAE
96.5 K / fl IoU 0.370 there against 18.4 K / 0.886 on the 30 laser-on frames before the
path end (Materials variations, 2026-09-01). `laser_on_frame_stop` is the single selector;
`beam_following_slices`, `to_steady`, `steady_entry_selection` and the eager
`_apply_moving_frame` all take its result.
"""
import numpy as np
import pytest

from lpno.custom_utils import npz_datasource as nds
from lpno.models.pipeline import frame_transfer as ft

L, V = 5.5e-4, 0.8              # anchorbox track: -0.275..+0.275 mm on a 0.9 mm grid
SCAN = L / V
T_END = SCAN * 1.25            # 20 % of the frames are overrun
NT = 100
T = np.linspace(0.0, T_END, NT)
N_ON = int(np.count_nonzero(T <= SCAN))


def _manifest(era="post", length=L):
    xs = np.linspace(-length / 2, length / 2, 41)   # path end at ix 72 of 90
    xy = np.stack([xs, np.zeros_like(xs)], axis=1).tolist()
    d = {"laser_path_xy_m": xy, "scan_time_s": SCAN, "scan_path_length_m": length}
    if era == "post":
        d.update(laser_off_time_s=None, laser_turn_off_at_end=False,
                 laser_on_past_declared_time=True, laser_overrun_distance_m=V * (T_END - SCAN))
    elif era == "pre":
        d.update(laser_off_time_s=SCAN, laser_turn_off_at_end=False)
    elif era == "stops":
        d.update(laser_off_time_s=SCAN, laser_turn_off_at_end=True, laser_on_past_declared_time=False)
    elif era == "dwell":
        d.update(laser_off_time_s=None, laser_turn_off_at_end=False, laser_on_past_declared_time=True,
                 laser_overrun_distance_m=0.0, scan_path_length_m=0.0,
                 laser_path_xy_m=[[0.0, 0.0], [0.0, 0.0]])   # stationary
    return {"input": {"laser": {"scan_speed_m_per_s": V, "beam_radius_m": 5e-5}}, "derived": d}


@pytest.mark.parametrize("era", ["post", "pre"])
def test_overrun_stops_the_tail_at_the_path_end(era):
    ft._STOP_ANNOUNCED.clear()
    assert ft.laser_on_frame_stop(_manifest(era), T) == N_ON < NT


@pytest.mark.parametrize("era", ["stops", "dwell"])
def test_no_moving_overrun_keeps_the_export_end(era):
    assert ft.laser_on_frame_stop(_manifest(era), T) == NT


def test_truncated_axis_agrees_with_the_selector():
    """A derived dataset already cut at the path end and this selector must not compound."""
    assert ft.laser_on_frame_stop(_manifest("post"), T[:N_ON]) == N_ON


def test_slices_come_from_the_laser_on_frames_and_are_beam_centred():
    """Export tail: beam 15+ cells past the nominal split (wall-pinned). Laser-on tail:
    centred where there is room, and within the 2-cell edge clamp on the LAST frame only
    (beam at the path end, start 31 wanted, 29 allowed). Over the 30-frame average a clean
    case's beam sits at window index ~40.4 (Materials variations, 7 cases), so that last
    frame's 42-43 is not the trained placement -- ~41 is."""
    x = np.linspace(-4.45e-4, 4.45e-4, 90); x0, dx = x[0], x[1] - x[0]   # path end at ix 72
    m = _manifest("post"); c = ft.beam_centres_from_manifest(m, T)
    stop = ft.laser_on_frame_stop(m, T)
    ix = lambda i: int(np.round((c[i, 0] - x0) / dx))
    tail_only = ft.beam_following_slices(c, x, steps=30)
    on_path = ft.beam_following_slices(c, x, steps=30, stop=stop)
    assert tail_only[-1].start == 90 - 61 == 29 and ix(NT - 1) - tail_only[-1].start >= 55
    assert on_path[0].start == ix(stop - 30) - 41, "unclamped where there is room"
    assert ix(stop - 1) - on_path[-1].start <= 43, "at most the 2-cell edge clamp at the path end"


def test_to_steady_and_entry_selection_honour_the_stop():
    a = np.arange(NT, dtype=np.float32).reshape(1, NT, 1, 1, 1, 1)
    assert float(ft.to_steady(a, 4, stop=N_ON)[0, 0, 0, 0, 0, 0]) == np.mean(np.arange(N_ON - 4, N_ON))
    entries = [(0, t) for t in range(NT)] + [(1, t) for t in range(NT)]
    sel, ids = ft.steady_entry_selection(entries, 3, stop_by_case={0: N_ON})
    assert sel[:3] == [N_ON - 3, N_ON - 2, N_ON - 1], "case 0: before its stop"
    assert sel[3:] == [NT + NT - 3, NT + NT - 2, NT + NT - 1], "case 1: export tail"


def test_eager_moving_frame_windows_each_case_on_its_own_stop():
    nx, ny, nz, steps = 90, 3, 2, 30
    x = np.linspace(-4.45e-4, 4.45e-4, nx); x0, dx = x[0], x[1] - x[0]
    field = np.broadcast_to(np.arange(NT, dtype=np.float32).reshape(1, NT, 1, 1, 1, 1),
                            (2, NT, 1, nx, ny, nz)).copy()
    m = _manifest("post"); c = ft.beam_centres_from_manifest(m, T)
    centres = np.stack([c, c], axis=0)
    out = nds._apply_moving_frame({"f": field}, steps, steps, 1, centres=centres, x0=x0, dx=dx,
                                  stops=np.array([N_ON, NT]))
    assert out["f"].shape == (2, 1, nx - 29, ny, nz)
    assert np.isclose(out["f"][0, 0, 0, 0, 0], np.mean(np.arange(N_ON - steps, N_ON)))
    assert np.isclose(out["f"][1, 0, 0, 0, 0], np.mean(np.arange(NT - steps, NT)))


def test_a_stop_shorter_than_the_window_is_refused():
    with pytest.raises(ValueError, match="fewer than moving_frame_steps"):
        ft.beam_following_slices(np.zeros((NT, 2)), np.linspace(0, 1, 90), steps=30, stop=10)


@pytest.mark.parametrize("beyond, expect_delta", [(0.3, 0), (0.7, 1)])
def test_last_frame_within_half_a_frame_of_the_path_end_is_on_path(beyond, expect_delta):
    """`finish_time == scan_time` exports end a rounding error past scan_time; that is not
    an overrun. Only a frame more than half an interval beyond the path end counts."""
    ft._STOP_ANNOUNCED.clear()
    dt = SCAN / 99
    t = np.concatenate([np.linspace(0.0, SCAN, 100)[:-1], [SCAN + beyond * dt]])
    assert ft.laser_on_frame_stop(_manifest("post"), t) == t.size - expect_delta
