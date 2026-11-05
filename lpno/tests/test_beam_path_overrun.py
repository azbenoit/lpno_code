"""A laser that keeps going past its declared path must keep going in the loader.

`turn_off_at_end: false` makes FLOW-3D hold the last laser row until finish_time: the beam
stays on and, for a scan, keeps moving. `_beam_path_over_time` treated the off time
correctly (`laser_off_time_s = None` -> never off) and then froze the POSITION at the last
knot anyway, so on overrun frames every path channel (beam_x/beam_y/beam_dist/path_dose/
conduction_field) and both moving-frame windows said the beam had stopped while the true
field showed it moving. Measured on ti64_anchorbox_lhs150 (Materials variations, eval 83535
of model 82910): manifest beam pinned at ix 72 for the last 30 frames, Heat Absorption Rate
argmax running 79 -> 85, and the derived steady target 5-12 cells AHEAD of its own beam in
all 15 anchor replicates -- a state no keyhole can be in.

Two manifest eras carry the condition differently, and both are covered:
  post-2026-08-15: `laser_on_past_declared_time: true`, `laser_overrun_distance_m`, off = None
  pre-2026-08-15:  `laser_turn_off_at_end: false` with `laser_off_time_s = scan_time` (the
                   writer's old lie, 2026-08-15)
A manifest with neither flag keeps the old clamp, so legacy fixtures are unchanged.
"""
import numpy as np
import pytest

from lpno.custom_utils import npz_datasource as nds

L, V, K = 1.0e-3, 0.8, 41
SCAN = L / V
FINISH = SCAN * 1.25
OVERRUN_M = V * (FINISH - SCAN)


def _manifest(era):
    xs = np.linspace(0.0, L, K)
    xy = np.stack([xs, np.zeros_like(xs)], axis=1).tolist()
    d = {"laser_path_xy_m": xy, "scan_time_s": SCAN, "scan_path_length_m": L}
    if era == "post":
        d.update(laser_off_time_s=None, laser_turn_off_at_end=False,
                 laser_on_past_declared_time=True, laser_overrun_distance_m=OVERRUN_M)
    elif era == "pre":
        d.update(laser_off_time_s=SCAN, laser_turn_off_at_end=False)
    elif era == "stops":
        d.update(laser_off_time_s=SCAN, laser_turn_off_at_end=True,
                 laser_on_past_declared_time=False, laser_overrun_distance_m=0.0)
    elif era == "legacy":
        d.update(laser_off_time_s=SCAN)
    return {"input": {"laser": {"scan_speed_m_per_s": V, "beam_radius_m": 5e-5}}, "derived": d}


T = np.linspace(0.0, FINISH, 26)   # last 5 frames are past the declared path end


@pytest.mark.parametrize("era", ["post", "pre"])
def test_centre_keeps_moving_past_the_path_end(era):
    centres, directions, speeds = nds._beam_path_over_time(_manifest(era), T)
    assert np.isclose(centres[-1, 0], L + OVERRUN_M), "centre at finish = path end + overrun"
    assert np.isclose(centres[-1, 1], 0.0)
    # Speed and direction stay live on the overrun frames, consistent with the motion.
    assert np.allclose(speeds[T > SCAN], V)
    assert np.allclose(directions[T > SCAN], [1.0, 0.0])
    # Still exact on the declared path.
    assert np.isclose(np.interp(SCAN, T, centres[:, 0]), L, rtol=1e-6)


def test_post_manifest_overrun_matches_the_recorded_distance():
    centres, _, _ = nds._beam_path_over_time(_manifest("post"), T)
    assert np.isclose(centres[-1, 0] - L, _manifest("post")["derived"]["laser_overrun_distance_m"])


def test_pre_manifest_off_time_is_overridden_not_trusted():
    """Old writer wrote laser_off_time_s = scan_time even when the laser never turned off;
    that zeroed speeds/directions past scan_time as well as freezing the centre."""
    _, directions, speeds = nds._beam_path_over_time(_manifest("pre"), T)
    assert (speeds[T > SCAN] > 0).all()
    assert (np.abs(directions[T > SCAN]).sum(axis=1) > 0).all()


def test_a_laser_that_stops_still_stops():
    centres, directions, speeds = nds._beam_path_over_time(_manifest("stops"), T)
    assert np.isclose(centres[-1, 0], L)
    assert np.allclose(speeds[T > SCAN], 0.0)
    assert np.allclose(directions[T > SCAN], 0.0)


def test_legacy_manifest_without_the_flag_keeps_the_clamp():
    centres, _, _ = nds._beam_path_over_time(_manifest("legacy"), T)
    assert np.isclose(centres[-1, 0], L)


def test_window_consumer_sees_the_overrun():
    """Both moving-frame windows read centres through `_beam_centres_over_time`."""
    c = nds._beam_centres_over_time(_manifest("post"), T)
    assert np.isclose(c[-1, 0], L + OVERRUN_M)


def test_variable_speed_path_extrapolates_along_its_last_leg():
    m = {
        "input": {"laser": {"scan_speed_m_per_s": 0.5, "beam_radius_m": 5e-5}},
        "derived": {
            "laser_path_xy_m": [[0.0, 0.0], [1.0e-3, 0.0], [1.0e-3, 1.0e-3]],
            "laser_path_times_s": [0.0, 1.0e-3, 2.0e-3],   # last leg: +y at 1.0 m/s
            "laser_off_time_s": None, "laser_turn_off_at_end": False,
            "laser_on_past_declared_time": True,
        },
    }
    t = np.array([0.0, 1.0e-3, 2.0e-3, 2.5e-3])
    centres, directions, speeds = nds._beam_path_over_time(m, t)
    assert np.allclose(centres[-1], [1.0e-3, 1.0e-3 + 1.0 * 0.5e-3])
    assert np.allclose(directions[-1], [0.0, 1.0]) and np.isclose(speeds[-1], 1.0)


def test_terminal_dwell_stays_put_even_when_the_laser_stays_on():
    m = {
        "input": {"laser": {"scan_speed_m_per_s": 0.5, "beam_radius_m": 5e-5}},
        "derived": {
            "laser_path_xy_m": [[0.0, 0.0], [1.0e-3, 0.0], [1.0e-3, 0.0]],
            "laser_path_times_s": [0.0, 1.0e-3, 2.0e-3],
            "laser_off_time_s": None, "laser_turn_off_at_end": False,
            "laser_on_past_declared_time": True,
        },
    }
    centres, _, _ = nds._beam_path_over_time(m, np.array([2.0e-3, 3.0e-3]))
    assert np.allclose(centres[-1], [1.0e-3, 0.0])
