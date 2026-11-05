"""A MULTI-BEAM case must get the track panel, not the beam-speed curve.

`beam_track_drawer` was written for exactly this ("a multi-beam case has no single scan
speed, so the varspeed curve has nothing to plot") and then never wired into `main()`:
`side_draw` defaulted to None on every CLI path, so every multibeam video rendered the
speed curve instead. That curve is the POWER-WEIGHTED CENTROID of the active beams, so it
jumps discontinuously whenever a beam switches off -- measured 2026-09-04 on mbc40_031,
where the panel shows spikes to 25 m/s on a case whose beams run at ~0.45 m/s. The spikes
are an artefact of the centroid definition, not beam motion, and they are the most
eye-catching thing in the frame.

Single-beam cases must keep the speed curve: there the centroid IS the beam and the trace
is the physical scan speed, which is the whole point of the varspeed figures.
"""
from __future__ import annotations

import pytest

from scripts.render_case_video import side_panel_for


def _manifest(n_beams, with_paths=True):
    beams = []
    for i in range(n_beams):
        b = {"index": i + 1, "power_w": 50.0 + i}
        if with_paths:
            b["path_xy_m"] = [[0.0, 0.0], [1e-4, 0.0]]
            b["path_times_s"] = [0.0, 1e-4]
        beams.append(b)
    return {"derived": {"beam_count": n_beams, "beams": beams}}


def test_single_beam_keeps_the_speed_curve():
    assert side_panel_for(_manifest(1), [0.0, 1e-4]) is None


def test_absent_beam_block_keeps_the_speed_curve():
    assert side_panel_for({"derived": {}}, [0.0, 1e-4]) is None


@pytest.mark.parametrize("n", [2, 3, 4])
def test_multi_beam_gets_the_track_panel(n):
    assert callable(side_panel_for(_manifest(n), [0.0, 1e-4]))


def test_multi_beam_without_staged_paths_falls_back_rather_than_drawing_nothing():
    """The node manifest carries EMPTY per-beam paths; an empty panel is worse than the curve."""
    assert side_panel_for(_manifest(3, with_paths=False), [0.0, 1e-4]) is None
