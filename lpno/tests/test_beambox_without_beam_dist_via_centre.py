"""The beam box must not require `beam_dist` when the beam centre is otherwise known.

Origin (2026-08-22): the 8-channel mref ablation returned NO beam box. Its key set is
[V_scan,P,H_norm,x,y,z,beam_x,beam_y] -- it states the beam position outright -- but the builder
keyed on `beam_dist` and declined. That made a channel ablation silently drop a whole metric
column, and it was the tenth time this metric had gone missing.
"""
import numpy as np

from lpno.models.pipeline.postprocessing import beam_box_from_input


def _grids(nx=41, ny=41, nz=12, off=(0.0, 0.0)):
    x = np.linspace(-1.0, 1.0, nx)[None, None, :, None, None] + off[0]
    y = np.linspace(-1.0, 1.0, ny)[None, None, None, :, None] + off[1]
    z = np.linspace(-1.0, 0.0, nz)[None, None, None, None, :]
    return np.broadcast_to(x, (1, 1, nx, ny, nz)).copy(), \
           np.broadcast_to(y, (1, 1, nx, ny, nz)).copy(), \
           np.broadcast_to(z, (1, 1, nx, ny, nz)).copy()


def test_beam_x_y_are_enough_no_beam_dist_and_no_origin_flag():
    xg, yg, zg = _grids()
    box = beam_box_from_input(
        {"x": xg, "y": yg, "beam_x": np.zeros_like(xg), "beam_y": np.zeros_like(yg)},
        zg, n_radii=2.0, beam_radius=0.25,
    )
    assert box is not None and box.any(), "a stated beam centre must be enough"


def test_the_box_FOLLOWS_the_stated_centre():
    """The centre must actually be used, not merely accepted."""
    xg, yg, zg = _grids()
    shifted = np.full_like(xg, 0.5)
    a = beam_box_from_input({"x": xg, "y": yg, "beam_x": np.zeros_like(xg),
                             "beam_y": np.zeros_like(yg)}, zg, n_radii=2.0, beam_radius=0.25)
    b = beam_box_from_input({"x": xg, "y": yg, "beam_x": shifted,
                             "beam_y": np.zeros_like(yg)}, zg, n_radii=2.0, beam_radius=0.25)
    ax = np.argwhere(a)[:, 2].mean()
    bx = np.argwhere(b)[:, 2].mean()
    assert bx > ax + 2, f"box did not move with beam_x: {ax:.1f} -> {bx:.1f}"


def test_still_declines_when_the_centre_is_genuinely_unknown():
    """A lab-frame run with no beam channel at all must still return None, not guess."""
    xg, yg, zg = _grids()
    assert beam_box_from_input({"x": xg, "y": yg}, zg, n_radii=2.0, beam_radius=0.25) is None
