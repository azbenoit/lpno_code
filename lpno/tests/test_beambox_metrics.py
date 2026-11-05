"""Errors restricted to the beam box: the region that actually matters.

Whole-domain and substrate-wide averages are diluted by quiescent metal, so they
cannot see a failure confined to the melt pool. Measured 2026-08-03: model 29125
held T RMSE 84 K and alpha IoU 0.998 while predicting ZERO melt (melt volume 0.0
against 357275 true) -- the entire error lived in a few hundred cells around the
beam, and every whole-domain average absorbed it.

The box is built from the `beam_dist` footprint, NOT from a configured radius.
A first version looked up `beam_radius_m` in `run_cfg.laser_params`: no saved run
config carries it (129/129 have only type/x0/z0), so the metric silently never
fired, and a metres radius compared against `x/ref_scales["L"]` would have been
wrong by 1e4 anyway. `beam_dist` is `exp(-r**2/2 sigma**2)` evaluated on the real
grid, so units and the per-frame beam position are correct by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import (
    beam_box_error_stats,
    beam_box_from_input,
    beam_box_mask,
    beam_sigma_from_footprint,
)

NX = NY = 21
NZ = 8
SIGMA = 2.0  # grid units


def _grid(beam_ix=10, beam_iy=10, nb=1):
    """(beam_dist, x, z) shaped (nb, 1, NX, NY, NZ); z<=0 for the lower half."""
    xs = np.arange(NX, dtype=np.float64) - 10.0
    ys = np.arange(NY, dtype=np.float64) - 10.0
    zs = np.arange(NZ, dtype=np.float64) - (NZ - 2)  # ..., -1, 0, 1
    beams = []
    for b in range(nb):
        cx = xs[beam_ix + b]
        cy = ys[beam_iy]
        r2 = (xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2
        fp = np.exp(-r2 / (2 * SIGMA**2))
        beams.append(np.broadcast_to(fp[:, :, None], (NX, NY, NZ)))
    beam = np.stack(beams)[:, None]
    x = np.broadcast_to(xs[None, None, :, None, None], (nb, 1, NX, NY, NZ))
    z = np.broadcast_to(zs[None, None, None, None, :], (nb, 1, NX, NY, NZ))
    return beam.copy(), x.copy(), z.copy()


def test_sigma_is_recovered_from_the_footprint():
    """No config plumbing: the exp(-0.5) contour is one sigma by construction."""
    beam, x, _ = _grid()
    assert beam_sigma_from_footprint(beam, x) == pytest.approx(SIGMA, abs=0.6)


def test_box_is_inside_the_metal_only():
    beam, x, z = _grid()
    box = beam_box_mask(beam, z, n_radii=2.0, sigma=SIGMA)
    assert box is not None
    assert not box[z > 0].any(), "box must never include the gas layer"


def test_depth_is_bounded_not_a_full_column():
    """Without a depth bound the box is mostly deep quiescent metal."""
    beam, x, z = _grid()
    bounded = beam_box_mask(beam, z, n_radii=2.0, sigma=SIGMA)
    unbounded = beam_box_mask(beam, z, n_radii=2.0, sigma=None)
    assert bounded.sum() < unbounded.sum()


def test_error_outside_the_box_is_not_counted():
    beam, x, z = _grid()
    box = beam_box_mask(beam, z, n_radii=2.0, sigma=SIGMA)
    truth = np.zeros_like(beam)
    pred = np.zeros_like(beam)
    pred[0, 0, 0, 0, 0] = 100.0  # far corner
    stats = beam_box_error_stats(pred, truth, box=box)
    assert stats["beambox_abs_err_mean"] == pytest.approx(0.0)
    assert stats["beambox_n"] > 0


def test_box_follows_a_per_frame_beam_position():
    """Each batch row carries its own beam position; the box must track it.

    `beam_x`/`beam_y` are per-frame fields, so a scalar beam centre taken from
    frame 0 would pin the box to the start of the track for the whole batch.
    Building the mask elementwise from `beam_dist` avoids that entirely.
    """
    beam, x, z = _grid(nb=3)
    box = beam_box_mask(beam, z, n_radii=2.0, sigma=SIGMA)
    assert box.shape[0] == 3
    centres = [np.argwhere(box[b, 0, :, :, :].any(axis=(1, 2))).mean() for b in range(3)]
    assert centres[0] < centres[1] < centres[2], f"box did not track the beam: {centres}"


def test_batch_greater_than_one_does_not_crash():
    """The first version reshaped coordinate fields and raised for B>1."""
    beam, x, z = _grid(nb=4)
    box = beam_box_mask(beam, z, n_radii=2.0, sigma=SIGMA)
    truth = np.zeros_like(beam)
    pred = np.ones_like(beam)
    stats = beam_box_error_stats(pred, truth, box=box)
    assert stats["beambox_abs_err_mean"] == pytest.approx(1.0)


def test_box_iou_catches_zero_melt_that_global_iou_misses():
    """The motivating failure: global alpha IoU 0.998, melt actually zero."""
    beam, x, z = _grid()
    box = beam_box_mask(beam, z, n_radii=2.0, sigma=SIGMA)
    truth = np.zeros_like(beam)
    truth[box] = 1.0            # all the melt is in the box
    pred = np.zeros_like(beam)  # model predicts none of it

    inside = beam_box_error_stats(pred, truth, box=box, iou=True)
    assert inside["beambox_iou"] == pytest.approx(0.0), "box IoU must expose the failure"

    whole = np.logical_and(pred > 0.5, truth > 0.5).sum() / max(
        np.logical_or(pred > 0.5, truth > 0.5).sum(), 1
    )
    everywhere = (pred > 0.5) == (truth > 0.5)
    assert everywhere.mean() > 0.9, "whole-domain agreement stays high — the dilution"
    assert whole < 0.5


def test_no_box_when_the_laser_is_off():
    """`beam_dist` is gated to zero past laser_off_time_s: report nothing."""
    beam, x, z = _grid()
    assert beam_box_mask(np.zeros_like(beam), z, n_radii=2.0, sigma=SIGMA) is None


def test_stats_empty_for_an_empty_box():
    beam, x, z = _grid()
    empty = np.zeros_like(beam, dtype=bool)
    assert beam_box_error_stats(beam, beam, box=empty) == {}


def test_sigma_and_box_do_not_depend_on_batch_size():
    """Two frames with the beam 10 cells apart, stacked as one batch (time_as_batch),
    must recover the per-frame radius, not the span of both beam positions -- and the
    box must be exactly the union of the two frames' own boxes (2cbfc60: batch 2 gave a
    1.71x larger box and flattered every tabled beam-box number)."""
    b1, x1, z1 = _grid(beam_ix=5)
    b2, x2, z2 = _grid(beam_ix=15)
    beam = np.concatenate([b1, b2]); x = np.concatenate([x1, x2]); z = np.concatenate([z1, z2])
    assert beam_sigma_from_footprint(beam, x) == pytest.approx(SIGMA, abs=0.6)
    batched = beam_box_from_input({"beam_dist": beam, "x": x}, z)
    single = [beam_box_from_input({"beam_dist": b, "x": xx}, zz) for b, xx, zz in ((b1, x1, z1), (b2, x2, z2))]
    assert batched.sum() == sum(int(s.sum()) for s in single)
