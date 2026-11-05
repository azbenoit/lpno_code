"""The beam box must be constructible in a beam-following frame without `beam_dist`.

The box is a determined geometric region: beam radius and beam centre fix it. On a
fixed-radius dataset like spkh123 (r0 = 50 um for every case) it is literally the SAME
box every time. Requiring the `beam_dist` CHANNEL is an implementation artifact, and it
silently removed the metric from every steady arm -- `beam_box_from_input` returned None,
`beam_box_validation_metrics` returned {}, and the manuscript's architecture table reached
a reviewer with no beam-box column and nothing saying why.

WHY THE CENTRE IS THE ORIGIN. Measured, not assumed: in the moving-reference frame the
stored `in_beam_dist` of eval 60434 peaks at x = -1.9 um, y = -5.0 um on a 10 um grid --
the origin within half a cell -- and its footprint > 0.5 disc spans +-55 um, matching
r < 1.177*sigma for sigma = 50 um. The beam is pinned at the origin by construction there.

WHY IT IS GATED ON THE FRAME. In the LAB frame the beam moves, so the origin assumption is
false and would place the box over quiescent metal. That is not a degraded metric, it is
the documented failure mode: twice (2026-08-10, 2026-08-15) a broken box silently
reproduced the substrate number under a beam-box name, 8x and 9.6x below the true box.
Lab-frame runs keep requiring the real channel.
"""
from __future__ import annotations

import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import (
    beam_box_from_input,
    beam_box_mask,
)

NX, NY, NZ = 41, 41, 20
SIGMA = 50e-6
CELL = 10e-6


def _grids():
    x = (np.arange(NX) - NX // 2) * CELL
    y = (np.arange(NY) - NY // 2) * CELL
    z = -(np.arange(NZ)[::-1]) * CELL
    xg = x.reshape(1, NX, 1, 1)
    yg = y.reshape(1, 1, NY, 1)
    zg = z.reshape(1, 1, 1, NZ)
    return xg, yg, zg


def _gaussian_beam_dist(xg, yg):
    """The pipeline's own footprint: exp(-r^2/2 sigma^2), peak 1, broadcast over z."""
    r2 = xg**2 + yg**2
    return np.broadcast_to(
        np.exp(-r2 / (2.0 * SIGMA * SIGMA)), (1, NX, NY, NZ)
    ).astype(np.float32)


def test_synthesized_box_equals_the_beam_dist_box():
    """THE anti-divergence test: both routes must produce the SAME mask.

    Both historical beam-box bugs were a second copy of the box definition drifting from
    the first. If the synthesized path ever stops matching the channel path, this fails.
    """
    xg, yg, zg = _grids()
    from_channel = beam_box_from_input(
        {"beam_dist": _gaussian_beam_dist(xg, yg), "x": xg}, zg, n_radii=2.0
    )
    synthesized = beam_box_from_input(
        {"x": xg, "y": yg, "beam_radius": SIGMA}, zg, n_radii=2.0, beam_at_origin=True
    )
    assert from_channel is not None and synthesized is not None
    assert from_channel.shape == synthesized.shape
    assert np.array_equal(from_channel, synthesized), (
        f"routes disagree on {int((from_channel != synthesized).sum())} cells"
    )


def test_lab_frame_still_refuses_without_the_channel():
    """Without beam_at_origin the beam position is unknown -- must not guess."""
    xg, yg, zg = _grids()
    assert beam_box_from_input({"x": xg, "y": yg, "beam_radius": SIGMA}, zg) is None


def test_missing_radius_uses_the_announced_default(capsys):
    """SUPERSEDED by the 2026-08-21 directive: fixed size beats no box.

    This originally asserted a raise. The user's instruction was explicit -- "If beam_dist or
    whatever is not there, USE THE BEAM CENTER WHICH YOU KNOW AND USE A FIXED BOX SIZE" -- because
    three eval rounds had produced an empty beam-box column and the raise was one more way to
    report nothing. The centre is exact in a beam-following frame; only the size is assumed, so
    the assumption is announced on stdout rather than hidden.
    """
    xg, yg, zg = _grids()
    box = beam_box_from_input({"x": xg, "y": yg}, zg, n_radii=2.0, beam_at_origin=True)
    assert box is not None and box.any()
    assert "DEFAULT" in capsys.readouterr().out, "the assumed size must be announced"


def test_beam_dist_still_wins_when_present():
    """The real channel is authoritative; synthesis is only a fallback."""
    xg, yg, zg = _grids()
    off_centre = np.broadcast_to(
        np.exp(-((xg - 20 * CELL) ** 2 + yg**2) / (2.0 * SIGMA * SIGMA)), (1, NX, NY, NZ)
    ).astype(np.float32)
    box = beam_box_from_input(
        {"beam_dist": off_centre, "x": xg, "y": yg, "beam_radius": SIGMA},
        zg, n_radii=2.0, beam_at_origin=True,
    )
    centred = beam_box_from_input(
        {"x": xg, "y": yg, "beam_radius": SIGMA}, zg, n_radii=2.0, beam_at_origin=True
    )
    assert not np.array_equal(box, centred), "beam_dist was ignored in favour of synthesis"


def test_box_is_the_expected_disc():
    """r <= n*sigma, inside the metal, within n*sigma of the surface."""
    xg, yg, zg = _grids()
    box = beam_box_from_input(
        {"x": xg, "y": yg, "beam_radius": SIGMA}, zg, n_radii=2.0, beam_at_origin=True
    )
    r = np.sqrt(np.broadcast_to(xg, box.shape) ** 2 + np.broadcast_to(yg, box.shape) ** 2)
    zz = np.broadcast_to(zg, box.shape)
    in_depth = (zz <= 1e-9) & (zz >= -2.0 * SIGMA)
    # Assert the PROPERTY with a one-cell margin either side of r = 2*sigma rather than
    # mirroring the implementation: at exactly r = 2*sigma the footprint equals the
    # threshold, so inclusion is a float32 tie and testing it would pin an accident.
    assert box[(r <= 2.0 * SIGMA - CELL) & in_depth].all(), "well inside the disc must be in"
    assert not box[r >= 2.0 * SIGMA + CELL].any(), "well outside the disc must be out"
    assert not box[zz > 1e-9].any(), "gas must be excluded"
    assert not box[zz < -2.0 * SIGMA - 1e-12].any(), "below the depth bound must be excluded"
    assert 0 < box.sum() < box.size, "box must be a proper subset of the domain"


def test_uniform_beam_uses_the_top_hat_threshold():
    """A uniform beam's footprint and threshold differ from the Gaussian branch."""
    xg, yg, zg = _grids()
    g = beam_box_from_input({"x": xg, "y": yg, "beam_radius": SIGMA}, zg,
                            n_radii=2.0, beam_at_origin=True)
    u = beam_box_from_input({"x": xg, "y": yg, "beam_radius": SIGMA,
                             "beam_shape_uniform": np.array([1.0])}, zg,
                            n_radii=2.0, beam_at_origin=True)
    assert g is not None and u is not None
    assert not np.array_equal(g, u), "uniform branch collapsed onto the Gaussian one"


def test_dataset_level_radius_is_used_when_no_channel():
    """spkh123 has no beam_radius CHANNEL; adopt_beam_radius supplies it per dataset."""
    xg, yg, zg = _grids()
    box = beam_box_from_input(
        {"x": xg, "y": yg}, zg, n_radii=2.0, beam_at_origin=True, beam_radius=SIGMA
    )
    reference = beam_box_from_input(
        {"x": xg, "y": yg, "beam_radius": SIGMA}, zg, n_radii=2.0, beam_at_origin=True
    )
    assert box is not None and np.array_equal(box, reference)


def test_per_case_channel_beats_the_dataset_fallback():
    """A varying-radius set must not be flattened to one dataset-level value."""
    xg, yg, zg = _grids()
    per_case = beam_box_from_input(
        {"x": xg, "y": yg, "beam_radius": 2.0 * SIGMA}, zg,
        n_radii=2.0, beam_at_origin=True, beam_radius=SIGMA,
    )
    fallback = beam_box_from_input(
        {"x": xg, "y": yg}, zg, n_radii=2.0, beam_at_origin=True, beam_radius=SIGMA
    )
    assert per_case.sum() > fallback.sum(), "the per-case channel was ignored"


def test_raw_metre_radius_against_normalised_grid_is_loud():
    """The unit mismatch that produced an EMPTY box on evals 61523/61528/61530.

    At the metrics stage x and y are already divided by ref_scales["L"], so passing a raw
    metre radius makes r >> sigma everywhere, the footprint underflows to zero and the box
    silently vanishes. It must raise instead.
    """
    # The REAL eval grid: 61x40x40 with x spanning -1.55..4.45 in ref_scales["L"] units.
    # It does not contain exactly x = 0, so with a raw-metre sigma every cell underflows.
    # (An idealised grid that happens to include the exact origin keeps one cell at 1.0
    # and hides the mismatch -- which is why this test uses the real spacing.)
    x = np.linspace(-1.55, 4.45, 61).reshape(1, 61, 1, 1)
    y = np.linspace(-1.95, 1.95, 40).reshape(1, 1, 40, 1)
    z = np.linspace(-2.95, 0.95, 40).reshape(1, 1, 1, 40)
    with pytest.raises(ValueError, match="UNIT|unit"):
        beam_box_from_input({"x": x, "y": y}, z,
                            n_radii=2.0, beam_at_origin=True, beam_radius=SIGMA)


def test_radius_normalised_to_grid_units_builds_the_box():
    """The same call with the radius in grid units works."""
    x = np.linspace(-1.55, 4.45, 61).reshape(1, 61, 1, 1)
    y = np.linspace(-1.95, 1.95, 40).reshape(1, 1, 40, 1)
    z = np.linspace(-2.95, 0.95, 40).reshape(1, 1, 1, 40)
    box = beam_box_from_input({"x": x, "y": y}, z,
                              n_radii=2.0, beam_at_origin=True, beam_radius=0.5)
    assert box is not None
    frac = box.sum() / box.size
    assert 0.01 < frac < 0.20, f"beam box is {frac:.1%} of the domain -- implausible"


def test_beam_radius_in_grid_units_helper():
    from types import SimpleNamespace
    from lpno.models.pipeline.postprocessing import _beam_radius_in_grid_units
    cfg = SimpleNamespace(laser_params={"beam_radius_m": 5e-5}, ref_scales={"L": 1e-4})
    assert _beam_radius_in_grid_units(cfg) == pytest.approx(0.5)
    assert _beam_radius_in_grid_units(SimpleNamespace(laser_params={}, ref_scales={"L": 1e-4})) is None


def test_lazy_metrics_projection_keeps_the_beam_CENTRE():
    """The lazy-eval call site must pass `beam_x`/`beam_y` to the shared builder.

    `beam_box_from_input` was taught (d0cb078, 2026-08-22) to build the box from the
    beam CENTRE so `beam_dist` is not required. That fix was unreachable from
    `run_lazy_dataset_metrics`, which projects the batch through a fixed key list
    before calling it -- and the list omitted `beam_x`/`beam_y`. So every arm whose
    channel set drops `beam_dist` (the 8-channel ablation among them) still returned
    NO beambox_* keys at all: measured on 82336/82400 (2026-09-01), which carry
    beam_x and beam_y and produced substrate metrics but no beam-box column.

    This is the fifth time this metric has gone missing through a builder fix that
    a caller never reached, so the contract is pinned here rather than left implicit.
    """
    import numpy as np
    from lpno.models.pipeline.postprocessing import (
        BEAM_BOX_INPUT_KEYS,
        beam_box_from_input,
    )

    for key in ("beam_x", "beam_y"):
        assert key in BEAM_BOX_INPUT_KEYS, (
            f"{key} missing from BEAM_BOX_INPUT_KEYS: the projection would drop the beam "
            "centre and the builder would decline, removing beambox_* silently."
        )

    # Behavioural: an 8-channel-style batch (centre, no beam_dist) projected through the
    # real key list must still yield a box in the LAB frame, where beam_at_origin is off.
    batch = {
        "x": np.linspace(-1.55, 4.45, 61).reshape(1, 61, 1, 1),
        "y": np.linspace(-1.95, 1.95, 40).reshape(1, 1, 40, 1),
        "beam_x": np.float32(0.0),
        "beam_y": np.float32(0.0),
        "T": np.zeros((1, 61, 40, 40), dtype=np.float32),  # not a beam-box input
    }
    z = np.linspace(-2.95, 0.95, 40).reshape(1, 1, 1, 40)
    projected = {k: batch[k] for k in BEAM_BOX_INPUT_KEYS if batch.get(k) is not None}
    assert "T" not in projected, "the projection should still drop non-beam-box inputs"

    box = beam_box_from_input(projected, z, n_radii=2.0,
                              beam_at_origin=False, beam_radius=0.5)
    assert box is not None, "no beam box from a batch that states the beam centre"
    frac = box.sum() / box.size
    assert 0.01 < frac < 0.20, f"beam box is {frac:.1%} of the domain -- implausible"
