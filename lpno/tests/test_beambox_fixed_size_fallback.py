"""A beam-following frame ALWAYS gets a box: known centre, fixed size if the radius is not.

User directive 2026-08-21: "If beam_dist or whatever is not there, USE THE BEAM CENTER WHICH
YOU KNOW AND USE A FIXED BOX SIZE." The column must not keep coming back empty.

Three eval rounds produced no beambox_* keys because `run_lazy_dataset_metrics` built the mask
itself -- a FOURTH copy of the construction, gated on `"beam_dist" in invar` -- so none of the
synthesis work in beam_box_from_input was reachable from the lazy path at all. Raising there
would not have helped either; nothing was calling it.

Precedence for the size: the per-case `beam_radius` channel, else the dataset-level value from
adopt_beam_radius, else DEFAULT_BEAM_RADIUS_M. The default is announced, never silent, because a
box of assumed size is a different claim from a box of measured size.
"""
from __future__ import annotations

import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import (
    DEFAULT_BEAM_RADIUS_M,
    beam_box_from_input,
)

L = 1e-4
SIGMA = 5e-5


def _grids():
    x = np.linspace(-1.55, 4.45, 61).reshape(1, 61, 1, 1)
    y = np.linspace(-1.95, 1.95, 40).reshape(1, 1, 40, 1)
    z = np.linspace(-2.95, 0.95, 40).reshape(1, 1, 1, 40)
    return x, y, z


def test_no_radius_anywhere_still_returns_a_box():
    """The directive: no beam_dist, no beam_radius -> fixed size, not None and not a raise."""
    x, y, z = _grids()
    box = beam_box_from_input({"x": x, "y": y}, z, n_radii=2.0, beam_at_origin=True)
    assert box is not None, "a beam-following frame must always yield a box"
    frac = box.sum() / box.size
    assert 0.01 < frac < 0.20, f"box is {frac:.1%} of the domain -- implausible"


def test_fixed_size_matches_the_declared_default():
    """The fallback must use DEFAULT_BEAM_RADIUS_M, not some other hidden number."""
    x, y, z = _grids()
    implicit = beam_box_from_input({"x": x, "y": y}, z, n_radii=2.0, beam_at_origin=True)
    explicit = beam_box_from_input(
        {"x": x, "y": y}, z, n_radii=2.0, beam_at_origin=True,
        beam_radius=DEFAULT_BEAM_RADIUS_M / L,
    )
    assert np.array_equal(implicit, explicit)


def test_supplied_radius_still_wins_over_the_default():
    x, y, z = _grids()
    wide = beam_box_from_input({"x": x, "y": y}, z, n_radii=2.0,
                               beam_at_origin=True, beam_radius=2.0 * SIGMA / L)
    default = beam_box_from_input({"x": x, "y": y}, z, n_radii=2.0, beam_at_origin=True)
    assert wide.sum() > default.sum(), "an explicit radius was ignored"


def test_lab_frame_is_unchanged_and_still_refuses():
    """Only a beam-following frame knows where the beam is. Lab frame must not guess."""
    x, y, z = _grids()
    assert beam_box_from_input({"x": x, "y": y}, z, n_radii=2.0) is None


def test_lazy_metrics_emits_beambox_without_beam_dist(monkeypatch):
    """The lazy path must route through the shared builder, not its own copy.

    This is the regression that matters: the previous code was gated on
    `"beam_dist" in invar`, so a steady eval silently produced no beambox_* keys.
    """
    import lpno.models.pipeline.postprocessing as pp
    import inspect

    src = inspect.getsource(pp.run_lazy_dataset_metrics)
    assert "beam_box_from_input" in src, (
        "run_lazy_dataset_metrics must use the shared beam_box_from_input builder; "
        "a private beam_box_mask call here is the fourth copy that caused this bug"
    )
    assert 'and "beam_dist" in invar' not in src, (
        "the beam box must no longer be gated on the beam_dist channel being present"
    )
