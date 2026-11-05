"""A 2D middle-plane run has no y extent, so the beam box is skipped, not fatal.

Synthesising the footprint (the path taken when `beam_dist` is absent but the beam
CENTRE is known -- a beam-following frame, or beam_x/beam_y) raised without both
grids. Right for a 3D run, where a missing grid IS a misconfiguration and this
metric has a documented history of vanishing silently. Wrong for a 2D run, where y
was sliced away on purpose and a beam BOX is not a thing that exists.

Regressed the `use_2D` eval path from 2026-08-20; CI's GPU smoke died with
`ValueError: ... needs both 'x' and 'y' grids` after the training half succeeded.

Skipped, not silenced: the 2D case says why, once.
"""
import numpy as np
import pytest

from lpno.models.pipeline.postprocessing import (
    _synthetic_beam_footprint,
    beam_box_from_input,
)


def test_2d_synthesis_reports_not_computable_instead_of_raising(capsys):
    beam, sigma = _synthetic_beam_footprint(
        {"x": np.linspace(-4e-4, 4e-4, 12)}, False, 5e-5, None)
    assert beam is None, "a 2D run must report 'not computable', not a footprint"
    assert "2D" in capsys.readouterr().out, "the skip must be announced, not silent"


def test_missing_x_still_raises():
    """The loud case is preserved: no x at all is a misconfiguration, not a 2D run."""
    with pytest.raises(ValueError, match="needs both 'x' and 'y' grids"):
        _synthetic_beam_footprint(
            {"y": np.linspace(-2e-4, 2e-4, 6)}, False, 5e-5, None)


def test_a_full_grid_still_synthesises():
    beam, sigma = _synthetic_beam_footprint(
        {"x": np.linspace(-4e-4, 4e-4, 12), "y": np.linspace(-2e-4, 2e-4, 12)},
        False, 5e-5, None)
    assert beam is not None, "a real 3D grid must still synthesise a footprint"
    assert np.isfinite(beam).any() and float(np.nanmax(beam)) > 0.0


def test_gas_crop_does_not_change_the_beam_box():
    """A gas-side z-crop leaves the box identical: it already excludes the gas.

    `beam_box_mask` bounds the box to the substrate (`mask & (z <= 1e-9)`), so the
    gas cells a crop removes were never in it. Cropping the gas therefore cannot
    make the box non-computable -- measured here as byte-identical masks, 2135 cells
    either way, with zero gas cells inside both.

    This is the distinction the 2D case above turns on, and it is easy to conflate:
    the box needs an x-y DISC, so it dies without **y** (2D middle plane); it does
    not care how much **z** it is given, because it clips z itself.
    """
    nx = ny = 21
    x = np.linspace(-2e-4, 2e-4, nx).reshape(nx, 1, 1)
    y = np.linspace(-2e-4, 2e-4, ny).reshape(1, ny, 1)
    full_z = np.linspace(-3e-4, 1e-4, 13).reshape(1, 1, 13)
    cropped_z = full_z[..., full_z.ravel() <= 0.0]          # what crop_z_top removes

    def _box(z):
        return beam_box_from_input({"x": x, "y": y, "beam_radius": 1e-4}, z,
                                   n_radii=2.0, beam_at_origin=True)

    full, cropped = _box(full_z), _box(cropped_z)
    assert full is not None and cropped is not None, "the gas crop must not kill the box"
    assert int(np.asarray(full).sum()) == int(np.asarray(cropped).sum()), (
        "cropping the gas changed the box, so the box was counting gas cells")
    for box, z in ((full, full_z), (cropped, cropped_z)):
        b = np.asarray(box)
        assert int((b & (np.broadcast_to(z, b.shape) > 0)).sum()) == 0, "gas inside the box"
