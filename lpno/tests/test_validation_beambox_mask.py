"""The training-time beam box must match the eval-path definition.

Bug found 2026-08-10 by adversarial review: the validator used
`beam.abs() <= n_radii` while `beam_dist` is a peak-1 GAUSSIAN FOOTPRINT, not a
distance in beam radii. Every voxel satisfies `<= 2`, so `beambox_*` during
training was the substrate under a beam-box name (val beambox_mae_T 0.00478 sat
on eval substrate_abs_err_mean 0.00439, 8x below the true eval beam-box 0.0402).

Second instance, found 2026-08-15 the same way and by the same mechanism: the
radial predicate was fixed but the eval's DEPTH bound (`z >= -n_radii*sigma`)
was not carried over, so the validator box stayed a full-depth column of mostly
quiescent metal. Measured on `rigno_e8_varspeed_35k` (model 47850): val
beambox_mae_T 0.005155 against the test-split eval's beambox_abs_err_mean
0.049499 -- 9.6x apart, with the val figure sitting right on the eval's
substrate_abs_err_mean 0.006176, the signature of a substrate-wide mean under a
beam-box name.

The 2026-08-10 test below did assert validator-equals-eval, but called the eval
with `sigma=None`, which switches the depth bound OFF on the eval side too. It
pinned the degraded case and passed throughout. A test that pins shared
semantics has to exercise them in the configuration production actually uses.
"""
import numpy as np
import torch

from lpno.models.pipeline.validation_beambox import _beam_box_from_stash
from lpno.models.pipeline.postprocessing import (
    beam_box_from_input,
    beam_box_mask,
)


def _footprint(n=40, nz=8, sigma=5e-5, extent=3e-4):
    """A peak-1 Gaussian footprint on a small grid, the beam_dist convention."""
    ax = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    beam = np.broadcast_to(
        np.exp(-(X**2 + Y**2) / (2 * sigma**2))[:, :, None], (n, n, nz)).copy()
    z = np.broadcast_to(
        np.linspace(-2e-4, 0.0, nz)[None, None, :], (n, n, nz)).copy()
    return beam, z


def _with_x(n=40, nz=8, sigma=5e-5, extent=3e-4):
    """Same grid plus the `x` channel, which is what makes sigma recoverable.

    `beam_sigma_from_footprint` reads the one-sigma contour off `beam_dist`
    against `x`, so without `x` the eval gets sigma=None and drops the depth
    bound -- the exact configuration the old test froze.
    """
    beam, z = _footprint(n=n, nz=nz, sigma=sigma, extent=extent)
    ax = np.linspace(-extent, extent, n)
    x = np.broadcast_to(ax[:, None, None], (n, n, nz)).copy()
    return beam, z, x


def test_validator_box_is_not_the_whole_substrate():
    beam, z = _footprint()
    box = _beam_box_from_stash(torch.tensor(beam), torch.tensor(z), n_radii=2.0)
    frac = float(box.float().mean())
    assert 0.0 < frac < 0.25, f"box covers {frac:.1%} of the grid; it must be a small disc"


def test_validator_box_matches_the_eval_definition():
    beam, z = _footprint()
    ours = _beam_box_from_stash(torch.tensor(beam), torch.tensor(z), n_radii=2.0).numpy()
    theirs = beam_box_mask(beam, z, n_radii=2.0, sigma=None)
    assert theirs is not None
    assert np.array_equal(ours, theirs), (
        f"validator selects {ours.sum()} cells, eval selects {theirs.sum()}"
    )


def test_validator_applies_the_depth_bound_when_x_is_available():
    """With `x` in hand the validator must bound depth exactly as the eval does.

    This is the 2026-08-15 regression: without the bound the box runs the full
    depth of the substrate and dilutes the metric back toward substrate-wide.
    """
    beam, z, x = _with_x()
    ours = _beam_box_from_stash(
        torch.tensor(beam), torch.tensor(z), x=torch.tensor(x), n_radii=2.0
    ).numpy()
    theirs = beam_box_from_input({"beam_dist": beam, "x": x}, z, n_radii=2.0)
    assert theirs is not None
    assert np.array_equal(ours, theirs), (
        f"validator selects {ours.sum()} cells, eval selects {theirs.sum()}"
    )


def test_depth_bound_actually_shrinks_the_box():
    """Guard the guard: if the fixture never triggers the bound, the test above
    would pass on a validator that ignores `x` entirely."""
    beam, z, x = _with_x()
    unbounded = beam_box_mask(beam, z, n_radii=2.0, sigma=None)
    bounded = beam_box_from_input({"beam_dist": beam, "x": x}, z, n_radii=2.0)
    assert bounded.sum() < unbounded.sum(), (
        "fixture does not exercise the depth bound, so the equality test above is vacuous"
    )


def test_uniform_footprint_threshold_is_carried_over():
    """A top-hat case must use the uniform inversion, not the Gaussian one.

    `beam_box_threshold` solves the uniform profile to r = 1.232 R rather than
    2 R; using exp(-n^2/2) on a top-hat scored 71 of the 230 PV-LHS cases over a
    2.6x smaller, hotter region under the same metric name.
    """
    beam, z, x = _with_x()
    shape = np.ones_like(beam)          # beam_shape_uniform = 1 -> top hat
    ours = _beam_box_from_stash(
        torch.tensor(beam), torch.tensor(z), x=torch.tensor(x),
        shape=torch.tensor(shape), n_radii=2.0,
    ).numpy()
    theirs = beam_box_from_input(
        {"beam_dist": beam, "x": x, "beam_shape_uniform": shape}, z, n_radii=2.0
    )
    assert theirs is not None
    assert np.array_equal(ours, theirs)
    gaussian = beam_box_from_input({"beam_dist": beam, "x": x}, z, n_radii=2.0)
    assert ours.sum() != gaussian.sum(), "uniform and Gaussian thresholds coincide in this fixture"
