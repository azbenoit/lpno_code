"""Regression: the field contour was drawn on the relative-error panel too.

The difference column is `D = (T - P) / mean(|T|)` on a symmetric "bwr" scale, so
`contour_val` (0.5 for alpha, T_s/ref for T) usually falls inside those limits.
The guard passed and `ax.contour` traced an isoline of the *error* field, which
the legend then labelled "T = 0.62" -- a reader takes that for the melt boundary.

The contour belongs on the true and pred panels only.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from test_plotting_validators_velocity import _base_invar, _find_main_axis  # noqa: E402

from lpno.plotting.validators import CustomSlicePlotter  # noqa: E402

CONTOUR_VAL = 0.5


def _ramp_field(batch=1, nx=5, ny=5, nz=5):
    """A field spanning 0..1 so CONTOUR_VAL sits inside its range on every panel."""
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    return np.broadcast_to(x.reshape(1, 1, nx, 1, 1), (batch, 1, nx, ny, nz)).copy()


def _figures():
    invar = _base_invar(nx=5, ny=5, nz=5)
    true_outvar = {"alpha": _ramp_field()}
    # Predictions offset so the error panel is nonzero and spans CONTOUR_VAL too.
    pred_outvar = {"alpha": _ramp_field() * 0.2}

    plotter = CustomSlicePlotter(
        xy_idx=2, yz_idx=2, xz_idx=2,
        variables=["alpha"],
        n_examples=1,
        debug=False,
        contours={"alpha": CONTOUR_VAL},
    )
    return plotter(invar, true_outvar, pred_outvar)


def _has_contour(ax) -> bool:
    # A legend is added only alongside the contour, and is the cleanest signal
    # that survives matplotlib's varying contour container types.
    return ax.get_legend() is not None


def test_error_panel_has_no_field_contour():
    for fig, _ in _figures():
        ax_err = _find_main_axis(fig, "rel error")
        assert not _has_contour(ax_err), "field contour drawn on the relative-error panel"


def test_true_and_pred_panels_still_get_the_contour():
    # The guard must not have thrown the baby out: the contour is the point of
    # the feature on the panels where it means something.
    drawn = 0
    for fig, _ in _figures():
        for ax in fig.axes:
            title = ax.get_title()
            if "rel error" in title or not title:
                continue
            if _has_contour(ax):
                drawn += 1
    assert drawn > 0, "contour disappeared from the true/pred panels as well"
