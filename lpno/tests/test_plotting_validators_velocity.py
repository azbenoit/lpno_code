import sys
import types

import matplotlib
import numpy as np


matplotlib.use("Agg")

def _install_plotter_stub() -> None:
    if "physicsnemo.sym.utils.io.plotter" in sys.modules:
        return

    physicsnemo = types.ModuleType("physicsnemo")
    sym = types.ModuleType("physicsnemo.sym")
    utils = types.ModuleType("physicsnemo.sym.utils")
    io = types.ModuleType("physicsnemo.sym.utils.io")
    plotter = types.ModuleType("physicsnemo.sym.utils.io.plotter")

    class GridValidatorPlotter:
        def __init__(self, n_examples=1):
            self.n_examples = n_examples

    plotter.GridValidatorPlotter = GridValidatorPlotter
    io.plotter = plotter
    utils.io = io
    sym.utils = utils
    physicsnemo.sym = sym

    sys.modules["physicsnemo"] = physicsnemo
    sys.modules["physicsnemo.sym"] = sym
    sys.modules["physicsnemo.sym.utils"] = utils
    sys.modules["physicsnemo.sym.utils.io"] = io
    sys.modules["physicsnemo.sym.utils.io.plotter"] = plotter


_install_plotter_stub()

from lpno.plotting.validators import CustomSlicePlotter, TemporalPatchPlotter  # noqa: E402


def _base_invar(batch=1, nx=3, ny=3, nz=3):
    return {
        "P": np.zeros((batch, 1, nx, ny, nz), dtype=np.float32),
        "V_scan": np.zeros((batch, 1, nx, ny, nz), dtype=np.float32),
    }


def _constant_field(value, batch=1, nx=3, ny=3, nz=3):
    arr = np.full((batch, 1, nx, ny, nz), value, dtype=np.float32)
    return arr


def _find_main_axis(fig, title_substring):
    for ax in fig.axes:
        if title_substring in ax.get_title():
            return ax
    raise AssertionError(f"Axis with title containing '{title_substring}' not found")


def test_velocity_triplet_masks_and_scalar_suppression():
    batch, nx, ny, nz = 1, 3, 3, 3
    invar = _base_invar(batch=batch, nx=nx, ny=ny, nz=nz)

    true_outvar = {
        "T": _constant_field(2.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "u": _constant_field(1.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "v": _constant_field(1.5, batch=batch, nx=nx, ny=ny, nz=nz),
        "w": _constant_field(2.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "fl": _constant_field(0.1, batch=batch, nx=nx, ny=ny, nz=nz),
    }
    pred_outvar = {
        "T": _constant_field(1.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "u": _constant_field(3.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "v": _constant_field(3.5, batch=batch, nx=nx, ny=ny, nz=nz),
        "w": _constant_field(4.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "fl": _constant_field(0.1, batch=batch, nx=nx, ny=ny, nz=nz),
    }

    # XZ (y=1) true mask: 2 points
    true_mask_xz = np.array(
        [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
        dtype=np.float32,
    )
    # XZ (y=1) pred mask: 2 points, one overlaps with true mask
    pred_mask_xz = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        dtype=np.float32,
    )
    true_outvar["fl"][0, 0, :, 1, :] = np.where(true_mask_xz > 0, 0.4, 0.1)
    pred_outvar["fl"][0, 0, :, 1, :] = np.where(pred_mask_xz > 0, 0.4, 0.1)

    plotter = CustomSlicePlotter(
        xy_idx=1,
        yz_idx=1,
        xz_idx=1,
        variables=["T", "u", "v", "w"],
        n_examples=1,
        debug=False,
    )
    figs = plotter(invar, true_outvar, pred_outvar)
    names = [name for _, name in figs]

    # Scalar T remains.
    t_names = [n for n in names if n.startswith("T_")]
    assert len(t_names) == 3

    # Per-component scalar plots are suppressed in combined velocity mode.
    assert not any(n.startswith("u_") for n in names)
    assert not any(n.startswith("v_") for n in names)
    assert not any(n.startswith("w_") for n in names)

    # Velocity triplet yields 3 plane figures.
    velocity_names = [n for n in names if n.startswith("velocity_")]
    assert len(velocity_names) == 3

    xz_name = next(n for n in velocity_names if n.startswith("velocity_xz_y1_"))
    fig_by_name = {name: fig for fig, name in figs}
    xz_fig = fig_by_name[xz_name]

    ax_true = _find_main_axis(xz_fig, "(true)")
    ax_pred = _find_main_axis(xz_fig, "(pred)")
    ax_diff = _find_main_axis(xz_fig, "error")

    # Quiver object is the first collection in each populated axis.
    q_true = ax_true.collections[0]
    q_pred = ax_pred.collections[0]
    q_diff = ax_diff.collections[0]

    assert len(q_true.U) == int(np.sum(true_mask_xz))
    assert len(q_pred.U) == int(np.sum(pred_mask_xz))
    assert len(q_diff.U) == int(np.sum((true_mask_xz > 0) & (pred_mask_xz > 0)))


def test_velocity_empty_masks_render_message():
    batch, nx, ny, nz = 1, 3, 3, 3
    invar = _base_invar(batch=batch, nx=nx, ny=ny, nz=nz)
    true_outvar = {
        "u": _constant_field(1.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "v": _constant_field(1.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "w": _constant_field(1.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "fl": _constant_field(0.1, batch=batch, nx=nx, ny=ny, nz=nz),
    }
    pred_outvar = {
        "u": _constant_field(1.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "v": _constant_field(1.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "w": _constant_field(1.0, batch=batch, nx=nx, ny=ny, nz=nz),
        "fl": _constant_field(0.1, batch=batch, nx=nx, ny=ny, nz=nz),
    }

    plotter = CustomSlicePlotter(
        xy_idx=1,
        yz_idx=1,
        xz_idx=1,
        variables=["u", "v", "w"],
        n_examples=1,
        debug=False,
    )
    figs = plotter(invar, true_outvar, pred_outvar)
    velocity_figs = [fig for fig, name in figs if name.startswith("velocity_")]
    assert len(velocity_figs) == 3

    for fig in velocity_figs:
        for token in ("(true)", "(pred)", "error"):
            ax = _find_main_axis(fig, token)
            assert len(ax.collections) == 0
            assert any("No points satisfy" in text.get_text() for text in ax.texts)


def test_temporal_patch_plotter_renders_context_prediction_and_error_panels():
    batch, nx, ny, nz = 1, 4, 4, 4
    invar = {
        "T_prev": _constant_field(0.25, batch=batch, nx=nx, ny=ny, nz=nz),
        "T_prev_mask": _constant_field(0.0, batch=batch, nx=nx, ny=ny, nz=nz),
    }
    invar["T_prev_mask"][0, 0, 1:3, 1:3, 1:3] = 1.0
    true_outvar = {"T": _constant_field(0.5, batch=batch, nx=nx, ny=ny, nz=nz)}
    pred_outvar = {"T": _constant_field(0.4, batch=batch, nx=nx, ny=ny, nz=nz)}

    plotter = TemporalPatchPlotter(variables=["T"], state_prev_suffix="_prev", n_examples=1)
    figs = plotter(invar, true_outvar, pred_outvar)

    assert len(figs) == 1
    fig, name = figs[0]
    assert name == "temporal_patch_T_b0"
    titles = [ax.get_title() for ax in fig.axes]
    assert any("prev" in title.lower() for title in titles)
    assert any("mask" in title.lower() for title in titles)
    assert any("true" in title.lower() for title in titles)
    assert any("pred" in title.lower() for title in titles)
    assert any("error" in title.lower() for title in titles)
