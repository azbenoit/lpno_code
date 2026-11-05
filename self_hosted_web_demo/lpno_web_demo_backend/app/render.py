from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


class RenderError(RuntimeError):
    pass


@dataclass(slots=True)
class RenderedPlot:
    png_bytes: bytes
    y_index_used: int
    grid_shape_xyz: tuple[int, int, int]
    width_px: int
    height_px: int


def _import_pyvista() -> Any:
    return importlib.import_module("pyvista")


def _extract_volume(preds: dict[str, np.ndarray], field: str) -> tuple[np.ndarray, tuple[int, int, int]]:
    if field not in preds:
        raise RenderError(f"Requested field '{field}' not present in predictions: {sorted(preds.keys())}")

    arr = np.asarray(preds[field])
    if arr.ndim != 5:
        raise RenderError(f"Expected 5D array (B,C,X,Y,Z) for field '{field}', got shape {arr.shape}")

    vol = arr[0, 0]
    if vol.ndim != 3:
        raise RenderError(f"Expected 3D volume after selecting [0,0], got shape {vol.shape}")
    nx, ny, nz = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))
    return vol.astype(np.float32, copy=False), (nx, ny, nz)


def _compose_3d_figure_with_colorbar(
    *,
    img: np.ndarray,
    color_field: str,
    clim: list[float] | None,
    width_px: int,
    height_px: int,
    title: str | None = None,
) -> bytes:
    dpi = 100
    fig = plt.figure(figsize=(max(width_px, 1) / dpi, max(height_px, 1) / dpi), dpi=dpi)
    # Leave headroom for the title so ~12 pt text is never clipped at dpi 100.
    ax_img = fig.add_axes([0.03, 0.02, 0.84, 0.86] if title else [0.03, 0.04, 0.84, 0.92])
    ax_img.imshow(img)
    ax_img.axis("off")
    if title:
        ax_img.set_title(title, fontsize=12, pad=8)

    if clim is not None:
        cax = fig.add_axes([0.90, 0.19, 0.026, 0.62])
        cmap = plt.get_cmap("inferno")
        norm = plt.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        unit = "[K]" if color_field == "T" else ""
        title = f"{color_field} {unit}".strip()
        cbar.ax.set_title(title, fontsize=11, pad=10)
        cbar.ax.tick_params(labelsize=9)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def _field_label(field: str) -> str:
    """Colorbar/panel label with unit, matching the 2D convention (T [K], alpha, fl)."""
    unit = {"T": "K", "alpha": "-", "fl": "-"}.get(field, "")
    return f"{field} [{unit}]" if unit else field


def _compose_3d_gt_pred_figure(
    *,
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    clim: list[float],
    width_px: int,
    time_label: str,
    gt_split: str | None = None,
    field_label: str = "T [K]",
) -> bytes:
    """Portrait stack of two same-camera PyVista screenshots — ground truth on
    top, prediction below (the 2D triple-panel convention, without the diff) —
    with one shared colorbar spanning both and the time label as the overall
    title. Height derives from the screenshot aspect; object API (no pyplot)
    so it stays thread-safe."""
    dpi = 100
    fig_w_in = max(width_px, 1) / dpi
    img_w_frac = 0.84  # same image box width as the single-panel composition
    ax_w_in = img_w_frac * fig_w_in
    sup_in, title_in, bottom_in = 0.40, 0.34, 0.06
    gt_tag = f" ({gt_split})" if gt_split else ""
    panels = [(f"Ground truth{gt_tag}  {field_label}", gt_img), (f"Prediction  {field_label}", pred_img)]
    ax_h_ins = [ax_w_in * (img.shape[0] / max(img.shape[1], 1)) for _, img in panels]
    fig_h_in = sup_in + sum(title_in + h for h in ax_h_ins) + bottom_in

    fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.suptitle(time_label, fontsize=12, y=1.0 - 0.45 * sup_in / fig_h_in)

    y_top_in = fig_h_in - sup_in
    y0_ins: list[float] = []
    for (panel_title, img), ax_h_in in zip(panels, ax_h_ins):
        y0_in = y_top_in - title_in - ax_h_in
        ax = fig.add_axes([0.03, y0_in / fig_h_in, img_w_frac, ax_h_in / fig_h_in])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(panel_title, fontsize=12, pad=6)
        y0_ins.append(y0_in)
        y_top_in = y0_in

    # One colorbar spanning both panels, inset like the single-panel layout.
    cb_lo_in = y0_ins[1] + 0.10 * ax_h_ins[1]
    cb_hi_in = y0_ins[0] + 0.90 * ax_h_ins[0]
    cax = fig.add_axes([0.90, cb_lo_in / fig_h_in, 0.026, (cb_hi_in - cb_lo_in) / fig_h_in])
    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=float(clim[0]), vmax=float(clim[1])),
        cmap=plt.get_cmap("inferno"),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.ax.set_title(field_label, fontsize=11, pad=10)
    cbar.ax.tick_params(labelsize=9)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pil_kwargs={"compress_level": 3})
    return buf.getvalue()


def _zoom_image(img: np.ndarray, *, crop_fraction: float = 0.10) -> np.ndarray:
    if img.ndim < 2:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    crop_y = max(0, int(round(h * float(crop_fraction))))
    crop_x = max(0, int(round(w * float(crop_fraction))))
    if crop_y * 2 >= h or crop_x * 2 >= w:
        return img
    return img[crop_y : h - crop_y, crop_x : w - crop_x, ...]


def _top_metal_projection(vol: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """(nx, ny) map of ``vol`` at the topmost metal cell (alpha >= 0.5, highest z)
    in each column; NaN where a column has no metal. z increases toward the free
    surface, so the topmost metal is the first metal cell scanning from the top."""
    metal = np.asarray(alpha) >= 0.5  # (nx, ny, nz)
    nx, ny, nz = metal.shape
    has = metal.any(axis=2)
    top_off = np.argmax(metal[:, :, ::-1], axis=2)  # offset of first metal from the surface
    zidx = np.clip(nz - 1 - top_off, 0, nz - 1)
    xi, yi = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    top = np.asarray(vol)[xi, yi, zidx].astype(np.float32)
    return np.where(has, top, np.nan)


# Melt-pool edge contour level (K): Ti-64 solidus/liquidus midpoint (1873/1923).
MELT_CONTOUR_K = 1898.0
_MELT_CONTOUR_COLOR = "orange"

# 2D display upsampling: spline-zoom the slice, then imshow with interpolation
# "nearest". This keeps the
# keyhole edge crisp instead of the mushy bilinear blur on the native grid.
_DISPLAY_ZOOM = 3


def _zoom_slice(arr2d: np.ndarray, factor: int = _DISPLAY_ZOOM, order: int = 1) -> np.ndarray:
    """Upsample a 2D slice by ``factor`` with a spline of ``order`` (1 = linear),
    for a crisp nearest-interpolation display. Same physical extent, more pixels."""
    if factor <= 1:
        return np.asarray(arr2d)
    from scipy.ndimage import zoom

    return zoom(np.asarray(arr2d, dtype=np.float32), factor, order=order)


def _draw_melt_contour(ax, t_2d: np.ndarray, extent, *, level: float = MELT_CONTOUR_K, add_label: bool = True):
    """Draw the melt-pool edge (``T = level``) on a matplotlib axis from the real
    raw served T slice ``t_2d`` (shape (vertical, x), origin lower), over
    ``extent`` = [xmin, xmax, vmin, vmax]. Returns the contour set so callers that
    reuse a figure (per-frame) can remove it. ``add_label`` adds a one-off legend
    entry "meltpool edge"; pass False when the legend is set once elsewhere."""
    t_2d = np.asarray(t_2d)
    xg = np.linspace(extent[0], extent[1], t_2d.shape[1])
    vg = np.linspace(extent[2], extent[3], t_2d.shape[0])
    cs = ax.contour(*np.meshgrid(xg, vg), t_2d, levels=[float(level)], colors=_MELT_CONTOUR_COLOR, linewidths=1.4)
    if add_label and len(cs.levels) > 0:
        ax.plot([], [], color=_MELT_CONTOUR_COLOR, linewidth=1.4, label="meltpool edge")
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    return cs


def render_xz_cross_section_png(
    *,
    preds: dict[str, np.ndarray],
    field: str,
    y_index: int,
    mesh_size_um: float,
    show_t_contour: bool,
    t_contour_k: float,
    width_px: int,
    height_px: int,
    plane: str = "xz",
) -> RenderedPlot:
    """Steady 2D slice. ``plane="xz"`` is the cross-section at a y-index (with an
    optional T contour); ``plane="xy"`` is the top view at the surface z-plane."""
    vol, (nx, ny, nz) = _extract_volume(preds, field)
    alpha_vol = _extract_volume(preds, "alpha")[0] if "alpha" in preds else None
    yi = ny // 2 if int(y_index) < 0 else max(0, min(int(y_index), ny - 1))

    if plane == "xy":
        # Melt-pool top view: the topmost metal cell (alpha >= 0.5) in each column,
        # not the raw free surface (which is vapor at the keyhole and saturates the
        # colour). Columns with no metal render as background.
        if alpha_vol is not None:
            slice_2d = _top_metal_projection(vol, alpha_vol).T  # (ny, nx)
        else:
            slice_2d = vol[:, :, nz - 1].T
        extent = [0.0, float(nx) * float(mesh_size_um), 0.0, float(ny) * float(mesh_size_um)]
        title = f"{field} XY top view (melt surface)"
        xlabel, ylabel = "x (µm)", "y (µm)"
    else:
        slice_2d = vol[:, yi, :].T  # (nz, nx)
        extent = [0.0, float(nx) * float(mesh_size_um), 0.0, float(nz) * float(mesh_size_um)]
        title = f"{field} XZ cross-section @ y-index={yi}"
        xlabel, ylabel = "x (µm)", "z (µm)"

    dpi = 100
    fig = plt.figure(figsize=(max(width_px, 1) / dpi, max(height_px, 1) / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    vmin = None
    vmax = None
    if field == "T":
        vmin, vmax = 300.0, 3200.0
    elif field in ("alpha", "fl"):
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(
        _zoom_slice(slice_2d),
        origin="lower",
        cmap="inferno",
        aspect="equal",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    unit_map = {"T": "K", "alpha": "-", "fl": "-", "u": "m/s", "v": "m/s", "w": "m/s"}
    unit = unit_map.get(field, "")
    cbar.set_label(f"{field} [{unit}]" if unit else field)

    # Melt-pool edge contour on the raw served T, aligned to the shown
    # slice: the melt-surface projection for the top view, the y-slice for the xz
    # cross-section. Same T-boundary the other 2D pages draw.
    if show_t_contour and "T" in preds and np.asarray(preds["T"]).ndim == 5:
        t_disp = np.asarray(preds["T"])[0, 0]
        alpha_disp = np.asarray(preds["alpha"])[0, 0] if "alpha" in preds else None
        if plane == "xy":
            t_slice = _top_metal_projection(t_disp, alpha_disp).T if alpha_disp is not None else t_disp[:, :, nz - 1].T
        else:
            t_slice = t_disp[:, yi, :].T
        _draw_melt_contour(ax, _zoom_slice(t_slice), extent, level=t_contour_k)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return RenderedPlot(
        png_bytes=buf.getvalue(),
        y_index_used=yi,
        grid_shape_xyz=(nx, ny, nz),
        width_px=width_px,
        height_px=height_px,
    )


def _steady_clim(field: str) -> tuple[float, float]:
    return (300.0, 3200.0) if field == "T" else (0.0, 1.0)


def _xz_slice(vol: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray, y_index: int) -> tuple[np.ndarray, list[float]]:
    """XZ cross-section at a y-index, plus its physical extent in mm."""
    nx, ny, nz = vol.shape
    yi = ny // 2 if int(y_index) < 0 else max(0, min(int(y_index), ny - 1))
    slice_xz = vol[:, yi, :].T  # (nz, nx)
    extent = [
        float(x_axis[0]) * 1e3,
        float(x_axis[-1]) * 1e3,
        float(z_axis[0]) * 1e3,
        float(z_axis[-1]) * 1e3,
    ]
    return slice_xz, extent


def render_steady_gt_compare_png(
    *,
    pred: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    pred_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    gt_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    field: str,
    y_index: int,
    width_px: int,
    height_px: int,
    gt_split: str | None = None,
) -> RenderedPlot:
    """Ground-truth vs prediction XZ cross-sections for the steady-state model,
    stacked (GT above prediction) with one shared colorbar. Each panel is drawn
    on its own physical axes (the FLOW-3D GT grid and the served prediction grid
    differ in x), so both show true physical millimetre extents. Object API (no
    pyplot) for thread safety.
    """
    pvol, (pnx, pny, pnz) = _extract_volume(pred, field)
    gvol = np.asarray(gt[field], dtype=np.float32)
    if gvol.ndim != 3:
        raise RenderError(f"GT field '{field}' must be a 3D volume, got shape {gvol.shape}")

    px, py, pz = pred_axes
    gx, gy, gz = gt_axes
    pred_slice, pred_extent = _xz_slice(pvol, px, pz, y_index)
    gt_slice, gt_extent = _xz_slice(gvol, gx, gz, y_index)
    # T slices for the melt-pool edge contour on each panel.
    t_pred_slice, _ = _xz_slice(_extract_volume(pred, "T")[0], px, pz, y_index)
    t_gt_slice, _ = _xz_slice(np.asarray(gt["T"], dtype=np.float32), gx, gz, y_index)
    clim = _steady_clim(field)
    unit = "[K]" if field == "T" else "[-]"
    gt_title = f"Ground truth ({gt_split})" if gt_split else "Ground truth"

    dpi = 100
    fig_w_in = max(width_px, 1) / dpi
    plot_w_frac = 0.80
    left_in = 0.72
    bottom_in, title_in, sup_in = 0.55, 0.34, 0.30
    plot_w_in = plot_w_frac * fig_w_in - left_in
    panels = [
        (gt_title, gt_slice, gt_extent, t_gt_slice),
        ("Prediction", pred_slice, pred_extent, t_pred_slice),
    ]
    # Panel heights follow each slice's physical z/x aspect so nothing distorts.
    ax_h_ins = [
        plot_w_in * (abs(ext[3] - ext[2]) / max(abs(ext[1] - ext[0]), 1e-9)) for _t, _s, ext, _tc in panels
    ]
    fig_h_in = sup_in + sum(title_in + bottom_in + h for h in ax_h_ins)

    fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.suptitle(f"{field} {unit}", fontsize=12, y=1.0 - 0.4 * sup_in / fig_h_in)

    ims = []
    y_top_in = fig_h_in - sup_in
    for (panel_title, slice_xz, extent, t_slice), ax_h_in in zip(panels, ax_h_ins):
        y0_in = y_top_in - title_in - ax_h_in
        ax = fig.add_axes([left_in / fig_w_in, y0_in / fig_h_in, plot_w_in / fig_w_in, ax_h_in / fig_h_in])
        im = ax.imshow(
            _zoom_slice(slice_xz),
            origin="lower",
            cmap="inferno",
            aspect="auto",
            extent=extent,
            vmin=clim[0],
            vmax=clim[1],
            interpolation="nearest",
        )
        ax.set_title(panel_title, fontsize=11, pad=4)
        ax.set_xlabel("x [mm]", fontsize=10)
        ax.set_ylabel("z [mm]", fontsize=10)
        ax.tick_params(labelsize=9)
        _draw_melt_contour(ax, _zoom_slice(t_slice), extent, add_label=(panel_title == "Prediction"))
        ims.append(im)
        y_top_in = y0_in - bottom_in

    cb_lo_in = (y_top_in + bottom_in) + 0.05 * ax_h_ins[-1]
    cb_hi_in = (fig_h_in - sup_in - title_in) - 0.05 * ax_h_ins[0]
    cax = fig.add_axes([left_in / fig_w_in + plot_w_in / fig_w_in + 0.02, cb_lo_in / fig_h_in, 0.022, (cb_hi_in - cb_lo_in) / fig_h_in])
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.ax.set_title(unit, fontsize=10, pad=8)
    cbar.ax.tick_params(labelsize=9)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pil_kwargs={"compress_level": 3})
    return RenderedPlot(
        png_bytes=buf.getvalue(),
        y_index_used=(pny // 2 if int(y_index) < 0 else max(0, min(int(y_index), pny - 1))),
        grid_shape_xyz=(pnx, pny, pnz),
        width_px=width_px,
        height_px=height_px,
    )


def _path_clim(field: str) -> tuple[float, float]:
    return (300.0, 3200.0) if field == "T" else (0.0, 1.0)


# Fixed panel margins in inches for the manual /infer_path layout. The axes box
# is sized to the physical domain aspect, so plots are never distorted. Sized
# for legible ~10-12 pt text at dpi 100 and a ~640 px request width with
# nothing clipped.
_PATH_MARGIN_LEFT_IN = 0.70  # y-label + ticks
_PATH_MARGIN_CBAR_IN = 0.95  # gap + colorbar + its tick labels
_PATH_MARGIN_BOTTOM_IN = 0.58  # x-label + ticks
_PATH_MARGIN_TOP_IN = 0.45  # panel title
_PATH_SUPTITLE_IN = 0.38
_TRIPLE_PLOT_FACTOR = 0.9  # triple panel keeps the request width; plots ~0.9x
_PATH_FONT = {"title": 12, "label": 11, "tick": 10}


class PathFrameRenderer:
    """Reusable figure for one (field, panel-type) of an /infer_path request.

    The Figure, axes, colorbars, and static path overlay are built once; per
    frame only the image data, diff clim, beam marker, and time label are
    updated before re-encoding. This skips ~all of the per-frame matplotlib
    layout cost of building 60 figures per request from scratch.

    Single panel for drawn paths; triple portrait stack (GT / prediction /
    diff) when ``has_gt``. Panels keep the true physical domain aspect; the
    figure height derives from ``width_px``, the aspect, and fixed margins.
    Object API only (no pyplot), so rendering stays thread-safe. Follows the
    submodule random_path_inference notebook show() conventions (arr.T, origin
    lower, extent in mm, inferno, cyan path, lime beam marker).
    """

    def __init__(
        self,
        *,
        field: str,
        has_gt: bool,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        path_xy: np.ndarray | None,
        width_px: int,
        diff_dmax: float = 1.0,
        gt_split: str | None = None,
        plane: str = "xy",
        contour_k: float | None = None,
        dpi: int = 100,
        beam_radius_m: float = 5.0e-5,  # generator beam_radius_m (= Gaussian sigma)
    ) -> None:
        self._unit = "[K]" if field == "T" else "[-]"
        self._field = field
        self._has_gt = has_gt
        self._plane = plane
        self._contour_k = contour_k
        self._beam_radius_mm = float(beam_radius_m) * 1e3
        self._axes: list = []
        self._contours: list = []
        extent_mm = [
            float(x_axis.min()) * 1e3,
            float(x_axis.max()) * 1e3,
            float(y_axis.min()) * 1e3,
            float(y_axis.max()) * 1e3,
        ]
        self._extent_mm = extent_mm  # for the per-frame melt-pool contour
        domain_aspect = (extent_mm[3] - extent_mm[2]) / max(extent_mm[1] - extent_mm[0], 1e-12)
        clim = _path_clim(field)
        n_panels = 3 if has_gt else 1

        fig_w_in = max(width_px, 1) / dpi
        inner_w_in = fig_w_in * (_TRIPLE_PLOT_FACTOR if has_gt else 1.0)
        x_pad_in = (fig_w_in - inner_w_in) / 2.0
        plot_w_in = max(inner_w_in - _PATH_MARGIN_LEFT_IN - _PATH_MARGIN_CBAR_IN, 0.5)
        plot_h_in = plot_w_in * domain_aspect
        panel_h_in = _PATH_MARGIN_BOTTOM_IN + plot_h_in + _PATH_MARGIN_TOP_IN
        fig_h_in = n_panels * panel_h_in + (_PATH_SUPTITLE_IN if has_gt else 0.0)

        self._fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        FigureCanvasAgg(self._fig)
        self._dpi = dpi

        gt_tag = f" ({gt_split})" if gt_split else ""
        if has_gt:
            titles = [
                f"Ground truth{gt_tag}  {field} {self._unit}",
                f"Prediction  {field} {self._unit}",
                f"Prediction - GT  {self._unit}",
            ]
            self._suptitle = self._fig.suptitle(
                "", fontsize=_PATH_FONT["title"], y=1.0 - 0.45 * _PATH_SUPTITLE_IN / fig_h_in
            )
        else:
            titles = [""]  # set per frame (includes the time label)
            self._suptitle = None

        # Placeholder at the upsampled display shape (frames are zoomed before
        # set_data), so imshow with "nearest" stays crisp instead of blurring.
        placeholder = np.zeros((y_axis.size * _DISPLAY_ZOOM, x_axis.size * _DISPLAY_ZOOM), dtype=np.float32)
        self._ims = []
        self._beams = []
        self._titles = []
        for k in range(n_panels):
            y0_in = (
                fig_h_in
                - (_PATH_SUPTITLE_IN if has_gt else 0.0)
                - (k + 1) * panel_h_in
                + _PATH_MARGIN_BOTTOM_IN
            )
            ax = self._fig.add_axes([
                (x_pad_in + _PATH_MARGIN_LEFT_IN) / fig_w_in,
                y0_in / fig_h_in,
                plot_w_in / fig_w_in,
                plot_h_in / fig_h_in,
            ])
            cax = self._fig.add_axes([
                (x_pad_in + _PATH_MARGIN_LEFT_IN + plot_w_in + 0.14) / fig_w_in,
                y0_in / fig_h_in,
                0.18 / fig_w_in,
                plot_h_in / fig_h_in,
            ])
            is_diff = has_gt and k == 2
            im = ax.imshow(
                placeholder,
                origin="lower",
                extent=extent_mm,
                cmap="coolwarm" if is_diff else "inferno",
                aspect="equal",
                interpolation="nearest",
            )
            # Fixed clim on every panel so the colorbar is constant across the
            # whole time series: field range for GT/prediction, one run-wide
            # symmetric range for the diff (computed over all frames by the
            # caller). Nothing is set per frame in render().
            if is_diff:
                d = max(float(diff_dmax), 1e-9)
                im.set_clim(-d, d)
            else:
                im.set_clim(*clim)
            self._fig.colorbar(im, cax=cax)
            cax.tick_params(labelsize=_PATH_FONT["tick"])
            ax.set_xlabel("x [mm]", fontsize=_PATH_FONT["label"])
            ax.set_ylabel("y [mm]" if plane == "xy" else "z [mm]", fontsize=_PATH_FONT["label"])
            ax.tick_params(labelsize=_PATH_FONT["tick"])
            title = ax.set_title(titles[k], fontsize=_PATH_FONT["title"])
            # The drawn path lives in the XY plane; only overlay it on top views.
            if path_xy is not None and plane == "xy":
                ax.plot(path_xy[:, 0] * 1e3, path_xy[:, 1] * 1e3, "-", color="cyan", lw=0.8, alpha=0.7)
            # Beam marker: a DATA-scaled open circle of one beam radius around the
            # instantaneous laser centre (was a fixed 7-pt dot whose size meant
            # nothing). Hidden until the first render() supplies a centre.
            from matplotlib.patches import Circle

            beam = Circle(
                (0.0, 0.0), radius=self._beam_radius_mm, fill=False,
                edgecolor="lime", linewidth=1.8, visible=False, zorder=5,
            )
            ax.add_patch(beam)
            # One-off legend on the GT + prediction panels (not the diff); the
            # per-frame contour set itself is replaced each frame. Proxy handles so
            # the entries render as compact line/marker samples; semi-transparent
            # frame in the lower-right corner (start region — coolest, least
            # informative part of the field once the weld is underway).
            if contour_k is not None and not (has_gt and k == 2):
                from matplotlib.lines import Line2D

                handles = [
                    Line2D([], [], color=_MELT_CONTOUR_COLOR, linewidth=1.4, label="meltpool edge"),
                    Line2D(
                        [], [], marker="o", color="lime", mfc="none", mew=1.8, ls="",
                        label=f"beam radius ({self._beam_radius_mm * 1e3:.0f} µm)",
                    ),
                ]
                ax.legend(handles=handles, loc="lower right", fontsize=7,
                          frameon=True, framealpha=0.75, borderpad=0.4,
                          handletextpad=0.5, labelspacing=0.3)
            self._ims.append(im)
            self._beams.append(beam)
            self._titles.append(title)
            self._axes.append(ax)

    def render(
        self,
        *,
        pred_xy: np.ndarray,
        gt_xy: np.ndarray | None,
        centre_xy: np.ndarray | None,
        time_s: float,
        t_pred_xy: np.ndarray | None = None,
        t_gt_xy: np.ndarray | None = None,
    ) -> bytes:
        time_label = f"t = {time_s * 1e3:.2f} ms"
        if self._has_gt:
            if gt_xy is None:
                raise RenderError("Renderer built with has_gt=True but no gt_xy given")
            # clim is fixed once in __init__ (run-wide); only the data changes.
            # Upsample each slice so imshow "nearest" stays crisp (keyhole edge).
            self._ims[0].set_data(_zoom_slice(gt_xy.T))
            self._ims[1].set_data(_zoom_slice(pred_xy.T))
            self._ims[2].set_data(_zoom_slice((pred_xy - gt_xy).T))
            self._suptitle.set_text(time_label)
        else:
            self._ims[0].set_data(_zoom_slice(pred_xy.T))
            self._titles[0].set_text(f"Prediction  {self._field} {self._unit}  |  {time_label}")
        if centre_xy is not None:
            for beam in self._beams:
                beam.center = (centre_xy[0] * 1e3, centre_xy[1] * 1e3)
                beam.set_visible(True)

        # Melt-pool edge contour on the GT + prediction panels (not the diff), from
        # the raw served T at this slice. Contour sets can't be updated in place,
        # so remove last frame's and redraw; upsample to match the crisp image.
        if self._contour_k is not None:
            for cs in self._contours:
                cs.remove()
            self._contours = []
            panel_t = [(0, t_gt_xy), (1, t_pred_xy)] if self._has_gt else [(0, t_pred_xy)]
            for k, t_xy in panel_t:
                if t_xy is None:
                    continue
                self._contours.append(
                    _draw_melt_contour(
                        self._axes[k], _zoom_slice(np.asarray(t_xy).T), self._extent_mm,
                        level=self._contour_k, add_label=False,
                    )
                )

        buf = BytesIO()
        # compress_level 3 (PIL) instead of Agg's default 6: ~same size class,
        # measurably faster to encode.
        self._fig.savefig(buf, format="png", dpi=self._dpi, pil_kwargs={"compress_level": 3})
        return buf.getvalue()


def _render_iso_surface_img(
    t_vol: np.ndarray,
    alpha_vol: np.ndarray,
    scalar_vol: np.ndarray,
    *,
    color_field: str,
    mesh_size_um: float,
    width_px: int,
    height_px: int,
    smoothing_iterations: int,
    plane: str,
    clim: list[float] | None = None,
) -> tuple[np.ndarray, list[float] | None]:
    """Render the predicted melt-pool surface of one steady volume, matching the
    straight/path 3D block look: threshold by the predicted melt fraction
    ``alpha >= 0.5`` (so the surface is the melt region regardless of peak T — a
    RIGNO that never exceeds the 3123 K fill still carves correctly), invert the
    free-surface T-fill so metal cells show real temperature, and use a pitch-only
    top camera / y-midplane cross-section (no yaw or roll, so +x stays horizontal).
    Takes 3D (X,Y,Z) T + alpha + color volumes so the 5D prediction and the plain
    3D ground truth share it with one camera + clim. Returns (img, clim)."""
    try:
        pv = _import_pyvista()
    except Exception as exc:  # noqa: BLE001
        raise RenderError(f"PyVista unavailable for 3D render: {type(exc).__name__}: {exc}") from exc

    t_vol = np.asarray(t_vol, dtype=np.float32)
    alpha_vol = np.asarray(alpha_vol, dtype=np.float32)
    scalar_vol = np.asarray(scalar_vol, dtype=np.float32)
    if not (t_vol.shape == alpha_vol.shape == scalar_vol.shape):
        raise RenderError(
            f"3D render shape mismatch: T={t_vol.shape}, alpha={alpha_vol.shape}, scalar={scalar_vol.shape}"
        )

    plot_val = scalar_vol  # raw served field (no fill inversion)

    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    pv.OFF_SCREEN = True
    try:
        pv.set_jupyter_backend("static")
    except Exception:
        pass

    grid = pv.ImageData(
        dimensions=(int(t_vol.shape[0]), int(t_vol.shape[1]), int(t_vol.shape[2])),
        spacing=(float(mesh_size_um), float(mesh_size_um), float(mesh_size_um)),
    )
    grid.point_data["alpha"] = alpha_vol.ravel(order="F")
    grid.point_data["plot_val"] = plot_val.ravel(order="F")
    # all_scalars: keep only cells whose every corner is metal, so the extracted
    # surface never carries gas-side (filled) values.
    thr = grid.threshold(0.5, scalars="alpha", all_scalars=True)
    if getattr(thr, "n_points", 0) == 0:
        raise RenderError("No 3D surface extracted for alpha >= 0.5")
    if plane == "xz":
        # Cut at the y-midplane and keep the FAR half so the exposed cut face is
        # viewed from -y (below) and +x runs left->right (the laser travel dir).
        thr = thr.clip(normal=(0.0, 1.0, 0.0), origin=grid.center, invert=False)
        if getattr(thr, "n_points", 0) == 0:
            raise RenderError("Empty steady block after y-midplane clip")
    surf = thr.extract_surface()
    # Smoothing rounds the flat cut face, so only soften the staircase for the top view.
    if plane != "xz" and int(smoothing_iterations) > 0:
        surf = surf.smooth(n_iter=int(smoothing_iterations))

    if clim is None:
        if color_field == "T":
            metal = alpha_vol >= 0.5
            clim = [0.0, float(np.max(t_vol[metal])) if np.any(metal) else 3200.0]
        else:
            clim = [0.0, 1.0]

    plotter = None
    try:
        plotter = pv.Plotter(off_screen=True, window_size=(int(width_px), int(height_px)))
        plotter.set_background("white")
        plotter.add_mesh(
            surf, scalars="plot_val", cmap="inferno", clim=list(clim),
            smooth_shading=True, ambient=0.25, diffuse=0.75, show_scalar_bar=False,
        )
        cx, cy, cz = grid.center
        if plane == "xz":
            # Look at the exposed y-midplane cut face from -y, tilted ~22 deg; up +z.
            tilt = np.deg2rad(22.0)
            plotter.camera_position = [(cx, cy - np.cos(tilt), cz + np.sin(tilt)), (cx, cy, cz), (0.0, 0.0, 1.0)]
            plotter.reset_camera()
            plotter.camera.zoom(1.5)
        else:
            # Pitch-only camera in the y-z plane above/in front of the block; no
            # yaw or roll, so the block's long x edges stay exactly horizontal.
            tilt = np.deg2rad(30.0)
            plotter.camera_position = [
                (cx, cy - np.sin(tilt), cz + np.cos(tilt)), (cx, cy, cz), (0.0, np.cos(tilt), np.sin(tilt)),
            ]
            plotter.reset_camera()
            plotter.camera.zoom(1.42)
        plotter.reset_camera_clipping_range()
        img = plotter.screenshot(return_img=True)
    except Exception as exc:  # noqa: BLE001
        raise RenderError(f"3D rendering failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if plotter is not None:
            plotter.close()

    return _zoom_image(img, crop_fraction=0.06), clim


def render_3d_surface_png(
    *,
    preds: dict[str, np.ndarray],
    field: str,
    mesh_size_um: float,
    width_px: int,
    height_px: int,
    smoothing_iterations: int,
    plane: str = "xz",
) -> RenderedPlot:
    if "T" not in preds or "alpha" not in preds:
        raise RenderError("3D surface rendering requires 'T' and 'alpha' in predictions")
    color_field = field if field in preds else "T"
    t_vol, grid_shape_xyz = _extract_volume(preds, "T")
    alpha_vol, _ = _extract_volume(preds, "alpha")
    scalar_vol, scalar_shape = _extract_volume(preds, color_field)
    if scalar_shape != grid_shape_xyz:
        raise RenderError(f"Shape mismatch for 3D render: T={grid_shape_xyz}, {color_field}={scalar_shape}")
    img, clim = _render_iso_surface_img(
        t_vol, alpha_vol, scalar_vol, color_field=color_field, mesh_size_um=mesh_size_um,
        width_px=width_px, height_px=height_px, smoothing_iterations=smoothing_iterations, plane=plane,
    )
    return RenderedPlot(
        png_bytes=_compose_3d_figure_with_colorbar(
            img=img, color_field=color_field, clim=clim, width_px=width_px, height_px=height_px,
        ),
        y_index_used=-1,
        grid_shape_xyz=grid_shape_xyz,
        width_px=width_px,
        height_px=height_px,
    )


def _interp_gt_to_grid(
    gt_vol: np.ndarray,
    gt_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    pred_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    fill: float,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Trilinearly resample a GT volume from its (x, y, z) axes onto the
    prediction's physical grid, so GT and prediction display at the same
    resolution. Prediction grid points outside the GT extent get ``fill``
    (``nan`` lets a caller mask them). Rendering uses float32; the metric in
    ``service._steady_resampled_metrics`` passes float64 so its numbers are
    unchanged from when it carried its own copy of this interpolator."""
    from scipy.interpolate import RegularGridInterpolator

    def _asc(axis: np.ndarray, vol_axis: int, vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(axis, dtype=np.float64)
        if a.size > 1 and a[0] > a[-1]:  # RegularGridInterpolator needs ascending axes
            a = a[::-1]
            vol = np.flip(vol, axis=vol_axis)
        return a, vol

    vol = np.asarray(gt_vol, dtype=dtype)
    gx, vol = _asc(gt_axes[0], 0, vol)
    gy, vol = _asc(gt_axes[1], 1, vol)
    gz, vol = _asc(gt_axes[2], 2, vol)
    interp = RegularGridInterpolator((gx, gy, gz), vol, bounds_error=False, fill_value=float(fill))
    px = np.asarray(pred_axes[0], dtype=np.float64)
    py = np.asarray(pred_axes[1], dtype=np.float64)
    pz = np.asarray(pred_axes[2], dtype=np.float64)
    PX, PY, PZ = np.meshgrid(px, py, pz, indexing="ij")
    out = interp(np.stack([PX.ravel(), PY.ravel(), PZ.ravel()], axis=-1))
    return out.reshape(px.size, py.size, pz.size).astype(dtype)


def render_steady_gt_compare_3d_png(
    *,
    pred: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    pred_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    gt_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    field: str,
    mesh_size_um: float,
    width_px: int,
    height_px: int,
    smoothing_iterations: int,
    plane: str = "xz",
    gt_split: str | None = None,
) -> RenderedPlot:
    """Steady 3D GT comparison: ground-truth melt-pool surface on top, prediction
    below, same camera + clim (the 3D analogue of the 2D stacked panels). The GT
    is trilinearly resampled onto the prediction's grid, so both blocks display at
    the same resolution and align cell-for-cell."""
    if "T" not in pred or "alpha" not in pred:
        raise RenderError("3D GT comparison requires 'T' and 'alpha' in predictions")
    pred_cf = field if field in pred else "T"
    t_pred, grid_shape_xyz = _extract_volume(pred, "T")
    alpha_pred, _ = _extract_volume(pred, "alpha")
    scalar_pred, _ = _extract_volume(pred, pred_cf)

    gt_cf = field if field in gt else "T"
    # Resample GT onto the prediction grid: same display resolution + alignment.
    t_gt = _interp_gt_to_grid(np.asarray(gt["T"]), gt_axes, pred_axes, fill=300.0)
    alpha_src = np.asarray(gt["alpha"]) if "alpha" in gt else np.zeros_like(np.asarray(gt["T"]))
    alpha_gt = _interp_gt_to_grid(alpha_src, gt_axes, pred_axes, fill=0.0)  # outside GT -> non-metal
    scalar_gt = _interp_gt_to_grid(
        np.asarray(gt[gt_cf]), gt_axes, pred_axes, fill=300.0 if gt_cf == "T" else 0.0
    )

    # One shared colorbar range across both blocks (raw metal T, or 0..1).
    if pred_cf == "T":
        peaks = []
        for tv, av in ((t_pred, alpha_pred), (t_gt, alpha_gt)):
            m = av >= 0.5
            if np.any(m):
                peaks.append(float(np.max(np.asarray(tv)[m])))
        clim: list[float] = [0.0, max(peaks) if peaks else 3200.0]
    else:
        clim = [0.0, 1.0]

    pred_img, _ = _render_iso_surface_img(
        t_pred, alpha_pred, scalar_pred, color_field=pred_cf, mesh_size_um=mesh_size_um,
        width_px=width_px, height_px=height_px, smoothing_iterations=smoothing_iterations, plane=plane, clim=clim,
    )
    gt_img, _ = _render_iso_surface_img(
        t_gt, alpha_gt, scalar_gt, color_field=gt_cf, mesh_size_um=mesh_size_um,
        width_px=width_px, height_px=height_px, smoothing_iterations=smoothing_iterations, plane=plane, clim=clim,
    )
    png = _compose_3d_gt_pred_figure(
        gt_img=gt_img,
        pred_img=pred_img,
        clim=clim,
        width_px=width_px,
        time_label="",
        gt_split=gt_split,
        field_label=_field_label(pred_cf),
    )
    return RenderedPlot(
        png_bytes=png, y_index_used=-1, grid_shape_xyz=grid_shape_xyz, width_px=width_px, height_px=height_px,
    )


# --------------------------------------------------------------------------- #
# /infer_path/render3d: per-frame 3D block PNGs
# --------------------------------------------------------------------------- #



def _block_time_label(time_s: float) -> str:
    return f"t = {time_s * 1e3:.2f} ms"


def render_3d_block_frames(
    *,
    preds: dict[str, np.ndarray],  # {"T": (nt,nx,ny,nz) Kelvin, "alpha": (nt,nx,ny,nz)}
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,  # metres
    times_s: np.ndarray,
    width_px: int,
    height_px: int,
    gt: dict[str, np.ndarray] | None = None,  # same shapes/masking as preds
    gt_split: str | None = None,  # train|val|test tag for the GT panel title
    plane: str = "xy",  # "xy" top view of the block | "xz" cut-away cross-section
) -> list[bytes]:
    """Per-frame PNGs of the metal block (alpha >= 0.5) colored by T.

    PyVista off-screen with a fixed pitch-only camera; each screenshot is
    composed with a matplotlib colorbar and time-label title outside the 3D
    image. With ``gt`` (preset paths), each frame is a portrait stack —
    ground truth above the prediction, same camera, one shared colorbar and
    one run-wide clim spanning both. Falls back to matplotlib top-surface
    frames when PyVista/EGL is unavailable (e.g. a stub test environment
    without a working headless GL); the fallback stays prediction-only.
    """
    try:
        frames_by_series, clim = _pyvista_block_frames(
            preds=preds, gt=gt, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
            width_px=width_px, height_px=height_px, plane=plane,
        )
    except Exception:  # noqa: BLE001 — contract must hold without headless GL
        logger.exception("PyVista block render failed; falling back to matplotlib frames")
        return _mpl_block_frames(
            preds=preds, x_axis=x_axis, y_axis=y_axis,
            times_s=times_s, width_px=width_px, height_px=height_px,
        )
    if gt is not None:
        gt_frames, pred_frames = frames_by_series
        return [
            _compose_3d_gt_pred_figure(
                gt_img=gt_frames[i],
                pred_img=pred_frames[i],
                clim=list(clim),
                width_px=width_px,
                time_label=_block_time_label(float(times_s[i])),
                gt_split=gt_split,
            )
            for i in range(len(pred_frames))
        ]
    (pred_frames,) = frames_by_series
    return [
        _compose_3d_figure_with_colorbar(
            img=img,
            color_field="T",
            clim=list(clim),
            width_px=width_px,
            height_px=height_px,
            title=_block_time_label(float(times_s[i])),
        )
        for i, img in enumerate(pred_frames)
    ]


def _block_run_clim(t_served: list[np.ndarray], metal_masks: list[np.ndarray]) -> tuple[float, float]:
    """One colorbar range for the whole run: floor 0 K up to the maximum
    temperature actually displayed on the thresholded (solid) block across all
    frames. The true peak temperatures sit in vaporized cells (alpha < 0.5)
    that the threshold removes, so the fixed 3200 K ceiling left the palette
    top unused."""
    tmax = 300.0  # minimum ceiling so vmax > vmin=0 even with no metal
    for t_i, mask in zip(t_served, metal_masks):
        if np.any(mask):
            tmax = max(tmax, float(np.max(t_i[mask])))
    return 0.0, tmax


def _pyvista_block_frames(
    *,
    preds: dict[str, np.ndarray],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    width_px: int,
    height_px: int,
    gt: dict[str, np.ndarray] | None = None,
    plane: str = "xy",
) -> tuple[list[list[np.ndarray]], tuple[float, float]]:
    """Screenshot every frame of one or two field series (GT first when given)
    with a single plotter, one fixed camera, and one run-wide clim spanning
    every displayed series. Returns one frame list per series. ``plane="xz"``
    clips the block at its y-midplane and looks at the exposed cross-section."""
    pv = _import_pyvista()
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    pv.OFF_SCREEN = True

    # Coordinates in mm: micrometre-scale metre values make VTK camera
    # clipping-range math fussy. Uniform spacing from the first grid step: the
    # training grids are uniform per axis.
    grid = pv.ImageData(
        dimensions=(int(x_axis.size), int(y_axis.size), int(z_axis.size)),
        spacing=tuple(float(a[1] - a[0]) * 1e3 for a in (x_axis, y_axis, z_axis)),
        origin=tuple(float(a[0]) * 1e3 for a in (x_axis, y_axis, z_axis)),
    )

    nt = int(preds["T"].shape[0])

    # Pass 1 (numpy only): collect the raw served T + metal masks, so one colorbar
    # range can span the whole run. No fill inversion — the surface is coloured by
    # the raw model output (T_served) the same as every other view.
    def _prep(series: dict[str, np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        t_list: list[np.ndarray] = []
        alphas: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for i in range(nt):
            t_i = np.asarray(series["T"][i], dtype=np.float32)
            alpha_i = np.asarray(series["alpha"][i], dtype=np.float32)
            t_list.append(t_i)
            alphas.append(alpha_i)
            masks.append(alpha_i >= 0.5)
        return t_list, alphas, masks

    series_data = [_prep(s) for s in ([gt, preds] if gt is not None else [preds])]
    clim = _block_run_clim(
        [t for t_list, _, _ in series_data for t in t_list],
        [m for _, _, masks in series_data for m in masks],
    )

    frames: list[list[np.ndarray]] = [[] for _ in series_data]
    plotter = pv.Plotter(off_screen=True, window_size=(int(width_px), int(height_px)))
    try:
        plotter.set_background("white")
        camera_set = False
        for i in range(nt):
            for s, (t_list, alphas, _masks) in enumerate(series_data):
                grid.point_data["T"] = t_list[i].ravel(order="F")
                grid.point_data["alpha"] = alphas[i].ravel(order="F")
                # all_scalars: keep only cells whose every point is metal, so the
                # extracted surface never carries gas-side (filled) temperatures.
                thr = grid.threshold(0.5, scalars="alpha", all_scalars=True)
                if getattr(thr, "n_points", 0) == 0:
                    raise RenderError("No block extracted for alpha >= 0.5")
                if plane == "xz":
                    # Cut the block at its y-midplane and keep the FAR half, so the
                    # exposed cut face is viewed from the -y side (below) and +x
                    # runs left->right (the laser's travel direction). Keeping the
                    # near half instead mirrors x and the motion looks reversed.
                    thr = thr.clip(normal=(0.0, 1.0, 0.0), origin=grid.center, invert=False)
                    if getattr(thr, "n_points", 0) == 0:
                        raise RenderError("Empty block after y-midplane clip")
                block = thr.extract_surface()
                # Smoothing rounds the flat cut face, so only soften the staircase
                # for the full-block top view.
                if plane != "xz":
                    block = block.smooth(n_iter=20)
                # Same actor name every frame: PyVista replaces the mesh in place,
                # so the plotter, scalar bar, and camera are built once.
                plotter.add_mesh(
                    block,
                    name="block",
                    scalars="T",
                    cmap="inferno",
                    clim=list(clim),
                    smooth_shading=True,
                    ambient=0.25,  # keep keyhole walls readable instead of black
                    diffuse=0.75,  # ambient+diffuse ~1 so the hot zone is not overexposed
                    show_scalar_bar=False,  # colorbar is composed outside, in matplotlib
                )
                if not camera_set:
                    cx, cy, cz = grid.center
                    if plane == "xz":
                        # Look at the exposed y-midplane cut face from the -y side,
                        # tilted ~22 deg above the x-axis so the top edge reads too;
                        # view-up +z. From -y looking +y, +x runs left->right.
                        tilt = np.deg2rad(22.0)
                        plotter.camera_position = [
                            (cx, cy - np.cos(tilt), cz + np.sin(tilt)),
                            (cx, cy, cz),
                            (0.0, 0.0, 1.0),
                        ]
                        plotter.reset_camera()
                        plotter.camera.zoom(1.5)
                    else:
                        # Fixed pitch-only camera in the y-z plane above/in front of
                        # the block, tilted ~30 deg from vertical. No yaw/roll, so
                        # the block's long x edges stay exactly horizontal.
                        tilt = np.deg2rad(30.0)
                        plotter.camera_position = [
                            (cx, cy - np.sin(tilt), cz + np.cos(tilt)),
                            (cx, cy, cz),
                            (0.0, np.cos(tilt), np.sin(tilt)),  # view-up in the y-z plane
                        ]
                        plotter.reset_camera()
                        plotter.camera.zoom(1.42)
                    camera_set = True
                plotter.reset_camera_clipping_range()
                frames[s].append(np.asarray(plotter.screenshot(return_img=True)))
    finally:
        plotter.close()
    return frames, clim


def _mpl_block_frames(
    *,
    preds: dict[str, np.ndarray],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    times_s: np.ndarray,
    width_px: int,
    height_px: int,
) -> list[bytes]:
    """Fallback: top-surface temperature (highest alpha>=0.5 cell per column)."""
    dpi = 100
    extent = [float(x_axis[0]) * 1e3, float(x_axis[-1]) * 1e3, float(y_axis[0]) * 1e3, float(y_axis[-1]) * 1e3]
    fig = Figure(figsize=(max(width_px, 1) / dpi, max(height_px, 1) / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    nt = int(preds["T"].shape[0])
    nz = int(preds["T"].shape[3])
    # Pass 1: top-surface temperatures per frame, and one run-wide colorbar
    # range from the displayed (solid) values.
    shown_frames: list[np.ndarray] = []
    shown_masks: list[np.ndarray] = []
    for i in range(nt):
        mask = np.asarray(preds["alpha"][i]) >= 0.5  # (nx,ny,nz)
        rev = mask[:, :, ::-1]
        has_solid = rev.any(axis=2)
        top = nz - 1 - rev.argmax(axis=2)  # highest solid z per (x,y) column
        t_surf = np.take_along_axis(np.asarray(preds["T"][i]), top[..., None], axis=2)[..., 0]
        shown_frames.append(np.where(has_solid, t_surf, np.nan))
        shown_masks.append(has_solid)
    clim = _block_run_clim(shown_frames, shown_masks)

    placeholder = np.full((y_axis.size, x_axis.size), np.nan, dtype=np.float32)
    im = ax.imshow(placeholder, origin="lower", extent=extent, cmap="inferno",
                   vmin=clim[0], vmax=clim[1], aspect="equal")
    fig.colorbar(im, ax=ax, label="T [K]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    title = ax.set_title("")

    frames: list[bytes] = []
    for i in range(nt):
        im.set_data(shown_frames[i].T)
        title.set_text(f"Block top surface  |  {_block_time_label(float(times_s[i]))}")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, pil_kwargs={"compress_level": 3})
        frames.append(buf.getvalue())
    return frames
