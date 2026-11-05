from __future__ import annotations

import numpy as np
import pytest

from app.render import (
    RenderError,
    render_3d_surface_png,
    render_xz_cross_section_png,
)


def test_render_xz_cross_section_png_returns_png_bytes() -> None:
    nx, ny, nz = 32, 20, 24
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(0, 1, ny, dtype=np.float32)
    z = np.linspace(0, 1, nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    t = (300 + 2800 * np.exp(-((X - 0.5) ** 2 / 0.05 + (Z - 0.4) ** 2 / 0.06))).astype(np.float32)
    alpha = np.clip((t - 1500) / 800, 0, 1).astype(np.float32)

    preds = {
        "T": t[None, None, ...],
        "alpha": alpha[None, None, ...],
        "fl": np.clip((t - 1700) / 300, 0, 1).astype(np.float32)[None, None, ...],
    }

    rendered = render_xz_cross_section_png(
        preds=preds,
        field="T",
        y_index=-1,
        mesh_size_um=10.0,
        show_t_contour=True,
        t_contour_k=1898.0,
        width_px=800,
        height_px=450,
    )

    assert rendered.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert rendered.grid_shape_xyz == (nx, ny, nz)
    assert rendered.width_px == 800
    assert rendered.height_px == 450
    assert 0 <= rendered.y_index_used < ny


class _FakeSurface:
    def __init__(self, n_points: int = 32) -> None:
        self.n_points = n_points
        self.smoothed_with = None

    def clip(self, *_args, **_kwargs):
        return self

    def extract_surface(self):
        return self

    def smooth(self, n_iter: int):
        self.smoothed_with = n_iter
        return self


class _FakeImageData:
    def __init__(self, *, dimensions, spacing) -> None:
        self.dimensions = dimensions
        self.spacing = spacing
        self.cell_data = {}
        self.point_data = {}
        self.center = (0.0, 0.0, 0.0)
        self.surface = _FakeSurface()

    def threshold(self, *_args, **_kwargs):
        return self.surface


class _FakeCamera:
    def __init__(self) -> None:
        self.azimuth = 0.0
        self.elevation = 0.0
        self.roll = 0.0

    def zoom(self, _factor: float) -> None:
        return None


class _FakePlotter:
    def __init__(self, *, off_screen: bool, window_size: tuple[int, int]) -> None:
        self.off_screen = off_screen
        self.window_size = window_size
        self.camera = _FakeCamera()
        self.camera_position = None

    def set_background(self, _color: str) -> None:
        return None

    def add_mesh(self, *_args, **_kwargs) -> None:
        return None

    def view_isometric(self) -> None:
        return None

    def view_xy(self) -> None:
        return None

    def reset_camera(self) -> None:
        return None

    def reset_camera_clipping_range(self) -> None:
        return None

    def screenshot(self, return_img: bool = False):
        assert return_img is True
        return np.ones((self.window_size[1], self.window_size[0], 3), dtype=np.float32)

    def close(self) -> None:
        return None


class _FakePyVista:
    OFF_SCREEN = False
    ImageData = _FakeImageData
    Plotter = _FakePlotter

    @staticmethod
    def set_jupyter_backend(_backend: str) -> None:
        return None


def test_render_3d_surface_png_returns_png_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    nx, ny, nz = 16, 10, 12
    t = np.full((1, 1, nx, ny, nz), 1800.0, dtype=np.float32)
    alpha = np.full((1, 1, nx, ny, nz), 0.7, dtype=np.float32)
    preds = {"T": t, "alpha": alpha}

    monkeypatch.setattr("app.render._import_pyvista", lambda: _FakePyVista())
    rendered = render_3d_surface_png(
        preds=preds,
        field="alpha",
        mesh_size_um=10.0,
        width_px=640,
        height_px=360,
        smoothing_iterations=20,
    )

    assert rendered.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert rendered.grid_shape_xyz == (nx, ny, nz)
    assert rendered.width_px == 640
    assert rendered.height_px == 360
    assert rendered.y_index_used == -1


def test_render_3d_surface_png_requires_temperature_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.render._import_pyvista", lambda: _FakePyVista())
    with pytest.raises(RenderError, match="requires 'T'"):
        render_3d_surface_png(
            preds={"alpha": np.zeros((1, 1, 8, 8, 8), dtype=np.float32)},
            field="alpha",
            mesh_size_um=10.0,
            width_px=640,
            height_px=360,
            smoothing_iterations=0,
        )


def test_render_3d_surface_png_rejects_invalid_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.render._import_pyvista", lambda: _FakePyVista())
    with pytest.raises(RenderError, match="Expected 5D array"):
        render_3d_surface_png(
            preds={"T": np.zeros((8, 8, 8), dtype=np.float32), "alpha": np.zeros((8, 8, 8), dtype=np.float32)},
            field="T",
            mesh_size_um=10.0,
            width_px=640,
            height_px=360,
            smoothing_iterations=0,
        )


def test_path_diff_colorbar_is_constant_across_frames() -> None:
    """The Prediction - GT diff panel colorbar must be fixed run-wide, not
    rescaled per frame: a later frame with a much larger diff must not widen
    the colorbar. Regression for the per-frame `set_clim` that made the diff
    colorbar values change over the time series.
    """
    from app.render import PathFrameRenderer

    x = np.linspace(-7e-4, 7e-4, 8, dtype=np.float32)
    y = np.linspace(-5e-4, 5e-4, 6, dtype=np.float32)
    r = PathFrameRenderer(
        field="T", has_gt=True, x_axis=x, y_axis=y, path_xy=None,
        width_px=480, diff_dmax=5.0,
    )
    same = np.ones((x.size, y.size), dtype=np.float32)
    r.render(pred_xy=same, gt_xy=same, centre_xy=None, time_s=0.0)  # ~zero diff
    clim0 = r._ims[2].get_clim()

    big = np.zeros((x.size, y.size), dtype=np.float32)
    big[0, 0] = 1000.0  # a huge diff in a later frame
    r.render(pred_xy=big, gt_xy=np.zeros((x.size, y.size), dtype=np.float32),
             centre_xy=None, time_s=1e-3)
    clim1 = r._ims[2].get_clim()

    assert clim0 == (-5.0, 5.0)
    assert clim1 == clim0  # constant across time despite the large frame-2 diff
