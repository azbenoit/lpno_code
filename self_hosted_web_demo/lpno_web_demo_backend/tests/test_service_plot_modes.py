from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from app.render import RenderedPlot
from app.service import InferenceService
from app.config import Settings
from app.schemas import InferRequest


class CountingAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, runtime, p_watts: float, v_scan_mps: float, x_shift_m=0.0):
        self.calls += 1
        nx, ny, nz = 12, 8, 10
        t = np.full((1, 1, nx, ny, nz), 1800.0 + p_watts, dtype=np.float32)
        alpha = np.full((1, 1, nx, ny, nz), min(v_scan_mps, 1.0), dtype=np.float32)
        fl = np.full((1, 1, nx, ny, nz), 0.4, dtype=np.float32)
        return {"T": t, "alpha": alpha, "fl": fl}

    def runtime_summary(self, runtime):
        return {"mode": "counting"}


def _req(*, seq: int, mode: str, plot_mode: str, field: str = "T") -> InferRequest:
    return InferRequest.model_validate({
        "client_request_id": f"req-{seq}",
        "client_seq": seq,
        "mode": mode,
        "inputs": {"p_watts": 80, "v_scan_mps": 0.3},
        "view": {
            "field": field,
            "plot_mode": plot_mode,
            "y_index": -1,
            "show_t_contour": plot_mode == "2d",
            "t_contour_k": 1800.0,
        },
        "render": {"width_px": 640, "height_px": 360},
    })


def test_service_dispatches_2d_and_3d_renderers(monkeypatch) -> None:
    settings = Settings(startup_load_runtime=False, use_stub_runtime=True)
    adapter = CountingAdapter()
    service = InferenceService(settings=settings, adapter=adapter, runtime=SimpleNamespace(h_norm_denom=1.0))
    calls: list[tuple[str, int]] = []

    def fake_2d(**kwargs):
        calls.append(("2d", kwargs["y_index"]))
        return RenderedPlot(b"png2d", kwargs["y_index"], (12, 8, 10), kwargs["width_px"], kwargs["height_px"])

    def fake_3d(**kwargs):
        calls.append(("3d", kwargs["smoothing_iterations"]))
        return RenderedPlot(b"png3d", -1, (12, 8, 10), kwargs["width_px"], kwargs["height_px"])

    monkeypatch.setattr("app.service.render_xz_cross_section_png", fake_2d)
    monkeypatch.setattr("app.service.render_3d_surface_png", fake_3d)

    resp_2d = service._infer_sync(_req(seq=1, mode="preview", plot_mode="2d"), "srv-1", time.perf_counter())
    resp_3d_preview = service._infer_sync(_req(seq=2, mode="preview", plot_mode="3d", field="alpha"), "srv-2", time.perf_counter())
    resp_3d_final = service._infer_sync(_req(seq=3, mode="final", plot_mode="3d", field="alpha"), "srv-3", time.perf_counter())

    assert resp_2d.result_meta.plot_mode_used == "2d"
    assert resp_3d_preview.result_meta.plot_mode_used == "3d"
    assert resp_3d_final.result_meta.plot_mode_used == "3d"
    assert calls == [("2d", -1), ("3d", 0), ("3d", 20)]


def test_service_reuses_cached_predictions_across_plot_modes(monkeypatch) -> None:
    settings = Settings(startup_load_runtime=False, use_stub_runtime=True)
    adapter = CountingAdapter()
    service = InferenceService(settings=settings, adapter=adapter, runtime=SimpleNamespace(h_norm_denom=1.0))

    monkeypatch.setattr(
        "app.service.render_xz_cross_section_png",
        lambda **kwargs: RenderedPlot(b"png2d", kwargs["y_index"], (12, 8, 10), kwargs["width_px"], kwargs["height_px"]),
    )
    monkeypatch.setattr(
        "app.service.render_3d_surface_png",
        lambda **kwargs: RenderedPlot(b"png3d", -1, (12, 8, 10), kwargs["width_px"], kwargs["height_px"]),
    )

    service._infer_sync(_req(seq=1, mode="preview", plot_mode="2d"), "srv-1", time.perf_counter())
    service._infer_sync(_req(seq=2, mode="final", plot_mode="3d", field="fl"), "srv-2", time.perf_counter())

    assert adapter.calls == 1
