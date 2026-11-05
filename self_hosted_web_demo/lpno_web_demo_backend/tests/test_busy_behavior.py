from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
from app.config import Settings
from app.schemas import InferRequest
from app.service import BusyError, InferenceService


class SlowAdapter:
    def infer(self, runtime, p_watts: float, v_scan_mps: float, x_shift_m=0.0):
        time.sleep(0.15)
        nx, ny, nz = 16, 8, 12
        t = np.full((1, 1, nx, ny, nz), 1800.0, dtype=np.float32)
        a = np.full((1, 1, nx, ny, nz), 0.7, dtype=np.float32)
        f = np.full((1, 1, nx, ny, nz), 0.9, dtype=np.float32)
        return {"T": t, "alpha": a, "fl": f}

    def runtime_summary(self, runtime):
        return {"mode": "test"}


def _req(seq: int) -> InferRequest:
    return InferRequest.model_validate({
        "client_request_id": f"req-{seq}",
        "client_seq": seq,
        "mode": "preview",
        "inputs": {"p_watts": 80, "v_scan_mps": 0.3},
        "view": {"field": "T", "y_index": -1, "show_t_contour": False, "t_contour_k": 1800.0},
        "render": {"width_px": 640, "height_px": 360},
    })


def test_concurrent_requests_busy_reject() -> None:
    async def _run() -> None:
        settings = Settings(startup_load_runtime=False, use_stub_runtime=True)
        service = InferenceService(settings=settings, adapter=SlowAdapter(), runtime=SimpleNamespace(h_norm_denom=1.0))

        t1 = asyncio.create_task(service.infer(_req(1), server_request_id="srv-1"))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(service.infer(_req(2), server_request_id="srv-2"))

        r1 = await t1
        r2 = await asyncio.gather(t2, return_exceptions=True)
        err = r2[0]

        assert r1.status == "ok"
        assert isinstance(err, BusyError)

    asyncio.run(_run())
