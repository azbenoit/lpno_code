"""The steady moving-frame window is per case, and the served window must follow it.

The steady models are trained on a 61-cell beam-following slice of the ABSOLUTE
domain. That slice is NOT in the same place for every case: measured over the 13
cases of ds 58056 (2026-09-03), raw x[0] runs -1.877e-4..-1.550e-4 and the beam
sits at window index 35..44. Serving every case on the runtime's single window
compared a melt pool against its own neighbourhood: the GT-compare panel reported
63-192 K where model 63732's recorded evaluation is 7.51 K substrate.

These pin the two halves of the fix: the shift is computed from the matched case,
and it actually translates the x channel the model sees.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.adapter_pino import PinoRuntimeAdapter
from app.service import InferenceService

REF_L = 1.0e-4


def _ctx(x_axis: np.ndarray) -> SimpleNamespace:
    """A context whose `infer` records the invar it was handed."""
    seen: dict = {}

    class _Recorder:
        def run(self, invar, **_kw):
            seen.update(invar)
            shape = invar[next(iter(invar))].shape
            return {"T": np.zeros(shape, dtype=np.float32)}

    return SimpleNamespace(
        infer=_Recorder(),
        seen=seen,
        input_keys=["x", "P", "V_scan"],
        output_keys=["T"],
        template_input={
            "x": x_axis.reshape(1, 1, -1, 1, 1).astype(np.float32) / REF_L,
            "P": np.zeros((1, 1, x_axis.size, 1, 1), dtype=np.float32),
            "V_scan": np.zeros((1, 1, x_axis.size, 1, 1), dtype=np.float32),
        },
        ref_scales={"P": 10.0, "V": 0.1, "T": 3000.0, "L": REF_L},
        material_params={},
        alpha_masking=False,
        h_norm_denom=275.0539301382,
        results_dir=None,
        bounds=[],
        x_phys=x_axis,
        y_phys=np.zeros(1),
        z_phys=np.zeros(1),
        norm_summary="",
        serving={},
    )


_IMPORTS = {
    "smooth_liquid_fraction_numpy": lambda *a, **k: None,
    "fill_where_mask_smooth_np": lambda *a, **k: None,
}


@pytest.mark.parametrize("shift_m", [0.0, -3.267e-5, 1.0e-5])
def test_x_shift_translates_the_served_window(shift_m: float) -> None:
    x = np.linspace(-1.55e-4, 4.45e-4, 61, dtype=np.float32)
    ctx = _ctx(x)
    PinoRuntimeAdapter()._run_inference_for_p_v(
        ctx, 150.0, 0.6, imports=_IMPORTS, x_shift_m=shift_m
    )
    served = ctx.seen["x"].reshape(-1) * REF_L
    assert np.allclose(served, x + shift_m, atol=1e-12)


def test_x_shift_leaves_the_template_untouched() -> None:
    """The shift must not accumulate across requests (template_input is shared)."""
    x = np.linspace(-1.55e-4, 4.45e-4, 61, dtype=np.float32)
    ctx = _ctx(x)
    a = PinoRuntimeAdapter()
    for _ in range(3):
        a._run_inference_for_p_v(ctx, 150.0, 0.6, imports=_IMPORTS, x_shift_m=-1.0e-5)
    served = ctx.seen["x"].reshape(-1) * REF_L
    assert np.allclose(served, x - 1.0e-5, atol=1e-12)


def _svc(runtime_x: np.ndarray | None) -> SimpleNamespace:
    rt = SimpleNamespace(ctx=SimpleNamespace(x_phys=runtime_x)) if runtime_x is not None else None
    return SimpleNamespace(_steady_runtime_for=lambda _m: rt)


def _gt(x: np.ndarray) -> tuple:
    return ({}, (x, np.zeros(1), np.zeros(1)), (0.0, 0.0), "test")


def test_window_shift_is_the_offset_between_the_two_windows() -> None:
    runtime_x = np.linspace(-1.55e-4, 4.45e-4, 61)
    case_x = np.linspace(-1.8767e-4, 4.1233e-4, 61)   # train_170W_0960mps, the worst offset
    got = InferenceService._steady_window_shift(_svc(runtime_x), "fno_cal", _gt(case_x))
    assert got == pytest.approx(-3.267e-5, abs=1e-9)


def test_window_shift_is_zero_without_a_matched_case() -> None:
    runtime_x = np.linspace(-1.55e-4, 4.45e-4, 61)
    assert InferenceService._steady_window_shift(_svc(runtime_x), "fno_cal", None) == 0.0


def test_window_shift_refuses_a_different_cell_count() -> None:
    """A different nx is a geometry mismatch, not an offset; shifting would hide it."""
    runtime_x = np.linspace(-1.55e-4, 4.45e-4, 61)
    case_x = np.linspace(-1.55e-4, 4.45e-4, 90)
    assert InferenceService._steady_window_shift(_svc(runtime_x), "fno_cal", _gt(case_x)) == 0.0
