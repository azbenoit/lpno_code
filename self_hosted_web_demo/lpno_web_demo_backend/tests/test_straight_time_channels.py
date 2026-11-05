"""The straight page must build its time-integral channels the way training did.

Found 2026-09-14 by diffing the demo's inputs for one case against the eval's own
preview batch: seven channels matched to the bit and three did not --
path_dose, path_dose_recent (1.97x too small) and conduction_field. Two causes,
both in StraightCase, both fixed by giving it what PathCase already had:

  * no `physics`  -> _path_channels' getattr fell back to power_weighted=False,
    although every lab model trains on power-weighted dose;
  * no `times_fine`/`frame_idx` -> the dose and conduction integrals ran on the
    20 rendered frames (dt up to 250 us) instead of the 5 us export cadence.

These pin the mechanism; the end-to-end check is the 13-case substrate MAE against
the recorded evaluation (2026-09-14).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.adapter_randompath import ModelPhysics
from app.adapter_straightpath import STRAIGHT_FRAME_DT_S, StraightPathAdapter
from app.config import Settings


@pytest.fixture(scope="module")
def adapter() -> StraightPathAdapter:
    return StraightPathAdapter(Settings(use_stub_runtime=True), stub=True)


def test_resolve_builds_a_fine_time_grid_at_the_export_cadence(adapter: StraightPathAdapter) -> None:
    case = adapter.resolve(p_watts=150.0, v_scan_mps=0.6)
    fine, idx = case.times_fine, case.frame_idx
    assert fine is not None and idx is not None
    assert fine[0] == 0.0
    assert np.all(np.diff(fine) <= STRAIGHT_FRAME_DT_S + 1e-12)
    assert fine[-1] >= case.laser_off_time_s - STRAIGHT_FRAME_DT_S
    # every rendered frame maps to the nearest fine sample
    assert idx.shape == case.times.shape
    assert np.all(np.abs(fine[idx] - case.times) <= STRAIGHT_FRAME_DT_S / 2 + 1e-12)
    # and the fine grid is genuinely finer than the rendered one
    assert fine.size >= 5 * case.times.size   # 0.995 ms at 5 us = 200 samples for 20 frames


def test_resolve_carries_the_serving_models_physics(adapter: StraightPathAdapter) -> None:
    ph = ModelPhysics(power_weighted_dose=True, h_norm_denom=275.0539301382)
    case = adapter.resolve(p_watts=150.0, v_scan_mps=0.6, physics=ph, h_ref=ph.h_norm_denom)
    assert case.physics is ph
    assert adapter.resolve(p_watts=150.0, v_scan_mps=0.6).physics is None


def test_dose_is_power_weighted_when_the_model_says_so(adapter: StraightPathAdapter) -> None:
    """Same track, same speed, double the power: a power-weighted dose doubles,
    an unweighted one does not move."""
    def dose(p_watts: float, weighted: bool) -> np.ndarray:
        ph = ModelPhysics(power_weighted_dose=weighted)
        case = adapter.resolve(p_watts=p_watts, v_scan_mps=0.6, physics=ph)
        return np.asarray(adapter._path_channels(case, keys=["path_dose"])["path_dose"])

    w100, w200 = dose(100.0, True), dose(200.0, True)
    u100, u200 = dose(100.0, False), dose(200.0, False)
    assert w200.max() > 0
    assert w200.max() == pytest.approx(2.0 * w100.max(), rel=1e-6)
    assert u200.max() == pytest.approx(u100.max(), rel=1e-6)


def test_channels_are_integrated_on_the_fine_grid(adapter: StraightPathAdapter) -> None:
    """The last rendered frame's dose must equal the fine-grid integral at that
    time, not a 20-sample quadrature of it: build both and compare."""
    from types import SimpleNamespace
    from dataclasses import fields

    ph = ModelPhysics(power_weighted_dose=True)
    case = adapter.resolve(p_watts=150.0, v_scan_mps=0.6, physics=ph)
    fine_dose = np.asarray(adapter._path_channels(case, keys=["path_dose"])["path_dose"])[-1]
    coarse = SimpleNamespace(**{f.name: getattr(case, f.name) for f in fields(case)})
    coarse.times_fine = None
    coarse.frame_idx = None
    coarse_dose = np.asarray(adapter._path_channels(coarse, keys=["path_dose"])["path_dose"])[-1]
    # a 20-sample quadrature of a beam-width kernel is not the same integral
    assert not np.allclose(fine_dose, coarse_dose, rtol=1e-3, atol=1e-6)
