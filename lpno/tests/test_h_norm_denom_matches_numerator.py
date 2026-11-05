"""The H_norm denominator must be computed with the SAME formula as its numerator.

2026-08-20 added the beam-radius term to H_norm's numerator
(`preprocessing_datasets._preprocess_var`) on the reasoning that "a CONSTANT radius
cancels exactly" under the shared denominator. That holds for the eager path, which
divides by the max of the very array it just built. It does NOT hold for the lazy
path: `lazy_runtime.compute_h_norm_denom` fits `P/sqrt(V)` with no radius term, so the
factor never cancels.

Measured consequence (2026-08-21): every steady arm fed the network
H_norm = 2,504,968 instead of 0.8856 -- the correct value times 1/sqrt(r0^3) for
r0 = 50 um = 2.83e6. The mref and lab arms read exactly 1.0 and were unaffected.
A conditioning channel six orders of magnitude out is not a subtle degradation.
"""
from __future__ import annotations

import numpy as np
import pytest

from lpno.custom_utils.utils import h_norm_raw


def test_h_norm_raw_without_radius_is_the_classic_form():
    assert h_norm_raw(70.0, 0.298) == pytest.approx(70.0 / np.sqrt(0.298))


def test_h_norm_raw_applies_the_radius_term():
    r0 = 50e-6
    got = h_norm_raw(70.0, 0.298, r0)
    assert got == pytest.approx(70.0 / np.sqrt(0.298) / np.sqrt(r0 ** 3))


def test_a_constant_radius_cancels_under_a_matching_denominator():
    """The whole justification for making the radius term unconditional."""
    r0 = 50e-6
    powers, speeds = [70.0, 160.0, 200.0], [0.298, 0.617, 0.119]
    with_r = [h_norm_raw(p, v, r0) for p, v in zip(powers, speeds)]
    without = [h_norm_raw(p, v) for p, v in zip(powers, speeds)]
    norm_with = [h / max(with_r) for h in with_r]
    norm_without = [h / max(without) for h in without]
    assert norm_with == pytest.approx(norm_without)


def test_a_radius_blind_denominator_does_NOT_cancel_it():
    """Pin the actual defect: mixing the two formulas leaves the 1/sqrt(r0^3) factor."""
    r0 = 50e-6
    numerator = h_norm_raw(70.0, 0.298, r0)
    radius_blind_denom = max(h_norm_raw(p, v) for p, v in
                             ((70.0, 0.298), (160.0, 0.617), (200.0, 0.119)))
    leaked = numerator / radius_blind_denom
    assert leaked > 1e5
    assert leaked == pytest.approx(
        (numerator / max(h_norm_raw(p, v, r0) for p, v in
                         ((70.0, 0.298), (160.0, 0.617), (200.0, 0.119))))
        / np.sqrt(r0 ** 3), rel=1e-6)


def test_lazy_denominator_uses_the_shared_helper(monkeypatch):
    """compute_h_norm_denom must route through h_norm_raw, not its own copy."""
    from lpno.models.pipeline import lazy_runtime

    class _Store:
        def has_time_axis(self):
            return False

        def time_steps(self, case_idx):
            return 1

        def scalar_value(self, key, case_idx, time_idx=None):
            return {"P": 70.0, "V_scan": 0.298, "beam_radius": 50e-6}[key]

    seen = []
    real = lazy_runtime.h_norm_raw
    monkeypatch.setattr(lazy_runtime, "h_norm_raw",
                        lambda *a, **k: (seen.append(a) or real(*a, **k)), raising=True)
    denom = lazy_runtime.compute_h_norm_denom(
        _Store(), [0], time_as_batch=False, include_radius=True)
    assert seen, "compute_h_norm_denom did not call the shared h_norm_raw"
    assert denom == pytest.approx(h_norm_raw(70.0, 0.298, 50e-6))
