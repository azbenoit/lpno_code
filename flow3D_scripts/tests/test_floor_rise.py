"""`domain_validity`'s floor check is blind to a preheated substrate.

`MAX_FLOOR_K = 400.0` is an ABSOLUTE threshold. It answers "did heat reach the
bottom boundary" only when the substrate starts near room temperature: a case
preheated to 500-770 K has its floor above 400 K on frame 0, before the laser
fires, so it is flagged invalid by construction.

Measured 2026-08-22 on the two ti64_preheat60 batches (60 + 60 cases): 47 of 60
in EACH read `ok = 0`, and invalidity tracks the preheat axis exactly -- every
case at or above 400 K preheat invalid, every case below it valid. The floor's
median RISE above its own initial temperature is +0.2 K, i.e. the floor is
untouched. Only 9 of 60 (extension) and 10 of 60 (parent) have heat genuinely
reaching the bottom.

So `floor_rise_k` measures the quantity the check actually wants. Following the
`end_wall_contact` precedent it is REPORTED, not fatal: flipping `ok` would
retroactively revalidate every preheated case in the back-catalogue, and that
promotion is a human's call.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import melt_metrics as mm  # noqa: E402

from test_melt_metrics import synthetic  # noqa: E402


def preheated(bulk_k: float, floor_k: float):
    """A case whose metal sits at `bulk_k` with its bottom row at `floor_k`."""
    d = synthetic(floor_k=floor_k)
    t = np.asarray(d["Temperature"], dtype=float).copy()
    t[...] = bulk_k
    t[:, :, :, 0] = floor_k
    d["Temperature"] = t
    return d


class FloorRiseTest(unittest.TestCase):
    def test_untouched_floor_of_a_preheated_case_reads_near_zero_rise(self):
        """700 K preheat, floor still at 700 K: no heat reached the bottom."""
        v = mm.domain_validity(preheated(bulk_k=700.0, floor_k=700.0))
        self.assertAlmostEqual(v.floor_rise_k, 0.0, places=6)

    def test_a_genuinely_heated_floor_reads_a_large_rise(self):
        """Same preheat, floor driven to 2000 K: heat did reach the bottom."""
        v = mm.domain_validity(preheated(bulk_k=700.0, floor_k=2000.0))
        self.assertGreater(v.floor_rise_k, 1000.0)

    def test_rise_separates_the_two_where_the_absolute_floor_does_not(self):
        """Both exceed MAX_FLOOR_K; only the rise tells them apart."""
        cold = mm.domain_validity(preheated(bulk_k=700.0, floor_k=700.0))
        hot = mm.domain_validity(preheated(bulk_k=700.0, floor_k=2000.0))
        self.assertGreater(cold.floor_k, mm.MAX_FLOOR_K)
        self.assertGreater(hot.floor_k, mm.MAX_FLOOR_K)
        self.assertGreater(hot.floor_rise_k - cold.floor_rise_k, 1000.0)

    def test_ok_is_NOT_changed_by_this(self):
        """Pin the deliberate non-change: `ok` still uses the absolute threshold.

        If a later change promotes floor_rise_k into `ok`, this test should be
        updated consciously rather than silently.
        """
        v = mm.domain_validity(preheated(bulk_k=700.0, floor_k=700.0))
        self.assertFalse(v.ok, "an untouched preheated floor still reads ok=0 today")

    def test_room_temperature_case_is_unaffected(self):
        """The ordinary case keeps its old meaning: rise ~ floor - ambient."""
        v = mm.domain_validity(synthetic(floor_k=300.0))
        self.assertTrue(v.ok)
        self.assertLess(abs(v.floor_rise_k), 200.0)


if __name__ == "__main__":
    unittest.main()
