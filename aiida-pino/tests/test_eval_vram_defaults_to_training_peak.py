"""A grid-bound eval booked below its model's measured peak is RAISED, not refused.

Of 505 failed eval processes since 2026-08-15 (classified 2026-09-06), 74
died on "eval VRAM below the model's measured training peak" and 52 more OOMed at eval, a
quarter of all failures between them. Both are the same gap: `evaluation_profile.gpu_vram_mb`
is supplied by hand, while the model node already carries `gpu_vram_peak_mb` measured during
training -- and the guard's own refusal message quotes that number, so the correct value is
known at the moment of refusal.

The auto-sizer in `EvaluateModelWorkChain` only fires when the profile supplies NOTHING, so a
submitter who supplies a too-low figure gets a refusal instead of a correction.

Scope is deliberately narrow and follows the existing guard, not the report: only grid-bound
architectures (RIGNO, GAOT), whose peak comes from the edge set / encoder graph that a smaller
eval batch does not shrink. FNO is excluded because its footprint DOES scale with the batch --
steady FNO evals peak at ~164 MiB against far larger training bookings, so defaulting those to
the training peak would over-book by 4-5x, which the workchain comments already reject.
Time-averaged evals stay exempt for the same measured reason (a RIGNO steady model trained at 12000 and
evaluated cleanly at 3000).

The correction anchors on a MEASUREMENT and only on one. Where no `gpu_vram_peak_mb` was
recorded the guard still REFUSES, unchanged: the training booking is "whatever a human or a
clone put there", so silently raising to it would repeat the model-66067 failure of forcing a
figure nothing on the box could free. That refusal is also the protection the origin case
needed (calc 61877 recorded no peak), and it stays pinned by
`tests/test_eval_vram_vs_training.py`, which this change does not touch.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.model_registry import _check_graph_eval_vram_against_training as check


def _frozen(config="config_flow3D_RIGNO_randompath", booked=20000):
    return {"config_name": config, "gpu_vram_mb": booked}


class RaiseRatherThanRefuseTest(unittest.TestCase):
    def test_an_under_booked_grid_bound_eval_is_raised_to_the_measured_peak(self):
        merged = {"gpu_vram_mb": 16000, "time_avg_window": 1}
        check(_frozen(), merged, 19530)
        self.assertEqual(merged["gpu_vram_mb"], 19530, "the booking was not corrected")

    def test_it_does_not_lower_a_sufficient_booking(self):
        merged = {"gpu_vram_mb": 24000, "time_avg_window": 1}
        check(_frozen(), merged, 19530)
        self.assertEqual(merged["gpu_vram_mb"], 24000)

    def test_fno_is_untouched(self):
        """FNO eval footprint scales with the batch; the training peak is 4-5x too large."""
        merged = {"gpu_vram_mb": 2000, "time_avg_window": 1}
        check({"config_name": "config_flow3D_FNO_pvlhs", "gpu_vram_mb": 20000}, merged, 19530)
        self.assertEqual(merged["gpu_vram_mb"], 2000)

    def test_a_time_averaged_eval_is_untouched(self):
        # Measured: RIGNO steady model trained at 12000, evaluated cleanly at 3000.
        merged = {"gpu_vram_mb": 3000, "time_avg_window": 30}
        check(_frozen(booked=12000), merged, 12000)
        self.assertEqual(merged["gpu_vram_mb"], 3000)

    def test_without_a_measured_peak_it_still_REFUSES(self):
        """Deliberately unchanged: only a MEASUREMENT is safe to auto-apply.

        Raising a booking to a number nothing measured is how one model was forced to a
        figure the box could not free. And the refusal is the protection the origin case
        needed -- calc 61877 recorded no peak -- so it stays, pinned by
        `tests/test_eval_vram_vs_training.py`, which this change deliberately does not touch.
        """
        merged = {"gpu_vram_mb": 8000, "time_avg_window": 1}
        with self.assertRaises(ValueError):
            check(_frozen(booked=14000), merged, None)

    def test_it_no_longer_refuses(self):
        """Refusing cost a whole submission round; the value needed was already in hand."""
        merged = {"gpu_vram_mb": 1000, "time_avg_window": 1}
        try:
            check(_frozen(), merged, 19530)
        except ValueError as exc:
            self.fail(f"still refuses instead of correcting: {exc}")
        self.assertEqual(merged["gpu_vram_mb"], 19530)


if __name__ == "__main__":
    unittest.main()
