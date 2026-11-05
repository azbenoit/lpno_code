"""Validators report absolute MAE + RMSE alongside l2_relative_error."""

import unittest

import torch

# Import triggers the module-level metric patch.
from lpno.models.pipeline import model_domain  # noqa: F401
from physicsnemo.sym.domain.validator import GridValidator, PointwiseValidator


class ValidatorMaeMetricTest(unittest.TestCase):
    def _check(self, cls):
        true = {"T": torch.tensor([[1.0], [2.0], [3.0]])}
        pred = {"T": torch.tensor([[1.5], [2.0], [2.0]])}  # abs errs 0.5, 0, 1.0
        out = cls._l2_relative_error(true, pred)
        self.assertIn("l2_relative_error_T", out)  # original kept
        self.assertAlmostEqual(out["mae_T"].item(), 0.5, places=5)
        self.assertAlmostEqual(out["rmse_T"].item(), (1.25 / 3) ** 0.5, places=5)

    def test_grid_validator(self):
        self._check(GridValidator)

    def test_pointwise_validator(self):
        self._check(PointwiseValidator)


if __name__ == "__main__":
    unittest.main()
