"""Validator plotters are off by default and enabled by custom.validator_plots."""

import unittest

from omegaconf import OmegaConf

from lpno.models.pipeline.model_domain import _validator_plotter


class ValidatorPlotterGateTest(unittest.TestCase):
    def test_default_is_no_plotter(self):
        cfg = OmegaConf.create({"custom": {"name": "x"}})
        self.assertIsNone(_validator_plotter(cfg, lambda: object()))

    def test_flag_enables_plotter(self):
        cfg = OmegaConf.create({"custom": {"validator_plots": True}})
        sentinel = object()
        self.assertIs(_validator_plotter(cfg, lambda: sentinel), sentinel)

    def test_explicit_false_is_no_plotter(self):
        cfg = OmegaConf.create({"custom": {"validator_plots": False}})
        self.assertIsNone(_validator_plotter(cfg, lambda: object()))

    def test_factory_not_called_when_disabled(self):
        cfg = OmegaConf.create({"custom": {}})
        calls = []
        _validator_plotter(cfg, lambda: calls.append(1))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
