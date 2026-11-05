"""GridGradientLoss: pointwise L2 + spatial-gradient mismatch term."""

import unittest

import torch

from lpno.custom_utils.losses import GridGradientLoss


def _mk(shape=(2, 1, 8, 6, 4), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g)


class TestGridGradientLoss(unittest.TestCase):
    def _loss(self, pred, true, w):
        fn = GridGradientLoss(gradient_weight=w)
        lam = {"T": torch.ones_like(pred)}
        return fn({}, {"T": pred}, {"T": true}, lam, step=0)["T"]

    def test_matches_pointwise_l2_when_weight_zero(self):
        true, pred = _mk(seed=1), _mk(seed=2)
        expected = (pred - true).pow(2).sum()
        self.assertTrue(torch.allclose(self._loss(pred, true, 0.0), expected))

    def test_zero_error_gives_zero_loss(self):
        true = _mk()
        self.assertEqual(float(self._loss(true.clone(), true, 1.0)), 0.0)

    def test_constant_offset_adds_no_gradient_penalty(self):
        # A constant bias has identical gradients everywhere: the gradient term
        # must contribute nothing, leaving exactly the pointwise loss.
        true = _mk()
        pred = true + 0.5
        self.assertTrue(
            torch.allclose(self._loss(pred, true, 1.0), self._loss(pred, true, 0.0))
        )

    def test_patchy_error_penalized_more_than_smooth(self):
        # Checkerboard vs constant perturbation with identical pointwise L2:
        # only the gradient term separates them, and it must punish patchiness.
        true = _mk()
        sign = torch.ones(8, 6, 4)
        sign[::2] *= -1.0
        sign[:, ::2] *= -1.0
        sign[:, :, ::2] *= -1.0
        patchy = true + 0.1 * sign
        smooth = true + 0.1
        self.assertTrue(
            torch.allclose(self._loss(patchy, true, 0.0), self._loss(smooth, true, 0.0))
        )
        self.assertGreater(
            float(self._loss(patchy, true, 1.0)), float(self._loss(smooth, true, 1.0))
        )

    def test_edge_downweight_zero_is_identity(self):
        # edge_downweight=0 must reproduce the un-masked loss exactly (backward compat).
        true, pred = _mk(seed=1), _mk(seed=2)
        lam = {"T": torch.ones_like(pred)}
        base = GridGradientLoss(gradient_weight=1.0)({}, {"T": pred}, {"T": true}, lam, 0)["T"]
        off = GridGradientLoss(gradient_weight=1.0, edge_downweight=0.0)(
            {}, {"T": pred}, {"T": true}, lam, 0
        )["T"]
        self.assertTrue(torch.allclose(base, off))

    def test_edge_downweight_lowers_sharp_region_contribution(self):
        # true: flat left (|grad|~0), ramp right (|grad|~1). The SAME-magnitude error
        # placed in the high-gradient region must contribute LESS to the loss than in
        # the smooth region — that is the keyhole-edge down-weighting.
        true = torch.zeros(1, 1, 8, 6, 4)
        true[0, 0, 4:, :, :] = torch.arange(4, dtype=torch.float32).view(4, 1, 1)

        def loss_with_err(x):
            pred = true.clone()
            pred[0, 0, x, 0, 0] += 0.5
            fn = GridGradientLoss(gradient_weight=0.0, edge_downweight=4.0)
            return float(fn({}, {"T": pred}, {"T": true}, {"T": torch.ones_like(true)}, 0)["T"])

        self.assertLess(loss_with_err(5), loss_with_err(1))  # sharp region < smooth region

    def test_multiple_output_keys(self):
        true_t, true_a = _mk(seed=3), _mk(seed=4)
        pred_t, pred_a = _mk(seed=5), _mk(seed=6)
        fn = GridGradientLoss(gradient_weight=1.0)
        lam = {"T": torch.ones_like(true_t), "alpha": torch.ones_like(true_a)}
        out = fn({}, {"T": pred_t, "alpha": pred_a}, {"T": true_t, "alpha": true_a}, lam, step=0)
        self.assertEqual(set(out), {"T", "alpha"})
        for v in out.values():
            self.assertGreater(float(v), 0.0)


if __name__ == "__main__":
    unittest.main()
