"""Melt-focus loss terms: soft-Dice on the pool/cavity, and melt-region weighting.

Both are gated by a step warmup so the term only switches on once the model
already predicts a rough melt pool -- before that the gradient of a region
overlap term points nowhere useful. With the weights at zero the loss must be
bit-for-bit what it was, because every existing run depends on that.
"""

import unittest

import torch

from lpno.custom_utils.losses import GridGradientLoss

T_S, T_L = 0.6, 0.7  # normalized solidus/liquidus (T/ref_scales["T"])


def _pool(shape=(1, 1, 8, 6, 4), hot=(slice(2, 6), slice(2, 4), slice(2, 4)), t=0.9):
    """Cold field at 0.2 with a hot (molten) block at ``t``."""
    field = torch.full(shape, 0.2)
    field[(slice(None), slice(None)) + hot] = t
    return field


def _cavity(shape=(1, 1, 8, 6, 4), hole=(slice(2, 6), slice(2, 4), slice(2, 4))):
    """alpha=1 metal with a gas cavity (alpha=0) block."""
    field = torch.ones(shape)
    field[(slice(None), slice(None)) + hole] = 0.0
    return field


def _loss(fn, pred, true, step, key="T"):
    lam = {key: torch.ones_like(pred)}
    return float(fn({}, {key: pred}, {key: true}, lam, step=step)[key])


class TestMeltFocusLoss(unittest.TestCase):
    def test_zero_weights_reproduce_the_plain_loss(self):
        true, pred = _pool(), _pool(hot=(slice(3, 7), slice(2, 4), slice(2, 4)))
        plain = GridGradientLoss(gradient_weight=1.0)
        withz = GridGradientLoss(
            gradient_weight=1.0, melt_dice_weight=0.0, melt_weight=0.0,
            t_solidus=T_S, t_liquidus=T_L,
        )
        self.assertEqual(_loss(plain, pred, true, 0), _loss(withz, pred, true, 0))

    def test_term_is_off_before_warmup_and_on_after(self):
        true, pred = _pool(), _pool(hot=(slice(3, 7), slice(2, 4), slice(2, 4)))
        plain = GridGradientLoss(gradient_weight=0.0)
        fn = GridGradientLoss(
            gradient_weight=0.0, melt_dice_weight=1.0, warmup_steps=100, ramp_steps=0,
            t_solidus=T_S, t_liquidus=T_L,
        )
        baseline = _loss(plain, pred, true, 0)
        self.assertEqual(_loss(fn, pred, true, 99), baseline)
        self.assertGreater(_loss(fn, pred, true, 100), baseline)

    def test_ramp_is_linear_between_warmup_and_full(self):
        true, pred = _pool(), _pool(hot=(slice(3, 7), slice(2, 4), slice(2, 4)))
        plain = GridGradientLoss(gradient_weight=0.0)
        fn = GridGradientLoss(
            gradient_weight=0.0, melt_dice_weight=1.0, warmup_steps=100, ramp_steps=100,
            t_solidus=T_S, t_liquidus=T_L,
        )
        base = _loss(plain, pred, true, 0)
        half = _loss(fn, pred, true, 150) - base
        full = _loss(fn, pred, true, 200) - base
        self.assertAlmostEqual(half, 0.5 * full, places=5)
        # and it saturates rather than growing past 1
        self.assertAlmostEqual(_loss(fn, pred, true, 10_000) - base, full, places=5)

    def test_dice_is_zero_for_a_perfect_pool(self):
        true = _pool()
        plain = GridGradientLoss(gradient_weight=0.0)
        fn = GridGradientLoss(
            gradient_weight=0.0, melt_dice_weight=1.0, t_solidus=T_S, t_liquidus=T_L,
        )
        self.assertAlmostEqual(
            _loss(fn, true.clone(), true, 0), _loss(plain, true.clone(), true, 0), places=6
        )

    def test_dice_grows_with_pool_displacement(self):
        """A pool off by two cells must cost more than one off by one."""
        true = _pool()
        near = _pool(hot=(slice(3, 7), slice(2, 4), slice(2, 4)))
        far = _pool(hot=(slice(4, 8), slice(2, 4), slice(2, 4)))
        fn = GridGradientLoss(
            gradient_weight=0.0, melt_dice_weight=1.0, t_solidus=T_S, t_liquidus=T_L,
        )
        plain = GridGradientLoss(gradient_weight=0.0)
        d_near = _loss(fn, near, true, 0) - _loss(plain, near, true, 0)
        d_far = _loss(fn, far, true, 0) - _loss(plain, far, true, 0)
        self.assertGreater(d_far, d_near)
        self.assertGreater(d_near, 0.0)

    def test_dice_on_the_alpha_cavity(self):
        true = _cavity()
        pred = _cavity(hole=(slice(2, 6), slice(2, 4), slice(1, 4)))  # one cell deeper
        fn = GridGradientLoss(gradient_weight=0.0, melt_dice_weight=1.0, melt_tau=0.05)
        plain = GridGradientLoss(gradient_weight=0.0)
        self.assertGreater(
            _loss(fn, pred, true, 0, key="alpha"), _loss(plain, pred, true, 0, key="alpha")
        )

    def test_melt_weight_upweights_error_inside_the_pool(self):
        """Same squared error, inside the melt vs outside it."""
        true = _pool()
        inside, outside = true.clone(), true.clone()
        inside[0, 0, 3, 2, 2] += 0.1   # in the hot block
        outside[0, 0, 0, 0, 0] += 0.1  # in the cold far field
        fn = GridGradientLoss(
            gradient_weight=0.0, melt_weight=4.0, t_solidus=T_S, t_liquidus=T_L,
        )
        self.assertGreater(_loss(fn, inside, true, 0), _loss(fn, outside, true, 0))
        plain = GridGradientLoss(gradient_weight=0.0)
        self.assertAlmostEqual(
            _loss(plain, inside, true, 0), _loss(plain, outside, true, 0), places=6
        )

    def test_melt_weight_respects_the_warmup(self):
        true = _pool()
        pred = true.clone()
        pred[0, 0, 3, 2, 2] += 0.1
        fn = GridGradientLoss(
            gradient_weight=0.0, melt_weight=4.0, warmup_steps=50,
            t_solidus=T_S, t_liquidus=T_L,
        )
        plain = GridGradientLoss(gradient_weight=0.0)
        self.assertEqual(_loss(fn, pred, true, 49), _loss(plain, pred, true, 49))
        self.assertGreater(_loss(fn, pred, true, 50), _loss(plain, pred, true, 50))

    def test_alpha_is_weighted_by_the_truth_melt_region(self):
        """The melt weight comes from true T, so it also applies to alpha."""
        t_true = _pool()
        a_true = torch.ones_like(t_true)
        a_pred = a_true.clone()
        a_pred[0, 0, 3, 2, 2] -= 0.1  # error inside the pool
        fn = GridGradientLoss(
            gradient_weight=0.0, melt_weight=4.0, t_solidus=T_S, t_liquidus=T_L,
        )
        lam = {"alpha": torch.ones_like(a_pred)}
        weighted = float(
            fn({}, {"alpha": a_pred}, {"alpha": a_true, "T": t_true}, lam, step=0)["alpha"]
        )
        plain = float((a_pred - a_true).pow(2).sum())
        self.assertGreater(weighted, plain * 1.5)


if __name__ == "__main__":
    unittest.main()
