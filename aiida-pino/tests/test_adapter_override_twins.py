"""Stored `+` overrides are promoted to `++` on replay, identically in both adapters.

Old models store single-plus overrides that were legal against the packaged config of
their day. Replayed onto the model's own archived `.hydra/config.yaml` -- which bakes
those keys in -- Hydra refuses the append (`An item is already at 'custom.seed'`), so the
era's models were unevaluable through exactly the path built to keep them evaluable.
Reported on one evaluation and verified on the node: `+custom.seed=42`,
`+custom.case_coherent_sampling=true`.
"""

import inspect
import unittest

from aiida_pino.scripts import eval_adapter, train_adapter


class PromotionTest(unittest.TestCase):
    def test_single_plus_is_promoted_and_nothing_else_is_touched(self):
        out = eval_adapter.promote_replayed_overrides(
            ["+custom.seed=42", "++already.double=1", "training.max_steps=5",
             "~scheduler", "+custom.case_coherent_sampling=true"]
        )
        self.assertEqual(out, ["++custom.seed=42", "++already.double=1",
                               "training.max_steps=5", "~scheduler",
                               "++custom.case_coherent_sampling=true"])

    def test_empty_and_none_are_harmless(self):
        self.assertEqual(eval_adapter.promote_replayed_overrides([]), [])
        self.assertEqual(eval_adapter.promote_replayed_overrides(None), [])

    def test_the_exact_stored_overrides_of_m47146_become_replayable(self):
        # The verified real-world payload: after promotion, none carries the bare `+`
        # that Hydra refuses against a config that already declares the key.
        out = eval_adapter.promote_replayed_overrides(
            ["+custom.seed=42", "+custom.case_coherent_sampling=true"]
        )
        self.assertTrue(all(o.startswith("++") for o in out))

    def test_both_command_builders_replay_through_the_promotion(self):
        for module, builder in ((eval_adapter, "build_eval_command"),
                                (train_adapter, "build_training_command")):
            src = inspect.getsource(getattr(module, builder, None) or module)
            self.assertIn('promote_replayed_overrides(profile.get("hydra_overrides"', src,
                          f"{module.__name__} replays raw overrides")

    def test_the_forced_twins_are_byte_identical(self):
        # Adapters are uploaded as single standalone files and cannot import each other on
        # the compute box, so the helper exists twice BY NECESSITY. Twins drift -- that is
        # this repo's oldest lesson -- so the copies are pinned identical here.
        self.assertEqual(
            inspect.getsource(eval_adapter.promote_replayed_overrides),
            inspect.getsource(train_adapter.promote_replayed_overrides),
        )


if __name__ == "__main__":
    unittest.main()
