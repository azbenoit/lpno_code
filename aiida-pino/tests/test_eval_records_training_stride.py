"""An eval records the stride its model TRAINED at, beside the one it scored at.

Scoring a model at a resolution it never trained on exits 0 and returns numbers that look
real. Measured 2026-09-06: an FNO trained at [2,2,1] and scored with
the stride blanked read substrate 43-55 K / beam box 145-167 K against a true 11.90 / 45.17 --
a 4x fake regression that ALSO inverted the ranking of the two arms being compared, and was
reported as a model result before being caught. Rescored at the training stride the same
weights give 5.36 K / 13.69 K.

It hides because it is architecture-dependent: RIGNO and GAOT are graph models and absorb the
change, so in a mixed batch two of three architectures reproduce and the third looks broken.

Both numbers already existed -- `training_stride` on the model node, `eval_stride` on the eval
-- and nothing compared them. This records the pair and the comparison so the mismatch is
queryable. It deliberately does NOT refuse: a refusal would change behaviour for grid-bound
and spectral architectures, and was considered and declined.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.model_registry import _model_training_stride


class _Model:
    def __init__(self, value):
        class _Extras:
            all = {"training_stride": value} if value is not None else {}

        self.base = type("B", (), {"extras": _Extras})


class TrainingStrideTest(unittest.TestCase):
    def test_it_reads_the_models_recorded_stride(self):
        self.assertEqual(_model_training_stride(_Model([2, 2, 1])), [2, 2, 1])

    def test_a_model_without_the_extra_is_unknown_not_full_resolution(self):
        """None must not read as [1,1,1]: that would call every old model full-resolution."""
        self.assertIsNone(_model_training_stride(_Model(None)))

    def test_a_malformed_value_is_unknown(self):
        for bad in ("2,2,1", [2, 2], [1, 2, "x"], 2):
            self.assertIsNone(_model_training_stride(_Model(bad)), f"{bad!r} was accepted")


class ExtrasTest(unittest.TestCase):
    def test_the_writer_stamps_the_pair_and_the_comparison(self):
        import inspect

        from aiida_pino.data import pino_model_evaluation as mod

        src = inspect.getsource(mod)
        self.assertIn('node.base.extras.set("training_stride"', src)
        self.assertIn("eval_resolution_matches_training", src)

    def test_the_flag_is_only_set_when_BOTH_are_known(self):
        """An unknown on either side must not produce a False that reads as a mismatch."""
        import inspect

        from aiida_pino.data import pino_model_evaluation as mod

        src = inspect.getsource(mod)
        flag = src.index("eval_resolution_matches_training")
        guard = src.index('if isinstance(trained, (list, tuple)) and len(trained) == 3:')
        self.assertLess(guard, flag, "the flag is not gated on the training stride being known")


if __name__ == "__main__":
    unittest.main()
