"""`eval_stride` must describe the grid the eval SERVED, not the one it requested.

`effective_stride` composes baked x loader by multiplication, which is right: a pre-strided
[2,2,1] copy read with `PINO_LAZY_STRIDE=2,2,1` still on really does serve [4,4,1]. But
`_eval_effective_stride` fed it the REQUESTED `derivation_profile` instead of the stride the
served dataset actually bakes. When no explicit derivation profile is passed, the eval builds
one FROM the model's training env -- so the same stride appears on both sides and multiplies
with itself.

Measured 2026-09-05: 20 evals since 2026-08-01 are labelled [4,4,1] from a model whose env is
`2,2,1`. They were not served that. Counting the cells settles it: two evals
labelled [4,4,1] report `beambox_n` 88552, identical to two evals
(labelled [2,2,1]) on the same split, against 397546 at [1,1,1]. A genuine 16x decimation
would read ~22k. `eval_stride` is COMPUTED and `beambox_n` is COUNTED, so the count decides.

The consequence was a label, not a wrong number -- but the label is what league tables and
"has this checkpoint been scored at two resolutions?" read, which is the question the field
was added for.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.model_registry import _eval_effective_stride


class _Model:
    def __init__(self, env):
        self._env = dict(env)

    def get_model_metadata(self):
        return {"training_profile": {"env": dict(self._env)}}


class _Dataset:
    """Only what the stride label needs: what the served arrays bake."""

    def __init__(self, baked):
        self._baked = baked

    def get_dataset_metadata(self):
        return {"derivation_profile": {"stride": list(self._baked)} if self._baked else {}}


STRIDING_MODEL = {"PINO_LAZY_STRIDE": "2,2,1"}


class ServedNotRequestedTest(unittest.TestCase):
    def test_a_full_res_dataset_read_with_a_loader_stride_is_2(self):
        """The 20 mislabelled evals: requested [2,2,1] built FROM the env, dataset full-res."""
        got = _eval_effective_stride(
            {}, _Model(STRIDING_MODEL), {"stride": [2, 2, 1]}, _Dataset([1, 1, 1])
        )
        self.assertEqual(list(got), [2, 2, 1], "the requested stride was composed with itself")

    def test_a_baked_dataset_whose_loader_stride_was_cleared_is_2(self):
        # The bake path: the variant carries the decimation, the env was popped.
        got = _eval_effective_stride({}, _Model({}), {"stride": [2, 2, 1]}, _Dataset([2, 2, 1]))
        self.assertEqual(list(got), [2, 2, 1])

    def test_a_baked_variant_read_with_a_stale_loader_stride_is_NOT_doubled(self):
        """They do not compose on this path, and the cells prove it.

        `apply_dataset_layout_to_training_profile` pops the loader stride whenever the served
        variant bakes one, so the decimation is applied once. One eval read a derived set (baked
        [2,2,1]) with `PINO_LAZY_STRIDE=2,2,1` still in its profile and counted `beambox_n`
        88552 -- identical to an eval that read the FULL-RES subset with the same env,
        and against 397546 at [1,1,1]. A real [4,4,1] would be ~22k. An earlier version asserted [4,4,1]
        here from the composition's docstring; the count says otherwise and the count wins.
        """
        got = _eval_effective_stride(
            {}, _Model(STRIDING_MODEL), {"stride": [2, 2, 1]}, _Dataset([2, 2, 1])
        )
        self.assertEqual(list(got), [2, 2, 1], "the popped loader stride was composed anyway")

    def test_full_resolution_stays_full_resolution(self):
        got = _eval_effective_stride({}, _Model({}), {"stride": [1, 1, 1]}, _Dataset([1, 1, 1]))
        self.assertEqual(list(got), [1, 1, 1])

    def test_an_explicit_eval_env_overrides_the_inherited_one(self):
        # The documented escape hatch: the evaluation profile's own env wins.
        got = _eval_effective_stride(
            {"env": {"PINO_LAZY_STRIDE": "1,1,1"}}, _Model(STRIDING_MODEL),
            {"stride": [1, 1, 1]}, _Dataset([1, 1, 1]),
        )
        self.assertEqual(list(got), [1, 1, 1])

    def test_an_unreadable_dataset_does_not_invent_full_resolution(self):
        class _Broken:
            def get_dataset_metadata(self):
                raise RuntimeError("unreadable")

        self.assertIsNone(_eval_effective_stride({}, _Model(STRIDING_MODEL), {}, _Broken()))


class BlankedAndDroppedStrideTest(unittest.TestCase):
    """Two ways a stride stops applying, both of which the label missed.

    One eval (created after the corrected rule was live) served FULL
    RESOLUTION and was labelled [2,2,1]. Two counted quantities say full-res -- `beambox_n`
    397546, matching the manuscript row on the same split where strided reads 88552, and
    `true_T_max_K` 4118 against 3853 on every strided eval of that model -- against one
    computed label (2026-09-05).

    Cause 1: its evaluation profile sets `PINO_LAZY_STRIDE: ''`, an explicit "no stride", and
    the recorder's `if not (...).strip()` treated PRESENT-but-blank as ABSENT and fell back to
    the model's `2,2,1`. Absence is not a value, here too.

    Cause 2: the full-res opt-in drops the inherited stride inside the workchain's merged
    profile, and the recorder is handed the RAW evaluation profile, so it never learns the
    drop happened. A profile built from a striding model always carries that model's stride,
    so derivation [1,1,1] against a striding model can only mean the request was explicit --
    which is what the opt-in acts on.
    """

    def test_a_blanked_eval_stride_means_none_not_inherit(self):
        got = _eval_effective_stride(
            {"env": {"PINO_LAZY_STRIDE": ""}}, _Model(STRIDING_MODEL),
            {"stride": [1, 1, 1]}, _Dataset([1, 1, 1]),
        )
        self.assertEqual(list(got), [1, 1, 1], "a blanked stride fell through to the model env")

    def test_the_effective_env_reports_what_the_run_actually_had(self):
        """The opt-in path: the workchain dropped the stride, so the run's env has none."""
        got = _eval_effective_stride(
            {}, _Model(STRIDING_MODEL), {"stride": [1, 1, 1]}, _Dataset([1, 1, 1]),
            effective_env={"PINO_LAZY_CACHE": "4"},
        )
        self.assertEqual(list(got), [1, 1, 1])

    def test_without_an_effective_env_history_is_not_assumed_full_res(self):
        """The inference this replaced would have been wrong for every pre-opt-in eval.

        One measured eval has exactly this shape -- explicit derivation [1,1,1], a striding model, no
        blanked env -- and it COUNTED 88552, the strided view, because the opt-in did not yet
        exist when it ran. Inferring [1,1,1] from the shape would corrupt it on any recompute.
        """
        got = _eval_effective_stride(
            {}, _Model(STRIDING_MODEL), {"stride": [1, 1, 1]}, _Dataset([1, 1, 1])
        )
        self.assertEqual(list(got), [2, 2, 1])

    def test_a_profile_built_from_a_striding_model_still_reads_strided(self):
        # Built-from-training carries the model's own stride, so this is NOT the opt-in.
        got = _eval_effective_stride(
            {}, _Model(STRIDING_MODEL), {"stride": [2, 2, 1]}, _Dataset([1, 1, 1])
        )
        self.assertEqual(list(got), [2, 2, 1])

    def test_an_eval_profile_that_sets_its_own_stride_still_wins(self):
        got = _eval_effective_stride(
            {"env": {"PINO_LAZY_STRIDE": "3,3,1"}}, _Model(STRIDING_MODEL),
            {"stride": [1, 1, 1]}, _Dataset([1, 1, 1]),
        )
        self.assertEqual(list(got), [3, 3, 1], "an explicit eval stride must not be dropped")


class ExplicitFullResolutionTest(unittest.TestCase):
    """The opt-in half: an explicit [1,1,1] request must clear the inherited loader stride."""

    def test_the_workchain_drops_an_inherited_stride_for_an_explicit_full_res_request(self):
        import inspect

        from aiida_pino.workflows import evaluate_model as mod

        src = inspect.getsource(mod.EvaluateModelWorkChain.run_evaluation)
        self.assertIn('"derivation_profile" in self.inputs', src,
                      "an inherited stride must only be dropped for an EXPLICIT request")
        self.assertIn('env.pop("PINO_LAZY_STRIDE", None)', src)
        self.assertIn("requested == [1, 1, 1]", src)

    def test_a_profile_built_from_training_is_left_alone(self):
        """Built-from-training carries the model's own stride and means 'score as trained'."""
        import inspect

        from aiida_pino.workflows import evaluate_model as mod

        src = inspect.getsource(mod.EvaluateModelWorkChain.run_evaluation)
        drop = src.index('env.pop("PINO_LAZY_STRIDE", None)')
        guard = src.index('"derivation_profile" in self.inputs')
        self.assertLess(guard, drop, "the drop is not gated on an explicit request")


if __name__ == "__main__":
    unittest.main()
