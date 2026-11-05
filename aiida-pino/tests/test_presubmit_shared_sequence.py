"""The submission sequence exists ONCE, and the dry-run runs the same copy.

`TrainModelWorkChain.validate_inputs` used to hold the only complete statement of the
resolution sequence, which made a presubmit dry-run impossible without duplicating it --
the duplicated-builder trap that hid the preheat bug for a week. Now
`resolve_training_submission` is the single copy and both callers delegate.
"""

import inspect
import unittest

from aiida_pino.utils import presubmit
from aiida_pino.workflows import train_model


class SingleCopyTest(unittest.TestCase):
    def test_the_workchain_delegates_instead_of_duplicating(self):
        src = inspect.getsource(train_model.TrainModelWorkChain.validate_inputs)
        self.assertIn("resolve_training_submission", src)
        # The inline copies must be GONE, not merely shadowed: these calls are the
        # sequence, and their presence in the workchain body would be the second copy.
        for marker in ("demote_for_code", "layouts_compatible",
                       "find_reusable_derived_dataset", "split_violation_message"):
            self.assertNotIn(marker, src, f"{marker} still inline in validate_inputs")
        self.assertFalse(hasattr(train_model.TrainModelWorkChain, "_resolve_source"),
                         "_resolve_source is the old second copy of the resolution")

    def test_every_problem_kind_maps_to_a_real_exit_code(self):
        wc = train_model.TrainModelWorkChain
        kinds = {getattr(presubmit, name) for name in dir(presubmit) if name.isupper()
                 and isinstance(getattr(presubmit, name), str)}
        mapped = set(wc._PROBLEM_EXIT)
        self.assertEqual(kinds, mapped, "problem kinds and the exit-code map disagree")
        spec = wc.spec()
        for exit_name in wc._PROBLEM_EXIT.values():
            self.assertIn(exit_name, spec.exit_codes, exit_name)

    def test_the_dry_run_never_materializes(self):
        src = inspect.getsource(presubmit.resolve_training_submission)
        # materialize_derived_dataset must be reachable ONLY behind the flag.
        before, sep, after = src.partition("if not materialize:")
        self.assertTrue(sep, "the materialize gate is gone")
        self.assertNotIn("materialize_derived_dataset", before)
        self.assertIn("materialize_derived_dataset", after)

    def test_staging_report_answers_dont_know_rather_than_guessing(self):
        # A dry-run that would materialize has no resolved node; staging for it cannot be
        # checked, and the report must say so instead of answering staged/not-staged.
        report = presubmit.staging_report({"resolved": None}, _RemoteCode())
        self.assertTrue(report["needed"])
        self.assertIsNone(report["staged"])

    def test_a_local_computer_still_pays_for_an_unbuilt_cache(self):
        """Changed 2026-09-11. This asserted "local needs no staging", which was the misleading
        sentence: a local job whose collection is unresolved (would materialise) or has no
        runtime cache EXPORTS it inside `do_upload`, GPUs idle, before it can start. The
        report now says so; nothing refuses on it (only `submit_train` prints it as a note).
        """
        report = presubmit.staging_report({"resolved": None}, _LocalCode())
        self.assertTrue(report["needed"])
        self.assertIn("upload", report["detail"])


class _LocalComputer:
    transport_type = "core.local"
    label = "localhost"


class _RemoteComputer:
    transport_type = "core.ssh_async"
    label = "somewhere"


class _LocalCode:
    computer = _LocalComputer()


class _RemoteCode:
    computer = _RemoteComputer()


if __name__ == "__main__":
    unittest.main()
