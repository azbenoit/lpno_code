"""A remote eval must fail BEFORE the CalcJob when its collection is not staged.

A remote eval that derives a new collection at submit time can never satisfy the staging
check, because the collection is created by the submission itself. Reproduced on one
submission (2026-09-06), timestamps one second apart:

    90701  EvalOnlyWorkChain           16:28:58   the submission
    90704  materialize_derived_dataset 16:29:01   derives the collection
    90705  PinoDatasetCollectionData   16:29:03   CREATED BY 90701 ITSELF
    90707  PinoEvalCalculation         16:29:04   excepts: 90705 must already be staged

The message was already right -- it even says "this is a DERIVED collection ... it could not
have been staged in advance" and prints a runnable `stage-remote` line. What was wrong is
WHEN: it raises inside CalcJob `presubmit`, so the failure arrives as an engine traceback on a
job node that should never have been created, and the parent reports the generic
ERROR_EVALUATION_FAILED (302) that hides 234 heterogeneous failures.

Checked before submitting, the same message becomes a clean exit code, and the pk it names now
EXISTS and can be staged -- 90705 was created two seconds before the job that demanded it.

The check must ask the SAME question the CalcJob asks. `aiida-pino staged --computer X` lists
BASE datasets, so a submitter can confirm their dataset appears there, submit, and still die
on the derivation+fields key -- which is how one campaign submitted wrong twice.
This calls the CalcJob's own `remote_dataset_symlink_entry` with fields from the CalcJob's own
`runtime_fields_from_training_profile`, so the two cannot drift apart.
"""

from __future__ import annotations

import inspect
import unittest


class PreSubmitCheckTest(unittest.TestCase):
    def test_the_workchain_checks_before_it_submits(self):
        from aiida_pino.workflows import evaluate_model as mod

        src = inspect.getsource(mod.EvaluateModelWorkChain.run_evaluation)
        self.assertIn("remote_dataset_symlink_entry", src, "no pre-submit staging check")
        check = src.index("remote_dataset_symlink_entry")
        submit = src.index("self.submit(")
        self.assertLess(check, submit, "the check runs after the CalcJob is created")

    def test_it_asks_the_same_question_the_calcjob_asks(self):
        """Fields from the same helper, so the check cannot drift into a weaker question."""
        from aiida_pino.workflows import evaluate_model as mod

        src = inspect.getsource(mod.EvaluateModelWorkChain.run_evaluation)
        self.assertIn("runtime_fields_from_training_profile", src)

    def test_it_only_applies_to_a_remote_computer(self):
        """A local job builds its cache at submit time; refusing it would be over-strict."""
        from aiida_pino.workflows import evaluate_model as mod

        src = inspect.getsource(mod.EvaluateModelWorkChain.run_evaluation)
        self.assertIn("pack_index_allowed_for_transport", src)

    def test_there_is_a_dedicated_exit_code(self):
        """Not the generic 302, which covers 234 heterogeneous failures."""
        from aiida_pino.workflows.evaluate_model import EvaluateModelWorkChain

        spec = EvaluateModelWorkChain.spec()
        codes = {c.status: c.message for c in spec.exit_codes.values()}
        self.assertIn(307, codes)
        self.assertIn("stage", str(codes[307]).lower())


if __name__ == "__main__":
    unittest.main()
