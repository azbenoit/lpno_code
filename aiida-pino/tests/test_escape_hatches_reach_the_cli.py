"""A refusal's documented remedy must be reachable through the sanctioned path.

User, 2026-09-07: "any refusal should be possible to bypass". A bypass that exists as a
WorkChain input but has no route from the CLI is not a bypass -- the error message names a
remedy the user cannot apply without hand-submitting the WorkChain, which bypasses the whole
integration, or re-deriving a dataset to make two bookkeeping numbers agree.

Found 2026-09-08 on a case where the geometry genuinely matches.
A model trained on a composed collection whose derivation BAKES crop_z_top=10,
so no read-time crop was requested and `training_crop_z_top` recorded 0 -- true of the
request, misleading about the extent, which was 30-deep. Scoring on an uncropped subset
([90,40,40]) with crop 10 also gives 30-deep. Same extent, verified by grid:

    train    composed  deriv_crop=10   [90, 40, 30]   <- trained on
    test     subset    deriv_crop=None [90, 40, 40]   <- +10 -> [90, 40, 30]

`validate_inputs` compares the recorded REQUEST (0) against the requested (10) and refuses
306. The guard has earned its keep twice, so it stays; what was missing is the way past it.

Two gaps, not one: `EvalOnlyWorkChain` forwarded `allow_training_data` but never declared
`allow_domain_change` at all, and neither reached the CLI. The same defect was already fixed
once for `full_frames`, whose comment in `cli.py` records it verbatim.
"""

from __future__ import annotations

import inspect
import unittest


class WorkChainForwardingTest(unittest.TestCase):
    def test_eval_only_declares_both_escape_hatches(self):
        from aiida_pino.workflows.eval_only import EvalOnlyWorkChain

        names = set(EvalOnlyWorkChain.spec().inputs.keys())
        self.assertIn("allow_training_data", names)
        self.assertIn("allow_domain_change", names)

    def test_it_forwards_them_to_the_child(self):
        from aiida_pino.workflows import eval_only as mod

        src = inspect.getsource(mod)
        self.assertIn('inputs["allow_domain_change"]', src)
        self.assertIn('inputs["allow_training_data"]', src)


class RequestSchemaTest(unittest.TestCase):
    def test_the_request_carries_both(self):
        from aiida_pino.console.schemas import EvalOnlyRequest

        fields = set(EvalOnlyRequest.model_fields)
        self.assertIn("allow_domain_change", fields)
        self.assertIn("allow_training_data", fields)

    def test_they_default_to_off(self):
        """A bypass must never be the default; the guard is right by default."""
        from aiida_pino.console.schemas import EvalOnlyRequest

        req = EvalOnlyRequest(dataset_pk=1, model_pk=2, python_code="x@y")
        self.assertFalse(req.allow_domain_change)
        self.assertFalse(req.allow_training_data)


class CliTest(unittest.TestCase):
    def test_both_flags_exist_on_submit_eval(self):
        from pathlib import Path

        import aiida_pino

        cli = (Path(aiida_pino.__file__).parent / "cli.py").read_text()
        self.assertIn('"--allow-domain-change"', cli)
        self.assertIn('"--allow-training-data"', cli)

    def test_the_service_passes_them_through(self):
        from aiida_pino.console.service import AiidaConsoleService

        src = inspect.getsource(AiidaConsoleService.submit_eval_only)
        self.assertIn("allow_domain_change", src)
        self.assertIn("allow_training_data", src)


if __name__ == "__main__":
    unittest.main()
