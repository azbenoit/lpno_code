"""`staged --check` must answer the question SUBMISSION asks, not a weaker one.

`aiida-pino staged --computer X` lists the runtime caches present there, and its help said
"check BEFORE submitting there". One submitter did exactly that: ran the listing, saw
their dataset uuids, concluded "staged", and submitted twice into a failure (2026-09-06).

The listing cannot answer the question. What a remote job consumes is not the dataset that was
passed -- it is the collection the derivation profile resolves to, keyed by
`cache_discriminator_for(node, runtime_fields, transport_type)` and `runtime_cache_name`. A
base dataset being present says nothing about whether THAT key is.

Same shape as reading `derivation_profile.stride` when the answer lives in `eval_stride`, and
the reporter noted it is the same shape as their own FNO mistake: the query they ran ("is the
uuid present") was a weaker claim than the one they needed, and it read as confirmation. That
is the expensive failure mode -- a check that returns a confident wrong answer to a human,
rather than a wrong default a guard catches.

Three states, and the third is the one that had no answer at all: a derivation that resolves
to no existing collection CANNOT be pre-staged, because submission will create it. Saying so
is the difference between an impossible instruction and a plan.
"""

from __future__ import annotations

import inspect
import unittest


class StagedCheckTest(unittest.TestCase):
    def test_the_service_exposes_a_check(self):
        from aiida_pino.console.service import AiidaConsoleService

        self.assertTrue(hasattr(AiidaConsoleService, "staged_check"))

    def test_it_resolves_the_derivation_rather_than_trusting_the_dataset(self):
        """The base dataset being staged is the weaker claim that caused the failure."""
        from aiida_pino.console.service import AiidaConsoleService

        src = inspect.getsource(AiidaConsoleService.staged_check)
        self.assertIn("find_reusable_derived_dataset", src)

    def test_it_keys_on_the_same_discriminator_submission_uses(self):
        from aiida_pino.console.service import AiidaConsoleService

        src = inspect.getsource(AiidaConsoleService.staged_check)
        self.assertIn("cache_discriminator_for", src)
        self.assertIn("runtime_cache_name", src)
        self.assertIn("find_remote_runtime_cache", src)

    def test_it_reports_the_would_be_created_case_distinctly(self):
        """A derivation resolving to nothing cannot be pre-staged; that is not 'not staged'."""
        from aiida_pino.console.service import AiidaConsoleService

        src = inspect.getsource(AiidaConsoleService.staged_check)
        self.assertIn("would_be_created", src)

    def test_the_cli_exposes_check_derivation_and_fields(self):
        from pathlib import Path

        import aiida_pino

        cli = (Path(aiida_pino.__file__).parent / "cli.py").read_text()
        self.assertIn('"--check"', cli)
        self.assertIn('"--derivation"', cli)
        self.assertIn("staged_check", cli)

    def test_the_listing_help_no_longer_claims_to_answer_it(self):
        """"check BEFORE submitting" is what the reporter reasonably relied on."""
        from pathlib import Path

        import aiida_pino

        cli = (Path(aiida_pino.__file__).parent / "cli.py").read_text()
        block = cli[cli.index('"staged"'):cli.index('"compare"')]
        self.assertNotIn("check BEFORE submitting there", block)


class ImportsResolveTest(unittest.TestCase):
    """Every deferred import inside `staged_check` must actually resolve.

    The source-inspection tests above all passed while the function raised ImportError on its
    first call: `find_remote_runtime_cache` lives in `utils.data_nodes`, not `stage_remote`.
    A test that reads source cannot catch that, and this check is meant to be run by a human
    BEFORE submitting, so a first-call crash would be the worst possible moment.
    """

    def test_the_deferred_imports_exist(self):
        import importlib
        import inspect
        import re

        from aiida_pino.console.service import AiidaConsoleService

        src = inspect.getsource(AiidaConsoleService.staged_check)
        pairs = re.findall(r"from ([\w.]+) import \(([^)]*)\)|from ([\w.]+) import ([\w, ]+)", src)
        checked = 0
        for grouped_mod, grouped_names, plain_mod, plain_names in pairs:
            module = grouped_mod or plain_mod
            names = (grouped_names or plain_names).replace("\n", " ")
            mod = importlib.import_module(module)
            for name in [n.strip().rstrip(",") for n in names.split(",") if n.strip()]:
                self.assertTrue(hasattr(mod, name), f"{module} has no {name}")
                checked += 1
        self.assertGreater(checked, 4, "the import scan matched nothing; the regex is wrong")


if __name__ == "__main__":
    unittest.main()
