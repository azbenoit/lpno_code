"""An env that was never recorded is unknown, not a confound.

`import-model` used to accept a partial profile and store `env: {}`. Comparing such a
model against a native one charged every env key as a confound -- 41 false confounds,
reported 2026-08-27. Absent is not empty.
"""

import unittest

from aiida_pino.utils import model_compare as mc

NATIVE_ENV = {"PINO_LAZY_CROP_Z_TOP": "10", "PINO_LOADER_WORKERS": "2"}


def _rows(fa, fb):
    unknown = mc._env_unrecorded(fa) or mc._env_unrecorded(fb)
    return mc._diff_maps("env", fa["env"], fb["env"], set(),
                         severity="unknown" if unknown else None)


class UnrecordedEnvTest(unittest.TestCase):
    def test_an_import_without_a_recorded_env_yields_unknown_not_confound(self):
        imported = {"env": {}, "has_calcjob": False}
        native = {"env": NATIVE_ENV, "has_calcjob": True}
        rows = _rows(imported, native)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["severity"] for r in rows}, {"unknown"})

    def test_a_native_run_with_a_genuinely_empty_env_still_confounds(self):
        # This is the case the fix must NOT swallow: the run recorded its env and it was
        # empty, so the other side's crop really is a different input geometry.
        native_empty = {"env": {}, "has_calcjob": True}
        native = {"env": NATIVE_ENV, "has_calcjob": True}
        rows = _rows(native_empty, native)
        self.assertTrue(rows)
        self.assertIn("confound", {r["severity"] for r in rows})

    def test_two_imports_that_agree_produce_no_rows_at_all(self):
        a = {"env": {}, "has_calcjob": False}
        self.assertEqual(_rows(a, dict(a)), [])

    def test_facts_records_the_discriminator(self):
        # `has_calcjob` is what separates "never recorded" from "recorded as empty";
        # a comparison that cannot see it cannot make this distinction.
        import inspect
        self.assertIn('"has_calcjob"', inspect.getsource(mc._facts))

    def test_the_verdict_treats_unknown_as_unverified_rather_than_measured(self):
        # The severity is only useful because the verdict already distinguishes it.
        import inspect
        src = inspect.getsource(mc.compare_models)
        self.assertIn('severity"] == "unknown"', src)
        self.assertIn("cannot be verified equal", src)


if __name__ == "__main__":
    unittest.main()
