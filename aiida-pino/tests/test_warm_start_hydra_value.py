"""A donor value must serialise to an override Hydra reconstructs as that value.

`hydra_value` bracketed real lists and fell through to `str(value)` for everything else.
physicsnemo's archived config stores `arch.fno.input_keys` as the STRING
`'x, y, z, V_scan, P'` (verified on donor 68423; `arch.decoder.input_keys` beside it is a
real list, which is why the feature landed green), so inheritance emitted the bare

    arch.fno.input_keys=x, y, z, V_scan, P

That is not a syntax error to Hydra -- its parser reads it as a **ChoiceSweep** of five
values, i.e. a multirun sweep, and a single run then refuses it as ambiguous. Every FNO warm
start died at step 0, AFTER taking a GPU seat and loading the donor weights, so the cost was
a seat plus a queue wait (2026-09-03).
RIGNO donors were unaffected: their inherited values are lists or scalars.

Quoting is used rather than splitting on commas, because the value must survive as the STRING
the donor ran -- splitting would silently retype it, and a comma-bearing string that is not a
list would be corrupted. Measured against Hydra's own parser: quoted round-trips to the
identical str; bare gives a ChoiceSweep. The command is passed to `subprocess.Popen` as an
argv list with no shell, so the quotes reach Hydra rather than being eaten by a shell.

Scalar-literal strings (`"true"`, `"3"`, `"null"`) also fail to round-trip -- they arrive as
bool/int/None -- but they yield a SCALAR the target coerces, they are not this defect, and
re-typing them would change overrides that work today. Deliberately left alone.
"""

from __future__ import annotations

import unittest

from aiida_pino.utils.warm_start import hydra_value

try:
    from hydra.core.override_parser.overrides_parser import OverridesParser

    _PARSER = OverridesParser.create()
except Exception:  # pragma: no cover - hydra present in this environment
    _PARSER = None


def _roundtrip(value):
    """What Hydra reconstructs from `hydra_value(value)`."""
    return _PARSER.parse_overrides([f"k={hydra_value(value)}"])[0].value()


class CommaStringTest(unittest.TestCase):
    def test_the_donor_string_survives_as_that_string(self):
        # The exact value stored on donor 68423.
        value = "x, y, z, V_scan, P"
        self.assertEqual(_roundtrip(value), value)

    def test_it_is_not_silently_retyped_into_a_list(self):
        """Splitting on commas would pass this test's sibling and corrupt a real string."""
        self.assertIsInstance(_roundtrip("x, y, z, V_scan, P"), str)

    def test_a_bare_comma_string_would_have_been_a_sweep(self):
        # Pins the mechanism, so a future "simplification" back to str() fails loudly here
        # rather than at step 0 of a warm start that already holds a seat.
        from hydra.core.override_parser.types import ChoiceSweep

        bare = _PARSER.parse_overrides(["k=x, y, z, V_scan, P"])[0].value()
        self.assertIsInstance(bare, ChoiceSweep)

    def test_container_and_punctuation_strings_survive(self):
        for value in ("a,b", "[1,2]", "{a:1}", "a=b", "a(b)", "it's", 'say "hi"'):
            self.assertEqual(_roundtrip(value), value, f"{value!r} did not round-trip")


class UnchangedBehaviourTest(unittest.TestCase):
    def test_real_lists_are_still_bracketed_without_spaces(self):
        self.assertEqual(hydra_value(["z", 32]), "[z,32]")
        self.assertEqual(hydra_value([20, 18, 8]), "[20,18,8]")

    def test_bools_and_none_are_unchanged(self):
        self.assertEqual(hydra_value(True), "true")
        self.assertEqual(hydra_value(False), "false")
        self.assertEqual(hydra_value(None), "null")

    def test_plain_strings_are_not_quoted(self):
        # Quoting everything would retype "true"/"3" and change overrides that work today.
        for value in ("plain", "a_b", "a-b", "a.b", "a/b", "a:b", "true", "3"):
            self.assertEqual(hydra_value(value), value)

    def test_a_list_of_comma_strings_quotes_its_items(self):
        self.assertEqual(_roundtrip(["a,b", "c"]), ["a,b", "c"])


if __name__ == "__main__":
    unittest.main()
