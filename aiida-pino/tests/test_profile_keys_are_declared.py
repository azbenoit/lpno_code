"""Every key the profiles module READS must be DECLARED in DEFAULT_TRAINING_PROFILE.

`TRAINING_PROFILE_KEYS = frozenset(DEFAULT_TRAINING_PROFILE)` is what the submit path
validates against, so a key that is read but not declared makes its own feature
unreachable: any submission naming it is refused as "unknown training_profile key(s)",
and the code reading it therefore only ever sees the default.

Three instances of this have been found in this one file:

* `gpu_seat_priority` -- "the key was only ever declared on the EVALUATION profile, so
  the console's unknown-key check refused it on a training and the working code was
  unreachable" (recorded in the module itself).
* `graph_nodes` (2026-08-26) -- `_check_host_ram_against_loader_settings` scaled the
  full-resolution host-RAM floor by `profile.get('graph_nodes')`, but the key was never
  declared, so the grid-aware branch could not execute and every run inherited the flat
  40960 anchor measured on a 600k-node GAOT build. A cropped 90x40x30 arm with a
  measured RSS of 8188 MiB was made to book 40960, capping a 128 GB box at two
  concurrent runs because host RAM is booked machine-wide.
* The same defect also corrupted a MEASUREMENT: "of 446 stored profiles, ZERO declare
  both a stride and `graph_nodes`" was read as "this combination is rare" when it
  actually meant "the check rejected all of them". A query cannot see what a validator
  refuses.

The failure is silent in the worst way -- no error at read time, just a default -- so it
is pinned structurally rather than case by case.
"""

import ast
import unittest
from pathlib import Path

from aiida_pino.utils.profiles import DEFAULT_TRAINING_PROFILE

MODULE = Path(__file__).resolve().parents[1] / "src" / "aiida_pino" / "utils" / "profiles.py"

#: Locals that hold a training profile in this module.
PROFILE_NAMES = {"profile", "training_profile"}

#: Keys read off a profile-shaped mapping that are legitimately not profile keys.
#: Keep this empty unless there is a real reason; each entry is a hole in the check.
ALLOWED_UNDECLARED: set[str] = set()


def _keys_read_from_a_profile(tree: ast.AST) -> set[str]:
    """String keys fetched off a variable named like a training profile."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in PROFILE_NAMES
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.add(node.args[0].value)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in PROFILE_NAMES
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            found.add(node.slice.value)
    return found


class TestProfileKeysAreDeclared(unittest.TestCase):
    def test_the_module_is_readable(self):
        self.assertTrue(MODULE.is_file(), f"profiles.py not found at {MODULE}")

    def test_the_scan_finds_something(self):
        """A scan that silently matches nothing would pass for the wrong reason."""
        keys = _keys_read_from_a_profile(ast.parse(MODULE.read_text()))
        self.assertGreater(
            len(keys), 10,
            "the AST scan found almost no profile reads -- the access pattern in "
            "profiles.py has probably changed and this test is no longer checking anything",
        )

    def test_every_key_read_is_declared(self):
        keys = _keys_read_from_a_profile(ast.parse(MODULE.read_text()))
        undeclared = sorted(k for k in keys if k not in DEFAULT_TRAINING_PROFILE and k not in ALLOWED_UNDECLARED)
        self.assertEqual(
            [], undeclared,
            "these keys are read off a training profile but are not in "
            "DEFAULT_TRAINING_PROFILE, so TRAINING_PROFILE_KEYS refuses any submission "
            "that sets them and the code reading them can only ever see the default:\n  "
            + "\n  ".join(undeclared),
        )

    def test_graph_nodes_specifically(self):
        """The instance that motivated this test, pinned by name."""
        self.assertIn("graph_nodes", DEFAULT_TRAINING_PROFILE)

    def test_gpu_seat_priority_specifically(self):
        """The earlier instance the module documents."""
        self.assertIn("gpu_seat_priority", DEFAULT_TRAINING_PROFILE)


if __name__ == "__main__":
    unittest.main()
