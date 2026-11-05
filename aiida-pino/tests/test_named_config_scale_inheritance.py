"""The shipped-config Kelvin scale must follow Hydra's `defaults:` chain.

Reading only the named file finds nothing for a config that inherits its scales, and most
do. Measured on the live monitor 2026-08-27: six running trainings had a beam-box MAE and
no Kelvin, so the console showed `0.381 norm.` where the repo rule is Kelvin. Across the
config names 471 stored models actually use, 122 resolved a factor before and 471 after.
"""

import unittest

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from aiida_pino.console.model_metrics import _named_scale_with_inheritance, named_config_scale

SCALES = "custom:\n  ref_scales:\n    T: {t}\n"


def _write(root, name, body):
    (Path(root) / f"{name}.yaml").write_text(body, encoding="utf-8")


def _resolve(root, name):
    return _named_scale_with_inheritance(Path(root), name, set())


class InheritanceTest(unittest.TestCase):
    def test_a_config_that_defines_its_own_scale_names_itself(self):
        with TemporaryDirectory() as d:
            _write(d, "own", SCALES.format(t=3000))
            self.assertEqual(_resolve(d, "own"), (3000.0, "own"))

    def test_an_inherited_scale_is_found_and_the_ancestor_is_named(self):
        # Naming the ancestor matters: the provenance string is what a reviewer checks, and
        # saying the named config carried the scale would be false.
        with TemporaryDirectory() as d:
            _write(d, "parent", SCALES.format(t=3000))
            _write(d, "child", "defaults:\n  - parent\n  - _self_\ncustom:\n  x: 1\n")
            self.assertEqual(_resolve(d, "child"), (3000.0, "parent"))

    def test_it_walks_more_than_one_level(self):
        with TemporaryDirectory() as d:
            _write(d, "grand", SCALES.format(t=2500))
            _write(d, "mid", "defaults:\n  - grand\n")
            _write(d, "leaf", "defaults:\n  - mid\n")
            self.assertEqual(_resolve(d, "leaf"), (2500.0, "grand"))

    def test_package_directed_and_self_entries_are_not_followed(self):
        # `/arch/fno_cfg@arch.fno` is an include from another group, not a config of that
        # name in this directory; `_self_` is a position in the composition order.
        with TemporaryDirectory() as d:
            _write(d, "leaf", "defaults:\n  - /arch/fno_cfg@arch.fno\n  - _self_\n")
            self.assertEqual(_resolve(d, "leaf"), (None, ""))

    def test_the_later_default_wins_as_hydra_composes_it(self):
        with TemporaryDirectory() as d:
            _write(d, "early", SCALES.format(t=1000))
            _write(d, "late", SCALES.format(t=3000))
            _write(d, "leaf", "defaults:\n  - early\n  - late\n")
            self.assertEqual(_resolve(d, "leaf"), (3000.0, "late"))

    def test_a_cycle_terminates_instead_of_recursing(self):
        with TemporaryDirectory() as d:
            _write(d, "a", "defaults:\n  - b\n")
            _write(d, "b", "defaults:\n  - a\n")
            self.assertEqual(_resolve(d, "a"), (None, ""))

    def test_missing_and_malformed_configs_return_no_factor(self):
        with TemporaryDirectory() as d:
            _write(d, "bad", "custom:\n  ref_scales:\n   - not: a mapping\n  : :\n")
            self.assertEqual(_resolve(d, "nope"), (None, ""))
            self.assertEqual(_resolve(d, "bad"), (None, ""))

    @pytest.mark.requires_lpno
    def test_the_real_shipped_configs_the_live_monitor_needed_now_resolve(self):
        # The two config names running on the box when this was found.
        for name in ("config_flow3D_RIGNO_pvlhs", "config_flow3D_FNO_randompath"):
            factor, provenance = named_config_scale(name, because="live training progress")
            self.assertEqual(factor, 3000.0, name)
            self.assertIn("inherited by", provenance)
            self.assertIn(name, provenance)


if __name__ == "__main__":
    unittest.main()
