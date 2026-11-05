"""param_dir default resolution is lenient; data/val stay strict.

param_dir is only read for piecewise-tabulated c(T) materials (random-path
materials use polynomial c(T)), so a missing default param path must not abort
config loading — existence is validated at read time in
_define_material_properties instead. data_dir/val_data_dir remain strict.
"""

import unittest
from pathlib import Path

from lpno.models.pipeline import config_loading as cl


class ParamDirLenientTest(unittest.TestCase):
    _MISSING = Path("nonexistent_dir_for_test_xyz")

    def test_strict_default_missing_raises(self):
        # data/val resolution keeps require_exists=True (default).
        with self.assertRaises(ValueError):
            cl._resolve_configured_data_path("DEFAULT", relative_path=self._MISSING, label="test data")

    def test_lenient_default_missing_ok(self):
        # param_dir passes require_exists=False -> resolves without raising.
        p = cl._resolve_configured_data_path(
            "DEFAULT", relative_path=self._MISSING, label="parameter data", require_exists=False
        )
        self.assertTrue(p.endswith(str(self._MISSING)))

    def test_explicit_value_passthrough(self):
        # An explicit configured path is returned verbatim (no default resolution).
        p = cl._resolve_configured_data_path("/an/explicit/path", relative_path=self._MISSING, label="x")
        self.assertEqual(p, "/an/explicit/path")


if __name__ == "__main__":
    unittest.main()
