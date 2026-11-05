#!/usr/bin/env python3
"""The README config-schema table is generated from python/config_schema.py.

These tests fail if the committed README block drifts from the generator, and
check the `--print-config-schema` CLI documented in the README.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RUNNER = ROOT / "python" / "run_rectangular_case.py"

sys.path.insert(0, str(ROOT / "python"))
import config_schema  # noqa: E402


class ConfigSchemaDocsTests(unittest.TestCase):
    def _readme_block(self) -> str:
        text = README.read_text(encoding="utf-8")
        start = text.index(config_schema.SCHEMA_BEGIN) + len(config_schema.SCHEMA_BEGIN)
        end = text.index(config_schema.SCHEMA_END)
        return text[start:end].strip()

    def test_readme_table_matches_generator(self) -> None:
        self.assertEqual(
            self._readme_block(),
            config_schema.to_markdown(),
            "README schema table is stale; run "
            "`python3 python/run_rectangular_case.py --print-config-schema` and paste the output.",
        )

    def test_every_key_the_runner_reads_has_a_row(self) -> None:
        """A config key the runner reads (``laser.get("x")``) and the table does not
        name is a knob users cannot find. Thirteen optics/vapour/pressure keys and
        ``laser.beams`` were readable and undocumented for months (public-cut
        audit, 2026-09-14); this keeps the table complete by construction."""
        source = RUNNER.read_text(encoding="utf-8")
        sections = ("parallel", "geometry", "mesh", "laser", "simulation")
        read = set()
        for section in sections:
            for key in re.findall(rf'\b{section}\.get\("([a-z_0-9]+)"', source):
                read.add(f"{section}.{key}")
        documented = "\n".join(value for value, _, _ in config_schema.CONFIG_SCHEMA)
        missing = sorted(k for k in read if f"`{k}`" not in documented)
        self.assertEqual(missing, [], f"config keys the runner reads but CONFIG_SCHEMA omits: {missing}")

    def test_print_config_schema_cli(self) -> None:
        result = subprocess.run(
            ["python3", str(RUNNER), "--print-config-schema"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), config_schema.to_markdown())


if __name__ == "__main__":
    unittest.main()
