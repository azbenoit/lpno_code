#!/usr/bin/env python3
"""The validation/ tree is the reproducibility deliverable: every campaign
config must stay renderable, self-contained (relative material paths), and
runnable through validation/reproduce.py without a license."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "validation"


class ValidationPipelineTests(unittest.TestCase):
    def test_tree_exists_and_is_self_contained(self) -> None:
        cfgs = sorted((V / "configs").glob("*.json"))
        self.assertGreater(len(cfgs), 40)
        for cfg in cfgs:
            c = json.loads(cfg.read_text())
            m = c.get("material", {})
            self.assertIn("file", m, f"{cfg.name}: material must be a file reference")
            self.assertFalse(Path(m["file"]).is_absolute(), f"{cfg.name}: absolute material path")
            self.assertTrue((cfg.parent / m["file"]).resolve().is_file(),
                            f"{cfg.name}: material file missing: {m['file']}")

    def test_reproduce_prepare_only_renders_a_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(
                [sys.executable, str(V / "reproduce.py"), "--only", "t2_156w", "--out-root", tmp],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout[-500:] + r.stderr[-500:])
            prepins = list(Path(tmp).glob("t2_156w/prepin.*"))
            self.assertEqual(len(prepins), 1)
            text = prepins[0].read_text(errors="ignore")
            self.assertIn("tv1=3520", text)          # the calibrated Ti64 recipe
            self.assertIn("if_prsrecoil=2", text)

    def test_unknown_case_is_rejected(self) -> None:
        r = subprocess.run(
            [sys.executable, str(V / "reproduce.py"), "--only", "no_such_case"],
            capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no_such_case", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
