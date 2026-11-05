"""Regression: the README's model-entrypoint list named a module that does not exist.

`NS_Energy_laser_PINO_standard.py` was listed as a compatibility wrapper, but
`models/` only ever had the unified entrypoint, the kfold wrapper and the eval
wrapper -- running the documented module raised ModuleNotFoundError. The list is
a user-facing map of the package, so it is checked against the tree.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "lpno" / "models"
SECTION = "### 2. Model Entrypoints + Shared Pipeline"


def _listed_modules() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    body = text.split(SECTION, 1)[1].split("\n### ", 1)[0]
    # Bullet lines of the form "- `some/path.py`: description"
    return re.findall(r"^- `([^`]+\.py)`", body, flags=re.MULTILINE)


def test_section_is_present_and_not_empty():
    assert _listed_modules(), f"README section {SECTION!r} lists no modules"


def test_every_listed_module_exists():
    missing = [m for m in _listed_modules() if not (MODELS / m).is_file()]
    assert not missing, f"README lists modules that do not exist under models/: {missing}"
