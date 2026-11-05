from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def has_served_gt(*rel_dirs: str) -> bool:
    """True when every given ground-truth dir (repo-relative) holds at least one
    NPZ. The served data is gitignored, so a fresh clone has the tracked
    meta.json files and none of the arrays; tests needing them skip, not fail."""
    return all(any((ROOT / d).glob("**/*.npz")) for d in rel_dirs)


def needs_served_gt(*rel_dirs: str):
    """Marker: skip unless the ground truth under ``rel_dirs`` is present."""
    return pytest.mark.skipif(
        not has_served_gt(*rel_dirs),
        reason=f"served ground truth not present under {list(rel_dirs)} (see README: fetch the asset bundle)",
    )


def skip_without_served_gt(*rel_dirs: str) -> None:
    """Same check from inside a fixture."""
    if not has_served_gt(*rel_dirs):
        pytest.skip(f"served ground truth not present under {list(rel_dirs)} (see README: fetch the asset bundle)")
