"""The recorded code version must be the code that RAN, not HEAD at run end.

`collect_environment()` was called after the payload finished. Of the 21 models
carrying an environment block, **21 of 21** record a SHA matching HEAD at
`started_utc + wall_seconds` and only 6 match HEAD at run start. Four models
sharing the identical `started_utc` recorded four different SHAs, each matching
its own end — which rules out "they all queued behind the GPU seat". Worst drift
+51.1 h; one model's recorded commit post-dates its own code load by 12.6 h and
nine commits.

These are editable checkouts that get edited while runs are queued, so the field
named commits that did not exist when the code was loaded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aiida_pino.scripts.live_tail import collect_environment

#: The adapters that record an environment; the others have nothing to capture.
ADAPTERS = sorted(
    p for p in (Path(__file__).resolve().parent.parent / "src" / "aiida_pino" / "scripts").glob("*_adapter.py")
    if "collect_environment" in p.read_text()
)

#: The late-capture shape: built straight into the summary after the run.
LATE_CAPTURE = re.compile(r'"environment":\s*collect_environment\(\)')


def test_adapters_exist():
    assert ADAPTERS, "no adapters found to check"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda p: p.name)
def test_no_adapter_captures_the_environment_after_the_payload(adapter: Path):
    source = adapter.read_text()
    assert not LATE_CAPTURE.search(source), (
        f"{adapter.name} builds the summary with collect_environment() inline, which runs "
        "after the payload; bind it before the run instead"
    )
    assert "environment = collect_environment()" in source, (
        f"{adapter.name} must capture the environment before the payload runs"
    )


def test_the_snapshot_says_when_it_was_taken():
    env = collect_environment()
    assert env.get("captured_at"), "a provenance snapshot with no timestamp cannot be checked"


def test_a_dirty_checkout_is_flagged():
    """A SHA from an edited working tree does not name the code that ran."""
    from aiida_pino.scripts.live_tail import _repo_identity

    # 3-tuple since 2026-08-21: `_repo_identity` also returns the src/ tree
    # hash. Only the arity changed here; what this test asserts is unchanged.
    sha, dirty, _src_tree = _repo_identity("aiida_pino")
    if not sha:
        pytest.skip("not an editable git checkout")
    assert isinstance(dirty, bool)
    env = collect_environment()
    assert env.get("aiida_pino_sha") == sha
    assert bool(env.get("aiida_pino_dirty", False)) is dirty
