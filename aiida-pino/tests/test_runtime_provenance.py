"""Compute time and environment identity are recorded and queryable.

Before 2026-07-30 a training's `/usr/bin/time -v` report sat unparsed in a
FolderData, FLOW-3D's solver clock sat nested in one Dict, evals measured
nothing, and no run recorded what code or hardware produced it. The adapters
now write a `runtime` block (RuntimeMeter + collect_environment) into every
summary, and the parser promotes the aggregates to calcjob extras.
"""

from __future__ import annotations

import pytest
import subprocess
import sys

from aiida_pino.parsers.common import stamp_runtime_extras
from aiida_pino.scripts.live_tail import RuntimeMeter, collect_environment


def _in_git_checkout() -> bool:
    """True when this test file sits in a git working tree and `git` can read it."""
    import shutil
    from pathlib import Path

    if shutil.which("git") is None:
        return False
    try:
        probe = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
    except OSError:
        return False
    return probe.returncode == 0 and probe.stdout.strip() == "true"


#: The source-identity tests read commit and tree hashes, which exist only in a git
#: checkout; a plain export of the repository has none to record.
requires_git_checkout = pytest.mark.skipif(
    not _in_git_checkout(), reason="not a git checkout: there is no commit or tree hash to record"
)

class _FakeNode:
    """Just enough of a node to receive extras."""

    def __init__(self, existing: dict | None = None):
        self._extras = dict(existing or {})
        self.base = self

    @property
    def extras(self):
        return self

    def get(self, key, default=None):
        return self._extras.get(key, default)

    def set(self, key, value):
        self._extras[key] = value


def test_runtime_meter_counts_child_cpu_and_wall() -> None:
    meter = RuntimeMeter()
    subprocess.run([sys.executable, "-c", "sum(i * i for i in range(2_000_000))"], check=True)
    result = meter.stop()
    assert result["wall_seconds"] > 0
    assert result["cpu_user_seconds"] + result["cpu_system_seconds"] > 0
    assert result["max_rss_mb"] > 0
    assert result["started_utc"].endswith("Z")


def test_collect_environment_identifies_the_machine_and_stack() -> None:
    env = collect_environment()
    assert env["hostname"]
    assert env["python"]
    # numpy is a hard dependency of this repo, so its version must be present
    assert env.get("numpy_version")


def test_stamping_promotes_adapter_runtime_block() -> None:
    node = _FakeNode()
    stamp_runtime_extras(node, {
        "gpu_vram_peak_mb": 3180,
        "runtime": {
            "wall_seconds": 3210.0, "cpu_user_seconds": 3158.0, "cpu_system_seconds": 12.0,
            "max_rss_mb": 2125,
            "environment": {"hostname": "labhost01", "gpu_name": "TITAN RTX"},
        },
    })
    assert node._extras == {
        "runtime_wall_seconds": 3210.0,
        "runtime_cpu_seconds": 3170.0,
        "runtime_max_rss_mb": 2125,
        "env_hostname": "labhost01",
        "env_gpu_name": "TITAN RTX",
        "gpu_vram_peak_mb": 3180,
    }


def test_stamping_promotes_flow3d_solver_clock() -> None:
    node = _FakeNode()
    stamp_runtime_extras(node, {
        "manifest": {
            "run": {"cpu_seconds": 96867.2, "elapsed_seconds": 12125.5},
            "parallel": {"cores": 8},
        },
    })
    assert node._extras["solver_cpu_seconds"] == 96867.2
    assert node._extras["solver_elapsed_seconds"] == 12125.5
    assert node._extras["solver_cores"] == 8


def test_stamping_never_overwrites_existing_extras() -> None:
    """Re-parses and the backfill must be idempotent: a measured value that is
    already recorded (e.g. gpu_vram_peak_mb from the auto-VRAM workchain) wins
    over anything computed later."""
    node = _FakeNode({"gpu_vram_peak_mb": 3000, "runtime_wall_seconds": 1.0})
    stamp_runtime_extras(node, {
        "gpu_vram_peak_mb": 9999,
        "runtime": {"wall_seconds": 9999.0},
    })
    assert node._extras["gpu_vram_peak_mb"] == 3000
    assert node._extras["runtime_wall_seconds"] == 1.0


def test_stamping_handles_empty_and_partial_summaries() -> None:
    node = _FakeNode()
    stamp_runtime_extras(node, None)
    stamp_runtime_extras(node, {})
    stamp_runtime_extras(node, {"runtime": {}})
    assert node._extras == {}


# --- a SHA is a mutable identifier; a tree hash is not ---------------------
# 2026-08-21: a fast-import purging `Important_models/` re-hashed every commit
# after 4b25162. Measured on the live profile: of 196 distinct
# `lpno_sha` values recorded across 1834 nodes, **137 stopped being
# ancestors of HEAD** and resolve in no fresh clone. They survive only because
# someone created a rescue branch before `gc` ran, and that branch cannot be
# published -- it carries 712 MB of exactly the content the purge removed
# because it was too big to push.
#
# Recording the sha EARLIER does not help; it was already recorded correctly at
# run time. The fragility is that a commit hash is not content-addressed. The
# `src/` TREE hash is, and it was verified identical across all 138 rewritten
# old/new pairs -- so a run that recorded its src tree needs no remap at all.

@requires_git_checkout
def test_repo_identity_records_the_source_tree_not_only_the_commit():
    import subprocess
    from pathlib import Path

    from aiida_pino.scripts import live_tail

    sha, dirty, src_tree = live_tail._repo_identity("aiida_pino")
    assert sha, "commit sha must still be recorded"
    assert src_tree, "src/ tree hash must be recorded alongside it"

    root = Path(live_tail.__file__).resolve()
    repo = next(p for p in root.parents if (p / ".git").exists())
    # The src directory addressed from the repo root: `src` here, `aiida-pino/src` in the
    # combined public repository, where the package sits one directory down.
    src_rel = root.parents[2].relative_to(repo).as_posix()  # src/aiida_pino/scripts/live_tail.py -> src
    expected = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short=12", f"HEAD:{src_rel}"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert src_tree == expected, "must be the real src/ tree of the recorded commit"


@requires_git_checkout
def test_repo_sha_still_returns_the_commit():
    """`_repo_sha` indexes the tuple; widening it must not shift that."""
    from aiida_pino.scripts import live_tail

    assert live_tail._repo_sha("aiida_pino") == live_tail._repo_identity("aiida_pino")[0]


@requires_git_checkout
def test_collect_environment_carries_the_tree_hash():
    from aiida_pino.scripts import live_tail

    env = live_tail.collect_environment()
    assert env.get("aiida_pino_sha"), "sanity: this repo is a git checkout"
    assert env.get("aiida_pino_src_tree"), (
        "the environment block must carry a rewrite-proof source identity, not "
        "only a commit hash that a history rewrite can invalidate"
    )
