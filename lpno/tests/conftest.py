from pathlib import Path
import os
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Persistent (so the built graphs stay cached between runs) but uid-scoped: this
# box runs the suite as both the developer and the CI runner, and a fixed
# /tmp/<name> created by whoever ran first is 0755, so the other user's run dies
# with PermissionError. Honours TMPDIR via gettempdir().
GRAPH_CACHE_DIR = os.path.join(tempfile.gettempdir(), f"rigno_test_graphs_{os.getuid()}")


import pytest


@pytest.fixture(autouse=True)
def _no_machine_load_in_tests(monkeypatch):
    """The gpu_spool CPU-headroom gate must never let the box's real load decide a test.

    Added with the gate itself (2026-08-20). The per-file spool fixtures already neutralize
    the two other environmental inputs (PINO_SPOOL_CHECK_FREE for a real GPU's free VRAM,
    PINO_SPOOL_DEFAULT_RAM_MB for real free RAM); load is the third and was about to make
    test_auto_seat_selects_an_admissible_gpu_and_injects_cuda_override fail purely because
    a FLOW-3D solve was running on the same workstation. Repo-wide rather than per-file so
    a future spool test cannot reintroduce the dependency by omission. A test that WANTS
    the gate sets PINO_SPOOL_MIN_FREE_CORES itself or calls cpu_headroom_ok directly with
    explicit load/cores, both of which override this.
    """
    monkeypatch.setenv("PINO_SPOOL_MIN_FREE_CORES", "0")
    # The admission gates are tested under `enforce` whatever the checkout's literal says:
    # the public copy ships `RESOURCE_GUARDS_DEFAULT = "warn"`, and a refusal test must not
    # depend on which copy runs it. A test that wants `warn` sets the env itself.
    monkeypatch.setenv("PINO_RESOURCE_GUARDS", "enforce")
