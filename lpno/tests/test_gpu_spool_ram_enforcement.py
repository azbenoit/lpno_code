"""BUG-43: a host-RAM booking was a promise nothing enforced.

Nothing capped the process, so a run that grew past its declaration took the RAM
anyway -- and the kernel's OOM killer picks the largest RSS, not the run that
overran, so the victim was usually the *honest* big job (the 86 GiB CI stage)
rather than the one that lied. A SIGKILL leaves no traceback either.

A cgroup cap makes an over-running run die as itself. It is applied only when the
run declared its own RAM: the 32 GB default is a guess, and capping a training at
a guessed number kills it at hour three.
"""
from __future__ import annotations

import pytest

from lpno import gpu_spool


# Deliberate behaviour change, recorded here:
# the old tests pinned "declared runs are capped BY DEFAULT". That default OOM-killed a
# healthy training at step ~900 (440-670 ms/iter, loss falling) when its validation
# pass pushed RSS + cgroup-charged PAGE CACHE (~42 GB of NPZ streamed per epoch) over
# the 1.5x ceiling -- the same failure mode flow3D_scripts had already documented for
# FLOW-3D's flsgrf write-back cache. The cap is now opt-in (PINO_SPOOL_ENFORCE_RAM=1)
# for workloads whose page cache is known-small; BUG-43's need is served by the train
# adapter's peak-RSS-vs-declaration reporting plus admission booking.
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PINO_SPOOL_ENFORCE_RAM", raising=False)
    monkeypatch.setattr(gpu_spool.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_default_is_reporting_not_capping():
    """An IO-heavy declared run must NOT be capped by default -- the cgroup charges
    its page cache, and the cap killed a healthy training at step ~900."""
    assert gpu_spool.memory_scope_prefix(8000) == []


def test_opt_in_enforcement_still_caps(monkeypatch):
    monkeypatch.setenv("PINO_SPOOL_ENFORCE_RAM", "1")
    prefix = gpu_spool.memory_scope_prefix(8000)
    assert prefix[:4] == ["systemd-run", "--user", "--scope", "-q"]
    assert prefix[-1] == "--"
    joined = " ".join(prefix)
    # 50% headroom over the declaration: using every MiB asked for is fine, a
    # runaway is not.
    assert "MemoryMax=12000M" in joined
    # Without capping swap the limit does not bite -- memory.max reclaims to the
    # shared 8 GB of swap instead of killing, which is the thrashing that locked
    # SSH out of this box. Measured: a 2 GB toy under MemoryMax=300M ran to
    # completion until swap was capped.
    assert "MemorySwapMax=0" in joined
    # No MemoryHigh: a scope that reaches it with unreclaimable anon memory is
    # throttled into a crawl rather than killed, wedging the run and its seat.
    assert "MemoryHigh" not in joined


def test_an_undeclared_run_is_not_capped():
    # None means "took the house default", which is a guess, not a measurement.
    assert gpu_spool.memory_scope_prefix(None) == []
    assert gpu_spool.memory_scope_prefix(0) == []


def test_explicit_zero_also_disables_it(monkeypatch):
    monkeypatch.setenv("PINO_SPOOL_ENFORCE_RAM", "0")
    assert gpu_spool.memory_scope_prefix(8000) == []


def test_no_systemd_means_no_cap_rather_than_no_run(monkeypatch):
    monkeypatch.setattr(gpu_spool.shutil, "which", lambda name: None)
    assert gpu_spool.memory_scope_prefix(8000) == []


def test_the_ledger_still_books_the_default_for_an_undeclared_run(tmp_path, monkeypatch):
    # The cap is opt-in by declaration, but the *booking* is not: an undeclared
    # run must stay visible to the next admission (the original BUG-43 fix).
    monkeypatch.delenv("PINO_SPOOL_DEFAULT_RAM_MB", raising=False)
    assert gpu_spool.default_ram_mb() == 32768
