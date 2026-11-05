"""A transient archive error on the SHARED runtime cache must not kill a run.

The staged cache is a mutable directory other jobs stage into, evict from and
repair. A file rewritten under an open handle surfaces as
`BadZipFile: Bad CRC-32` even though the file verifies clean moments later --
measured twice on the varspeed cache (2026-08-13/14), each time killing a
training at step ~8000 after hours of work. One reopen absorbs it; a second
failure is real and must propagate.
"""
from __future__ import annotations

import zipfile

import pytest

from lpno.models.pipeline.npz_lazy_store import NpzDirStore


class _Store(NpzDirStore):
    def __init__(self, fail_times):        # noqa: D107 - test double
        self._calls = 0
        self._fail_times = fail_times
        self._cache = {}

    def _case_uncached(self, case_idx):
        self._calls += 1
        if self._calls <= self._fail_times:
            raise zipfile.BadZipFile("Bad CRC-32 for file 'Fraction Of Fluid.npy'")
        return f"case{case_idx}"


def test_one_transient_error_is_retried(capsys):
    s = _Store(fail_times=1)
    assert s._case(3) == "case3"
    assert s._calls == 2
    assert "transient archive error" in capsys.readouterr().out


def test_a_persistent_error_propagates():
    s = _Store(fail_times=2)
    with pytest.raises(zipfile.BadZipFile):
        s._case(0)
    assert s._calls == 2, "must retry exactly once, not loop"


def test_no_error_means_no_retry():
    s = _Store(fail_times=0)
    assert s._case(1) == "case1"
    assert s._calls == 1
