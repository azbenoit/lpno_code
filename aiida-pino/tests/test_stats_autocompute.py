"""Background melt-pool stats worker: what it must and must not do.

Written before the implementation. The three that matter:

* it must pick up simulations whose stats are absent OR stale (older
  STATS_VERSION), because a version bump is exactly when a re-compute is owed;
* it must NOT retry a simulation whose fields cannot produce melt-pool metrics.
  Datasets ingested with the ML training profile carry no `Melt Region`, so
  `compute_case_metrics` raises KeyError for every one of their cases; without a
  tombstone the worker spins on them forever and never reaches real work
  (one dataset holds 150 such cases);
* it must stop when asked, so it can be paused alongside the FLOW-3D seat.
"""
from __future__ import annotations

import pytest

from aiida_pino.utils import stats_autocompute as sa


class FakeExtras:
    def __init__(self, data=None):
        self._d = dict(data or {})

    def get(self, key, default=None):
        return self._d.get(key, default)

    def set(self, key, value):
        self._d[key] = value


class FakeBase:
    def __init__(self, extras=None, attributes=None):
        self.extras = FakeExtras(extras)
        self.attributes = FakeExtras(attributes)


class FakeSim:
    def __init__(self, pk, extras=None, attributes=None):
        self.pk = pk
        self.uuid = f"uuid-{pk}"
        self.base = FakeBase(extras, attributes)


def sized_sim(pk, gib, *, extras=None):
    """A simulation whose three full-stack fields total roughly `gib` GiB."""
    # compute_case_metrics loads each of FULL_STACK_FIELDS as a whole
    # (T, X, Y, Z) float32 stack, so 4 bytes an element is the floor.
    per = max(1, int(gib * (2 ** 30) / 4 / len(sa.full_stack_fields())))
    aliases = {f: f.lower().replace(" ", "_") for f in sa.full_stack_fields()}
    shapes = {stored: [per] for stored in aliases.values()}
    return FakeSim(pk, extras,
                   {"array_aliases": aliases, "array_shapes": shapes})


def test_selects_absent_and_stale_but_not_current():
    current = {"version": sa.stats_version()}
    stale = {"version": sa.stats_version() - 1}
    sims = [FakeSim(1), FakeSim(2, {"meltpool_stats": current}),
            FakeSim(3, {"meltpool_stats": stale})]
    todo = [s.pk for s in sims if sa.needs_stats(s)]
    assert todo == [1, 3]


def test_unavailable_is_tombstoned_and_not_retried():
    """A case whose fields cannot yield melt metrics must be marked, not retried."""
    sim = FakeSim(4)

    def compute(_s):
        raise KeyError("Melt Region")

    assert sa.needs_stats(sim) is True
    handled = sa.compute_one(sim, compute=compute)
    assert handled is False                      # nothing computed
    assert sa.needs_stats(sim) is False          # and never offered again
    mark = sim.base.extras.get(sa.UNAVAILABLE_KEY)
    assert mark and "Melt Region" in str(mark["reason"])


def test_tombstone_is_reconsidered_after_a_version_bump():
    """A later metric revision may be able to score what the old one could not."""
    sim = FakeSim(5, {sa.UNAVAILABLE_KEY: {"version": sa.stats_version() - 1,
                                           "reason": "missing field Melt Region"}})
    assert sa.needs_stats(sim) is True


def test_successful_compute_writes_stats_and_clears_todo():
    sim = FakeSim(6)
    stats = {"version": sa.stats_version(), "depth_um": 42.0}
    assert sa.compute_one(sim, compute=lambda _s: stats) is True
    assert sim.base.extras.get("meltpool_stats") == stats
    assert sa.needs_stats(sim) is False


def test_stop_predicate_halts_the_drain():
    sims = [FakeSim(i) for i in range(5)]
    seen = []

    def compute(s):
        seen.append(s.pk)
        return {"version": sa.stats_version()}

    n = sa.drain(sims, compute=compute, should_stop=lambda: len(seen) >= 2)
    assert n == 2
    assert len(seen) == 2


def test_drain_survives_one_bad_case():
    """One exploding case must not abort the batch."""
    sims = [FakeSim(7), FakeSim(8), FakeSim(9)]

    def compute(s):
        if s.pk == 8:
            raise RuntimeError("boom")
        return {"version": sa.stats_version()}

    assert sa.drain(sims, compute=compute) == 2


# --- the poison pill -------------------------------------------------------
# 2026-08-21: the worker had restarted 16 times without once logging
# `computed N/50`. `_pending` takes the newest pending cases in a fixed order,
# so a case too big to hold in RAM was re-picked on every restart, OOM-killed
# the process, and `Restart=always` brought it straight back to the same case.
# Measured: the pending queue is small (median 0.46 GiB of stacks, p95 2.31)
# with a tail -- one case needs 13.86 GiB before the boolean-mask copies.
# The worker must step over what it cannot lift instead of dying on it.

def test_oversized_case_is_skipped_not_attempted():
    """The whole point: a case too big to hold must never reach `compute`."""
    sim = sized_sim(10, 40.0)
    called = []

    assert sa.needs_stats(sim) is True
    handled = sa.compute_one(sim, compute=lambda s: called.append(s) or {},
                             max_stack_gib=6.0)
    assert handled is False
    assert called == []                      # never loaded -- no OOM
    assert sa.needs_stats(sim, max_stack_gib=6.0) is False   # and not re-picked
    mark = sim.base.extras.get(sa.UNAVAILABLE_KEY)
    assert mark and mark["max_stack_gib"] == 6.0
    assert "too large" in str(mark["reason"])


def test_a_case_within_budget_is_computed_normally():
    sim = sized_sim(11, 0.5)
    assert sa.compute_one(sim, compute=lambda _s: {"version": sa.stats_version()},
                          max_stack_gib=6.0) is True
    assert sa.needs_stats(sim, max_stack_gib=6.0) is False


def test_drain_steps_over_the_poison_pill_and_finishes_the_rest():
    """The regression: one huge case must not cost the other 49."""
    sims = [sized_sim(12, 0.5), sized_sim(13, 40.0), sized_sim(14, 0.5)]
    done = []

    def compute(s):
        done.append(s.pk)
        return {"version": sa.stats_version()}

    assert sa.drain(sims, compute=compute, max_stack_gib=6.0) == 2
    assert done == [12, 14]                  # 13 skipped, not attempted


def test_raising_the_budget_reconsiders_a_size_tombstone():
    """A size skip is provisional; a missing-field skip stays permanent."""
    big = sized_sim(15, 10.0)
    sa.compute_one(big, compute=lambda _s: {}, max_stack_gib=6.0)
    assert sa.needs_stats(big, max_stack_gib=6.0) is False     # still too big
    assert sa.needs_stats(big, max_stack_gib=32.0) is True     # now liftable

    missing = FakeSim(16)
    def raises(_s):
        raise KeyError("Melt Region")
    sa.compute_one(missing, compute=raises)
    assert sa.needs_stats(missing, max_stack_gib=32.0) is False  # never returns


def test_unknown_size_is_attempted_rather_than_skipped():
    """No array_shapes means unknown, not huge -- skipping would strand cases."""
    sim = FakeSim(17)
    assert sa.needs_stats(sim, max_stack_gib=6.0) is True
    assert sa.compute_one(sim, compute=lambda _s: {"version": sa.stats_version()},
                          max_stack_gib=6.0) is True
