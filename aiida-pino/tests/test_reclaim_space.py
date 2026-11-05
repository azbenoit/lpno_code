"""Targeted reclaim: free disk from evicted/unreferenced objects.

Eviction drops a node's reference to its `payload.npz`, but on a packed store
the bytes stay inside the pack file until a repack. `verdi storage maintain
--full` repacks all packs; this reclaims the same space by repacking *only* the
packs that actually hold unreferenced bytes, so the exclusive-lock window and
the I/O are proportional to what was evicted, not to the whole 700 GB store.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.console import maintenance
from web.app.aiida_service import AiidaConsoleService
from web.app.config import Settings


@pytest.fixture
def console_service():
    return AiidaConsoleService(Settings())


def _packed_sim(factory, name, seed=0):
    from aiida.manage import get_manager

    # Distinct content per sim, else the content-addressed store dedups them into
    # one object and evicting one leaves it referenced by the other.
    arr = (np.arange(64, dtype=np.float32) + seed * 1000.0).reshape(1, 1, 4, 4, 4)
    sim = factory(name, arrays={"Temperature": arr})
    # Pack loose objects so the reclaim has to go through repack, like production.
    get_manager().get_profile_storage().get_repository()._container.pack_all_loose()
    return sim


def test_plan_reports_unreferenced_bytes_and_affected_packs(aiida_profile_clean, pino_simulation_factory) -> None:
    from aiida.manage import get_manager

    keep = _packed_sim(pino_simulation_factory, "keep", seed=1)
    drop = _packed_sim(pino_simulation_factory, "drop", seed=2)
    drop.evict_data()

    be = get_manager().get_profile_storage()
    unref = list(be.get_unreferenced_keyset())
    plan = maintenance.plan_reclaim(be.get_repository()._container, unref)

    assert plan["unreferenced"] >= 1
    assert plan["reclaimable_bytes"] > 0
    assert plan["packs"], "an evicted packed object should mark its pack for repacking"
    # keep's data must not be in the unreferenced set.
    assert keep.base.repository.list_object_names()


def test_reclaim_frees_evicted_bytes_and_preserves_the_rest(aiida_profile_clean, pino_simulation_factory, console_service) -> None:
    from aiida.manage import get_manager

    keep = _packed_sim(pino_simulation_factory, "keep", seed=1)
    drop = _packed_sim(pino_simulation_factory, "drop", seed=2)
    keep_name = keep.list_original_array_names()[0]
    drop.evict_data()

    before = get_manager().get_profile_storage().get_repository()._container.get_total_size()

    dry = console_service.reclaim_space(dry_run=True)
    assert dry["dry_run"] is True and dry["unreferenced"] >= 1
    # a dry run must not have deleted anything
    assert list(get_manager().get_profile_storage().get_unreferenced_keyset())

    done = console_service.reclaim_space(dry_run=False)
    assert done["dry_run"] is False
    assert done["packs_repacked"] >= 1

    be = get_manager().get_profile_storage()
    assert list(be.get_unreferenced_keyset()) == [], "unreferenced objects survived reclaim"
    after = be.get_repository()._container.get_total_size()
    assert after.total_size_packed < before.total_size_packed, "reclaim did not shrink the packed size"
    # the kept simulation still reads — repack must never touch referenced data.
    assert np.asarray(keep.get_array(keep_name)).size == 64


def test_reclaim_is_a_noop_when_nothing_is_unreferenced(aiida_profile_clean, pino_simulation_factory, console_service) -> None:
    _packed_sim(pino_simulation_factory, "keep")
    report = console_service.reclaim_space(dry_run=False)
    assert report["unreferenced"] == 0 and report["packs"] == []
