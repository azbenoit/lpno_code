"""Runtime-cache eviction: only untouched entries no in-flight job holds."""
from __future__ import annotations

import os
import time

import pytest

from aiida_pino.utils.runtime_evict import entry_size, stale_entries

LIVE = "11111111-1111-1111-1111-111111111111"
DEAD = "22222222-2222-2222-2222-222222222222"
FRESH = "33333333-3333-3333-3333-333333333333"


def _mkentry(root, name, *, age_hours=0.0, size=0):
    path = root / name
    path.mkdir()
    if size:
        (path / "payload.npz").write_bytes(b"\0" * size)
    stamp = time.time() - age_hours * 3600.0
    os.utime(path, (stamp, stamp))
    return path


def test_evicts_only_old_unused_entries(tmp_path):
    _mkentry(tmp_path, LIVE, age_hours=48.0)
    dead = _mkentry(tmp_path, DEAD, age_hours=48.0)
    _mkentry(tmp_path, FRESH, age_hours=1.0)

    stale = stale_entries(tmp_path, in_use={LIVE}, grace_hours=6.0)

    assert stale == [dead], "an in-use uuid or a recently touched entry must survive"


def test_ignores_unrelated_names_but_reaps_old_build_dirs(tmp_path):
    _mkentry(tmp_path, "not-a-uuid", age_hours=48.0)
    _mkentry(tmp_path, f".build-{DEAD}-999", age_hours=1.0)
    crashed = _mkentry(tmp_path, f".build-{DEAD}-123", age_hours=48.0)
    repair = _mkentry(tmp_path, f".repair-{DEAD}-456", age_hours=48.0)

    stale = stale_entries(tmp_path, in_use=set(), grace_hours=6.0)

    assert sorted(stale) == sorted([crashed, repair])


def test_in_use_wins_over_age_for_build_dirs(tmp_path):
    _mkentry(tmp_path, f".build-{LIVE}-123", age_hours=48.0)

    assert stale_entries(tmp_path, in_use={LIVE}, grace_hours=6.0) == []


def test_missing_root_is_not_an_error(tmp_path):
    assert stale_entries(tmp_path / "absent", in_use=set()) == []


def test_entry_size_counts_a_hardlinked_payload_once(tmp_path):
    entry = _mkentry(tmp_path, DEAD, age_hours=48.0, size=4096)
    try:
        os.link(entry / "payload.npz", entry / "alias.npz")
    except OSError:  # pragma: no cover - filesystem without hard links
        pytest.skip("no hard-link support here")

    assert entry_size(entry) == 4096


def test_discriminated_cache_names_are_recognised_at_every_shape():
    """A field-trimmed cache is `<uuid>-<hash12>`, and this tool could not see it.

    `_entry_uuid` re-derived the uuid with an anchored regex, so a discriminated
    entry -- published, `.build-` residue, or `.repair-` residue -- returned
    None and was skipped at any age. That is a silent leak in the documented
    reclaim path on a box where disk is a hard resource. It fails closed, which
    is why nobody noticed.
    """
    from aiida_pino.utils.runtime_evict import _entry_uuid

    uuid = "dd250548-1d94-44e1-ab06-ca955d3cf9fb"
    disc = "6029bce7b6e4"
    assert _entry_uuid(uuid) == uuid
    assert _entry_uuid(f"{uuid}-{disc}") == uuid, "published discriminated entry"
    assert _entry_uuid(f".build-{uuid}-12345") == uuid
    assert _entry_uuid(f".build-{uuid}-{disc}-12345") == uuid, "crashed discriminated build"
    assert _entry_uuid(f".repair-{uuid}-{disc}-99") == uuid, "abandoned repair directory"

    # Still strict: an unrelated directory must never be claimed as a cache.
    assert _entry_uuid("unrelated-dir") is None
    assert _entry_uuid(".build-not-a-uuid-1") is None


# --- a cache whose SOURCE was evicted is the last copy ---------------------
# 2026-08-21: the evictor's model is "a runtime cache entry is not data, it
# re-exports from the object store, so the cost of deleting one is time". That
# is right, and it is why deleting 964 entries to free 55.3 GB was safe. Two entries had
# been flagged as must-not-touch on the grounds they were "local-only",
# which conflated "not re-fetchable from the remote box" with "not re-derivable at all".
# Checked afterwards: both sources were intact (`data_evicted` unset), so they
# re-export and the flag was wrong.
#
# But the model has one real exception, and this repo ships the mechanism for
# it. `evict_dataset_data` deletes a dataset's array data while keeping its
# metadata and provenance -- documented practice for reclaiming disk. A cache
# built from a collection that has since been evicted CANNOT be re-exported:
# there is nothing left to export from, and the cache is the last usable copy
# of those bytes. Deleting it is then a loss, not a delay.


def test_a_cache_whose_source_was_evicted_is_never_stale(tmp_path):
    from aiida_pino.utils.runtime_evict import stale_entries

    old = 0.0  # far older than any grace window
    for name in ("aaaaaaaa-0000-0000-0000-000000000001",
                 "bbbbbbbb-0000-0000-0000-000000000002"):
        d = tmp_path / name
        d.mkdir()
        (d / "payload.npz").write_bytes(b"x")
        import os
        os.utime(d, (old, old))

    evicted = {"bbbbbbbb-0000-0000-0000-000000000002"}
    stale = stale_entries(tmp_path, set(), grace_hours=1.0, evicted_sources=evicted)
    names = {p.name for p in stale}

    assert "aaaaaaaa-0000-0000-0000-000000000001" in names, "an intact source still evicts"
    assert "bbbbbbbb-0000-0000-0000-000000000002" not in names, (
        "a cache whose source data was evicted cannot be re-exported, so deleting "
        "it destroys the last copy rather than costing a re-export"
    )


def test_evicted_sources_defaults_to_empty_and_changes_nothing(tmp_path):
    """The guard must be opt-in for callers that cannot query, not a behaviour change."""
    import os

    from aiida_pino.utils.runtime_evict import stale_entries

    d = tmp_path / "cccccccc-0000-0000-0000-000000000003"
    d.mkdir()
    os.utime(d, (0.0, 0.0))
    assert [p.name for p in stale_entries(tmp_path, set(), grace_hours=1.0)] == [d.name]


def test_a_collection_a_live_workchain_holds_through_a_finished_child_is_in_use(aiida_profile_clean, aiida_localhost):
    """The guard held DIRECT inputs of non-terminal processes and nothing else.

    `TrainModelWorkChain` resolves a derived collection, runs a short VRAM SMOKE CalcJob on it
    (whose upload builds the runtime cache), waits for it to FINISH, then creates the real
    training CalcJob on the same collection. Between those two the collection is an input of
    a terminal child and a ctx value of the live parent -- held by nobody the query could see.
    On 2026-09-11 a disk-space guard script ran `runtime_evict --apply` every few
    minutes below 60 GB and evicted exactly that entry each cycle: a 4.8 GB cache re-derived
    for ~1 h per job (self-improve measured 60.2 GB, three times, on one day's members).
    A live process holds everything its called descendants consumed or produced.
    """
    from aiida import orm
    from aiida.common.links import LinkType

    from aiida_pino.utils.runtime_evict import in_use_uuids

    asked = orm.Data().store()          # what the submitter passed: the parent's INPUT
    resolved = orm.Data().store()       # what presubmit resolved to: the cache is keyed by THIS
    produced = orm.Data().store()       # a derived collection a finished child RETURNED

    parent = orm.WorkChainNode()
    parent.base.links.add_incoming(asked, LinkType.INPUT_WORK, "train_dataset")
    parent.set_process_state("waiting")
    parent.store()

    smoke = orm.CalcJobNode(computer=aiida_localhost)
    smoke.set_option("resources", {"num_machines": 1})
    smoke.base.links.add_incoming(parent, LinkType.CALL_CALC, "CALL")
    smoke.base.links.add_incoming(resolved, LinkType.INPUT_CALC, "train_dataset")
    smoke.set_process_state("finished")
    smoke.store()

    child = orm.WorkChainNode()
    child.base.links.add_incoming(parent, LinkType.CALL_WORK, "CALL")
    child.set_process_state("finished")
    child.store()
    produced.base.links.add_incoming(child, LinkType.RETURN, "derived_dataset")

    held = in_use_uuids()
    assert asked.uuid in held
    assert resolved.uuid in held, "input of a FINISHED child of a live parent must be held"
    assert produced.uuid in held, "output of a finished child of a live parent must be held"


def test_nothing_is_held_once_the_parent_is_terminal(aiida_profile_clean, aiida_localhost):
    from aiida import orm
    from aiida.common.links import LinkType

    from aiida_pino.utils.runtime_evict import in_use_uuids

    resolved = orm.Data().store()
    parent = orm.WorkChainNode()
    parent.set_process_state("finished")
    parent.store()
    smoke = orm.CalcJobNode(computer=aiida_localhost)
    smoke.set_option("resources", {"num_machines": 1})
    smoke.base.links.add_incoming(parent, LinkType.CALL_CALC, "CALL")
    smoke.base.links.add_incoming(resolved, LinkType.INPUT_CALC, "train_dataset")
    smoke.set_process_state("finished")
    smoke.store()
    assert resolved.uuid not in in_use_uuids()
