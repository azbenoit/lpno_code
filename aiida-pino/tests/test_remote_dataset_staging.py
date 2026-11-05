"""A job on a remote computer must stage a PRE-STAGED remote copy, or refuse.

Before this, `stage_dataset_input` emitted `(computer_uuid, <local cache path>,
target)` whatever the computer was. On a remote box that path does not exist, so
AiiDA created a symlink to nothing and the job started and died at step 0 —
`train_adapter` reporting a missing runtime manifest, which reads as a corrupt
dataset rather than a machine that was never given one.

The lookup is by CACHE NAME, not by path: the remote cache root is the compute
box's, unknown here, and guessing it is how you get a confident wrong answer on
a box that happens to share `/home/<user>` with the primary — which some compute boxes do.

Nothing here opens a transport, so it is all unit-testable: a `Computer` with a
non-local `transport_type` is enough, and no authinfo is ever needed.
"""

from __future__ import annotations

import pytest
from aiida import orm

from aiida_pino.utils.data_nodes import (
    find_remote_runtime_cache,
    register_remote_runtime_cache,
)


def _computer(label: str, transport: str) -> orm.Computer:
    computer = orm.Computer(
        label=label,
        hostname="localhost",
        transport_type=transport,
        scheduler_type="core.direct",
        workdir="/tmp/aiida_run",
    )
    computer.store()
    return computer


CACHE = "11111111-2222-3333-4444-555555555555-abc123def456"


def test_a_cache_is_found_by_name_not_by_local_path(aiida_profile_clean):
    """The recorded remote path is the box's, and must survive verbatim."""
    remote = _computer("box-a", "core.ssh_async")
    node = register_remote_runtime_cache(
        remote, cache_name=CACHE, remote_path="/home/other/.cache/aiida-pino/runtime/" + CACHE
    )
    found = find_remote_runtime_cache(remote, CACHE)
    assert found is not None and found.pk == node.pk
    assert found.get_remote_path() == "/home/other/.cache/aiida-pino/runtime/" + CACHE


def test_a_remote_path_is_never_resolved_against_the_local_filesystem(aiida_profile_clean, tmp_path, monkeypatch):
    """`expanduser`/`resolve` are local operations and must not touch this path.

    A compute box can share `/home/<user>` with the primary, so a locally-resolved path can
    look plausible and be wrong. `~` must survive as `~`, because only the
    remote knows whose home it is.
    """
    remote = _computer("box-b", "core.ssh_async")
    monkeypatch.setenv("HOME", str(tmp_path))
    register_remote_runtime_cache(remote, cache_name=CACHE, remote_path="~/.cache/aiida-pino/runtime/" + CACHE)
    found = find_remote_runtime_cache(remote, CACHE)
    assert found.get_remote_path() == "~/.cache/aiida-pino/runtime/" + CACHE
    assert str(tmp_path) not in found.get_remote_path()


def test_a_cache_registered_for_one_box_is_not_found_on_another(aiida_profile_clean):
    """Two boxes, same dataset: staging must not hand box B box A's path."""
    box_a = _computer("box-c", "core.ssh_async")
    box_b = _computer("box-d", "core.ssh_async")
    register_remote_runtime_cache(box_a, cache_name=CACHE, remote_path="/data/" + CACHE)
    assert find_remote_runtime_cache(box_a, CACHE) is not None
    assert find_remote_runtime_cache(box_b, CACHE) is None


def test_a_different_cache_name_is_a_miss(aiida_profile_clean):
    """A field-set change renames the cache, and must miss rather than reuse."""
    remote = _computer("box-e", "core.ssh_async")
    register_remote_runtime_cache(remote, cache_name=CACHE, remote_path="/data/" + CACHE)
    assert find_remote_runtime_cache(remote, "11111111-2222-3333-4444-555555555555-999999999999") is None


def test_re_registering_the_same_cache_reuses_the_node(aiida_profile_clean):
    """Re-staging must not mint a second RemoteData for the same copy."""
    remote = _computer("box-f", "core.ssh_async")
    first = register_remote_runtime_cache(remote, cache_name=CACHE, remote_path="/data/" + CACHE)
    second = register_remote_runtime_cache(remote, cache_name=CACHE, remote_path="/data/" + CACHE)
    assert first.pk == second.pk


def test_re_registering_at_a_new_path_supersedes_the_old_entry(aiida_profile_clean):
    """Staging to a different root must not leave two answers for one name."""
    remote = _computer("box-g", "core.ssh_async")
    register_remote_runtime_cache(remote, cache_name=CACHE, remote_path="/old/" + CACHE)
    register_remote_runtime_cache(remote, cache_name=CACHE, remote_path="/new/" + CACHE)
    found = find_remote_runtime_cache(remote, CACHE)
    assert found.get_remote_path() == "/new/" + CACHE


class _Collection:
    """A native dataset node: uuid, exporter signature, and an identity."""

    def __init__(self, uuid: str, label: str = "ds"):
        self.uuid = uuid
        self.label = label
        self.pk = 4242

    def export_runtime_arraydir(self, path, runtime_fields=None, allow_pack_index=True):
        raise NotImplementedError


UUID = "11111111-2222-3333-4444-555555555555"


def test_a_staged_remote_cache_becomes_the_symlink_source(aiida_profile_clean):
    from aiida_pino.calculations.base import remote_dataset_symlink_entry
    from aiida_pino.utils.runtime_cache import cache_discriminator_for, runtime_cache_name

    remote = _computer("box-h", "core.ssh_async")
    node = _Collection(UUID)
    name = runtime_cache_name(UUID, cache_discriminator_for(node, runtime_fields=None, transport_type="core.ssh_async"))
    register_remote_runtime_cache(remote, cache_name=name, remote_path=f"/remote/cache/{name}")

    entry = remote_dataset_symlink_entry(node, remote, runtime_fields=None, target="train_dataset")
    assert entry == (remote.uuid, f"/remote/cache/{name}", "train_dataset")


def test_a_missing_remote_cache_refuses_with_the_command_that_fixes_it(aiida_profile_clean):
    """The error must name the collection actually resolved and what to run."""
    from aiida_pino.calculations.base import remote_dataset_symlink_entry

    remote = _computer("box-i", "core.ssh_async")
    with pytest.raises(ValueError) as excinfo:
        remote_dataset_symlink_entry(_Collection(UUID), remote, runtime_fields=None, target="train_dataset")
    message = str(excinfo.value)
    assert "stage-remote" in message
    assert UUID in message
    assert remote.label in message


def test_a_field_selection_looks_up_its_own_cache(aiida_profile_clean):
    """Staging a full-field copy must not satisfy a job asking for two fields."""
    from aiida_pino.calculations.base import remote_dataset_symlink_entry
    from aiida_pino.utils.runtime_cache import cache_discriminator_for, runtime_cache_name

    remote = _computer("box-j", "core.ssh_async")
    node = _Collection(UUID)
    full = runtime_cache_name(UUID, cache_discriminator_for(node, runtime_fields=None, transport_type="core.ssh_async"))
    register_remote_runtime_cache(remote, cache_name=full, remote_path=f"/remote/{full}")

    with pytest.raises(ValueError):
        remote_dataset_symlink_entry(node, remote, runtime_fields=["T"], target="train_dataset")


def test_a_union_on_a_remote_computer_is_refused_not_mis_staged(aiida_profile_clean):
    """`export_dataset_union` exports members WITHOUT allow_pack_index=False.

    So a union cache carries absolute pack paths into this machine's object
    store, and the union branch never reached the guard that would catch it.
    Refusing is the honest answer until the union exporter can build a
    transportable copy.
    """
    from aiida_pino.calculations.base import remote_dataset_symlink_entry

    class _Union(_Collection):
        pass

    remote = _computer("box-k", "core.ssh_async")
    with pytest.raises(ValueError) as excinfo:
        remote_dataset_symlink_entry(_Union(UUID), remote, runtime_fields=None, target="t", is_union=True)
    assert "union" in str(excinfo.value).lower()


def test_the_refusal_says_when_the_collection_is_DERIVED(aiida_profile_clean):
    """A training's inputs are base collections; the job consumes derived ones.

    The derived collection is created at submit time from the derivation
    profile, so it cannot be staged in advance — and someone who dutifully
    staged the dataset they passed then reads a refusal naming a uuid they have
    never seen. Without this sentence the obvious move is to re-stage the input
    and watch it fail identically.
    """
    from aiida_pino.calculations.base import remote_dataset_symlink_entry

    class _Derived(_Collection):
        def get_dataset_metadata(self):
            return {"dataset_role": "derived", "parent_collection_pk": 55011}

    remote = _computer("box-derived", "core.ssh_async")
    with pytest.raises(ValueError) as excinfo:
        remote_dataset_symlink_entry(_Derived(UUID), remote, runtime_fields=None, target="train_dataset")
    message = str(excinfo.value)
    assert "DERIVED" in message
    assert "55011" in message, "name the parent, so the user can connect it to what they passed"
    assert "stage-remote" in message


def test_a_base_collection_refusal_stays_short(aiida_profile_clean):
    """No derivation note when there is no derivation to explain."""
    from aiida_pino.calculations.base import remote_dataset_symlink_entry

    class _Base(_Collection):
        def get_dataset_metadata(self):
            return {"dataset_role": "base"}

    remote = _computer("box-base", "core.ssh_async")
    with pytest.raises(ValueError) as excinfo:
        remote_dataset_symlink_entry(_Base(UUID), remote, runtime_fields=None, target="train_dataset")
    assert "DERIVED" not in str(excinfo.value)


def test_a_process_with_no_calcjob_datasets_is_refused_not_silently_empty(aiida_profile_clean):
    """`--for-training` on the wrong pk must say so rather than stage nothing.

    Staging nothing and reporting success is the failure shape this whole path
    exists to prevent; a resolver that returns an empty list is exactly that.
    """
    from aiida import orm

    from aiida_pino.utils.stage_remote import collections_consumed_by

    node = orm.Dict(dict={"not": "a process"}).store()
    assert collections_consumed_by(node.pk) == []
