"""One cache name, computed in one place, for the local and the remote path.

`stage_dataset_input` builds a runtime cache under `<uuid>` or
`<uuid>-<discriminator-hash>`; `aiida-pino stage-remote` has to put its copy on
the compute box under the *same* name, because staging looks it up by that name
at `prepare_for_submission`. If the two ever disagree the failure is not an
error, it is a job that finds nothing staged -- so the naming must be shared
code, not two implementations that happen to match today.

The local name must also stay BYTE-IDENTICAL to what it was before remote
support existed, or every cache already on disk is orphaned.
"""

from __future__ import annotations

import pytest

from aiida_pino.utils.runtime_cache import (
    cache_discriminator_for,
    runtime_cache_name,
)


class _Node:
    """Stands in for a collection: a uuid and a full exporter signature."""

    def __init__(self, uuid: str):
        self.uuid = uuid

    def export_runtime_arraydir(self, path, runtime_fields=None, allow_pack_index=True):
        raise NotImplementedError


class _MinimalNode:
    """An exporter that accepts neither flag -- the signature-drift case."""

    def __init__(self, uuid: str):
        self.uuid = uuid

    def export_runtime_arraydir(self, path):
        raise NotImplementedError


UUID = "11111111-2222-3333-4444-555555555555"


def test_no_discriminator_means_the_bare_uuid():
    """The pre-remote name. Changing it orphans every cache already on disk."""
    assert runtime_cache_name(UUID, None) == UUID
    assert runtime_cache_name(UUID, {}) == UUID


def test_a_discriminator_appends_a_stable_short_hash():
    name = runtime_cache_name(UUID, {"runtime_fields": ["T", "fl"]})
    assert name.startswith(f"{UUID}-")
    assert len(name) == len(UUID) + 1 + 12
    # Stable across calls and insensitive to key order.
    assert name == runtime_cache_name(UUID, {"runtime_fields": ["T", "fl"]})
    assert runtime_cache_name(UUID, {"a": 1, "b": 2}) == runtime_cache_name(UUID, {"b": 2, "a": 1})


def test_field_order_is_significant_because_the_export_differs():
    assert runtime_cache_name(UUID, {"runtime_fields": ["T", "fl"]}) != runtime_cache_name(
        UUID, {"runtime_fields": ["fl", "T"]}
    )


def test_a_local_computer_keeps_the_historical_name():
    """core.local with no field selection must still be the bare uuid."""
    node = _Node(UUID)
    disc = cache_discriminator_for(node, runtime_fields=None, transport_type="core.local")
    assert disc == {}
    assert runtime_cache_name(UUID, disc) == UUID


def test_a_remote_computer_gets_a_different_cache_than_local():
    """A local cache holds pack-index entries into THIS machine's object store.

    Handing that to a remote job stages paths that do not exist there, so the
    two must never share a key.
    """
    node = _Node(UUID)
    local = cache_discriminator_for(node, runtime_fields=None, transport_type="core.local")
    remote = cache_discriminator_for(node, runtime_fields=None, transport_type="core.ssh_async")
    assert remote == {"pack_index": False}
    assert runtime_cache_name(UUID, local) != runtime_cache_name(UUID, remote)


def test_runtime_fields_and_transport_combine():
    node = _Node(UUID)
    disc = cache_discriminator_for(node, runtime_fields=["T"], transport_type="core.ssh_async")
    assert disc == {"runtime_fields": ["T"], "pack_index": False}


def test_a_field_mismatch_misses_loudly_rather_than_training_on_the_wrong_fields():
    """Different field sets are different caches, so a mismatch cannot silently reuse."""
    node = _Node(UUID)
    a = cache_discriminator_for(node, runtime_fields=["T"], transport_type="core.ssh_async")
    b = cache_discriminator_for(node, runtime_fields=["T", "fl"], transport_type="core.ssh_async")
    assert runtime_cache_name(UUID, a) != runtime_cache_name(UUID, b)


def test_an_exporter_that_cannot_take_the_flags_does_not_record_them():
    """The discriminator must describe what was actually passed to the exporter."""
    node = _MinimalNode(UUID)
    disc = cache_discriminator_for(node, runtime_fields=["T"], transport_type="core.ssh_async")
    assert disc == {}


@pytest.mark.parametrize("transport", ["core.ssh", "core.ssh_async", "", None])
def test_every_non_local_transport_is_treated_as_remote(transport):
    node = _Node(UUID)
    disc = cache_discriminator_for(node, runtime_fields=None, transport_type=transport)
    assert disc == {"pack_index": False}
