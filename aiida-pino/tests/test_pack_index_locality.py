"""Pack-index staging is LOCAL-ONLY, and the guard must not rely on discipline.

A `pack_index.json` entry stores `pack_path` -- an absolute path into this
machine's disk-objectstore container -- plus an offset and size
(`data/pino_simulation.py:1107-1125`). On any computer other than `local` those
bytes do not exist, so a dataset staged that way resolves to nothing. Inert
while one computer exists; live the day the second workstation lands
(`REMOTE_GPU_PLAN.md`), and WIDER since 2026-08-17 (8ca4a3d) because
identity-derived collections stage by index too -- which is most of what a
training materialises.

The subtle half is the cache, not the export: `_dataset_runtime_cache` is keyed
by node uuid, so a cache built for a LOCAL job would otherwise be handed
straight to a remote one. The two must not share a key.
"""

from __future__ import annotations

import json

import pytest

from aiida_pino.data.pino_dataset_collection import (
    assert_cache_usable_for_computer,
    pack_index_allowed_for_transport,
)


def test_index_allowed_only_for_a_local_transport():
    assert pack_index_allowed_for_transport("core.local") is True
    assert pack_index_allowed_for_transport("core.ssh") is False
    assert pack_index_allowed_for_transport("core.ssh_async") is False
    # Unknown transport: fail safe, no index.
    assert pack_index_allowed_for_transport("") is False
    assert pack_index_allowed_for_transport(None) is False


def test_an_indexed_cache_is_refused_for_a_remote_computer(tmp_path):
    """The failure must be a loud refusal, not a job that stages nothing."""
    (tmp_path / "pack_index.json").write_text(json.dumps({"entries": [{"pack_path": "/x"}]}))

    # Local: fine, this is the normal fast path.
    assert_cache_usable_for_computer(tmp_path, "core.local")

    with pytest.raises(ValueError) as excinfo:
        assert_cache_usable_for_computer(tmp_path, "core.ssh_async")
    message = str(excinfo.value)
    assert "pack_index" in message
    # The message has to say what to do, not just what is wrong.
    assert "local" in message.lower()


def test_a_copied_cache_is_usable_anywhere(tmp_path):
    """No index means real payloads, which travel."""
    (tmp_path / "manifest.json").write_text("{}")
    assert_cache_usable_for_computer(tmp_path, "core.local")
    assert_cache_usable_for_computer(tmp_path, "core.ssh_async")
