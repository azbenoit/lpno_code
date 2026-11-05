"""A field-discriminated runtime cache must still be repairable.

`stage_dataset_input` gained a cache discriminator so a compact
Temperature/fl export cannot collide with a full-field one: the directory is
named ``<node-uuid>-<fieldhash>`` instead of ``<node-uuid>``. `runtime_repair`
still gated on a fully anchored uuid regex, so every discriminated cache failed
the gate and `repair_dataset_dir` silently returned False.

That is the repair path the gpu-seat shim runs *after* the seat wait, precisely
to rebuild a cache evicted while the job queued. Losing it for exactly the runs
that trim fields is what took out the whole varspeed vscan arm on 2026-07-31:
five workchains dead with `BadZipFile: Bad CRC-32` reading a cache directory
that nothing rebuilt.

The uuid is still recoverable from the name, and the fields come from a sidecar
written beside the cache at build time (outside the directory, so it survives
the directory being evicted).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aiida_pino.utils.runtime_repair import cache_identity


def test_plain_uuid_directory_is_recognised():
    uuid = "316f068e-a56f-4d3f-98b5-a72ada9f351a"
    assert cache_identity(uuid) == (uuid, None)


def test_field_discriminated_directory_is_recognised():
    uuid = "316f068e-a56f-4d3f-98b5-a72ada9f351a"
    assert cache_identity(f"{uuid}-92d6e922223e") == (uuid, "92d6e922223e")


@pytest.mark.parametrize(
    "name",
    [
        "not-a-uuid",
        "316f068e-a56f-4d3f-98b5",                              # truncated uuid
        "316f068e-a56f-4d3f-98b5-a72ada9f351a-",                # empty discriminator
        "316f068e-a56f-4d3f-98b5-a72ada9f351a-NOTHEX",          # non-hex discriminator
        "316f068e-a56f-4d3f-98b5-a72ada9f351a-abc-def",         # two discriminators
    ],
)
def test_unrecognised_names_are_rejected(name):
    """A wrong guess here re-exports over some other job's cache."""
    assert cache_identity(name) is None


def test_sidecar_round_trips_the_runtime_fields(tmp_path: Path):
    """Repair cannot invert the field hash, so the fields are recorded beside it."""
    from aiida_pino.utils.runtime_repair import read_cache_sidecar, write_cache_sidecar

    fields = ["Temperature", "Fraction Of Fluid"]
    write_cache_sidecar(tmp_path, "abc-123", runtime_fields=fields)
    assert read_cache_sidecar(tmp_path, "abc-123") == fields


def test_missing_sidecar_reads_as_none(tmp_path: Path):
    """No sidecar means a full-field export, which is the pre-existing behaviour."""
    from aiida_pino.utils.runtime_repair import read_cache_sidecar

    assert read_cache_sidecar(tmp_path, "nothing-here") is None


def test_a_remote_built_cache_records_that_it_has_no_pack_index(tmp_path):
    """A cache built for another machine must not be REPAIRED with an index.

    The discriminator is a hash, so `<uuid>-<hash>` cannot be inverted: without
    this the sidecar said only which fields were used, and a repair would
    re-export the same name WITH pack-index entries — absolute paths into this
    machine's object store, which resolve to nothing on the box that reads them.
    """
    from aiida_pino.utils.runtime_repair import (
        read_cache_allows_pack_index,
        read_cache_sidecar,
        write_cache_sidecar,
    )

    write_cache_sidecar(tmp_path, "remote-cache", runtime_fields=["T"], allow_pack_index=False)
    assert read_cache_allows_pack_index(tmp_path, "remote-cache") is False
    assert read_cache_sidecar(tmp_path, "remote-cache") == ["T"], "fields must still be readable"


def test_a_local_cache_sidecar_is_unchanged_on_disk(tmp_path):
    """Byte-identical to what previous submits wrote, or every sidecar is stale."""
    import json

    from aiida_pino.utils.runtime_repair import write_cache_sidecar

    write_cache_sidecar(tmp_path, "local-cache", runtime_fields=["T", "fl"])
    written = json.loads((tmp_path / "local-cache.fields.json").read_text())
    assert written == ["T", "fl"], "the legacy list form, unchanged"


def test_a_legacy_sidecar_is_read_as_allowing_an_index(tmp_path):
    """Sidecars already on disk say nothing about pack_index; assume the old default."""
    import json

    from aiida_pino.utils.runtime_repair import read_cache_allows_pack_index, read_cache_sidecar

    (tmp_path / "old-cache.fields.json").write_text(json.dumps(["T"]))
    assert read_cache_sidecar(tmp_path, "old-cache") == ["T"]
    assert read_cache_allows_pack_index(tmp_path, "old-cache") is True


def test_pack_index_is_recorded_even_with_no_field_selection(tmp_path):
    """A remote full-field cache still must not be repaired with an index."""
    from aiida_pino.utils.runtime_repair import read_cache_allows_pack_index, write_cache_sidecar

    write_cache_sidecar(tmp_path, "remote-full", runtime_fields=None, allow_pack_index=False)
    assert read_cache_allows_pack_index(tmp_path, "remote-full") is False


def test_nothing_is_written_when_there_is_nothing_to_record(tmp_path):
    from aiida_pino.utils.runtime_repair import write_cache_sidecar

    write_cache_sidecar(tmp_path, "plain", runtime_fields=None)
    assert not (tmp_path / "plain.fields.json").exists()
