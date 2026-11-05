"""A repack must never silently substitute one case's data for another's.

Reproduced 2026-08-14 before the fix: two same-shape cases produce identical
1062-byte NPZs (the normal situation here — these datasets hold same-grid
cases). After reordering the pack, a read asking for case_0 returned case_1 and
the EOCD validation PASSED, because the last 22 bytes of the window prove it
ends at *a* ZIP, not at *the* ZIP indexed. Fix (c): each open re-resolves the
object's CURRENT location by content hash through the container's own index,
which a repack updates in the same operation that moves the bytes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

disk_objectstore = pytest.importorskip("disk_objectstore")
from disk_objectstore import Container  # noqa: E402

from lpno.custom_utils import pack_slice  # noqa: E402
from lpno.custom_utils.pack_slice import PackSlicePath  # noqa: E402


def _npz_bytes(case: str) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, Temperature=np.full((4, 4, 4), float(case[-1])),
             case_name=np.str_(case))
    return buf.getvalue()


@pytest.fixture
def container(tmp_path):
    pack_slice._CONTAINERS.clear()  # the cache is per-process; tests share one
    c = Container(tmp_path / "container")
    c.init_container()
    yield c
    pack_slice._CONTAINERS.clear()
    c.close()


def _entry(container: Container, hashkey: str, name: str) -> dict:
    meta = container.get_object_meta(hashkey)
    root = Path(container.get_folder())
    return {
        "filename": name,
        "pack_path": str(root / "packs" / str(meta.pack_id)),
        "offset": int(meta.pack_offset),
        "size": int(meta.pack_length),
        "hashkey": hashkey,
    }


def test_a_stale_offset_onto_a_same_size_neighbour_returns_the_RIGHT_case(container, tmp_path):
    """The exact reproduced failure: equal-size blobs, index points at the wrong one."""
    blob0, blob1 = _npz_bytes("case_0"), _npz_bytes("case_1")
    assert len(blob0) == len(blob1), "same-shape cases must produce same-size blobs"
    keys = container.add_objects_to_pack([blob0, blob1])

    entry = _entry(container, keys[0], "case_0.npz")
    # Simulate the repack: the index remembers case_0's window, but that byte
    # range now belongs to case_1 (same size, so bounds and EOCD both pass).
    wrong = _entry(container, keys[1], "case_1.npz")
    stale = {**entry, "offset": wrong["offset"], "size": wrong["size"],
             "pack_path": wrong["pack_path"]}

    got = np.load(PackSlicePath(tmp_path, stale).readable(), allow_pickle=True)
    assert str(got["case_name"]) == "case_0", \
        "resolution by hash must return the indexed case, not whatever sits at the offset"


def test_a_deleted_object_fails_loudly_not_with_garbage(container, tmp_path):
    (key,) = container.add_objects_to_pack([_npz_bytes("case_0")])
    entry = _entry(container, key, "case_0.npz")
    entry["hashkey"] = "0" * 64  # a hash the container has never seen
    with pytest.raises(FileNotFoundError, match="does not know object"):
        PackSlicePath(tmp_path, entry).readable()


def test_pre_hashkey_indexes_keep_the_old_path(container, tmp_path):
    """Entries staged before hashkeys existed still open via the stored offset."""
    (key,) = container.add_objects_to_pack([_npz_bytes("case_0")])
    entry = _entry(container, key, "case_0.npz")
    entry["hashkey"] = None
    got = np.load(PackSlicePath(tmp_path, entry).readable(), allow_pickle=True)
    assert str(got["case_name"]) == "case_0"


def test_a_loose_object_resolves_too(container, tmp_path):
    """An object not yet packed lives under loose/; hash resolution finds it."""
    stream = io.BytesIO(_npz_bytes("case_0"))
    key = container.add_streamed_object(stream)
    root = Path(container.get_folder())
    entry = {"filename": "case_0.npz", "hashkey": key,
             # a plausible-but-wrong pack window: only the hash is right
             "pack_path": str(root / "packs" / "0"), "offset": 0, "size": 10}
    got = np.load(PackSlicePath(tmp_path, entry).readable(), allow_pickle=True)
    assert str(got["case_name"]) == "case_0"


def test_index_round_trip_still_loads(container, tmp_path):
    """The staged-dir contract end to end: write pack_index.json, load it back."""
    (key,) = container.add_objects_to_pack([_npz_bytes("case_0")])
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / pack_slice.PACK_INDEX_FILENAME).write_text(json.dumps({
        "format": pack_slice.PACK_INDEX_FORMAT,
        "entries": [_entry(container, key, "case_0.npz")],
    }))
    paths = pack_slice.load_pack_index(staged)
    assert len(paths) == 1
    got = np.load(paths[0].readable(), allow_pickle=True)
    assert str(got["case_name"]) == "case_0"
