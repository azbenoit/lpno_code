"""Zero-copy case NPZs: read them straight out of AiiDA's object-store packs.

On the the authors' database profile every `payload.npz` is stored **uncompressed and
contiguous** inside a disk-objectstore pack (measured 2026-08-05:
`size == pack_length` on every sampled object — the NPZs are already deflated
internally, so the container never re-compresses them). A per-case NPZ is
therefore a byte range of a pack file, and copying it into the staging cache
buys nothing: the cache reached 352 GB of duplicated bytes on a disk at 88%,
and rebuilding one dataset's worth used to take hours.

With a `pack_index.json` in the staged dataset dir, this module lets
`NpzDirStore` open each case as a bounded window over the pack instead:
staging writes a few KB of JSON, and reads are ordinary seeks into a file the
OS page cache already knows.

Safety model, in order of the failure that motivates it:

* **Repacking moves offsets — so offsets are not trusted.** `verdi storage
  maintain` rewrites packs, and it REORDERS objects, not merely truncates: a
  stale window can land on a DIFFERENT object of the same size, and since these
  datasets hold same-grid cases, equal sizes are the norm. The EOCD check alone
  passed that case (reproduced 2026-08-14: asked for case_0, got case_1,
  validation passed — silent wrong training data). Each open therefore
  re-resolves the object's CURRENT location by content hash through the
  container's own index (`disk_objectstore.Container.get_object_meta`), which a
  repack updates atomically. `disk_objectstore` is a small standalone library,
  not AiiDA — no profile, no session, no ORM — so the no-AiiDA-in-workers rule
  holds. The stored offset is only a fallback for pre-hashkey indexes and for a
  container that cannot be found, and that fallback keeps the EOCD+bounds check
  with its documented same-size blind spot.
* **Kill switch.** `PINO_PACK_INDEX_DISABLE=1` makes the loader ignore every
  index, so one env var returns any job to materialized files (after a
  re-stage) without a code change.
* **Mixed dirs are normal.** An index covers the members whose payloads were
  locatable; anything else is staged as a real file beside it. Real files win
  on name collisions.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

PACK_INDEX_FILENAME = "pack_index.json"
PACK_INDEX_FORMAT = "pino_pack_index_v1"
_EOCD_MAGIC = b"PK\x05\x06"

#: One Container handle per container root, per process. DataLoader workers are
#: separate processes and each builds its own on first use; the handle holds a
#: read-only SQLite session over the container's own index.
_CONTAINERS: Dict[str, Any] = {}


def _container_for(pack_path: str):
    """The disk-objectstore Container owning `pack_path`, or None.

    Pack files live at `<container>/packs/<pack_id>`, so the container root is
    two levels up; `config.json` at that root is what makes it a container
    rather than a coincidence of directory shape. Returns None — never raises —
    when disk_objectstore is not importable or the root is not a container:
    the caller falls back to the stored offset.
    """
    root = str(Path(pack_path).parent.parent)
    if root in _CONTAINERS:
        return _CONTAINERS[root]
    container = None
    try:
        if (Path(root) / "config.json").is_file():
            from disk_objectstore import Container

            candidate = Container(root)
            if candidate.is_initialised:
                container = candidate
    except Exception:  # noqa: BLE001 - resolution is best-effort by design
        container = None
    _CONTAINERS[root] = container
    return container


class PackSliceReader(io.RawIOBase):
    """A seekable read-only window ``[offset, offset+size)`` over a pack file."""

    def __init__(self, path: str, offset: int, size: int):
        self._fh = open(path, "rb")
        self._offset = int(offset)
        self._size = int(size)
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self._pos = pos
        elif whence == os.SEEK_CUR:
            self._pos += pos
        elif whence == os.SEEK_END:
            self._pos = self._size + pos
        else:
            raise ValueError(f"bad whence {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buffer) -> int:
        remaining = self._size - self._pos
        if remaining <= 0:
            return 0
        want = min(len(buffer), remaining)
        self._fh.seek(self._offset + self._pos)
        data = self._fh.read(want)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


class PackSlicePath:
    """Quacks like the ``Path`` of a case NPZ, backed by a pack byte range.

    Supports exactly what `NpzDirStore`/`npz_datasource` do with a case path:
    `.name`/`.stem`/`.suffix`/`.parent`/`.with_name()` (sidecars live as real
    files in the staged dir), `.exists()`, sorting by name, and `readable()`
    for `np.load`.
    """

    __slots__ = ("name", "parent", "pack_path", "offset", "size", "hashkey")

    def __init__(self, staged_dir: Path, entry: Dict[str, Any]):
        self.name = str(entry["filename"])
        self.parent = Path(staged_dir)
        self.pack_path = str(entry["pack_path"])
        self.offset = int(entry["offset"])
        self.size = int(entry["size"])
        self.hashkey = entry.get("hashkey")

    # ---- Path-alikes used by the datasource --------------------------------
    @property
    def stem(self) -> str:
        return self.name.rsplit(".", 1)[0]

    @property
    def suffix(self) -> str:
        return "." + self.name.rsplit(".", 1)[1] if "." in self.name else ""

    def with_name(self, name: str) -> Path:
        return self.parent / name

    def exists(self) -> bool:
        try:
            return os.path.getsize(self.pack_path) >= self.offset + self.size
        except OSError:
            return False

    def __fspath__(self):  # keep os.fspath()/np.load(str-path) from silently opening the whole pack
        raise TypeError(
            f"{self.name} is a pack slice, not a file on disk; open it with .readable() "
            "or re-stage the dataset without the pack index"
        )

    def __repr__(self) -> str:
        return f"PackSlicePath({self.name} @ {Path(self.pack_path).name}+{self.offset}:{self.size})"

    def __lt__(self, other) -> bool:
        return self.name < getattr(other, "name", str(other))

    # ---- the actual read ----------------------------------------------------
    def readable(self) -> io.BufferedReader:
        """A fresh bounded reader, validated. `np.load` accepts it directly.

        Resolution order:
        1. By CONTENT HASH through the container's own index — repack-proof,
           because a repack updates that index in the same operation that moves
           the bytes. Still zero-copy: the resolved location is read as a
           bounded window exactly like before, just at the CURRENT offset.
        2. The stored offset, only when the entry predates hashkeys or the
           container cannot be found. This path keeps the EOCD check and its
           documented blind spot (a same-size neighbour after a repack).
        """
        resolved = self._resolve_current()
        if resolved is not None:
            return resolved
        if not self.exists():
            raise FileNotFoundError(
                f"[pack-index] {self.name}: pack {self.pack_path} is missing or shorter than "
                f"offset {self.offset} + size {self.size}. The container was likely repacked "
                "(`verdi storage maintain`) after this dataset was staged — re-stage it "
                "(delete its runtime-cache entry; it rebuilds on next use)."
            )
        reader = io.BufferedReader(PackSliceReader(self.pack_path, self.offset, self.size))
        # A ZIP archive ends with the end-of-central-directory record. 22 bytes
        # from the window's tail decide whether this byte range still IS the
        # NPZ it was indexed as, before numpy reads structure from garbage.
        reader.seek(-22, os.SEEK_END)
        tail = reader.read(22)
        reader.seek(0)
        if _EOCD_MAGIC not in tail:
            reader.close()
            raise ValueError(
                f"[pack-index] {self.name}: the byte range in {Path(self.pack_path).name} no longer "
                "holds a ZIP archive — the container was repacked after staging. Re-stage the "
                "dataset (delete its runtime-cache entry; it rebuilds on next use)."
            )
        return reader

    def _resolve_current(self) -> io.BufferedReader | None:
        """The object at its CURRENT location, by hash. None -> use the fallback."""
        if not self.hashkey:
            return None
        container = _container_for(self.pack_path)
        if container is None:
            return None
        try:
            meta = container.get_object_meta(self.hashkey)
        except Exception as exc:  # noqa: BLE001 - unknown hash IS the loud case
            raise FileNotFoundError(
                f"[pack-index] {self.name}: the container at "
                f"{Path(self.pack_path).parent.parent} does not know object "
                f"{self.hashkey} — it was deleted (evict?) after this dataset was "
                "staged. Re-stage the dataset."
            ) from exc
        kind = str(getattr(meta, "type", ""))
        if kind.endswith("missing"):
            raise FileNotFoundError(
                f"[pack-index] {self.name}: object {self.hashkey} is gone from the "
                "container — deleted after staging. Re-stage the dataset."
            )
        if kind.endswith("packed") and not meta.pack_compressed:
            pack = str(Path(self.pack_path).parent.parent / "packs" / str(meta.pack_id))
            return io.BufferedReader(
                PackSliceReader(pack, int(meta.pack_offset), int(meta.pack_length))
            )
        if kind.endswith("loose"):
            loose = (Path(self.pack_path).parent.parent / "loose"
                     / self.hashkey[:2] / self.hashkey[2:])
            if loose.is_file():
                return open(loose, "rb")
        # Re-packed WITH compression (the staging-time eligibility no longer
        # holds): a bounded raw window would hand zlib bytes to np.load. Rare,
        # so correctness over speed — materialise through the container.
        with container.get_object_stream(self.hashkey) as stream:
            return io.BytesIO(stream.read())  # np.load accepts it; BufferedReader cannot wrap BytesIO


def pack_index_disabled() -> bool:
    return os.environ.get("PINO_PACK_INDEX_DISABLE", "").strip() not in ("", "0", "false", "False")


def load_pack_index(directory: Path) -> List[PackSlicePath]:
    """The pack-slice cases of a staged dir; [] when absent or disabled."""
    index_path = Path(directory) / PACK_INDEX_FILENAME
    if pack_index_disabled() or not index_path.is_file():
        return []
    document = json.loads(index_path.read_text(encoding="utf-8"))
    if document.get("format") != PACK_INDEX_FORMAT:
        raise ValueError(
            f"[pack-index] {index_path}: unknown format {document.get('format')!r} "
            f"(this loader reads {PACK_INDEX_FORMAT})."
        )
    return [PackSlicePath(Path(directory), entry) for entry in document.get("entries", [])]


def open_case(path) -> Any:
    """What `np.load` should be handed for a case path, real or sliced."""
    if isinstance(path, PackSlicePath):
        return path.readable()
    return path
