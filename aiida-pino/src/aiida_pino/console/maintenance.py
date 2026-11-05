"""Targeted object-store reclaim: free the disk left behind by eviction.

Eviction drops a node's reference to its arrays, but on a packed store the
bytes live inside a pack file and are only reclaimed by a repack. AiiDA's
``verdi storage maintain --full`` repacks *every* pack (an exclusive-lock churn
over the whole store); this repacks only the packs that actually hold
unreferenced bytes, so the lock window and I/O scale with what was evicted, not
with the 700 GB store. Deleting the objects uses the same
``get_unreferenced_keyset`` mark-and-sweep as ``maintain``, so it is exactly as
safe: only objects no node references are ever removed.
"""

from __future__ import annotations

from typing import Any


def plan_reclaim(container, unref_keys: list[str]) -> dict[str, Any]:
    """What deleting these unreferenced objects frees, and which packs to repack.

    Loose unreferenced objects are freed immediately by the delete (no repack);
    packed ones need their pack repacked to reclaim the hole.
    """
    loose_bytes = 0
    packs: dict[int, int] = {}
    for _hashkey, meta in container.get_objects_meta(list(unref_keys)):
        size = int(getattr(meta, "size", 0) or 0)
        pack_id = getattr(meta, "pack_id", None)
        if pack_id is None:
            loose_bytes += size
        else:
            packs[pack_id] = packs.get(pack_id, 0) + size
    packed_bytes = sum(packs.values())
    return {
        "unreferenced": len(unref_keys),
        "reclaimable_bytes": loose_bytes + packed_bytes,
        "loose_bytes": loose_bytes,
        "packed_bytes": packed_bytes,
        "packs": sorted(packs),
        "pack_reclaim_bytes": {int(k): int(v) for k, v in packs.items()},
    }
