"""Was this value recorded when the node was made, or reconstructed later?

The two are not equally trustworthy and the database could not tell them apart.
A value written at materialisation can only be wrong through a code bug. A
backfilled one is an inference made after the fact -- from a manifest, a
sibling, a log tail, a case-name match -- and inference carries error the
original never had: in one day of work, a backfill read the wrong `[step: N]`
line, another mistook a peak-frame width for a section width, and a lineage
pointer said "derived from" on case-name evidence that the grids later refuted.
Each was caught, but nothing on the node would have shown a reader which values
were exposed to that class of mistake.

A few keys already carry a semantic source (`material_source`,
`training_seed_source`, `steps_trained_source`, `training_stride_source`).
Those say WHERE a value came from conceptually -- they do not say whether it was
written at creation. `material_source: "manifest"` sits on 3,894 simulations and
every one of them was stamped by a backfill; the tag is true and still answers a
different question.

`backfilled_keys` is that missing answer: the sorted list of extras written
after the node was stored. One key to look at, one filter to find them:

    filters={"extras.backfilled_keys": {"contains": ["held_out"]}}
"""

from __future__ import annotations

from typing import Any, Iterable

KEY = "backfilled_keys"


def mark_backfilled(node, keys: Iterable[str], *, apply: bool = True) -> list[str]:
    """Record that `keys` were written after `node` was created.

    Idempotent and additive: re-running a backfill does not duplicate entries,
    and marking one key never erases another's mark.
    """
    existing = list(node.base.extras.get(KEY, None) or [])
    merged = sorted({*existing, *(str(k) for k in keys if k)})
    if apply and merged != existing:
        node.base.extras.set(KEY, merged)
    return merged


def is_backfilled(node, key: str) -> bool:
    """Whether `key` on this node was reconstructed rather than recorded."""
    return str(key) in (node.base.extras.get(KEY, None) or [])


def provenance_of(node, key: str) -> dict[str, Any]:
    """How `key` got onto this node, as far as the node can say.

    `mode` is "backfilled" or "native"; "native" means only that no backfill
    marked it -- a value written before this marker existed and never
    backfilled since reads as native, which is the honest default but not a
    guarantee. `source` carries the semantic tag when the key has one.
    """
    value = node.base.extras.get(key, None)
    source_key = f"{key}_source"
    source = node.base.extras.get(source_key, None)
    if source is None and key in ("solidus_k", "material_name", "liquidus_k", "material_file"):
        source = node.base.extras.get("material_source", None)
    return {
        "key": key,
        "value": value,
        "present": value is not None,
        "mode": "backfilled" if is_backfilled(node, key) else "native",
        "source": source,
        "source_key": source_key if node.base.extras.get(source_key, None) is not None else None,
    }
