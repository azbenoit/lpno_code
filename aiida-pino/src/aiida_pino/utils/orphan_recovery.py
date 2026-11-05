"""Recover the completed cases of a KILLED batch workchain.

`verdi process kill` skips `finalize`, which is where `Flow3DBatchWorkChain`
files its simulations into a named base dataset. Killing a sweep at case 20 of
40 therefore leaves 20 cases ingested but UNFILED: absent from every dataset and
folder view, not selectable by name, indistinguishable from cases that never
ran. Measured on one batch killed at 20 of 40. Killing is the right response to a design flaw found
mid-sweep, so the behaviour penalises the correct call.

Deliberately a RECOVERY command rather than an on-kill hook: after a kill you
usually want to split valid from invalid before filing. That recovery
was two datasets — 16 valid and 4 floor-invalid marked do-not-train — which an
automatic hook would have bundled into one.
"""

from __future__ import annotations

from typing import Any, Iterable


def plan_orphan_filing(
    *,
    completed: Iterable[dict[str, Any]],
    owned_uuids: set[str],
    dataset_name: str,
    only_cases: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Decide which completed cases still need filing. Pure; no DB writes.

    `completed` is a list of `{"uuid", "case_name"}`. `owned_uuids` are the
    simulations a base dataset already claims — they are skipped rather than
    re-filed, because a simulation belongs to at most one base dataset and
    `create_base_dataset` refuses to steal.
    """
    rows = [dict(row) for row in completed]
    if only_cases is not None:
        wanted = [str(name) for name in only_cases]
        known = {str(row.get("case_name")) for row in rows}
        missing = [name for name in wanted if name not in known]
        if missing:
            # A typo must not silently file fewer cases than intended -- the
            # whole point of this command is that cases went missing quietly.
            raise ValueError(
                f"case(s) not among the batch's completed children: {', '.join(missing)}. "
                f"Known: {', '.join(sorted(known))}"
            )
        rows = [row for row in rows if str(row.get("case_name")) in set(wanted)]

    to_file = [row for row in rows if str(row.get("uuid")) not in set(owned_uuids)]
    return {
        "dataset_name": str(dataset_name),
        "to_file": to_file,
        "already_filed": len(rows) - len(to_file),
        "total_considered": len(rows),
    }


def completed_children_of_batch(pk: int) -> list[dict[str, Any]]:
    """`{"uuid", "case_name", "pk"}` for every simulation a batch actually produced.

    Reads the workchain's descendants rather than `batch_summary`, which a
    killed batch never wrote.
    """
    from aiida import orm

    from aiida_pino.data import PinoSimulationData

    node = orm.load_node(pk)
    seen: dict[str, dict[str, Any]] = {}
    for descendant in [node, *_walk_called(node)]:
        for link in descendant.base.links.get_outgoing().all():
            target = link.node
            if isinstance(target, PinoSimulationData):
                seen[str(target.uuid)] = {
                    "uuid": str(target.uuid),
                    "pk": target.pk,
                    "case_name": str(target.base.extras.all.get("case_name") or target.label or ""),
                }
    return sorted(seen.values(), key=lambda row: row["case_name"])


def _walk_called(node, _depth: int = 0):
    """Every process this one called, recursively (batch -> generate -> run)."""
    if _depth > 8:  # a batch nests three deep; the cap is a cycle guard
        return []
    out = []
    for link in node.base.links.get_outgoing().all():
        target = link.node
        if hasattr(target, "base") and hasattr(target, "process_state"):
            out.append(target)
            out.extend(_walk_called(target, _depth + 1))
    return out


def batch_dataset_profile(pk: int) -> dict[str, Any]:
    """The batch's own `dataset_profile` input, which survives a kill."""
    from aiida import orm

    node = orm.load_node(pk)
    for link in node.base.links.get_incoming().all():
        if link.link_label == "dataset_profile" and isinstance(link.node, orm.Dict):
            return dict(link.node.get_dict())
    return {}
