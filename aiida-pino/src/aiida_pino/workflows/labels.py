"""Give a child process its caller's label, so the run list is readable.

`_label_submitted_process` labels what a *caller* submits; children stay blank,
and the console renders a blank label as the class name. Measured on the authors' database
2026-08-06: **5406 of 6258 processes (86%) carry no label** — 1249
`Flow3DRunCalculation`, 1244 `Flow3DGenerateWorkChain`, 1184
`RawFlow3DToDatasetCalculation`, 460 `TrainModelWorkChain`, 454
`PinoTrainCalculation`. So a `TrainEvalWorkChain` shows `gaot_sw_arm0_seed43`
and the two rows directly beneath it show class names.

A mixin rather than 20 edits at the `self.submit(...)` call sites: it covers the
sites that exist, and the ones nobody has written yet.
"""

from __future__ import annotations


class InheritsCallerLabel:
    """Mixin: a submitted child with no label of its own takes the caller's.

    A label the caller sets deliberately always wins — this only fills a blank.
    """

    def submit(self, process, inputs=None, **kwargs):  # noqa: D102 - contract above
        label = str(getattr(self.node, "label", "") or "").strip()
        if label:
            target = inputs if isinstance(inputs, dict) else kwargs
            metadata = dict(target.get("metadata") or {})
            if not str(metadata.get("label") or "").strip():
                metadata["label"] = label
                target["metadata"] = metadata
        return super().submit(process, inputs, **kwargs)


#: Input namespace -> the key inside it that names a run, most specific first.
_NAME_SOURCES = (
    ("training_profile", "run_name"),
    ("evaluation_profile", "run_name"),
    ("run_config", "case_name"),
    ("execution_profile", "case_name"),
    ("simulation_profile", "case_name"),
)


def label_from_inputs(node) -> str:
    """A label built from what the process itself was given.

    The same fields `_label_submitted_process` reads at submit time, applied to
    a node that predates it. A short process label plus the run's own name beats
    a bare class name, which is what a blank label renders as.
    """
    try:
        inputs = node.inputs
    except Exception:  # noqa: BLE001 - a process with no inputs namespace
        return ""
    for namespace, key in _NAME_SOURCES:
        try:
            value = getattr(inputs, namespace).get_dict().get(key)
        except Exception:  # noqa: BLE001 - absent namespace or not a Dict
            continue
        name = str(value or "").strip()
        if name:
            kind = str(node.process_label or "").removesuffix("WorkChain").removesuffix("Calculation")
            return f"{kind} | {name}" if kind else name
    return ""


def backfill_process_labels(*, apply: bool = False, profile: str | None = None) -> dict:
    """Label already-stored children from their caller. Reads only unless `apply`.

    Walks up the caller chain rather than one level, because an unlabelled
    workchain between a labelled one and its calcjob would otherwise stop the
    inheritance dead.
    """
    from aiida import orm

    if profile is not None:
        from aiida import load_profile

        load_profile(profile)

    stats = {"processes": 0, "already": 0, "labelled": 0, "no_labelled_caller": 0}
    by_class: dict[str, int] = {}
    for pk in orm.QueryBuilder().append(orm.ProcessNode, project=["id"]).all(flat=True):
        node = orm.load_node(pk)
        stats["processes"] += 1
        if str(node.label or "").strip():
            stats["already"] += 1
            continue
        caller, label = node.caller, ""
        while caller is not None:
            label = str(caller.label or "").strip()
            if label:
                break
            caller = caller.caller
        if not label:
            # No labelled ancestor: 4026 processes are top-level submits made
            # before labelling existed, so there is nothing above them to
            # inherit. Their own inputs still name them — the same fields
            # `_label_submitted_process` reads at submit time.
            label = label_from_inputs(node)
        if not label:
            stats["no_labelled_caller"] += 1
            continue
        stats["labelled"] += 1
        by_class[node.process_label or "?"] = by_class.get(node.process_label or "?", 0) + 1
        if apply:
            node.label = label
    stats["by_class"] = dict(sorted(by_class.items(), key=lambda kv: -kv[1])[:8])
    stats["applied"] = bool(apply)
    return stats
