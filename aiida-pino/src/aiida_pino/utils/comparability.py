"""Whether two evaluations measure the same thing, stated as a set difference.

Every axis here cost a wrong conclusion in one week: a val-scored number
compared against test-scored ones and published; two evals of one model at
different frame counts producing a "regime gap" that was an artifact of N;
full-resolution GAOT against half-resolution RIGNO, unseparated; and a
steady-protocol eval that silently re-derived lab frame.

The purpose is not to block comparisons. It is that a difference is VISIBLE to
someone who did not know to look — the same reason `numerical_regime` is an
extra rather than a value inside a summary Dict.
"""

from __future__ import annotations

#: The axes that decide whether two evaluation numbers mean the same thing.
COMPARABILITY_AXES = (
    "dataset_uuid",
    "split",
    "derivation_kind",
    "served_grid",
    "eval_frame_stride",
    "training_stride",
)

#: Axes whose disagreement invalidates MAX statistics but not per-cell means.
#: Peak statistics grow as sqrt(2 ln N), so a frame-count difference manufactures
#: a gap; means are robust to it (23.8 -> 23.9 K across 109 -> 121 frames).
_MAX_ONLY_AXES = ("eval_frame_stride",)


def max_statistics_comparable(axes_by_pk: dict) -> bool:
    """Whether peak/max metrics may be diffed across these evaluations."""
    for axis in _MAX_ONLY_AXES:
        values = {repr(a.get(axis)) for a in axes_by_pk.values()}
        if len(values) > 1:
            return False
    return True


def compare_axis_sets(axes_by_pk: dict) -> dict:
    """Which axes differ, which are unknown, and what may still be compared.

    An UNKNOWN value is never treated as agreement. That is the failure this
    exists to prevent: a missing training stride reading as "the same as the
    other one" is how a full-res model got compared with a half-res one.

    Unknowns are reported at two strengths. `unknown_axes` is every axis some
    evaluation did not record; `asymmetric_unknown_axes` is the subset recorded
    on some and not others, which additionally says the two were produced by
    different generations of the metric code.
    """
    differing: list[str] = []
    unknown: list[str] = []
    asymmetric: list[str] = []
    values: dict[str, dict] = {}
    for axis in COMPARABILITY_AXES:
        per_pk = {pk: axes.get(axis) for pk, axes in axes_by_pk.items()}
        values[axis] = per_pk
        if any(value is None for value in per_pk.values()):
            unknown.append(axis)
            # Absent on ONE side is not the same fact as absent on both. Absent
            # on both means neither eval ever recorded the axis -- uninformative.
            # Absent on one means the two were produced by different generations
            # of the metric code, which is a strong signal on its own: the
            # 2026-08-21 fusion-IoU anomaly (0.407 against 0.766) was exactly
            # this, a control whose metrics predate `n_frames` compared against
            # an eval that scored 643 frames. Collapsed together, a reader sees
            # `differing_axes: []` and has to notice `unknown` is doing the work.
            if any(value is not None for value in per_pk.values()):
                asymmetric.append(axis)
            continue
        if len({repr(value) for value in per_pk.values()}) > 1:
            differing.append(axis)

    max_ok = max_statistics_comparable(axes_by_pk) and not unknown
    mean_blockers = [a for a in differing if a not in _MAX_ONLY_AXES]
    return {
        "pks": sorted(axes_by_pk),
        "differing_axes": differing,
        "unknown_axes": unknown,
        #: Subset of `unknown_axes`: recorded on some evaluations and not
        #: others. Refines `unknown_axes`, never contradicts it.
        "asymmetric_unknown_axes": asymmetric,
        "values": values,
        "max_statistics_comparable": max_ok,
        "mean_statistics_comparable": not mean_blockers and not unknown,
        "comparable": not differing and not unknown,
    }


def _metric_field(node, key):
    """One field out of an evaluation's stored metrics, or None."""
    try:
        for link in node.base.links.get_outgoing().all():
            target = link.node
            getter = getattr(target, "get_dict", None)
            if getter is None:
                continue
            payload = getter() or {}
            if key in payload:
                return payload[key]
            metrics = payload.get("metrics")
            if isinstance(metrics, dict) and key in metrics:
                return metrics[key]
    except Exception:  # noqa: BLE001 - unknown stays unknown
        return None
    return None


def _dataset_of(node):
    """The dataset an evaluation actually consumed."""
    for link in node.base.links.get_incoming().all():
        if link.link_label in ("dataset", "eval_dataset"):
            return link.node
    return None


def comparability_axes(node) -> dict:
    """Read the six axes off a stored evaluation. Unknown stays None.

    Read from the NODE GRAPH rather than from `training_profile.env`:
    `dataset_derivation.py` POPS `PINO_LAZY_STRIDE` after baking it into the
    variant, so auditing the profile env misses 15 of 52 strided models — and a
    comparison tool that read env would repeat the error it exists to prevent.
    """
    axes = dict.fromkeys(COMPARABILITY_AXES)
    extras = node.base.extras
    # NOT `eval_stride`: that extra is the SPATIAL stride [i,j,k]
    # (`data/pino_model_evaluation.py`), while the frame count comes from
    # PINO_EVAL_FRAME_STRIDE, which stamps `eval_frame_stride` into the stored
    # METRICS (`workflows/evaluate_model.py:46`). Reading the spatial one here
    # would make the max-statistics refusal key off the wrong field — the exact
    # class of mix-up this module exists to catch. The spatial stride is already
    # represented by `served_grid`.
    frame_stride = extras.get("eval_frame_stride", None)
    if frame_stride is None:
        frame_stride = _metric_field(node, "eval_frame_stride")
    if frame_stride is not None:
        axes["eval_frame_stride"] = frame_stride
    training_stride = extras.get("training_stride", None)
    if training_stride is None and extras.get("eval_full_resolution", None) is not None:
        training_stride = "full" if extras.get("eval_full_resolution") else None
    if training_stride is not None:
        axes["training_stride"] = str(training_stride)

    dataset = _dataset_of(node)
    if dataset is None:
        return axes
    axes["dataset_uuid"] = str(dataset.uuid)
    try:
        metadata = dataset.get_dataset_metadata()
    except Exception:  # noqa: BLE001 - a missing metadata block is "unknown", not an error
        return axes
    profile = metadata.get("derivation_profile") or {}
    if profile:
        axes["derivation_kind"] = (
            "steady"
            if profile.get("time_avg_window", 1) and int(profile.get("time_avg_window", 1)) > 1
            else ("moving-ref" if profile.get("moving_ref_frame") else "lab")
        )
    shape = metadata.get("spatial_shape")
    if shape:
        axes["served_grid"] = "x".join(str(int(v)) for v in shape)
    label = str(dataset.label or "")
    for candidate in ("test", "val", "train"):
        if f"[{candidate}]" in label or label.endswith(f"-{candidate}"):
            axes["split"] = candidate
            break
    return axes


def compare_evaluations(pks) -> dict:
    """Axes for each evaluation, plus what differs between them."""
    from aiida import orm

    axes_by_pk = {int(pk): comparability_axes(orm.load_node(int(pk))) for pk in pks}
    report = compare_axis_sets(axes_by_pk)
    report["axes_by_pk"] = axes_by_pk
    return report
