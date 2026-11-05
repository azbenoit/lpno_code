"""Where a freshly ingested simulation goes.

One resolver for both FLOW-3D entry points, so a single run and a batch cannot
disagree about which dataset their cases land in.

Two ways to say where:

- **explicitly**, by passing a `target_dataset` node — the run joins exactly that
  dataset and nothing else is consulted;
- **by name**, the historical behaviour — the batch keyword (or `flow3d_runs`
  for an unnamed single run) is looked up among base datasets and extended if it
  exists, created if it does not.

Explicit wins, because a name is ambiguous the moment two datasets share a
display name and a caller who names a node has said something more specific than
a caller who names a string.
"""

from __future__ import annotations

from typing import Any


def resolve_filing_target(*, target_dataset, name: str):
    """The base dataset to file into, or None to create one called `name`.

    `target_dataset` may be a `PinoDatasetCollectionData` (or None). A target
    that has been superseded is refused rather than silently extended: its
    replacement is what everything else now reads, so adding cases to it would
    put them somewhere no listing shows.
    """
    from aiida_pino.data.dataset_hierarchy import iter_base_datasets

    from .flow3d_batch import dataset_display_name

    if target_dataset is not None:
        if bool(target_dataset.base.extras.get("superseded", False)):
            raise ValueError(
                f"dataset {target_dataset.pk} is superseded (replaced by "
                f"{target_dataset.base.extras.get('replaced_by_pk', '?')}); "
                "file into the replacement instead"
            )
        return target_dataset
    return next((d for d in iter_base_datasets() if dataset_display_name(d) == name), None)


def auto_description(name: str, simulations, *, batch_keyword: str | None = None) -> str:
    """A factual description for a dataset nobody wrote one for.

    Refusing to file would throw away a finished solve,
    so automated filing never blocks on this — but it must not leave a blank
    either: 713 of 802 collections had no description, which is how a dataset's
    purpose ends up living only in whoever made it. Stating what it is made of
    is worth more than an empty string, and it is honestly marked as generated.
    """
    powers, speeds = [], []
    for sim in simulations:
        meta = getattr(sim, "get_simulation_metadata", dict)() or {}
        if isinstance(meta.get("laser_power"), (int, float)):
            powers.append(float(meta["laser_power"]))
        if isinstance(meta.get("V_scan"), (int, float)):
            speeds.append(float(meta["V_scan"]))
    bits = [f"{len(list(simulations))} FLOW-3D case(s) filed automatically as `{name}`"]
    if batch_keyword:
        bits.append(f"batch `{batch_keyword}`")
    if powers:
        bits.append(f"P {min(powers):g}-{max(powers):g} W" if min(powers) != max(powers) else f"P {powers[0]:g} W")
    if speeds:
        bits.append(f"V {min(speeds):g}-{max(speeds):g} m/s" if min(speeds) != max(speeds) else f"V {speeds[0]:g} m/s")
    return "; ".join(bits) + ". (auto-generated: no description was supplied — replace it.)"


def dataset_metadata_from_profile(profile: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """The dataset extras a `dataset_profile` contributes, defined once.

    `batch_keyword` NAMES the dataset; `keywords` are the dataset's own tags.
    The two read alike and that is exactly how they diverged: the batch path
    forwarded `keywords`, the generate path hand-built a dict with only
    `description` and `batch_keyword`, and a one-off or requeue produced a
    dataset with zero keywords -- silently, since `dataset_profile_metadata_gaps`
    sees both of ITS fields present. `keywords` is what the `old_physics` /
    `experimentally_anchored` era selection and every console keyword query read.
    """
    return {
        "description": profile.get("description"),
        "keywords": profile.get("keywords"),
        "batch_keyword": profile.get("batch_keyword"),
        **extra,
    }


def file_simulations(
    simulations,
    *,
    name: str,
    target_dataset=None,
    extra_metadata: dict[str, Any] | None = None,
):
    """Attach `simulations` to their base dataset. Returns `(dataset, action)`.

    Only simulations that do not already belong to a base dataset are filed —
    a case is owned by one base dataset, and a requeue must not steal it from
    the sweep it already belongs to. Returns `(None, "nothing")` when there is
    nothing left to attach.
    """
    from aiida_pino.data.dataset_hierarchy import (
        base_dataset_of,
        create_base_dataset,
        extend_base_dataset,
    )

    unattached = [sim for sim in simulations if base_dataset_of(sim) is None]
    if not unattached:
        return None, "nothing"
    existing = resolve_filing_target(target_dataset=target_dataset, name=name)
    if existing is not None:
        return extend_base_dataset(existing, unattached, name=name), "extended"
    metadata = dict(extra_metadata or {})
    if not str(metadata.get("description") or "").strip():
        metadata["description"] = auto_description(
            name, unattached, batch_keyword=metadata.get("batch_keyword")
        )
        metadata["description_auto_generated"] = True
    dataset, _created = create_base_dataset(unattached, name=name, extra_metadata=metadata)
    return dataset, "created"
