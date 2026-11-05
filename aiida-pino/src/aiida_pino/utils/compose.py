"""Compose dataset collections *through a calcfunction*, so membership is a graph edge.

Review criticism A, membership half: 18,707 of 19,822 (simulation, collection)
pairs asserted by `member_simulation_uuids` had no link-path equivalent, and
only 476 of 3,093 simulations were link-reachable to any collection. The
consequence a reviewer meets first is that `verdi archive create -N 1601` on the
300-member reference dataset exports **one node and zero files** — the attribute
says what is in the dataset, the graph does not.

The pattern was already implemented and running in the console
(`create_collection`); it simply was not what the dataset-filing paths used.
This module is that pattern, in one place, for both.
"""

from __future__ import annotations

from typing import Any


def compose_dataset_calcfunction(parameters, **inputs):
    """Run the composition calcfunction (deferred import keeps module load light)."""
    from aiida.engine import calcfunction

    from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData

    @calcfunction
    def compose_dataset(parameters, **inputs):
        """Union the member simulations of the input datasets (plus any
        explicitly picked simulations) into one composed collection."""
        meta = dict(parameters["metadata"])
        pool = {}
        for node in inputs.values():
            if isinstance(node, PinoSimulationData):
                pool[str(node.uuid)] = node
            elif isinstance(node, PinoDatasetCollectionData):
                for sim in node.iter_member_simulations():
                    pool[str(sim.uuid)] = sim
        members = [pool[str(u)] for u in parameters["member_uuids"]]
        return PinoDatasetCollectionData.from_simulations(members, metadata=meta)

    return compose_dataset(parameters, **inputs)


def compose_collection(
    sims,
    metadata: dict[str, Any],
    *,
    extra_parameters: dict[str, Any] | None = None,
    **inputs,
):
    """Reuse an identical collection, else create one through the calcfunction.

    Returns `(node, created)`, like `get_or_create_dataset_collection` — which
    this replaces wherever the caller knows the provenance inputs. Passing the
    member simulations as `inputs` is what puts membership in the graph:
    `simulation -> compose_dataset -> collection`.
    """
    from aiida.orm import Dict

    from aiida_pino.data.pino_dataset_collection import validate_collection_members
    from aiida_pino.utils.model_registry import (
        find_reusable_dataset_collection,
        supersede_equivalent_dataset_collections,
    )

    members, _ = validate_collection_members(
        sims, field_policy=str(metadata.get("field_policy") or "strict")
    )
    existing = find_reusable_dataset_collection(members, metadata=metadata)
    if existing is not None:
        return existing, False
    parameters = {"metadata": metadata, "member_uuids": [s.uuid for s in members]}
    parameters.update(extra_parameters or {})
    node = compose_dataset_calcfunction(Dict(dict=parameters), **inputs)
    supersede_equivalent_dataset_collections(node)
    return node, True


def member_inputs(members, prefix: str = "simulation") -> dict[str, Any]:
    """`{simulation_0: node, ...}` link labels for a member list.

    Zero-padded so a 300-member dataset reads in order in `verdi node show`.
    """
    return {f"{prefix}_{index:04d}": member for index, member in enumerate(members)}
