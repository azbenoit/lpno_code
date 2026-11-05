"""Parser for VTK-to-native-dataset conversion jobs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aiida import orm
from aiida.common import exceptions
from aiida.parsers import Parser

from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData
from aiida_pino.data.pino_simulation import payload_content_hash
from aiida_pino.utils.model_registry import supersede_equivalent_dataset_collections


def _query_simulation_by_fingerprint(fingerprint: str, *, oldest_first: bool = False) -> PinoSimulationData | None:
    """Return one stored simulation with the given fingerprint, or None."""
    builder = orm.QueryBuilder()
    builder.append(
        PinoSimulationData,
        filters={"extras.simulation_fingerprint": {"==": fingerprint}},
        project=["*"],
    )
    builder.order_by({PinoSimulationData: {"ctime": "asc" if oldest_first else "desc"}})
    return builder.first(flat=True)


def _physics_key(node) -> tuple:
    """The conditioning that identical bytes do NOT prove identical.

    `P` and `V_scan` are broadcast channels: they are synthesized from metadata
    at read time and never stored, so two cases can hold byte-identical payloads
    while describing different physics. Bytes plus this tuple is a case; bytes
    alone are not.
    """
    meta = node.get_simulation_metadata() or {}
    return (
        meta.get("laser_power"),
        meta.get("V_scan"),
        meta.get("beam_radius"),
        meta.get("num_timesteps"),
        tuple(meta.get("spatial_shape") or ()),
    )


def _query_simulation_by_content(content_hash: str, candidate=None) -> PinoSimulationData | None:
    """The oldest stored simulation with the same bytes *and* the same physics."""
    if not content_hash:
        return None
    builder = orm.QueryBuilder()
    builder.append(
        PinoSimulationData,
        filters={"extras.content_hash": {"==": content_hash}, "extras.superseded": {"!==": True}},
        project=["*"],
    )
    builder.order_by({PinoSimulationData: {"ctime": "asc"}})
    wanted = _physics_key(candidate) if candidate is not None else None
    for node in builder.all(flat=True):
        if wanted is None or _physics_key(node) == wanted:
            return node
    return None


def _resolve_deduplicated_simulation(candidate: PinoSimulationData) -> tuple[PinoSimulationData, bool]:
    """Deduplicate a candidate simulation by fingerprint (BUG-20).

    The fast-path query keeps repeated imports cheap, but check-then-store
    races: two parallel parsers can both miss and both store. After storing,
    re-query oldest-first: the oldest node is the deterministic winner, and a
    losing candidate is marked superseded and points at the winner.

    Returns `(node, created_here)`. The flag decides whether this job may claim
    the node as its output: a Data node has exactly one CREATE link, so a
    simulation that was already in the database keeps the creator it already
    has, and a race loser is claimed by nobody.
    """
    fingerprint = candidate.get_simulation_fingerprint()
    existing = _query_simulation_by_fingerprint(fingerprint)
    if existing is not None:
        return existing, False
    candidate.store()
    # The fingerprint hashes the *path*, so the same case imported from two
    # directories misses the check above. Now that the node is stored its
    # content hash is free to read (the object store is sha256-addressed), so
    # ask the stronger question too, and stamp the answer for next time.
    content = payload_content_hash(candidate)
    if content:
        candidate.base.extras.set("content_hash", content)
        twin = _query_simulation_by_content(content, candidate)
        if twin is not None and twin.uuid != candidate.uuid:
            candidate.base.extras.set("superseded", True)
            candidate.base.extras.set("duplicate_of_uuid", twin.uuid)
            return twin, False
    winner = _query_simulation_by_fingerprint(fingerprint, oldest_first=True)
    if winner is None or winner.uuid == candidate.uuid:
        return candidate, True
    candidate.base.extras.set("superseded", True)
    candidate.base.extras.set("duplicate_of_uuid", winner.uuid)
    return winner, False


class VtkToDatasetParser(Parser):
    """Parse retrieved simulation payloads into a native dataset collection node."""

    dataset_label = "dataset"
    dataset_subpath = "dataset_output"
    summary_label = "dataset_summary"
    summary_filename = "dataset_summary.json"
    simulations_label = "simulations"

    def read_json(self, filename: str) -> dict:
        """Read a JSON file from the retrieved folder."""
        try:
            with self.retrieved.base.repository.open(filename, "r") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise exceptions.ParsingError(f"Failed to read parser JSON file `{filename}`.") from exc

    def read_text(self, filename: str) -> str:
        """Read a plain text file from the retrieved folder if present."""
        try:
            with self.retrieved.base.repository.open(filename, "r") as handle:
                return handle.read().strip()
        except OSError:
            return ""

    def parse(self, **kwargs):
        try:
            summary_data = self.read_json(self.summary_filename)
        except exceptions.ParsingError:
            scheduler_stderr = self.read_text("_scheduler-stderr.txt")
            aiida_out = self.read_text("aiida.out")
            details = scheduler_stderr or aiida_out
            if details:
                self.logger.error(details)
            return self.exit_codes.ERROR_MISSING_SUMMARY

        summary = orm.Dict(dict=summary_data)
        summary.label = self.summary_label
        self.out(self.summary_label, summary)
        if summary_data.get("status") == "error":
            return self.exit_codes.ERROR_CONVERSION_FAILED

        retrieved_temporary_folder = kwargs.get("retrieved_temporary_folder")
        if not retrieved_temporary_folder:
            return self.exit_codes.ERROR_MISSING_DATASET_PAYLOAD

        try:
            payload_root = Path(retrieved_temporary_folder) / self.dataset_subpath
            if not payload_root.exists():
                payload_root = Path(retrieved_temporary_folder)
            metadata_path = payload_root / "metadata.json"
            manifest_path = payload_root / "manifest.json"
            if not metadata_path.exists() or not manifest_path.exists():
                return self.exit_codes.ERROR_MISSING_DATASET_PAYLOAD
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            self.logger.error(f"Retrieved output folder `{self.dataset_subpath}` is missing: {exc}")
            return self.exit_codes.ERROR_MISSING_DATASET_PAYLOAD
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.logger.error(f"Retrieved dataset payload is invalid: {exc}")
            return self.exit_codes.ERROR_INVALID_DATASET_PAYLOAD

        source_uri = metadata.get("source_uri", "")
        source = None
        if source_uri:
            source = {
                "uri": source_uri,
                "description": metadata.get(
                    "source_description",
                    "VTK dataset imported through aiida-pino.",
                ),
            }

        source_folder = getattr(self.node.inputs, "vtk_folder", None)
        if source_folder is not None:
            metadata["source_folder_pk"] = source_folder.pk
            metadata["source_folder_uuid"] = source_folder.uuid

        try:
            simulations: list[PinoSimulationData] = []
            created_here: list[PinoSimulationData] = []

            def claim(simulation) -> None:
                """Emit a newly created simulation *immediately*.

                The member loop stores each candidate as it goes and can still
                return an exit code afterwards; AiiDA wraps no transaction around
                parsing, so those stores commit. Emitting only after the loop
                meant a failure at member k left k-1 simulations stored,
                unclaimed and unreachable — and a retry finds them by fingerprint
                and returns them as pre-existing, so they are never claimed at
                all. Claiming as we go bounds the damage to the member that
                actually failed.
                """
                created_here.append(simulation)
                self.out(f"{self.simulations_label}.sim_{len(created_here) - 1:04d}", simulation)
            for member in manifest.get("members", []):
                simulation_metadata = dict(member.get("simulation_metadata", {}))
                if member.get("storage_mode") == "npz_file":
                    # Compact per-case NPZ stored verbatim. Either the original
                    # file straight from its canonical realpath (nothing moved
                    # through the workdir), or a frame-sliced compact copy the
                    # adapter wrote into the retrieved payload dir.
                    if member.get("source_npz_realpath"):
                        source_npz = Path(str(member["source_npz_realpath"]))
                    else:
                        source_npz = payload_root / str(member.get("payload_file", ""))
                    if not source_npz.is_file():
                        self.logger.error(
                            f"npz_file member `{member.get('group', '?')}` source path is missing: {source_npz}"
                        )
                        return self.exit_codes.ERROR_MISSING_DATASET_PAYLOAD
                    candidate = PinoSimulationData.from_npz_file(
                        source_npz, metadata=simulation_metadata, source=source
                    )
                else:
                    npz_path = payload_root / str(member.get("payload_file", ""))
                    if not npz_path.exists():
                        return self.exit_codes.ERROR_MISSING_DATASET_PAYLOAD
                    with np.load(npz_path, allow_pickle=False) as archive:
                        arrays = {name: archive[name] for name in archive.files}
                    candidate = PinoSimulationData.from_arrays(arrays, metadata=simulation_metadata, source=source)
                simulation, created = _resolve_deduplicated_simulation(candidate)
                simulations.append(simulation)
                if created:
                    claim(simulation)
            dataset = PinoDatasetCollectionData.from_simulations(simulations, metadata=metadata).store()
            supersede_equivalent_dataset_collections(dataset)
        except (OSError, ValueError) as exc:
            self.logger.error(f"Retrieved dataset payload is invalid: {exc}")
            return self.exit_codes.ERROR_INVALID_DATASET_PAYLOAD
        dataset.label = metadata.get("label", dataset.label)
        self.out(self.dataset_label, dataset)
        # Claim the simulations this job actually created, so the graph records
        # what produced them. Before this, no simulation in the database had an
        # incoming link from any process (review criticism A) and
        # `verdi archive create -N <dataset>` exported one node and zero files.
        # Only newly created nodes are emitted: a Data node has exactly one
        # CREATE link, so a re-imported simulation keeps its original creator.
        # (simulations were claimed as they were created, above)
