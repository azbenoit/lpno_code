"""An import claims the simulations it creates, as real graph edges.

Review criticism A (blocker): not one of the 3,093 stored simulations had an
incoming link from any process —

    QueryBuilder().append(ProcessNode, tag='p') \
        .append(PinoSimulationData, with_incoming='p').count()   # -> 0

— so `verdi archive create -N <dataset>` exported one node and zero data files,
and `verdi node delete` on a simulation never mentioned the collections holding
it. The ingest parser builds the simulations and must therefore emit them.

The parser is driven end to end here rather than unit-tested, because the thing
under test is whether AiiDA accepts an already-stored node as a CalcJob output
and writes the CREATE link — a claim no amount of reading the code settles.
"""

from __future__ import annotations

import json
import pathlib
import sys
from uuid import uuid4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from aiida import orm
from aiida.common.links import LinkType

from aiida_pino.data import PinoSimulationData
from aiida_pino.parsers.vtk_to_dataset import VtkToDatasetParser

ENTRY_POINT = "pino.raw_flow3d_to_dataset"


def _member_metadata(group: str, token: str) -> dict:
    return {
        "source_type": "flow3d",
        "source_origin_kind": "import",
        "source_folder_name": f"ingest_{token}",
        "source_group": group,
        "source_uri": f"file:///tmp/ingest_{token}/{group}",
        "requested_fields": ["Temperature", "alpha"],
        "grid_params": {
            "x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "z_range": [0.0, 1.0],
            "resolution": [4, 4, 4],
        },
        "relative_paths": [f"{group}/results_0.vtk", f"{group}/results_1.vtk"],
        "time_values_s": [0.0, 1.0],
        "sample_metadata": [
            {
                "group": group, "name": f"results_{i}.vtk",
                "relative_path": f"{group}/results_{i}.vtk", "stem": f"results_{i}",
                "params": {"V_scan": 0.01, "laser_power": 5.0, "time_s": float(i)},
            }
            for i in range(2)
        ],
    }


def _payload(tmp_path, token: str, groups: list[str]) -> pathlib.Path:
    """A retrieved payload directory as the ingest adapter writes one."""
    root = tmp_path / "dataset_output"
    root.mkdir(parents=True, exist_ok=True)
    members = []
    for index, group in enumerate(groups):
        # Distinct cases need distinct bytes: imports dedupe on payload content
        # as well as on the source path, so two cases sharing an array would
        # correctly collapse into one simulation and this fixture would be
        # testing nothing.
        fill = float(int(token[:6], 16) % 997) + index
        arrays = {
            "Temperature": np.full((2, 1, 4, 4, 4), fill, dtype=np.float32),
            "alpha": np.full((2, 1, 4, 4, 4), fill + 0.5, dtype=np.float32),
        }
        name = f"{group}.npz"
        np.savez(root / name, **arrays)
        members.append({
            "group": group,
            "payload_file": name,
            "simulation_metadata": _member_metadata(group, token),
        })
    (root / "manifest.json").write_text(json.dumps({"members": members}), encoding="utf-8")
    (root / "metadata.json").write_text(
        json.dumps({
            "source_type": "flow3d",
            "source_folder_name": f"ingest_{token}",
            "source_uri": f"file:///tmp/ingest_{token}",
            "dataset_role": "base",
            "label": f"ingest_{token}",
        }),
        encoding="utf-8",
    )
    return root


def _calcjob(computer) -> orm.CalcJobNode:
    node = orm.CalcJobNode(computer=computer, process_type=f"aiida.calculations:{ENTRY_POINT}")
    node.set_option("resources", {"num_machines": 1})
    node.set_option("output_filename", "aiida.out")
    node.store()
    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_bytes(
        json.dumps({"status": "ok"}).encode("utf-8"), "dataset_summary.json"
    )
    retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
    retrieved.store()
    return node


def _parse(node, payload_root):
    """Parse, then attach the outputs exactly as the CalcJob engine does.

    `aiida.engine.processes.process.Process.update_outputs` is one line:
    `output.base.links.add_incoming(self.node, LinkType.CREATE, link_label)`,
    with nested namespaces flattened on `__`. Doing that here puts the real
    link validation under test — which is the whole question, since these nodes
    are already stored when the parser emits them.

    Deliberately not `Parser.parse_from_node`: that wraps the re-parse in a
    *calcfunction*, and `CalcFunctionNode.validate_outgoing` rejects stored data
    outright. `CalcJobNode` inherits the generic `ProcessNode.validate_outgoing`,
    which only refuses to link from a sealed node — the difference this parser
    has always relied on to emit its stored `dataset`.
    """
    parser = VtkToDatasetParser(node)
    exit_code = parser.parse(retrieved_temporary_folder=str(payload_root.parent))
    assert exit_code is None, f"parser returned {exit_code}"
    results = {}
    for label, output in _flat(parser.outputs):
        output.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label=label)
        results[label] = output
    return results


def _flat(outputs, prefix=""):
    """(link_label, node) pairs with the label AiiDA will actually store.

    A Parser keeps `self.out("simulations.sim_0000", node)` as a literal key;
    the CalcJob then replays it through `Process.out`, which splits on `.` into
    a namespace and stores the link label joined by `__`.
    """
    for key, value in outputs.items():
        label = f"{prefix}{key}".replace(".", "__")
        if isinstance(value, orm.Node):
            yield label, value
        else:
            yield from _flat(value, prefix=f"{label}__")


def test_import_creates_a_provenance_edge_to_each_simulation(aiida_localhost, tmp_path):
    token = uuid4().hex[:8]
    node = _calcjob(aiida_localhost)
    outputs = _parse(node, _payload(tmp_path, token, ["case_a", "case_b"]))

    emitted = {k: v for k, v in outputs.items() if k.startswith("simulations__")}
    assert len(emitted) == 2, f"expected both new simulations to be claimed, got {sorted(emitted)}"
    for simulation in emitted.values():
        assert isinstance(simulation, PinoSimulationData)
        # the edge the reviewer could not find anywhere in the database
        assert simulation.creator is not None
        creators = orm.QueryBuilder().append(
            orm.ProcessNode, tag="p"
        ).append(PinoSimulationData, with_incoming="p", filters={"uuid": simulation.uuid}).count()
        assert creators == 1


def test_reimported_simulations_keep_their_original_creator(aiida_localhost, tmp_path):
    """A Data node has exactly one CREATE link. Importing the same case twice
    must not have the second job claim it — that would raise, and the honest
    record is that the first import created it."""
    token = uuid4().hex[:8]
    first = _calcjob(aiida_localhost)
    payload = _payload(tmp_path, token, ["case_a"])
    first_out = _parse(first, payload)
    claimed = [v for k, v in first_out.items() if k.startswith("simulations__")]
    assert len(claimed) == 1
    original = claimed[0]

    second = _calcjob(aiida_localhost)
    second_out = _parse(second, _payload(tmp_path / "again", token, ["case_a"]))
    assert not [k for k in second_out if k.startswith("simulations__")]
    assert original.creator.pk == first.pk  # unchanged by the second import
    # ...and the dataset is still produced, pointing at the same simulation
    assert second_out["dataset"].get_simulation_count() == 1
    assert (
        second_out["dataset"].get_member_simulation_uuids()
        == first_out["dataset"].get_member_simulation_uuids()
    )


def test_base_dataset_membership_is_a_graph_path(aiida_profile_clean, pino_simulation_factory):
    """The reviewer's own probe: are members link-reachable from the dataset?

    18,707 of 19,822 (simulation, collection) pairs had no link path — the
    attribute said what was in the dataset and the graph did not, which is why
    `verdi archive create -N <collection pk>` exported one node and zero data files.
    """
    from aiida_pino.data.dataset_hierarchy import create_base_dataset

    sims = [pino_simulation_factory("linked", group_name=f"lk-{i}") for i in range(3)]
    dataset, created = create_base_dataset(sims, name="linked")
    assert created

    # dataset <- compose_dataset <- each simulation
    creator = dataset.creator
    assert creator is not None, "the dataset has no creating process"
    input_uuids = {link.node.uuid for link in creator.base.links.get_incoming().all()}
    assert {sim.uuid for sim in sims} <= input_uuids

    # ...and the traversal the reviewer ran, from the simulation's side
    reachable = orm.QueryBuilder().append(
        PinoSimulationData, tag="sim", filters={"uuid": sims[0].uuid}
    ).append(orm.CalcFunctionNode, with_incoming="sim", tag="calc").append(
        type(dataset), with_incoming="calc", project=["uuid"]
    ).all(flat=True)
    assert str(dataset.uuid) in {str(u) for u in reachable}


def test_extending_a_base_dataset_records_the_dataset_it_grew_from(
    aiida_profile_clean, pino_simulation_factory
):
    from aiida_pino.data.dataset_hierarchy import create_base_dataset, extend_base_dataset

    first = [pino_simulation_factory("grow", group_name=f"g-{i}") for i in range(2)]
    base, _ = create_base_dataset(first, name="grow")
    extra = [pino_simulation_factory("grow", group_name="g-2")]
    grown = extend_base_dataset(base, extra)

    assert grown.uuid != base.uuid
    assert grown.get_simulation_count() == 3
    inputs = {link.link_label: link.node for link in grown.creator.base.links.get_incoming().all()}
    assert inputs["previous"].uuid == base.uuid  # the supersession chain, in the graph
    assert base.base.extras.get("superseded") is True


def test_a_split_subset_puts_its_members_in_the_graph(aiida_profile_clean, pino_simulation_factory):
    """The production path, and the one that was missed.

    Split subsets are what every reported metric is evaluated on. They were
    composed with only the *base dataset* as an input — enough for the
    calcfunction to find the members, but it puts no simulation in the graph, so
    `verdi archive create -N <split>` exported 4 nodes and none of the data for
    the six splits created after the membership fix landed.
    """
    from aiida_pino.console.config import Settings
    from aiida_pino.console.service import AiidaConsoleService
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    sims = [pino_simulation_factory("splitgraph", group_name=f"sg-{i}") for i in range(4)]
    for sim, split in zip(sims, ["train", "train", "val", "test"]):
        sim.base.extras.set("split", split)
    meta = build_collection_metadata_from_simulations(sims, collection_name="splitgraph")
    base, _ = get_or_create_dataset_collection(sims, metadata=meta)

    subset, _note = AiidaConsoleService(Settings())._test_subset(base)
    assert subset is not None
    inputs = {link.node.uuid for link in subset.creator.base.links.get_incoming().all()}
    test_member = next(s for s in sims if s.base.extras.get("split") == "test")
    assert test_member.uuid in inputs, "the split's own members are not inputs to the process that made it"
    assert base.uuid in inputs, "the base it was split from is still recorded"


def test_a_failure_partway_through_still_claims_what_was_created(aiida_localhost, tmp_path):
    """An import that dies on member k must not strand members 0..k-1.

    The parser stores each candidate as it goes and AiiDA wraps no transaction
    around parsing, so those stores commit even when the parser then returns an
    exit code. Emitting the outputs only after the loop meant the survivors were
    stored, unclaimed and unreachable — and a retry finds them by fingerprint and
    returns them as pre-existing, so they are never claimed at all.
    """
    token = uuid4().hex[:8]
    node = _calcjob(aiida_localhost)
    root = _payload(tmp_path, token, ["good_a", "good_b", "bad_c"])

    # break the third member only: its payload file disappears
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["members"][2]["payload_file"] = "does_not_exist.npz"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parser = VtkToDatasetParser(node)
    exit_code = parser.parse(retrieved_temporary_folder=str(root.parent))
    assert exit_code is not None, "the broken member should have produced an exit code"

    claimed = [label for label, _ in _flat(parser.outputs) if label.startswith("simulations__")]
    assert len(claimed) == 2, f"the two good members were not claimed: {claimed}"
