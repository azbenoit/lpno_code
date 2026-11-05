"""A finished batch must file its simulations into one readable base dataset.

Ingested-but-unfiled simulations are invisible in the dataset views and easy to
mistake for training data: on 2026-07-28 four completed batches (74 sims,
including a 60-case evaluation-only set) sat with zero dataset membership
because filing was a separate manual step. `Flow3DBatchWorkChain.finalize`
now does it.
"""

from __future__ import annotations

import pytest

from aiida_pino.workflows import flow3d_batch


class _Attrs(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeSim:
    def __init__(self, pk):
        self.pk = pk


def test_dataset_name_prefers_batch_keyword():
    name = flow3d_batch.base_dataset_name({"batch_keyword": "material_ck300",
                                           "description": "whatever"}, "mck")
    assert name == "material_ck300"


def test_dataset_name_falls_back_to_case_base():
    name = flow3d_batch.base_dataset_name({"description": "no keyword here"}, "t64v")
    assert name == "t64v"


def test_dataset_name_last_resort_is_stable():
    assert flow3d_batch.base_dataset_name({}, None)


def test_collect_simulations_skips_children_without_a_dataset():
    summaries = [
        {"pk": 1, "case_name": "a", "dataset_pk": 11},
        {"pk": 2, "case_name": "b"},                       # failed / no ingest
        {"pk": 3, "case_name": "c", "dataset_pk": 13},
    ]
    assert flow3d_batch.ingested_dataset_pks(summaries) == [11, 13]


def test_standalone_generate_files_but_batch_child_does_not():
    """Only a standalone generate run files itself.

    Flow3DBatchWorkChain submits Flow3DGenerateWorkChain children, so if every
    generate filed itself the batch would end up as N one-member datasets and its
    own filing step would find nothing left to attach. A child (caller is not
    None) therefore defers to the batch; a standalone run (requeue, one-off
    validation case, the web console's single-run endpoint) files itself.
    """
    from aiida_pino.workflows import flow3d_generate as fg

    assert fg.should_file_own_dataset(caller=None) is True
    assert fg.should_file_own_dataset(caller=object()) is False


def test_member_simulations_prefers_the_collection_method():
    """Membership lives in a collection attribute, not in graph links.

    The npz_file ingest stores its simulations as standalone nodes referenced
    only by the collection's attribute; it creates no graph links to them. On
    2026-07-29 the link-based discovery (`get_incoming(node_class=
    PinoSimulationData)`) matched zero nodes and every batch of the calibration
    campaign reported "No unattached simulations to file" -- 23 sims orphaned.
    """

    class _FakeCollection:
        def iter_member_simulations(self):
            return ["sim-a", "sim-b"]

    assert flow3d_batch.member_simulations(_FakeCollection()) == ["sim-a", "sim-b"]


def test_member_simulations_resolve_by_uuid_never_by_pk(monkeypatch):
    """Primary keys are not stable across an archive round-trip.

    The 2026-07-21 migration renumbered every node, leaving 162 collections
    whose stored `member_simulation_pks` load *different* simulations with no
    error raised (review criticism E). Resolution must read the uuid list, so a
    node whose pk list is stale still yields the right members.
    """

    class _Base:
        def __init__(self, attrs):
            self.attributes = _Attrs(attrs)

    class _FakeNode:
        def __init__(self, attrs):
            self.base = _Base(attrs)

    loaded = []
    monkeypatch.setattr(flow3d_batch.orm, "load_node", lambda ident: loaded.append(ident) or f"node:{ident}")
    node = _FakeNode({
        "member_simulation_uuids": ["uuid-1", "uuid-2"],
        "member_simulation_pks": [999, 1000],  # stale, and must be ignored
    })
    assert flow3d_batch.member_simulations(node) == ["node:uuid-1", "node:uuid-2"]
    assert loaded == ["uuid-1", "uuid-2"]
    assert flow3d_batch.member_simulations(_FakeNode({})) == []


def test_dataset_display_name_comes_from_the_label():
    class _FakeDs:
        label = "bundle_gate_gan_domain | var_1_time_noref | flow3d"

    assert flow3d_batch.dataset_display_name(_FakeDs()) == "bundle_gate_gan_domain"


def test_batch_filing_extends_an_existing_dataset_of_the_same_name():
    """A second batch under one keyword must EXTEND, not mint a twin.

    `create_base_dataset` never merges, so a campaign that submits several
    batches with the same `batch_keyword` ends up with N datasets sharing a
    display name and selection-by-name becomes ambiguous (observed
    2026-07-29: two `bundle_gate` datasets). The standalone generate path
    already extends by name; the batch path must too.
    """
    assert flow3d_batch.should_extend_existing(existing=object()) is True
    assert flow3d_batch.should_extend_existing(existing=None) is False
