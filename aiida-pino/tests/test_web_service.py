"""Service-layer tests against the AiiDA pytest profile with seeded nodes.

These exercise the broker-free browse/monitor paths on real ORM data and the
submit-gating behaviour (the test profile has no running daemon, so every
submission must raise DaemonUnavailableError rather than block or crash).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from pathlib import Path

from web.app.config import Settings
from web.app.aiida_service import AiidaConsoleService
from web.app.errors import DaemonUnavailableError, InvalidRequestError, NodeNotFoundError
from web.app.schemas import DeriveDatasetRequest, RawFlow3DImportRequest, TrainEvalRequest


@pytest.fixture
def svc():
    return AiidaConsoleService(Settings())


def test_daemon_unavailable_in_test_profile(svc, aiida_localhost):
    available, reason = svc.daemon_available()
    assert available is False
    assert isinstance(reason, str) and reason


def test_env_status_shape(svc, local_code_factory):
    local_code_factory("/bin/bash", label="pino-python-web-test")
    env = svc.env_status()
    assert set(env) >= {"lpno_version", "profile", "daemon", "codes", "config_names"}
    assert env["daemon"]["available"] is False
    assert any(c["label"].startswith("pino-python-web-test") for c in env["codes"]["all"])
    assert isinstance(env["config_names"], list)


def test_list_and_detail_dataset(svc, pino_dataset_factory):
    ds = pino_dataset_factory("web-console-ds")
    rows = svc.list_datasets(limit=50)
    assert any(r["pk"] == ds.pk for r in rows)
    detail = svc.dataset_detail(ds.pk)
    assert detail["pk"] == ds.pk
    assert "family" in detail and "describe" in detail
    # The list row shows both of these; the pane asked for them by name and the
    # payload did not carry them, so kv() silently dropped the two rows.
    row = next(r for r in rows if r["pk"] == ds.pk)
    assert detail["simulation_count"] == row["simulation_count"]
    assert "num_timesteps" in detail


def test_list_and_detail_model(svc, pino_model_factory):
    model = pino_model_factory()
    rows = svc.list_models(limit=50)
    assert any(r["pk"] == model.pk for r in rows)
    detail = svc.model_detail(model.pk)
    assert detail["pk"] == model.pk
    assert "training_profile" in detail


def test_list_processes_returns_list(svc, aiida_localhost):
    assert isinstance(svc.list_processes(limit=10), list)


def test_load_node_missing_raises_not_found(svc, aiida_localhost):
    with pytest.raises(NodeNotFoundError):
        svc.dataset_detail(99_000_001)


def test_detail_routes_reject_wrong_type_pk(svc, pino_dataset_factory):
    # An existing pk of the wrong node type must 404 (NodeNotFoundError), not
    # 500 from a downstream accessor. A dataset is neither a model nor a process.
    ds = pino_dataset_factory("web-wrongtype-ds")
    with pytest.raises(NodeNotFoundError):
        svc.model_detail(ds.pk)
    with pytest.raises(NodeNotFoundError):
        svc.process_detail(ds.pk)


def test_submit_is_gated_on_daemon(svc, pino_dataset_factory, local_code_factory):
    # A submit builder that needs no training-profile validation must still fail
    # at the daemon gate (503), not attempt a broker connection.
    local_code_factory("/bin/bash", label="pino-python-gate")
    req = RawFlow3DImportRequest(source_path="/tmp/does-not-matter", python_code="pino-python-gate")
    with pytest.raises(DaemonUnavailableError):
        svc.submit_raw_flow3d(req)


def _collection_of(sims, name):
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    meta = build_collection_metadata_from_simulations(sims, collection_name=name)
    node, _ = get_or_create_dataset_collection(sims, metadata=meta)
    return node


def test_assign_splits_proportions_and_determinism(svc, pino_simulation_factory):
    sims = [pino_simulation_factory("split-ds", group_name=f"split-{i:02d}") for i in range(10)]
    ds = _collection_of(sims, "split-ds")
    out = svc.assign_splits(ds.pk, train=0.6, val=0.2, test=0.2)
    assert out["assigned"] == {"train": 6, "val": 2, "test": 2}
    assert out["skipped"] == 0
    splits = [s.base.extras.get("split", None) for s in sims]
    assert sorted(splits).count("train") == 6
    # deterministic: same seed reproduces the same assignment
    first = {s.pk: s.base.extras.get("split", None) for s in sims}
    svc.assign_splits(ds.pk, train=0.6, val=0.2, test=0.2, only_unassigned=False, seed=0)
    assert {s.pk: s.base.extras.get("split", None) for s in sims} == first
    # the recipe that produced those labels is recorded, not just the labels
    recipe = ds.base.extras.get("split_recipe")
    assert recipe["method"] == "random"
    assert recipe["seed"] == 0
    assert recipe["fractions"] == {"train": 0.6, "val": 0.2, "test": 0.2}
    assert recipe["only_unassigned"] is False  # the reshuffle overwrote the first record
    assert recipe["assigned"] == {"train": 6, "val": 2, "test": 2}
    assert len(recipe["member_uuids_sha256"]) == 64
    assert out["recipe"]["only_unassigned"] is True  # returned by the first call


def test_assign_splits_only_unassigned_and_validation(svc, pino_simulation_factory):
    sims = [pino_simulation_factory("split-ds2", group_name=f"split2-{i:02d}") for i in range(5)]
    sims[0].base.extras.set("split", "test")
    ds = _collection_of(sims, "split-ds2")
    out = svc.assign_splits(ds.pk, train=0.5, val=0.25, test=0.25)
    assert out["skipped"] == 1
    assert sims[0].base.extras.get("split", None) == "test"  # untouched
    assert sum(out["assigned"].values()) == 4
    with pytest.raises(InvalidRequestError):
        svc.assign_splits(ds.pk, train=0.9, val=0.2, test=0.2)  # sums past 1
    with pytest.raises(InvalidRequestError):
        svc.assign_splits(ds.pk, train=-0.5, val=1.0, test=0.5)


def test_dataset_rows_step_stats(svc, pino_simulation_factory):
    import numpy as np

    a = pino_simulation_factory("steps-ds", group_name="st-a")  # 2 timesteps
    b = pino_simulation_factory(
        "steps-ds",
        group_name="st-b",
        arrays={
            "Temperature": np.zeros((3, 1, 4, 4, 4), np.float32),
            "alpha": np.ones((3, 1, 4, 4, 4), np.float32),
        },
    )
    ds = _collection_of([a, b], "steps-ds")
    row = next(r for r in svc.list_datasets(role="all", limit=50) if r["pk"] == ds.pk)
    assert (row["steps_min"], row["steps_max"]) == (2, 3)
    assert "step_size_s" in row


def test_composed_collection_role(svc, pino_simulation_factory):
    sims = [pino_simulation_factory("comp-ds", group_name=f"comp-{i}") for i in range(3)]
    _collection_of(sims, "comp-parent")  # composed subset of this must stay composed
    out = svc.create_collection([s.pk for s in sims[:2]], "comp-child", description="composition test")
    rows = {r["pk"]: r for r in svc.list_datasets(role="all", limit=50)}
    assert rows[out["pk"]]["dataset_role"] == "composed"
    assert svc.dataset_detail(out["pk"])["dataset_role"] == "composed"


def test_preview_collection_returns_structured_pks(svc, pino_simulation_factory):
    sims = [pino_simulation_factory("prev-ds", group_name=f"prev-{i}") for i in range(2)]
    out = svc.preview_collection([s.pk for s in sims])
    assert out["member_pks"] == [s.pk for s in sims]
    assert out["member_count"] == 2
    assert isinstance(out["signature"], str) and out["signature"]
    assert "result" not in out  # was str((members, signature)): a Python repr inside the JSON


def test_create_collection_from_datasets_union(svc, pino_simulation_factory):
    a = [pino_simulation_factory("union-a", group_name=f"ua-{i}") for i in range(2)]
    b = [pino_simulation_factory("union-b", group_name=f"ub-{i}") for i in range(2)]
    ds_a = _collection_of(a, "union-a")
    ds_b = _collection_of([a[1], *b], "union-b")  # overlaps a[1]
    out = svc.create_collection([], "union-ab", description="composition test", dataset_pks=[ds_a.pk, ds_b.pk])
    detail = svc.dataset_detail(out["pk"])
    assert detail["dataset_role"] == "composed"
    assert detail["num_samples"] == 4  # deduped union
    # composition provenance: the source datasets hang off the composed row...
    rows = {r["pk"]: r for r in svc.list_datasets(role="all", limit=50)}
    assert rows[out["pk"]]["component_pks"] == [ds_a.pk, ds_b.pk]
    # ...and both sources are still listed top-level in their own right
    assert ds_a.pk in rows and ds_b.pk in rows
    # detail carries clickable refs
    assert [c["pk"] for c in detail["composed_from"]] == [ds_a.pk, ds_b.pk]
    # real AiiDA provenance: the composed collection was created by a
    # calcfunction whose inputs include both source datasets
    from aiida.orm import load_node

    creator = load_node(out["pk"]).creator
    assert creator is not None
    input_pks = {link.node.pk for link in creator.base.links.get_incoming().all()}
    assert {ds_a.pk, ds_b.pk} <= input_pks
    # a source may sit under several compositions
    out2 = svc.create_collection([], "union-ab2", description="composition test", dataset_pks=[ds_a.pk])
    rows = {r["pk"]: r for r in svc.list_datasets(role="all", limit=50)}
    assert rows[out2["pk"]]["component_pks"] == [ds_a.pk]
    with pytest.raises(InvalidRequestError):
        svc.create_collection([], "union-empty", description="composition test")  # no sims, no datasets


def _runner_valid_config(case_name: str) -> dict:
    """A config the real runner accepts, obtained by asking it rather than guessing.

    These tests used to pass skeletal configs (`{"geometry": {}, "mesh": {}, ...}`)
    and still reach the daemon gate, because the console only ran the runner's own
    `load_and_validate` "when the runner checkout is known" — and with no root
    configured it never was. Once an unset root started resolving to the sibling
    checkout, the deep check ran and refused them: `case_name must use letters,
    digits, _ and - only`, because there was no case_name at all.

    The values below are the minimum the runner accepts, found by feeding it a
    config and repairing whatever it named until it stopped complaining: a
    case_name, a beam radius, a focal distance and lens radius, a scan speed,
    an end point, an initial temperature, a max iteration count, a finish time
    longer than the laser travel (0.0016 s here), and a cell size no larger than
    half the beam radius. That list is itself the point of the fix — every one of
    them used to be discovered only when a child job died.
    """
    return {
        "schema_version": 1,
        "case_name": case_name,
        "geometry": {"length_m": 0.0024},
        "mesh": {"cell_size_m": 2e-5},
        "laser": {
            "power_w": 150.0,
            "beam_radius_m": 5e-5,
            "focal_distance_m": 0.2,
            "lens_radius_m": 0.01,
            "scan_speed_m_per_s": 0.5,
            "end_point_m": [0.0008, 0.0, 0.0],
        },
        "simulation": {
            "finish_time_s": 3e-3,
            "initial_temperature_k": 300.0,
            "max_iterations": 100000,
        },
    }


def test_flow3d_generate_validates_profiles_before_gate(svc, local_code_factory):
    from aiida_pino.console.schemas import Flow3DGenerateRequest

    local_code_factory("/bin/bash", label="pino-python-f3d")
    cfg = _runner_valid_config("console_case_gate")
    # unknown execution key => 422-class error before any daemon contact
    with pytest.raises(InvalidRequestError):
        svc.submit_flow3d_generate(Flow3DGenerateRequest(
            python_code="pino-python-f3d", run_config=cfg, execution_profile={"nope": 1}))
    # license material is refused
    with pytest.raises(InvalidRequestError):
        svc.submit_flow3d_generate(Flow3DGenerateRequest(
            python_code="pino-python-f3d", run_config=cfg, execution_profile={"license_server": "x"}))
    with pytest.raises(InvalidRequestError):
        svc.submit_flow3d_generate(Flow3DGenerateRequest(python_code="pino-python-f3d", run_config={}))
    # valid request stops at the daemon gate in the test profile
    with pytest.raises(DaemonUnavailableError):
        svc.submit_flow3d_generate(Flow3DGenerateRequest(
            python_code="pino-python-f3d", run_config=cfg,
            execution_profile={"prepare_only": True, "cores": 2}))


def test_flow3d_batch_validates_and_gates(svc, local_code_factory):
    from aiida_pino.console.schemas import Flow3DBatchRequest

    local_code_factory("/bin/bash", label="pino-python-f3db")
    with pytest.raises(InvalidRequestError):
        svc.submit_flow3d_batch(Flow3DBatchRequest(python_code="pino-python-f3db", run_configs=[]))
    with pytest.raises(DaemonUnavailableError):
        svc.submit_flow3d_batch(Flow3DBatchRequest(
            python_code="pino-python-f3db",
            run_configs=[
                _runner_valid_config("console_batch_a"),
                _runner_valid_config("console_batch_b"),
            ],
            execution_profile={"prepare_only": True}))


def test_flow3d_queue_status_reads_spool(svc, tmp_path, monkeypatch):
    import json as _json

    spool = tmp_path / "spool"
    (spool / "queue").mkdir(parents=True)
    (spool / "running").mkdir()
    (spool / "queue" / "001.task").write_text(_json.dumps({"name": "case_a", "cmd": "x", "added": "t"}))
    (spool / ".seat.lock").touch()
    monkeypatch.setenv("F3D_SPOOL_DIR", str(spool))
    out = svc.flow3d_queue_status()
    assert out["exists"] is True
    assert out["queued_count"] == 1 and out["queued"][0]["name"] == "case_a"
    assert out["running"] == []
    assert out["seat_held"] is False  # lock file present but nobody holds the flock
    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path / "nowhere"))
    assert svc.flow3d_queue_status()["exists"] is False


def test_monitor_status_counts_aiida_driven_jobs(svc, aiida_profile_clean, aiida_localhost, tmp_path, monkeypatch):
    """The queue panels read spool dirs only, so AiiDA-driven work is invisible
    there: an empty spool must not read as an idle machine.

    Needs a clean profile because it asserts exact counts over every process on
    it: run after `test_process_state_filter.py`, which stores a
    `Flow3DRunCalculation` of its own, the count was 2 and the file passed or
    failed depending on which other file ran in between.
    """
    from aiida.orm import CalcJobNode

    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path / "no-spool"))  # nothing queued anywhere

    def _running(label, ptype):
        node = CalcJobNode(computer=aiida_localhost)
        node.set_process_type(ptype)
        node.base.attributes.set("process_label", label)
        node.base.attributes.set("process_state", "running")
        node.store()
        return node

    _running("Flow3DRunCalculation", "aiida.calculations:pino.flow3d_run")
    _running("PinoTrainCalculation", "aiida.calculations:pino.train")
    _running("PinoTrainCalculation", "aiida.calculations:pino.train")
    _running("PinoEvalCalculation", "aiida.calculations:pino.eval")

    m = svc.monitor_status()
    assert m["aiida_calcjobs"]["Flow3DRunCalculation"] == 1
    assert m["aiida_calcjobs"]["PinoTrainCalculation"] == 2
    assert m["aiida_calcjobs"]["PinoEvalCalculation"] == 1
    # GPU work driven by AiiDA rather than queued in gpu_spool
    assert m["aiida_gpu_running"] == 3
    # these are running, not queued, so nothing AiiDA-driven is in the queue
    # (real-machine spool tasks may still appear; only the aiida entries are ours)
    assert m["aiida_gpu_queued"] == 0
    assert not any(e["source"] == "aiida" for e in m["gpu_queue"])
    assert not any(e["source"] == "aiida" for e in m["flow3d"]["queue"])
    # and the FLOW-3D panel knows the seat is busy even with an empty spool
    assert m["flow3d"]["aiida_running"] == 1
    assert m["flow3d"]["exists"] is False


def test_monitor_status_merges_gpu_seat_holders(svc, tmp_path, monkeypatch):
    """A card booked through `gpu_spool seat` has an empty spool queue, so the
    row must carry the lease ledger's verdict, not just the queue's."""
    import json as _json
    import os as _os

    from aiida_pino.console import monitor

    ledger = tmp_path / "leases" / "gpu0"
    ledger.mkdir(parents=True)
    (ledger / "a.lease").write_text(_json.dumps(
        {"name": "train_a", "vram_mb": 9000, "ram_mb": 0, "pid": _os.getpid(), "user": "someone"}))
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path / "no-spool"))
    monkeypatch.setattr(monitor, "gpu_spool_snapshot", lambda *a, **k: [
        {"gpu": "gpu0", "running": [], "queued_count": 0, "queued": []},
    ])

    row = svc.monitor_status()["gpu_spool"][0]
    assert row["seat_count"] == 1
    assert row["seat_vram_mb"] == 9000
    assert row["busy"] is True          # nothing queued, nothing spool-running
    assert row["seats"][0]["name"] == "train_a"


def test_monitor_status_aiida_counts_present_when_idle(svc, aiida_profile_clean, tmp_path, monkeypatch):
    """No active runs => explicit zeros, so the panel can say so rather than
    omitting the field (a missing key renders as blank, which reads as idle)."""
    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path / "no-spool"))
    m = svc.monitor_status()
    assert m["aiida_calcjobs"] == {}
    assert m["aiida_gpu_running"] == 0
    assert m["flow3d"]["aiida_running"] == 0


def test_submit_honours_split_extras(svc, pino_simulation_factory, local_code_factory):
    from aiida_pino.console.schemas import TrainEvalRequest

    local_code_factory("/bin/bash", label="pino-python-split")
    sims = [pino_simulation_factory("split-honour", group_name=f"sh-{i}") for i in range(5)]
    for sim, split in zip(sims, ("train", "train", "train", "val", "test")):
        sim.base.extras.set("split", split)
    ds = _collection_of(sims, "split-honour")
    req = TrainEvalRequest(train_dataset_pk=ds.pk, python_code="pino-python-split")
    # the daemon gate fires last; the split subsets are composed before it
    with pytest.raises(DaemonUnavailableError):
        svc.submit_train_eval(req)
    rows = {r["source_folder_name"]: r for r in svc.list_datasets(role="all", limit=50)}
    assert rows["split-honour_train_split"]["simulation_count"] == 3
    assert rows["split-honour_val_split"]["simulation_count"] == 1
    # Auto splits are `subset` views of one base dataset, not composed unions,
    # and record their base for provenance (2026-07-24).
    assert rows["split-honour_train_split"]["dataset_role"] == "subset"
    assert rows["split-honour_train_split"]["base_collection_pk"] == ds.pk


def test_ragged_collection_records_servable_layout(svc, pino_simulation_factory):
    # A ragged member set must not record the canonical time_as_dim layout,
    # which the training path can never serve for it.
    import numpy as np

    a = pino_simulation_factory("ragged", group_name="rg-a")  # 2 timesteps
    b = pino_simulation_factory(
        "ragged", group_name="rg-b",
        arrays={"Temperature": np.zeros((3, 1, 4, 4, 4), np.float32), "alpha": np.ones((3, 1, 4, 4, 4), np.float32)},
    )
    ds = _collection_of([a, b], "ragged-coll")
    profile = ds.get_dataset_metadata().get("derivation_profile") or {}
    assert profile.get("time_as_dim") is False
    assert profile.get("time_as_batch") is True
    # uniform sets keep the canonical layout -- which is now time_as_batch too,
    # so both ragged and uniform collections land on one servable layout.
    from aiida_pino.utils import canonical_base_layout_profile

    u = _collection_of([pino_simulation_factory("uni", group_name=f"u-{i}") for i in range(2)], "uni-coll")
    uprofile = u.get_dataset_metadata().get("derivation_profile") or {}
    assert uprofile == canonical_base_layout_profile()
    assert uprofile.get("time_as_batch") is True


def test_composed_collection_is_its_own_derivation_base(svc, pino_simulation_factory):
    from aiida_pino.utils.dataset_derivation import resolve_base_dataset

    sims = [pino_simulation_factory("comp-base", group_name=f"cb-{i}") for i in range(2)]
    out = svc.create_collection([s.pk for s in sims], "comp-base-coll", description="composition test")
    from aiida.orm import load_node

    node = load_node(out["pk"])
    assert resolve_base_dataset(node).pk == node.pk  # no more 'not canonical base' error


def test_partial_training_profile_merges_over_defaults(svc):
    # Regression for the CLI/API friction: a partial profile no longer fails
    # with "missing required keys".
    merged = svc._validated_training_profile({"max_steps": 123, "run_name": "partial"}, split_mode="standard")
    assert merged["max_steps"] == 123
    assert merged["run_name"] == "partial"
    assert merged["cuda_device"] is not None and "hydra_overrides" in merged


@pytest.mark.requires_lpno
def test_partial_training_profile_inherits_selected_config_defaults(svc):
    # Regression: the pre-fill for partial profiles must come from the profile's
    # own config_name, not DEFAULT_TRAINING_PROFILE — the latter stamped the
    # default config's values (time_as_batch=False, lr=6e-5, batch=20) as
    # explicitly-provided keys, silently overriding the selected config.
    merged = svc._validated_training_profile(
        {"config_name": "config_flow3D_RIGNO_randompath", "run_name": "x", "max_steps": 8000},
        split_mode="standard",
    )
    assert merged["time_as_batch"] is True  # randompath config value, not the default False
    assert merged["learning_rate"] == pytest.approx(1e-4)
    assert merged["batch_size_grid"] == 1
    assert merged["scheduler_decay_steps"] == 250
    # Explicit keys still win over the config-derived base.
    assert merged["max_steps"] == 8000


def test_training_presets_roundtrip(svc):
    out = svc.save_training_preset(
        name="rigno-quick", description="short RIGNO smoke", training_profile={"max_steps": 500}
    )
    rows = svc.list_training_presets()
    mine = next(r for r in rows if r["pk"] == out["pk"])
    assert mine["name"] == "rigno-quick"
    assert mine["description"] == "short RIGNO smoke"
    assert mine["training_profile"] == {"max_steps": 500}  # partial presets allowed
    with pytest.raises(InvalidRequestError):
        svc.save_training_preset(name="  ", training_profile={})
    with pytest.raises(InvalidRequestError):
        # invalid VALUES are rejected the same way a submit would reject them
        svc.save_training_preset(name="bad", training_profile={"max_steps": -5})


def test_all_packaged_configs_are_listed():
    pytest.importorskip("lpno")
    from aiida_pino.utils.profiles import available_config_names

    names = available_config_names()
    # The old listing hid everything outside the config_NS_energy_PINO prefix
    # (e.g. the RIGNO and flow3D baseline configs) even though submission
    # accepted them.
    assert any(not n.startswith("config_NS_energy_PINO") for n in names)
    assert all(n.startswith("config_") for n in names)


def test_train_submit_profile_validation_before_gate(svc, pino_dataset_factory, local_code_factory):
    # Since the partial-profile fix, an EMPTY profile is valid (it overlays
    # the packaged defaults) and proceeds to the daemon gate; an invalid
    # VALUE still fails as a 422-class error before any daemon contact.
    ds = pino_dataset_factory("web-train-ds")
    local_code_factory("/bin/bash", label="pino-python-train")
    with pytest.raises(DaemonUnavailableError):
        svc.submit_train_eval(TrainEvalRequest(
            train_dataset_pk=ds.pk, python_code="pino-python-train", training_profile={}))
    with pytest.raises(InvalidRequestError):
        svc.submit_train_eval(TrainEvalRequest(
            train_dataset_pk=ds.pk, python_code="pino-python-train",
            training_profile={"max_steps": -5}))


def test_derive_requires_collection_node(svc, pino_model_factory):
    # A model pk is not a dataset collection => clean 404-class NodeNotFoundError.
    model = pino_model_factory()
    req = DeriveDatasetRequest(dataset_pk=model.pk, derivation_profile={})
    with pytest.raises(NodeNotFoundError):
        svc.submit_derive(req)


def test_delete_dry_run_does_not_delete(svc, pino_dataset_factory):
    ds = pino_dataset_factory("web-del-ds")
    result = svc.delete_nodes([ds.pk], dry_run=True)
    assert result["dry_run"] is True
    assert ds.pk in result["affected"]
    # Node still loads after a dry run.
    assert svc.dataset_detail(ds.pk)["pk"] == ds.pk


# --- annotations -------------------------------------------------------------
def test_update_annotations_on_simulation(svc, pino_simulation_factory):
    sim = pino_simulation_factory("web-ann-sim")
    out = svc.update_annotations(sim.pk, description="d1", split="test", keywords_set=["keyhole"])
    assert out["pk"] == sim.pk
    assert out["kind"] == "simulation"
    assert out["description"] == "d1"
    assert out["split"] == "test"
    assert out["keywords"] == ["keyhole"]
    # description lands on the native column AND the sim extras mirror
    assert sim.description == "d1"
    assert sim.base.extras.get("description") == "d1"
    # add/remove semantics through the service
    out = svc.update_annotations(sim.pk, keywords_add=["smoke"], keywords_remove=["keyhole"])
    assert out["keywords"] == ["smoke"]


def test_update_annotations_invalid_split_rejected(svc, pino_simulation_factory):
    sim = pino_simulation_factory("web-ann-badsplit")
    with pytest.raises(InvalidRequestError):
        svc.update_annotations(sim.pk, split="bogus")


def test_update_annotations_split_on_model_rejected(svc, pino_model_factory):
    model = pino_model_factory("web-ann-model")
    with pytest.raises(InvalidRequestError):
        svc.update_annotations(model.pk, split="train")


def test_update_annotations_unknown_pk(svc, aiida_localhost):
    with pytest.raises(NodeNotFoundError):
        svc.update_annotations(99_000_002, split="train")


def test_list_simulations_filters_by_split_and_keywords(svc, pino_simulation_factory):
    tagged = pino_simulation_factory("web-flt-tagged")
    plain = pino_simulation_factory("web-flt-plain")
    svc.update_annotations(tagged.pk, split="train", keywords_set=["keyhole", "smoke"])
    svc.update_annotations(plain.pk, keywords_set=["keyhole"])

    pks = {r["pk"] for r in svc.list_simulations(split="train")}
    assert tagged.pk in pks and plain.pk not in pks
    # "unassigned" matches nodes without a split extra
    pks = {r["pk"] for r in svc.list_simulations(split="unassigned")}
    assert plain.pk in pks and tagged.pk not in pks
    # keywords use AND semantics
    pks = {r["pk"] for r in svc.list_simulations(keywords=["keyhole", "smoke"])}
    assert tagged.pk in pks and plain.pk not in pks
    rows = svc.list_simulations(keywords=["keyhole"])
    pks = {r["pk"] for r in rows}
    assert {tagged.pk, plain.pk} <= pks
    # rows carry split + keywords, defaulting to unassigned/[]
    row = next(r for r in rows if r["pk"] == tagged.pk)
    assert row["split"] == "train" and row["keywords"] == ["keyhole", "smoke"]
    row = next(r for r in rows if r["pk"] == plain.pk)
    assert row["split"] == "unassigned"


def test_list_datasets_keyword_filter_and_row_fields(svc, pino_dataset_factory):
    ds = pino_dataset_factory("web-kw-ds")
    svc.update_annotations(ds.pk, description="ds notes", keywords_set=["steady"])
    rows = svc.list_datasets(keywords=["steady"])
    row = next(r for r in rows if r["pk"] == ds.pk)
    assert row["keywords"] == ["steady"]
    assert row["description"] == "ds notes"
    assert all("steady" in r["keywords"] for r in rows)
    assert not any(r["pk"] == ds.pk for r in svc.list_datasets(keywords=["steady", "absent-kw"]))


def test_list_models_keyword_filter(svc, pino_model_factory):
    model = pino_model_factory("web-kw-model")
    svc.update_annotations(model.pk, keywords_set=["downsampled"])
    rows = svc.list_models(keywords=["downsampled"])
    row = next(r for r in rows if r["pk"] == model.pk)
    assert row["keywords"] == ["downsampled"]
    assert "description" in row
    assert not any(r["pk"] == model.pk for r in svc.list_models(keywords=["absent-kw"]))


def test_create_collection_stores_description_and_keywords(svc, pino_simulation_factory):
    sims = [
        pino_simulation_factory("web-cc", group_name=f"web-cc-{index:03d}") for index in range(2)
    ]
    out = svc.create_collection(
        [s.pk for s in sims],
        "web-cc-collection",
        description="cc notes",
        keywords=["random path"],
    )
    from aiida.orm import load_node

    node = load_node(out["pk"])
    assert node.description == "cc notes"
    assert node.base.extras.get("keywords") == ["random path"]
    # and it shows up through the dataset keyword filter
    assert any(r["pk"] == node.pk for r in svc.list_datasets(keywords=["random path"]))


def test_dataset_rows_carry_display_fields(svc, pino_dataset_factory):
    ds = pino_dataset_factory("web-display-ds")
    row = next(r for r in svc.list_datasets(limit=50) if r["pk"] == ds.pk)
    assert row["display_name"]  # prettified, never empty
    assert "_" not in row["display_name"].strip() or row["display_name"] != row["source_folder_name"]
    assert isinstance(row["split_counts"], dict)
    assert "parent_pk" in row
    assert row["dataset_role"]  # effective role, never empty
    detail = svc.dataset_detail(ds.pk)
    assert detail["display_name"]
    assert isinstance(detail["split_counts"], dict)


def test_dataset_cases_browser(svc, pino_dataset_factory):
    ds = pino_dataset_factory("web-cases-ds")
    body = svc.dataset_cases(ds.pk)
    assert body["pk"] == ds.pk
    assert body["count"] == len(body["cases"]) > 0
    case = body["cases"][0]
    assert set(case) >= {"pk", "case_name", "split", "keywords", "laser_power", "V_scan", "num_timesteps"}
    assert case["split"] in ("train", "val", "test", "unassigned")


def test_dataset_cases_rejects_wrong_type(svc, pino_model_factory):
    model = pino_model_factory()
    with pytest.raises(NodeNotFoundError):
        svc.dataset_cases(model.pk)


def test_subset_detection_marks_strict_subsets(svc, pino_simulation_factory):
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    sims = [pino_simulation_factory(f"subset-sim-{i}") for i in range(3)]
    meta_all = build_collection_metadata_from_simulations(sims, collection_name="subset-parent")
    parent, _ = get_or_create_dataset_collection(sims, metadata=meta_all)
    meta_sub = build_collection_metadata_from_simulations(sims[:2], collection_name="subset-child")
    child, _ = get_or_create_dataset_collection(sims[:2], metadata=meta_sub)
    rows = {r["pk"]: r for r in svc.list_datasets(limit=50)}
    assert rows[child.pk]["parent_pk"] == parent.pk
    assert rows[child.pk]["dataset_role"] == "subset"
    assert rows[parent.pk]["parent_pk"] is None


def test_folder_lineage_marks_derived_sets(svc, pino_simulation_factory):
    # Disjoint member sets, so only the folder-name lineage map can link them.
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    sims = [pino_simulation_factory(f"lineage-sim-{i}") for i in range(4)]
    base, _ = get_or_create_dataset_collection(
        sims[:2], metadata=build_collection_metadata_from_simulations(sims[:2], collection_name="ti64_random150")
    )
    derived, _ = get_or_create_dataset_collection(
        sims[2:],
        metadata=build_collection_metadata_from_simulations(sims[2:], collection_name="ti64_random150_ds_split"),
    )
    rows = {r["pk"]: r for r in svc.list_datasets(limit=50)}
    assert rows[base.pk]["dataset_role"] == "base"
    assert rows[derived.pk]["dataset_role"] == "derived"
    assert rows[derived.pk]["parent_pk"] == base.pk
    # role filter acts on the effective role
    assert derived.pk in {r["pk"] for r in svc.list_datasets(role="derived", limit=50)}
    assert derived.pk not in {r["pk"] for r in svc.list_datasets(role="base", limit=50)}
    # detail carries the lineage both ways
    assert svc.dataset_detail(derived.pk)["derived_from"]["pk"] == base.pk
    assert {s["pk"] for s in svc.dataset_detail(base.pk)["derived_sets"]} == {derived.pk}


def test_model_rows_carry_registry_fields(svc, pino_model_factory):
    model = pino_model_factory()
    row = next(r for r in svc.list_models(limit=50) if r["pk"] == model.pk)
    assert row["architecture"]  # derived family label, never empty
    assert "training_source_display" in row
    assert isinstance(row["evals"], list)
    detail = svc.model_detail(model.pk)
    assert detail["architecture"] == row["architecture"]
    assert isinstance(detail["eval_rows"], list)


def test_classify_run_running_vs_queued() -> None:
    from aiida_pino.console.service import classify_run

    # GPU: on a card, or logging steps, is running; waiting-for-seat is queued.
    assert classify_run("TrainModelWorkChain", {"step": 500}, on_gpu=False, has_seat=False) == ("gpu", "running")
    assert classify_run("TrainModelWorkChain", None, on_gpu=True, has_seat=False) == ("gpu", "running")
    assert classify_run("TrainModelWorkChain", {"waiting_for_gpu": True}, on_gpu=False, has_seat=False) == ("gpu", "queued")
    assert classify_run("EvalOnlyWorkChain", None, on_gpu=False, has_seat=False) == ("gpu", "queued")
    # a job that logged waiting stays queued even if a stale step is present
    assert classify_run("TrainEvalWorkChain", {"step": 10, "waiting_for_gpu": True}, on_gpu=False, has_seat=False) == ("gpu", "queued")
    # FLOW-3D: running only while holding the seat.
    assert classify_run("Flow3DBatchWorkChain", None, on_gpu=False, has_seat=True) == ("flow3d", "running")
    assert classify_run("Flow3DGenerateWorkChain", None, on_gpu=False, has_seat=False) == ("flow3d", "queued")
    # non-resource work is never queued behind a seat
    assert classify_run("SomeOtherWorkChain", None, on_gpu=False, has_seat=False) == ("other", "running")


def test_a_collection_needs_a_description_unless_skipped(svc, pino_simulation_factory):
    """713 of 802 collections had none, so a dataset's purpose lived only in
    whoever made it. A name is not a description: "bundle_gate" does not say
    what was gated or why."""
    sims = [pino_simulation_factory("needs-desc", group_name=f"nd-{i}") for i in range(2)]
    pks = [s.pk for s in sims]

    with pytest.raises(InvalidRequestError, match="description is required"):
        svc.create_collection(pks, "no-desc")
    with pytest.raises(InvalidRequestError, match="description is required"):
        svc.create_collection(pks, "blank-desc", description="   ")

    # ...and it can still be skipped on purpose
    out = svc.create_collection(pks, "skipped-desc", skip_description=True)
    assert out["pk"]


def test_broker_fault_tells_a_dead_broker_from_a_busy_database():
    """A pool timeout used to be reported as "the daemon is not reachable".

    `_submit` calls `daemon_available()` — a real RPC — three lines before the
    relabel, so the message contradicted a liveness check the same function had
    just passed, and the documented response (restart the container, restart the
    daemon) was aimed at the wrong service with runs in flight.
    """
    from sqlalchemy.exc import TimeoutError as SATimeoutError

    from aiida_pino.console.service import _broker_fault

    assert _broker_fault(ConnectionError("broker refused")) is True
    assert _broker_fault(SATimeoutError("QueuePool limit of size 5 overflow 10 reached")) is False
    assert _broker_fault(ValueError("workchain spec rejected the port")) is False
    # The real one arrives wrapped by whatever called it.
    wrapped = RuntimeError("submit failed")
    wrapped.__cause__ = SATimeoutError("QueuePool limit reached")
    assert _broker_fault(wrapped) is False


def test_flow3d_run_config_is_checked_at_submit_time():
    """The form's own default used to be a config the runner rejects.

    `run_config` was validated for truthiness and stored verbatim, while
    `execution_profile` beside it rejects unknown keys. flow3D_scripts requires
    `schema_version` to be exactly 1, so the untouched form (which shipped 3)
    built a workchain and a calcjob that could not succeed — and with
    prepare_only off, that malformed case waited behind the seat lock, days
    deep, to fail when it reached the front.
    """
    import pytest as _pytest

    from aiida_pino.utils.flow3d_config import validate_run_config

    good = {
        "schema_version": 1,
        "case_name": "console_case_1",
        "geometry": {"length_m": 0.0024},
        "mesh": {"cell_size_m": 7.5e-5},
        "laser": {"power_w": 300.0},
        "simulation": {"finish_time_s": 0.0085},
    }
    validate_run_config(good)  # must not raise

    with _pytest.raises(ValueError, match="schema_version must be exactly 1"):
        validate_run_config({**good, "schema_version": 3})
    with _pytest.raises(ValueError, match="missing required section"):
        validate_run_config({"schema_version": 1})
    with _pytest.raises(ValueError, match="case_name"):
        validate_run_config({**good, "case_name": "bad name/with slash"})


def test_parse_proc_locks_separates_the_holder_from_the_queue():
    """The spool's `running/` entry is not the seat.

    On 2026-08-04 it named a task queued since 08-02 while the flock was held by
    an AiiDA `flow3d_run_adapter.py`, with that spool task and two more AiiDA
    runs blocked behind it — so the console reported three FLOW-3D runs
    "running" under one lock and an empty queue. Only the kernel knows.
    """
    from aiida_pino.console.monitor import parse_proc_locks

    text = (
        "1: POSIX  ADVISORY  WRITE 111 103:02:99 0 EOF\n"
        "38: FLOCK  ADVISORY  WRITE 660466 103:02:73800397 0 EOF\n"
        "38: -> FLOCK  ADVISORY  WRITE 675609 103:02:73800397 0 EOF\n"
        "38:  -> FLOCK  ADVISORY  WRITE 698592 103:02:73800397 0 EOF\n"
        "40: FLOCK  ADVISORY  WRITE 900001 103:02:55 0 EOF\n"
    )
    holder, waiters = parse_proc_locks(text, 73800397)
    assert holder == 660466
    assert waiters == [675609, 698592]
    assert parse_proc_locks(text, 1234) == (None, [])


def test_split_qualified_names_the_split():
    """Two evaluations of one model rendered as identical rows.

    Model 1308's test and val evaluations both displayed "Straight path Ti64",
    same pk, in one table — and the split is the single most important qualifier
    on a reported metric.
    """
    from aiida_pino.console.service import _split_qualified

    assert _split_qualified("Straight path Ti64", "…noref (test split)") == "Straight path Ti64 (test split)"
    assert _split_qualified("Straight path Ti64", "…noref (val split)") == "Straight path Ti64 (val split)"
    # already qualified, or nothing to add
    assert _split_qualified("Ti64 test split", "x (test split)") == "Ti64 test split"
    assert _split_qualified("Ti64", "no split named here") == "Ti64"
    assert _split_qualified(None, "(test split)") is None


def test_unknown_filter_values_are_rejected(svc, aiida_localhost):
    """A typo'd filter returned an empty list under HTTP 200.

    "No datasets have that role" and "that is not a role" are different answers,
    and only one of them is true. Checked in the service so the CLI, which does
    not go through HTTP, is covered too.
    """
    svc.list_datasets(role="subset", limit=1)
    svc.list_datasets(source="all", limit=1)
    with pytest.raises(InvalidRequestError, match="unknown role"):
        svc.list_datasets(role="nonsense")
    with pytest.raises(InvalidRequestError, match="unknown source"):
        svc.list_datasets(source="nonsense")


def test_clamp_limit_only_treats_zero_as_no_cap():
    """`if limit <= 0` let -1 inherit the "no cap" sentinel.

    A `parseInt` over an empty input, or a decrementing control, then fired the
    most expensive query in the store instead of erroring.
    """
    from aiida_pino.console.config import Settings

    settings = Settings()
    assert settings.clamp_limit(0) is None                 # documented sentinel
    assert settings.clamp_limit(None) == settings.list_default_limit
    assert settings.clamp_limit(50) == 50
    for bad in (-1, -999):
        with pytest.raises(InvalidRequestError, match="must be 0"):
            settings.clamp_limit(bad)


def test_not_found_messages_name_the_expected_type(svc, pino_model_factory):
    """"node 999 not found: NotExistent" names AiiDA's class, not the thing."""
    from aiida_pino.console.service import _article, _human_type
    from aiida_pino.data import PinoDatasetCollectionData

    assert _human_type(PinoDatasetCollectionData) == "dataset"
    assert _human_type(None) == "node"
    assert _article("InstalledCode") == "an" and _article("PinoModelData") == "a"

    model = pino_model_factory()
    # dataset_detail accepts several node classes, so it names them all.
    with pytest.raises(NodeNotFoundError, match=r"^dataset( or \w+)* 999000111 not found$"):
        svc.dataset_detail(999_000_111)
    with pytest.raises(NodeNotFoundError, match=r"not a dataset"):
        svc.dataset_detail(model.pk)


def test_superseded_dataset_is_noted_not_refused(svc, pino_dataset_factory):
    """Filing *into* a retired dataset is refused; reading one is only noted.

    Re-running an old training to reproduce a published number is legitimate,
    but the pk can equally come from a stale note, and the picker hides
    superseded rows -- so nothing on this path said a word.
    """
    dataset = pino_dataset_factory("superseded-source")
    assert svc._superseded_note(dataset, "training") is None
    dataset.base.extras.set("superseded", True)
    dataset.base.extras.set("replaced_by_pk", 4242)
    note = svc._superseded_note(dataset, "training")
    assert note is not None
    assert "superseded" in note and "#4242" in note and "submitted" in note


def test_submit_validation_uses_the_runner_as_the_authority(tmp_path):
    """A copy of the runner's rules drifts; the runner itself does not.

    Measured 2026-08-05: eight configs the runner refuses in under 1.3 ms were
    accepted at submit — missing material file, mesh coarser than the beam, end
    point off the plate, negative power, T0 = 0 K, `recoil_model: 7`, a finish
    time shorter than the laser travel. Each cost a round trip of minutes ending
    in "runner exited with code 2".
    """
    import copy

    from aiida_pino.utils.flow3d_config import validate_run_config, validate_with_runner

    # Sibling checkout beside this repo, not an absolute home path.
    runner_root = str(Path(__file__).resolve().parents[2] / "flow3D_scripts")
    if not (Path(runner_root) / "python" / "run_rectangular_case.py").is_file():
        pytest.skip("flow3D_scripts checkout not available here")

    good = {
        "schema_version": 1,
        "case_name": "validation_probe",
        "material": {"name": "ti64_v3_calibrated"},
        "geometry": {"length_m": 0.0024, "width_m": 0.0012, "thickness_m": 0.0004, "gas_clearance_m": 0.0004},
        "mesh": {"cell_size_m": 7.5e-05},
        "laser": {
            "power_w": 300.0, "beam_radius_m": 0.00015, "focal_distance_m": 0.01,
            "lens_radius_m": 0.0003, "start_point_m": [0.0, 0.0, 0.0],
            "end_point_m": [0.0008, 0.0, 0.0], "scan_speed_m_per_s": 0.1,
        },
        "simulation": {"initial_temperature_k": 293.15, "finish_time_s": 0.0085, "max_iterations": 5000},
    }
    validate_run_config(good)
    validate_with_runner(good, runner_root)  # must not raise

    for dotted, value, needle in [
        ("mesh.cell_size_m", 1.5e-3, "cell_size_m"),
        ("laser.power_w", -300.0, "must not be negative"),
        ("simulation.initial_temperature_k", 0.0, "greater than zero"),
        ("simulation.recoil_model", 7, "recoil_model"),
    ]:
        bad = copy.deepcopy(good)
        section, _, key = dotted.partition(".")
        bad[section][key] = value
        with pytest.raises(ValueError, match=needle):
            validate_with_runner(bad, runner_root)


def test_validate_with_runner_is_a_no_op_without_a_checkout(tmp_path):
    """Best effort: no runner means the shallow checks still stand alone."""
    from aiida_pino.utils.flow3d_config import validate_with_runner

    validate_with_runner({"schema_version": 1}, tmp_path / "nowhere")  # must not raise


def test_batch_pending_cases_are_visible_in_the_queue(svc, aiida_profile_clean, tmp_path, monkeypatch):
    """With max_concurrent_runs=1 a 300-case batch is ONE process; the other 299
    cases exist only in its `run_configs` input, so every process tally missed
    them — one batch read as "nothing queued" while ~3 days of seat time was
    committed. The queue now reads pending = planned - submitted off the extras
    the workchain already stamps, and a pre-extras batch stays unknown (None),
    never 0.
    """
    from aiida.orm import WorkChainNode

    monkeypatch.setenv("F3D_SPOOL_DIR", str(tmp_path / "no-spool"))

    def _batch(extras):
        node = WorkChainNode()
        node.set_process_type("aiida.workflows:pino.flow3d_batch")
        node.base.attributes.set("process_label", "Flow3DBatchWorkChain")
        node.base.attributes.set("process_state", "running")
        node.store()
        for key, value in extras.items():
            node.base.extras.set(key, value)
        return node

    stamped = _batch({"planned_runs": 300, "completed_runs": 1})
    legacy = _batch({})  # pre-extras batch: unknown is not an empty queue

    out = svc.flow3d_queue_status()
    rows = {row["pk"]: row for row in out["aiida"]["batches"]}
    assert rows[stamped.pk]["pending_runs"] == 299
    assert rows[stamped.pk]["planned_runs"] == 300
    assert rows[legacy.pk]["pending_runs"] is None
    assert out["aiida"]["pending_batch_cases"] == 299
    assert out["total_waiting"] == 299, "committed batch cases must count as waiting"

    m = svc.monitor_status()
    assert m["flow3d"]["pending_batch_cases"] == 299


def test_tombstoned_cases_do_not_read_as_missing_stats(svc, monkeypatch):
    """A case the worker tombstoned as unavailable must not count as "missing".

    The background worker's `needs_stats()` honours `meltpool_stats_unavailable`
    (reduced-field datasets can never yield melt metrics), but the console's
    stats rows only ever looked for a current-version `meltpool_stats`. So every
    tombstoned case stayed in "Compute missing (N)" forever, and each click
    re-ran a multi-minute compute guaranteed to raise KeyError again. The two
    definitions of "owes a compute" have to agree.
    """
    import aiida.orm as _orm
    from aiida_pino.console import viz
    from aiida_pino.utils.stats_autocompute import UNAVAILABLE_KEY

    version = viz.STATS_VERSION
    store = {
        "u-ok": {viz.STATS_EXTRA_KEY: {"version": version, "regime": "keyhole"}},
        "u-dead": {UNAVAILABLE_KEY: {"version": version, "reason": "missing field Melt Region"}},
        "u-todo": {},
    }

    class _Extras:
        def __init__(self, d): self._d = d
        def get(self, key, default=None): return self._d.get(key, default)

    class _Node:
        def __init__(self, d): self.base = type("B", (), {"extras": _Extras(d)})()

    monkeypatch.setattr(_orm, "load_node", lambda uuid: _Node(store[uuid]))
    monkeypatch.setattr(svc, "dataset_cases", lambda pk: {"cases": [
        {"uuid": "u-ok", "pk": 1, "case_name": "ok"},
        {"uuid": "u-dead", "pk": 2, "case_name": "dead"},
        {"uuid": "u-todo", "pk": 3, "case_name": "todo"},
    ]})

    summary = svc.dataset_stats(7)["summary"]
    assert summary["total"] == 3
    assert summary["computed"] == 1
    assert summary["unavailable"] == 1, "tombstoned case must be reported, not hidden in 'missing'"
    # what the button offers to compute: exactly the one real todo
    assert summary["total"] - summary["computed"] - summary["unavailable"] == 1

    # Compute now happens in a child process (see `_compute_case_stats_isolated`),
    # so count at that seam; the assertion is unchanged.
    from aiida_pino.console import service as service_mod

    calls = []
    monkeypatch.setattr(service_mod, "_compute_case_stats_isolated",
                        lambda uuid: calls.append(uuid) or True)
    result = svc.dataset_stats_compute(7, limit=25)
    assert calls == ["u-todo"], "must not recompute the tombstoned case"
    assert result["remaining"] == 0


def test_batch_stats_compute_releases_memory_after_the_batch(svc, monkeypatch):
    """The batch path must hand freed memory back, not grow the web process.

    `compute_case_metrics` loads whole field stacks (810 MB for one field of a
    337-frame 600k-cell case). Python frees them, but glibc raises its dynamic
    mmap threshold once it has seen large frees, so those blocks come from the
    arena and RSS never drops: one measured case grew the console 12.03 ->
    16.40 GiB and stayed there, which is how a long-lived console reached 89 GB
    and drove the box under its MemAvailable floor. A batch visits each case
    once, so the viewer's field-stack cache buys it nothing either.
    """
    from aiida_pino.console import service as service_mod
    from aiida_pino.console import viz

    monkeypatch.setattr(svc, "dataset_cases", lambda pk: {"cases": [
        {"uuid": "u-1", "pk": 1, "case_name": "one"},
    ]})
    monkeypatch.setattr(svc, "_dataset_stats_rows", lambda pk: [
        {"uuid": "u-1", "pk": 1, "case_name": "one", "stats": None, "unavailable": None},
    ])
    monkeypatch.setattr(service_mod, "load_node", lambda uuid: object(), raising=False)
    monkeypatch.setattr(viz, "get_or_compute_case_stats", lambda sim: {"version": viz.STATS_VERSION})

    calls = {"cleared": 0, "trimmed": 0, "isolated": []}
    monkeypatch.setattr(viz, "clear_caches", lambda: calls.__setitem__("cleared", calls["cleared"] + 1))
    monkeypatch.setattr(service_mod, "_trim_malloc_arena",
                        lambda: calls.__setitem__("trimmed", calls["trimmed"] + 1))
    # The compute itself must NOT run in this process: malloc_trim recovered
    # only 10 of the 22 GiB a single measured case added, so the web process
    # still ratchets. A child process returns 100% of it by exiting.
    monkeypatch.setattr(service_mod, "_compute_case_stats_isolated",
                        lambda uuid: calls["isolated"].append(uuid) or True)
    monkeypatch.setattr(viz, "get_or_compute_case_stats",
                        lambda sim: pytest.fail("stats must not be computed in the web process"))

    result = svc.dataset_stats_compute(7, limit=25)
    assert calls["isolated"] == ["u-1"], "each case must be computed in a child process"
    assert result["computed_now"] == 1
    assert calls["cleared"] == 1, "batch must drop the field-stack caches it filled"
    assert calls["trimmed"] == 1, "batch must return the freed arena to the OS"
