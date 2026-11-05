"""Restoring a published deposit's payloads into the deposited nodes themselves.

2026-10-07, reproduction test from the deposit: the archive's simulation nodes are
metadata-only, and the documented way back (`from_npz_file`) made NEW nodes that no deposited
collection lists, so nothing downstream could be re-run. `restore_data` writes each payload
back into its own node, keeping the UUID every collection, model and evaluation refers to.
"""
from __future__ import annotations

import csv
import hashlib
import tarfile

import numpy as np
import pytest

from aiida_pino.data.pino_simulation import DataEvictedError, PinoSimulationData, payload_content_hash
from aiida_pino.utils.deposit_restore import DepositPayloads, restore_from_deposit


def _deposited_sim(tmp_path, case_name="dep_case", nt=4):
    rng = np.random.default_rng(1)
    path = tmp_path / f"{case_name}.npz"
    np.savez_compressed(
        path,
        x=np.linspace(0.0, 1.0, 4).astype(np.float32),
        y=np.linspace(0.0, 2.0, 3).astype(np.float32),
        z=np.linspace(0.0, 3.0, 5).astype(np.float32),
        time_s=np.linspace(1e-6, 1e-4, nt).astype(np.float64),
        power_w=np.float64(200.0),
        scan_speed_m_per_s=np.float64(0.5),
        case_name=np.str_(case_name),
        Temperature=rng.random((nt, 4, 3, 5), dtype=np.float32) * 3000.0,
    )
    metadata = {"source_type": "flow3d", "case_name": case_name, "laser_power": 200.0, "V_scan": 0.5}
    sim = PinoSimulationData.from_npz_file(path, metadata=metadata).store()
    return sim, path


def _deposit(tmp_path, sim, payload, *, family="fam", sha=None):
    root = tmp_path / "deposit"
    root.mkdir()
    with tarfile.open(root / f"payloads_{family}.tar", "w") as archive:
        archive.add(payload, arcname=f"{sim.uuid}.npz")
    digest = sha or hashlib.sha256(payload.read_bytes()).hexdigest()
    with open(root / "payload_manifest.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uuid", "family", "split", "case_name", "bytes", "sha256"])
        writer.writerow([sim.uuid, family, "test", "dep_case", payload.stat().st_size, digest])
    return root


def test_restore_puts_the_arrays_back_into_the_same_node(aiida_profile_clean, tmp_path):
    from aiida import orm

    sim, payload = _deposited_sim(tmp_path)
    name = sim.list_original_array_names()[-1]
    before, times = sim.get_array_by_original_name(name), sim.get_time_values_s()
    key = payload_content_hash(sim)
    sim.evict_data(reason="deposit")
    with pytest.raises(DataEvictedError, match="restore-payloads"):
        sim.get_array_by_original_name(name)
    with pytest.raises(DataEvictedError):  # it used to answer None, which read as "no time axis"
        sim.get_time_values_s()

    plan = restore_from_deposit(DepositPayloads(_deposit(tmp_path, sim, payload)), [sim])
    assert plan["dry_run"] and plan["to_restore"] == 1 and not plan["missing_tars"]
    assert sim.is_data_evicted(), "a dry run writes nothing"

    done = restore_from_deposit(DepositPayloads(tmp_path / "deposit"), [sim], dry_run=False)
    assert done["restored"] == 1
    again = orm.load_node(sim.uuid)
    assert not again.is_data_evicted()
    np.testing.assert_array_equal(again.get_array_by_original_name(name), before)
    np.testing.assert_array_equal(again.get_time_values_s(), times)
    assert payload_content_hash(again) == key
    assert again.base.extras.get("data_restored")["sha256"] == key


def test_a_payload_with_the_wrong_hash_is_refused_and_nothing_persists(aiida_profile_clean, tmp_path):
    sim, payload = _deposited_sim(tmp_path)
    sim.evict_data()
    deposit = DepositPayloads(_deposit(tmp_path, sim, payload, sha="0" * 64))
    with pytest.raises(ValueError, match="sha256"):
        restore_from_deposit(deposit, [sim], dry_run=False)
    assert sim.is_data_evicted()
    assert sim.base.repository.list_object_names() == []


def test_restoring_a_present_simulation_is_a_no_op(aiida_profile_clean, tmp_path):
    sim, payload = _deposited_sim(tmp_path)
    assert sim.restore_data(payload)["restored"] is False
