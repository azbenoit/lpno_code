"""The 2026-08-09 friction batch: names, manifests, sizes, stale pks, lost CSVs.

Six items were logged as friction; each premise was measured before the
fix landed (see database.md "Names, sizes and stale pks").
"""

from __future__ import annotations

import json

from aiida import orm

from aiida_pino.data.pino_simulation import case_manifest


def test_case_manifest_accepts_every_shape_it_meets(tmp_path):
    manifest = {"run": {"case_name": "c1"}, "input": {"laser": {"power_w": 200.0}},
                "source_config": "/somewhere/config.json"}

    # the dict shapes: bare manifest, metadata wrapper
    assert case_manifest(manifest) == manifest
    assert case_manifest({"flow3d_case_manifest": manifest}) == manifest

    # the string shapes: JSON text, and a sidecar file on disk
    assert case_manifest(json.dumps(manifest)) == manifest
    sidecar = tmp_path / "c1.manifest.json"
    sidecar.write_text(json.dumps(manifest))
    assert case_manifest(sidecar) == manifest
    assert case_manifest(str(sidecar)) == manifest

    # a metadata dict whose manifest is (wrongly but observedly) a string
    assert case_manifest({"flow3d_case_manifest": json.dumps(manifest)}) == manifest

    # emptiness never raises
    assert case_manifest(None) == {}
    assert case_manifest({}) == {}


def test_source_config_is_a_path_by_design():
    """3,857 of 3,942 sims carry a path string there; its parsed form is `input`.

    The defensive double-parse came from trying to json.loads it.
    """
    manifest = case_manifest({"input": {"a": 1}, "source_config": "/cfg/run.json"})
    assert isinstance(manifest["source_config"], str)
    assert isinstance(manifest["input"], dict)


def test_collections_stamp_num_members(aiida_profile_clean, tmp_path):
    """Finding a collection by size is one filter, not a client-side len()."""
    from aiida_pino.data import PinoDatasetCollectionData

    from test_pack_index_export import _collection, _npz_file_sim

    sims = [_npz_file_sim(tmp_path / f"src_{i}", f"case_{i}", seed=i) for i in range(3)]
    coll = _collection(sims, "num_members_probe")
    assert coll.base.extras.get("num_members") == 3
    found = orm.QueryBuilder().append(
        PinoDatasetCollectionData, filters={"extras.num_members": 3}, project=["id"],
    ).all(flat=True)
    assert coll.pk in found


def test_model_archive_keeps_the_sigma_csvs(aiida_profile_clean, tmp_path):
    """The per-seed validator CSVs used to die with the /tmp workdir."""
    from aiida_pino.data.pino_model import (
        MODEL_KEEP_DIRECTORIES,
        MODEL_KEEP_EXTENSIONS,
        _copy_selected_model_payload,
    )

    assert ".csv" in MODEL_KEEP_EXTENSIONS
    assert "validators" in MODEL_KEEP_DIRECTORIES

    src = tmp_path / "training_output"
    (src / "validators").mkdir(parents=True)
    (src / "validators" / "val_error.csv").write_text("step,mae\n1,0.5\n")
    (src / "u_network.0.pth").write_bytes(b"ckpt")
    (src / "noise.log").write_text("not archived")
    folder = orm.FolderData(tree=str(src)).store()

    out = tmp_path / "model_payload"
    out.mkdir()
    copied = _copy_selected_model_payload(folder, out)
    assert "validators/val_error.csv" in copied
    assert "u_network.0.pth" in copied
    assert "noise.log" not in copied
