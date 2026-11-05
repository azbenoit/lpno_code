from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_registry import discover_models


def _mk_model(root, mid, meta=None, with_ckpt=True):
    d = root / mid
    d.mkdir(parents=True)
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
    if with_ckpt:
        (d / "u_network.0.pth").write_bytes(b"x")


def test_discover_models_sorted_with_meta(tmp_path):
    _mk_model(tmp_path, "rigno", {"label": "RIGNO", "type": "rigno", "order": 2, "default": True})
    _mk_model(tmp_path, "fno", {"label": "FNO", "type": "fno", "order": 1})
    _mk_model(tmp_path, "bare", None)  # no meta but has a checkpoint -> label = id, order last
    got = discover_models(tmp_path)
    assert [m["id"] for m in got] == ["fno", "rigno", "bare"]
    assert got[0]["label"] == "FNO"
    assert got[1]["default"] is True
    assert got[2]["label"] == "bare" and got[2]["default"] is False


def test_discover_models_skips_non_model_dirs(tmp_path):
    (tmp_path / "empty").mkdir()  # neither meta.json nor a checkpoint -> skipped
    assert discover_models(tmp_path) == []
    assert discover_models(tmp_path / "missing") == []  # missing dir -> empty, no error


def test_models_endpoint_lists_folder(tmp_path):
    _mk_model(tmp_path, "fno", {"label": "FNO", "order": 1})
    app = create_app(
        settings=Settings(startup_load_runtime=False, use_stub_runtime=True, straight_models_dir=str(tmp_path))
    )
    with TestClient(app) as client:
        resp = client.get("/models?page=straight")
        assert resp.status_code == 200
        body = resp.json()
        assert body["page"] == "straight"
        assert [m["id"] for m in body["models"]] == ["fno"]
        assert client.get("/models?page=bogus").status_code == 422  # page is a Literal


def test_discover_models_is_cached_until_the_folder_changes(tmp_path):
    """Discovery runs per request (id validation on every route), so it is
    served from a cache keyed on the directory's and each meta.json's mtime."""
    import os

    from app.model_registry import model_entry, physics_for

    _mk_model(tmp_path, "a", {"label": "A", "physics": {"p_watts": {"min": 1, "max": 2, "default": 1.5}}})
    first = discover_models(tmp_path)
    assert discover_models(tmp_path) is first                 # hit: same list object
    assert physics_for(tmp_path, "a").p_range == (1.0, 2.0)   # parsed once, on the entry
    assert physics_for(tmp_path, "nope").p_range is None      # unknown -> defaults, no raise
    assert model_entry(tmp_path, "nope") is None

    # An in-place meta.json edit: bump the mtime explicitly (kernel timestamp
    # granularity can be coarser than this test) -> re-read.
    (tmp_path / "a" / "meta.json").write_text(json.dumps({"label": "A2"}))
    os.utime(tmp_path / "a" / "meta.json", ns=(1, 10**18))
    assert discover_models(tmp_path)[0]["label"] == "A2"
    assert physics_for(tmp_path, "a").p_range is None

    # A folder added -> the directory mtime moves -> re-read.
    _mk_model(tmp_path, "b", {"label": "B"})
    os.utime(tmp_path, ns=(1, 10**18))
    assert [m["id"] for m in discover_models(tmp_path)] == ["a", "b"]

    # A meta-less folder that gains its checkpoint later: only the subfolder's
    # mtime moves, and that must be enough to list it.
    (tmp_path / "c").mkdir()
    os.utime(tmp_path, ns=(1, 10**18))
    assert [m["id"] for m in discover_models(tmp_path)] == ["a", "b"]
    (tmp_path / "c" / "u_network.0.pth").write_bytes(b"")
    os.utime(tmp_path / "c", ns=(1, 10**18))
    assert [m["id"] for m in discover_models(tmp_path)] == ["a", "b", "c"]
