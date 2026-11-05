"""Runtime-cache self-repair: a dangling staged symlink rebuilds itself."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiida_pino.utils import runtime_repair

UUID = "1fe4799c-5514-47f3-a901-aa70a896b7c2"


def _staged(tmp_path: Path, cache_root: Path) -> Path:
    target = cache_root / UUID
    link = tmp_path / "train_dataset"
    link.symlink_to(target)
    return link


def test_noop_when_cache_is_healthy(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / UUID).mkdir(parents=True)
    (cache / UUID / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(cache))
    assert runtime_repair.repair_dataset_dir(_staged(tmp_path, cache)) is False


def test_noop_when_target_is_not_a_cache_uuid(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(cache))
    other = tmp_path / "elsewhere" / "not-a-uuid"
    link = tmp_path / "train_dataset"
    link.symlink_to(other)
    assert runtime_repair.repair_dataset_dir(link) is False


def test_dangling_cache_symlink_is_rebuilt_from_the_node(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(cache))
    link = _staged(tmp_path, cache)
    assert not (cache / UUID).exists()

    class _Node:
        def export_runtime_arraydir(self, path: Path) -> None:
            Path(path).mkdir(parents=True)
            (Path(path) / "manifest.json").write_text(json.dumps({"members": []}), encoding="utf-8")

    monkeypatch.setattr(runtime_repair, "load_profile", lambda *a, **k: None, raising=False)
    import aiida.orm

    monkeypatch.setattr("aiida.orm.load_node", lambda uuid: _Node())
    monkeypatch.setattr("aiida.load_profile", lambda *a, **k: None)

    assert runtime_repair.repair_dataset_dir(link) is True
    assert (cache / UUID / "manifest.json").is_file()
    # The staged symlink now resolves again.
    assert (link / "manifest.json").is_file()


def test_repair_staged_symlinks_scans_cwd(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(cache))
    job = tmp_path / "job"
    job.mkdir()
    (job / "train_dataset").symlink_to(cache / UUID)
    (job / "val_dataset").symlink_to(cache / UUID)
    (job / "not_a_dataset").symlink_to(tmp_path / "elsewhere")  # ignored

    class _Node:
        def export_runtime_arraydir(self, path: Path) -> None:
            Path(path).mkdir(parents=True)
            (Path(path) / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("aiida.orm.load_node", lambda uuid: _Node())
    monkeypatch.setattr("aiida.load_profile", lambda *a, **k: None)

    n = runtime_repair.repair_staged_symlinks_in(job)
    assert n == 1  # one uuid rebuilt (both links resolve to it; second is a no-op)
    assert (cache / UUID / "manifest.json").is_file()
    assert (job / "train_dataset" / "manifest.json").is_file()


def test_export_without_manifest_is_a_hard_error(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(cache))
    link = _staged(tmp_path, cache)

    class _BadNode:
        def export_runtime_arraydir(self, path: Path) -> None:
            Path(path).mkdir(parents=True)  # no manifest written

    monkeypatch.setattr("aiida.orm.load_node", lambda uuid: _BadNode())
    monkeypatch.setattr("aiida.load_profile", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        runtime_repair.repair_dataset_dir(link)
