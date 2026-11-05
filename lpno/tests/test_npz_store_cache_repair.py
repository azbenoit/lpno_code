"""A case file that vanishes mid-run is repaired once, not fatal.

The staged dataset dir is a symlink into aiida-pino's runtime cache, and that
cache is (was) documented as safe to `rm -rf`. Clearing it while a training
reads from it kills the run: a DataLoader worker raises `FileNotFoundError` for
one case NPZ and the whole job dies. Measured cost so far: seven runs, including
a RIGNO training killed at step 18000 after ten hours.

`aiida_pino.utils.runtime_repair.repair_dataset_dir` can rebuild the cache from
the symlink's uuid, so the store asks it once before giving up. A single retry
is the right shape: it turns a lost run into a pause, while a genuinely missing
file still fails instead of looping.

lpno does not depend on aiida-pino, so the hook is resolved lazily
and its absence is not an error.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from lpno.models.pipeline.npz_lazy_store import NpzDirStore
from lpno.custom_utils import npz_datasource as nds

from test_npz_lazy_store import _write_cases


def _store(tmp_path, **kw):
    return NpzDirStore(
        tmp_path,
        requested_fields=["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"],
        path_channels=list(nds.PATH_CHANNEL_NAMES),
        material_channels=list(nds.MATERIAL_CHANNEL_NAMES),
        **kw,
    )


@pytest.fixture
def fake_repair(monkeypatch):
    """Install a stub aiida_pino.utils.runtime_repair that records its calls."""
    calls: list[str] = []
    restored: dict[str, bytes] = {}

    def repair_dataset_dir(path):
        calls.append(str(path))
        for name, blob in restored.items():
            (path / name).write_bytes(blob)
        return bool(restored)

    mod = types.ModuleType("aiida_pino.utils.runtime_repair")
    mod.repair_dataset_dir = repair_dataset_dir
    pkg = types.ModuleType("aiida_pino")
    utils = types.ModuleType("aiida_pino.utils")
    monkeypatch.setitem(sys.modules, "aiida_pino", pkg)
    monkeypatch.setitem(sys.modules, "aiida_pino.utils", utils)
    monkeypatch.setitem(sys.modules, "aiida_pino.utils.runtime_repair", mod)
    return calls, restored


def test_missing_case_is_repaired_and_read(tmp_path, fake_repair):
    calls, restored = fake_repair
    _write_cases(tmp_path, n=2)
    s = _store(tmp_path)
    target = s.cases[0]
    restored[target.name] = target.read_bytes()
    restored[target.stem + ".manifest.json"] = (tmp_path / f"{target.stem}.manifest.json").read_bytes()
    target.unlink()

    raw = s.read_case(0, time_idx=1)  # must not raise
    assert raw["Temperature"].shape[0] == 1
    assert len(calls) == 1, "repair should be attempted exactly once"


def test_repair_is_not_called_when_the_file_is_present(tmp_path, fake_repair):
    calls, _ = fake_repair
    _write_cases(tmp_path, n=2)
    s = _store(tmp_path)
    s.read_case(0, time_idx=0)
    assert calls == []


def test_a_genuinely_missing_case_still_raises(tmp_path, fake_repair):
    """Repair that restores nothing must not loop or mask the error."""
    calls, _ = fake_repair  # restored stays empty
    _write_cases(tmp_path, n=2)
    s = _store(tmp_path)
    s.cases[0].unlink()
    with pytest.raises(FileNotFoundError):
        s.read_case(0, time_idx=0)
    assert len(calls) == 1


def test_missing_aiida_pino_is_not_an_error(tmp_path, monkeypatch):
    """lpno must stay usable without aiida-pino installed."""
    monkeypatch.setitem(sys.modules, "aiida_pino", None)
    _write_cases(tmp_path, n=2)
    s = _store(tmp_path)
    s.cases[0].unlink()
    with pytest.raises(FileNotFoundError):
        s.read_case(0, time_idx=0)


def test_missing_manifest_sidecar_is_repaired_too(tmp_path, fake_repair):
    """The sidecar half of the same failure: guarded like the NPZ half.

    Clearing the runtime cache removes `<case>.manifest.json` alongside
    `<case>.npz`. The NPZ half retried `repair_dataset_dir` (`_existing_case_path`);
    the manifest half raised straight out of `_require_manifest`, so which error a
    run died with depended only on which file the code touched first.

    Two of the three `_require_manifest` call sites index `self.cases` directly and
    never pass through `_existing_case_path`, so they had no repair at all.
    """
    calls, restored = fake_repair
    _write_cases(tmp_path, n=2)
    s = _store(tmp_path)
    target = s.cases[0]
    sidecar = tmp_path / f"{target.stem}.manifest.json"
    restored[sidecar.name] = sidecar.read_bytes()
    sidecar.unlink()

    manifest = s._require_manifest(target)  # must not raise
    assert isinstance(manifest, dict) and manifest
    assert len(calls) == 1, "repair should be attempted exactly once"


def test_genuinely_absent_manifest_still_raises(tmp_path, fake_repair):
    """Repair that cannot help must not turn a real error into a loop or a silence."""
    calls, restored = fake_repair  # `restored` left empty: repair fixes nothing
    _write_cases(tmp_path, n=1)
    s = _store(tmp_path)
    target = s.cases[0]
    (tmp_path / f"{target.stem}.manifest.json").unlink()

    with pytest.raises(ValueError, match="manifest"):
        s._require_manifest(target)
    assert len(calls) == 1, "one attempt, then give up"


def test_manifest_repair_survives_absent_aiida_pino(tmp_path, monkeypatch):
    """lpno does not depend on aiida-pino; its absence is not an error."""
    import builtins

    _write_cases(tmp_path, n=1)
    s = _store(tmp_path)
    target = s.cases[0]
    (tmp_path / f"{target.stem}.manifest.json").unlink()

    real_import = builtins.__import__

    def no_aiida(name, *a, **kw):
        if name.startswith("aiida_pino"):
            raise ImportError("no aiida-pino here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_aiida)
    with pytest.raises(ValueError, match="manifest"):
        s._require_manifest(target)
