"""Pack-slice reading: byte-identity with real files, and loud failure on repack.

The index turns a staged dataset from tens of GB of copies into a few KB of
JSON, so the one thing these tests must make impossible is a *silent* wrong
read: a moved offset must raise naming the cause, never return plausible
arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lpno.custom_utils.npz_datasource import _case_files
from lpno.custom_utils.pack_slice import (
    PACK_INDEX_FILENAME,
    PackSlicePath,
    load_pack_index,
    open_case,
)

_SIGNATURE = dict(
    x=np.linspace(0, 2e-3, 6).astype(np.float32),
    y=np.linspace(-5e-4, 5e-4, 5).astype(np.float32),
    z=np.linspace(-5e-4, 2e-4, 4).astype(np.float32),
    time_s=np.linspace(0, 1e-3, 3).astype(np.float64),
    power_w=np.float64(150.0),
    scan_speed_m_per_s=np.float64(1.2),
)


def _case_bytes(case_name: str, seed: int) -> bytes:
    import io

    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        case_name=np.str_(case_name),
        Temperature=(300 + 1500 * rng.random((3, 6, 5, 4))).astype(np.float32),
        **_SIGNATURE,
    )
    return buffer.getvalue()


def _pack_with_index(root: Path, cases: dict[str, bytes]) -> Path:
    """One fake pack holding every case back to back, plus the index."""
    root.mkdir(parents=True, exist_ok=True)
    pack = root / "pack_0"
    entries = []
    blob = b""
    for name, payload in cases.items():
        entries.append({
            "filename": f"{name}.npz",
            "pack_path": str(pack),
            "offset": len(blob) + 7,   # deliberate non-zero, non-aligned offsets
            "size": len(payload),
        })
        blob += b"\x00" * 7 + payload
    pack.write_bytes(blob)
    (root / PACK_INDEX_FILENAME).write_text(json.dumps({
        "format": "pino_pack_index_v1",
        "entries": entries,
    }))
    return root


def test_sliced_cases_read_byte_identical_to_real_files(tmp_path):
    payloads = {f"case_{i}": _case_bytes(f"case_{i}", i) for i in range(3)}
    staged = _pack_with_index(tmp_path / "staged", payloads)

    cases = _case_files(staged)
    assert len(cases) == 3 and all(isinstance(c, PackSlicePath) for c in cases)

    for case in cases:
        with np.load(open_case(case)) as sliced:
            real = np.load(__import__("io").BytesIO(payloads[case.stem]))
            for key in real.files:
                np.testing.assert_array_equal(sliced[key], real[key])
            real.close()


def test_a_real_file_wins_a_name_collision(tmp_path):
    payloads = {"case_0": _case_bytes("case_0", 0)}
    staged = _pack_with_index(tmp_path / "staged", payloads)
    # A repair materialized the same case as a real file with DIFFERENT content:
    # the real file is the truth the repair re-exported.
    (staged / "case_0.npz").write_bytes(_case_bytes("case_0", 99))

    cases = _case_files(staged)
    assert len(cases) == 1
    assert isinstance(cases[0], Path)


def test_repack_is_loud_not_silent(tmp_path):
    payloads = {"case_0": _case_bytes("case_0", 0)}
    staged = _pack_with_index(tmp_path / "staged", payloads)
    pack = staged / "pack_0"

    # Repack scenario 1: the pack shrank (offsets past the end).
    pack.write_bytes(pack.read_bytes()[: 100])
    entry = load_pack_index(staged)[0]
    with pytest.raises(FileNotFoundError, match="repacked"):
        entry.readable()

    # Repack scenario 2: same length, different content at the offset.
    pack.write_bytes(b"\x01" * (7 + len(payloads["case_0"])))
    with pytest.raises(ValueError, match="repacked"):
        entry.readable()


def test_kill_switch_disables_every_index(tmp_path, monkeypatch):
    payloads = {"case_0": _case_bytes("case_0", 0)}
    staged = _pack_with_index(tmp_path / "staged", payloads)
    monkeypatch.setenv("PINO_PACK_INDEX_DISABLE", "1")
    assert load_pack_index(staged) == []
    with pytest.raises(ValueError, match="No consolidated case NPZ"):
        _case_files(staged)


def test_fspath_refuses_to_leak_the_whole_pack(tmp_path):
    payloads = {"case_0": _case_bytes("case_0", 0)}
    staged = _pack_with_index(tmp_path / "staged", payloads)
    entry = load_pack_index(staged)[0]
    import os

    with pytest.raises(TypeError, match="pack slice"):
        os.fspath(entry)


def test_case_discovery_survives_pack_slice_being_unavailable(tmp_path, monkeypatch):
    """An optional accelerator must never break the core path.

    The first version imported `pack_slice` at the top of `_case_files`, so a
    checkout carrying the npz_datasource change without `pack_slice.py` — the
    window between the two files being written — broke case discovery outright,
    and a reviewer hit exactly that. Zero-copy staging is an accelerator:
    without it, real files must still load.
    """
    from lpno.custom_utils import npz_datasource

    real = tmp_path / "case_0.npz"
    real.write_bytes(_case_bytes("case_0", 0))

    # What the guarded import does when pack_slice cannot be imported.
    monkeypatch.setattr(npz_datasource, "_pack_index_entries", lambda directory: [])
    cases = npz_datasource._case_files(tmp_path)
    assert [c.name for c in cases] == ["case_0.npz"]
    with np.load(npz_datasource._open_case_safely(cases[0])) as data:
        assert "time_s" in data.files


def test_index_only_staging_is_recognized_as_consolidated_npz(tmp_path):
    """The 2026-08-14 rollout failure: pure index staging must dispatch to NpzDirStore.

    With zero-copy staging on by default, a staged dataset has pack_index.json and
    NO real per-case .npz files. `is_consolidated_npz` only globbed real files, so
    `load_and_preprocess_data` fell through to the arraydir store, which rejects
    the npz_file member manifests: run 46347 died with "Unsupported runtime format
    ... aiida_npz_file_v1" before training a single step. Detection must consult
    the index exactly like `_case_files` does.
    """
    from lpno.custom_utils.npz_datasource import is_consolidated_npz

    payloads = {f"case_{i}": _case_bytes(f"case_{i}", i) for i in range(2)}
    staged = _pack_with_index(tmp_path / "staged", payloads)
    assert not list(staged.glob("*.npz")), "precondition: no real npz files"

    assert is_consolidated_npz(staged) is True
