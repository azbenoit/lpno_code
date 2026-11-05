"""Unit test for build_random_path_split.py --from-split.

Rebuilding a split with a spatial stride normally re-randomizes the assignment,
which would break provenance: the AiiDA collection records which case is in
train/val/test. --from-split copies an existing split's membership verbatim and
only applies the stride, so a downsampled dataset stays the same dataset.
Numpy-only, no GPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "build_random_path_split",
    Path(__file__).resolve().parents[1] / "scripts" / "build_random_path_split.py",
)
_brps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_brps)

NX, NY, NZ = 8, 6, 4


def _write_case(root: Path, name: str, nt: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / f"{name}.npz",
        x=np.linspace(0.0, 7e-5, NX).astype(np.float32),
        y=np.linspace(0.0, 5e-5, NY).astype(np.float32),
        z=np.linspace(-3e-5, 0.0, NZ).astype(np.float32),
        time_s=np.linspace(0.0, 1e-4, nt).astype(np.float32),
        power_w=np.float64(120.0),
        Temperature=np.arange(nt * NX * NY * NZ, dtype=np.float32).reshape(nt, NX, NY, NZ),
    )
    (root / f"{name}.manifest.json").write_text(json.dumps({"input": {}}))


def test_from_split_preserves_membership_and_strides(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "src"
    # Deliberately uneven, variable-length cases: the lazy path's whole point.
    _write_case(src / "train", "c_000", nt=5)
    _write_case(src / "train", "c_001", nt=9)
    _write_case(src / "val", "c_002", nt=7)
    _write_case(src / "test", "c_003", nt=6)
    dest = tmp_path / "dest"

    monkeypatch.setattr(
        "sys.argv",
        ["build_random_path_split.py", "--from-split", str(src), "--dest", str(dest),
         "--frames", "0", "--stride", "2", "2", "1"],
    )
    _brps.main()

    assert sorted(p.stem for p in (dest / "train").glob("*.npz")) == ["c_000", "c_001"]
    assert [p.stem for p in (dest / "val").glob("*.npz")] == ["c_002"]
    assert [p.stem for p in (dest / "test").glob("*.npz")] == ["c_003"]

    with np.load(dest / "train" / "c_001.npz") as d:
        assert d["Temperature"].shape == (9, NX // 2, NY // 2, NZ)  # all frames, strided space
        assert d["x"].size == NX // 2 and d["y"].size == NY // 2 and d["z"].size == NZ
    with np.load(dest / "val" / "c_002.npz") as d:
        assert d["Temperature"].shape[0] == 7  # per-case length preserved, not equalized
    assert (dest / "train" / "c_001.manifest.json").is_file()


def test_from_split_is_resumable(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "src"
    _write_case(src / "train", "c_000", nt=5)
    dest = tmp_path / "dest"
    argv = ["build_random_path_split.py", "--from-split", str(src), "--dest", str(dest),
            "--frames", "0", "--stride", "2", "2", "1"]

    monkeypatch.setattr("sys.argv", argv)
    _brps.main()
    stamp = (dest / "train" / "c_000.npz").stat().st_mtime_ns
    monkeypatch.setattr("sys.argv", argv)
    _brps.main()  # second pass must skip the finished case, not rewrite it

    assert (dest / "train" / "c_000.npz").stat().st_mtime_ns == stamp
