"""Regression: aiida-pino stages npz_file-mode collections the training path must read.

`aiida-pino` exports a dataset collection whose members are stored as whole case
NPZs by hardlinking each `<case>.npz` flat into the collection root and writing a
member manifest tagged `aiida_npz_file_v1` (PinoSimulationData.NPZ_RUNTIME_MANIFEST_VERSION).
The root manifest still says `aiida_arraydir_v2`, so `is_arraydir_runtime` matched
and `ArrayDirCollectionStore` tried to read the members — which only accept
`aiida_arraydir_v2` — and every AiiDA-launched training on such a dataset died with

    ValueError: Unsupported runtime format in .../members/simulation_0000: aiida_npz_file_v1

The flat case NPZs at the root are exactly what `NpzDirStore` consumes, and it is
also the only store that synthesizes the laser-path channels (beam_x/beam_dist/
path_dose/path_dose_recent) that random-path configs list in FNO_input_keys, so
the loader routes npz_file-mode collections there.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lpno.models.pipeline.lazy_runtime import (
    is_arraydir_runtime,
    is_npz_file_collection,
)
from lpno.models.pipeline.preprocessing_datasets import load_and_preprocess_data
from lpno.models.pipeline.npz_lazy_store import NpzDirStore

from test_npz_lazy_store import _manifest, _write_cases


def _stage_aiida_npz_file_collection(root: Path, n: int = 2) -> None:
    """Mirror aiida-pino's npz_file staging: flat case NPZs + v1 member manifests."""
    nx, ny, nz, nt = _write_cases(root, n=n)
    members = []
    for i in range(n):
        case = f"mlhs_{i:03d}"
        member_dir = root / "members" / f"simulation_{i:04d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        (member_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "aiida_npz_file_v1",
                    "source_kind": "simulation",
                    "storage_mode": "npz_file",
                    "simulation_uuid": f"sim-{i}",
                    "simulation_pk": i + 1,
                    "case_name": case,
                    "npz_file": f"{case}.npz",
                    "case_manifest_file": f"{case}.manifest.json",
                    "array_aliases": {},
                    "simulation_metadata": {"source_type": "flow3d"},
                }
            ),
            encoding="utf-8",
        )
        members.append(
            {
                "relative_path": str(member_dir.relative_to(root)),
                "manifest_file": str((member_dir / "manifest.json").relative_to(root)),
                "case_index": i,
                "simulation_pk": i + 1,
                "simulation_uuid": f"sim-{i}",
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "aiida_arraydir_v2",
                "source_kind": "dataset_collection",
                "collection_uuid": "collection",
                "collection_pk": 1,
                "dataset_identifier": "collection",
                "source_type": "flow3d",
                "simulation_count": n,
                "members": members,
                "dataset_metadata": {"source_type": "flow3d"},
            }
        ),
        encoding="utf-8",
    )
    return nx, ny, nz, nt


def _run_cfg(tmp_path: Path):
    """Minimal run config: a random-path model asking for the path channels."""
    return SimpleNamespace(
        data_directory_path=str(tmp_path),
        vtk_fields=["Temperature", "Fraction Of Fluid"],
        keys=SimpleNamespace(
            str_fno_input_keys=[
                "V_scan", "P", "x", "y", "z",
                "beam_x", "beam_y", "beam_dist", "path_dose", "path_dose_recent",
            ]
        ),
        moving_ref_frame=False,
        time_as_batch=True,
        time_as_dim=False,
        include_time=True,
        lazy_npz=False,  # deliberately off: the arraydir path must not depend on it
    )


def test_npz_file_collection_is_detected(tmp_path: Path) -> None:
    _stage_aiida_npz_file_collection(tmp_path)
    # The root still advertises arraydir_v2 (that is what broke it) ...
    assert is_arraydir_runtime(tmp_path)
    # ... but the members are whole-NPZ, which is what decides the reader.
    assert is_npz_file_collection(tmp_path)


def test_npz_file_collection_routes_to_the_npz_store(tmp_path: Path) -> None:
    """Before the fix this raised ValueError: Unsupported runtime format ... aiida_npz_file_v1."""
    nx, ny, nz, nt = _stage_aiida_npz_file_collection(tmp_path, n=2)
    store = load_and_preprocess_data(_run_cfg(tmp_path), input_dir=str(tmp_path))

    assert isinstance(store, NpzDirStore)
    assert store.sample_count() == 2
    assert store.spatial_shape() == (nx, ny, nz)
    assert store.time_steps(case_idx=0) == nt


def test_npz_file_collection_supplies_laser_path_channels(tmp_path: Path) -> None:
    """The arraydir store cannot build these; routing to NpzDirStore is what provides them."""
    _stage_aiida_npz_file_collection(tmp_path, n=1)
    store = load_and_preprocess_data(_run_cfg(tmp_path), input_dir=str(tmp_path))

    frame = store.read_case(0, time_idx=1)
    for channel in ("beam_x", "beam_y", "beam_dist", "path_dose", "path_dose_recent"):
        assert channel in frame, f"missing laser-path channel {channel}"
        assert np.isfinite(np.asarray(frame[channel])).all()


def test_plain_arraydir_collection_still_uses_the_arraydir_store(tmp_path: Path) -> None:
    """Guard the existing path: real arraydir members must not be diverted."""
    from test_lazy_runtime import _write_runtime_simulation

    first = _write_runtime_simulation(tmp_path / "members" / "simulation_0000", offset=0.0)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "aiida_arraydir_v2",
                "source_kind": "dataset_collection",
                "collection_uuid": "c",
                "collection_pk": 1,
                "dataset_identifier": "c",
                "source_type": "flow3d",
                "simulation_count": 1,
                "members": [
                    {
                        "relative_path": str(first.relative_to(tmp_path)),
                        "manifest_file": str(first.relative_to(tmp_path) / "manifest.json"),
                        "case_index": 0,
                        "simulation_pk": 1,
                        "simulation_uuid": "sim-0",
                    }
                ],
                "dataset_metadata": {"source_type": "flow3d"},
            }
        ),
        encoding="utf-8",
    )
    assert not is_npz_file_collection(tmp_path)
