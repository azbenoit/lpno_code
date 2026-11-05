"""Cross-repo contract: runtime exports must be readable by lpno.

The 2026-07-20 RIGNO run found that aiida-pino can emit a runtime layout the
consumer never declared support for (npz_file member manifests tagged
`aiida_npz_file_v1` under an `aiida_arraydir_v2` root), which broke every
AiiDA-launched training on the rebuilt DB. These tests pin the pair: the
exported layout must be recognized by the consumer's actual routing functions.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def _write_case_npz(path, name):
    nx, ny, nz, nt = 6, 5, 4, 3
    rng = np.random.default_rng(7)
    np.savez_compressed(
        path,
        x=np.linspace(0.0, 5e-4, nx).astype(np.float32),
        y=np.linspace(-2e-4, 2e-4, ny).astype(np.float32),
        z=np.linspace(-3e-4, 1e-4, nz).astype(np.float32),
        time_s=np.linspace(0.0, 1e-3, nt).astype(np.float64),
        power_w=np.float64(150.0),
        scan_speed_m_per_s=np.float64(0.8),
        case_name=np.str_(name),
        **{
            "Temperature": (300 + rng.random((nt, nx, ny, nz))).astype(np.float32),
            "Fraction Of Fluid": rng.random((nt, nx, ny, nz)).astype(np.float32),
        },
    )


def test_npz_file_collection_export_is_recognized_by_consumer(tmp_path, aiida_profile_clean, monkeypatch):
    # Pins the FLAT per-case staging path explicitly. 9d6d8bd made
    # zero-copy pack-index staging the default, and these assertions are
    # about the flat path (real per-case files, byte-identical payloads,
    # duplicate-name refusal), so they must name the mode instead of
    # inheriting whichever one happens to be default. The new default has
    # its own coverage in tests/test_pack_index_export.py.
    monkeypatch.setenv("AIIDA_PINO_PACK_INDEX", "0")
    lazy_runtime = pytest.importorskip("lpno.models.pipeline.lazy_runtime")
    npz_lazy_store = pytest.importorskip("lpno.models.pipeline.npz_lazy_store")

    from aiida_pino.data import PinoSimulationData
    from aiida_pino.utils import build_collection_metadata_from_simulations, get_or_create_dataset_collection

    sims = []
    for i in range(2):
        npz = tmp_path / f"case_{i}.npz"
        _write_case_npz(npz, f"case_{i}")
        node = PinoSimulationData.from_npz_file(
            npz, metadata={"source_folder_name": "contract", "source_group": ""}
        )
        node.store()
        sims.append(node)
    meta = build_collection_metadata_from_simulations(sims, collection_name="contract")
    collection, _ = get_or_create_dataset_collection(sims, metadata=meta)

    export = tmp_path / "runtime"
    collection.export_runtime_arraydir(export)

    # The consumer's routing (preprocessing_datasets.load path) must claim this
    # layout end to end: arraydir root, npz_file member routing, and the
    # consolidated-NPZ signature NpzDirStore actually reads.
    assert lazy_runtime.is_arraydir_runtime(export), "consumer no longer recognizes the exported root manifest"
    assert lazy_runtime.is_npz_file_collection(export), (
        "consumer no longer routes npz_file member manifests to NpzDirStore — "
        "this is the exact break that blocked all AiiDA trainings on 2026-07-20"
    )
    assert npz_lazy_store.is_consolidated_npz(export), (
        "exported case NPZs no longer match the consolidated-NPZ signature NpzDirStore reads"
    )
