"""Zero-copy staging: the pack index written by the export, read by the pino loader.

The full chain under test is the production one: an npz_file simulation stored
through AiiDA, staged with `AIIDA_PINO_PACK_INDEX=1`, and opened by
lpno' own case discovery — byte-identical to the repository copy,
with no payload bytes duplicated on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _npz_file_sim(tmp_path: Path, case_name: str, *, seed: int = 7):
    from aiida_pino.data import PinoSimulationData

    rng = np.random.default_rng(seed)
    nx, ny, nz = 6, 5, 4
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{case_name}.npz"
    np.savez_compressed(
        path,
        x=np.linspace(0.0, 2e-3, nx).astype(np.float32),
        y=np.linspace(-5e-4, 5e-4, ny).astype(np.float32),
        z=np.linspace(-5e-4, 2e-4, nz).astype(np.float32),
        time_s=np.linspace(0.0, 1e-3, 3).astype(np.float64),
        power_w=np.float64(150.0),
        scan_speed_m_per_s=np.float64(1.2),
        case_name=np.str_(case_name),
        Temperature=(300.0 + 1500.0 * rng.random((3, nx, ny, nz))).astype(np.float32),
    )
    node = PinoSimulationData.from_npz_file(
        path,
        metadata={"source_folder_name": "pack_index_test", "source_group": case_name,
                  "laser_power": 150.0, "V_scan": 1.2},
    )
    return node.store()


def _collection(sims, name):
    from aiida_pino.utils import build_collection_metadata_from_simulations
    from aiida_pino.utils.compose import compose_collection, member_inputs

    metadata = build_collection_metadata_from_simulations(
        sims, collection_name=name, dataset_role="base", composition_kind="import"
    )
    return compose_collection(sims, metadata, **member_inputs(sims))[0]


def test_pack_index_staging_writes_kilobytes_not_copies(aiida_profile_clean, tmp_path, monkeypatch):
    sims = [_npz_file_sim(tmp_path / f"src_{i}", f"case_{i}", seed=i) for i in range(3)]
    node = _collection(sims, "pack_index_export")

    monkeypatch.setenv("AIIDA_PINO_PACK_INDEX", "1")
    staged = tmp_path / "staged"
    node.export_runtime_arraydir(staged)

    index = json.loads((staged / "pack_index.json").read_text())
    assert index["format"] == "pino_pack_index_v1"
    assert len(index["entries"]) == 3
    # No payload copies: the flat root holds sidecars and the index only.
    assert list(staged.glob("*.npz")) == []
    # The staged root without members/ is KB, not the dataset's MB.
    flat_bytes = sum(f.stat().st_size for f in staged.iterdir() if f.is_file())
    assert flat_bytes < 100_000
    # Member manifests record how the case was staged.
    member_manifest = json.loads(next(staged.glob("members/*/manifest.json")).read_text())
    assert member_manifest["npz_via_pack_index"] is True


def test_pack_index_on_by_default(aiida_profile_clean, tmp_path, monkeypatch):
    """Zero-copy staging is the DEFAULT (since 2026-08-13): the opt-in
    shipped 2026-08-05 was never rolled out, and the full-copy default filled
    the root disk (362 GB of staging cache). Failure on repack is loud, so
    opt-out is the safe polarity."""
    monkeypatch.delenv("AIIDA_PINO_PACK_INDEX", raising=False)
    sims = [_npz_file_sim(tmp_path / "src", "case_solo")]
    node = _collection(sims, "pack_index_default_on")
    staged = tmp_path / "staged"
    node.export_runtime_arraydir(staged)
    assert (staged / "pack_index.json").exists()


def test_pack_index_explicit_opt_out_still_copies(aiida_profile_clean, tmp_path, monkeypatch):
    monkeypatch.setenv("AIIDA_PINO_PACK_INDEX", "0")
    sims = [_npz_file_sim(tmp_path / "src", "case_solo")]
    node = _collection(sims, "pack_index_opt_out")
    staged = tmp_path / "staged"
    node.export_runtime_arraydir(staged)
    assert not (staged / "pack_index.json").exists()
    assert (staged / "case_solo.npz").is_file()


def test_pino_loader_reads_the_index_byte_identical(aiida_profile_clean, tmp_path, monkeypatch):
    """End to end across the repo boundary, against the repository bytes."""
    pytest.importorskip("lpno")
    from lpno.custom_utils.npz_datasource import _case_files
    from lpno.custom_utils.pack_slice import PackSlicePath, open_case

    sims = [_npz_file_sim(tmp_path / f"src_{i}", f"case_{i}", seed=i) for i in range(2)]
    node = _collection(sims, "pack_index_end_to_end")
    monkeypatch.setenv("AIIDA_PINO_PACK_INDEX", "1")
    staged = tmp_path / "staged"
    node.export_runtime_arraydir(staged)

    cases = _case_files(staged)
    assert len(cases) == 2 and all(isinstance(c, PackSlicePath) for c in cases)
    by_name = {c.stem: c for c in cases}
    for sim in sims:
        case_name = sim.get_simulation_metadata()["case_name"]
        with sim.base.repository.open(sim.PAYLOAD_FILENAME, mode="rb") as handle:
            reference = handle.read()
        reader = open_case(by_name[case_name])
        assert reader.read() == reference, f"{case_name}: sliced bytes differ from the repository"
        reader.close()
