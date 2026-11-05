"""Tests for flat-folder FLOW-3D NPZ ingest (per-case NPZ + manifest sidecars).

Covers: recursive discovery with skips/excludes, adapter emission of
``npz_file`` members without payload copies, laser-scalar fallback and guards,
symlink realpath normalization, and one end-to-end CalcJob run through the
parser (`from_npz_file` storage plus fingerprint dedupe across a symlink view).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aiida_pino.scripts import raw_flow3d_to_dataset_adapter as adapter
from aiida_pino.utils.flow3d_import import discover_flow3d_cases
from aiida_pino.utils.profiles import validate_flow3d_npz_profile

GRID = (5, 4, 3)
NUM_FRAMES = 4


def _sidecar_manifest(case_name: str, *, power_w: float, scan_speed: float) -> dict:
    return {
        "case_name": case_name,
        "schema_version": 3,
        "units": {"length": "m", "time": "s"},
        "input": {
            "laser": {"power_w": power_w, "scan_speed_m_per_s": scan_speed, "beam_radius_m": 5e-5},
            "simulation": {"finish_time_s": 1e-3},
        },
        "derived": {"scan_time_s": 1e-3},
        "material": {"name": "ti_imi318_mills", "file": "materials/ti_imi318_mills/ti_imi318_mills.json"},
    }


def _write_flat_case(
    directory: Path,
    case_name: str,
    *,
    power_w: float = 200.0,
    scan_speed: float = 1.0,
    with_manifest: bool = True,
    npz_power_w: float | None = None,
    npz_scan_speed: float | None = None,
    seed: int = 3,
) -> Path:
    nx, ny, nz = GRID
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case_name}.npz"
    np.savez_compressed(
        path,
        x=np.linspace(0.0, 2e-3, nx).astype(np.float32),
        y=np.linspace(-5e-4, 5e-4, ny).astype(np.float32),
        z=np.linspace(-5e-4, 2e-4, nz).astype(np.float32),
        time_s=np.linspace(0.0, 1e-3, NUM_FRAMES).astype(np.float64),
        power_w=np.float64(power_w if npz_power_w is None else npz_power_w),
        scan_speed_m_per_s=np.float64(scan_speed if npz_scan_speed is None else npz_scan_speed),
        case_name=np.str_(case_name),
        **{
            "Temperature": (300.0 + rng.random((NUM_FRAMES, nx, ny, nz))).astype(np.float32),
            "Melt Region": rng.random((NUM_FRAMES, nx, ny, nz)).astype(np.float32),
            "Velocity": rng.standard_normal((NUM_FRAMES, nx, ny, nz, 3)).astype(np.float32),
        },
    )
    if with_manifest:
        sidecar = directory / f"{case_name}.manifest.json"
        sidecar.write_text(
            json.dumps(_sidecar_manifest(case_name, power_w=power_w, scan_speed=scan_speed)),
            encoding="utf-8",
        )
    return path


def _run_main(monkeypatch, tmp_path: Path, source: Path, profile: dict | None = None) -> dict:
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(profile or {}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "raw_flow3d_to_dataset_adapter.py",
            "--raw-source-path",
            str(source),
            "--output-dir",
            "dataset_output",
            "--profile-file",
            str(profile_file),
        ],
    )
    adapter._main()
    return json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))


# --- discovery ---------------------------------------------------------------


def test_discover_flat_cases_recursive_with_groups_and_skips(tmp_path: Path) -> None:
    root = tmp_path / "flatset"
    _write_flat_case(root, "case_root")
    _write_flat_case(root / "train", "case_train", with_manifest=False)
    _write_flat_case(root / ".aiida_pino_cache", "cached")
    _write_flat_case(root / "npz_tools", "tooling")
    (root / "var_1_time_dim_base.npz").write_bytes(b"not a case")

    cases = discover_flow3d_cases(root)

    assert [entry["case_name"] for entry in cases] == ["case_root", "case_train"]
    by_name = {entry["case_name"]: entry for entry in cases}
    assert by_name["case_root"]["layout"] == "flat_npz"
    assert by_name["case_root"]["source_group"] == ""
    assert by_name["case_root"]["manifest_path"] is not None
    assert by_name["case_train"]["source_group"] == "train"
    assert by_name["case_train"]["manifest_path"] is None
    assert all(entry["has_npz"] for entry in cases)


def test_discover_exclude_globs_filters_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "flatset"
    _write_flat_case(root, "keep_me")
    _write_flat_case(root / "smoke", "smoke_case")
    _write_flat_case(root, "abx_twin")

    cases = discover_flow3d_cases(root, exclude_globs=["smoke/*", "*_twin.npz"])
    assert [entry["case_name"] for entry in cases] == ["keep_me"]


def test_discover_mixed_run_dir_and_flat_does_not_double_count(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    run_dir = root / "run_case"
    run_dir.mkdir(parents=True)
    (run_dir / "case_manifest.json").write_text(
        json.dumps(_sidecar_manifest("run_case", power_w=100.0, scan_speed=0.5)), encoding="utf-8"
    )
    (run_dir / "npz").mkdir()
    _write_flat_case(run_dir / "npz", "run_case", with_manifest=False)
    _write_flat_case(root, "flat_case")

    cases = discover_flow3d_cases(root)
    assert [(entry["case_name"], entry.get("layout", "run_dir")) for entry in cases] == [
        ("run_case", "run_dir"),
        ("flat_case", "flat_npz"),
    ]
    assert cases[0]["has_npz"] is True


def test_profile_validation_accepts_and_rejects_exclude_globs() -> None:
    validate_flow3d_npz_profile({"exclude_globs": ["smoke/*", "*_twin.npz"]})
    with pytest.raises(ValueError, match="exclude_globs"):
        validate_flow3d_npz_profile({"exclude_globs": "smoke/*"})
    with pytest.raises(ValueError, match="exclude_globs"):
        validate_flow3d_npz_profile({"exclude_globs": [""]})


# --- adapter -----------------------------------------------------------------


def test_flat_main_emits_npz_file_members_without_payload_copies(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "flatset"
    _write_flat_case(source, "case_a", power_w=180.0, scan_speed=0.9, seed=1)
    _write_flat_case(source / "train", "case_b", power_w=220.0, scan_speed=1.1, seed=2)

    summary = _run_main(monkeypatch, tmp_path, source)

    assert summary["status"] == "ok"
    assert summary["num_simulations"] == 2
    assert sorted(summary["flat_npz_cases"]) == ["case_a", "case_b"]
    # No array payloads went through the workdir.
    assert not list((tmp_path / "dataset_output" / "payloads").iterdir())

    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    members = {member["group"]: member for member in manifest["members"]}
    member = members["case_a"]
    assert member["storage_mode"] == "npz_file"
    assert "payload_file" not in member
    assert member["source_npz_realpath"] == str((source / "case_a.npz").resolve())
    metadata = member["simulation_metadata"]
    assert metadata["case_name"] == "case_a"
    assert metadata["source_group"] == ""
    assert metadata["laser_power"] == pytest.approx(180.0)
    assert metadata["V_scan"] == pytest.approx(0.9)
    assert metadata["source_origin_kind"] == "flow3d_npz_file_import"
    assert metadata["flow3d_case_manifest"]["input"]["laser"]["power_w"] == pytest.approx(180.0)
    assert members["case_b"]["simulation_metadata"]["source_group"] == "train"


def test_flat_main_falls_back_to_npz_provenance_scalars(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "flatset"
    _write_flat_case(source, "case_bare", power_w=140.0, scan_speed=0.7, with_manifest=False)

    summary = _run_main(monkeypatch, tmp_path, source)
    assert summary["status"] == "ok"
    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["members"][0]["simulation_metadata"]
    assert metadata["laser_power"] == pytest.approx(140.0)
    assert metadata["V_scan"] == pytest.approx(0.7)
    assert "flow3d_case_manifest" not in metadata


def test_flat_main_nan_scalars_without_manifest_error(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "flatset"
    _write_flat_case(source, "case_nan", npz_power_w=float("nan"), with_manifest=False)

    with pytest.raises(ValueError, match="laser_power"):
        _run_main(monkeypatch, tmp_path, source)
    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert "case_nan" in summary["error_message"]


def test_flat_main_rejects_time_target_frames_subsampling(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "flatset"
    _write_flat_case(source, "case_frames")

    with pytest.raises(ValueError, match="time_target_frames"):
        _run_main(monkeypatch, tmp_path, source, {"time_target_frames": 2})

    # A target >= the stored frame count is a no-op and passes.
    summary = _run_main(monkeypatch, tmp_path, source, {"time_target_frames": NUM_FRAMES})
    assert summary["status"] == "ok"


def test_flat_main_resolves_symlinks_to_realpath(monkeypatch, tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    path = _write_flat_case(canonical, "case_link")
    view = tmp_path / "view" / "train"
    view.mkdir(parents=True)
    (view / "case_link.npz").symlink_to(path)
    sidecar = canonical / "case_link.manifest.json"
    (view / "case_link.manifest.json").symlink_to(sidecar)

    summary = _run_main(monkeypatch, tmp_path, tmp_path / "view")
    assert summary["status"] == "ok"
    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    member = manifest["members"][0]
    assert member["source_npz_realpath"] == str(path.resolve())
    assert member["simulation_metadata"]["source_group"] == "train"


# --- end-to-end CalcJob + parser ---------------------------------------------


def test_flat_ingest_integration_and_symlink_dedupe(local_code_factory, tmp_path: Path) -> None:
    """Full CalcJob run: flat folder -> npz_file simulations -> dedupe via realpath."""
    from aiida import orm
    from aiida.engine import run_get_node

    from aiida_pino.calculations.raw_flow3d_to_dataset import RawFlow3DToDatasetCalculation
    from aiida_pino.data import PinoDatasetCollectionData

    source = tmp_path / "flatset"
    cases = {"case_a": (150.0, 0.8), "case_b": (250.0, 1.2)}
    for index, (name, (power, speed)) in enumerate(cases.items()):
        _write_flat_case(source, name, power_w=power, scan_speed=speed, seed=index)

    python = local_code_factory(sys.executable, label="real-python-flat-npz")
    results, node = run_get_node(
        RawFlow3DToDatasetCalculation,
        code=python,
        raw_source_path=orm.Str(str(source)),
        dataset_profile=orm.Dict(dict={}),
    )
    assert node.is_finished_ok, results.get("dataset_summary") and results["dataset_summary"].get_dict()
    dataset = results["dataset"]
    assert isinstance(dataset, PinoDatasetCollectionData)
    assert dataset.get_simulation_count() == 2

    member_uuids = set()
    for simulation in dataset.iter_member_simulations():
        member_uuids.add(simulation.uuid)
        assert simulation.get_storage_mode() == "npz_file"
        metadata = simulation.get_simulation_metadata()
        power, speed = cases[metadata["case_name"]]
        values = np.unique(simulation.get_array_by_original_name("P"))
        assert values.tolist() == [pytest.approx(power)]
        assert simulation.get_array_by_original_name("Velocity").shape == (NUM_FRAMES, 3, *GRID)
        # Verbatim payload.
        with simulation.base.repository.open("payload.npz", mode="rb") as handle:
            assert handle.read() == (source / f"{metadata['case_name']}.npz").read_bytes()

    # A symlink view (e.g. a curated split folder) dedupes onto the same
    # simulation nodes: realpath-normalized fingerprints match.
    view = tmp_path / "view"
    view.mkdir()
    for name in cases:
        (view / f"{name}.npz").symlink_to(source / f"{name}.npz")
        (view / f"{name}.manifest.json").symlink_to(source / f"{name}.manifest.json")
    second_results, second_node = run_get_node(
        RawFlow3DToDatasetCalculation,
        code=python,
        raw_source_path=orm.Str(str(view)),
        dataset_profile=orm.Dict(dict={}),
    )
    assert second_node.is_finished_ok
    second_uuids = {sim.uuid for sim in second_results["dataset"].iter_member_simulations()}
    assert second_uuids == member_uuids
