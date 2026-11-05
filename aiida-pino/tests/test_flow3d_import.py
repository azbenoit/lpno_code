"""Unit tests for the FLOW-3D run-directory metadata reader and case discovery.

BUG-02: the FLOW-3D ingest path must read per-simulation laser/timing metadata
from the `case_manifest.json` that every `flow3D_scripts` run directory carries,
mirroring `read_openfoam_case_metadata` on the OpenFOAM side. Pure filesystem
tests: no AiiDA profile, no solver, no `lpno` import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiida_pino.utils.flow3d_import import discover_flow3d_cases, read_flow3d_case_metadata


def _manifest(case_name: str = "case_fine_1_keyhole", *, with_material: bool = True) -> dict:
    payload = {
        "case_name": case_name,
        "schema_version": 3,
        "source_config": "configs/rectangular_moving_laser_melt.json",
        "units": "SI",
        "input": {
            "laser": {
                "power_w": 250,  # deliberately int: reader must return floats
                "scan_speed_m_per_s": 1.2,
                "beam_radius_m": 6.5e-5,
                "focal_distance_m": 0.01,
            },
            "simulation": {"finish_time_s": 2.0e-3},
            "geometry": {"length_m": 2e-3, "width_m": 1e-3, "thickness_m": 5e-4, "gas_clearance_m": 2e-4},
            "mesh": {"cell_size_m": 1e-5},
            "material": "ti_imi318_mills",
        },
        "derived": {
            "scan_time_s": 1.25e-3,
            "laser_off_time_s": 1.25e-3,
            "beam_velocity_m_per_s": 1.2,
        },
    }
    if with_material:
        payload["material"] = {
            "name": "ti_imi318_mills",
            "file": "materials/ti_imi318_mills/ti_imi318_mills.json",
            "flow3d_label": "Ti IMI318 (Mills)",
            "properties": {"rho_s": 4420.0, "t_liquidus_k": 1923.0},
            "optical": {"absorptivity": 0.35},
            "table_rows": 42,
        }
        payload["parallel"] = {"cores": 4, "ranks": 1, "threads_per_rank": 4}
    return payload


def _write_case(root: Path, name: str, manifest: dict | None = None) -> Path:
    case_dir = root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest if manifest is not None else _manifest(name)
    (case_dir / "case_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return case_dir


# --- read_flow3d_case_metadata ----------------------------------------------


def test_reader_extracts_laser_scalars_as_floats(tmp_path: Path) -> None:
    case_dir = _write_case(tmp_path / "runs", "case_fine_1_keyhole")

    metadata = read_flow3d_case_metadata(case_dir)

    assert metadata["source_origin_kind"] == "flow3d_run_import"
    assert metadata["source_folder_name"] == "runs"
    assert metadata["source_group"] == "case_fine_1_keyhole"
    assert metadata["source_uri"] == f"file://{case_dir.resolve()}"
    assert metadata["case_name"] == "case_fine_1_keyhole"
    for key, expected in (
        ("V_scan", 1.2),
        ("laser_power", 250.0),
        ("beam_radius", 6.5e-5),
        ("effective_end_time_s", 2.0e-3),
    ):
        assert metadata[key] == pytest.approx(expected)
        assert isinstance(metadata[key], float)
    # FLOW-3D has no beam_blocks concept: always None.
    assert metadata["beam_blocks"] is None
    assert metadata["flow3d_case_manifest"] == _manifest("case_fine_1_keyhole")
    assert metadata["flow3d_material"] == _manifest("case_fine_1_keyhole")["material"]


def test_reader_tolerates_old_v1_manifest_without_material(tmp_path: Path) -> None:
    case_dir = _write_case(tmp_path / "runs", "rectangular_moving_laser_melt_v5", _manifest(
        "rectangular_moving_laser_melt_v5", with_material=False
    ))

    metadata = read_flow3d_case_metadata(case_dir)

    assert metadata["laser_power"] == pytest.approx(250.0)
    assert "flow3d_material" not in metadata
    assert "flow3d_material_source" not in metadata
    assert metadata["beam_blocks"] is None


def test_reader_resolves_material_source_under_materials_root(tmp_path: Path) -> None:
    case_dir = _write_case(tmp_path / "runs", "case_a")
    material_payload = {"name": "ti_imi318_mills", "tables": {"rows": 42}}
    material_file = tmp_path / "repo" / "materials" / "ti_imi318_mills" / "ti_imi318_mills.json"
    material_file.parent.mkdir(parents=True)
    material_file.write_text(json.dumps(material_payload), encoding="utf-8")

    metadata = read_flow3d_case_metadata(case_dir, materials_root=tmp_path / "repo")

    assert metadata["flow3d_material_source"] == material_payload


def test_reader_resolves_flat_legacy_material_file_via_directory_layout(tmp_path: Path) -> None:
    # Manifests rendered before 2026-06-30 record the flat layout
    # (materials/<name>.json); the JSON now lives at materials/<name>/<name>.json.
    manifest = _manifest("case_a")
    manifest["material"]["file"] = "materials/ti_imi318_mills.json"
    case_dir = _write_case(tmp_path / "runs", "case_a", manifest)
    material_payload = {"name": "ti_imi318_mills", "tables": {"rows": 42}}
    material_file = tmp_path / "repo" / "materials" / "ti_imi318_mills" / "ti_imi318_mills.json"
    material_file.parent.mkdir(parents=True)
    material_file.write_text(json.dumps(material_payload), encoding="utf-8")

    metadata = read_flow3d_case_metadata(case_dir, materials_root=tmp_path / "repo")

    assert metadata["flow3d_material_source"] == material_payload


def test_reader_omits_material_source_when_file_missing(tmp_path: Path) -> None:
    case_dir = _write_case(tmp_path / "runs", "case_a")

    metadata = read_flow3d_case_metadata(case_dir, materials_root=tmp_path / "empty-root")

    assert "flow3d_material_source" not in metadata
    assert metadata["flow3d_material"]["name"] == "ti_imi318_mills"


def test_reader_rejects_missing_case_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        read_flow3d_case_metadata(tmp_path / "missing")


def test_reader_rejects_dir_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "not_a_run").mkdir()
    with pytest.raises(ValueError, match="case_manifest.json"):
        read_flow3d_case_metadata(tmp_path / "not_a_run")


# --- discover_flow3d_cases ---------------------------------------------------


def test_discover_returns_root_itself_when_it_is_a_case(tmp_path: Path) -> None:
    case_dir = _write_case(tmp_path, "single_case")

    cases = discover_flow3d_cases(case_dir)

    assert len(cases) == 1
    assert cases[0]["case_dir"] == str(case_dir.resolve())
    assert cases[0]["case_name"] == "single_case"
    assert cases[0]["has_npz"] is False


def test_discover_finds_immediate_subdir_cases_with_npz_flag(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    case_a = _write_case(root, "case_a")
    _write_case(root, "case_b")
    (root / "not_a_case").mkdir()  # no manifest -> ignored
    npz_dir = case_a / "npz"
    npz_dir.mkdir()
    (npz_dir / "case_a.npz").write_bytes(b"npz")

    cases = discover_flow3d_cases(root)

    assert [entry["case_name"] for entry in cases] == ["case_a", "case_b"]
    assert [entry["has_npz"] for entry in cases] == [True, False]


def test_discover_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        discover_flow3d_cases(tmp_path / "missing")


def test_discover_flat_follows_directory_symlinks(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "case.npz").write_bytes(b"npz")
    (canonical / "case.manifest.json").write_text(json.dumps(_manifest("case")), encoding="utf-8")
    view = tmp_path / "view"
    view.mkdir()
    (view / "linked").symlink_to(canonical, target_is_directory=True)

    cases = discover_flow3d_cases(view)

    assert [entry["case_name"] for entry in cases] == ["case"]
    assert cases[0]["layout"] == "flat_npz"
    assert cases[0]["manifest_path"] is not None


def test_discover_flat_symlink_cycle_yields_each_case_once(tmp_path: Path) -> None:
    root = tmp_path / "view"
    root.mkdir()
    (root / "case.npz").write_bytes(b"npz")
    (root / "self").symlink_to(root, target_is_directory=True)  # cycle

    cases = discover_flow3d_cases(root)

    assert [entry["case_name"] for entry in cases] == ["case"]
