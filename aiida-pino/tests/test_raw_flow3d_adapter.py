"""Unit tests for the single-step FLOW-3D NPZ importer adapter.

Pure filesystem/numpy tests: no AiiDA profile, no solver, no `lpno`
import, no ParaView. Run directories mirror the `flow3D_scripts` contract:
`case_manifest.json` plus a compact `npz/<case>.npz` export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aiida_pino.scripts import raw_flow3d_to_dataset_adapter as adapter
from aiida_pino.utils.profiles import validate_flow3d_npz_profile

GRID = (6, 5, 4)
NUM_FRAMES = 6


def _manifest(case_name: str, *, power_w: float = 250.0, scan_speed: float = 1.2) -> dict:
    return {
        "case_name": case_name,
        "schema_version": 3,
        "source_config": "configs/rectangular_moving_laser_melt.json",
        "units": "SI",
        "input": {
            "laser": {"power_w": power_w, "scan_speed_m_per_s": scan_speed, "beam_radius_m": 6.5e-5},
            "simulation": {"finish_time_s": 2.0e-3},
            "geometry": {"length_m": 2e-3, "width_m": 1e-3, "thickness_m": 5e-4, "gas_clearance_m": 2e-4},
            "mesh": {"cell_size_m": 1e-5},
        },
        "derived": {"scan_time_s": 1.25e-3},
        "material": {
            "name": "ti_imi318_mills",
            "file": "materials/ti_imi318_mills/ti_imi318_mills.json",
            "flow3d_label": "Ti IMI318 (Mills)",
            "properties": {"rho_s": 4420.0},
            "optical": {"absorptivity": 0.35},
            "table_rows": 42,
        },
    }


def _write_case(
    root: Path,
    name: str,
    *,
    manifest: dict | None = None,
    with_npz: bool = True,
    npz_power_w: float = float("nan"),
    npz_scan_speed: float = float("nan"),
    temperature: np.ndarray | None = None,
    grid: tuple[int, int, int] | None = None,
) -> Path:
    case_dir = root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest if manifest is not None else _manifest(name)
    (case_dir / "case_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    if with_npz:
        nx, ny, nz = grid or GRID
        rng = np.random.default_rng(7)
        if temperature is None:
            temperature = (300.0 + 1500.0 * rng.random((NUM_FRAMES, nx, ny, nz))).astype(np.float32)
        (case_dir / "npz").mkdir(exist_ok=True)
        np.savez(
            case_dir / "npz" / f"{name}.npz",
            x=np.linspace(0.0, 2e-3, nx).astype(np.float32),
            y=np.linspace(-5e-4, 5e-4, ny).astype(np.float32),
            z=np.linspace(-5e-4, 2e-4, nz).astype(np.float32),
            time_s=np.linspace(0.0, 1e-3, NUM_FRAMES).astype(np.float64),
            power_w=np.float64(npz_power_w),
            scan_speed_m_per_s=np.float64(npz_scan_speed),
            case_name=np.str_(name),
            **{
                "Temperature": temperature,
                "Melt Region": rng.random((NUM_FRAMES, nx, ny, nz)).astype(np.float32),
                "Velocity": rng.standard_normal((NUM_FRAMES, nx, ny, nz, 3)).astype(np.float32),
            },
        )
    return case_dir


def _run_main(monkeypatch, tmp_path: Path, source: Path, profile: dict | None = None) -> Path:
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
    return tmp_path


# --- payload assembly --------------------------------------------------------


def test_member_is_verbatim_npz_file_with_canonical_shapes_and_metadata(monkeypatch, tmp_path: Path) -> None:
    """Run-dir members store the compact NPZ verbatim — never broadcast arrays.

    Until 2026-07-27 this path materialized the canonical `(T, C, nx, ny, nz)`
    arrays into a workdir payload that the parser stored uncompressed via
    `from_arrays` (~20x the compact size). The member is now
    `npz_file` mode pointing at the source file's realpath, and no payload
    moves through the workdir.
    """
    source = tmp_path / "runs"
    _write_case(source, "case_a")

    _run_main(monkeypatch, tmp_path, source)

    nx, ny, nz = GRID
    assert not (tmp_path / "dataset_output" / "payloads" / "simulation_0000.npz").exists(), (
        "full-frame run-dir imports must not copy arrays through the workdir"
    )
    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    member = manifest["members"][0]
    metadata = member["simulation_metadata"]
    assert member["storage_mode"] == "npz_file"
    assert Path(member["source_npz_realpath"]) == (source / "case_a" / "npz" / "case_a.npz").resolve()
    # The canonical (T, C, nx, ny, nz) shapes are still advertised — they are
    # synthesized at read time from the compact payload, not stored.
    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["array_shapes"]["Velocity"] == [NUM_FRAMES, 3, nx, ny, nz]
    assert summary["array_shapes"]["Temperature"] == [NUM_FRAMES, 1, nx, ny, nz]
    assert summary["array_shapes"]["x"] == [NUM_FRAMES, 1, nx, ny, nz]
    # Dedup identity is pinned to the legacy from_arrays recipe: hashing the
    # member metadata itself must reproduce the recorded fingerprint, so
    # re-imports of already-ingested runs keep resolving onto the same sims.
    from aiida_pino.data.pino_simulation import build_simulation_fingerprint

    assert metadata["simulation_fingerprint"] == build_simulation_fingerprint(metadata)
    assert member["group"] == "case_a"
    assert metadata["source_origin_kind"] == "flow3d_npz_import"
    assert metadata["requested_fields"] == ["Temperature", "Melt Region", "Velocity"]
    assert metadata["grid_params"]["resolution"] == [nx, ny, nz]
    # One relative_path per selected frame so the fingerprint tracks subsampling.
    assert metadata["relative_paths"] == [f"npz/case_a.npz#frame={i}" for i in range(NUM_FRAMES)]
    assert metadata["laser_power"] == pytest.approx(250.0)
    assert metadata["V_scan"] == pytest.approx(1.2)
    assert metadata["beam_radius"] == pytest.approx(6.5e-5)
    assert metadata["beam_blocks"] is None
    assert len(metadata["time_values_s"]) == NUM_FRAMES
    assert metadata["effective_end_time_s"] == pytest.approx(1e-3)
    assert metadata["delta_t_s"] == pytest.approx(1e-3 / (NUM_FRAMES - 1))
    assert metadata["flow3d_case_manifest"]["input"]["laser"]["power_w"] == 250.0
    assert metadata["flow3d_material"]["name"] == "ti_imi318_mills"
    assert metadata["sample_metadata"][0] == {
        "relative_path": "npz/case_a.npz",
        "frame_index": 0,
        "params": {"time_s": 0.0},
    }

    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["grouping_mode"] == "raw_flow3d_npz"
    assert summary["imported_cases"] == ["case_a"]
    # NaN provenance scalars carry no information and produce no warning.
    assert summary["provenance_warnings"] == []


def test_profile_annotations_flow_into_member_metadata(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_a")

    _run_main(
        monkeypatch,
        tmp_path,
        source,
        {
            "description": "reproduced from flow3D_scripts melt config",
            "batch_keyword": "lhs_melt_2026",
            "usable_for_training": False,
        },
    )

    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["members"][0]["simulation_metadata"]
    assert metadata["description"] == "reproduced from flow3D_scripts melt config"
    assert metadata["batch_keyword"] == "lhs_melt_2026"
    assert metadata["usable_for_training"] is False


def test_member_annotations_default_when_profile_omits_them(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_a")

    _run_main(monkeypatch, tmp_path, source)

    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["members"][0]["simulation_metadata"]
    # The adapter records the profile value verbatim (None when unset); the node
    # layer normalizes None -> "" / True at from_arrays time.
    assert metadata["description"] is None
    assert metadata["batch_keyword"] is None
    assert metadata["usable_for_training"] is True


def test_npz_fields_profile_selects_subset(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_a")

    _run_main(monkeypatch, tmp_path, source, {"npz_fields": ["Temperature"]})

    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    member = manifest["members"][0]
    # A field subset keeps the verbatim store; `requested_fields` is what
    # filters the served arrays at read time.
    assert member["storage_mode"] == "npz_file" and "source_npz_realpath" in member
    assert member["simulation_metadata"]["requested_fields"] == ["Temperature"]


# --- time subsampling --------------------------------------------------------


def test_select_frame_indices_keeps_first_and_last() -> None:
    assert adapter.select_frame_indices(6, 3) == [0, 2, 5]
    assert adapter.select_frame_indices(6, None) == [0, 1, 2, 3, 4, 5]
    assert adapter.select_frame_indices(4, 10) == [0, 1, 2, 3]
    assert adapter.select_frame_indices(101, 2) == [0, 100]


def test_time_target_frames_subsamples_including_endpoints(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_a")

    _run_main(monkeypatch, tmp_path, source, {"time_target_frames": 3})

    nx, ny, nz = GRID
    with np.load(tmp_path / "dataset_output" / "payloads" / "simulation_0000.npz") as payload:
        # The sliced payload keeps the *compact* schema: 1-D axes, fields at
        # (T, nx, ny, nz[, C]) and their original dtype — no broadcast arrays.
        assert payload["x"].shape == (nx,)
        assert payload["time_s"].shape == (3,)
        assert payload["Temperature"].shape == (3, nx, ny, nz)
        assert payload["Velocity"].shape == (3, nx, ny, nz, 3)
    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    member = manifest["members"][0]
    assert member["storage_mode"] == "npz_file" and "payload_file" in member
    metadata = member["simulation_metadata"]
    times = np.linspace(0.0, 1e-3, NUM_FRAMES)
    assert metadata["time_values_s"] == pytest.approx([times[0], times[2], times[5]])
    assert [entry["frame_index"] for entry in metadata["sample_metadata"]] == [0, 2, 5]
    assert metadata["effective_end_time_s"] == pytest.approx(times[-1])
    assert metadata["delta_t_s"] is None  # 0, 2, 5 is not uniform


def test_frame_subsampling_changes_simulation_fingerprint(monkeypatch, tmp_path: Path) -> None:
    """Distinct `time_target_frames` of one NPZ must not dedupe onto one identity.

    Regression: `build_simulation_fingerprint` hashes `relative_paths`, so a
    full-frame import and a subsampled import of the same case must differ, or
    the subsampled collection would silently reuse the stale full-frame members
    (and `supersede_equivalent_dataset_collections` would mark the honest one
    superseded).
    """
    from aiida_pino.data.pino_simulation import build_simulation_fingerprint

    source = tmp_path / "runs"
    _write_case(source, "case_a")

    _run_main(monkeypatch, tmp_path, source)  # full frames
    full = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    full_meta = full["members"][0]["simulation_metadata"]

    _run_main(monkeypatch, tmp_path, source, {"time_target_frames": 3})  # subsampled
    sub = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    sub_meta = sub["members"][0]["simulation_metadata"]

    assert len(full_meta["relative_paths"]) == NUM_FRAMES
    assert sub_meta["relative_paths"] == [f"npz/case_a.npz#frame={i}" for i in (0, 2, 5)]
    assert full_meta["relative_paths"] != sub_meta["relative_paths"]
    assert build_simulation_fingerprint(full_meta) != build_simulation_fingerprint(sub_meta)


# --- guards ------------------------------------------------------------------


def test_zero_power_guard_raises_loudly(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_a", manifest=_manifest("case_a", power_w=0.0))

    with pytest.raises(ValueError, match="laser_power"):
        _run_main(monkeypatch, tmp_path, source)

    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert "case_a" in summary["error_message"]


def test_non_finite_laser_value_guard_raises(monkeypatch, tmp_path: Path) -> None:
    """A NaN/inf manifest laser value must fail loudly, not fill P/V_scan with NaN.

    Regression: the guard used `float(value) <= 0.0`, which is False for NaN, so
    a NaN laser power slipped through into all-NaN conditioning grids.
    """
    source = tmp_path / "runs"
    _write_case(source, "case_nan_power", manifest=_manifest("case_nan_power", power_w=float("nan")))

    with pytest.raises(ValueError, match="laser_power"):
        _run_main(monkeypatch, tmp_path, source)

    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert "case_nan_power" in summary["error_message"]


def test_non_finite_field_guard_names_field_and_case(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    nx, ny, nz = GRID
    temperature = np.full((NUM_FRAMES, nx, ny, nz), 300.0, dtype=np.float32)
    temperature[2, 1, 1, 1] = np.nan
    _write_case(source, "case_nan", temperature=temperature)

    with pytest.raises(ValueError, match="`Temperature`.*non-finite|case_nan.*Temperature"):
        _run_main(monkeypatch, tmp_path, source)

    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert "Temperature" in summary["error_message"]
    assert "case_nan" in summary["error_message"]


def test_provenance_mismatch_records_warning_and_manifest_wins(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_a", npz_power_w=300.0, npz_scan_speed=1.2)

    _run_main(monkeypatch, tmp_path, source)

    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert len(summary["provenance_warnings"]) == 1
    assert "power_w" in summary["provenance_warnings"][0]
    # Manifest wins: the P channel is synthesized at read time from this value.
    manifest = json.loads((tmp_path / "dataset_output" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["members"][0]["simulation_metadata"]["laser_power"] == pytest.approx(250.0)


# --- case discovery / skip ---------------------------------------------------


def test_manifest_only_case_is_skipped_and_recorded(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_a")
    _write_case(source, "case_manifest_only", with_npz=False)

    _run_main(monkeypatch, tmp_path, source)

    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["imported_cases"] == ["case_a"]
    assert summary["skipped_cases"] == ["case_manifest_only"]


def test_zero_ingestable_cases_writes_error_summary_and_raises(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_manifest_only", with_npz=False)

    with pytest.raises(ValueError, match="npz"):
        _run_main(monkeypatch, tmp_path, source)

    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["skipped_cases"] == ["case_manifest_only"]


# --- profile validation ------------------------------------------------------


def test_profile_validation_rejects_unknown_keys_and_bad_types() -> None:
    validate_flow3d_npz_profile({})
    validate_flow3d_npz_profile({"npz_fields": ["Temperature"], "time_target_frames": 40})
    with pytest.raises(ValueError, match="unknown keys"):
        validate_flow3d_npz_profile({"vtk_fields": ["Temperature"]})
    with pytest.raises(ValueError, match="npz_fields"):
        validate_flow3d_npz_profile({"npz_fields": "Temperature"})
    with pytest.raises(ValueError, match="time_target_frames"):
        validate_flow3d_npz_profile({"time_target_frames": 1})


def test_profile_validation_accepts_and_type_checks_annotations() -> None:
    validate_flow3d_npz_profile({"description": "x", "batch_keyword": "b", "usable_for_training": True})
    with pytest.raises(ValueError, match="description"):
        validate_flow3d_npz_profile({"description": 5})
    with pytest.raises(ValueError, match="batch_keyword"):
        validate_flow3d_npz_profile({"batch_keyword": ["b"]})
    with pytest.raises(ValueError, match="usable_for_training"):
        validate_flow3d_npz_profile({"usable_for_training": "yes"})


def test_main_rejects_unknown_profile_key_with_error_summary(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "runs"
    _write_case(source, "case_a")

    with pytest.raises(ValueError, match="unknown keys"):
        _run_main(monkeypatch, tmp_path, source, {"vtk_fields": ["Temperature"]})

    summary = json.loads((tmp_path / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"


def test_scalar_metadata_is_not_validated_as_a_spatial_field(tmp_path) -> None:
    """A new scalar in the NPZ must not be mistaken for a physical field.

    On 2026-08-02 the flow3D_scripts exporter gained `frame_stride` (recording time
    subsampling). Ingest validated every non-axis array against
    (T, nx, ny, nz[, C]) using a hardcoded name list, so four live AiiDA batches
    failed with "frame_stride has shape [], expected (T, nx, ny, nz[, C])" purely
    because the name was not on the list. Rank-0 arrays are metadata by shape, so
    the next such scalar is additive rather than breaking.
    """
    import numpy as np

    from aiida_pino.utils.flow3d_npz import is_metadata_array, read_npz_headers

    path = tmp_path / "case.npz"
    np.savez_compressed(
        path,
        x=np.arange(3.0), y=np.arange(4.0), z=np.arange(5.0),
        time_s=np.arange(2.0),
        power_w=np.asarray(90.0),
        frame_stride=np.asarray(4, dtype=np.int64),
        a_future_scalar=np.asarray(1.5),
        Temperature=np.zeros((2, 3, 4, 5), dtype=np.float32),
    )
    headers = read_npz_headers(str(path))
    fields = [n for n in headers if not is_metadata_array(n, tuple(headers[n][0]))]

    assert fields == ["Temperature"]
    assert is_metadata_array("frame_stride", ())
    assert is_metadata_array("a_future_scalar", ())          # unnamed scalar, by shape
    assert not is_metadata_array("Temperature", (2, 3, 4, 5))


# --- mixed grids -------------------------------------------------------------


def test_mixed_grid_import_completes_and_reports_nonuniform(monkeypatch, tmp_path: Path) -> None:
    """Mixed extents import instead of exiting 410.

    `_accumulate` used to raise "incompatible grids" the moment a second case
    disagreed with the first, killing multi-case back-fills of exactly the
    dataset `ti64_geom40` is: one resolution, many domain extents. The refusal
    is now a provenance warning, and the manifest carries what the collection
    layer needs to report `uniform_grid: False` honestly.
    """
    import json as _json

    source = tmp_path / "mixed"
    _write_case(source, "geom_small")
    _write_case(source, "geom_large", grid=(9, 7, 4))

    _run_main(monkeypatch, tmp_path, source)

    manifest = _json.loads((tmp_path / "dataset_output" / "manifest.json").read_text())
    names = sorted(m["group"] for m in manifest["members"])
    assert names == ["geom_large", "geom_small"], "both cases must import"
    summary = _json.loads((tmp_path / "dataset_summary.json").read_text())
    warnings = " ".join(summary.get("provenance_warnings") or [])
    assert "mixed grids" in warnings, "the divergence must be recorded, not silent"
    # `resolved_geometry` keeps the FIRST case's grid, matching what
    # `grid_params` means everywhere else.
    assert summary["resolved_geometry"] is not None
