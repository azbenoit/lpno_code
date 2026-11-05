"""Unit tests for the Flow3D ingestion path adapters.

These cover the Flow3D -> VTK -> dataset seam that must become the primary data
source in place of OpenFOAM. They are pure-function / filesystem tests: no AiiDA
daemon, no compute resource, and no license. The heavy `lpno` and
ParaView dependencies are never imported (the Flow3D converter is loaded lazily
inside the adapter and is stubbed here).

Background (`assumptions.md` A4/A6): the legacy datasource encodes the laser
power and scan velocity through the *output filename*, so a real FLOW-3D case
name carries no `P`/`V_scan` and, without the manifest overlay, those training
channels would silently become zero. The `legacy_output_name` tests below pin
that mechanism.
"""

from __future__ import annotations

import pytest
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from aiida_pino.scripts import flow3d_to_vtk_adapter, vtk_to_dataset_adapter

LEGACY_CASE_RE = re.compile(r"^(?P<P>\d{2,3})W_(?P<V>\d{1,4})mps_line_track$")


# --- flow3d_to_vtk_adapter.legacy_output_name -------------------------------


@pytest.mark.parametrize(
    ("case_name", "expected"),
    [
        ("95W_1mps_line_track", "results_V_scan_1p0_laser_power_95"),
        ("95W_1200mps_line_track", "results_V_scan_1p200_laser_power_95"),
        ("100W_8mps_line_track", "results_V_scan_8p0_laser_power_100"),
    ],
)
def test_legacy_output_name_encodes_line_track_params(case_name: str, expected: str) -> None:
    match = LEGACY_CASE_RE.match(case_name)
    assert flow3d_to_vtk_adapter.legacy_output_name(case_name, match, LEGACY_CASE_RE) == expected


@pytest.mark.parametrize(
    "case_name",
    [
        "rectangular_moving_laser_melt",  # FLOW-3D_scripts maintained reference case
        "case_fine_1_keyhole",
        "5W_1mps_line_track",  # single-digit power: regex requires 2-3 digits, no match
    ],
)
def test_legacy_output_name_passes_through_real_flow3d_names(case_name: str) -> None:
    """A case name that does not match the legacy regex is returned unchanged.

    This is the root of BUG-01/BUG-02: the basename then carries no power/velocity
    token, so the PINO datasource fills the `P`/`V_scan` channels with 0.0. When
    the metadata-driven fix lands, this passthrough should be replaced (the params
    should come from the config/manifest, not the filename) and this test updated.
    """
    match = LEGACY_CASE_RE.match(case_name)
    assert flow3d_to_vtk_adapter.legacy_output_name(case_name, match, LEGACY_CASE_RE) == case_name


# --- flow3d_to_vtk_adapter._main case discovery / skip ----------------------


def _prepare_main(monkeypatch, tmp_path: Path, profile: dict, save_stub) -> Path:
    """Stub the heavy converter imports and stage argv/cwd for a `_main` call."""
    source = tmp_path / "flow3d_runs"
    source.mkdir(exist_ok=True)
    monkeypatch.setattr(flow3d_to_vtk_adapter, "resolve_flow3d_script_path", lambda: Path("/unused"))
    monkeypatch.setattr(flow3d_to_vtk_adapter, "load_flow3d_functions", lambda _path: (save_stub, LEGACY_CASE_RE))
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flow3d_to_vtk_adapter",
            "--source-path",
            str(source),
            "--output-dir",
            str(tmp_path / "vtk_output"),
            "--profile-file",
            str(profile_file),
        ],
    )
    return source


def test_main_rejects_profile_missing_required_keys(monkeypatch, tmp_path: Path) -> None:
    """BUG-21: a malformed profile must raise a clear error, not a raw KeyError."""
    _prepare_main(monkeypatch, tmp_path, {"case_regex": r"^case_.*$"}, lambda **kwargs: None)

    with pytest.raises(ValueError, match="num_timesteps") as excinfo:
        flow3d_to_vtk_adapter._main()
    assert "output_basename_mode" in str(excinfo.value)


def test_main_continues_after_case_failure_and_records_it(monkeypatch, tmp_path: Path) -> None:
    """BUG-15: one bad case must not abort the batch; the failure lands in the summary."""

    def failing_save(**kwargs):
        if kwargs["case_name"] == "case_a":
            raise RuntimeError("boom")

    profile = {"case_regex": r"^case_.*$", "num_timesteps": 1, "output_basename_mode": "case_name"}
    source = _prepare_main(monkeypatch, tmp_path, profile, failing_save)
    for name in ("case_a", "case_b"):
        (source / name).mkdir()
        (source / name / f"flsgrf1.{name}").write_text("flsgrf", encoding="utf-8")

    flow3d_to_vtk_adapter._main()

    summary = json.loads((tmp_path / "flow3d_summary.json").read_text(encoding="utf-8"))
    assert [c["case_name"] for c in summary["converted_cases"]] == ["case_b"]
    assert summary["failed_cases"] == [{"case_name": "case_a", "error": "boom"}]


def test_main_writes_error_summary_before_reraising(monkeypatch, tmp_path: Path) -> None:
    """BUG-15: a fatal error must still leave a `status: error` summary behind."""
    profile = {"case_regex": r"^case_.*$", "num_timesteps": 1, "output_basename_mode": "case_name"}
    source = _prepare_main(monkeypatch, tmp_path, profile, lambda **kwargs: None)
    source.rmdir()  # iterating the missing source dir raises

    with pytest.raises(OSError):
        flow3d_to_vtk_adapter._main()

    summary = json.loads((tmp_path / "flow3d_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["error_message"]


def test_main_zero_conversions_writes_error_summary_and_raises(monkeypatch, tmp_path: Path) -> None:
    """BUG-06: converting zero cases must not look like success."""
    profile = {"case_regex": r"^case_.*$", "num_timesteps": 1, "output_basename_mode": "case_name"}
    source = _prepare_main(monkeypatch, tmp_path, profile, lambda **kwargs: None)
    (source / "other").mkdir()  # does not match case_regex -> skipped

    with pytest.raises(ValueError, match="zero"):
        flow3d_to_vtk_adapter._main()

    summary = json.loads((tmp_path / "flow3d_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "error"
    assert summary["converted_cases"] == []
    assert "other" in summary["error_message"]


def test_main_converts_matching_cases_and_records_skips(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "flow3d_runs"
    source.mkdir()
    (source / "case_a").mkdir()
    (source / "case_a" / "flsgrf1.case_a").write_text("flsgrf", encoding="utf-8")
    (source / "case_b").mkdir()  # matches regex but missing flsgrf1.* -> skipped
    (source / "other").mkdir()  # does not match case_regex -> skipped

    calls: list[dict] = []

    def recording_save(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(flow3d_to_vtk_adapter, "resolve_flow3d_script_path", lambda: Path("/unused"))
    monkeypatch.setattr(flow3d_to_vtk_adapter, "load_flow3d_functions", lambda _path: (recording_save, LEGACY_CASE_RE))

    profile = {"case_regex": r"^case_.*$", "num_timesteps": 1, "output_basename_mode": "case_name"}
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(profile), encoding="utf-8")

    output_dir = tmp_path / "vtk_output"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["flow3d_to_vtk_adapter", "--source-path", str(source), "--output-dir", str(output_dir), "--profile-file", str(profile_file)],
    )

    flow3d_to_vtk_adapter._main()

    assert len(calls) == 1
    assert calls[0]["case_name"] == "case_a"
    assert calls[0]["output_filename"] == "case_a"

    summary = json.loads((tmp_path / "flow3d_summary.json").read_text(encoding="utf-8"))
    assert [c["case_name"] for c in summary["converted_cases"]] == ["case_a"]
    assert sorted(summary["skipped_cases"]) == ["case_b", "other"]


# --- vtk_to_dataset_adapter._run_summary_member_overrides -------------------


def test_run_summary_overrides_renames_and_types_scalars() -> None:
    base = {
        "laserbeamfoam": {
            "run_summary": {
                "simulation_profile": {"V_scan": 0.8, "laser_power": 100, "beam_radius": 5e-5, "beam_blocks": 3},
                "derived_parameters": {"effective_end_time": 1.2e-3, "deltaT": 5e-6},
            }
        }
    }
    overrides = vtk_to_dataset_adapter._run_summary_member_overrides(base)
    assert overrides["V_scan"] == pytest.approx(0.8)
    assert overrides["laser_power"] == pytest.approx(100.0)
    assert overrides["beam_radius"] == pytest.approx(5e-5)
    assert overrides["beam_blocks"] == 3
    assert isinstance(overrides["beam_blocks"], int)
    assert overrides["effective_end_time_s"] == pytest.approx(1.2e-3)  # renamed
    assert overrides["delta_t_s"] == pytest.approx(5e-6)  # renamed from deltaT


def test_run_summary_overrides_empty_when_no_run_summary() -> None:
    # A FLOW-3D import supplies no `laserbeamfoam.run_summary` (BUG-02): no spurious keys.
    assert vtk_to_dataset_adapter._run_summary_member_overrides({"source_type": "flow3d"}) == {}


def test_run_summary_overrides_omits_missing_scalars() -> None:
    base = {"laserbeamfoam": {"run_summary": {"simulation_profile": {"V_scan": 0.5}}}}
    overrides = vtk_to_dataset_adapter._run_summary_member_overrides(base)
    assert overrides == {"V_scan": 0.5}


# --- BUG-18: run-summary overrides are single-group-only --------------------


_LASERBEAMFOAM_RUN_SUMMARY = {
    "run_summary": {"simulation_profile": {"V_scan": 0.8, "laser_power": 100.0}}
}


def test_split_single_group_applies_run_summary_overrides() -> None:
    arrays = {"Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "run", "files": []}]
    base = _base_collection_metadata(source_type="openfoam", laserbeamfoam=_LASERBEAMFOAM_RUN_SUMMARY)

    members, collection = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    assert members[0]["simulation_metadata"]["laser_power"] == pytest.approx(100.0)
    assert members[0]["simulation_metadata"]["V_scan"] == pytest.approx(0.8)
    assert "run_summary_overrides_applied" not in collection


def test_split_multi_group_skips_run_summary_overrides() -> None:
    # BUG-18: one laserbeamfoam run_summary cannot describe several groups, so
    # its laser scalars must not be stamped onto every member.
    arrays = {"Temperature": np.ones((2, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "sim_0", "files": []}, {"group": "sim_1", "files": []}]
    base = _base_collection_metadata(source_type="openfoam", laserbeamfoam=_LASERBEAMFOAM_RUN_SUMMARY)

    members, collection = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    for member in members:
        assert "laser_power" not in member["simulation_metadata"]
        assert "V_scan" not in member["simulation_metadata"]
    assert collection["run_summary_overrides_applied"] is False


def test_split_multi_group_flow3d_manifests_still_apply_when_run_summary_skipped() -> None:
    arrays = {"Temperature": np.ones((2, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "case_a", "files": []}, {"group": "case_b", "files": []}]
    base = _flow3d_base_metadata(
        {
            "case_a": _flow3d_case_entry("case_a", laser_power=100.0, v_scan=0.5),
            "case_b": _flow3d_case_entry("case_b", laser_power=200.0, v_scan=0.8),
        },
        laserbeamfoam={"run_summary": {"simulation_profile": {"V_scan": 9.9, "laser_power": 999.0}}},
    )

    members, collection = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    # Per-group manifest values win; the shared run-summary scalars are skipped.
    assert members[0]["simulation_metadata"]["laser_power"] == pytest.approx(100.0)
    assert members[1]["simulation_metadata"]["laser_power"] == pytest.approx(200.0)
    assert collection["run_summary_overrides_applied"] is False


# --- vtk_to_dataset_adapter.split_collection_payload ------------------------


def _base_collection_metadata(**extra) -> dict:
    base = {
        "source_type": "flow3d",
        "source_folder_name": "run",
        "source_uri": "/data/run",
        "grid_params": {"nx": 2, "ny": 2, "nz": 2},
        "requested_fields": ["Temperature"],
    }
    base.update(extra)
    return base


def test_split_collection_payload_slices_batch_axis_per_group() -> None:
    arrays = {"Temperature": np.arange(2 * 1 * 2 * 2 * 2, dtype=np.float64).reshape(2, 1, 2, 2, 2)}
    grouped = [{"group": "sim_0", "files": []}, {"group": "sim_1", "files": []}]
    # BUG-01: a flow3d import without laser metadata now fails loudly, so this
    # slicing test supplies per-group case manifests to satisfy the guard.
    base = _flow3d_base_metadata(
        {
            "sim_0": _flow3d_case_entry("sim_0", laser_power=100.0, v_scan=0.5),
            "sim_1": _flow3d_case_entry("sim_1", laser_power=200.0, v_scan=0.8),
        }
    )

    members, collection = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    assert len(members) == 2
    for i, member in enumerate(members):
        payload = member["payload_arrays"]["Temperature"]
        np.testing.assert_array_equal(payload, arrays["Temperature"][i])
        assert payload.dtype == np.float32  # batch axis dropped, cast to float32
        assert member["simulation_metadata"]["source_type"] == "flow3d"
        assert member["simulation_metadata"]["grid_params"] == {"nx": 2, "ny": 2, "nz": 2}
    assert collection["simulation_count"] == 2
    assert collection["composition_kind"] == "import"
    assert collection["source_origin_kind"] == "vtk_import"


def test_split_collection_payload_rejects_batch_axis_mismatch() -> None:
    arrays = {"Temperature": np.zeros((3, 1, 2, 2, 2), dtype=np.float32)}  # 3 != 2 groups
    grouped = [{"group": "sim_0", "files": []}, {"group": "sim_1", "files": []}]
    with pytest.raises(ValueError, match="batch axis"):
        vtk_to_dataset_adapter.split_collection_payload(
            arrays, grouped_metadata=grouped, base_collection_metadata=_base_collection_metadata()
        )


# --- vtk_to_dataset_adapter._flow3d_member_overrides (BUG-02) ---------------


def _flow3d_case_entry(case_name: str, *, laser_power: float, v_scan: float) -> dict:
    return {
        "source_origin_kind": "flow3d_run_import",
        "case_name": case_name,
        "V_scan": v_scan,
        "laser_power": laser_power,
        "beam_radius": 6.5e-5,
        "beam_blocks": None,
        "effective_end_time_s": 2.0e-3,
        "flow3d_case_manifest": {"case_name": case_name, "schema_version": 3},
        "flow3d_material": {"name": "ti_imi318_mills"},
    }


def _flow3d_base_metadata(case_manifests: dict, **extra) -> dict:
    return _base_collection_metadata(flow3d={"case_manifests": case_manifests}, **extra)


def test_flow3d_member_overrides_exact_group_match() -> None:
    base = _flow3d_base_metadata({"case_a": _flow3d_case_entry("case_a", laser_power=100.0, v_scan=0.5)})

    overrides = vtk_to_dataset_adapter._flow3d_member_overrides(base, "case_a")

    assert overrides["laser_power"] == pytest.approx(100.0)
    assert overrides["V_scan"] == pytest.approx(0.5)
    assert overrides["beam_radius"] == pytest.approx(6.5e-5)
    assert overrides["beam_blocks"] is None
    assert overrides["effective_end_time_s"] == pytest.approx(2.0e-3)
    assert overrides["flow3d_case_manifest"]["case_name"] == "case_a"
    assert overrides["flow3d_material"] == {"name": "ti_imi318_mills"}


def test_flow3d_member_overrides_unique_prefix_match() -> None:
    # Legacy basename mode / time grouping can rename the VTK group; a unique
    # prefix relation with the case dir name must still resolve the manifest.
    base = _flow3d_base_metadata({"case_a": _flow3d_case_entry("case_a", laser_power=100.0, v_scan=0.5)})

    overrides = vtk_to_dataset_adapter._flow3d_member_overrides(base, "case_a_0000")

    assert overrides["laser_power"] == pytest.approx(100.0)


def test_flow3d_member_overrides_ambiguous_prefix_returns_empty() -> None:
    base = _flow3d_base_metadata(
        {
            "case_a": _flow3d_case_entry("case_a", laser_power=100.0, v_scan=0.5),
            "case_ab": _flow3d_case_entry("case_ab", laser_power=200.0, v_scan=0.8),
        }
    )
    assert vtk_to_dataset_adapter._flow3d_member_overrides(base, "case_") == {}


def test_flow3d_member_overrides_empty_without_manifests() -> None:
    assert vtk_to_dataset_adapter._flow3d_member_overrides(_base_collection_metadata(), "case_a") == {}


def test_split_collection_payload_applies_per_group_flow3d_manifests() -> None:
    arrays = {"Temperature": np.ones((2, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "case_a", "files": []}, {"group": "case_b", "files": []}]
    base = _flow3d_base_metadata(
        {
            "case_a": _flow3d_case_entry("case_a", laser_power=100.0, v_scan=0.5),
            "case_b": _flow3d_case_entry("case_b", laser_power=200.0, v_scan=0.8),
        }
    )

    members, collection = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    # Each member gets ITS OWN scalars, not one shared set (cf. BUG-18).
    assert members[0]["simulation_metadata"]["laser_power"] == pytest.approx(100.0)
    assert members[0]["simulation_metadata"]["V_scan"] == pytest.approx(0.5)
    assert members[1]["simulation_metadata"]["laser_power"] == pytest.approx(200.0)
    assert members[1]["simulation_metadata"]["V_scan"] == pytest.approx(0.8)
    for member in members:
        assert member["simulation_metadata"]["flow3d_case_manifest"]["schema_version"] == 3
        assert member["simulation_metadata"]["beam_blocks"] is None
    assert collection["flow3d_manifest_groups"] == {"matched": ["case_a", "case_b"], "missed": []}


def test_split_collection_payload_records_missed_manifest_groups() -> None:
    arrays = {"Temperature": np.ones((2, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "case_a", "files": []}, {"group": "unrelated", "files": []}]
    base = _flow3d_base_metadata(
        {"case_a": _flow3d_case_entry("case_a", laser_power=100.0, v_scan=0.5)},
        source_type="openfoam",
    )

    _, collection = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    assert collection["flow3d_manifest_groups"] == {"matched": ["case_a"], "missed": ["unrelated"]}


# --- BUG-01: FLOW-3D `P`/`V_scan` grids come from manifests, not filenames ---


def test_split_flow3d_zero_power_grid_without_manifest_raises() -> None:
    arrays = {
        "Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float64),
        "P": np.zeros((1, 1, 2, 2, 2), dtype=np.float64),
        "V_scan": np.zeros((1, 1, 2, 2, 2), dtype=np.float64),
    }
    grouped = [{"group": "rectangular_moving_laser_melt", "files": []}]

    with pytest.raises(ValueError, match="rectangular_moving_laser_melt") as excinfo:
        vtk_to_dataset_adapter.split_collection_payload(
            arrays, grouped_metadata=grouped, base_collection_metadata=_base_collection_metadata()
        )
    assert "case_manifest.json" in str(excinfo.value)


def test_split_flow3d_missing_metadata_without_manifest_raises() -> None:
    # No P/V_scan grids at all and no manifest: the laser scalars resolve to None.
    arrays = {"Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "case_fine_1_keyhole", "files": []}]

    with pytest.raises(ValueError, match="case_fine_1_keyhole"):
        vtk_to_dataset_adapter.split_collection_payload(
            arrays, grouped_metadata=grouped, base_collection_metadata=_base_collection_metadata()
        )


def test_split_flow3d_manifest_overrides_rewrite_channel_grids() -> None:
    arrays = {
        "Temperature": np.ones((2, 1, 2, 2, 2), dtype=np.float64),
        "P": np.zeros((2, 1, 2, 2, 2), dtype=np.float64),  # filename-derived zeros: no longer trusted
        "V_scan": np.zeros((2, 1, 2, 2, 2), dtype=np.float64),
    }
    grouped = [{"group": "case_a", "files": []}, {"group": "case_b", "files": []}]
    base = _flow3d_base_metadata(
        {
            "case_a": _flow3d_case_entry("case_a", laser_power=100.0, v_scan=0.5),
            "case_b": _flow3d_case_entry("case_b", laser_power=200.0, v_scan=0.8),
        }
    )

    members, _ = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    for member, power, v_scan in zip(members, (100.0, 200.0), (0.5, 0.8)):
        np.testing.assert_allclose(member["payload_arrays"]["P"], np.full((1, 2, 2, 2), power, dtype=np.float32))
        np.testing.assert_allclose(
            member["payload_arrays"]["V_scan"], np.full((1, 2, 2, 2), v_scan, dtype=np.float32)
        )
        assert member["payload_arrays"]["P"].dtype == np.float32
        assert member["simulation_metadata"]["laser_power"] == pytest.approx(power)
        assert member["simulation_metadata"]["V_scan"] == pytest.approx(v_scan)


def test_split_flow3d_manifest_overrides_create_missing_channel_grids() -> None:
    arrays = {"Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "case_a", "files": []}]
    base = _flow3d_base_metadata({"case_a": _flow3d_case_entry("case_a", laser_power=100.0, v_scan=0.5)})

    members, _ = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    np.testing.assert_allclose(members[0]["payload_arrays"]["P"], np.full((1, 2, 2, 2), 100.0, dtype=np.float32))
    np.testing.assert_allclose(
        members[0]["payload_arrays"]["V_scan"], np.full((1, 2, 2, 2), 0.5, dtype=np.float32)
    )


def test_split_openfoam_zero_power_grid_is_not_blocked() -> None:
    # The guard is flow3d-only: OpenFOAM steady cases may legitimately carry zero grids.
    arrays = {
        "Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float64),
        "P": np.zeros((1, 1, 2, 2, 2), dtype=np.float64),
    }
    grouped = [{"group": "steady_case", "files": []}]

    members, _ = vtk_to_dataset_adapter.split_collection_payload(
        arrays,
        grouped_metadata=grouped,
        base_collection_metadata=_base_collection_metadata(source_type="openfoam"),
    )

    np.testing.assert_array_equal(members[0]["payload_arrays"]["P"], np.zeros((1, 2, 2, 2), dtype=np.float32))


def test_split_collection_payload_rejects_empty_group_list() -> None:
    with pytest.raises(ValueError, match="No grouped simulations"):
        vtk_to_dataset_adapter.split_collection_payload(
            {"Temperature": np.zeros((0, 1, 2, 2, 2), dtype=np.float32)},
            grouped_metadata=[],
            base_collection_metadata=_base_collection_metadata(),
        )


def test_stationary_case_may_have_zero_scan_speed(tmp_path):
    """A dwell case legitimately scans at 0 m/s and must still ingest.

    Regression for 2026-07-28: dwell_ti64_130w solved and exported, then the
    ingest refused it because the V_scan guard demanded a strictly positive
    speed. The guard exists to catch a *dropped* speed silently becoming zero,
    so it now yields only when the manifest declares the run stationary.
    """
    from aiida_pino.scripts import raw_flow3d_to_dataset_adapter as ad

    meta = {"laser_power": 130.0, "V_scan": 0.0}
    ad._require_positive_laser_metadata(meta, "dwell", "manifest", stationary=True)  # no raise

    with pytest.raises(ValueError):
        ad._require_positive_laser_metadata(meta, "moving", "manifest", stationary=False)

    # power must still be positive even for a dwell
    with pytest.raises(ValueError):
        ad._require_positive_laser_metadata(
            {"laser_power": 0.0, "V_scan": 0.0}, "dwell", "manifest", stationary=True
        )


def test_is_stationary_reads_the_manifest_flag(tmp_path):
    from aiida_pino.scripts import raw_flow3d_to_dataset_adapter as ad
    import json as _json

    assert ad._is_stationary(tmp_path) is False           # no manifest -> strict
    (tmp_path / "case_manifest.json").write_text(_json.dumps({"derived": {"laser_stationary": True}}))
    assert ad._is_stationary(tmp_path) is True
    (tmp_path / "case_manifest.json").write_text(_json.dumps({"derived": {}}))
    assert ad._is_stationary(tmp_path) is False


def test_stationary_case_with_absent_scan_speed_normalizes_to_zero():
    """A dwell manifest has no scan speed at all, not a zero.

    Regression for 2026-07-28: rs0p1_dwell_130w solved, exported, and still
    failed ingest -- the stationary branch accepted 0.0 but the value was None,
    because a dwell config omits laser.scan_speed_m_per_s entirely.
    """
    from aiida_pino.scripts import raw_flow3d_to_dataset_adapter as ad

    meta = {"laser_power": 130.0, "V_scan": None}
    ad._require_positive_laser_metadata(meta, "dwell", "manifest", stationary=True)
    assert meta["V_scan"] == 0.0          # normalized in place, not left as a hole

    with pytest.raises(ValueError):       # a moving case still may not lose its speed
        ad._require_positive_laser_metadata(
            {"laser_power": 130.0, "V_scan": None}, "moving", "manifest", stationary=False
        )


# --- the third copy of the laser guard ------------------------------------
# 2026-08-21 duplication audit: the "V_scan = 0 is real for a dwell weld" rule
# was hand-written twice (data/pino_simulation.py and
# scripts/raw_flow3d_to_dataset_adapter.py, converged only after two separate
# fixes a week apart) and a third site here never got either. A spot run
# therefore ingests through the NPZ path and is refused by the legacy VTK path,
# with a message telling you to supply a manifest that is already present.


def _stationary_case_entry(case_name: str, *, laser_power: float) -> dict:
    entry = _flow3d_case_entry(case_name, laser_power=laser_power, v_scan=0.0)
    entry["flow3d_case_manifest"]["derived"] = {"laser_stationary": True}
    return entry


def test_split_flow3d_stationary_dwell_case_is_accepted() -> None:
    """A spot weld's all-zero V_scan grid is the measurement, not a dropped value."""
    arrays = {
        "Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float64),
        "P": np.zeros((1, 1, 2, 2, 2), dtype=np.float64),
        "V_scan": np.zeros((1, 1, 2, 2, 2), dtype=np.float64),
    }
    grouped = [{"group": "spot_a", "files": []}]
    base = _flow3d_base_metadata({"spot_a": _stationary_case_entry("spot_a", laser_power=100.0)})

    members, _ = vtk_to_dataset_adapter.split_collection_payload(
        arrays, grouped_metadata=grouped, base_collection_metadata=base
    )

    np.testing.assert_allclose(
        members[0]["payload_arrays"]["V_scan"], np.zeros((1, 2, 2, 2), dtype=np.float32)
    )
    np.testing.assert_allclose(
        members[0]["payload_arrays"]["P"], np.full((1, 2, 2, 2), 100.0, dtype=np.float32)
    )


def test_split_flow3d_non_stationary_zero_v_scan_still_raises() -> None:
    """Accepting the dwell case must not blanket-accept a dropped scan speed."""
    arrays = {"Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "moving_a", "files": []}]
    base = _flow3d_base_metadata({"moving_a": _flow3d_case_entry("moving_a", laser_power=100.0, v_scan=0.0)})

    with pytest.raises(ValueError, match="moving_a"):
        vtk_to_dataset_adapter.split_collection_payload(
            arrays, grouped_metadata=grouped, base_collection_metadata=base
        )


def test_split_flow3d_nan_laser_metadata_is_rejected() -> None:
    """The old guard skipped the scalar check entirely whenever a manifest existed."""
    arrays = {"Temperature": np.ones((1, 1, 2, 2, 2), dtype=np.float64)}
    grouped = [{"group": "case_nan", "files": []}]
    base = _flow3d_base_metadata(
        {"case_nan": _flow3d_case_entry("case_nan", laser_power=float("nan"), v_scan=0.5)}
    )

    with pytest.raises(ValueError, match="case_nan"):
        vtk_to_dataset_adapter.split_collection_payload(
            arrays, grouped_metadata=grouped, base_collection_metadata=base
        )
