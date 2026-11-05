"""Flat-NPZ ingest must not crash on the run-directory-only `case_dir` local.

`_flat_case_member` referenced `case_dir` — a name bound only in the
run-directory path — so *every* flat-layout ingest died with
`NameError: name 'case_dir' is not defined` (hit 2026-07-29 ingesting a flat
directory of steady-frame NPZs; the workchain surfaced it only as exit 410).

The stationary check must also work for the flat layout, where the manifest is
a `<case>.manifest.json` sidecar rather than `<dir>/case_manifest.json`.
"""

from __future__ import annotations

import json

from aiida_pino.scripts import raw_flow3d_to_dataset_adapter as adapter


def test_is_stationary_reads_a_sidecar_manifest_file(tmp_path):
    sidecar = tmp_path / "case_a.manifest.json"
    sidecar.write_text(json.dumps({"derived": {"laser_stationary": True}}), encoding="utf-8")
    assert adapter._is_stationary(sidecar) is True

    moving = tmp_path / "case_b.manifest.json"
    moving.write_text(json.dumps({"derived": {"laser_stationary": False}}), encoding="utf-8")
    assert adapter._is_stationary(moving) is False


def test_is_stationary_still_reads_a_run_directory(tmp_path):
    run_dir = tmp_path / "alsi_c000"
    run_dir.mkdir()
    (run_dir / "case_manifest.json").write_text(
        json.dumps({"derived": {"laser_stationary": True}}), encoding="utf-8"
    )
    assert adapter._is_stationary(run_dir) is True


def test_is_stationary_defaults_false_when_absent(tmp_path):
    # missing/broken manifest keeps the strict V_scan rule
    assert adapter._is_stationary(tmp_path / "nope.manifest.json") is False
    assert adapter._is_stationary(tmp_path) is False


def test_flat_case_member_has_no_case_dir_reference():
    """The flat path must not reference the run-directory-only local."""
    import inspect

    src = inspect.getsource(adapter._flat_case_member)
    assert "_is_stationary(case_dir)" not in src, (
        "flat-case path references `case_dir`, which is unbound there — "
        "this is the NameError that broke every flat-NPZ ingest"
    )


def test_both_payload_paths_forward_the_stationary_flag():
    """The adapter resolves `laser_stationary`, then must forward it.

    It set the flag on `case_metadata` for `_require_positive_laser_metadata`
    and then built `simulation_metadata` field by field without it, so the value
    never reached the node. Every SS304L spot weld stored
    `laser_stationary=False` beside `V_scan=0.0` -- a moving laser at zero speed,
    which is precisely what the guard exists to reject. Ingest passed because the
    guard read the local, and the node was wrong because storage read the payload.
    """
    import inspect

    for fn in (adapter._case_member_payload, adapter._flat_case_member):
        src = inspect.getsource(fn)
        assert '"laser_stationary"' in src, (
            f"{fn.__name__} builds simulation_metadata without laser_stationary; "
            "the flag it resolved for the guard never reaches the stored node"
        )


def test_multi_beam_manifest_summarises_its_beams():
    """A multi-beam case has no top-level power, and must not read as None.

    Independent-beam configs (`mb40_*`) put power, path and turn-off on each beam,
    so `laser.power_w` and `laser.scan_speed_m_per_s` are absent and both fields
    parsed to None. `_require_positive_laser_metadata` then rejects the case, so
    none of the 23 solved multi-beam runs could be ingested at all.

    The summary matches what the exporter already wrote into the NPZ
    (flow3D_scripts `export_provenance_scalars`): total delivered power, mean
    speed. Verified against mb40_000, whose NPZ holds 244.871 W / 0.3231 m/s.
    """
    from aiida_pino.utils.flow3d_import import parse_flow3d_case_manifest

    manifest = {
        "case_name": "unit_mb",
        "input": {
            "laser": {
                "beam_radius_m": 6.0e-5,
                "beams": [
                    {"power_w": 100.0, "scan_speed_m_per_s": 0.2},
                    {"power_w": 144.871, "scan_speed_m_per_s": 0.4462},
                ],
            },
            "simulation": {"finish_time_s": 3.0e-3},
        },
    }
    m = parse_flow3d_case_manifest(manifest, case_name_fallback="unit_mb")
    assert abs(m["laser_power"] - 244.871) < 1e-6
    assert abs(m["V_scan"] - 0.3231) < 1e-6
    assert m["beam_count"] == 2


def test_single_beam_manifest_is_untouched():
    from aiida_pino.utils.flow3d_import import parse_flow3d_case_manifest

    manifest = {
        "case_name": "unit_sb",
        "input": {"laser": {"power_w": 195.0, "scan_speed_m_per_s": 0.7,
                            "beam_radius_m": 6.125e-5},
                  "simulation": {"finish_time_s": 1.0e-3}},
    }
    m = parse_flow3d_case_manifest(manifest, case_name_fallback="unit_sb")
    assert abs(m["laser_power"] - 195.0) < 1e-9
    assert abs(m["V_scan"] - 0.7) < 1e-9
    assert "beam_count" not in m
