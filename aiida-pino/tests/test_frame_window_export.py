"""A moving-frame export reads only the frames it keeps, and writes the same bits as a full read.

2026-10-08, reproduction from the deposit: an evaluation's export canonicalized every frame of
every field before keeping the last `moving_frame_steps`, about 8.5 GB per worker for the
longest case, and was OOM-killed in a 32 GB job. `derived_member_worker` now reads the window
alone; `frame_window: False` (or AIIDA_PINO_EXPORT_FULL_FRAMES=1) restores the full read.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.utils.runtime_export import derived_member_worker, member_frame_window, payload_locator
from test_runtime_export import BOTH_PATHS_PROFILES, _beam_track_task_fields, _npz_file_sim


@pytest.mark.parametrize("profile_name", sorted(BOTH_PATHS_PROFILES))
def test_the_window_gives_the_bits_of_the_full_read(aiida_profile_clean, tmp_path, profile_name):
    from aiida_pino.data.pino_dataset_collection import merged_case_manifest

    profile = BOTH_PATHS_PROFILES[profile_name]
    sim = _npz_file_sim(tmp_path / "src", "window_case", frames=9)
    time_values = sim.get_time_values_s()
    task = {
        "index": 0,
        "case_name": "window_case",
        "laser_power": float(sim.base.attributes.get("laser_power")),
        "v_scan": float(sim.base.attributes.get("V_scan")),
        "requested_fields": None,
        "profile": dict(profile),
        "time_values_s": time_values.tolist(),
        "locator": payload_locator(sim),
        **_beam_track_task_fields(sim, profile),
    }
    if profile.get("derived_fields"):
        task["case_manifest"] = merged_case_manifest(sim)
    window = member_frame_window(task, profile)
    assert window is not None and window[0] > 0, f"no frames cut ({window}); the test would be vacuous"
    assert member_frame_window({**task, "frame_window": False}, profile) is None

    full = derived_member_worker({**task, "frame_window": False, "target_dir": str(tmp_path / "full")})
    cut = derived_member_worker({**task, "target_dir": str(tmp_path / "cut")})

    assert cut["array_shapes"] == full["array_shapes"]
    files = {e["original_name"]: e["filename"] for e in full["array_entries"]}
    assert files == {e["original_name"]: e["filename"] for e in cut["array_entries"]}
    for name, filename in files.items():   # coordinates included
        np.testing.assert_array_equal(np.load(tmp_path / "cut" / filename), np.load(tmp_path / "full" / filename),
                                      err_msg=f"{name} differs between the windowed and the full read")
    np.testing.assert_array_equal(np.load(tmp_path / "cut" / "time_values_s.npy"),
                                  np.load(tmp_path / "full" / "time_values_s.npy"))


def test_the_environment_switch_turns_the_window_off(monkeypatch):
    task, profile = {"moving_frame_stop": 8}, {"moving_ref_frame": True, "moving_frame_steps": 3}
    assert member_frame_window(task, profile) == (5, 8)
    monkeypatch.setenv("AIIDA_PINO_EXPORT_FULL_FRAMES", "1")
    assert member_frame_window(task, profile) is None
    monkeypatch.delenv("AIIDA_PINO_EXPORT_FULL_FRAMES")
    assert member_frame_window({"moving_frame_stop": 2}, profile) is None      # stop before a full window
    assert member_frame_window(task, {"moving_ref_frame": False}) is None      # lab frame: every frame kept
