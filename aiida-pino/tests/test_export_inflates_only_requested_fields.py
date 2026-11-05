"""A derived export inflates only the fields the consumer reads.

`canonicalize_flow3d_case_arrays` decompresses exactly `selected_fields`, but the export task
carried the SIMULATION's import-time list -- all 17 arrays -- and ignored the `runtime_fields`
it was given, so a steady export inflated Velocity and eleven scalars to keep Temperature and
Fraction Of Fluid (self-improve, 2026-09-14; ~85 s per member where ~20 s is the two fields).
The memory estimate is sized on the same trimmed set, so the pool widens with it.
"""

from __future__ import annotations

from types import SimpleNamespace

from aiida_pino.data.pino_dataset_collection import trimmed_fields
from aiida_pino.utils.export_pool import per_member_export_peak_bytes

GB = 1e9
SHAPES = {"Temperature": [798, 1, 90, 60, 55], "Fraction Of Fluid": [798, 1, 90, 60, 55],
          "Velocity": [798, 3, 90, 60, 55], **{f"S{i}": [798, 1, 90, 60, 55] for i in range(12)}}


def _sim(names=tuple(SHAPES)):
    attrs = {"requested_fields": list(SHAPES)}
    return SimpleNamespace(
        list_original_array_names=lambda: list(names),
        get_array_shapes=lambda: dict(SHAPES),
        base=SimpleNamespace(attributes=SimpleNamespace(get=lambda k, d=None: attrs.get(k, d))),
    )


def test_runtime_fields_pass_through_as_names_for_the_worker_to_intersect():
    got = trimmed_fields(_sim(), ["Temperature", "Fraction Of Fluid", "V_scan", "P"])
    assert got == ["Temperature", "Fraction Of Fluid", "V_scan", "P"]


def test_no_runtime_fields_keeps_the_import_time_list():
    assert trimmed_fields(_sim(), None) == list(SHAPES)


def test_the_worker_inflates_only_the_requested_archive_members(aiida_profile_clean, tmp_path):
    """The training's real list names V_scan and P, which are synthesised, not stored: the
    worker must intersect with the archive and not raise (every member of a warm failed on
    `NPZ has no field(s) P, V_scan` when the filter ran against the canonical names)."""
    from aiida_pino.utils import flow3d_npz
    from aiida_pino.utils.runtime_export import derived_member_worker, payload_locator
    from test_runtime_export import _beam_track_task_fields, _npz_file_sim

    sim = _npz_file_sim(tmp_path / "src", "trim_case")
    seen = {}
    real = flow3d_npz.canonicalize_flow3d_case_arrays

    def spy(archive, **kw):
        seen["selected"] = kw.get("selected_fields")
        return real(archive, **kw)

    original = flow3d_npz.canonicalize_flow3d_case_arrays  # the worker imports it at call time
    flow3d_npz.canonicalize_flow3d_case_arrays = spy
    try:
        profile = {"stride": [1, 1, 1], "time_as_dim": False, "include_time": True, "time_avg_window": 1,
                   "moving_ref_frame": False, "super_res_factor": 1, "moving_frame_steps": 1, "crop_z_top": 0}
        tv = sim.get_time_values_s()
        task = {"index": 0, "target_dir": str(tmp_path / "member"), "case_name": "trim_case",
                "laser_power": 150.0, "v_scan": 1.2,
                "requested_fields": ["Temperature", "Fraction Of Fluid", "V_scan", "P"],
                "profile": profile, "time_values_s": tv.tolist(), "locator": payload_locator(sim),
                **_beam_track_task_fields(sim, profile)}
        result = derived_member_worker(task)
    finally:
        flow3d_npz.canonicalize_flow3d_case_arrays = original
    assert seen["selected"] == ["Temperature"]  # the only requested name the archive holds
    assert "Temperature" in result["array_shapes"] and "Melt Region" not in result["array_shapes"]
    assert "P" in result["array_shapes"] and "V_scan" in result["array_shapes"]  # synthesised, still served


def test_the_estimate_is_sized_on_the_trimmed_fields():
    steady = {"moving_ref_frame": True, "time_avg_window": 30, "derived_fields": ["conduction_field"]}
    full = per_member_export_peak_bytes([_sim()], steady)
    two = per_member_export_peak_bytes([_sim()], steady, fields=["Temperature", "Fraction Of Fluid", "V_scan"])
    # Two of fifteen equal-sized fields, but the conduction-field build (four times one
    # field, float64 temporaries over every frame) is charged either way: measured 6.75 GB
    # against 19.2 GB on one dataset member (2026-09-14), so about a third, not a fifteenth.
    assert two < full / 3, f"{two / GB:.1f} GB should be well under {full / GB:.1f} GB"
    assert two >= (2 + 4) * 798 * 90 * 60 * 55 * 4  # the two fields plus the build term
    # Names that match nothing size the whole member, as the worker then inflates it.
    assert per_member_export_peak_bytes([_sim()], steady, fields=["nothing"]) == full
