"""A derived member stages as ONE compressed NPZ in the flat root, not loose `.npy`.

2026-08-26, decided because local disk space is scarce. `derived_member_worker` wrote every array with
`np.save` -- per-array UNCOMPRESSED object storage, the pattern that once cost this
project ~754 GiB in the object store, reappearing in the runtime cache where no rule
watched. It refused a 208-member steady build at 887 GB and a 60-member test at 225 GB
on a box with 186 GB free.

Two properties, and the second is the one that is easy to get wrong silently:

- COMPRESSED: measured 58x on the gascrop val split (0.05 GB/member against 2.9 GB).
  Material channels make it starker: each is `np.full(shape, value)`, one scalar over the
  whole grid, twelve of them.
- FLAT ROOT: this is what decides WHICH READER runs. A dir of `<case>.npz` satisfies
  `is_consolidated_npz`, so `NpzDirStore` serves it as archives. Left under `members/` it
  stays an arraydir, read by `ArrayDir*Store` with `mmap_mode="r"` -- and mmap is IGNORED
  on a zip, so compressing in place would not error, it would quietly load whole arrays
  into RAM.
"""

from __future__ import annotations

import numpy as np


def _fake_task(tmp_path, *, nt=4, nx=6, ny=5, nz=8, profile=None):
    import io

    root = tmp_path / "collection"
    member = root / "members" / "simulation_0000"
    member.mkdir(parents=True)

    buf = io.BytesIO()
    rng = np.random.default_rng(0)
    np.savez(
        buf,
        Temperature=rng.random((nt, nx, ny, nz), dtype=np.float32),
        x=np.linspace(0, 1, nx, dtype=np.float32),
        y=np.linspace(0, 1, ny, dtype=np.float32),
        z=np.linspace(0, 2, nz, dtype=np.float32),
        time_s=np.linspace(0, 1e-3, nt, dtype=np.float32),
        power_w=np.float32(100.0),
        scan_speed_m_per_s=np.float32(0.5),
        case_name=np.asarray("case_a"),
    )
    staged = tmp_path / "payload.npz"
    staged.write_bytes(buf.getvalue())

    return {
        "index": 0,
        "target_dir": str(member),
        "staged_path": str(staged),
        "profile": profile or {},
        "case_name": "case_a",
        "laser_power": 100.0,
        "v_scan": 0.5,
        "time_values_s": np.linspace(0, 1e-3, nt).tolist(),
        "requested_fields": ["Temperature"],
        "case_manifest": {"case_name": "case_a"},
    }, root


def test_it_writes_one_flat_npz_and_no_loose_npy(tmp_path):
    from aiida_pino.utils.runtime_export import derived_member_worker

    task, root = _fake_task(tmp_path)
    derived_member_worker(task)

    assert sorted(f.name for f in root.glob("*.npz")) == ["case_a.npz"]
    assert list(root.rglob("*.npy")) == [], "loose .npy is the defect"


def test_the_staged_directory_is_recognised_as_consolidated_npz(tmp_path):
    """The whole point: this is what routes it to the archive reader."""
    from lpno.custom_utils.npz_datasource import is_consolidated_npz

    from aiida_pino.utils.runtime_export import derived_member_worker

    task, root = _fake_task(tmp_path)
    derived_member_worker(task)
    assert is_consolidated_npz(root)


def test_the_signature_keys_are_present(tmp_path):
    """Without these three the dir is NOT consolidated and the loader falls back."""
    from aiida_pino.utils.runtime_export import derived_member_worker

    task, root = _fake_task(tmp_path)
    derived_member_worker(task)
    with np.load(root / "case_a.npz", allow_pickle=True) as archive:
        assert {"power_w", "scan_speed_m_per_s", "case_name"} <= set(archive.files)


def test_it_is_actually_compressed(tmp_path):
    """A constant field is the material-channel case: it must not cost its full extent."""
    from aiida_pino.utils.runtime_export import derived_member_worker

    task, root = _fake_task(tmp_path, nt=8, nx=20, ny=20, nz=20)
    derived_member_worker(task)
    size = (root / "case_a.npz").stat().st_size
    uncompressed = 8 * 20 * 20 * 20 * 4
    assert size < uncompressed, f"{size} not smaller than {uncompressed}"


def test_axes_stay_1d_and_match_the_derived_field(tmp_path):
    """The consolidated format wants 1-D axes; a mismatch misreports the grid.

    Cropping z must shorten the z AXIS by the same amount as the field, or `raw_nz` and the
    field disagree and every downstream shape check reads a different grid.
    """
    from aiida_pino.utils.runtime_export import derived_member_worker

    task, root = _fake_task(tmp_path, nz=8, profile={"crop_z_top": 3})
    derived_member_worker(task)
    with np.load(root / "case_a.npz", allow_pickle=True) as archive:
        temperature = np.asarray(archive["Temperature"])
        # NATIVE rank. This test originally asserted ndim == 5 -- it pinned the defect:
        # the canonical channel axis leaking into the file is what shifted every axis
        # under NpzDirStore's positional indexing (one training arm). The flat
        # npz convention is (nt, nx, ny, nz), axes at 1/2/3.
        assert temperature.ndim == 4, temperature.shape
        for axis_name, axis_index in (("x", 1), ("y", 2), ("z", 3)):
            axis = np.asarray(archive[axis_name])
            assert axis.ndim == 1, f"{axis_name} must stay 1-D"
            assert axis.size == temperature.shape[axis_index], axis_name
        assert temperature.shape[3] == 5, "crop_z_top=3 of 8 leaves 5"


def test_the_size_gate_estimates_the_compressed_write(tmp_path, monkeypatch):
    """An uncompressed estimate would refuse exactly the builds compression enables."""
    import shutil

    from aiida_pino.utils.runtime_cache import estimated_derived_bytes, refuse_build_that_cannot_fit

    class _Member:
        def get_simulation_metadata(self):
            return {"array_shapes": {f"f{i}": [400, 1, 150, 100, 40] for i in range(17)}}

    members = [_Member() for _ in range(60)]
    monkeypatch.setattr(
        "aiida_pino.utils.runtime_cache._stored_compressed_bytes", lambda _m: 7 * 1024**3
    )
    assert estimated_derived_bytes(members, {}) < 10 * 1024**3

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: type("U", (), {"free": 186 * 1024**3, "total": 4000 * 1024**3})())
    refuse_build_that_cannot_fit(tmp_path, members, {}, label="ck300 test")


def test_an_unknown_stored_size_does_not_read_as_zero(tmp_path, monkeypatch):
    """Falling back to the uncompressed count is the safe direction."""
    from aiida_pino.utils.runtime_cache import estimated_arraydir_bytes, estimated_derived_bytes

    class _Member:
        def get_simulation_metadata(self):
            return {"array_shapes": {"f": [10, 1, 10, 10, 10]}}

    members = [_Member()]
    monkeypatch.setattr("aiida_pino.utils.runtime_cache._stored_compressed_bytes", lambda _m: 0)
    assert estimated_derived_bytes(members, {}) == estimated_arraydir_bytes(members, {})


def test_a_materialising_run_announces_its_cost_before_building():
    """User: the loader-side frame is the normal route; this one must announce itself."""
    from aiida_pino.utils import report_materialisation_cost

    class _Member:
        def get_simulation_metadata(self):
            return {"array_shapes": {"f": [400, 1, 150, 100, 40]}}

    class _Base:
        def iter_member_simulations(self):
            return [_Member() for _ in range(60)]

    lines: list[str] = []
    report_materialisation_cost(lines.append, _Base(), {"moving_ref_frame": True, "time_avg_window": 30})
    assert lines and "GB" in lines[0]
    assert "PINO_LAZY_MOVING_REF" in lines[0], "must name the loader-side alternative"


def test_the_warning_never_breaks_a_submission():
    """A cost estimate is not worth failing a run over."""
    from aiida_pino.utils import report_materialisation_cost

    class _Broken:
        def iter_member_simulations(self):
            raise RuntimeError("no members")

    lines: list[str] = []
    report_materialisation_cost(lines.append, _Broken(), {})
    assert lines == []


def test_the_sidecar_lands_beside_the_flat_npz_with_the_derived_block(tmp_path):
    """A derived member without its case-manifest sidecar kills training at data-build.

    NpzDirStore requires `<case>.manifest.json` beside `<case>.npz` to synthesise the
    trajectory channels, and `_beam_path_over_time` reads `manifest["derived"]
    ["laser_path_xy_m"]` -- a block only the REAL flow3d case manifest carries, not the
    arraydir wrapper (copying the wrapper was tried as a workaround and failed exactly
    there: KeyError 'laser_path_xy_m'). Four arms died on 2026-08-28 because the parent
    shipped the manifest to the worker only when `derived_fields` asked for it, so a
    `time_frames` or stride derivation wrote no sidecar at all.
    """
    import json

    from aiida_pino.utils.runtime_export import derived_member_worker

    task, root = _fake_task(tmp_path, profile={"time_frames": 2})
    task["case_manifest"] = {"case_name": "case_a",
                             "derived": {"laser_path_xy_m": [[0.0, 0.0], [1e-3, 0.0]]}}
    derived_member_worker(task)
    sidecar = root / "case_a.manifest.json"
    npz = root / "case_a.npz"
    assert npz.is_file()
    assert sidecar.is_file(), "no sidecar beside the flat npz"
    written = json.loads(sidecar.read_text())
    assert written["derived"]["laser_path_xy_m"] == [[0.0, 0.0], [1e-3, 0.0]]


def test_the_worker_task_ships_the_manifest_for_every_transforming_profile(tmp_path):
    """The bug was the CONDITION, so pin the condition, not only the writer.

    `task["case_manifest"]` was set only under `profile.get("derived_fields")`; every
    other transforming profile shipped none and the (correct) sidecar writer had nothing
    to write. The assignment must now be unconditional.
    """
    import inspect

    from aiida_pino.data import pino_dataset_collection as pdc

    src = inspect.getsource(pdc.PinoDatasetCollectionData._export_members)
    idx = src.index('task["case_manifest"] = merged_case_manifest(simulation)')
    guard = src[:idx].rsplit("if profile.get(", 1)
    # The nearest preceding `if profile.get(...)` must not be the gate for this line:
    # cheap structural check that the assignment is not nested under derived_fields.
    assert 'if profile.get("derived_fields"):' not in src.split("case_manifest")[0].rsplit("\n\n", 1)[-1]
    line = [l for l in src.splitlines() if 'task["case_manifest"]' in l][0]
    assert line.startswith("            task"), f"still nested deeper than the task loop: {line!r}"


def test_an_empty_manifest_writes_no_sidecar(tmp_path):
    # A simulation that records no case manifest must not gain a fabricated one.
    from aiida_pino.utils.runtime_export import derived_member_worker

    task, root = _fake_task(tmp_path, profile={"time_frames": 2})
    task["case_manifest"] = {}
    derived_member_worker(task)
    assert not (root / "case_a.manifest.json").exists()


def test_time_frames_is_applied_native_shaped_and_time_consistent(tmp_path):
    """One test, three writer defects (2026-08-28).

    The derived flat npz must (a) actually carry `time_frames` frames -- the profile
    recorded 60 while the arrays kept all native frames, the preheat lesson verbatim;
    (b) match the NATIVE flat-npz rank -- the canonical (nt, 1, nx, ny, nz) channel axis
    leaked into the file, and NpzDirStore indexes positionally, so every axis shifted
    (loud on mismatched dims, silently TRANSPOSED on coinciding ones); (c) keep `time_s`
    values that are a subset of the native ones, or beam centres evaluate at wrong times.
    """
    import numpy as np

    from aiida_pino.utils.runtime_export import derived_member_worker

    nt, keep = 12, 5
    task, root = _fake_task(tmp_path, nt=nt, profile={"time_frames": keep})
    native_times = np.load(task["staged_path"])["time_s"]

    # Production hands the worker CANONICAL arrays -- (nt, 1, nx, ny, nz). The fixture's
    # payload is native 4-D; the transform path broadcasts through either, and this test
    # must exercise the canonical rank or the squeeze goes untested.
    derived_member_worker(task)

    with np.load(root / "case_a.npz") as archive:
        temperature = archive["Temperature"]
        time_s = archive["time_s"]
    # (a) the subsample is REAL, not recorded intent
    assert temperature.shape[0] == keep, temperature.shape
    assert time_s.shape[0] == keep
    # (b) native rank: (nt, nx, ny, nz), no channel axis
    assert temperature.ndim == 4, temperature.shape
    # (c) the kept instants are the native ones, ends included
    assert set(np.round(time_s, 9)).issubset(set(np.round(native_times, 9)))
    assert time_s[0] == native_times[0] and time_s[-1] == native_times[-1]


def test_the_canonical_channel_axis_is_squeezed_at_write(tmp_path):
    """Feed the worker a 5-D canonical array directly; the file must be 4-D."""
    import numpy as np

    from aiida_pino.utils.runtime_export import write_derived_case_npz

    written = write_derived_case_npz(
        tmp_path,
        case_name="case_c",
        derived_arrays={"Temperature": np.zeros((6, 1, 4, 3, 2), dtype=np.float32)},
        time_values=np.linspace(0, 1, 6, dtype=np.float32),
        raw_axes={},
        laser_power=100.0,
        v_scan=0.5,
        profile={},
        case_manifest=None,
    )
    assert written["array_shapes"]["Temperature"] == [6, 4, 3, 2]
    with np.load(tmp_path / "case_c.npz") as archive:
        assert archive["Temperature"].shape == (6, 4, 3, 2)


def test_a_transformed_x_axis_comes_from_the_broadcast_not_a_respanned_linspace(tmp_path):
    """Defect four (case ti64_ph_018): the mref x sidecar was wrong.

    The moving-frame x-crop keeps a beam-following window with the ORIGINAL spacing; the
    old fallback rebuilt x as linspace(raw[0], raw[-1], n) -- the full absolute span at a
    stretched dx (measured 14.8 um for a true 10.0 um). The transformed coordinate
    broadcast went through the same `apply_derivation` as the data and is authoritative
    whenever the axis length changed; the raw axis stays exact when it did not.
    """
    import numpy as np

    from aiida_pino.utils.runtime_export import write_derived_case_npz

    nt, nx_raw, nx_win, ny, nz = 3, 90, 61, 4, 5
    raw_x = np.linspace(-4.45e-4, 4.45e-4, nx_raw, dtype=np.float32)
    # The beam-relative window the transform produced: original dx, shifted origin.
    window_x = (np.arange(nx_win, dtype=np.float32) * 1.0e-5) - 1.73e-4
    x_broadcast = np.broadcast_to(
        window_x[None, None, :, None, None], (nt, 1, nx_win, ny, nz)
    ).copy()
    written = write_derived_case_npz(
        tmp_path,
        case_name="case_m",
        derived_arrays={
            "Temperature": np.zeros((nt, 1, nx_win, ny, nz), dtype=np.float32),
            "x": x_broadcast,
        },
        time_values=np.linspace(0, 1, nt, dtype=np.float32),
        raw_axes={"x": raw_x, "y": np.zeros(ny, dtype=np.float32),
                  "z": np.zeros(nz, dtype=np.float32)},
        laser_power=100.0,
        v_scan=0.5,
        profile={"moving_ref_frame": True, "moving_frame_steps": nt},
        case_manifest=None,
    )
    with np.load(tmp_path / "case_m.npz") as archive:
        staged_x = np.asarray(archive["x"])
        assert staged_x.ndim == 1 and staged_x.size == nx_win
        # The broadcast's own line, not a re-span of the raw axis.
        np.testing.assert_allclose(staged_x, window_x, rtol=0, atol=1e-12)
        dx = np.diff(staged_x)
        assert abs(float(dx.mean()) - 1.0e-5) < 1e-9, "spacing must be the original dx"
        # Untransformed axes keep the exact raw values.
        assert np.asarray(archive["y"]).size == ny
