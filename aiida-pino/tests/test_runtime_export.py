"""The parallel runtime export: locators, workers, and equivalence to the serial path.

The bar for all of this is byte-identity: the worker path must produce exactly
the arrays `derive_simulation_arrays` produces, or a training staged through
one path is not reproducible through the other.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from aiida_pino.utils.runtime_export import (
    apply_derivation,
    derive_time_values,
    derived_member_worker,
    payload_locator,
    read_locator_bytes,
    sendfile_locator,
)


def _npz_file_sim(tmp_path: Path, case_name: str, *, frames: int = 5, seed: int = 7):
    """A stored npz_file-mode simulation, like the production ingest makes."""
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
        time_s=np.linspace(0.0, 1e-3, frames).astype(np.float64),
        power_w=np.float64(150.0),
        scan_speed_m_per_s=np.float64(1.2),
        case_name=np.str_(case_name),
        Temperature=(300.0 + 1500.0 * rng.random((frames, nx, ny, nz))).astype(np.float32),
        **{"Melt Region": rng.random((frames, nx, ny, nz)).astype(np.float32)},
    )
    # The moving frame follows the beam's REAL position, so a fixture that exercises it
    # has to say where the beam goes. Straight +x across the domain at `scan_speed`.
    xs = np.linspace(0.0, 2e-3, 21)
    manifest = {
        "input": {"laser": {"scan_speed_m_per_s": 1.2, "beam_radius_m": 5e-5}},
        "derived": {
            "laser_path_xy_m": np.stack([xs, np.zeros_like(xs)], axis=1).tolist(),
            "scan_time_s": 2e-3 / 1.2,
            "laser_off_time_s": 2e-3 / 1.2,
        },
    }
    node = PinoSimulationData.from_npz_file(
        path,
        metadata={"source_folder_name": "runtime_export_test", "source_group": case_name,
                  "laser_power": 150.0, "V_scan": 1.2,
                  "flow3d_case_manifest": manifest},
    )
    return node.store()

MOVING_PROFILE = {
    "stride": [1, 1, 1],
    "time_as_dim": False,
    "include_time": True,
    "time_as_batch": False,
    "time_avg_window": 3,
    "moving_ref_frame": True,
    "super_res_factor": 1,
    "moving_frame_steps": 3,
}

#: Same transform PLUS a built channel. Split out as its own profile because the
#: single-profile version of the both-paths test below could not catch the
#: 2026-08-22 defect: `derived_fields` shipped into `derive_simulation_arrays`
#: only, `derived_member_worker` had no code that read it, and the test passed
#: because MOVING_PROFILE leaves the key at its `[]` default. A key that no
#: profile sets is a key this test does not test.
DERIVED_FIELDS_PROFILE = {**MOVING_PROFILE, "derived_fields": ["conduction_field"]}

#: A z-cropped variant. `crop_z_top` restricts the DOMAIN before every other transform, so
#: it has to agree across both writers like anything else -- and it is the key most likely
#: to disagree, because it changes an array's shape rather than its values.
CROPPED_PROFILE = {**MOVING_PROFILE, "crop_z_top": 1}

#: Every profile the worker-vs-serial equivalence runs against.
BOTH_PATHS_PROFILES = {
    "moving": MOVING_PROFILE,
    "derived_fields": DERIVED_FIELDS_PROFILE,
    "cropped": CROPPED_PROFILE,
}

#: Derivation keys deliberately NOT exercised by the matrix above, each with the
#: reason. `test_every_derivation_key_is_exercised_or_exempt` fails when a new key
#: is neither set by a profile nor listed here -- so adding one forces the choice
#: instead of silently landing untested on one of the two paths.
BOTH_PATHS_UNEXERCISED = {
    "time_as_dim": "layout metadata; consumed at load, does not transform arrays",
    "time_as_batch": "layout metadata; consumed at load, does not transform arrays",
    "time_frames": "member selection, applied before this transform",
    "include_time": "mutually exclusive with moving_ref_frame, which the matrix sets",
    "stride": "exercised by test_strided_npz_matches_the_serial_derivation",
    "super_res_factor": "upsampling is covered by tests/test_super_resolution.py",
    "materialize": "staging directive: chooses the ENCODING of the staged store, "
                   "not what the transform computes; the arrays are byte-identical "
                   "either way (measured 2026-09-09)",
}


def test_locator_range_read_and_sendfile(tmp_path):
    """A locator is just (path, offset, size); both readers must honour it."""
    blob = bytes(range(256)) * 64
    pack = tmp_path / "pack"
    pack.write_bytes(b"HEAD" + blob + b"TAIL")
    locator = {"kind": "pack", "path": str(pack), "offset": 4, "size": len(blob)}

    assert read_locator_bytes(locator) == blob

    out = tmp_path / "staged.npz"
    sendfile_locator(locator, out)
    assert out.read_bytes() == blob


def test_locator_short_read_names_the_repack_risk(tmp_path):
    pack = tmp_path / "pack"
    pack.write_bytes(b"0123456789")
    locator = {"kind": "pack", "path": str(pack), "offset": 4, "size": 100}
    with pytest.raises(OSError, match="repacked underneath"):
        read_locator_bytes(locator)


def test_payload_locator_finds_a_loose_object(aiida_profile_clean, tmp_path):
    """A freshly stored npz_file simulation is a loose object: locatable."""
    sim = _npz_file_sim(tmp_path, "loc_case")
    locator = payload_locator(sim)
    assert locator is not None and locator["kind"] == "loose"
    raw = read_locator_bytes(locator)
    with sim.base.repository.open(sim.PAYLOAD_FILENAME, mode="rb") as handle:
        assert raw == handle.read()


def _beam_track_task_fields(sim, profile) -> dict:
    """The beam-track fields production ships with a task: slices AND the tail stop.

    Both come from one call because they describe the same frames -- `moving_frame_stop`
    says where the window ends when the laser overruns its declared path, and slices built
    for those frames must be applied to those frames. Mirrored here in one helper so a
    field added to the production task cannot be silently missing from this parity test.
    """
    from aiida_pino.data.pino_dataset_collection import _beam_following_slices_for

    slices, stop = _beam_following_slices_for(
        sim,
        int(profile["moving_frame_steps"]),
        int(profile["super_res_factor"]),
    )
    return {
        "moving_frame_slices": [(int(sl.start), int(sl.stop)) for sl in slices],
        "moving_frame_stop": int(stop),
    }


@pytest.mark.parametrize("profile_name", sorted(BOTH_PATHS_PROFILES))
def test_worker_matches_the_serial_derivation(aiida_profile_clean, tmp_path, profile_name):
    """Byte-identity between derived_member_worker and derive_simulation_arrays."""
    from aiida_pino.data.pino_dataset_collection import (
        _beam_following_slices_for,
        derive_simulation_arrays,
        merged_case_manifest,
    )

    profile = BOTH_PATHS_PROFILES[profile_name]
    sim = _npz_file_sim(tmp_path / "src", "equiv_case")

    serial_arrays, serial_time = derive_simulation_arrays(sim, profile)

    time_values = sim.get_time_values_s()
    task = {
        "index": 0,
        "target_dir": str(tmp_path / "member"),
        "case_name": str(sim.get_simulation_metadata().get("case_name") or sim.get_simulation_identifier()),
        "laser_power": float(sim.base.attributes.get("laser_power")),
        "v_scan": float(sim.base.attributes.get("V_scan")),
        "requested_fields": list(sim.base.attributes.get("requested_fields", []) or []) or None,
        "profile": dict(profile),
        "time_values_s": time_values.tolist() if time_values is not None else None,
        "locator": payload_locator(sim),
        # Production sends the beam track with the task: the worker has no AiiDA session
        # and cannot read the case manifest itself. Same helper, so this stays a test of
        # worker-vs-serial identity rather than of two different windows.
        **_beam_track_task_fields(sim, profile),
    }
    if profile.get("derived_fields"):
        # Production ships this for the same reason as the beam track above.
        task["case_manifest"] = merged_case_manifest(sim)
    assert task["locator"] is not None
    result = derived_member_worker(task)

    assert set(result["array_shapes"]) == set(serial_arrays)
    for name in profile.get("derived_fields") or []:
        # Guards the degenerate pass: if the serial path also stopped building the
        # channel, the set-equality above would still hold with both sides missing it.
        assert name in serial_arrays, f"serial path built no {name}"
        assert name in result["array_shapes"], f"worker built no {name}"
    # LAYOUT-AWARE since the steady staging fix (61f464d): every profile in this matrix
    # carries `moving_ref_frame=True`, and a moving-frame member now stages as ARRAYDIR
    # (loose `.npy` + array_entries) because the flat consolidated npz routes to a store
    # that refuses baked mref by design. The equivalence CONTRACT is unchanged and
    # layout-independent -- the pooled worker's output must byte-equal the serial
    # derivation -- and that is still what is asserted; only where the bytes live moved.
    coordinate_names = {"x", "y", "z", "time_s"}
    assert result["layout"] == "arraydir", "an mref profile must stage arraydir"
    member = tmp_path / "member"
    stored = {e["original_name"]: member / e["filename"] for e in result["array_entries"]}
    compared = 0
    for name, serial in serial_arrays.items():
        if name in coordinate_names:
            continue
        assert name in stored, f"worker staged no file for {name}"
        np.testing.assert_array_equal(
            np.load(stored[name]), serial,
            err_msg=f"array {name} diverges from the serial path",
        )
        compared += 1
    assert compared, "no field arrays compared -- the test would pass vacuously"
    # The arraydir layout ships the derived time axis as its own file, and NO flat npz:
    # the flat file's presence is exactly what routed steady evals to the refusing store.
    assert result["time_values_filename"] == "time_values_s.npy"
    assert (member / "time_values_s.npy").is_file()
    assert not list(member.glob("*.npz")), "a flat npz on an mref member is the regression"


def test_strided_fast_path_refuses_derived_fields(aiida_profile_clean, tmp_path):
    """The compact writer copies RAW archive members, so it cannot serve a BUILT one.

    Third writer of this transform (after the serial path and `derived_member_worker`).
    It qualifies on a strided, non-moving profile, and would stage a member missing the
    channel -- silently, exactly like the worker did. Refusing the fast path sends the
    member to the transform path, which builds it.
    """
    from aiida_pino.data.pino_dataset_collection import PinoDatasetCollectionData

    sim = _npz_file_sim(tmp_path / "src", "strided_case")
    qualifying = {
        "stride": [2, 2, 2],
        "include_time": True,
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "time_avg_window": 1,
        "super_res_factor": 1,
    }
    assert PinoDatasetCollectionData._can_write_strided_npz(qualifying, sim) is True
    with_built = {**qualifying, "derived_fields": ["conduction_field"]}
    assert PinoDatasetCollectionData._can_write_strided_npz(with_built, sim) is False


def test_every_derivation_key_is_exercised_or_exempt():
    """A new derivation key must reach BOTH paths, or say in writing why it need not.

    This is the check that was missing on 2026-08-22: `derived_fields` was added to
    the profile schema and implemented on the serial path only, and every existing
    test still passed. The transform has two implementations, so a key tested against
    one of them is untested by definition: duplicated builders diverge.
    """
    from aiida_pino.utils.dataset_derivation import DERIVATION_PROFILE_KEYS

    exercised = {k for profile in BOTH_PATHS_PROFILES.values() for k in profile}
    unaccounted = set(DERIVATION_PROFILE_KEYS) - exercised - set(BOTH_PATHS_UNEXERCISED)
    assert not unaccounted, (
        f"derivation key(s) {sorted(unaccounted)} are set by no profile in "
        f"BOTH_PATHS_PROFILES and listed in no BOTH_PATHS_UNEXERCISED reason. Add a "
        f"profile that exercises the key against both the serial and worker paths, or "
        f"record why it cannot affect the transform."
    )
    assert not set(BOTH_PATHS_UNEXERCISED) - set(DERIVATION_PROFILE_KEYS), (
        "BOTH_PATHS_UNEXERCISED names a key that is no longer a derivation key"
    )


def test_collection_export_uses_the_pool_and_matches_manifest_shape(
    aiida_profile_clean, tmp_path, monkeypatch
):
    """A derived collection staged through the pool has the same manifest contract."""
    from aiida_pino.utils import build_collection_metadata_from_simulations
    from aiida_pino.utils.compose import compose_collection, member_inputs

    sims = [_npz_file_sim(tmp_path / f"src_{i}", f"pool_{i}", seed=i) for i in range(3)]
    metadata = build_collection_metadata_from_simulations(
        sims,
        collection_name="pool_export",
        dataset_role="derived",
        composition_kind="composed",
        extra_metadata={"derivation_profile": dict(MOVING_PROFILE)},
    )
    node, _ = compose_collection(sims, metadata, **member_inputs(sims))

    monkeypatch.setenv("AIIDA_PINO_EXPORT_WORKERS", "2")
    out = tmp_path / "staged"
    node.export_runtime_arraydir(out)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["simulation_count"] == 3
    for member in manifest["members"]:
        member_manifest = json.loads((out / member["manifest_file"]).read_text())
        # This collection's profile is MOVING_PROFILE, and a moving-frame member stages
        # as ARRAYDIR since 61f464d -- the flat npz routes to a store that refuses baked
        # mref by design (the assertion here used to pin the flat contract for ALL
        # derived members, which was the regression). The arraydir manifest lists
        # `array_entries` naming per-array `.npy` files that must exist, and names NO
        # flat npz.
        assert "npz_file" not in member_manifest
        entries = member_manifest["array_entries"]
        assert entries, "arraydir member manifest lists no arrays"
        shapes = member_manifest["simulation_metadata"]["array_shapes"]
        assert shapes, "worker wrote no arrays"
        for entry in entries:
            staged = (out / member["relative_path"] / entry["filename"]).resolve()
            assert staged.is_file(), f"array_entries names a missing file: {entry['filename']}"
            assert list(np.load(staged).shape) == list(shapes[entry["original_name"]])
