"""Tests for npz_file-mode simulation storage (verbatim compact-NPZ payloads).

Covers: byte-identical parity between `from_arrays` (canonicalized) and
`from_npz_file` (raw payload) reads, header-only metadata extraction,
realpath-normalized fingerprints, flat hardlink materialization of collections,
and collection compatibility checks driven by metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiida import orm

from aiida_pino.data import PinoDatasetCollectionData, PinoSimulationData
from aiida_pino.utils.flow3d_npz import (
    canonical_array_names,
    canonical_shapes_from_headers,
    canonicalize_flow3d_case_arrays,
    grid_params_from_axes,
    read_npz_headers,
)

GRID = (4, 3, 3)
NUM_FRAMES = 3
POWER_W = 150.0
SCAN_SPEED = 1.2


def _write_case_npz(
    directory: Path,
    case_name: str,
    *,
    power_w: float = POWER_W,
    scan_speed: float = SCAN_SPEED,
    fields: dict | None = None,
    seed: int = 7,
) -> Path:
    nx, ny, nz = GRID
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    payload = fields
    if payload is None:
        payload = {
            "Temperature": (300.0 + 1500.0 * rng.random((NUM_FRAMES, nx, ny, nz))).astype(np.float32),
            "Melt Region": rng.random((NUM_FRAMES, nx, ny, nz)).astype(np.float32),
            "Velocity": rng.standard_normal((NUM_FRAMES, nx, ny, nz, 3)).astype(np.float32),
        }
    path = directory / f"{case_name}.npz"
    np.savez_compressed(
        path,
        x=np.linspace(0.0, 2e-3, nx).astype(np.float32),
        y=np.linspace(-5e-4, 5e-4, ny).astype(np.float32),
        z=np.linspace(-5e-4, 2e-4, nz).astype(np.float32),
        time_s=np.linspace(0.0, 1e-3, NUM_FRAMES).astype(np.float64),
        power_w=np.float64(power_w),
        scan_speed_m_per_s=np.float64(scan_speed),
        case_name=np.str_(case_name),
        **payload,
    )
    return path


def _npz_node(path: Path, *, metadata: dict | None = None) -> PinoSimulationData:
    base = {"source_folder_name": "flatset", "source_group": ""}
    base.update(metadata or {})
    return PinoSimulationData.from_npz_file(path, metadata=base)


# --- parity ------------------------------------------------------------------


def test_from_npz_file_matches_from_arrays_byte_identical(tmp_path: Path) -> None:
    path = _write_case_npz(tmp_path, "case_a")

    with np.load(path, allow_pickle=False) as archive:
        canonical = canonicalize_flow3d_case_arrays(
            archive, case_name="case_a", laser_power=POWER_W, v_scan=SCAN_SPEED
        )
    arrays_node = PinoSimulationData.from_arrays(
        canonical.arrays,
        metadata={
            "source_folder_name": "flatset",
            "source_group": "case_a",
            "requested_fields": canonical.fields,
            "grid_params": grid_params_from_axes(canonical.x, canonical.y, canonical.z),
            "time_values_s": [float(value) for value in canonical.selected_times],
            "laser_power": POWER_W,
            "V_scan": SCAN_SPEED,
        },
    )
    npz_node = _npz_node(path, metadata={"laser_power": POWER_W, "V_scan": SCAN_SPEED})

    assert npz_node.get_storage_mode() == "npz_file"
    assert arrays_node.get_storage_mode() == "arrays"
    assert npz_node.get_array_aliases() == arrays_node.get_array_aliases()
    assert list(npz_node.get_array_aliases()) == canonical_array_names(canonical.fields)
    for original_name in npz_node.get_array_aliases():
        expected = arrays_node.get_array_by_original_name(original_name)
        actual = npz_node.get_array_by_original_name(original_name)
        assert actual.dtype == expected.dtype, original_name
        assert actual.shape == expected.shape, original_name
        assert actual.tobytes() == expected.tobytes(), original_name

    assert npz_node.get_num_timesteps() == arrays_node.get_num_timesteps() == NUM_FRAMES
    np.testing.assert_array_equal(npz_node.get_time_values_s(), arrays_node.get_time_values_s())
    metadata = npz_node.get_simulation_metadata()
    assert metadata["delta_t_s"] == pytest.approx(arrays_node.get_simulation_metadata()["delta_t_s"])
    assert metadata["spatial_shape"] == list(GRID)


def test_npz_node_reads_after_store_and_reload(tmp_path: Path) -> None:
    path = _write_case_npz(tmp_path, "case_reload")
    node = _npz_node(path)
    node.store()
    reloaded = orm.load_node(node.uuid)
    temperature = reloaded.get_array_by_original_name("Temperature")
    assert temperature.shape == (NUM_FRAMES, 1, *GRID)
    assert reloaded.get_array_by_original_name("Velocity").shape == (NUM_FRAMES, 3, *GRID)
    values = np.unique(reloaded.get_array_by_original_name("P"))
    assert values.tolist() == [pytest.approx(POWER_W)]


# --- metadata / headers ------------------------------------------------------


def test_from_npz_file_metadata_headers_and_extras(tmp_path: Path) -> None:
    path = _write_case_npz(tmp_path, "case_meta")
    node = _npz_node(path)

    # Verbatim payload object.
    with node.base.repository.open(PinoSimulationData.PAYLOAD_FILENAME, mode="rb") as handle:
        assert handle.read() == path.read_bytes()

    # Header-derived canonical shapes (no array loads needed).
    shapes = node.get_array_shapes()
    assert shapes["Temperature"] == [NUM_FRAMES, 1, *GRID]
    assert shapes["Velocity"] == [NUM_FRAMES, 3, *GRID]
    assert shapes["x"] == [NUM_FRAMES, 1, *GRID]

    metadata = node.get_simulation_metadata()
    assert metadata["case_name"] == "case_meta"
    # Laser scalars fall back to the NPZ provenance scalars.
    assert node.base.attributes.get("laser_power") == pytest.approx(POWER_W)
    assert node.base.attributes.get("V_scan") == pytest.approx(SCAN_SPEED)
    assert node.base.attributes.get("storage_mode") == "npz_file"
    assert node.base.extras.get("storage_mode") == "npz_file"
    assert node.base.extras.get("split") == "unassigned"
    assert node.base.extras.get("keywords") == []

    entries = {entry["original_name"]: entry for entry in node.iter_runtime_array_entries()}
    assert entries["Velocity"]["shape"] == [NUM_FRAMES, 3, *GRID]
    assert entries["Velocity"]["dtype"] == "float32"


def test_read_npz_headers_matches_actual_arrays(tmp_path: Path) -> None:
    path = _write_case_npz(tmp_path, "case_hdr")
    headers = read_npz_headers(str(path))
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            shape, dtype = headers[name]
            assert tuple(shape) == archive[name].shape
            assert dtype == archive[name].dtype
    shapes, fields = canonical_shapes_from_headers(headers, case_name="case_hdr")
    assert set(fields) == {"Temperature", "Melt Region", "Velocity"}
    assert shapes["Melt Region"] == [NUM_FRAMES, 1, *GRID]


def test_get_array_rejects_unknown_and_unnamed(tmp_path: Path) -> None:
    path = _write_case_npz(tmp_path, "case_err")
    node = _npz_node(path)
    with pytest.raises(KeyError, match="no array"):
        node.get_array("NoSuchField")
    with pytest.raises(ValueError, match="by name"):
        node.get_array()


def test_from_npz_file_rejects_missing_file_and_bad_laser(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        PinoSimulationData.from_npz_file(tmp_path / "missing.npz", metadata={})
    path = _write_case_npz(tmp_path, "case_nan", power_w=float("nan"))
    with pytest.raises(ValueError, match="laser_power"):
        PinoSimulationData.from_npz_file(path, metadata={})


# --- fingerprints ------------------------------------------------------------


def test_fingerprint_is_realpath_stable_across_symlinks(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical"
    path = _write_case_npz(canonical_dir, "case_fp")
    view_dir = tmp_path / "splits" / "train"
    view_dir.mkdir(parents=True)
    symlink = view_dir / "case_fp.npz"
    symlink.symlink_to(path)

    direct = _npz_node(path, metadata={"source_folder_name": "canonical", "source_group": ""})
    via_symlink = _npz_node(
        symlink, metadata={"source_folder_name": "splits", "source_group": "train"}
    )
    assert direct.get_simulation_fingerprint() == via_symlink.get_simulation_fingerprint()
    assert via_symlink.get_simulation_metadata()["source_npz_realpath"] == str(path.resolve())

    # A different field selection is a different identity.
    subset = _npz_node(path, metadata={"requested_fields": ["Temperature"]})
    assert subset.get_simulation_fingerprint() != direct.get_simulation_fingerprint()


# --- collections and materialization -----------------------------------------


def _case_manifest(case_name: str) -> dict:
    return {
        "case_name": case_name,
        "schema_version": 3,
        "input": {
            "laser": {"power_w": POWER_W, "scan_speed_m_per_s": SCAN_SPEED, "beam_radius_m": 5e-5},
            "simulation": {"finish_time_s": 1e-3},
        },
        "derived": {"scan_time_s": 1e-3},
    }


def _stored_collection(tmp_path: Path, names: list[str]) -> tuple[PinoDatasetCollectionData, dict[str, Path]]:
    paths = {}
    simulations = []
    for index, name in enumerate(names):
        path = _write_case_npz(tmp_path / "source", name, seed=10 + index)
        paths[name] = path
        node = _npz_node(
            path,
            metadata={"flow3d_case_manifest": _case_manifest(name)},
        )
        node.store()
        simulations.append(node)
    collection = PinoDatasetCollectionData.from_simulations(
        simulations, metadata={"source_folder_name": "flatset"}
    ).store()
    return collection, paths


def test_collection_materializes_flat_npz_with_original_bytes(tmp_path: Path, monkeypatch) -> None:
    # Pins the FLAT per-case staging path explicitly. 9d6d8bd made
    # zero-copy pack-index staging the default, and these assertions are
    # about the flat path (real per-case files, byte-identical payloads,
    # duplicate-name refusal), so they must name the mode instead of
    # inheriting whichever one happens to be default. The new default has
    # its own coverage in tests/test_pack_index_export.py.
    monkeypatch.setenv("AIIDA_PINO_PACK_INDEX", "0")
    collection, paths = _stored_collection(tmp_path, ["case_a", "case_b"])
    target = tmp_path / "staged"
    manifest_path = collection.export_runtime_arraydir(target)
    assert manifest_path.is_file()

    # Flat per-case NPZs with the original bytes at the staging root.
    for name, source_path in paths.items():
        staged = target / f"{name}.npz"
        assert staged.is_file()
        assert staged.read_bytes() == source_path.read_bytes()
        sidecar = target / f"{name}.manifest.json"
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text(encoding="utf-8"))["case_name"] == name

    # Member manifests record the npz_file storage mode.
    member_manifest = json.loads(
        (target / "members" / "simulation_0000" / "manifest.json").read_text(encoding="utf-8")
    )
    assert member_manifest["format"] == PinoSimulationData.NPZ_RUNTIME_MANIFEST_VERSION
    assert member_manifest["storage_mode"] == "npz_file"
    assert member_manifest["case_name"] == "case_a"
    # No bulk .npy shards in the member directories (zero payload duplication).
    assert not list((target / "members" / "simulation_0000").glob("*.npy"))

    # The staged folder validates as a training input without lpno.
    from aiida_pino.scripts.train_adapter import has_consolidated_case_npz, validate_dataset_input_dir

    assert has_consolidated_case_npz(target)
    validate_dataset_input_dir(target, label="Training dataset directory")


def test_derived_strided_npz_collection_stages_compact_npz(tmp_path: Path) -> None:
    from aiida import orm
    from aiida_pino.workflows.derive_dataset import materialize_derived_dataset

    collection, paths = _stored_collection(tmp_path, ["case_a"])
    profile = {
        "include_time": True,
        "time_as_batch": True,
        "time_as_dim": False,
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "time_avg_window": 1,
        "super_res_factor": 1,
        "stride": [2, 2, 1],
    }
    derived = materialize_derived_dataset(collection, orm.Dict(dict=profile))

    target = tmp_path / "staged_strided"
    derived.export_runtime_arraydir(target)

    staged = target / "case_a.npz"
    assert staged.is_file()
    assert not list((target / "members" / "simulation_0000").glob("*.npy"))

    with np.load(paths["case_a"], allow_pickle=False) as source, np.load(staged, allow_pickle=False) as actual:
        assert actual["x"].shape == (2,)
        assert actual["y"].shape == (2,)
        assert actual["z"].shape == (3,)
        np.testing.assert_array_equal(actual["x"], source["x"][::2])
        np.testing.assert_array_equal(actual["y"], source["y"][::2])
        np.testing.assert_array_equal(actual["z"], source["z"])
        np.testing.assert_array_equal(actual["Temperature"], source["Temperature"][:, ::2, ::2, ::1])
        assert "Velocity" in actual.files
        np.testing.assert_array_equal(actual["Velocity"], source["Velocity"][:, ::2, ::2, ::1])

    member_manifest = json.loads(
        (target / "members" / "simulation_0000" / "manifest.json").read_text(encoding="utf-8")
    )
    assert member_manifest["format"] == PinoSimulationData.NPZ_RUNTIME_MANIFEST_VERSION
    assert member_manifest["storage_mode"] == "npz_file"
    assert member_manifest["simulation_metadata"]["spatial_shape"] == [2, 2, 3]


def test_derived_strided_npz_collection_filters_runtime_fields(tmp_path: Path) -> None:
    from aiida import orm
    from aiida_pino.workflows.derive_dataset import materialize_derived_dataset

    collection, paths = _stored_collection(tmp_path, ["case_a"])
    profile = {
        "include_time": True,
        "time_as_batch": True,
        "time_as_dim": False,
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "time_avg_window": 1,
        "super_res_factor": 1,
        "stride": [2, 2, 1],
    }
    derived = materialize_derived_dataset(collection, orm.Dict(dict=profile))

    target = tmp_path / "staged_strided_filtered"
    derived.export_runtime_arraydir(target, runtime_fields=["Temperature", "Melt Region"])

    with np.load(paths["case_a"], allow_pickle=False) as source, np.load(target / "case_a.npz", allow_pickle=False) as actual:
        assert "Temperature" in actual.files
        assert "Melt Region" in actual.files
        assert "Velocity" not in actual.files
        np.testing.assert_array_equal(actual["Temperature"], source["Temperature"][:, ::2, ::2, ::1])


def test_derived_strided_npz_collection_defaults_to_supervised_fields(tmp_path: Path) -> None:
    from aiida import orm
    from aiida_pino.workflows.derive_dataset import materialize_derived_dataset

    fields = {
        "Temperature": np.ones((NUM_FRAMES, *GRID), dtype=np.float32),
        "Fraction Of Fluid": np.ones((NUM_FRAMES, *GRID), dtype=np.float32),
        "Pressure": np.ones((NUM_FRAMES, *GRID), dtype=np.float32),
        "Velocity": np.ones((NUM_FRAMES, *GRID, 3), dtype=np.float32),
    }
    path = _write_case_npz(tmp_path / "source", "case_supervised", fields=fields)
    node = _npz_node(
        path,
        metadata={"requested_fields": ["Temperature", "Fraction Of Fluid", "Pressure", "Velocity"]},
    ).store()
    collection = PinoDatasetCollectionData.from_simulations(
        [node],
        metadata={"dataset_role": "base", "source_folder_name": "dataset"},
    ).store()
    profile = {
        "include_time": True,
        "time_as_batch": True,
        "time_as_dim": False,
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "time_avg_window": 1,
        "super_res_factor": 1,
        "stride": [2, 2, 1],
    }
    derived = materialize_derived_dataset(collection, orm.Dict(dict=profile))

    target = tmp_path / "staged_strided_default_fields"
    derived.export_runtime_arraydir(target)

    with np.load(target / "case_supervised.npz", allow_pickle=False) as actual:
        assert "Temperature" in actual.files
        assert "Fraction Of Fluid" in actual.files
        assert "Pressure" not in actual.files
        assert "Velocity" not in actual.files


def test_collection_export_rejects_duplicate_case_names(tmp_path: Path, monkeypatch) -> None:
    # Pins the FLAT per-case staging path explicitly. 9d6d8bd made
    # zero-copy pack-index staging the default, and these assertions are
    # about the flat path (real per-case files, byte-identical payloads,
    # duplicate-name refusal), so they must name the mode instead of
    # inheriting whichever one happens to be default. The new default has
    # its own coverage in tests/test_pack_index_export.py.
    monkeypatch.setenv("AIIDA_PINO_PACK_INDEX", "0")
    path = _write_case_npz(tmp_path / "source", "case_dup")
    first = _npz_node(path, metadata={"source_group": "a"})
    first.store()
    # The same file stored twice (parser-level dedupe is bypassed here): the
    # two members share one case_name, so staging must fail loudly instead of
    # silently overwriting one member's payload with the other.
    second = _npz_node(path, metadata={"source_group": "b"})
    second.store()
    collection = PinoDatasetCollectionData.from_simulations(
        [first, second], metadata={"source_folder_name": "flatset"}
    ).store()
    with pytest.raises(ValueError, match="Duplicate case name"):
        collection.export_runtime_arraydir(tmp_path / "staged")


def test_direct_simulation_export_stages_into_target_dir(tmp_path: Path) -> None:
    path = _write_case_npz(tmp_path / "source", "case_direct")
    node = _npz_node(path, metadata={"flow3d_case_manifest": _case_manifest("case_direct")})
    node.store()
    target = tmp_path / "single"
    node.export_runtime_arraydir(target)
    assert (target / "case_direct.npz").read_bytes() == path.read_bytes()
    assert (target / "case_direct.manifest.json").is_file()
    assert (target / "manifest.json").is_file()


def test_collection_signature_mismatch_raises_for_different_fields(tmp_path: Path) -> None:
    path_a = _write_case_npz(tmp_path / "source", "case_full", seed=1)
    path_b = _write_case_npz(tmp_path / "source", "case_subset", seed=2)
    full = _npz_node(path_a)
    full.store()
    subset = _npz_node(path_b, metadata={"requested_fields": ["Temperature"]})
    subset.store()
    with pytest.raises(ValueError, match="not compatible"):
        PinoDatasetCollectionData.from_simulations(
            [full, subset], metadata={"source_folder_name": "flatset"}
        )


def test_bookkeeping_keys_are_not_fields(tmp_path):
    # The VTK->NPZ conversion writes 1-D bookkeeping arrays (frame_index,
    # skipped_fields); they must be skipped as fields, not shape-rejected.
    nx, ny, nz = GRID
    fields = {
        "Temperature": np.full((NUM_FRAMES, nx, ny, nz), 400.0, dtype=np.float32),
        "frame_index": np.arange(NUM_FRAMES),
        "skipped_fields": np.array(["a", "b"], dtype="U1"),
    }
    path = _write_case_npz(tmp_path, "bookkeeping_case", fields=fields)
    node = _npz_node(path)
    aliases = node.base.attributes.get("array_aliases")
    joined = " ".join(aliases)
    assert "Temperature" in joined
    assert "frame_index" not in joined and "skipped_fields" not in joined
    # The stored payload still carries the bookkeeping keys verbatim.
    import io
    import zipfile
    with node.base.repository.open("payload.npz", "rb") as fh:
        names = zipfile.ZipFile(io.BytesIO(fh.read())).namelist()
    assert "frame_index.npy" in names
