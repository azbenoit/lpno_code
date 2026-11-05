"""Converting legacy arrays-mode simulations to compressed npz_file mode.

arrays mode stored every canonical `(T, C, nx, ny, nz)` array as its own
uncompressed `.npy` object (~20x the compact NPZ). The converter de-broadcasts
the stored arrays back to the compact FLOW-3D schema, verifies the round-trip
through `canonicalize_flow3d_case_arrays` byte-for-byte, then swaps the
repository objects in place: same pk/uuid/links/extras, identical reads,
a fraction of the bytes.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_pino.data.pino_simulation import PinoSimulationData


def _compact_case(tmp_path, *, case_name="conv_case", nt=3, nx=4, ny=3, nz=5, seed=0):
    """Write a compact FLOW-3D per-case NPZ like the flow3D_scripts export."""
    rng = np.random.default_rng(seed)
    path = tmp_path / f"{case_name}.npz"
    np.savez_compressed(
        path,
        x=np.linspace(0.0, 1.0, nx).astype(np.float32),
        y=np.linspace(0.0, 2.0, ny).astype(np.float32),
        z=np.linspace(0.0, 3.0, nz).astype(np.float32),
        time_s=np.linspace(1e-6, 1e-4, nt).astype(np.float64),
        power_w=np.float64(200.0),
        scan_speed_m_per_s=np.float64(0.5),
        case_name=np.str_(case_name),
        Temperature=rng.random((nt, nx, ny, nz), dtype=np.float32) * 3000.0,
        Velocity=rng.random((nt, nx, ny, nz, 3), dtype=np.float32),
    )
    return path


def _arrays_mode_sim(tmp_path, *, case_name="conv_case", seed=0, tamper=None):
    """Store a sim the way the (buggy) run-dir import path did: canonical
    broadcast arrays via ``from_arrays``."""
    from aiida_pino.utils.flow3d_npz import canonicalize_flow3d_case_arrays, grid_params_from_axes

    path = _compact_case(tmp_path, case_name=case_name, seed=seed)
    with np.load(path, allow_pickle=False) as archive:
        case = canonicalize_flow3d_case_arrays(
            archive, case_name=case_name, laser_power=200.0, v_scan=0.5
        )
    arrays = {name: np.array(value) for name, value in case.arrays.items()}
    if tamper:
        tamper(arrays)
    metadata = {
        "source_type": "flow3d",
        "source_origin_kind": "flow3d_npz_import",
        "source_folder_name": case_name,
        "source_group": case_name,
        "case_name": case_name,
        "source_uri": f"file://{path}",
        "requested_fields": list(case.fields),
        "grid_params": grid_params_from_axes(case.x, case.y, case.z),
        "relative_paths": [f"npz/{case_name}.npz#frame={i}" for i in case.frame_indices],
        "time_values_s": [float(v) for v in case.selected_times],
        "V_scan": 0.5,
        "laser_power": 200.0,
    }
    sim = PinoSimulationData.from_arrays(arrays, metadata=metadata, source={"uri": metadata["source_uri"]})
    # Deliberate legacy-mode storage (session fixture sets the escape hatch).
    return sim.store(), arrays


def test_conversion_flips_mode_and_shrinks(aiida_profile_clean, tmp_path) -> None:
    sim, _ = _arrays_mode_sim(tmp_path)
    assert sim.get_storage_mode() == PinoSimulationData.STORAGE_MODE_ARRAYS
    before = sim.data_size_bytes()

    report = sim.compress_to_npz_file()

    assert report["bytes_before"] == before
    assert 0 < report["bytes_after"] < before
    reloaded = PinoSimulationData.collection.get(uuid=sim.uuid)
    assert reloaded.get_storage_mode() == PinoSimulationData.STORAGE_MODE_NPZ_FILE
    assert sorted(reloaded.base.repository.list_object_names()) == sorted(
        [PinoSimulationData.PAYLOAD_FILENAME, f"{PinoSimulationData.TIME_VALUES_STORAGE_NAME}.npy"]
    )


def test_conversion_preserves_reads_and_identity(aiida_profile_clean, tmp_path) -> None:
    sim, stored = _arrays_mode_sim(tmp_path)
    pk, uuid = sim.pk, sim.uuid
    fingerprint = sim.get_simulation_fingerprint()
    sim.base.extras.set("split", "train")
    aliases = sim.get_array_aliases()

    sim.compress_to_npz_file()

    reloaded = PinoSimulationData.collection.get(uuid=uuid)
    assert reloaded.pk == pk
    assert reloaded.get_simulation_fingerprint() == fingerprint
    assert reloaded.base.extras.get("split") == "train"
    for original, alias in aliases.items():
        got = reloaded.get_array(alias)
        assert np.array_equal(np.asarray(got), stored[original]), f"array `{original}` changed"
    times = reloaded.get_time_values_s()
    assert times is not None and times.shape[0] == 3


def test_conversion_is_idempotent(aiida_profile_clean, tmp_path) -> None:
    sim, _ = _arrays_mode_sim(tmp_path)
    assert sim.compress_to_npz_file()["bytes_after"] > 0
    second = sim.compress_to_npz_file()
    assert second["bytes_before"] == second["bytes_after"] == 0, "second conversion should no-op"


def test_conversion_refuses_non_broadcast_channels(aiida_profile_clean, tmp_path) -> None:
    """If a stored array is not an exact canonicalization image, refuse and
    leave the node untouched — never guess data back."""

    def tamper(arrays):
        x = np.array(arrays["x"])
        x[1, 0, 2, 1, 3] += 1e-3  # breaks the pure-broadcast structure
        arrays["x"] = x

    sim, stored = _arrays_mode_sim(tmp_path, tamper=tamper)
    with pytest.raises(ValueError, match="x"):
        sim.compress_to_npz_file()
    assert sim.get_storage_mode() == PinoSimulationData.STORAGE_MODE_ARRAYS
    got = sim.get_array(sim.get_array_aliases()["x"])
    assert np.array_equal(np.asarray(got), stored["x"]), "failed conversion mutated the node"


def test_conversion_refuses_evicted_and_foreign_layouts(aiida_profile_clean, tmp_path, pino_simulation_factory) -> None:
    sim, _ = _arrays_mode_sim(tmp_path)
    sim.evict_data()
    with pytest.raises(Exception):
        sim.compress_to_npz_file()

    # A sim without the six canonical channels is not a run-dir import.
    foreign = pino_simulation_factory("not_canonical")
    with pytest.raises(ValueError, match="channel"):
        foreign.compress_to_npz_file()


def test_provenance_links_survive_conversion(aiida_profile_clean, tmp_path) -> None:
    from aiida_pino.data.dataset_hierarchy import create_base_dataset

    sims = []
    for i in range(2):
        sim, _ = _arrays_mode_sim(tmp_path, case_name=f"conv_{i}", seed=i)
        sims.append(sim)
    base, _ = create_base_dataset(sims, name="convert_me")

    for sim in sims:
        sim.compress_to_npz_file()

    members = list(base.iter_member_simulations())
    assert len(members) == 2
    assert all(m.get_storage_mode() == PinoSimulationData.STORAGE_MODE_NPZ_FILE for m in members)


def test_storing_arrays_mode_sim_is_refused_by_default(aiida_profile_clean, monkeypatch) -> None:
    """The store()-level guard: nothing can persist an uncompressed arrays-mode
    simulation unless the escape hatch is set explicitly. This is what makes
    the 2026-07-27 bug structurally unrepeatable, whatever the import path."""
    monkeypatch.delenv(PinoSimulationData.ARRAYS_STORE_ENV, raising=False)
    arrays = {"Temperature": np.ones((2, 1, 4, 4, 4), dtype=np.float32)}
    metadata = {"source_folder_name": "guarded", "source_group": "guarded"}
    sim = PinoSimulationData.from_arrays(arrays, metadata=metadata)
    with pytest.raises(ValueError, match="npz_file"):
        sim.store()

    monkeypatch.setenv(PinoSimulationData.ARRAYS_STORE_ENV, "1")
    sim2 = PinoSimulationData.from_arrays(arrays, metadata=dict(metadata))
    assert sim2.store().pk > 0, "escape hatch must still allow legacy/test storage"


def _svc():
    from web.app.aiida_service import AiidaConsoleService
    from web.app.config import Settings

    return AiidaConsoleService(Settings())


def test_service_compress_dry_run_then_apply(aiida_profile_clean, tmp_path) -> None:
    from aiida_pino.data.dataset_hierarchy import create_base_dataset

    sims = []
    for i in range(2):
        sim, _ = _arrays_mode_sim(tmp_path, case_name=f"svc_{i}", seed=10 + i)
        sims.append(sim)
    base, _ = create_base_dataset(sims, name="svc_convert")
    svc = _svc()

    report = svc.compress_dataset_data(base.pk, dry_run=True)
    assert report["would_convert"] == 2 and report["bytes_before"] > 0
    assert all(s.get_storage_mode() == PinoSimulationData.STORAGE_MODE_ARRAYS for s in sims)

    done = svc.compress_dataset_data(base.pk, dry_run=False)
    assert done["converted"] == 2
    assert 0 < done["bytes_after"] < done["bytes_before"]
    again = svc.compress_dataset_data(base.pk, dry_run=True)
    assert again["would_convert"] == 0, "post-import regression check must report clean"
