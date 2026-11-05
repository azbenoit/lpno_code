"""Tests for the consolidated-NPZ datasource and data-derived geometry.

The consolidated NPZ format is a single ``.npz`` per case that already stores a
uniform grid (explicit ``x``/``y``/``z`` axes) plus every field over time, e.g.
produced by the flow3D_scripts compact export. Unlike the per-timestep VTK path
it carries its own rectangular domain, so geometry is derived from the file
rather than a hardcoded config block.

These tests build tiny synthetic NPZ files (CPU-only, no license, no pyvista):
detection, geometry derivation, tensor layout, moving-frame parity, velocity
component synthesis, and the missing-field error.
"""

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/lpno/custom_utils/npz_datasource.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tests._npz_datasource_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NPZ = _load_module()


def _make_npz(path, *, nx=6, ny=5, nz=4, nt=8, power=150.0, speed=0.8,
              with_components=False, case="c1", seed=0, with_manifest=False):
    x = np.linspace(-1.0e-3, 1.0e-3, nx).astype(np.float32)
    y = np.linspace(-5.0e-4, 5.0e-4, ny).astype(np.float32)
    z = np.linspace(-3.0e-4, 1.0e-4, nz).astype(np.float32)
    t = np.linspace(0.0, 2.0e-3, nt).astype(np.float64)
    rng = np.random.default_rng(seed)
    payload = {
        "x": x, "y": y, "z": z, "time_s": t,
        "power_w": np.float64(power),
        "scan_speed_m_per_s": np.float64(speed),
        "case_name": np.array(case),
    }
    for name in ("Temperature", "Fraction Of Fluid", "Pressure", "Solid Fraction"):
        payload[name] = rng.random((nt, nx, ny, nz)).astype(np.float32)
    payload["Velocity"] = rng.random((nt, nx, ny, nz, 3)).astype(np.float32)
    if with_components:
        for i, comp in enumerate(("X-velocity", "Y-velocity", "Z-velocity")):
            payload[comp] = payload["Velocity"][..., i].copy()
    np.savez(path, **payload)
    if with_manifest:
        # The moving frame follows the beam's real position, so a case that asks for it
        # has to say where the beam goes. Straight +x across the domain at `speed`.
        import json
        xs = np.linspace(float(x[0]), float(x[-1]), 61)
        json.dump(
            {
                "input": {"laser": {"scan_speed_m_per_s": float(speed), "beam_radius_m": 5e-5}},
                "derived": {
                    "laser_path_xy_m": np.stack([xs, np.zeros_like(xs)], axis=1).tolist(),
                    "scan_time_s": float(x[-1] - x[0]) / float(speed),
                    "laser_off_time_s": float(x[-1] - x[0]) / float(speed),
                },
            },
            open(str(path).replace(".npz", ".manifest.json"), "w"),
        )
    return path


def test_is_consolidated_npz_file(tmp_path):
    good = _make_npz(tmp_path / "case.npz")
    assert NPZ.is_consolidated_npz(good) is True


def test_is_consolidated_npz_rejects_var_cache(tmp_path):
    # A datasource var_*.npz cache has x/y/z/P/V_scan but no consolidated signature keys.
    cache = tmp_path / "var_1.npz"
    np.savez(cache, x=np.zeros(3), y=np.zeros(3), z=np.zeros(3),
             P=np.zeros(3), V_scan=np.zeros(3), Temperature=np.zeros(3))
    assert NPZ.is_consolidated_npz(cache) is False


def test_is_consolidated_npz_directory(tmp_path):
    d = tmp_path / "cases"
    d.mkdir()
    _make_npz(d / "a.npz", case="a")
    # A stray var cache alongside must not confuse detection.
    np.savez(d / "var_1.npz", x=np.zeros(3), P=np.zeros(3))
    assert NPZ.is_consolidated_npz(d) is True


def test_is_consolidated_npz_directory_only_cache(tmp_path):
    d = tmp_path / "vtkdir"
    d.mkdir()
    np.savez(d / "var_1_time_steps30_avg1.npz", x=np.zeros(3), P=np.zeros(3))
    assert NPZ.is_consolidated_npz(d) is False


def test_derive_geometry_from_npz(tmp_path):
    _make_npz(tmp_path / "case.npz", nx=6, ny=5, nz=4)
    geo = NPZ.derive_geometry_from_npz(tmp_path / "case.npz")
    assert geo["resolution"] == (6, 5, 4)
    assert np.isclose(geo["x_range"][0], -1.0e-3) and np.isclose(geo["x_range"][1], 1.0e-3)
    assert np.isclose(geo["y_range"][0], -5.0e-4) and np.isclose(geo["y_range"][1], 5.0e-4)
    assert np.isclose(geo["z_range"][0], -3.0e-4) and np.isclose(geo["z_range"][1], 1.0e-4)


def test_load_consolidated_npz_raw_layout(tmp_path):
    _make_npz(tmp_path / "case.npz", nx=6, ny=5, nz=4, nt=8, power=150.0, speed=0.8)
    var = NPZ.load_consolidated_npz(
        tmp_path / "case.npz",
        requested_fields=["Temperature", "Velocity", "Pressure", "Fraction Of Fluid"],
        time_axis=True,
        moving_ref_frame=False,   # keep the full time axis, no crop/shift
    )
    assert var["Temperature"].shape == (1, 8, 1, 6, 5, 4)
    assert var["Velocity"].shape == (1, 8, 3, 6, 5, 4)
    # x/y/z broadcast onto their own axis
    assert np.allclose(var["x"][0, 0, 0, :, 0, 0], np.linspace(-1.0e-3, 1.0e-3, 6))
    assert np.allclose(var["y"][0, 0, 0, 0, :, 0], np.linspace(-5.0e-4, 5.0e-4, 5))
    assert np.allclose(var["z"][0, 0, 0, 0, 0, :], np.linspace(-3.0e-4, 1.0e-4, 4))
    # scalar conditioning grids
    assert np.allclose(var["P"], 150.0)
    assert np.allclose(var["V_scan"], 0.8)
    assert np.allclose(var["time_s"][0, :, 0, 0, 0, 0], np.linspace(0.0, 2.0e-3, 8))


def test_load_consolidated_npz_moving_frame_crops_x(tmp_path):
    _make_npz(tmp_path / "case.npz", nx=10, ny=3, nz=2, nt=8, with_manifest=True)
    var = NPZ.load_consolidated_npz(
        tmp_path / "case.npz",
        requested_fields=["Temperature"],
        time_axis=True,
        moving_ref_frame=True,
        moving_frame_steps=4,
        avg_window_size=1,
        super_res_factor=1,
    )
    # the beam-following window keeps the last Nsteps frames and the legacy width
    # nx - (Nsteps-1)*sr, so this shape is unchanged by the 2026-08-20 fix; only
    # where the window SITS changed. See tests/test_frame_transfer.py.
    assert var["Temperature"].shape == (1, 4, 1, 10 - 3, 3, 2)


def test_velocity_components_synthesized_from_vector(tmp_path):
    # NPZ has Velocity(...,3) but not the separate X/Y/Z-velocity scalars.
    _make_npz(tmp_path / "case.npz", nx=4, ny=3, nz=2, nt=5, with_components=False)
    var = NPZ.load_consolidated_npz(
        tmp_path / "case.npz",
        requested_fields=["Velocity", "X-velocity", "Y-velocity", "Z-velocity"],
        time_axis=True,
        moving_ref_frame=False,
    )
    assert var["X-velocity"].shape == (1, 5, 1, 4, 3, 2)
    assert np.allclose(var["X-velocity"][0, :, 0], var["Velocity"][0, :, 0])
    assert np.allclose(var["Y-velocity"][0, :, 0], var["Velocity"][0, :, 1])
    assert np.allclose(var["Z-velocity"][0, :, 0], var["Velocity"][0, :, 2])


def test_missing_required_field_raises(tmp_path):
    _make_npz(tmp_path / "case.npz")
    try:
        NPZ.load_consolidated_npz(
            tmp_path / "case.npz",
            requested_fields=["Temperature", "Nonexistent Field"],
            time_axis=True,
            moving_ref_frame=False,
        )
    except KeyError as exc:
        assert "Nonexistent Field" in str(exc)
    else:
        raise AssertionError("expected KeyError for a missing requested field")


def test_multi_case_directory_stacks_batch(tmp_path):
    d = tmp_path / "cases"
    d.mkdir()
    _make_npz(d / "a.npz", nx=4, ny=3, nz=2, nt=5, power=100.0, speed=0.5, case="a", seed=1)
    _make_npz(d / "b.npz", nx=4, ny=3, nz=2, nt=5, power=200.0, speed=1.0, case="b", seed=2)
    var = NPZ.load_consolidated_npz(
        d,
        requested_fields=["Temperature"],
        time_axis=True,
        moving_ref_frame=False,
    )
    assert var["Temperature"].shape == (2, 5, 1, 4, 3, 2)
    # P grid carries each case's power on its own batch row
    assert np.allclose(var["P"][0], 100.0)
    assert np.allclose(var["P"][1], 200.0)
