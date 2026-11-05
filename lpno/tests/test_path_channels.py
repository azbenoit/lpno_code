"""Unit tests for laser-path conditioning channels (arbitrary weld path inputs).

Covers the geometric derivation (beam-centre interpolation, footprint, dose) and
the datasource plumbing (manifest read, channel attachment, missing-manifest
error). Numpy-only, no GPU/license.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lpno.custom_utils import npz_datasource as nds


def _straight_manifest(length=1.0e-3, speed=0.8, k=61, beam_r=5e-5, off=None):
    """Manifest for a straight +x path from (0,0) to (length,0), arc-uniform."""
    xs = np.linspace(0.0, length, k)
    xy = np.stack([xs, np.zeros_like(xs)], axis=1).tolist()
    scan_time = length / speed
    return {
        "input": {"laser": {"scan_speed_m_per_s": speed, "beam_radius_m": beam_r}},
        "derived": {
            "laser_path_xy_m": xy,
            "scan_time_s": scan_time,
            "laser_off_time_s": scan_time if off is None else off,
        },
    }


def _variable_speed_manifest(beam_r=5e-5):
    """Variable-speed path with a dwell at the middle knot."""
    return {
        "input": {"laser": {"scan_speed_m_per_s": 0.5, "beam_radius_m": beam_r}},
        "derived": {
            "laser_path_xy_m": [[0.0, 0.0], [1.0e-3, 0.0], [1.0e-3, 0.0], [2.0e-3, 0.0]],
            "laser_path_times_s": [0.0, 1.0e-3, 1.5e-3, 2.0e-3],
            "laser_path_variable_speed": True,
            "laser_path_leg_speeds_m_per_s": [1.0, 0.0, 2.0],
            "scan_time_s": 2.0e-3,
            "laser_off_time_s": 2.0e-3,
            "surface_z_m": 0.0,
            "laser_wavelength_m": 1.064e-6,
        },
    }


def test_beam_centre_endpoints_and_midpoint():
    man = _straight_manifest(length=1.0e-3, speed=0.8)
    scan_time = 1.0e-3 / 0.8
    t = np.array([0.0, scan_time / 2, scan_time, 2 * scan_time])  # last is past laser-off
    c = nds._beam_centres_over_time(man, t)
    assert np.allclose(c[0], [0.0, 0.0], atol=1e-9)  # start
    assert np.allclose(c[1], [0.5e-3, 0.0], atol=1e-6)  # halfway along x
    assert np.allclose(c[2], [1.0e-3, 0.0], atol=1e-6)  # end
    assert np.allclose(c[3], [1.0e-3, 0.0], atol=1e-6)  # clamped at end after laser off


def test_variable_speed_uses_manifest_times_and_marks_dwell():
    man = _variable_speed_manifest()
    t = np.array([0.5e-3, 1.25e-3, 1.75e-3, 2.5e-3])
    centres, dirs, speeds = nds._beam_path_over_time(man, t)
    assert np.allclose(centres[0], [0.5e-3, 0.0], atol=1e-9)
    assert np.allclose(centres[1], [1.0e-3, 0.0], atol=1e-9)  # dwell holds position
    assert np.allclose(centres[2], [1.5e-3, 0.0], atol=1e-9)
    assert np.allclose(centres[3], [2.0e-3, 0.0], atol=1e-9)  # off/end clamp
    assert np.allclose(speeds, [1.0, 0.0, 2.0, 0.0], atol=1e-9)
    assert np.allclose(dirs[1], [0.0, 0.0], atol=1e-12)


def test_build_path_channels_shapes_and_ranges():
    man = _straight_manifest()
    x = np.linspace(-1e-4, 1.1e-3, 24, dtype=np.float32)
    y = np.linspace(-2e-4, 2e-4, 16, dtype=np.float32)
    z = np.linspace(-3e-4, 0.0, 8, dtype=np.float32)
    t = np.linspace(0.0, 1.0e-3 / 0.8, 10)
    ch = nds._build_path_channels(nds.PATH_CHANNEL_NAMES, man, x, y, z, t)
    assert set(ch) == set(nds.PATH_CHANNEL_NAMES)
    for k, v in ch.items():
        assert v.shape == (1, t.size, 1, x.size, y.size, z.size), k
        assert not np.isnan(v).any(), k
    # Footprint peaks at 1 on the beam; on this coarse grid the nearest node is
    # close but not exact, so allow a sampling gap.
    assert 0.9 <= ch["beam_dist"].max() <= 1.0
    assert ch["beam_dist"].min() >= 0.0


def test_anisotropic_channels_are_signed_and_speed_is_local():
    man = _straight_manifest(length=1.0e-3, speed=0.8)
    x = np.array([0.45e-3, 0.5e-3, 0.55e-3], dtype=np.float32)
    y = np.array([-5e-5, 0.0, 5e-5], dtype=np.float32)
    z = np.array([0.0], dtype=np.float32)
    t = np.array([0.5e-3 / 0.8], dtype=np.float32)
    ch = nds._build_path_channels(["beam_s", "beam_n", "beam_speed"], man, x, y, z, t)
    s = ch["beam_s"][0, 0, 0, :, 1, 0]
    n = ch["beam_n"][0, 0, 0, 1, :, 0]
    assert np.allclose(s, [-1.0, 0.0, 1.0], atol=1e-5)
    assert np.allclose(n, [-1.0, 0.0, 1.0], atol=1e-5)
    assert np.allclose(ch["beam_speed"], 0.8)


def test_depth_aware_sigma_and_exp_channels_vary_with_z(monkeypatch):
    monkeypatch.setenv("PINO_BEAM_RAYLEIGH_M", "1e-4")
    monkeypatch.setenv("PINO_BEAM_ABS_DEPTH_M", "1e-4")
    man = _straight_manifest(length=1.0e-3, speed=0.8)
    man["derived"]["surface_z_m"] = 0.0
    x = np.array([5e-5], dtype=np.float32)  # one sigma from the beam centre at t=0
    y = np.array([0.0], dtype=np.float32)
    z = np.array([0.0, -2e-4], dtype=np.float32)
    t = np.array([0.0, 1e-4], dtype=np.float32)
    ch = nds._build_path_channels(
        ["beam_dist_sigma", "beam_dist_sigma_exp", "path_dose_sigma", "path_dose_sigma_exp"],
        man, x, y, z, t,
    )
    sigma_surface = ch["beam_dist_sigma"][0, 0, 0, 0, 0, 0]
    sigma_deep = ch["beam_dist_sigma"][0, 0, 0, 0, 0, 1]
    exp_surface = ch["beam_dist_sigma_exp"][0, 0, 0, 0, 0, 0]
    exp_deep = ch["beam_dist_sigma_exp"][0, 0, 0, 0, 0, 1]
    assert sigma_deep > sigma_surface  # wider beam at depth gives larger off-axis value
    assert exp_deep < exp_surface
    assert ch["path_dose_sigma"].shape == (1, t.size, 1, x.size, y.size, z.size)
    assert ch["path_dose_sigma_exp"].shape == (1, t.size, 1, x.size, y.size, z.size)


def test_dose_monotonic_and_stops_after_laser_off():
    # Laser off well before the last frame; dose must plateau afterwards.
    man = _straight_manifest(length=1.0e-3, speed=0.8, off=1.0e-3 / 0.8)
    x = np.linspace(-1e-4, 1.1e-3, 24, dtype=np.float32)
    y = np.linspace(-2e-4, 2e-4, 16, dtype=np.float32)
    z = np.linspace(-3e-4, 0.0, 4, dtype=np.float32)
    t = np.linspace(0.0, 2.0e-3, 20)  # runs past laser_off
    dose = nds._build_path_channels(["path_dose"], man, x, y, z, t)["path_dose"]
    peak_t = dose[0, :, 0].reshape(t.size, -1).max(axis=1)
    assert np.all(np.diff(peak_t) >= -1e-6)  # non-decreasing in time
    off_frames = t > man["derived"]["laser_off_time_s"]
    if off_frames.sum() >= 2:
        after = peak_t[off_frames]
        assert np.allclose(after, after[0], atol=1e-6)  # plateau once laser is off
    assert dose.max() <= 1.0 + 1e-6  # fraction-of-dwell normalization


def test_beam_dist_zero_during_cooling_tail():
    """Frames after laser_off_time_s must have a ZERO instantaneous footprint.

    Cooling-tail datasets (material_ck300: finish = 1.25*track/V, laser powers
    off at the track end) have frames where the melt pool collapses with no
    beam. Un-gated, beam_dist kept a full-strength Gaussian parked at the track
    end through the whole tail -- the one channel telling the model the laser
    was still ON while the target field cooled."""
    scan_time = 1.0e-3 / 0.8
    man = _straight_manifest(off=scan_time)
    x = np.linspace(-1e-4, 1.1e-3, 24, dtype=np.float32)
    y = np.linspace(-2e-4, 2e-4, 16, dtype=np.float32)
    z = np.linspace(-3e-4, 0.0, 8, dtype=np.float32)
    # 8 frames scanning + 4 frames cooling tail
    t = np.concatenate([np.linspace(0.0, scan_time, 8),
                        scan_time + np.linspace(0.1, 1.0, 4) * 0.25 * scan_time])
    ch = nds._build_path_channels(("beam_dist", "path_dose"), man, x, y, z, t)
    bd = ch["beam_dist"][0, :, 0]                     # (nt, nx, ny, nz)
    assert bd[:8].max() > 0.9, "beam on while scanning"
    assert bd[8:].max() == 0.0, "beam_dist must vanish after laser-off"
    # dose stays frozen (already-gated behaviour, keep it locked)
    assert np.allclose(ch["path_dose"][0, 8:, 0], ch["path_dose"][0, 7, 0])


def test_load_consolidated_npz_attaches_channels_and_requires_manifest(tmp_path):
    nx, ny, nz, nt = 12, 10, 6, 5
    x = np.linspace(-1e-4, 1.1e-3, nx).astype(np.float32)
    y = np.linspace(-2e-4, 2e-4, ny).astype(np.float32)
    z = np.linspace(-3e-4, 0.0, nz).astype(np.float32)
    t = np.linspace(0.0, 1.0e-3 / 0.8, nt).astype(np.float32)
    fields = {
        "x": x, "y": y, "z": z, "time_s": t,
        "power_w": np.float64(150.0), "scan_speed_m_per_s": np.float64(0.8),
        "case_name": np.array("c0"),
        "Temperature": np.random.rand(nt, nx, ny, nz).astype(np.float32),
    }
    np.savez(tmp_path / "c0.npz", **fields)

    # No manifest yet -> requesting path channels must fail loudly (no silent zeros).
    try:
        nds.load_consolidated_npz(
            tmp_path, requested_fields=["Temperature"], time_axis=True,
            avg_window_size=None, moving_frame_steps=None, moving_ref_frame=False,
            path_channels=["beam_x", "path_dose"],
        )
        raised = False
    except ValueError:
        raised = True
    assert raised, "missing manifest should raise"

    (tmp_path / "c0.manifest.json").write_text(json.dumps(_straight_manifest()))
    var = nds.load_consolidated_npz(
        tmp_path, requested_fields=["Temperature"], time_axis=True,
        avg_window_size=None, moving_frame_steps=None, moving_ref_frame=False,
        path_channels=["beam_x", "beam_y", "beam_dist", "path_dose"],
    )
    for k in ("beam_x", "beam_y", "beam_dist", "path_dose"):
        assert k in var, k
        assert var[k].shape == (1, nt, 1, nx, ny, nz), (k, var[k].shape)
    # beam advances in +x over time -> beam_x increases frame to frame.
    bx = var["beam_x"][0, :, 0, 0, 0, 0]
    assert np.all(np.diff(bx) >= -1e-9)


def test_no_path_channels_when_not_requested(tmp_path):
    nx, ny, nz, nt = 8, 6, 4, 3
    fields = {
        "x": np.linspace(0, 1e-3, nx).astype(np.float32),
        "y": np.linspace(0, 4e-4, ny).astype(np.float32),
        "z": np.linspace(-3e-4, 0, nz).astype(np.float32),
        "time_s": np.linspace(0, 1e-3, nt).astype(np.float32),
        "power_w": np.float64(150.0), "scan_speed_m_per_s": np.float64(0.8),
        "case_name": np.array("c0"),
        "Temperature": np.random.rand(nt, nx, ny, nz).astype(np.float32),
    }
    np.savez(tmp_path / "c0.npz", **fields)
    var = nds.load_consolidated_npz(
        tmp_path, requested_fields=["Temperature"], time_axis=True,
        avg_window_size=None, moving_frame_steps=None, moving_ref_frame=False,
        path_channels=None,
    )
    assert not any(k in var for k in nds.PATH_CHANNEL_NAMES)


def _l_shaped_manifest(leg=1.0e-3, speed=0.8, k=81, beam_r=5e-5):
    """Right-angle path: +x for one leg, then +y for one leg, arc-uniform."""
    per = k // 2
    xs = np.concatenate([np.linspace(0, leg, per), np.full(k - per, leg)])
    ys = np.concatenate([np.zeros(per), np.linspace(0, leg, k - per)])
    xy = np.stack([xs, ys], axis=1).tolist()
    scan_time = 2 * leg / speed
    return {
        "input": {"laser": {"scan_speed_m_per_s": speed, "beam_radius_m": beam_r}},
        "derived": {"laser_path_xy_m": xy, "scan_time_s": scan_time,
                    "laser_off_time_s": scan_time},
    }


def test_arc_frame_equals_tangent_frame_on_straight_path():
    man = _straight_manifest()
    x = np.linspace(0.0, 1.0e-3, 21)
    y = np.linspace(-2e-4, 2e-4, 9)
    z = np.array([0.0])
    t = np.linspace(0.0, man["derived"]["scan_time_s"], 13)
    ch = nds._build_path_channels(
        ["beam_s", "beam_n", "beam_s_arc", "beam_n_arc"], man, x, y, z, t)
    # on a straight track the arc frame IS the tangent frame
    assert np.allclose(ch["beam_s_arc"], ch["beam_s"], atol=0.02)
    assert np.allclose(ch["beam_n_arc"], ch["beam_n"], atol=0.02)


def test_arc_frame_follows_the_bend_where_tangent_frame_cannot():
    man = _l_shaped_manifest()
    sigma = 5e-5
    x = np.linspace(0.0, 1.0e-3, 41)
    y = np.linspace(0.0, 1.0e-3, 41)
    z = np.array([0.0])
    t = np.linspace(0.0, man["derived"]["scan_time_s"], 21)
    ch = nds._build_path_channels(
        ["beam_s", "beam_n", "beam_s_arc", "beam_n_arc"], man, x, y, z, t)
    ix = int(np.argmin(np.abs(x - 0.5e-3)))
    iy = int(np.argmin(np.abs(y - 0.0)))
    s_arc = float(ch["beam_s_arc"][0, -1, 0, ix, iy, 0])
    n_arc = float(ch["beam_n_arc"][0, -1, 0, ix, iy, 0])
    n_tan = float(ch["beam_n"][0, -1, 0, ix, iy, 0])
    # the point (0.5 mm, 0) sits ON the first leg: true along-path distance to
    # the beam (at the end of leg 2) is 1.5 mm behind, and it is on the path
    assert abs(s_arc + 1.5e-3 / sigma) < 1.0          # ~ -30 sigma
    assert abs(n_arc) < 0.5                            # on the track
    # the tangent frame at the final heading (+y) calls the same point
    # 10 sigma off-track sideways, which is the failure the arc frame removes
    assert abs(n_tan) > 5.0


def test_beam_veff_reduces_to_speed_on_straight_path():
    man = _straight_manifest(speed=0.8)
    x = np.linspace(0.0, 1.0e-3, 15)
    y = np.linspace(-2e-4, 2e-4, 7)
    t = np.linspace(0.0, man["derived"]["scan_time_s"], 9)
    ch = nds._build_path_channels(["beam_veff"], man, x, y, np.array([0.0]), t)
    v = ch["beam_veff"][0, :, 0, :, :, 0]
    # straight track: kappa = 0, so v_eff is the plain speed everywhere
    assert np.allclose(v[1:-1], 0.8, atol=1e-3)


def _arc_manifest(R=3.0e-4, speed=0.8, k=121, beam_r=5e-5):
    """Quarter circle, counterclockwise (a steady LEFT turn), arc-uniform."""
    th = np.linspace(0.0, np.pi / 2, k)
    xy = np.stack([R * np.cos(th), R * np.sin(th)], axis=1).tolist()
    scan_time = (np.pi / 2) * R / speed
    return {
        "input": {"laser": {"scan_speed_m_per_s": speed, "beam_radius_m": beam_r}},
        "derived": {"laser_path_xy_m": xy, "scan_time_s": scan_time,
                    "laser_off_time_s": scan_time},
    }


def test_beam_veff_asymmetry_on_a_steady_bend():
    man = _arc_manifest()
    x = np.linspace(-1e-4, 4.0e-4, 41)
    y = np.linspace(-1e-4, 4.0e-4, 41)
    t = np.linspace(0.0, man["derived"]["scan_time_s"], 41)
    ch = nds._build_path_channels(["beam_veff", "beam_n"], man, x, y, np.array([0.0]), t)
    f = 20                                          # mid-arc: kappa = 1/R steady
    v = ch["beam_veff"][0, f, 0, :, :, 0]
    n = ch["beam_n"][0, f, 0, :, :, 0]
    # LEFT turn: the left side (n > 0, toward the centre) is the inside and
    # must see the slower sweep
    assert v[n > 1.0].mean() < 0.8 < v[n < -1.0].mean()
    assert v.min() > 0.0                            # clamp keeps it positive


def test_conduction_field_depth_decay_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PINO_CONDUCTION_CACHE", str(tmp_path))
    man = _straight_manifest(speed=0.4)
    man["input"]["material"] = {"properties": {
        "density_kg_m3": 4420.0, "specific_heat_j_kg_k": 670.0,
        "thermal_conductivity_w_m_k": 8.9}}
    man["derived"]["surface_z_m"] = 0.0
    man["case_name"] = "conduction_cache_test"
    x = np.linspace(0.0, 1.0e-3, 24)
    y = np.linspace(-2e-4, 2e-4, 16)
    z = np.linspace(-3.0e-4, 5e-5, 15)              # metal below 0, gas above
    t = np.linspace(0.0, man["derived"]["scan_time_s"], 25)
    import time as _time
    t0 = _time.perf_counter()
    ch = nds._build_path_channels(["conduction_field"], man, x, y, z, t)
    cold = _time.perf_counter() - t0
    g = ch["conduction_field"][0, -1, 0]            # (nx, ny, nz), final frame
    assert np.isfinite(g).all() and g.max() > 0
    # gas planes carry nothing; the field decays into the metal
    assert np.allclose(g[:, :, z > 1e-9], 0.0)      # tolerance: linspace epsilon
    metal = np.where(z <= 1e-9)[0]
    surf, deep = metal.max(), metal.min()
    hot = np.unravel_index(np.argmax(g[:, :, surf]), g[:, :, surf].shape)
    assert g[hot[0], hot[1], surf] > 3 * g[hot[0], hot[1], deep]
    # second build must come from the cache: identical and much faster
    t0 = _time.perf_counter()
    ch2 = nds._build_path_channels(["conduction_field"], man, x, y, z, t)
    warm = _time.perf_counter() - t0
    assert np.allclose(ch2["conduction_field"], ch["conduction_field"])
    assert warm < cold / 3
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_null_laser_off_time_is_treated_as_never_off():
    """`laser_off_time_s: None` must not crash the beam-path builder.

    The default in `_beam_path_over_time` only applies when the key is ABSENT;
    a present-but-null value reached `float(None)` and raised TypeError. Found
    2026-08-18: 24 of the 123 members of
    ti64_straight_path_calibrated_with_keyholes (the `khc_*` keyhole cases)
    record `laser_off_time_s: null`, and both GAOT and RIGNO died at step 0.

    None means "never turns off" here, not "unknown": those cases carry
    `turn_off_at_end: False` and their `scan_time_s` equals the run's
    `effective_end_time_s`, i.e. the beam is on for the whole simulated
    duration and there is no cooling tail to gate.
    """
    import numpy as np

    from lpno.custom_utils import npz_datasource as nds

    x = np.linspace(-2.0e-4, 1.2e-3, 40).astype(np.float64)
    y = np.linspace(-2.0e-4, 2.0e-4, 20).astype(np.float64)
    z = np.linspace(-4.0e-4, 1.0e-4, 8).astype(np.float64)
    t = np.linspace(0.0, 1.0e-3, 6)

    man = {
        "input": {"laser": {"scan_speed_m_per_s": 0.129, "beam_radius_m": 5e-5,
                            "power_w": 120.0, "turn_off_at_end": False}},
        "derived": {"laser_path_xy_m": [[0.0, 0.0], [2.97e-4, 0.0]],
                    "scan_time_s": 2.3e-3,
                    "laser_off_time_s": None},
    }
    ch = nds._build_path_channels(["beam_dist", "path_dose"], man, x, y, z, t)
    assert ch["beam_dist"].shape[1] == t.size
    # never off => the footprint is live in the final frame, not gated to zero
    assert float(ch["beam_dist"][0, -1].max()) > 0.5


def test_moving_frame_keeps_every_trajectory_channel(tmp_path, capsys):
    """Steady/moving-frame arms inherit the SAME input keys as their lab partners.

    User decision 2026-08-19: keep all fields even where one goes redundant, so the
    three frame treatments (lab / moving-ref / steady) stay comparable and
    warm-startable from one another. `_reject_lab_frame_keys_in_moving_frame` used to
    refuse this outright, on a docstring claim that the datasource "never builds"
    trajectory channels in the moving frame. That is stale for the consolidated-NPZ
    path: they are built at npz_datasource.py:1230 and then cropped and shifted with
    every other key by `_apply_moving_frame`, so they stay consistent with the fields.
    """
    from lpno.models.pipeline.config_loading import (
        _reject_lab_frame_keys_in_moving_frame,
    )

    keys = ["V_scan", "P", "x", "y", "z", "beam_x", "beam_y", "beam_dist",
            "path_dose", "path_dose_recent"]
    _reject_lab_frame_keys_in_moving_frame(keys, moving_ref_frame=True)   # must not raise
    warning = capsys.readouterr().out
    assert "beam_x" in warning and "VTK" in warning, warning

    nx, ny, nz, nt = 40, 10, 6, 40
    np.savez(
        tmp_path / "c0.npz",
        x=np.linspace(-1e-4, 1.1e-3, nx).astype(np.float32),
        y=np.linspace(-2e-4, 2e-4, ny).astype(np.float32),
        z=np.linspace(-3e-4, 0.0, nz).astype(np.float32),
        time_s=np.linspace(0.0, 1.0e-3 / 0.8, nt).astype(np.float32),
        power_w=np.float64(150.0), scan_speed_m_per_s=np.float64(0.8),
        case_name=np.array("c0"),
        Temperature=np.random.rand(nt, nx, ny, nz).astype(np.float32),
    )
    (tmp_path / "c0.manifest.json").write_text(json.dumps(_straight_manifest()))

    channels = ["beam_x", "beam_y", "beam_dist", "path_dose", "path_dose_recent"]
    var = nds.load_consolidated_npz(
        tmp_path, requested_fields=["Temperature"], time_axis=True,
        avg_window_size=30, moving_frame_steps=30, moving_ref_frame=True,
        path_channels=channels,
    )
    for k in channels:
        assert k in var, f"{k} dropped by the moving frame"
        assert var[k].shape == var["Temperature"].shape, (k, var[k].shape)
