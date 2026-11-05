"""Unit test for scripts/build_steady_split.collapse_case_to_steady.

Checks the beam-centered co-moving collapse: a hot spot that moves with the laser
must land, sharply, at a fixed column (LASER_COL) in the steady frame regardless
of how many cells/frame the laser advances -- the property the old integer-shift
version broke on the time-even material data. Also checks output shape and that
the existing lazy store reads the result. Numpy-only, no GPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from lpno.models.pipeline.npz_lazy_store import NpzDirStore
from lpno.custom_utils import npz_datasource as nds
from lpno.custom_utils.npz_datasource import _beam_centres_over_time

_SPEC = importlib.util.spec_from_file_location(
    "build_steady_split",
    Path(__file__).resolve().parents[1] / "scripts" / "build_steady_split.py",
)
_bss = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_bss)

_PROPS = dict(rho=4124.66, cp=752.99, k=78.43, latent=248130.7, Tl=1388.33, A=0.38554)
# grid: x in [0, 60*dx), laser scans cells ~10->40; a window of 31 with the laser
# at col 15 fits for the last 20 frames.
NX, NY, NZ, NT = 60, 8, 6, 40
DX = 1e-5
X0, X1 = 1e-4, 4e-4          # laser path endpoints (metres)
SPEED = 0.3


def _manifest():
    return {
        "input": {"laser": {"scan_speed_m_per_s": SPEED, "beam_radius_m": 5e-5}},
        "derived": {"laser_path_xy_m": [[X0, 0.0], [X1, 0.0]],
                    "scan_time_s": (X1 - X0) / SPEED, "laser_off_time_s": 1e9},
        "material": {
            "properties": {"density_kg_m3": _PROPS["rho"], "specific_heat_j_kg_k": _PROPS["cp"],
                           "thermal_conductivity_w_m_k": _PROPS["k"], "latent_heat_j_kg": _PROPS["latent"],
                           "liquidus_temperature_k": _PROPS["Tl"]},
            "optical": {"normal_absorptivity": _PROPS["A"]},
        },
    }


def _write_moving_hotspot(tmp_path):
    x = (np.arange(NX) * DX).astype(np.float32)
    t = np.linspace(0.0, (X1 - X0) / SPEED, NT).astype(np.float64)
    man = _manifest()
    cell = np.round((_beam_centres_over_time(man, t)[:, 0] - x[0]) / DX).astype(int)
    # Temperature: ambient 300, a 2000 K spike at the laser cell each frame.
    temp = np.full((NT, NX, NY, NZ), 300.0, np.float32)
    for i in range(NT):
        temp[i, cell[i], :, :] = 2000.0
    fields = {
        "x": x, "y": np.linspace(-2e-4, 2e-4, NY).astype(np.float32),
        "z": np.linspace(-3e-4, 0.0, NZ).astype(np.float32),
        "time_s": t, "power_w": np.float64(120.0), "scan_speed_m_per_s": np.float64(SPEED),
        "case_name": np.array("mlhs_000"),
        "Temperature": temp,
        "Velocity": np.random.rand(NT, NX, NY, NZ, 3).astype(np.float32),
        "Fraction Of Fluid": np.random.rand(NT, NX, NY, NZ).astype(np.float32),
        "Pressure": np.random.rand(NT, NX, NY, NZ).astype(np.float32),
    }
    np.savez(tmp_path / "mlhs_000.npz", **fields)
    (tmp_path / "mlhs_000.manifest.json").write_text(json.dumps(man))
    return cell


def test_moving_hotspot_is_pinned_to_laser_col(tmp_path):
    cell = _write_moving_hotspot(tmp_path)
    fr, W, col = 20, 31, 15
    out = _bss.collapse_case_to_steady(tmp_path / "mlhs_000.npz", frame_steps=fr, out_width=W, laser_col=col)

    assert out["time_s"].shape == (1,)
    assert out["x"].shape == (W,)
    assert out["Temperature"].shape == (1, W, NY, NZ)
    assert out["Velocity"].shape == (1, W, NY, NZ, 3)

    # the moving hot spot must collapse onto the laser column, sharply.
    Tx = out["Temperature"][0, :, 0, 0]
    assert int(np.argmax(Tx)) == col, (int(np.argmax(Tx)), col)
    # sharp: the laser column is the clear peak, its immediate neighbours much cooler,
    # and the far background stays near ambient (not smeared across the window).
    assert Tx[col] > 1500.0
    assert Tx[0] < 500.0 and Tx[-1] < 500.0
    # laser-relative x: column `col` is x == 0.
    assert abs(float(out["x"][col])) < 1e-9
    # sanity: the laser genuinely moved across cells (else the test proves nothing).
    assert np.ptp(cell[NT - fr:]) >= 5

    # beam_dist: co-moving footprint, peak 1.0 at the laser column, spatially varying.
    assert out["beam_dist"].shape == (1, W, NY, NZ)
    bd = out["beam_dist"][0, :, :, 0]
    assert bd.max() > 0.8   # Gaussian peak (<1 unless a grid point hits the beam center exactly)
    assert int(np.argmax(bd[:, NY // 2])) == col   # peak x is the laser column
    assert bd.std() > 0.05                          # genuinely spatially varying

    # path_dose / path_dose_recent: co-moving history channels, ASYMMETRIC about the
    # laser (behind >> ahead), unlike the symmetric beam_dist -- they encode the
    # trailing scanned/heated region.
    for ch in ('path_dose', 'path_dose_recent'):
        assert out[ch].shape == (1, W, NY, NZ), (ch, out[ch].shape)
        prof = out[ch][0, :, NY // 2, NZ - 1]       # x-profile near the surface
        behind, ahead = prof[:col].mean(), prof[col + 1:].mean()  # laser scans +x
        assert behind > ahead + 1e-6, (ch, float(behind), float(ahead))


def test_steady_npz_round_trips_through_lazy_store(tmp_path):
    _write_moving_hotspot(tmp_path)
    fr, W, col = 20, 31, 15
    dst = tmp_path / "steady"
    dst.mkdir()
    out = _bss.collapse_case_to_steady(tmp_path / "mlhs_000.npz", frame_steps=fr, out_width=W, laser_col=col)
    np.savez_compressed(dst / "mlhs_000.npz", **out)
    (dst / "mlhs_000.manifest.json").write_text((tmp_path / "mlhs_000.manifest.json").read_text())

    store = NpzDirStore(
        dst,
        requested_fields=["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"],
        path_channels=["beam_dist", "path_dose", "path_dose_recent"],
        material_channels=list(nds.MATERIAL_CHANNEL_NAMES),
        cache_size=2,
    )
    assert store.sample_count() == 1
    assert store.time_steps(0) == 1
    assert store.spatial_shape() == (W, NY, NZ)
    rc = store.read_case(0, time_idx=0)
    assert rc["Temperature"].shape == (1, 1, W, NY, NZ)
    assert np.isclose(float(rc["mat_A"].flat[0]), _PROPS["A"], atol=1e-4)
    # the store reads the PRECOMPUTED co-moving path channels from the NPZ (not
    # rebuilt from the manifest): beam_dist peaks ~1 at the laser column, and the
    # dose channels are present and spatially varying.
    assert rc["beam_dist"].shape == (1, 1, W, NY, NZ)
    assert float(rc["beam_dist"].max()) > 0.8
    assert float(rc["beam_dist"].std()) > 0.05
    for ch in ('path_dose', 'path_dose_recent'):
        assert rc[ch].shape == (1, 1, W, NY, NZ)
        assert float(rc[ch].std()) > 1e-4
