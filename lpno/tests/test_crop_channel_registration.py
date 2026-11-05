"""The z-crop must not change any input channel on the retained planes.

Why this exists: run 82773 (RIGNO d18 16ch, the first random-path arm trained under
the post-2026-08-26 `PINO_LAZY_CROP_Z_TOP=10` default) collapsed -- 60x worse on
beam-box and 20x on alpha than m67210 at the same step -- and the first suspicion
was mis-registered channels: a dose or beam field built against raw z-indices would
land 10 cells below the surface and read exactly like that failure. The audit
(2026-09-01) found the opposite: 15 of 16 channels bit-identical on the shared
planes, because every builder receives the ALREADY-cropped coordinate arrays. This
test pins that property so a future builder that indexes raw z breaks loudly.

The one intended exception is `h_norm_local`: its `_soak_field` treats the z extents
as Dirichlet faces, so cropping moves the top face from the gas top to the surface
and shifts the channel by ~0.8%. That is a boundary-semantics change, tolerated with
a loose bound here, not exempted.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lpno.models.pipeline.npz_lazy_store import NpzDirStore

NT, NX, NY, NZ = 4, 8, 6, 8
GAS = 2                       # top z cells: the crop target


@pytest.fixture()
def case_dir(tmp_path):
    x = np.linspace(0.0, 7e-4, NX, dtype=np.float32)
    y = np.linspace(0.0, 5e-4, NY, dtype=np.float32)
    z = np.linspace(0.0, 7e-5, NZ, dtype=np.float32)
    surface = float(z[NZ - GAS - 1])          # last metal plane
    t = np.linspace(0.0, 3e-4, NT, dtype=np.float32)
    rng = np.random.default_rng(7)
    T = rng.uniform(300, 3000, (NT, NX, NY, NZ)).astype(np.float32)
    F = rng.uniform(0, 1, (NT, NX, NY, NZ)).astype(np.float32)
    np.savez(tmp_path / "case_000.npz", x=x, y=y, z=z, time_s=t,
             power_w=np.float32(100.0), scan_speed_m_per_s=np.float32(0.5),
             case_name=np.str_("case_000"),
             **{"Temperature": T, "Fraction Of Fluid": F})
    manifest = {
        "case_name": "case_000",
        "input": {"laser": {"beam_radius_m": 5e-5,
                            "path": {"type": "line", "length_m": 5e-4},
                            "scan_speed_m_per_s": 0.5, "power_w": 100.0},
                  "initial": {"temperature_k": 293.0}},
        "derived": {"surface_z_m": surface,
                    "beam_focus_start_m": [0.0, 0.0, 0.0],
                    "laser_path_xy_m": [[1e-4, 2.5e-4], [6e-4, 2.5e-4]],
                    "laser_off_time_s": float(t[-1])},
    }
    (tmp_path / "case_000.manifest.json").write_text(json.dumps(manifest))
    return tmp_path


KEYS = ["beam_x", "beam_y", "beam_dist", "path_dose", "path_dose_recent",
        "preheat_T0", "beam_shape_uniform", "focus_factor", "h_norm_local"]


def _read(root, crop, stride=(1, 1, 1)):
    store = NpzDirStore(root, requested_fields=["Temperature", "Fraction Of Fluid"],
                        path_channels=KEYS, spatial_stride=stride, crop_z_top=crop)
    return store.read_case(0, time_idx=NT - 2)


def test_crop_preserves_every_channel_on_the_retained_planes(case_dir):
    full, cropped = _read(case_dir, 0), _read(case_dir, GAS)
    checked = 0
    for k in sorted(set(full) & set(cropped)):
        va, vb = np.asarray(full[k]), np.asarray(cropped[k])
        if va.ndim < 3:
            continue
        overlap = va[..., : vb.shape[-1]]
        if k == "h_norm_local":
            # boundary-semantics shift, small by construction
            assert np.allclose(overlap, vb, rtol=0.05, atol=0.05), k
        else:
            np.testing.assert_array_equal(overlap, vb, err_msg=k)
        checked += 1
    assert checked >= 10          # the guard is vacuous if keys stopped resolving


def test_crop_composes_with_stride_in_raw_cells(case_dir):
    # crop first (raw cells), stride second: 8 -> 6 -> 3, never 8 -> 4 -> 3
    r = _read(case_dir, GAS, stride=(1, 1, 2))
    assert np.asarray(r["Temperature"]).shape[-1] == (NZ - GAS + 1) // 2
    full, cropped = _read(case_dir, 0, (1, 1, 2)), _read(case_dir, GAS, (1, 1, 2))
    ta = np.asarray(full["Temperature"])
    tb = np.asarray(cropped["Temperature"])
    np.testing.assert_array_equal(ta[..., : tb.shape[-1]], tb)
