"""Multi-beam conditioning channels: per-beam paths, power-weighted combination.

mbc40 cases carry 2-4 beams, each with its own path/power/turn-off, recovered
from the prepin velocity tables into `derived.beams[i].path_xy_m/path_times_s`
(the single-path `derived.laser_path_xy_m` is degenerate for these cases).

Combination rules, chosen so a single beam reproduces the old channels exactly:
- beam_dist:   sum_b w_b * gauss_b, with w_b = P_b / sum(P)  (power fraction)
- beam_x/y:    power-weighted centroid of the ACTIVE beams' centres
- path_dose*:  accumulate the combined footprint (same accumulators as before)
- a beam past its `turn_off_time_s` contributes nothing (weight 0), and the
  centroid re-normalises over the still-active beams.
"""

from __future__ import annotations

import numpy as np
import pytest

from lpno.custom_utils.npz_datasource import (
    _beam_path_over_time,
    _multibeam_specs,
)


def _single(manifest_xy=((0.0, 0.0), (1.0e-3, 0.0)), speed=1.0):
    return {
        "derived": {"laser_path_xy_m": [list(p) for p in manifest_xy],
                     "laser_off_time_s": 1.0e-3},
        "input": {"laser": {"scan_speed_m_per_s": speed}},
    }


def _multi():
    beams = [
        {"index": 0, "power_w": 60.0, "path_xy_m": [[0.0, 0.0], [1.0e-3, 0.0]],
         "path_times_s": [0.0, 1.0e-3], "turn_off_time_s": 1.0e-3,
         "scan_speed_m_per_s": 1.0},
        {"index": 1, "power_w": 40.0, "path_xy_m": [[0.0, 4.0e-4], [0.0, 4.0e-4]],
         "path_times_s": [0.0, 5.0e-4], "turn_off_time_s": 5.0e-4,
         "scan_speed_m_per_s": 0.0},
    ]
    return {
        "derived": {"beams": beams, "laser_path_xy_m": [[0.0, 0.0], [0.0, 0.0]],
                     "laser_off_time_s": 1.0e-3},
        "input": {"laser": {"scan_speed_m_per_s": 0.5,
                             "beams": [{}, {}]}},
    }


def test_single_beam_manifest_yields_no_multibeam_specs():
    assert _multibeam_specs(_single()) is None


def test_multibeam_specs_carry_paths_powers_and_offs():
    specs = _multibeam_specs(_multi())
    assert specs is not None and len(specs) == 2
    t = np.array([0.0, 2.5e-4, 7.5e-4])
    c0, _, _ = _beam_path_over_time(specs[0].manifest, t)
    c1, _, _ = _beam_path_over_time(specs[1].manifest, t)
    assert c0[1, 0] == pytest.approx(2.5e-4)     # beam 0 moving in +x
    assert c1[1, 1] == pytest.approx(4.0e-4)     # beam 1 stationary at y=0.4mm
    assert specs[0].power_fraction == pytest.approx(0.6)
    assert specs[1].power_fraction == pytest.approx(0.4)
    assert specs[1].turn_off_time_s == pytest.approx(5.0e-4)


def test_beam_weights_at_time_gate_on_turn_off():
    specs = _multibeam_specs(_multi())
    from lpno.custom_utils.npz_datasource import _beam_weights_at
    t = np.array([2.0e-4, 7.5e-4])
    w = _beam_weights_at(specs, t)          # (nt, nbeams)
    assert w[0] == pytest.approx([0.6, 0.4])
    assert w[1] == pytest.approx([0.6, 0.0])  # beam 1 off after 0.5 ms


def _grid():
    x = np.linspace(0, 1.0e-3, 21)
    y = np.linspace(0, 8.0e-4, 17)
    z = np.linspace(0, 1.0e-4, 3)
    t = np.linspace(0, 1.0e-3, 5)
    return x, y, z, t


def _mb_manifest():
    m = _multi()
    m["input"]["laser"]["beam_radius_m"] = 5.0e-5
    return m


def test_beam_dist_shows_both_beams():
    from lpno.custom_utils.npz_datasource import _build_path_channels
    x, y, z, t = _grid()
    out = _build_path_channels(["beam_dist", "beam_x", "beam_y"], _mb_manifest(), x, y, z, t)
    bd = out["beam_dist"][0, 0, 0, :, :, 0]      # frame 0
    # beam 0 at (0,0) with w=0.6; beam 1 at (0, 0.4mm) with w=0.4
    i0, j0 = 0, 0
    j1 = int(np.argmin(np.abs(y - 4.0e-4)))
    assert bd[i0, j0] == pytest.approx(0.6, abs=0.05)
    assert bd[i0, j1] == pytest.approx(0.4, abs=0.05)


def test_beam1_disappears_after_its_turn_off():
    from lpno.custom_utils.npz_datasource import _build_path_channels
    x, y, z, t = _grid()
    out = _build_path_channels(["beam_dist"], _mb_manifest(), x, y, z, t)
    j1 = int(np.argmin(np.abs(y - 4.0e-4)))
    late = out["beam_dist"][0, -1, 0, 0, j1, 0]   # t=1ms > beam1 off at 0.5ms
    assert late < 0.05


def test_path_dose_accumulates_both_beams():
    from lpno.custom_utils.npz_datasource import _build_path_channels
    x, y, z, t = _grid()
    out = _build_path_channels(["path_dose"], _mb_manifest(), x, y, z, t)
    pd = out["path_dose"][0, -1, 0]               # final frame
    j1 = int(np.argmin(np.abs(y - 4.0e-4)))
    assert pd[0, j1, 0] > 0, "stationary beam 1 left no dose"
    assert pd[-1, 0, 0] > 0 or pd[:, 0, 0].max() > 0, "moving beam 0 left no dose"


def test_single_beam_channels_are_bit_identical_to_before():
    """Regression: a single-beam manifest must not enter the multibeam path."""
    from lpno.custom_utils.npz_datasource import _build_path_channels
    x, y, z, t = _grid()
    m = _single()
    m["input"]["laser"]["beam_radius_m"] = 5.0e-5
    out = _build_path_channels(["beam_dist", "beam_x"], m, x, y, z, t)
    c, _, _ = _beam_path_over_time(m, t)
    assert np.allclose(out["beam_x"][0, :, 0, 0, 0, 0], c[:, 0])
    bd0 = out["beam_dist"][0, 0, 0, :, :, 0]
    i_c = int(np.argmin(np.abs(x - c[0, 0])))
    assert bd0[i_c, 0] == pytest.approx(1.0, abs=1e-5)


def test_beam_centres_helper_is_multibeam_aware():
    """The lazy store calls _beam_centres_over_time DIRECTLY, bypassing
    _build_path_channels; it must combine beams rather than fall through to the
    single-path branch (which raised KeyError: 'scan_speed_m_per_s' because
    mbc40 manifests carry per-beam speeds only)."""
    from lpno.custom_utils.npz_datasource import _beam_centres_over_time
    m = _mb_manifest()
    m["input"]["laser"].pop("scan_speed_m_per_s", None)   # as the real manifests are
    t = np.array([0.0, 2.5e-4, 7.5e-4])
    c = _beam_centres_over_time(m, t)
    assert c.shape == (3, 2)
    # frame 0: beams at (0,0) w=.6 and (0,4e-4) w=.4 -> centroid y = .4*4e-4
    assert c[0, 1] == pytest.approx(0.4 * 4.0e-4, rel=1e-6)
    # after beam 1's turn-off only beam 0 remains -> centroid is beam 0
    assert c[2, 1] == pytest.approx(0.0, abs=1e-9)


def test_conduction_cache_key_distinguishes_beams(tmp_path, monkeypatch):
    """Two beams of one case must not share a conduction cache entry.

    The key hashed case name + grid + times only, so per-beam superposition
    calls all hit the same file: beam 1 read beam 0's cached field and the
    multibeam conduction channel was silently N x beam 0.
    """
    import numpy as np
    from lpno.custom_utils.npz_datasource import _conduction_field
    monkeypatch.setenv("PINO_CONDUCTION_CACHE", str(tmp_path))
    m = {"case_name": "c", "derived": {"surface_z_m": 1.0e-4},
         "material": {"properties": {}}, "input": {"laser": {}}}
    x = np.linspace(0, 3e-4, 8); y = np.linspace(0, 2e-4, 6)
    z = np.linspace(0, 1e-4, 4); t = np.linspace(0, 1e-3, 3)
    dt = np.gradient(t); on = np.ones_like(t)
    c0 = np.tile([[0.0, 0.0]], (3, 1))
    c1 = np.tile([[2.0e-4, 1.0e-4]], (3, 1))
    f0 = _conduction_field(m, x, y, z, t, c0, dt, on, 5e-5)
    f1 = _conduction_field(m, x, y, z, t, c1, dt, on, 5e-5)
    assert not np.allclose(f0, f1), "different beam paths returned the same cached field"
    assert len(list(tmp_path.glob("conduction_*.npy"))) == 2, "beams shared one cache file"


def test_multibeam_manifest_without_merged_paths_fails_loudly():
    """A >=2-beam manifest whose beams lack paths must raise, not fall through.

    Live instance: eval calc 46611 (2026-08-14) read a runtime cache staged
    minutes before aiida-pino's second-sidecar-writer merge fix (4649a5a), so
    `derived.beams` carried no `path_xy_m`/`path_times_s` and the single-beam
    fallback died three frames deep on `KeyError: 'scan_speed_m_per_s'` (the
    degenerate top-level stub has no speed). The named error points at the
    stale-cache remedy instead of leaving the next reader a puzzle -- and it
    also guards the silent-worse case where the stub WOULD interpolate.
    """
    m = _multi()
    for b in m["derived"]["beams"]:
        b.pop("path_xy_m")
        b.pop("path_times_s")
    with pytest.raises(RuntimeError, match="per-beam paths"):
        _multibeam_specs(m)


def test_multibeam_manifest_with_zero_total_power_fails_loudly():
    """All-zero beam powers cannot be weighted; falling back would be wrong too."""
    m = _multi()
    for b in m["derived"]["beams"]:
        b["power_w"] = 0.0
    with pytest.raises(RuntimeError, match="power"):
        _multibeam_specs(m)
