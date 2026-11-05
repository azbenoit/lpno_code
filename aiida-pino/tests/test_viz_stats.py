"""Melt-pool stats + plot rendering tests on hand-built synthetic cases.

The synthetic NPZ has an explicit z axis with metal (z < 0) and gas planes and
hand-carved Fraction Of Fluid / Melt Region masks, so every metric has a known
expected value (notebook `case_metrics` semantics).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pytest_plugins = ["aiida.tools.pytest_fixtures"]

NX, NY, NZ = 8, 7, 6  # z: 4 metal planes (z<0) + 2 gas planes
NT = 3


def _write_synthetic(
    tmp_path, name="viz_case", *,
    gas_depth_cells=2, gas_ylen=3, melt_cells=10, tmax=2500.0, gas_xlen=2, gas_x0=3, gas_y0=2,
    melt_xlen=None, melt_ylen=None,
):
    x = np.linspace(0.0, 7e-4, NX).astype(np.float32)
    y = np.linspace(-3e-4, 3e-4, NY).astype(np.float32)
    z = np.array([-4e-4, -3e-4, -2e-4, -1e-4, 1e-4, 2e-4], dtype=np.float32)  # 4 metal, 2 gas
    nzm = 4
    F = np.ones((NT, NX, NY, NZ), dtype=np.float32)
    F[:, :, :, nzm:] = 0.0  # gas planes are void everywhere
    # Carve the depression at the LAST frame (peak frame = 2): interior columns
    # starting at (gas_x0, gas_y0), the top `gas_depth_cells` metal planes.
    for d in range(gas_depth_cells):
        F[NT - 1, gas_x0 : gas_x0 + gas_xlen, gas_y0 : gas_y0 + gas_ylen, nzm - 1 - d] = 0.0
    M = np.zeros((NT, NX, NY, NZ), dtype=np.float32)
    # melt blob at frame 1 (its own peak frame), top metal plane. A
    # melt_xlen/melt_ylen pair carves a clean rectangle instead, so the
    # ellipse-fit melt radii have a hand-computable expected value.
    if melt_xlen and melt_ylen:
        M[1, :melt_xlen, :melt_ylen, nzm - 1] = 1.0
    else:
        flat = np.zeros((NX, NY), dtype=np.float32).reshape(-1)
        flat[:melt_cells] = 1.0
        M[1, :, :, nzm - 1] = flat.reshape(NX, NY)
    T = np.full((NT, NX, NY, NZ), 300.0, dtype=np.float32)
    T[1, 0, 0, 0] = tmax
    path = tmp_path / f"{name}.npz"
    np.savez_compressed(
        path,
        x=x, y=y, z=z,
        time_s=np.linspace(0, 1e-3, NT).astype(np.float64),
        power_w=np.float64(150.0), scan_speed_m_per_s=np.float64(0.5),
        case_name=np.str_(name),
        **{"Fraction Of Fluid": F, "Melt Region": M, "Temperature": T},
    )
    return path


def _sim(tmp_path, aiida_profile_clean, **kw):
    from aiida_pino.data import PinoSimulationData

    path = _write_synthetic(tmp_path, **kw)
    node = PinoSimulationData.from_npz_file(path, metadata={"source_folder_name": "vizset", "source_group": ""})
    node.store()
    return node


def test_case_metrics_known_values(tmp_path, aiida_profile_clean):
    from aiida_pino.console import viz

    sim = _sim(tmp_path, aiida_profile_clean, gas_depth_cells=3, gas_xlen=1, gas_ylen=2, melt_cells=10, tmax=2500.0)
    st = viz.compute_case_metrics(sim)
    assert st["nz_metal"] == 4
    assert st["peak_frame"] == NT - 1
    assert st["vol_cells"] == 1 * 2 * 3  # x-span 1 * ylen 2 * depth 3
    assert st["depth_cells"] == 3
    assert st["xlen_cells"] == 1
    assert st["ylen_cells"] == 2
    assert st["melt_vol_cells"] == 10
    assert st["melt_peak_frame"] == 1
    assert st["melt_depth_cells"] == 1
    assert st["tmax_k"] == pytest.approx(2500.0)
    dz_um = st["dz_m"] * 1e6
    # Ellipse fit of the 1x2 footprint: minor-axis variance is the dz^2/12
    # per-cell term alone; major axis adds the two-point spread 0.25*dz^2.
    r_min_expect = 2.0 * (dz_um**2 / 12) ** 0.5
    r_max_expect = 2.0 * (dz_um**2 * (0.25 + 1 / 12)) ** 0.5
    assert st["cavity_rmin_um"] == pytest.approx(r_min_expect)
    assert st["cavity_rmax_um"] == pytest.approx(r_max_expect)
    # chimney cavity: depth 300 um / r_min ~57.7 um => mean aspect ~5.2 > 1,
    # and 300 um clears the 2.1 x 50 um beam-radius depth gate
    assert st["aspect_max"] == pytest.approx(300.0 / r_min_expect)
    assert st["aspect_mean"] == pytest.approx(300.0 / r_min_expect)  # one cavity frame
    assert st["aspect_max_frame"] == NT - 1
    assert st["cavity_depth_max_um"] == pytest.approx(300.0)
    assert st["keyhole_min_depth_um"] == pytest.approx(105.0)
    assert st["regime"] == "keyhole"
    assert st["melt_vol_um3"] == pytest.approx(10 * dz_um**3)


def test_regime_branches(tmp_path, aiida_profile_clean):
    from aiida_pino.console import viz

    lof = _sim(tmp_path, aiida_profile_clean, name="lof", gas_depth_cells=1, gas_ylen=3)
    assert viz.compute_case_metrics(lof)["regime"] == "LoF"
    # Wide shallow cavity: 5x5 interior footprint, depth 2 cells (200 um) =>
    # r_min ~289 um so mean aspect ~0.69 < 1 => conduction.
    cond = _sim(tmp_path, aiida_profile_clean, name="cond",
                gas_depth_cells=2, gas_xlen=5, gas_x0=1, gas_ylen=5, gas_y0=1)
    st = viz.compute_case_metrics(cond)
    assert st["cavity_rmin_um"] == pytest.approx(2.0 * (100.0**2 * (2 + 1 / 12)) ** 0.5, rel=1e-3)
    assert st["aspect_mean"] < viz.KEYHOLE_ASPECT_MEAN
    assert st["regime"] == "conduction"

    # Narrow but not deep *relative to the beam*: aspect passes, the absolute
    # depth gate (2.5 beam radii) does not, so it is not a keyhole.
    shallow = _sim(tmp_path, aiida_profile_clean, name="shallow",
                   gas_depth_cells=2, gas_xlen=1, gas_ylen=1)
    shallow.base.extras.set("beam_radius", 2e-4)  # 200 um beam => 420 um gate
    st = viz.compute_case_metrics(shallow)
    assert st["aspect_mean"] > viz.KEYHOLE_ASPECT_MEAN
    assert st["keyhole_min_depth_um"] == pytest.approx(420.0)
    assert st["cavity_depth_max_um"] == pytest.approx(200.0)
    assert st["cavity_depth_max_um"] < st["keyhole_min_depth_um"]
    assert st["regime"] == "conduction"


# Four cases from original_straight_path_data whose regime was
# fixed by eye on 2026-07-23. Their measured (aspect_mean, max cavity
# depth) are what the depth gate has to separate; all four have a 50 um beam
# and an aspect_mean above 1, so the gate is the only discriminator.
CALIBRATION_ANCHORS = [
    # pk,   case,               aspect_mean, cavity_depth_max_um, expected
    (2065, "130W_0561mps", 1.7272, 120.0, "keyhole"),
    (2033, "110W_0589mps", 1.2658, 90.0, "conduction"),
    (179, "90W_0323mps", 1.6068, 120.0, "keyhole"),
    (64, "70W_0298mps", 1.0366, 70.0, "conduction"),
]


@pytest.mark.parametrize("pk,case,aspect_mean,depth_max_um,expected", CALIBRATION_ANCHORS)
def test_keyhole_gate_matches_visual_calibration(pk, case, aspect_mean, depth_max_um, expected):
    """The depth gate must reproduce the hand-labelled regimes of the straight-path anchors."""
    from aiida_pino.console import viz

    beam_radius_um = 50.0
    got = viz.classify_regime(
        depth_cells=4,
        aspect_mean=aspect_mean,
        cavity_depth_max_um=depth_max_um,
        min_depth_um=viz.KEYHOLE_MIN_DEPTH_RADII * beam_radius_um,
    )
    assert got == expected, f"pk {pk} ({case}) classified {got}, expected {expected}"


def test_keyhole_gate_keeps_shallow_random_path_tracks_conduction():
    """Fixed-P,V random-path tracks bottom out at 100 um and stay conduction.

    Their speckled depressions read aspect 1.05-1.42 (nominally keyhole-like)
    but are visually conduction, so only the gate excludes them. A gate at or
    below 100 um would flip the whole family.
    """
    from aiida_pino.console import viz

    assert viz.KEYHOLE_MIN_DEPTH_RADII * 50.0 > 100.0
    assert viz.classify_regime(
        depth_cells=4, aspect_mean=1.42, cavity_depth_max_um=100.0,
        min_depth_um=viz.KEYHOLE_MIN_DEPTH_RADII * 50.0,
    ) == "conduction"


def test_melt_pool_and_cavity_width_length(tmp_path, aiida_profile_clean):
    """Melt pool gets the same ellipse-fit radii the cavity already had."""
    from aiida_pino.console import viz

    # 2x4 melt rectangle, 3x1 cavity -- both hand-computable.
    sim = _sim(tmp_path, aiida_profile_clean, name="mp",
               gas_depth_cells=2, gas_xlen=3, gas_ylen=1, melt_xlen=2, melt_ylen=4)
    st = viz.compute_case_metrics(sim)
    c = st["dz_m"] * 1e6  # cubic cells

    # A rectangle of n cells along an axis fits semi-axis c*n/sqrt(3):
    # discrete variance c^2(n^2-1)/12 plus the c^2/12 per-cell term = c^2n^2/12.
    assert st["melt_vol_cells"] == 2 * 4
    assert st["melt_rmin_um"] == pytest.approx(c * 2 / 3**0.5)
    assert st["melt_rmax_um"] == pytest.approx(c * 4 / 3**0.5)
    assert st["melt_xlen_cells"] == 2
    assert st["melt_ylen_cells"] == 4

    # width/length are full axes, i.e. twice the fitted radii
    assert st["melt_width_um"] == pytest.approx(2 * st["melt_rmin_um"])
    assert st["melt_length_um"] == pytest.approx(2 * st["melt_rmax_um"])
    assert st["cavity_width_um"] == pytest.approx(2 * st["cavity_rmin_um"])
    assert st["cavity_length_um"] == pytest.approx(2 * st["cavity_rmax_um"])
    assert st["cavity_rmin_um"] == pytest.approx(c * 1 / 3**0.5)
    assert st["cavity_rmax_um"] == pytest.approx(c * 3 / 3**0.5)


def test_melt_stats_absent_without_melt_region(tmp_path, aiida_profile_clean):
    """No Melt Region alias => no melt width/length keys, not zeros."""
    from aiida_pino.console import viz
    from aiida_pino.data import PinoSimulationData

    path = _write_synthetic(tmp_path, name="nomelt")
    with np.load(path) as zf:
        payload = {k: zf[k] for k in zf.files if k != "Melt Region"}
    p2 = path.with_name("nomelt2.npz")
    np.savez_compressed(p2, **payload)
    sim = PinoSimulationData.from_npz_file(p2, metadata={"source_folder_name": "vizset", "source_group": ""})
    sim.store()
    st = viz.compute_case_metrics(sim)
    assert "melt_width_um" not in st
    assert "melt_rmin_um" not in st


def test_interior_only_gas_ignored(tmp_path, aiida_profile_clean):
    from aiida_pino.console import viz
    from aiida_pino.data import PinoSimulationData

    path = _write_synthetic(tmp_path, name="edgecase", gas_depth_cells=0)
    with np.load(path) as zf:
        payload = {k: zf[k] for k in zf.files}
    F = payload["Fraction Of Fluid"]
    F[:, 0, :, 0:4] = 0.0  # gas along the x=0 boundary column: must not count
    p2 = path.with_name("edgecase2.npz")
    np.savez_compressed(p2, **payload)
    sim = PinoSimulationData.from_npz_file(p2, metadata={"source_folder_name": "vizset", "source_group": ""})
    sim.store()
    st = viz.compute_case_metrics(sim)
    assert st["vol_cells"] == 0
    assert st["regime"] == "LoF"


def test_stats_cached_in_extras(tmp_path, aiida_profile_clean):
    from aiida_pino.console import viz

    sim = _sim(tmp_path, aiida_profile_clean, name="cache")
    st1 = viz.get_or_compute_case_stats(sim)
    stored = sim.base.extras.get(viz.STATS_EXTRA_KEY, None)
    assert set(stored) == set(st1)
    for k, v in st1.items():
        # extras round-trip through JSON at ~14 significant digits
        assert stored[k] == pytest.approx(v) if isinstance(v, float) else stored[k] == v
    # cached path returns the stored dict without recompute
    assert viz.get_or_compute_case_stats(sim) == stored


def test_render_case_frame_png_all_views(tmp_path, aiida_profile_clean):
    from aiida_pino.console import viz

    sim = _sim(tmp_path, aiida_profile_clean, name="plotcase")
    for view in viz.VIEWS:
        png = viz.render_case_frame_png(sim, field="Temperature", view=view, frame=1)
        assert png[:8] == b"\x89PNG\r\n\x1a\n", view
    with pytest.raises(ValueError):
        viz.render_case_frame_png(sim, field="Nope", view="long")
    with pytest.raises(ValueError):
        viz.render_case_frame_png(sim, field="Temperature", view="diagonal")


def test_dataset_stats_pngs(tmp_path, aiida_profile_clean):
    from aiida_pino.console import viz

    rows = [
        {"laser_power": 100.0, "V_scan": 0.5,
         "stats": {"regime": "keyhole", "depth_um": 400.0, "cavity_rmin_um": 120.0, "melt_vol_um3": 1e6}},
        {"laser_power": 150.0, "V_scan": 1.0,
         "stats": {"regime": "conduction", "depth_um": 200.0, "cavity_rmin_um": 300.0, "melt_vol_um3": 2e6}},
        {"laser_power": 120.0, "V_scan": 0.7, "stats": None},
    ]
    for kind in viz.DATASET_PLOT_KINDS:
        png = viz.render_dataset_stats_png(rows, kind=kind)
        assert png[:8] == b"\x89PNG\r\n\x1a\n", kind
    with pytest.raises(ValueError):
        viz.render_dataset_stats_png(rows, kind="pie")


def test_regime_keyword_is_stamped_and_exclusive(tmp_path, aiida_profile_clean):
    from aiida_pino.console import viz

    sim = _sim(tmp_path, aiida_profile_clean, name="kwcase", gas_depth_cells=3, gas_xlen=1, gas_ylen=2)
    sim.base.extras.set("keywords", ["straight path", "conduction"])
    st = viz.get_or_compute_case_stats(sim)
    kws = sim.base.extras.get("keywords", None)
    assert st["regime"] == "keyhole"
    assert "keyhole" in kws
    assert "conduction" not in kws  # regime labels are mutually exclusive
    assert "straight path" in kws  # unrelated keywords survive


# --- the axis cache must not pin the stack it was sliced from --------------
# 2026-08-21: `pino-stats-autocompute` grew 0.128 -> 11.0 GiB over four cases
# and gc.collect() freed none of it. Cause: the canonical x/y/z/time_s arrays
# are stored broadcast to the full (T, 1, X, Y, Z) grid, and `_axes_cached`
# reduced them with basic slicing -- `a[0, 0, :, 0, 0]` is a VIEW, so each
# cached 1-D axis kept its whole float64 base alive. Measured on one dataset:
# four axes holding 0.0023 MiB of data pinned 0.660 GiB, and the cache holds
# 64 entries, so the ceiling was 42 GiB of arrays that should occupy ~150 KB.

def test_reduced_axis_does_not_pin_the_full_grid():
    from aiida_pino.console import viz

    full = np.zeros((5, 1, 8, 7, 6), dtype=np.float64)   # broadcast axis array
    for alias, expected in (("x", 8), ("y", 7), ("z", 6), ("time_s", 5)):
        axis = viz._reduce_axis(full, alias)
        assert axis.shape == (expected,)
        # the whole point: no reference back to `full`
        assert axis.base is None, (
            f"{alias} axis is a view pinning {full.nbytes} bytes")


def test_reduced_axis_keeps_the_right_values():
    """A copy is only correct if it copies the same elements the view saw."""
    from aiida_pino.console import viz

    full = np.zeros((5, 1, 8, 7, 6), dtype=np.float64)
    full[0, 0, :, 0, 0] = np.arange(8) * 10.0       # x varies along axis 2
    full[0, 0, 0, :, 0] = np.arange(7) * 100.0      # y along axis 3
    full[0, 0, 0, 0, :] = np.arange(6) * 1000.0     # z along axis 4
    full[:, 0, 0, 0, 0] = np.arange(5) * 0.5        # time along axis 0

    assert viz._reduce_axis(full, "y")[3] == 300.0
    assert viz._reduce_axis(full, "z")[5] == 5000.0
    assert viz._reduce_axis(full, "time_s")[4] == 2.0
    # x and time_s both touch element [0,0,0,0,0]; check a non-shared one
    assert viz._reduce_axis(full, "x")[7] == 70.0


def test_one_dimensional_axis_passes_through():
    from aiida_pino.console import viz

    flat = np.arange(6, dtype=np.float64)
    out = viz._reduce_axis(flat, "z")
    assert out.tolist() == [0, 1, 2, 3, 4, 5]
