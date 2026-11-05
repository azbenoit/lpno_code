#!/usr/bin/env python3
"""Melt-pool metrics: one definition, used everywhere.

Every geometric metric in the calibration campaign lived in a separate throwaway
watcher script, each with its own copy of the code. The copies diverged, and the
divergences produced real retractions: an aspect ratio taking width from a
different frame than depth, and four separate comparisons of a simulated vapour
depression against a metallographic fusion depth. These tests pin the definitions
written down in validation/ACCEPTANCE.md.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import melt_metrics as mm  # noqa: E402


def synthetic(*, nt=20, nx=9, ny=11, n_metal=10, n_gas_above=4, dz=1e-5,
              melt_depth_cells=4, melt_half_width=3,
              cavity_depth_cells=2, floor_k=300.0):
    """A case with exactly known geometry.

    z ascends; the first ``n_metal`` cells are metal (z < 0), the rest gas above
    the surface. Arrays are [t, x, y, z] to match the NPZ export layout.
    """
    nz = n_metal + n_gas_above
    z = (np.arange(nz) - n_metal + 0.5) * dz          # metal cells strictly < 0
    y = np.arange(ny) * dz
    x = np.arange(nx) * dz

    fof = np.ones((nt, nx, ny, nz))
    fof[:, :, :, n_metal:] = 0.0                       # gas above the surface
    melt = np.zeros((nt, nx, ny, nz))
    temp = np.full((nt, nx, ny, nz), 400.0)
    temp[:, :, :, 0] = floor_k                         # bottom row = the floor

    xc, yc = nx // 2, ny // 2
    for t in range(nt):
        # melt blob at mid-track, present only in the second half of the run so a
        # time-union differs from any single frame
        if t >= nt // 2:
            melt[t, xc - 1:xc + 2,
                 yc - melt_half_width:yc + melt_half_width + 1,
                 n_metal - melt_depth_cells:n_metal] = 1.0
        # vapour cavity: carve gas into the metal at mid-track
        if t >= nt // 2:
            fof[t, xc, yc, n_metal - cavity_depth_cells:n_metal] = 0.0
    return {"Fraction Of Fluid": fof, "Melt Region": melt, "Temperature": temp,
            "x": x, "y": y, "z": z}


class FusionGeometryTests(unittest.TestCase):
    def test_depth_and_width_are_the_time_union(self) -> None:
        d = synthetic(melt_depth_cells=4, melt_half_width=3, dz=1e-5)
        g = mm.fusion_geometry(d)
        self.assertAlmostEqual(g.depth_um, 4 * 10.0)        # 4 cells x 10 um
        self.assertAlmostEqual(g.width_um, 7 * 10.0)        # 2*3+1 cells
        self.assertAlmostEqual(g.aspect, 40.0 / 70.0)

    def test_no_melt_reports_zero_not_a_crash(self) -> None:
        d = synthetic(melt_depth_cells=0, melt_half_width=0)
        d["Melt Region"][:] = 0.0
        g = mm.fusion_geometry(d)
        self.assertEqual((g.depth_um, g.width_um), (0.0, 0.0))
        self.assertEqual(g.aspect, 0.0)


class DepressionTests(unittest.TestCase):
    def test_steady_depth_ignores_the_edge_columns(self) -> None:
        d = synthetic(cavity_depth_cells=2, dz=1e-5)
        # a spurious deep gas column on the x edge must not be counted
        d["Fraction Of Fluid"][:, 0, :, :] = 0.0
        r = mm.depression(d)
        self.assertAlmostEqual(r.depth_um, 2 * 10.0)

    def test_width_uses_the_same_window_as_depth(self) -> None:
        """The round-4 bug: width came from the global max-gas frame.

        Here the cavity is WIDE only in an early frame that the steady window
        excludes, so a width drawn from the whole run would be larger.
        """
        d = synthetic(cavity_depth_cells=2)
        nt, nx, ny, nz = d["Fraction Of Fluid"].shape
        d["Fraction Of Fluid"][1, nx // 2, :, 8:10] = 0.0    # early, full-width
        r = mm.depression(d)
        self.assertLess(r.width_um, ny * 10.0,
                        "width must come from the steady window, not the widest frame")
        self.assertAlmostEqual(r.aspect, r.depth_um / r.width_um)


class ValidityAndModeTests(unittest.TestCase):
    def test_hot_floor_invalidates_the_case(self) -> None:
        self.assertTrue(mm.domain_validity(synthetic(floor_k=300.0)).ok)
        self.assertFalse(mm.domain_validity(synthetic(floor_k=776.0)).ok,
                         "a domain-clipped case must count for nothing")

    def test_melt_spanning_the_domain_invalidates_the_case(self) -> None:
        d = synthetic()
        d["Melt Region"][:, :, :, :] = 1.0
        self.assertFalse(mm.domain_validity(d).ok)

    def test_melt_reaching_the_array_floor_is_invalid(self) -> None:
        """Depth truncation was unchecked: only lateral span was.

        For a refined run the exported NPZ is the fine block alone, so its floor is
        the refinement box's floor, not the substrate's -- a pool that outgrows the
        box is measured against a boundary. The failure under-reports depth, the
        same direction as the deficit the refinement exists to investigate, so it
        cannot be allowed to pass silently."""
        deep = synthetic(n_metal=10, melt_depth_cells=10)   # melt reaches z index 0
        v = mm.domain_validity(deep)
        self.assertTrue(v.depth_truncated)
        self.assertFalse(v.ok)

    def test_melt_clear_of_the_floor_stays_valid(self) -> None:
        shallow = synthetic(n_metal=10, melt_depth_cells=4)
        v = mm.domain_validity(shallow)
        self.assertFalse(v.depth_truncated)
        self.assertTrue(v.ok)


class DepthKindGuardTests(unittest.TestCase):
    def test_comparing_across_depth_kinds_is_refused(self) -> None:
        """Not a judgement call -- a bug. Committed four times in one night."""
        with self.assertRaises(mm.DepthKindMismatch):
            mm.score_point(sim_depth_um=29.0, measured_depth_um=128.25,
                           sim_kind="depression", measured_kind="fusion")

    def test_matching_kinds_score_normally(self) -> None:
        s = mm.score_point(sim_depth_um=60.0, measured_depth_um=128.25,
                           sim_kind="fusion", measured_kind="fusion")
        self.assertAlmostEqual(s.error_pct, 100 * (60.0 - 128.25) / 128.25, places=6)
        self.assertFalse(s.within_band)

    def test_band_defaults_to_the_pre_registered_30_percent(self) -> None:
        self.assertTrue(mm.score_point(sim_depth_um=100.0, measured_depth_um=128.25,
                                       sim_kind="fusion", measured_kind="fusion").within_band)
        self.assertFalse(mm.score_point(sim_depth_um=80.0, measured_depth_um=128.25,
                                        sim_kind="fusion", measured_kind="fusion").within_band)


class LadderTests(unittest.TestCase):
    def test_mean_signed_error_is_reported_not_just_pass_count(self) -> None:
        """A systematic one-sign offset is a finding even when every point passes."""
        errs = [-22.0, -25.0, -20.0, -24.0]
        summary = mm.ladder_summary(errs)
        self.assertEqual(summary.n_within_band, 4)
        self.assertAlmostEqual(summary.mean_signed_error_pct, -22.75)
        self.assertTrue(summary.systematic,
                        "all-same-sign errors must be flagged even when all pass")

    def test_mixed_signs_are_not_systematic(self) -> None:
        self.assertFalse(mm.ladder_summary([-10.0, 8.0, -3.0, 12.0]).systematic)


class AbsorbedFractionTests(unittest.TestCase):
    """Net absorbed fraction from the solver energy audit.

    This lived only in the reproducibility notebook. It is the metric behind the
    NIST absorptance diagnostic, so it belongs with the others.
    """

    def _hd3out(self, tmp: Path) -> Path:
        # 10 mJ gained over 1 ms at 100 W (= 100 mJ delivered) -> 10% absorbed
        p = tmp / "hd3out.case"
        p.write_text(
            "  time = 1.0000E-04\n  total fluid #1 thermal energy =  1.000E-03\n"
            "  time = 1.1000E-03\n  total fluid #1 thermal energy =  1.100E-02\n")
        return p

    def test_ratio_of_energy_gain_to_delivered_energy(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = self._hd3out(Path(d))
            got = mm.absorbed_fraction(p, power_w=100.0, t0=0.0, t1=1.0)
            # (0.011 - 0.001) J over (1.1e-3 - 1.0e-4) s at 100 W = 0.10
            self.assertAlmostEqual(got, 0.010 / (100.0 * 1.0e-3), places=6)
            self.assertAlmostEqual(got, 0.10, places=6)

    def test_window_with_too_few_samples_returns_nan(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = self._hd3out(Path(d))
            self.assertTrue(np.isnan(mm.absorbed_fraction(p, power_w=100.0,
                                                          t0=0.5, t1=0.6)))

    def test_missing_file_returns_nan_rather_than_raising(self) -> None:
        self.assertTrue(np.isnan(mm.absorbed_fraction(Path("/nonexistent/hd3out.x"),
                                                      power_w=100.0)))


class PerCaseCellSizeTests(unittest.TestCase):
    """The notebook hardcoded CELL_UM = 10.0 while 14 of 96 cases were not 10 um.

    SS304L ran at 25 and 50 um and Vaglio/spot25 at 6 um, so every length the
    notebook reported for them was wrong by 2.5x, 5x and 1.67x respectively. Cell
    size must come from the case's own coordinate arrays.
    """

    def test_cell_size_is_read_from_the_case(self) -> None:
        for dz in (6e-6, 1e-5, 2.5e-5, 5e-5):
            self.assertAlmostEqual(mm.cell_size_um(synthetic(dz=dz)), dz * 1e6, places=6)

    def test_geometry_scales_with_the_case_cell_size(self) -> None:
        coarse = mm.fusion_geometry(synthetic(dz=5e-5, melt_depth_cells=4))
        fine = mm.fusion_geometry(synthetic(dz=6e-6, melt_depth_cells=4))
        self.assertAlmostEqual(coarse.depth_um, 200.0)
        self.assertAlmostEqual(fine.depth_um, 24.0)


class DepressionFrameTests(unittest.TestCase):
    def test_single_frame_geometry_for_imagery(self) -> None:
        d = synthetic(cavity_depth_cells=3, nt=20)
        g = mm.depression_frame(d, 19)          # cavity present in the second half
        self.assertAlmostEqual(g.depth_um, 30.0)
        self.assertGreater(g.width_um, 0.0)

    def test_frame_without_a_cavity_is_zero(self) -> None:
        d = synthetic(cavity_depth_cells=3, nt=20)
        g = mm.depression_frame(d, 0)           # first half has no cavity
        self.assertEqual((g.depth_um, g.width_um), (0.0, 0.0))


class LaserOnWindowTests(unittest.TestCase):
    """The 0.85 laser-on fraction is a heuristic and is wrong for some datasets.

    It was chosen because the campaign's own cases turn the laser off at ~80% of
    the run. But the 84-case straight-path sweep sets finish_time == travel time,
    so the laser is on for 100% of every run and 0.85 silently discards the final
    15% -- the deepest part of the track -- as a cooling tail that does not exist.
    When the caller knows the laser-off time, it must be usable.
    """

    def _timed(self, n=20, laser_off_s=None, dt=1e-4):
        d = synthetic(nt=n)
        d = dict(d)
        d["time_s"] = np.arange(n) * dt
        return d

    def test_laser_off_time_defines_the_window(self) -> None:
        d = self._timed(n=20, dt=1e-4)          # t = 0 .. 1.9e-3
        # laser off at 1.0e-3 -> frames 0..10 are laser-on; tail 30% of those
        win = mm.steady_window(20, laser_off_s=1.0e-3, times_s=d["time_s"])
        self.assertLessEqual(win.stop, 11)
        self.assertGreater(win.stop, 8)

    def test_laser_on_for_the_whole_run_uses_every_frame(self) -> None:
        d = self._timed(n=20, dt=1e-4)
        win = mm.steady_window(20, laser_off_s=1.9e-3, times_s=d["time_s"])
        self.assertEqual(win.stop, 20,
                         "if the laser never turns off, the last frame is still laser-on")

    def test_falls_back_to_the_documented_fraction_when_unknown(self) -> None:
        self.assertEqual(mm.steady_window(20), mm.steady_window(20, laser_off_s=None))
        self.assertEqual(mm.steady_window(20).stop, int(20 * mm.LASER_ON_FRACTION))

    def test_depression_accepts_a_laser_off_time(self) -> None:
        d = self._timed(n=20, dt=1e-4)
        d["Fraction Of Fluid"] = synthetic(nt=20)["Fraction Of Fluid"]
        full = mm.depression(d, laser_off_s=1.9e-3)
        clipped = mm.depression(d)
        self.assertIsInstance(full.depth_um, float)
        self.assertIsInstance(clipped.depth_um, float)


class MeltSpanIsMeasuredInMetalTests(unittest.TestCase):
    """The span check asks "did the POOL reach the side walls", so it must ignore
    ejected melt above the surface.

    Measuring it over all z counted spatter in the gas layer and falsely
    invalidated three SS304L 456 W cases (all-z spans 0.84/0.87/0.94 against
    metal-only 0.30/0.52/0.33) plus a CoCr case, excluding them from summaries for
    a reason unrelated to whether the domain constrained the fusion geometry.
    """

    def test_spatter_above_the_surface_does_not_invalidate_a_case(self) -> None:
        d = synthetic(melt_half_width=2)          # narrow pool in the metal
        nt, nx, ny, nz = d["Melt Region"].shape
        n_metal = int((np.asarray(d["z"]) < 0).sum())
        # ejected droplets spanning the full width, but ABOVE the surface
        d["Melt Region"][:, :, :, n_metal:] = 1.0
        v = mm.domain_validity(d)
        self.assertLess(v.melt_span_fraction, 0.8)
        self.assertTrue(v.ok, "gas-layer spatter must not invalidate the geometry")

    def test_a_pool_that_really_reaches_the_walls_still_invalidates(self) -> None:
        d = synthetic()
        n_metal = int((np.asarray(d["z"]) < 0).sum())
        d["Melt Region"][:, :, :, :n_metal] = 1.0     # melt fills the metal laterally
        v = mm.domain_validity(d)
        self.assertGreaterEqual(v.melt_span_fraction, 0.8)
        self.assertFalse(v.ok)


class MeshAdequacyTests(unittest.TestCase):
    """An a-priori resolution criterion, since the deficit is set by the FEATURE.

    Cells-per-beam-radius does not predict anything here: every Gan case is 4.75
    cells/radius and passes, every Galiegue rung is 6.0 and fails. What separates
    them is cells across the vapour cavity, and the cavity depth is predictable
    before running from the Gan scaling law.
    """

    def test_adequacy_is_measured_against_the_feature(self) -> None:
        # 20 um cavity on a 10 um grid = 2 cells -> inadequate
        self.assertFalse(mm.mesh_adequate(feature_um=20.0, cell_um=10.0).ok)
        # same cavity on a 2 um grid = 10 cells -> adequate
        self.assertTrue(mm.mesh_adequate(feature_um=20.0, cell_um=2.0).ok)

    def test_it_reports_the_cell_size_that_would_suffice(self) -> None:
        r = mm.mesh_adequate(feature_um=20.0, cell_um=10.0)
        self.assertAlmostEqual(r.required_cell_um, 20.0 / mm.MIN_CELLS_PER_FEATURE)
        self.assertAlmostEqual(r.cells, 2.0)

    def test_the_campaign_cases_are_retrodicted(self) -> None:
        """The criterion must reproduce which cases actually failed."""
        passing = [234.0, 133.0, 93.0]        # Gan 206/156/129 W, all PASS
        failing = [30.0, 20.0, 10.0]          # Galiegue 110/90/80 W, all FAIL
        for d in passing:
            self.assertTrue(mm.mesh_adequate(feature_um=d, cell_um=10.0).ok, d)
        for d in failing:
            self.assertFalse(mm.mesh_adequate(feature_um=d, cell_um=10.0).ok, d)


if __name__ == "__main__":
    unittest.main()


class PathAwareFusionGeometryTests(unittest.TestCase):
    """Cross-section geometry for a beam that does not travel along x.

    `fusion_geometry` slices a +-5 cell band at mid-x, which assumes the track
    runs along x through the middle of the domain. For a random or multi-beam
    path there is no such track, so the band lands wherever the domain centre
    happens to be: on the 2026-08-05 multi-beam batch it returned depth 0 AND
    width 0 for `mb40_008` and `mb40_023` while both carried >2 % melt.

    `fusion_geometry_at_beam` finds the beam from the deposition field itself
    (`Heat Absorption Rate`, whose surface peak is where the laser is), takes the
    plane perpendicular to the beam's own direction of travel, and unions melt
    over time in that plane. For a straight x-track the perpendicular plane IS
    the y-z plane at mid-x, so it must reproduce the original number.
    """

    @staticmethod
    def _synthetic(track="x"):
        """A cuboid melt pool swept along x or along the diagonal."""
        nt, nx, ny, nz = 12, 40, 40, 20
        x = np.linspace(-200e-6, 200e-6, nx).astype(np.float32)
        y = np.linspace(-200e-6, 200e-6, ny).astype(np.float32)
        z = np.linspace(-150e-6, 50e-6, nz).astype(np.float32)
        melt = np.zeros((nt, nx, ny, nz), np.float32)
        har = np.zeros((nt, nx, ny, nz), np.float32)
        surf = int(np.argmin(np.abs(z)))
        for k in range(nt):
            ix = 6 + k * 2
            iy = ny // 2 if track == "x" else 6 + k * 2
            melt[k, max(ix - 3, 0):ix + 4, max(iy - 4, 0):iy + 5, surf - 5:surf + 1] = 1.0
            har[k, ix, iy, surf] = 1.0
        return {"x": x, "y": y, "z": z,
                "time_s": np.linspace(0, 1e-3, nt),
                "Melt Region": melt, "Heat Absorption Rate": har}

    def test_straight_track_matches_the_original_metric(self):
        d = self._synthetic("x")
        old = mm.fusion_geometry(d)
        new = mm.fusion_geometry_at_beam(d)
        dy = float(d["y"][1] - d["y"][0])
        self.assertAlmostEqual(new.depth_um, old.depth_um, delta=1.0)
        self.assertAlmostEqual(new.width_um, old.width_um, delta=2 * dy * 1e6)

    def test_diagonal_track_is_measured_where_the_beam_is(self):
        """The original metric misses a diagonal track; the new one must not."""
        d = self._synthetic("diag")
        new = mm.fusion_geometry_at_beam(d)
        self.assertGreater(new.depth_um, 0.0)
        self.assertGreater(new.width_um, 0.0)

    def test_returns_zero_without_a_deposition_field(self):
        d = self._synthetic("x")
        d.pop("Heat Absorption Rate")
        g = mm.fusion_geometry_at_beam(d)
        self.assertEqual((g.depth_um, g.width_um), (0.0, 0.0))


class MeltMaskFallbackTests(unittest.TestCase):
    """Fusion geometry from Temperature when `Melt Region` was not exported.

    Datasets ingested with the ML training profile carry only Fraction Of Fluid,
    Pressure, Temperature and Velocity -- `ti64_varspeed150` is 150 such cases --
    so every melt metric returned nothing for them and the dataset could not be
    scored from AiiDA at all.

    `Melt Region` is the INSTANTANEOUS molten fraction and is not a temperature
    threshold: per frame it agrees with `T > solidus` at only IoU 0.36-0.49,
    because it tracks the solidus-liquidus fraction rather than a hard cut. The
    fusion zone is the time UNION though, and that reproduces at IoU 0.965. At
    the metric level the agreement is tighter still, measured on four multi-beam
    cases: depth identical in all four, width identical in two and one cell out in
    the other two.

    So the fallback is sound for fusion geometry specifically. It is NOT a general
    substitute for the field.
    """

    @staticmethod
    def _case(with_melt=True):
        nt, nx, ny, nz = 6, 30, 30, 16
        x = np.linspace(-150e-6, 150e-6, nx).astype(np.float32)
        y = np.linspace(-150e-6, 150e-6, ny).astype(np.float32)
        z = np.linspace(-120e-6, 40e-6, nz).astype(np.float32)
        surf = int(np.argmin(np.abs(z)))
        T = np.full((nt, nx, ny, nz), 300.0, np.float32)
        melt = np.zeros((nt, nx, ny, nz), np.float32)
        har = np.zeros((nt, nx, ny, nz), np.float32)
        for k in range(nt):
            ix = 6 + k * 3
            T[k, ix - 2:ix + 3, ny // 2 - 3:ny // 2 + 4, surf - 4:surf + 1] = 2400.0
            melt[k, ix - 2:ix + 3, ny // 2 - 3:ny // 2 + 4, surf - 4:surf + 1] = 1.0
            har[k, ix, ny // 2, surf] = 1.0
        d = {"x": x, "y": y, "z": z, "time_s": np.linspace(0, 1e-3, nt),
             "Temperature": T, "Heat Absorption Rate": har}
        if with_melt:
            d["Melt Region"] = melt
        return d

    def test_falls_back_to_temperature_when_melt_region_is_absent(self):
        with_m = mm.fusion_geometry_at_beam(self._case(True))
        without = mm.fusion_geometry_at_beam(self._case(False), solidus_k=1873.15)
        self.assertGreater(without.depth_um, 0.0)
        self.assertAlmostEqual(without.depth_um, with_m.depth_um, delta=1e-6)
        self.assertAlmostEqual(without.width_um, with_m.width_um, delta=1e-6)

    def test_melt_region_still_wins_when_present(self):
        """The real field must not be overridden by the approximation."""
        d = self._case(True)
        d["Melt Region"][:] = 0.0           # nothing molten, though T is hot
        g = mm.fusion_geometry_at_beam(d, solidus_k=1873.15)
        self.assertEqual((g.depth_um, g.width_um), (0.0, 0.0))

    def test_no_fallback_without_a_stated_solidus(self):
        """Guessing an alloy's solidus would silently score the wrong material."""
        g = mm.fusion_geometry_at_beam(self._case(False))
        self.assertEqual((g.depth_um, g.width_um), (0.0, 0.0))


class BeamTrackJitterTests(unittest.TestCase):
    """The deposition peak jitters, and the direction must survive it.

    `beam_track` reads the argmax of a cell-quantised field, which hops around
    the cavity walls: on t2_nist195 the median frame-to-frame step is 22 um while
    the beam advances only 3.5 um per frame. Taking the direction between adjacent
    frames therefore pointed the normal ALONG the track and measured its length as
    width. Real straight-track cases read 810 um against a true 130 um
    (rep3_in718_p245) and 790 um against 160 um (t2_nist195).

    Fixed by smoothing the track and taking the direction over a multi-cell
    baseline. On real straight-track cases the metric now reproduces
    `fusion_geometry` exactly in depth and within two cells in width.
    """

    @staticmethod
    def _jittery_x_track(jitter_cells=3):
        nt, nx, ny, nz = 40, 60, 40, 16
        x = np.linspace(-300e-6, 300e-6, nx).astype(np.float32)
        y = np.linspace(-200e-6, 200e-6, ny).astype(np.float32)
        z = np.linspace(-120e-6, 40e-6, nz).astype(np.float32)
        surf = int(np.argmin(np.abs(z)))
        melt = np.zeros((nt, nx, ny, nz), np.float32)
        har = np.zeros((nt, nx, ny, nz), np.float32)
        rng = np.random.default_rng(0)
        for k in range(nt):
            ix = 5 + k                      # a clean straight advance along x
            iy = ny // 2
            melt[k, max(ix - 2, 0):ix + 3, iy - 3:iy + 4, surf - 4:surf + 1] = 1.0
            # the deposition peak hops by a few cells, as the real field does
            jx = int(np.clip(ix + rng.integers(-jitter_cells, jitter_cells + 1), 0, nx - 1))
            jy = int(np.clip(iy + rng.integers(-jitter_cells, jitter_cells + 1), 0, ny - 1))
            har[k, jx, jy, surf] = 1.0
        return {"x": x, "y": y, "z": z, "time_s": np.linspace(0, 1e-3, nt),
                "Melt Region": melt, "Heat Absorption Rate": har}

    def test_jitter_does_not_rotate_the_measurement_plane(self):
        d = self._jittery_x_track()
        old = mm.fusion_geometry(d)
        new = mm.fusion_geometry_at_beam(d)
        dy = float(d["y"][1] - d["y"][0]) * 1e6
        self.assertAlmostEqual(new.depth_um, old.depth_um, delta=1e-6)
        # a rotated plane would report the TRACK LENGTH, several times the width
        self.assertLess(new.width_um, old.width_um + 3 * dy,
                        "width inflated: the normal is pointing along the track")
