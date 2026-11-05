"""Parametric multibeam paths are reconstructed at the ONE seam every writer shares.

The mbs40 family defines each beam as `{"type": "random", "seed": ..., "length_m": ...,
"start_point_m": [...]}` rather than as a velocity table, so the prepin extraction has
nothing to extract and the beams reach the loader without `path_xy_m`. pino then refuses
("2 beams but per-beam paths for only 0") and the mb80 campaign is dead.

Two lessons are pinned here, both paid for:

1. **The path is RECONSTRUCTED from the generating seed, never synthesized.** The first
   proposed remedy was a straight line from `start_m` at `scan_speed_m_per_s`; these
   beams are `path_type='random'`, a seeded walk, and a line would have produced
   plausible geometry that silently mistrains.
2. **The reconstruction lives in `merge_multibeam_paths`, which all THREE sidecar writers
   route through.** Putting it in `merged_case_manifest` reached only the transform route:
   the pack-index staging kept writing pathless sidecars and a rebuilt cache still failed
   (seen on one RIGNO calcjob).
"""

from __future__ import annotations

import inspect
import unittest

from aiida_pino.data.pino_simulation import (
    merge_multibeam_paths,
    rebuild_parametric_beam_paths,
)


def _parametric_manifest(n=2):
    """A manifest shaped like mbs40's: parametric beam specs, no paths."""
    return {
        "input": {
            "geometry": {"length_m": 0.0015, "width_m": 0.001},
            "laser": {
                "beam_radius_m": 4.75e-05,
                "beams": [
                    {
                        "power_w": 100.0 + i,
                        "scan_speed_m_per_s": 0.3,
                        "path": {
                            "type": "random",
                            "seed": 11 + i,
                            "length_m": 0.0004,
                            "start_point_m": [0.0002 + i * 1e-5, -2e-05],
                        },
                    }
                    for i in range(n)
                ],
            },
        },
        "derived": {
            "beams": [
                {
                    "index": i + 1,
                    "start_m": [0.0002 + i * 1e-5, -2e-05],
                    "scan_speed_m_per_s": 0.3,
                    "path_type": "random",
                }
                for i in range(n)
            ]
        },
    }


class ReconstructionTest(unittest.TestCase):
    def test_parametric_beams_gain_a_path_and_matching_times(self):
        out = rebuild_parametric_beam_paths(_parametric_manifest(), "mbs40_001")
        beams = out["derived"]["beams"]
        self.assertEqual(len(beams), 2)
        for i, beam in enumerate(beams):
            path = beam.get("path_xy_m")
            self.assertTrue(path, f"beam {i} still has no path")
            self.assertEqual(len(beam["path_times_s"]), len(path))
            self.assertEqual([float(v) for v in path[0]],
                             [float(v) for v in beam["start_m"]])
            # Times are non-decreasing arc length over speed, starting at zero.
            times = beam["path_times_s"]
            self.assertEqual(times[0], 0.0)
            self.assertTrue(all(b >= a for a, b in zip(times, times[1:])))

    def test_a_random_path_is_not_the_straight_line_that_was_proposed(self):
        # The whole reason the parametric summary is insufficient.
        out = rebuild_parametric_beam_paths(_parametric_manifest(1), "mbs40_001")
        path = out["derived"]["beams"][0]["path_xy_m"]
        self.assertGreater(len(path), 2)
        (x0, y0), (x1, y1) = path[0], path[-1]
        chord = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        travelled = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                        for a, b in zip(path, path[1:]))
        self.assertGreater(travelled, chord * 1.05)

    def test_a_disagreeing_rebuild_is_refused_rather_than_served(self):
        # If the rebuild's sample count contradicts what the RUN recorded, it is not the
        # same path; serving it would score the model on geometry it never saw.
        manifest = _parametric_manifest(1)
        manifest["derived"]["beams"][0]["num_samples"] = 9999
        out = rebuild_parametric_beam_paths(manifest, "mbs40_001")
        self.assertIsNone(out["derived"]["beams"][0].get("path_xy_m"))

    def test_beams_that_already_have_paths_are_untouched(self):
        # The tabular family must not regress: extraction is authoritative.
        manifest = _parametric_manifest(1)
        manifest["derived"]["beams"][0]["path_xy_m"] = [[0.0, 0.0], [1.0, 1.0]]
        out = rebuild_parametric_beam_paths(manifest, "mbc40_000")
        self.assertEqual(out["derived"]["beams"][0]["path_xy_m"], [[0.0, 0.0], [1.0, 1.0]])

    def test_a_manifest_without_specs_is_returned_unchanged(self):
        self.assertEqual(rebuild_parametric_beam_paths({}, "x"), {})


class OneSeamTest(unittest.TestCase):
    def test_the_merge_reconstructs_on_both_of_its_exits(self):
        """Every sidecar writer calls `merge_multibeam_paths`; it must rebuild either way.

        The early return (nothing extracted -- the parametric case) is the one that was
        missed, and it is exactly the path mbs40 takes.
        """
        src = inspect.getsource(merge_multibeam_paths)
        self.assertEqual(src.count("rebuild_parametric_beam_paths"), 2, src)
        # No extraction at all: still reconstructed.
        out = merge_multibeam_paths(_parametric_manifest(1), None, "mbs40_001")
        self.assertTrue(out["derived"]["beams"][0].get("path_xy_m"))

    def test_all_three_sidecar_writers_go_through_the_merge(self):
        from aiida_pino.data import pino_dataset_collection as pdc
        from aiida_pino.data import pino_simulation as ps

        for label, func in (
            ("pack-index staging", ps.PinoSimulationData._export_runtime_npz_case),
            ("strided npz", pdc.PinoDatasetCollectionData._write_runtime_strided_npz_manifest),
            ("merged_case_manifest", pdc.merged_case_manifest),
        ):
            self.assertIn("merge_multibeam_paths", inspect.getsource(func),
                          f"{label} does not route through the merge")

    def test_callers_do_not_rebuild_a_second_time(self):
        # A caller repeating the rebuild is the drift this consolidation removes.
        from aiida_pino.data import pino_dataset_collection as pdc

        self.assertNotIn("rebuild_parametric_beam_paths",
                         inspect.getsource(pdc.merged_case_manifest))


if __name__ == "__main__":
    unittest.main()
