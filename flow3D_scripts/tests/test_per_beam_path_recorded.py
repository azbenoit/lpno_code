"""Each beam's manifest entry carries its actual path, not just a parametric summary.

Recording only `start_m` + `scan_speed_m_per_s` + `duration_s` describes the beam well
enough for FLOW-3D (which is handed the velocity table) but not for a consumer: the ML
pipeline synthesises per-beam channels from `path_xy_m`, and a `random` path cannot be
inferred from a start point and a speed. 31 of 61 members of the mb80 campaign were
unusable for exactly this (RIGNO session, 2026-08-28), while the tabular beam family was
fine -- two families describing themselves differently.
"""

from __future__ import annotations

import unittest

from test_multi_beam_paths import TWO_BEAMS, _config, _render


class PerBeamPathIsRecorded(unittest.TestCase):
    def test_every_beam_records_its_sampled_path(self) -> None:
        _, _, man = _render(_config(beams=TWO_BEAMS), "unit_mb_pathxy")
        beams = man["derived"]["beams"]
        self.assertEqual(len(beams), len(TWO_BEAMS))
        for i, beam in enumerate(beams):
            path = beam.get("path_xy_m")
            self.assertTrue(path, f"beam {i} records no path_xy_m")
            self.assertEqual(len(path), beam["num_samples"],
                             f"beam {i}: path length disagrees with num_samples")
            self.assertEqual([float(v) for v in path[0]], [float(v) for v in beam["start_m"]],
                             f"beam {i}: path must start at the recorded start_m")
            self.assertTrue(all(len(pt) == 2 for pt in path), "points must be [x, y]")

    def test_a_random_path_is_not_a_straight_line(self) -> None:
        # The property that makes the parametric summary insufficient: if these paths were
        # straight, start + speed + duration WOULD have been enough and this field would be
        # redundant. Guards against a future "simplification" back to the broken form.
        _, _, man = _render(_config(beams=[{
            "power_w": 100.0, "scan_speed_m_per_s": 0.2,
            "path": {"type": "random", "seed": 7, "length_m": 0.0012},
        }]), "unit_mb_notstraight")
        path = man["derived"]["beams"][0]["path_xy_m"]
        self.assertGreater(len(path), 2)
        (x0, y0), (x1, y1) = path[0], path[-1]
        chord = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        travelled = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                        for a, b in zip(path, path[1:]))
        self.assertGreater(travelled, chord * 1.05,
                           "a random walk must be longer than its chord")


if __name__ == "__main__":
    unittest.main()
