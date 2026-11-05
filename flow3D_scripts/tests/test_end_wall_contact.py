"""`domain_validity` must report melt touching the SCAN-DIRECTION end walls.

It already checks the floor, depth truncation and the lateral (y) melt span --
but the y span is computed with `axis=(0, 2)`, which collapses x, so nothing
looked at the x faces at all. That is not an oversight in the span check: for a
scanned track an x-span fraction would false-positive on the normal trailing
pool. End-wall CONTACT is a different question and was simply untested.

Found 2026-08-16: all 7 scored Galiegue spot60 cases have melt against the
x_max wall (which reaches 1537-3179 K, above the 1923 K liquidus in 5 of them)
because `turn_off_at_end: false` carries the beam 42 um PAST the domain edge.
An adiabatic wall cannot conduct heat away, so the pool deepens there -- and at
150/175/200 W the deepest melt, the number that gets scored, forms within
50-80 um of it. `domain_validity` returned ok for all 7.

`end_wall_contact` REPORTS this; it deliberately does not flip `ok`. Flipping it
would retroactively invalidate every previously-scored scanned case in the
project, which is a human's call, not this function's.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import melt_metrics as mm  # noqa: E402


def _case(melt_x_slice, nx=12, ny=8, nz_metal=6, nz_gas=2):
    """Minimal export: melt placed at chosen x columns, never at the floor."""
    nz = nz_metal + nz_gas
    melt = np.zeros((3, nx, ny, nz), dtype=float)
    # Melt occupies mid-depth only, so depth_truncated stays False.
    melt[:, melt_x_slice, 2:ny - 2, nz_metal - 3:nz_metal] = 1.0
    z = np.concatenate([np.linspace(-60e-6, -10e-6, nz_metal),
                        np.linspace(10e-6, 20e-6, nz_gas)])
    return {
        "Melt Region": melt,
        "Temperature": np.full((3, nx, ny, nz), 300.0),
        "x": np.linspace(-100e-6, 100e-6, nx),
        "y": np.linspace(-40e-6, 40e-6, ny),
        "z": z,
    }


class EndWallContactTest(unittest.TestCase):
    def test_interior_track_reports_no_contact(self):
        v = mm.domain_validity(_case(slice(3, 9)))
        self.assertFalse(v.end_wall_contact)
        self.assertTrue(v.ok)

    def test_melt_on_the_x_max_wall_is_reported(self):
        v = mm.domain_validity(_case(slice(3, 12)))
        self.assertTrue(v.end_wall_contact,
                        "melt in the last x column must be reported")

    def test_melt_on_the_x_min_wall_is_reported(self):
        v = mm.domain_validity(_case(slice(0, 9)))
        self.assertTrue(v.end_wall_contact)

    def test_contact_does_not_flip_ok(self):
        """Reporting only. Flipping ok would retroactively invalidate the
        entire scanned back-catalogue, which is a human's decision."""
        v = mm.domain_validity(_case(slice(0, 12)))
        self.assertTrue(v.end_wall_contact)
        self.assertTrue(v.ok, "end-wall contact must not silently invalidate a run")

    def test_real_case_signature(self):
        """The failure that motivated this: a track melting into both x walls."""
        v = mm.domain_validity(_case(slice(0, 12)))
        self.assertTrue(v.end_wall_contact)
        self.assertFalse(v.depth_truncated)


if __name__ == "__main__":
    unittest.main()
