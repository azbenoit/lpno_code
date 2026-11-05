#!/usr/bin/env python3
"""Unit tests for the multi-rank NPZ block merge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import npz_block_merge as merge  # noqa: E402


def cell_axis(n: int, x0: float = 0.0, dx: float = 1.0) -> np.ndarray:
    return x0 + (np.arange(n) + 0.5) * dx


class NpzBlockMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reference global grid 4x3x2, a scalar and a 3-vector with unique values.
        self.gx, self.gy, self.gz = cell_axis(4), cell_axis(3), cell_axis(2)
        self.scalar = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
        self.vector = np.arange(4 * 3 * 2 * 3, dtype=np.float32).reshape(4, 3, 2, 3)

    def _block(self, xsl: slice):
        ax = self.gx[xsl]
        return {
            "axes": (ax, self.gy, self.gz),
            "fields": {"Temperature": self.scalar[xsl], "Velocity": self.vector[xsl]},
        }

    def test_two_blocks_split_along_x_reconstruct_the_global_grid(self) -> None:
        blocks = [self._block(slice(0, 2)), self._block(slice(2, 4))]
        (gx, gy, gz), fields = merge.stitch_blocks(blocks)
        np.testing.assert_allclose(gx, self.gx)
        np.testing.assert_allclose(gy, self.gy)
        np.testing.assert_allclose(gz, self.gz)
        np.testing.assert_array_equal(fields["Temperature"], self.scalar)
        np.testing.assert_array_equal(fields["Velocity"], self.vector)

    def test_block_order_does_not_matter(self) -> None:
        blocks = [self._block(slice(2, 4)), self._block(slice(0, 2))]  # reversed
        _, fields = merge.stitch_blocks(blocks)
        np.testing.assert_array_equal(fields["Temperature"], self.scalar)

    def test_single_block_is_identity(self) -> None:
        (gx, _, _), fields = merge.stitch_blocks([self._block(slice(0, 4))])
        np.testing.assert_allclose(gx, self.gx)
        np.testing.assert_array_equal(fields["Temperature"], self.scalar)

    def test_ragged_blocks_that_do_not_tile_are_rejected(self) -> None:
        # Block B is missing the top y-row, so cell (x>=2, y=2) is uncovered even
        # though every coordinate appears in the union -> coverage check fires.
        block_a = {
            "axes": (self.gx[0:2], self.gy, self.gz),
            "fields": {"Temperature": self.scalar[0:2]},
        }
        block_b = {
            "axes": (self.gx[2:4], self.gy[0:2], self.gz),
            "fields": {"Temperature": self.scalar[2:4, 0:2]},
        }
        with self.assertRaisesRegex(ValueError, "do not tile"):
            merge.stitch_blocks([block_a, block_b])


if __name__ == "__main__":
    unittest.main()
