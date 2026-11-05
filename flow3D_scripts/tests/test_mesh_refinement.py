"""Nested mesh refinement around the laser track.

A refined case renders a second, finer `&mesh` block covering the laser path, so
the melt region is resolved without paying uniform-refinement cost over the whole
domain (about 12x CPU time for 10 -> 5 um, measured on a matched pair).

The two things that can silently go wrong, and are pinned here:

* a nested block whose faces do not land on parent cell faces is not a legal
  nesting, and FLOW-3D's interpolation across the interface degrades quietly;
* the NPZ exporter stitches blocks assuming they are disjoint bricks that tile
  the domain (npz_block_merge), which a nested block violates -- it overlaps the
  parent at a different spacing.  The finest block must be selected, not merged.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

import npz_block_merge  # noqa: E402
import run_rectangular_case as rectangular  # noqa: E402

SMOKE_CONFIG = REPO / "configs" / "rectangular_moving_laser_smoke.json"


def _config(tmp: str, **mesh_extra) -> Path:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["mesh"].update(mesh_extra)
    path = Path(tmp) / "refined.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _blocks(text: str) -> list[dict]:
    """Parse every `&mesh` namelist out of a rendered prepin."""
    out = []
    for chunk in text.split("&mesh")[1:]:
        body = chunk.split("/")[0]
        block = {}
        for key in ("size", "px(1)", "px(2)", "py(1)", "py(2)", "pz(1)", "pz(2)"):
            for line in body.splitlines():
                line = line.strip().rstrip(",")
                if line.startswith(key + "="):
                    block[key] = float(line.split("=", 1)[1])
        out.append(block)
    return out


class MeshRefinementConfigTests(unittest.TestCase):
    def test_absent_refinement_renders_exactly_one_mesh_block(self) -> None:
        """Default path is untouched: no refinement key, one block, no stray token."""
        case, repl, manifest = rectangular.load_and_validate(SMOKE_CONFIG, "unit_norefine")
        self.assertEqual(repl["__EXTRA_MESH_BLOCKS__"], "")
        self.assertIsNone(manifest["derived"]["mesh_refinement"])
        with tempfile.TemporaryDirectory() as tmp:
            text = rectangular.render(Path(tmp) / "c", case, repl, manifest).read_text(encoding="utf-8")
        self.assertEqual(len(_blocks(text)), 1)
        self.assertNotRegex(text, r"__[A-Z0-9_]+__")

    def test_refined_case_renders_a_finer_second_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, refinement={"factor": 2, "margin_m": 0.0003})
            case, repl, manifest = rectangular.load_and_validate(cfg, "unit_refine")
            text = rectangular.render(Path(tmp) / "c", case, repl, manifest).read_text(encoding="utf-8")

        blocks = _blocks(text)
        self.assertEqual(len(blocks), 2)
        parent, fine = blocks
        self.assertAlmostEqual(fine["size"], parent["size"] / 2)
        self.assertAlmostEqual(manifest["derived"]["mesh_refinement"]["cell_size_m"], parent["size"] / 2)

    def test_fine_block_faces_land_on_parent_cell_faces(self) -> None:
        """Illegal nesting is silent in FLOW-3D, so it is pinned here."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, refinement={"factor": 2, "margin_m": 0.00031})
            case, repl, manifest = rectangular.load_and_validate(cfg, "unit_snap")
            text = rectangular.render(Path(tmp) / "c", case, repl, manifest).read_text(encoding="utf-8")

        parent, fine = _blocks(text)
        cell = parent["size"]
        for axis in ("px", "py", "pz"):
            for index in (1, 2):
                key = f"{axis}({index})"
                offset = (fine[key] - parent[f"{axis}(1)"]) / cell
                self.assertAlmostEqual(offset, round(offset), places=6,
                                       msg=f"{key} does not sit on a parent cell face")

    def test_fine_block_is_strictly_inside_the_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, refinement={"factor": 2, "margin_m": 0.0002})
            case, repl, manifest = rectangular.load_and_validate(cfg, "unit_inside")
            text = rectangular.render(Path(tmp) / "c", case, repl, manifest).read_text(encoding="utf-8")

        parent, fine = _blocks(text)
        for axis in ("px", "py", "pz"):
            self.assertGreaterEqual(fine[f"{axis}(1)"], parent[f"{axis}(1)"])
            self.assertLessEqual(fine[f"{axis}(2)"], parent[f"{axis}(2)"])
            self.assertLess(fine[f"{axis}(1)"], fine[f"{axis}(2)"])

    def test_fine_block_covers_the_laser_track_and_beam(self) -> None:
        """The metrics only look inside the fine block, so the melt must fit in it."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, refinement={"factor": 2, "margin_m": 0.0003})
            case, repl, manifest = rectangular.load_and_validate(cfg, "unit_cover")
            text = rectangular.render(Path(tmp) / "c", case, repl, manifest).read_text(encoding="utf-8")

        _, fine = _blocks(text)
        d = manifest["derived"]
        start, end = d["beam_focus_start_m"], d["beam_focus_end_m"]
        radius = manifest["input"]["laser"]["beam_radius_m"]
        self.assertLessEqual(fine["px(1)"], min(start[0], end[0]) - radius)
        self.assertGreaterEqual(fine["px(2)"], max(start[0], end[0]) + radius)
        self.assertLessEqual(fine["py(1)"], min(start[1], end[1]) - radius)
        self.assertGreaterEqual(fine["py(2)"], max(start[1], end[1]) + radius)
        # The free surface, where depression and fusion depth are measured.
        self.assertLess(fine["pz(1)"], d["surface_z_m"])
        self.assertGreater(fine["pz(2)"], d["surface_z_m"])

    def test_refinement_rejects_non_integer_and_out_of_range_factors(self) -> None:
        for bad in (1, 4, 2.5, 0, -2):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = _config(tmp, refinement={"factor": bad, "margin_m": 0.0003})
                with self.assertRaises(rectangular.ConfigError):
                    rectangular.load_and_validate(cfg, "unit_badfactor")

    def test_refinement_rejects_non_positive_margin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, refinement={"factor": 2, "margin_m": 0.0})
            with self.assertRaises(rectangular.ConfigError):
                rectangular.load_and_validate(cfg, "unit_badmargin")

    def test_refinement_rejects_multi_rank_runs(self) -> None:
        """Each mesh block is decomposed per rank; the exporter cannot tell the
        resulting bricks apart from a nested block, so refinement is single-rank."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, refinement={"factor": 2, "margin_m": 0.0003})
            with self.assertRaisesRegex(rectangular.ConfigError, "rank"):
                rectangular.load_and_validate(cfg, "unit_ranks", cores_override=4, threads_override=1)

    def test_margin_that_would_fill_the_domain_is_rejected(self) -> None:
        """A fine block equal to the parent is uniform refinement wearing a hat."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, refinement={"factor": 2, "margin_m": 0.01})
            with self.assertRaisesRegex(rectangular.ConfigError, "cell_size_m"):
                rectangular.load_and_validate(cfg, "unit_toobig")


class FinestBlockSelectionTests(unittest.TestCase):
    """The exporter must pick the fine block, not stitch overlapping blocks."""

    @staticmethod
    def _payload(x0: float, x1: float, step: float, value: float) -> dict:
        axis = np.arange(x0, x1 + step / 2, step)
        shape = (axis.size, 2, 2)
        return {
            "axes": (axis, np.array([0.0, 1.0]), np.array([0.0, 1.0])),
            "fields": {"T": np.full(shape, value)},
        }

    def test_single_spacing_is_stitched_unchanged(self) -> None:
        """Multi-rank decomposition still merges; refinement is the only new case."""
        blocks = [self._payload(0.0, 3.0, 1.0, 1.0), self._payload(4.0, 7.0, 1.0, 2.0)]
        kept = npz_block_merge.select_finest(blocks)
        self.assertEqual(len(kept), 2)

    def test_finest_spacing_wins_and_coarse_blocks_are_dropped(self) -> None:
        coarse = self._payload(0.0, 8.0, 2.0, 1.0)
        fine = self._payload(2.0, 6.0, 0.5, 2.0)
        kept = npz_block_merge.select_finest([coarse, fine])
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(float(np.diff(kept[0]["axes"][0])[0]), 0.5)
        self.assertTrue(np.allclose(kept[0]["fields"]["T"], 2.0))

    def test_stitching_overlapping_blocks_raises_rather_than_corrupting(self) -> None:
        """Why select_finest exists.  stitch_blocks assumes disjoint bricks; handed
        a nested pair it refuses (good -- a refined run would fail loudly at export,
        not silently write a ragged grid).  select_finest is what makes it not
        reach that point."""
        coarse = self._payload(0.0, 8.0, 2.0, 1.0)
        fine = self._payload(2.0, 6.0, 0.5, 2.0)
        with self.assertRaisesRegex(ValueError, "does not align"):
            npz_block_merge.stitch_blocks([coarse, fine])
        # ...and with selection first, the same input exports cleanly.
        (gx, _, _), merged = npz_block_merge.stitch_blocks(
            npz_block_merge.select_finest([coarse, fine])
        )
        self.assertTrue(np.allclose(np.diff(gx), 0.5))
        self.assertTrue(np.allclose(merged["T"], 2.0))


if __name__ == "__main__":
    unittest.main()
