#!/usr/bin/env python3
"""Stitch FLOW-3D domain-decomposed structured blocks into one global grid.

A multi-rank run (``ranks > 1``) decomposes the single rectangular mesh into one
block per rank.  Once the reader's ghost shell is stripped, each block's real
cells are a disjoint, axis-aligned brick of the global grid, so the bricks tile
the domain exactly.  This rebuilds the global cell-centre axes and scatters each
block's fields into place, so the structured NPZ export works for a multi-rank
run as well as a single-rank one.

Pure numpy (no paraview), so the merge can be unit-tested without FLOW-3D POST.
"""

from __future__ import annotations

import numpy as np


def _merge_axis(block_coords: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Sorted union of 1D cell-centre coords; returns (axis, match tolerance).

    Real cells never overlap between blocks, so the union is already unique; the
    tolerance (a quarter of the smallest cell spacing) only guards float wobble
    when matching a block back onto the global axis."""
    allc = np.sort(np.concatenate([np.asarray(c, dtype=np.float64) for c in block_coords]))
    if allc.size == 0:
        return allc, 1e-12
    diffs = np.diff(allc)
    positive = diffs[diffs > 0]
    tol = 0.25 * float(positive.min()) if positive.size else 1e-12
    keep = [allc[0]]
    for value in allc[1:]:
        if value - keep[-1] > tol:
            keep.append(value)
    return np.asarray(keep), tol


def _offset(global_axis: np.ndarray, block_axis: np.ndarray, tol: float) -> int:
    """Index where this block's cells start in the global axis."""
    start = int(np.searchsorted(global_axis, block_axis[0] - tol))
    segment = global_axis[start : start + block_axis.size]
    if segment.size != block_axis.size or not np.allclose(segment, block_axis, atol=tol, rtol=0.0):
        raise ValueError("a decomposed block's cell-centre axis does not align with the global grid")
    return start


def _spacing(block: dict) -> float:
    """Smallest cell spacing in a block, over whichever axes have >1 cell."""
    steps = []
    for coords in block["axes"]:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.size > 1:
            steps.append(float(np.diff(coords).min()))
    return min(steps) if steps else float("inf")


def select_finest(blocks: list[dict], rtol: float = 1e-6) -> list[dict]:
    """Keep only the blocks at the finest spacing present.

    A domain-decomposed run gives one spacing across every block, so this is a
    no-op and stitch_blocks merges them as before.  A *nested refinement* run
    gives two: the fine block overlaps the coarse one at half (or a third of) its
    spacing, which stitch_blocks rejects outright -- the blocks no longer tile a
    grid.  The fine block is the one that covers the melt, and is the only region
    the metrics read, so the coarse far field is dropped rather than resampled.

    Dropping is deliberate: resampling the coarse block onto the fine grid would
    invent data at the interface and make the exported NPZ lossy, which is exactly
    what the archive format exists to avoid."""
    if not blocks:
        return []
    finest = min(_spacing(b) for b in blocks)
    return [b for b in blocks if _spacing(b) <= finest * (1.0 + rtol)]


def stitch_blocks(blocks: list[dict]) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    """Merge ghost-stripped structured blocks into one global grid.

    Each block is ``{"axes": (x, y, z), "fields": {name: array}}`` where every
    field array has shape ``(bx, by, bz)`` or ``(bx, by, bz, ncomp)`` matching the
    block's axis lengths.  Returns ``((gx, gy, gz), {name: global array})`` with the
    same axis order.  Raises ``ValueError`` if the blocks do not tile a complete
    rectangular grid (a gap would silently leave cells unset)."""
    if not blocks:
        raise ValueError("no blocks to stitch")
    gx, tol_x = _merge_axis([b["axes"][0] for b in blocks])
    gy, tol_y = _merge_axis([b["axes"][1] for b in blocks])
    gz, tol_z = _merge_axis([b["axes"][2] for b in blocks])
    global_shape = (gx.size, gy.size, gz.size)
    covered = np.zeros(global_shape, dtype=bool)
    merged: dict = {}
    for block in blocks:
        ax, ay, az = (np.asarray(c, dtype=np.float64) for c in block["axes"])
        ox, oy, oz = _offset(gx, ax, tol_x), _offset(gy, ay, tol_y), _offset(gz, az, tol_z)
        box = (slice(ox, ox + ax.size), slice(oy, oy + ay.size), slice(oz, oz + az.size))
        covered[box] = True
        for name, array in block["fields"].items():
            if array.shape[:3] != (ax.size, ay.size, az.size):
                raise ValueError(f"field '{name}' shape {array.shape} does not match its block axes")
            if name not in merged:
                merged[name] = np.zeros(global_shape + array.shape[3:], dtype=array.dtype)
            merged[name][box] = array
    if not covered.all():
        raise ValueError(
            f"decomposed blocks do not tile the grid: {int((~covered).sum())} of {covered.size} cells unfilled"
        )
    return (gx, gy, gz), merged
