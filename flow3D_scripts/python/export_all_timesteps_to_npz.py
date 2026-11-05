#!/usr/bin/env pvpython
"""Export every stored FLOW-3D FLSGRF time step to one structured-array NPZ.

FLOW-3D runs the maintained case on a static, rectangular mesh, so the result is
a regular grid.  This reads it straight from the FLSGRF reader (no isosurface, no
resampling), strips the one-cell ghost shell the reader pads around each block,
and writes one compressed ``.npz`` per simulation.  A multi-rank run decomposes
the mesh into one block per rank; the blocks are stitched back into the global
grid (npz_block_merge) before writing, so multi-core runs export too.  That is
far smaller than the per-step VTK series and keeps every stored cell, field, and
time step.

Layout convention (lab frame, no reference-frame change, no time averaging):
  x, y, z        1D cell-centre axes, shape (nx,), (ny,), (nz,)
  time_s         1D stored solver times, shape (T,)
  <field>        (T, nx, ny, nz) scalar, or (T, nx, ny, nz, 3) vector
                 with axis order (x, y, z); FLOW-3D field names kept verbatim
  power_w, scan_speed_m_per_s, case_name   provenance scalars

Requires FLOW-3D POST's bundled pvpython and the FLSGRF reader plug-in.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for npz_block_merge

try:
    import numpy as np
    from paraview.simple import LoadPlugin  # type: ignore[import-not-found]
    from paraview import servermanager as sm  # type: ignore[import-not-found]
    from vtkmodules.util.numpy_support import vtk_to_numpy  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "Run this script with FLOW-3D POST's bundled pvpython, not the Ubuntu system pvpython."
    ) from exc

from npz_block_merge import select_finest, stitch_blocks  # noqa: E402  (numpy-only, no paraview)
from verify_npz_checksums import write_sidecar  # noqa: E402  (stdlib only, safe under pvpython)
from resource_guards import resource_guard  # noqa: E402  (stdlib only, safe under pvpython)


POST_HOME = Path(
    os.environ.get("F3D_POST_HOME", "~/.local/opt/flow3d/FLOW3D_POST_2025R1")
).expanduser()
READER_PLUGIN = POST_HOME / "lib/paraview-5.13/plugins/FLSGRFReader/FLSGRFReader.so"
if not READER_PLUGIN.is_file():
    raise SystemExit(f"FLOW-3D POST FLSGRF reader was not found: {READER_PLUGIN}")

LoadPlugin(str(READER_PLUGIN), remote=False, ns=globals())
from paraview.simple import Selectedtimeline  # type: ignore[import-not-found]  # noqa: E402


# Request every field the reader can materialise; we export the physics ones.
CELL_ARRAYS = [
    "Cell Volume Fraction",
    "Fraction Of Fluid",
    "Heat Absorption Rate",
    "Heat Flux Spatial Distribution",
    "Melt Region",
    "Melt Region Contour",
    "Pressure",
    "Solid Fraction",
    "Temperature",
    "Temperature Gradient At Tgrdout",
    "Vapor Recoil Pressure",
    "Velocity",
    "Volume Fraction After AVRCK Adjustment",
    "X-velocity",
    "Y-velocity",
    "Z-velocity",
]

# Mesh-internal bookkeeping arrays and the per-component velocities, which are
# redundant with the 3-component "Velocity" vector.  Everything else is kept.
SKIP_ARRAYS = {
    "Cell Type",
    "Component Number",
    "vtkGhostType",
    "X-velocity",
    "Y-velocity",
    "Z-velocity",
}


def structured_blocks(source, time: float) -> list:
    """Fetch the FLOW-3D mesh as a list of vtkStructuredGrid blocks.

    A single-rank run is one block; a multi-rank run is one block per rank (the
    decomposition).  Both are merged into the global grid by stitch_blocks."""
    source.UpdatePipeline(time=time)
    dataset = sm.Fetch(source)
    grids = []
    if dataset.GetClassName() == "vtkMultiBlockDataSet":
        for index in range(dataset.GetNumberOfBlocks()):
            block = dataset.GetBlock(index)
            if block is None:
                continue
            if not block.IsA("vtkStructuredGrid"):
                raise SystemExit(f"FLSGRF block {index} is {block.GetClassName()}, not a structured grid.")
            grids.append(block)
    elif dataset.IsA("vtkStructuredGrid"):
        grids.append(dataset)
    else:
        raise SystemExit(f"FLSGRF dataset is {dataset.GetClassName()}, not structured or multiblock.")
    if not grids:
        raise SystemExit("FLSGRF reader returned no structured mesh blocks.")
    return grids


def block_payload(grid) -> dict:
    """One block -> {"axes": (x, y, z), "fields": {name: ghost-stripped array}}."""
    ncx, ncy, ncz = cell_dims(grid)
    sx, sy, sz = real_cell_box(grid, ncx, ncy, ncz)
    gx, gy, gz = cell_axes(grid, ncx, ncy, ncz)
    cell_data = grid.GetCellData()
    fields = {}
    for i in range(cell_data.GetNumberOfArrays()):
        name = cell_data.GetArrayName(i)
        if name in SKIP_ARRAYS:
            continue
        array = cell_data.GetArray(name)
        if array is None:
            continue
        field = reshape_cells(vtk_to_numpy(array).astype(np.float32), ncx, ncy, ncz)
        fields[name] = field[sx, sy, sz]
    return {"axes": (gx[sx], gy[sy], gz[sz]), "fields": fields}


def merged_grid(source, time: float):
    """Read every block at this time and stitch them into one global grid.

    With nested mesh refinement the blocks overlap at two spacings and do not tile
    a grid, so the finest ones are selected first; for a single-spacing run
    (one block, or a multi-rank decomposition) select_finest keeps everything and
    this is the same merge as before.

    Returns ((gx, gy, gz), {field: (nx, ny, nz[, ncomp]) array})."""
    blocks = [block_payload(g) for g in structured_blocks(source, time)]
    return stitch_blocks(select_finest(blocks))


def cell_dims(grid) -> tuple[int, int, int]:
    nxp, nyp, nzp = grid.GetDimensions()
    return nxp - 1, nyp - 1, nzp - 1


def reshape_cells(values: "np.ndarray", ncx: int, ncy: int, ncz: int) -> "np.ndarray":
    """VTK cell order has x fastest; return axis order (x, y, z[, comp])."""
    if values.ndim == 1:
        return values.reshape(ncz, ncy, ncx).transpose(2, 1, 0)
    return values.reshape(ncz, ncy, ncx, values.shape[1]).transpose(2, 1, 0, 3)


def real_cell_box(grid, ncx: int, ncy: int, ncz: int):
    """Index slices that drop the reader's ghost shell (vtkGhostType != 0)."""
    ghost_array = grid.GetCellData().GetArray("vtkGhostType")
    if ghost_array is None:
        return slice(None), slice(None), slice(None)
    ghost = reshape_cells(vtk_to_numpy(ghost_array).astype(np.int64), ncx, ncy, ncz)
    real = ghost == 0
    if not real.any():
        raise SystemExit("Every cell is flagged as a ghost cell; cannot strip the ghost shell.")
    ix = np.where(real.any(axis=(1, 2)))[0]
    iy = np.where(real.any(axis=(0, 2)))[0]
    iz = np.where(real.any(axis=(0, 1)))[0]
    return (
        slice(ix[0], ix[-1] + 1),
        slice(iy[0], iy[-1] + 1),
        slice(iz[0], iz[-1] + 1),
    )


def cell_axes(grid, ncx: int, ncy: int, ncz: int):
    """Cell-centre coordinate axes from the structured point coordinates."""
    nxp, nyp, nzp = grid.GetDimensions()
    pts = vtk_to_numpy(grid.GetPoints().GetData()).reshape(nzp, nyp, nxp, 3)
    xpt, ypt, zpt = pts[0, 0, :, 0], pts[0, :, 0, 1], pts[:, 0, 0, 2]
    return (
        0.5 * (xpt[:-1] + xpt[1:]),
        0.5 * (ypt[:-1] + ypt[1:]),
        0.5 * (zpt[:-1] + zpt[1:]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="FLOW-3D flsgrf1.<case> file")
    parser.add_argument("output_dir", type=Path, help="Directory for the <prefix>.npz file")
    parser.add_argument("--prefix", help="Output basename (default: input result stem)")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing an existing NPZ")
    parser.add_argument("--power-w", type=float, default=float("nan"), help="Laser power recorded for provenance")
    parser.add_argument("--scan-speed-m-per-s", type=float, default=float("nan"), help="Scan speed recorded for provenance")
    parser.add_argument(
        "--frame-stride", type=int, default=1,
        help="Export every Nth stored time step (default 1 = all). Trades time "
             "resolution for peak memory on large meshes; recorded in the NPZ as "
             "frame_stride. Validate the metric against a stride-1 export of a "
             "smaller case before trusting a strided series.")
    args = parser.parse_args()

    result = args.result.resolve()
    output_dir = args.output_dir.resolve()
    prefix = args.prefix or result.stem
    if not result.is_file():
        raise SystemExit(f"Result file does not exist: {result}")
    if not prefix or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for char in prefix):
        raise SystemExit("--prefix must contain only letters, digits, _ and -")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{prefix}.npz"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Existing export found: {out_path}; use a new directory or --overwrite.")

    source = Selectedtimeline(registrationName=result.stem, FileName=str(result))
    source.Noninertialreferenceframe = 0  # lab frame: no reference-frame change
    source.CellArrays = CELL_ARRAYS
    time_steps = list(source.TimestepValues)
    if not time_steps:
        raise SystemExit(f"No output time steps found in: {result}")

    # The mesh is static, so derive geometry once from the first frame; for a
    # multi-rank run the blocks are stitched back into the global grid here.
    (gx, gy, gz), merged0 = merged_grid(source, time_steps[0])
    field_names = sorted(merged0)
    if not field_names:
        raise SystemExit("No exportable physics fields were materialised on the FLSGRF grid.")

    # Accumulate through disk-backed memmaps, not RAM.
    #
    # History: the original code appended every frame to a list and np.stack'd it,
    # peaking at TWICE the series. It was written for ~10^2 steps x 10^5 cells and
    # died on the first genuinely larger mesh -- 5 um spot60_p90_mesh5 is 689 frames
    # x 94 MB = 64.5 GB for one copy, ~129 GB at the stack -- killed at frame 655/689
    # on 2026-07-31. Preallocating removed the doubling but 64.5 GB still exceeded the
    # ~58 GB free, so full resolution was unreachable either way.
    #
    # A memmap per field moves the series to disk: peak *anonymous* memory is one
    # frame (~94 MB here), and np.savez_compressed streams from a memmap in ~16 MB
    # chunks (measured: writing a 0.96 GB memmap grew RSS by 16 MB). The pages the
    # memmap does occupy are file-backed and reclaimable, which is the distinction
    # that matters -- the same holds for FLOW-3D's own write-back cache, where anon
    # RSS stays ~1.2 GB behind a ~13 GB page cache.
    stride = max(1, args.frame_stride)
    selected = list(range(0, len(time_steps), stride))
    selected_times = [time_steps[i] for i in selected]

    frame_bytes = sum(
        int(np.prod(np.shape(merged0[n]))) * np.dtype(merged0[n].dtype).itemsize
        for n in field_names
    )
    need = frame_bytes * len(selected)
    free = shutil.disk_usage(output_dir).free
    # Scratch holds one uncompressed copy; the NPZ lands beside it (~20x smaller).
    # A disk guard: refuses here, warns in the public copy (resource_guards.py). If
    # it warns and the disk does fill, the export fails part-way and the runner
    # keeps flsgrf, so no completed solve is lost.
    if free < need * 1.15:
        resource_guard(
            f"Need ~{need / 1e9:.1f} GB of scratch for {len(selected)} frames but only "
            f"{free / 1e9:.1f} GB is free under {output_dir}. Free space, pass an "
            f"output_dir on a roomier filesystem, or raise --frame-stride.",
            exc=SystemExit,
        )

    scratch = Path(tempfile.mkdtemp(prefix=".export_scratch_", dir=str(output_dir)))
    try:
        stores = {
            name: np.lib.format.open_memmap(
                scratch / f"f{i}.npy", mode="w+",
                dtype=merged0[name].dtype,
                shape=(len(selected),) + np.shape(merged0[name]),
            )
            for i, name in enumerate(field_names)
        }
        for out_index, index in enumerate(selected):
            time = time_steps[index]
            merged = merged0 if index == 0 else merged_grid(source, time)[1]
            for name in field_names:
                if name not in merged:
                    raise SystemExit(f"Field '{name}' vanished at t={time:.8g}; the field set must be constant.")
                stores[name][out_index] = merged[name]
            print(f"[{out_index + 1}/{len(selected)}] t={time:.8g}")
        for store in stores.values():
            store.flush()

        payload = {
            "x": gx.astype(np.float32),
            "y": gy.astype(np.float32),
            "z": gz.astype(np.float32),
            "time_s": np.asarray(selected_times, dtype=np.float64),
            "power_w": np.asarray(args.power_w, dtype=np.float64),
            "scan_speed_m_per_s": np.asarray(args.scan_speed_m_per_s, dtype=np.float64),
            "case_name": np.asarray(prefix),
            # 1 = every stored step. Lets a consumer tell a subsampled series from a
            # complete one without counting frames.
            "frame_stride": np.asarray(stride, dtype=np.int64),
        }
        payload.update(stores)
        shapes = {name: stores[name].shape for name in field_names}
        np.savez_compressed(out_path, **payload)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # Record the digest while the bytes are known good. Without it a later
    # corruption is undetectable short of decompressing everything: on 2026-08-05 a
    # `ti64_random150` case had a CRC-broken `Temperature.npy` that surfaced only
    # when an eval crashed on it days afterwards. Verify with
    # `python3 python/verify_npz_checksums.py <dir>`.
    digest = write_sidecar(out_path)
    print(f"sha256 {digest}  {out_path.name}")

    print(
        f"Wrote {len(selected)} of {len(time_steps)} time steps (stride {stride}) "
        f"to {out_path} (grid {gx.size}x{gy.size}x{gz.size}, fields {sorted(field_names)})"
    )
    print(f"Field shapes: {shapes}")


if __name__ == "__main__":
    main()
