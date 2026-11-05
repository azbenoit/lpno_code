#!/usr/bin/env pvpython
"""Export every stored FLOW-3D FLSGRF time step as VTK plus a ParaView PVD collection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from xml.sax.saxutils import escape

try:
    from paraview.simple import CellDatatoPointData, LoadPlugin, SaveData  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "Run this script with FLOW-3D POST's bundled pvpython, not the Ubuntu system pvpython."
    ) from exc


POST_HOME = Path(
    os.environ.get("F3D_POST_HOME", "~/.local/opt/flow3d/FLOW3D_POST_2025R1")
).expanduser()
READER_PLUGIN = POST_HOME / "lib/paraview-5.13/plugins/FLSGRFReader/FLSGRFReader.so"
if not READER_PLUGIN.is_file():
    raise SystemExit(f"FLOW-3D POST FLSGRF reader was not found: {READER_PLUGIN}")

LoadPlugin(str(READER_PLUGIN), remote=False, ns=globals())
from paraview.simple import FLSGRFIsoObject, Selectedtimeline  # type: ignore[import-not-found]  # noqa: E402


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
    "vtkGhostType",
]

# FLOW-3D fields are cell-centered; ParaView's Contour and other point-based
# filters need point data, so the export also interpolates cell -> point and
# writes both. vtkGhostType is cell-only bookkeeping, not interpolated to points.
POINT_ARRAYS = [name for name in CELL_ARRAYS if name != "vtkGhostType"]


def write_pvd(path: Path, frames: list[tuple[float, str]]) -> None:
    datasets = "\n".join(
        f'    <DataSet timestep="{time:.17g}" group="" part="0" file="{escape(filename)}"/>'
        for time, filename in frames
    )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        '  <Collection>\n'
        f"{datasets}\n"
        '  </Collection>\n'
        '</VTKFile>\n',
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="FLOW-3D flsgrf1.<case> file")
    parser.add_argument("output_dir", type=Path, help="Directory for .vtk frames and the .pvd collection")
    parser.add_argument("--prefix", help="Output basename (default: input result stem)")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing previously exported frames")
    parser.add_argument(
        "--representation",
        choices=["volume", "surface"],
        default="volume",
        help=(
            "VTK representation to write. volume exports the 3D Fluid isovolume "
            "for surrogate training. surface exports the Solidified Liquid 0.5 "
            "isosurface for geometry inspection."
        ),
    )
    args = parser.parse_args()

    result = args.result.resolve()
    output_dir = args.output_dir.resolve()
    prefix = args.prefix or result.stem
    if not result.is_file():
        raise SystemExit(f"Result file does not exist: {result}")
    if not prefix or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for char in prefix):
        raise SystemExit("--prefix must contain only letters, digits, _ and -")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [*output_dir.glob(f"{prefix}_*.vtk"), output_dir / f"{prefix}.pvd"]
    if not args.overwrite and any(path.exists() for path in existing):
        raise SystemExit(f"Existing export found in {output_dir}; use a new directory or --overwrite.")

    results = Selectedtimeline(registrationName=result.stem, FileName=str(result))
    results.Noninertialreferenceframe = 0
    results.CellArrays = CELL_ARRAYS
    time_steps = list(results.TimestepValues)
    if not time_steps:
        raise SystemExit(f"No output time steps found in: {result}")

    if args.representation == "volume":
        fluid = FLSGRFIsoObject(registrationName="Fluid Volume", Input=results)
        fluid.Surface = "Fluid"
        fluid.Type = "Isovolume"
        fluid.IsoValue = 0
    else:
        fluid = FLSGRFIsoObject(registrationName="Solidified Liquid Surface", Input=results)
        fluid.Surface = "Solidified Liquid"
        fluid.Type = "Isosurface"
        fluid.IsoValue = 0.5
    fluid.a3DClip = 0
    fluid.Symmetry = "None"
    fluid.Usethreads = 1
    fluid.Colors = CELL_ARRAYS

    # Interpolate the cell fields to points (PassCellData keeps the exact cell
    # values) so each VTK carries both, and ParaView Contour works on the result.
    point_data = CellDatatoPointData(registrationName="Cell To Point", Input=fluid)
    point_data.ProcessAllArrays = 1
    point_data.PassCellData = 1

    frames: list[tuple[float, str]] = []
    for index, time in enumerate(time_steps):
        filename = f"{prefix}_{index:05d}.vtk"
        output_path = output_dir / filename
        point_data.UpdatePipeline(time=time)
        SaveData(
            str(output_path),
            proxy=point_data,
            ChooseArraysToWrite=1,
            CellDataArrays=CELL_ARRAYS,
            PointDataArrays=POINT_ARRAYS,
            FieldDataArrays=[],
            FileType="Binary",
        )
        frames.append((time, filename))
        print(f"[{index + 1}/{len(time_steps)}] t={time:.8g}: {output_path.name}")

    collection = output_dir / f"{prefix}.pvd"
    write_pvd(collection, frames)
    print(f"Wrote {len(frames)} frames and ParaView collection: {collection}")


if __name__ == "__main__":
    main()
