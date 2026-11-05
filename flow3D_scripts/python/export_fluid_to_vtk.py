#!/usr/bin/env pvpython
"""Export the final FLOW-3D material surface from an FLSGRF result to VTK.

Requires the FLOW-3D FLSGRF ParaView reader plug-in.  This plug-in provides
the ``Selectedtimeline`` and ``FLSGRFIsoObject`` ParaView sources; it is not
part of the stock ParaView package.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from paraview.simple import CellDatatoPointData, LoadPlugin, SaveData  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "Run this script with FLOW-3D POST's bundled pvpython, not the "
        "Ubuntu system pvpython."
    ) from exc


POST_HOME = Path(
    os.environ.get(
        "F3D_POST_HOME", "~/.local/opt/flow3d/FLOW3D_POST_2025R1"
    )
).expanduser()
READER_PLUGIN = POST_HOME / "lib/paraview-5.13/plugins/FLSGRFReader/FLSGRFReader.so"
if not READER_PLUGIN.is_file():
    raise SystemExit(f"FLOW-3D POST FLSGRF reader was not found: {READER_PLUGIN}")

LoadPlugin(str(READER_PLUGIN), remote=False, ns=globals())
from paraview.simple import FLSGRFIsoObject, Selectedtimeline  # type: ignore[import-not-found]  # noqa: E402


CELL_ARRAYS = [
    "Temperature",
    "Melt Region",
    "Solid Fraction",
    "Pressure",
    "Heat Absorption Rate",
    "Heat Flux Spatial Distribution",
    "X-velocity",
    "Y-velocity",
    "Z-velocity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="FLOW-3D flsgrf1.<case> file")
    parser.add_argument("output", type=Path, help="Output .vtk file")
    args = parser.parse_args()

    result = args.result.resolve()
    output = args.output.resolve()
    if not result.is_file():
        raise SystemExit(f"Result file does not exist: {result}")
    if output.suffix.lower() != ".vtk":
        raise SystemExit("Output file must use the .vtk extension")

    output.parent.mkdir(parents=True, exist_ok=True)
    results = Selectedtimeline(registrationName=result.stem, FileName=str(result))
    results.Noninertialreferenceframe = 0
    results.CellArrays = CELL_ARRAYS

    time_steps = list(results.TimestepValues)
    if not time_steps:
        raise SystemExit(f"No output time steps found in: {result}")

    # Use the material boundary, not the full Cartesian fluid mesh.  The
    # simulation starts with solidified steel, so this is its workpiece surface.
    fluid = FLSGRFIsoObject(registrationName="Solidified Liquid Surface", Input=results)
    fluid.Surface = "Solidified Liquid"
    fluid.Type = "Isosurface"
    fluid.IsoValue = 0.5
    fluid.a3DClip = 0
    fluid.Symmetry = "None"
    fluid.Usethreads = 1
    # The reader only materializes arrays listed in Colors.  Without this,
    # SaveData writes geometry but no useful FLOW-3D result fields.
    fluid.Colors = CELL_ARRAYS + ["vtkGhostType"]

    # FLOW-3D fields are cell-centered; interpolate cell -> point (PassCellData
    # keeps the exact cell values) so the VTK carries both and ParaView Contour
    # works directly on the surface.
    point_data = CellDatatoPointData(registrationName="Cell To Point", Input=fluid)
    point_data.ProcessAllArrays = 1
    point_data.PassCellData = 1

    final_time = time_steps[-1]
    point_data.UpdatePipeline(time=final_time)
    SaveData(
        str(output),
        proxy=point_data,
        ChooseArraysToWrite=1,
        CellDataArrays=CELL_ARRAYS,
        PointDataArrays=CELL_ARRAYS,
        FieldDataArrays=[],
        FileType="Binary",
    )
    print(f"Wrote final time step {final_time:.8g} to {output}")


if __name__ == "__main__":
    main()
