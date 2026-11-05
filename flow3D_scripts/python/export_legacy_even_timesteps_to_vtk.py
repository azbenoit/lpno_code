#!/usr/bin/env pvpython
"""Export the old 60-frame FLOW-3D Fluid-isovolume VTK sequence.

This matches the historical PINO data-prep behavior: choose evenly spaced
stored FLOW-3D result indices from the last stored time back to the first, and
name each VTK by ``int(time_s * 1e6)``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    import numpy as np
    from paraview.simple import LoadPlugin, SaveData  # type: ignore[import-not-found]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="FLOW-3D flsgrf1.<case> file")
    parser.add_argument("output_dir", type=Path, help="Directory for legacy .vtk frames")
    parser.add_argument("--prefix", help="Output basename. Default is the result case name.")
    parser.add_argument("--frames", type=int, default=60, help="Number of evenly indexed stored frames to write")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing existing VTK frames")
    args = parser.parse_args()

    result = args.result.resolve()
    output_dir = args.output_dir.resolve()
    prefix = args.prefix or result.name.split(".", 1)[-1]
    if not result.is_file():
        raise SystemExit(f"Result file does not exist: {result}")
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if not prefix or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for char in prefix):
        raise SystemExit("--prefix must contain only letters, digits, _ and -")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(output_dir.glob(f"{prefix}_*.vtk")):
        raise SystemExit(f"Existing legacy export found in {output_dir}; use a new directory or --overwrite.")

    results = Selectedtimeline(registrationName=result.stem, FileName=str(result))
    results.Noninertialreferenceframe = 0
    results.CellArrays = CELL_ARRAYS

    fluid = FLSGRFIsoObject(registrationName="Fluid", Input=results)
    fluid.Surface = "Fluid"
    fluid.Type = "Isovolume"
    fluid.IsoValue = 0
    fluid.a3DClip = 0
    fluid.Box = "Box"
    fluid.Symmetry = "None"
    fluid.Usethreads = 1
    fluid.Colors = CELL_ARRAYS

    time_steps = list(results.TimestepValues)
    if not time_steps:
        raise SystemExit(f"No output time steps found in: {result}")
    selected_count = min(args.frames, len(time_steps))
    indices = np.linspace(len(time_steps) - 1, 0, selected_count, dtype=int)
    selected_times = [time_steps[int(round(index))] for index in indices]

    for index, time in enumerate(selected_times, start=1):
        output_path = output_dir / f"{prefix}_{int(time * 1e6)}.vtk"
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing frame: {output_path}")
        fluid.UpdatePipeline(time=time)
        SaveData(
            str(output_path),
            proxy=fluid,
            ChooseArraysToWrite=0,
            PointDataArrays=[],
            CellDataArrays=CELL_ARRAYS,
            FieldDataArrays=[],
            VertexDataArrays=[],
            EdgeDataArrays=[],
            RowDataArrays=[],
            Writetimestepsasfileseries=0,
            WriteSeriesMetaFile=0,
            FileType="Binary",
        )
        print(f"[{index}/{selected_count}] t={time:.8g}: {output_path.name}")

    print(f"Wrote {selected_count} legacy VTK frames to {output_dir}")


if __name__ == "__main__":
    main()
