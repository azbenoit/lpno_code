#!/usr/bin/env pvpython
"""Export ONLY the final stored FLOW-3D frame as a volume or surface VTK.

A lightweight companion to ``export_all_timesteps_to_vtk.py`` (whose reader and
field setup it reuses) for when only the final solidified state is wanted -- e.g.
to keep a small artifact after the multi-GB FLSGRF result is deleted.  Run with
FLOW-3D POST's bundled pvpython:

    F3D_POST_HOME=<POST dir> "$F3D_POST_HOME/bin/pvpython" \\
      python/export_final_frame_vtk.py runs/<case>/flsgrf1.<case> \\
      runs/<case>/paraview/final --prefix <case> --representation volume
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_all_timesteps_to_vtk as base  # noqa: E402  (import loads the FLSGRF plugin)
from paraview.simple import CellDatatoPointData, SaveData  # noqa: E402


def export_last(result: str, out_dir: str, prefix: str, representation: str) -> Path:
    results = base.Selectedtimeline(registrationName=Path(result).stem, FileName=str(result))
    results.Noninertialreferenceframe = 0
    results.CellArrays = base.CELL_ARRAYS
    times = list(results.TimestepValues)
    if not times:
        raise SystemExit(f"No output time steps found in: {result}")
    last = times[-1]
    if representation == "volume":
        fluid = base.FLSGRFIsoObject(registrationName="Fluid Volume", Input=results)
        fluid.Surface, fluid.Type, fluid.IsoValue = "Fluid", "Isovolume", 0
    else:
        fluid = base.FLSGRFIsoObject(registrationName="Solidified Liquid", Input=results)
        fluid.Surface, fluid.Type, fluid.IsoValue = "Solidified Liquid", "Isosurface", 0.5
    fluid.a3DClip, fluid.Symmetry, fluid.Usethreads, fluid.Colors = 0, "None", 1, base.CELL_ARRAYS
    points = CellDatatoPointData(registrationName="Cell To Point", Input=fluid)
    points.ProcessAllArrays, points.PassCellData = 1, 1
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{prefix}_{representation}_final.vtk"
    points.UpdatePipeline(time=last)
    SaveData(str(dest), proxy=points, ChooseArraysToWrite=1, CellDataArrays=base.CELL_ARRAYS,
             PointDataArrays=base.POINT_ARRAYS, FieldDataArrays=[], FileType="Binary")
    print(f"Wrote final ({representation}) frame at t={last:.8g}: {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result", help="FLOW-3D flsgrf1.<case> file")
    parser.add_argument("output_dir", help="Directory for the final-frame VTK")
    parser.add_argument("--prefix", required=True, help="Output basename")
    parser.add_argument("--representation", choices=["volume", "surface"], required=True)
    args = parser.parse_args()
    export_last(args.result, args.output_dir, args.prefix, args.representation)


if __name__ == "__main__":
    main()
