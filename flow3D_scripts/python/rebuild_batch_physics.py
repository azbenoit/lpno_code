#!/usr/bin/env python3
"""Rewrite an existing batch's configs onto the calibrated physics recipe.

Every sweep config written before the 2026-07-28/29 calibration campaign
renders pre-calibration physics: no ``simulation.recoil_model`` (so the solver
uses the legacy analytic recoil), no ``simulation.void_vapor_pressure_pa`` (so
the profile's GUI-relic ``pv0=50650`` stands), and a Ti64 material whose
saturation anchor the campaign replaced. Regenerating training data from those
files silently reproduces the old model -- the failure mode the pre-regeneration
review flagged as Tier A.

This tool rewrites a whole batch in one pass and records what it did, so the
regenerated sweep carries its recipe provenance next to its configs.

    python3 python/rebuild_batch_physics.py <src-batch-dir> <out-dir> \\
        --material materials/ti64_v3_calibrated/ti64_v3_calibrated.json \\
        --recoil-model 2 --void-vapor-pressure-pa 101325

Process conditions (power, speed, spot, mesh, geometry, timing) are copied
verbatim: only the physics recipe changes, so a regenerated case stays
comparable to its pre-campaign twin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fields retired by the campaign; carried by no validated recipe (assumptions A31 v4).
RETIRED_LASER_FIELDS = ("plume_attenuation", "plume_radius_m", "plume_condensation")


def rebuild(src_batch: Path, out_dir: Path, *, material: str, recoil_model: int,
            void_vapor_pressure_pa: float, vapor_pressure_b: float | None = None) -> int:
    """Rewrite every config under ``src_batch`` (recursively) into ``out_dir``.

    Returns the number of cases written. Writes ``rebuild_manifest.json``
    alongside them recording the source and the applied recipe.
    """
    sources = sorted(src_batch.rglob("configs/*.json"))
    if not sources:
        sources = sorted(src_batch.glob("*.json"))
    out_configs = out_dir / "configs"
    out_configs.mkdir(parents=True, exist_ok=True)

    for path in sources:
        config = json.loads(path.read_text())
        config["case_name"] = path.stem
        config["material"] = {"file": material}
        simulation = config.setdefault("simulation", {})
        simulation["recoil_model"] = recoil_model
        simulation["void_vapor_pressure_pa"] = void_vapor_pressure_pa
        laser = config.setdefault("laser", {})
        for field in RETIRED_LASER_FIELDS:
            laser.pop(field, None)
        # A pinned vapor_pressure_b beats the material file, so leaving one in
        # place renders the pre-campaign saturation curve under a calibrated
        # material name. Default is to drop it and let the material govern.
        if vapor_pressure_b is None:
            laser.pop("vapor_pressure_b", None)
        else:
            laser["vapor_pressure_b"] = vapor_pressure_b
        (out_configs / path.name).write_text(json.dumps(config, indent=1) + "\n")

    (out_dir / "rebuild_manifest.json").write_text(json.dumps({
        "source": str(src_batch),
        "cases": len(sources),
        "recipe": {
            "material": material,
            "recoil_model": recoil_model,
            "void_vapor_pressure_pa": void_vapor_pressure_pa,
            "vapor_pressure_b": vapor_pressure_b,
            "retired_fields_stripped": list(RETIRED_LASER_FIELDS),
        },
    }, indent=1) + "\n")
    return len(sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_batch", type=Path, help="batch directory holding the original configs")
    parser.add_argument("out_dir", type=Path, help="where the rebuilt configs are written")
    parser.add_argument("--material", required=True, help="path to the calibrated material JSON")
    parser.add_argument("--recoil-model", type=int, default=2)
    parser.add_argument("--void-vapor-pressure-pa", type=float, default=101325.0)
    parser.add_argument("--vapor-pressure-b", type=float, default=None,
                        help="set when the material's anchor differs from the source configs' B")
    args = parser.parse_args()

    count = rebuild(args.src_batch, args.out_dir, material=args.material,
                    recoil_model=args.recoil_model,
                    void_vapor_pressure_pa=args.void_vapor_pressure_pa,
                    vapor_pressure_b=args.vapor_pressure_b)
    print(f"rebuilt {count} configs -> {args.out_dir/'configs'}")
    print(f"recipe recorded in {args.out_dir/'rebuild_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
