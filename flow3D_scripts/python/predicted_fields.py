#!/usr/bin/env python3
"""Make a model prediction measurable by `melt_metrics`, unchanged.

`melt_metrics` takes any Mapping with `x`, `y`, `z` (metres, metal at z<0),
`time_s`, `Temperature` (K) and `Fraction Of Fluid`. A frame dump from a PINO
eval carries `pred_T`/`pred_alpha` (normalised, no coordinates), so this is the
whole adapter: rescale T, gate it on alpha, and re-attach coordinates from the
source simulation's own NPZ. No new metric, no new band -- the scoring stays
`melt_metrics.score_point` / `score_anchors.score_row`.

Two things that are not cosmetic:

* `postprocessing._apply_prediction_postprocessing` fills predicted T with
  T_boil/ref wherever pred_alpha < 0.5 (alpha_masking is true on both anchorbox
  models) and the dump is taken AFTER that fill. An ungated `T > solidus` mask
  therefore selects the vapour cavity as if it were pool: 6.3x more cells on the
  one preview volume that was measured. `alpha_gate` is what stops that.
* `kelvin` is the model's recorded ref_scales.T. Pass the number
  `recorded_kelvin_factor()` returns; never derive it from a metrics.json
  (pred_max is the boiling fill, so pred_T_max_K/pred_max gives 2808, not 3000).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def predicted_case(pred_T: np.ndarray, pred_alpha: np.ndarray,
                   source: Mapping[str, np.ndarray], *,
                   kelvin: float, alpha_gate: float = 0.5) -> dict[str, np.ndarray]:
    """A melt_metrics-shaped dict for one case's prediction.

    `pred_T`/`pred_alpha` are (nt, nx, ny, nz) slices of an eval frame dump.
    `source` is the case's own exported NPZ (for x/y/z/time_s and the cell size).
    A served grid smaller than the source is assumed to be a top crop in z
    (`crop_z_top` drops gas cells) and a beam-centred window in x; y is never
    cropped by any profile in this project.
    """
    pred_T = np.asarray(pred_T, dtype=float)
    pred_alpha = np.asarray(pred_alpha, dtype=float)
    if pred_T.shape != pred_alpha.shape:
        raise ValueError(f"T {pred_T.shape} and alpha {pred_alpha.shape} disagree")
    nt, nx, ny, nz = pred_T.shape

    x = np.asarray(source["x"], dtype=float)
    y = np.asarray(source["y"], dtype=float)
    z = np.asarray(source["z"], dtype=float)
    if len(y) != ny:
        raise ValueError(f"y is {len(y)} in the source, {ny} in the prediction")
    if len(z) < nz:
        raise ValueError(f"prediction has {nz} z-cells, source only {len(z)}")
    z = z[:nz]                      # crop_z_top removes gas cells off the top
    if len(x) != nx:                # moving reference frame: beam-relative window
        dx = float(x[1] - x[0])
        x = (np.arange(nx) - nx // 2) * dx

    t = np.asarray(source["time_s"], dtype=float)
    if len(t) != nt:                # time-averaged samples have no source frame
        t = np.linspace(float(t[0]), float(t[-1]), nt)

    temperature = pred_T * float(kelvin)
    temperature[pred_alpha < alpha_gate] = 0.0
    return {"x": x, "y": y, "z": z, "time_s": t,
            "Temperature": temperature,
            "Fraction Of Fluid": np.clip(pred_alpha, 0.0, 1.0)}


def _self_check() -> None:
    """Round-trip: the adapter must reproduce the raw NPZ's own geometry.

    Feeds the ground-truth fields back through as if they were a prediction, so
    any coordinate or scaling mistake shows up as a changed depth/width. Compares
    against `fusion_geometry` (the mid-x band) rather than `fusion_geometry_auto`,
    because a prediction carries no `Heat Absorption Rate` to locate the beam.
    """
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import melt_metrics as mm

    p = pathlib.Path(__file__).resolve().parents[1] / (
        "runs/batches/ti64_anchorbox_lhs150_20260830/runs/abx_015_gan129/"
        "npz/abx_015_gan129.npz")
    if not p.exists():
        raise SystemExit(f"self-check needs {p}")
    with np.load(p) as raw:
        src = {k: raw[k] for k in ("x", "y", "z", "time_s")}
        T = np.asarray(raw["Temperature"], dtype=float)
        a = np.asarray(raw["Fraction Of Fluid"], dtype=float)
        want_f = mm.fusion_geometry(dict(raw), solidus_k=1898.0)
        want_d = mm.depression(dict(raw), laser_off_s=float(raw["time_s"][-1]))

    case = predicted_case(T / 3000.0, a, src, kelvin=3000.0)
    got_f = mm.fusion_geometry(case, solidus_k=1898.0)
    got_d = mm.depression(case, laser_off_s=float(case["time_s"][-1]))
    assert abs(got_f.depth_um - want_f.depth_um) < 1e-6, (got_f, want_f)
    assert abs(got_f.width_um - want_f.width_um) < 1e-6, (got_f, want_f)
    assert abs(got_d.depth_um - want_d.depth_um) < 1e-6, (got_d, want_d)

    # The alpha gate is the point: without it the vapour cavity reads as pool.
    ungated = dict(case, Temperature=T)
    assert (mm.fusion_geometry(ungated, solidus_k=1898.0).depth_um
            >= got_f.depth_um), "gate check is not exercising anything"
    print(f"ok  fusion {got_f.depth_um:.1f} x {got_f.width_um:.1f} um  "
          f"depression {got_d.depth_um:.1f} +- {got_d.depth_sd_um:.1f} um  "
          f"cell {mm.cell_size_um(case):.1f} um")


if __name__ == "__main__":
    _self_check()
