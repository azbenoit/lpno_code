"""Steady-state (P-V, time-averaged) ground-truth loader for /infer/gt_compare.

Numpy-only (no torch), so it is cheap and safe in stub mode. The steady models
are served on a moving-frame, super-resolved grid (see adapter_pino); the ground
truth is the native 10 µm ``var_1_time_steps30_avg30`` cache per split under
``settings.steady_gt_data_dir`` (the calibrated 123-case set by default; the
legacy ``original_straight_path_data`` cache is the same layout). The 3D
comparison interpolates it up to the prediction grid so both display at the
same resolution. var_1 exists for every split, so all train/val/test P-V points
get a GT panel, and the same small files back the P-V catalog (``catalog``).

Masking matches the served prediction's postprocess (adapter_pino): T is filled
toward 3123 K where Fraction-Of-Fluid < 0.5 via a smooth tanh gate; alpha and fl
are gated the same way. Identical convention to the random-path / straight GT.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from .gt import mask_fields, nearest_pv


# Catalog cache per GT root, invalidated when the split files change (mtime).
_CATALOG: dict[str, tuple[tuple, list[dict]]] = {}

SPLITS = ("test", "train", "val")  # match priority: the held-out set first
CATALOG_ORDER = ("train", "val", "test")  # listing order of /gt_catalog (test dots drawn last)


def _split_files(root: Path) -> list[tuple[str, str]]:
    """(split, npz path) of every var_1 steady cache under root, in split priority."""
    return [
        (split, f)
        for split in SPLITS
        for f in sorted(glob.glob(str(root / split / "var_1_time_steps30_avg30*.npz")))
    ]


def catalog(root: str | Path) -> list[dict]:
    """One entry per steady case across train/val/test: ``{name, p_watts,
    v_scan_mps, split, file, index}``, P/V read from the small var_1 caches
    (present in every split; P/V are preprocessing-independent). Cached per
    root until a split file changes. The client plots one point per case,
    coloured by split; a click sends the case P/V to /infer/gt_compare."""
    root = Path(root)
    files = _split_files(root)
    stamp = tuple((f, Path(f).stat().st_mtime_ns) for _s, f in files)
    key = str(root.resolve())
    hit = _CATALOG.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    out: list[dict] = []
    for split, f in files:
        with np.load(f) as data:
            p_arr = np.asarray(data["P"])
            v_arr = np.asarray(data["V_scan"])
        for i in range(int(p_arr.shape[0])):
            p = float(p_arr[i].flat[0])
            v = float(v_arr[i].flat[0])
            out.append(
                {
                    "name": f"{p:.0f}W_{v * 1000:.0f}mps",
                    "p_watts": p,
                    "v_scan_mps": v,
                    "split": split,
                    "file": f,
                    "index": i,
                }
            )
    out.sort(key=lambda e: CATALOG_ORDER.index(e["split"]))  # stable: case order kept within a split
    _CATALOG[key] = (stamp, out)
    return out


def load_steady_gt(
    root: str | Path, p_watts: float, v_scan_mps: float
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[float, float], str] | None:
    """Return ``(gt_dict, (x, y, z) axes in metres, (P, V), split)`` for the
    nearest steady case within tolerance, or ``None`` when no case matches.

    Uses the native 10 µm mesh time-average (var_1), the resolution the model
    was trained on; the var_2 cache is a 2x-upsampled derivative and the
    separate fine_mesh_vtk sims are physically slightly different, so neither
    is used as GT -- the 3D comparison interpolates this GT up to the
    prediction grid instead. var_1 exists for every split, so all
    train/val/test P-V points get a panel. Splits are searched in priority
    order (test, then train, val), the nearest case within the box winning
    inside a split file. ``gt_dict`` has masked ``T`` (Kelvin), ``alpha`` and
    ``fl`` (0..1), each on the case's own physical grid.
    """
    entries = catalog(root)
    for _split, f in _split_files(Path(root)):
        best = nearest_pv([e for e in entries if e["file"] == f], p_watts, v_scan_mps)
        if best is None:
            continue
        i = int(best["index"])
        with np.load(f) as d:
            t_raw = np.asarray(d["Temperature"][i, 0], dtype=np.float32)
            fof = np.asarray(d["Fraction Of Fluid"][i, 0], dtype=np.float32)
            x = np.asarray(d["x"][i, 0, :, 0, 0], dtype=np.float32)
            y = np.asarray(d["y"][i, 0, 0, :, 0], dtype=np.float32)
            z = np.asarray(d["z"][i, 0, 0, 0, :], dtype=np.float32)
        return mask_fields(t_raw, fof), (x, y, z), (best["p_watts"], best["v_scan_mps"]), best["split"]

    return None
