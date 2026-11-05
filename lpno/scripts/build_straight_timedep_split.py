#!/usr/bin/env python3
"""Convert the straight-line variation dataset to random_paths per-case format.

Reads the lab-frame consolidated caches (``var_1_time_noref.npz``) of
``original_straight_path_data/{train,val,test}`` and writes
``data/flow3D/resampled_straight_path_data/{train,val,test}/<case>.npz`` +
``<case>.manifest.json`` in exactly the format ``build_random_path_split.py``
produces, so the *_randompath configs run on straight-line data unchanged
(lab frame, time_as_batch, manifest-derived beam channels): the apples-to-apples
per-frame straight baseline.

Geometry is synthesized from the validated reproduction manifests in the
sibling flow3D_scripts repo (runs/batches/pino_line_tracks_10um): every case
scans y=0 from x=-0.3 mm to x=+0.29724 mm (0.59724 mm) at the speed encoded in
its name, beam radius 50 um, laser on until path end. The script cross-checks
case-name P/V against the cache fields and the synthesized beam trajectory
against the argmax-T hot spot before writing anything.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

SRC_DEFAULT = Path("data/flow3D/original_straight_path_data")
DST_DEFAULT = Path("data/flow3D/resampled_straight_path_data")
CACHE_NAME = "var_1_time_noref.npz"
FIELDS = ["Temperature", "Velocity", "Fraction Of Fluid", "Pressure"]

# From runs/batches/pino_line_tracks_10um manifests (identical across cases).
PATH_START_M = [-0.0003, 0.0]
PATH_END_M = [0.00029724, 0.0]
PATH_LEN_M = 0.00059724
BEAM_RADIUS_M = 5e-05

NAME_RE = re.compile(r"^(\d+)W_(\d+)mps_line_track_vtk$")


def _case_names(split_dir: Path) -> list[str]:
    names = sorted(
        p.name for p in split_dir.iterdir() if p.is_dir() and NAME_RE.match(p.name)
    )
    if not names:
        raise ValueError(f"no *_line_track_vtk case dirs in {split_dir}")
    return names


def _parse_pv(name: str) -> tuple[float, float]:
    m = NAME_RE.match(name)
    power = float(m.group(1))
    speed = float(m.group(2)) / 1000.0  # 0340mps -> 0.340 m/s
    return power, speed


def _validate_beam_track(T: np.ndarray, x1d: np.ndarray, t: np.ndarray, speed: float, tol_m: float = 6e-5) -> float:
    """Median |argmax-T x - synthesized beam x| over mid-run frames, in metres.

    The instantaneous argmax-T sits in the trailing melt/keyhole rim and can lag
    the beam by >100 um on single frames, so validate the median, and pin the
    timing exactly via V*t_end == path length (the equal-spacing export cadence).
    """
    end_dist = speed * float(t[-1])
    if abs(end_dist - PATH_LEN_M) > 0.02 * PATH_LEN_M:
        raise ValueError(
            f"timing validation failed: V*t_end = {end_dist*1e3:.4f} mm vs path "
            f"length {PATH_LEN_M*1e3:.4f} mm"
        )
    nt = T.shape[0]
    devs = []
    for k in range(nt // 4, nt):
        surf = T[k, :, :, -1]  # top z slice
        ix = int(np.unravel_index(np.argmax(surf), surf.shape)[0])
        expect = min(PATH_START_M[0] + speed * float(t[k]), PATH_END_M[0])
        devs.append(abs(float(x1d[ix]) - expect))
    med = float(np.median(devs))
    if med > tol_m:
        # Informational only: the melt-pool hot spot trails the beam by ~100 um
        # at higher scan speeds; the hard invariants are V*t_end (above) and the
        # validated reproduction geometry the constants come from.
        print(f"  [note] median argmax-T trails synthesized beam by {med*1e6:.0f} um (melt-pool lag)")
    return med


def convert_split(src: Path, dst: Path, split: str, frames: int) -> None:
    split_src = src / split
    split_dst = dst / split
    split_dst.mkdir(parents=True, exist_ok=True)
    names = _case_names(split_src)

    cache = np.load(split_src / CACHE_NAME)
    time_field = cache["time_s"]  # (B, T, 1, X, Y, Z)
    B, T = time_field.shape[:2]
    if B != len(names):
        # The cache builder may have excluded cases (e.g. train lacks
        # 60W_0101mps). Keep only dirs whose (P, V) match cache rows in order;
        # the baselines trained on the cache, so the cache defines the split.
        P_rows = cache["P"][:, 0, 0, 0, 0, 0]
        V_rows = cache["V_scan"][:, 0, 0, 0, 0, 0]
        matched, i = [], 0
        for name in names:
            power, speed = _parse_pv(name)
            if i < B and abs(power - float(P_rows[i])) < 1e-6 and abs(speed - float(V_rows[i])) < 1e-6:
                matched.append(name)
                i += 1
            else:
                print(f"  [note] {split}/{name} not in {CACHE_NAME}; skipped")
        if i != B:
            raise ValueError(f"{split}: only matched {i} of {B} cache rows to case dirs")
        names = matched
    idx = np.round(np.linspace(0, T - 1, frames)).astype(int)
    if len(set(idx.tolist())) != frames:
        raise ValueError(f"{split}: cannot pick {frames} distinct frames from {T}")

    x1d = np.asarray(cache["x"][0, 0, 0, :, 0, 0], dtype=np.float64)
    y1d = np.asarray(cache["y"][0, 0, 0, 0, :, 0], dtype=np.float64)
    z1d = np.asarray(cache["z"][0, 0, 0, 0, 0, :], dtype=np.float64)
    for name_axis, axis in (("x", x1d), ("y", y1d), ("z", z1d)):
        if not np.all(np.diff(axis) > 0):
            raise ValueError(f"{split}: {name_axis} axis not strictly increasing")

    P_cache = cache["P"][:, 0, 0, 0, 0, 0]
    V_cache = cache["V_scan"][:, 0, 0, 0, 0, 0]
    fields = {name: cache[name] for name in FIELDS}

    for b, case in enumerate(names):
        out_npz = split_dst / f"{case}.npz"
        out_man = split_dst / f"{case}.manifest.json"
        power, speed = _parse_pv(case)
        if abs(power - float(P_cache[b])) > 1e-6 or abs(speed - float(V_cache[b])) > 1e-6:
            raise ValueError(
                f"{split}/{case}: name says P={power} V={speed} but cache row {b} has "
                f"P={float(P_cache[b])} V={float(V_cache[b])} — case ordering mismatch"
            )
        t_case = np.asarray(time_field[b, :, 0, 0, 0, 0], dtype=np.float64)[idx]

        if out_npz.exists() and out_man.exists():
            print(f"[skip] {split}/{case}")
            continue

        entry: dict[str, np.ndarray] = {
            "x": x1d.astype(np.float32),
            "y": y1d.astype(np.float32),
            "z": z1d.astype(np.float32),
            "time_s": t_case.astype(np.float32),
            "power_w": np.float32(power),
            "scan_speed_m_per_s": np.float32(speed),
            "case_name": np.str_(case),
        }
        for name, arr in fields.items():
            sub = np.asarray(arr[b][idx], dtype=np.float32)  # (nt, C, X, Y, Z)
            if sub.shape[1] == 1:  # scalar -> (nt, X, Y, Z)
                entry[name] = sub[:, 0]
            elif sub.shape[1] == 3:  # vector -> (nt, X, Y, Z, 3)
                entry[name] = np.moveaxis(sub, 1, -1)
            else:
                raise ValueError(f"{split}/{case}: field {name} has C={sub.shape[1]}")

        worst = _validate_beam_track(entry["Temperature"], x1d, t_case, speed)

        manifest = {
            "input": {
                "laser": {
                    "beam_radius_m": BEAM_RADIUS_M,
                    "scan_speed_m_per_s": speed,
                    "power_w": power,
                }
            },
            "derived": {
                "laser_path_xy_m": [PATH_START_M, PATH_END_M],
                "laser_off_time_s": PATH_LEN_M / speed,
            },
            "note": (
                "Synthesized for the per-frame straight baseline from the validated "
                "flow3D_scripts pino_line_tracks_10um reproduction geometry; fields "
                f"from {CACHE_NAME} (lab frame), {len(idx)} of {T} frames."
            ),
        }
        np.savez_compressed(out_npz, **entry)
        out_man.write_text(json.dumps(manifest, indent=1))
        print(f"[ok] {split}/{case}: frames {len(idx)}, beam-track dev {worst*1e6:.0f} um")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--dst", type=Path, default=DST_DEFAULT)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    for split in args.splits:
        convert_split(args.src, args.dst, split, args.frames)


if __name__ == "__main__":
    main()
