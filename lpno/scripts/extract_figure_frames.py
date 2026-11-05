#!/usr/bin/env python3
"""Pull the frames a manuscript panel needs out of an eval's `eval_frames_dump.npz`.

WHY A SCRIPT. The dump is a bare concatenation of every scored frame -- `pred_T`,
`true_T` and the two alphas, no case labels and no times -- so picking "case 026 at
0.925 ms" out of it is a two-step lookup that is easy to get subtly wrong. Doing it
by hand once already produced two mis-split videos that both looked plausible
(2026-08-22): the dump is ordered ALPHABETICALLY by case file name, not by
the order the cases were passed to `collection create`.

So the ordering is never assumed here. `check_segments` re-derives the boundaries
from discontinuities in the data and refuses if they disagree with the frame counts
read from the simulations themselves.

    python3 scripts/extract_figure_frames.py --dump DUMP.npz --out figure_frames.npz \
        --case mbc40_026=39091@0.925 --case mbc40_031=39192@1.025
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from render_case_video import check_segments  # noqa: E402  - one implementation, not two


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--case", action="append", required=True,
                    help="NAME=SIMPK@T_MS, repeatable; order is irrelevant (sorted by NAME)")
    ap.add_argument("--counts", default="",
                    help="comma-separated DUMPED frames per case in alphabetical order, when "
                         "striding does not give a uniform ceil/floor")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="the eval's PINO_EVAL_FRAME_STRIDE; the dump then holds every Nth "
                         "frame and the times must be subsampled to match")
    ap.add_argument("--profile", default=None, help="AiiDA profile (default: the configured default profile)")
    a = ap.parse_args(argv)

    spec = {}
    for item in a.case:
        name, rest = item.split("=", 1)
        pk, t_ms = rest.split("@", 1)
        spec[name] = (int(pk), float(t_ms))
    # The dump is in alphabetical case order; everything below follows that, not argv.
    names = sorted(spec)

    from aiida import load_profile, orm
    load_profile(a.profile)

    times, counts = {}, []
    for name in names:
        sim = orm.load_node(spec[name][0])
        got = sim.get_simulation_metadata().get("case_name")
        if got != name:
            raise SystemExit(f"pk {spec[name][0]} is case {got!r}, not {name!r}")
        with sim._open_payload_archive() as z:  # noqa: SLF001
            t_all = np.asarray(z["time_s"])
        # A strided eval dumps every Nth frame, so the TIME axis must be subsampled the
        # same way or the chosen instant maps to the wrong volume. The per-case dumped
        # count is not always ceil(n/stride) -- measured 2026-09-03, one case of six gave
        # floor -- so the caller passes the real counts and we only align the times here.
        times[name] = t_all[:: max(1, int(a.frame_stride))]
        counts.append(len(times[name]))

    with np.load(a.dump) as z:
        true_t = z["true_T"]
        if a.counts:
            counts = [int(c) for c in a.counts.split(",")]
            print(f"using caller-supplied dumped counts {counts}")
            for name, n in zip(names, counts):
                times[name] = times[name][:n]
        if len(true_t) != sum(counts):
            raise SystemExit(
                f"dump holds {len(true_t)} frames, the {len(names)} cases hold {sum(counts)} "
                f"({counts}). A strided eval cannot be indexed by source frame number.")
        check_segments(true_t, counts)      # refuses on a boundary disagreement
        out, start = {}, 0
        for name, n in zip(names, counts):
            t_ms = times[name] * 1e3
            want = spec[name][1]
            i = int(np.argmin(np.abs(t_ms - want)))
            print(f"{name}: frame {i} at {t_ms[i]:.4f} ms (asked {want} ms), "
                  f"dump index {start + i} of {len(true_t)}")
            for key, arr in (("pred_T", z["pred_T"]), ("true_T", true_t),
                             ("pred_alpha", z["pred_alpha"]), ("true_alpha", z["true_alpha"])):
                out[f"{name}_{key}"] = np.squeeze(arr[start + i])
            out[f"{name}_frame"] = np.asarray(i)
            out[f"{name}_t_s"] = times[name]
            start += n

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
