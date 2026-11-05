#!/usr/bin/env python3
"""fig:al-rounds: test-set error of the FNO models after each acquisition round.

Two panels, beam-box MAE and melt-pool IoU. Three model series: the maximin half, the bandit
half and the joint model (both halves combined), each retrained from scratch every round on
the Ti-6Al-4V set plus the IN718 picks so far. Each is scored on the 24-simulation IN718 test
set (solid line, filled marker) and the 38-simulation Ti-6Al-4V test set (dashed line, open
marker). Round 0 is the Ti-6Al-4V-only FNO (no IN718 data), scored on both.

Default (since 2026-10-13): the alloy-label FNO models, data/al_rounds_alloyflag.json, one row
per model and test set, three seeds (42, 43, 44) per round and series. The marker is the mean
and an error bar spans the three seeds; at these axis ranges every bar is at most about the size
of its marker (the round-1 IN718 beam-box bars of the joint and maximin series reach 3-5 K past
it). Each row names the deposited AiiDA evaluation it comes from; the melt-pool IoU scores each
case with its own alloy's solidus/liquidus window (iou_ti64_window keeps the earlier value,
which scored IN718 with the Ti-6Al-4V window). Round 0 is scored with its
own training normalization.

LPNO_AL_ROUNDS=previous plots the earlier alloy-blind points, one seed (42) per point, from
the deposit's provenance logs:
  data_deposit/provenance_logs/provenance_log.csv         round 0 (fno_steady_edge75) and round 5 (the seed-42
                                     models fno_twoalloy_{maximin,bandit,combined}_r05_m1)
  data_deposit/provenance_logs/provenance_log_rounds.csv  rounds 1-4 (fno_{maximin,bandit,joint}_r0N_s42), the
                                     per-round retrains of 2026-10-08, trained from the deposit
                                     and evaluated through AiiDA (the deposit's addition)
Columns used: beambox_mae_K and meltpool_iou, the convention of every table. The round-1 models
include the five replacement cases per half, so they differ from the loop's first round-1
models (v1), which were trained before those cases existed.

    python3 reproducibility/figures/make_al_rounds_fig.py --refresh [--rounds-log PATH]   # re-read the logs
    LPNO_REPLACE_FIGURES=1 python3 reproducibility/figures/make_al_rounds_fig.py
"""
import csv
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

DATA = HERE / "data" / "al_rounds.json"
DATA_ALLOYFLAG = HERE / "data" / "al_rounds_alloyflag.json"
USE_PREVIOUS = os.environ.get("LPNO_AL_ROUNDS") == "previous"
# the public code repo beside the manuscript (LPNO_CODE_REPO overrides), found by walking
# up, so a checkout nested inside the manuscript folder finds it too
_CODE = os.environ.get("LPNO_CODE_REPO") or next(
    (str(p / "lpno_code") for p in HERE.parents if (p / "lpno_code").is_dir()), "")
DEPOSIT_DOCS = pathlib.Path(_CODE) / "data_deposit" / "provenance_logs"
TESTS = {"IN718": "09e2a9c5", "Ti64": "3c7af352"}          # scored-dataset uuid prefixes
SERIES = [("maximin", figstyle.PALETTE[0], "o", "fno_twoalloy_maximin_r05_m1", "fno_maximin_r0{r}_s42"),
          ("bandit", figstyle.PALETTE[1], "s", "fno_twoalloy_bandit_r05_m1", "fno_bandit_r0{r}_s42"),
          ("joint", figstyle.PALETTE[2], "^", "fno_twoalloy_combined_r05_m1", "fno_joint_r0{r}_s42")]
ROUND0_MODEL = "fno_steady_edge75"
FRAC, ASPECT = 0.49, 0.62
LEGEND_FRAC = 1.0                 # six entries on one line need the full text width


def _rows(path):
    return list(csv.DictReader(open(path)))


def _pick(rows, model, test, required=True):
    hits = [r for r in rows if r["model"] == model and r["scored_dataset_uuid"].startswith(TESTS[test])]
    if not hits and not required:
        print(f"  not yet in the log: {model} on {test}")
        return None
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} rows for {model} on {test}; expected exactly one")
    r = hits[0]
    return {"bb_mae_k": float(r["beambox_mae_K"]), "iou": float(r["meltpool_iou"]),
            "evaluation_uuid": r["evaluation_uuid"], "model_uuid": r["model_uuid"]}


def refresh(rounds_log):
    main = _rows(DEPOSIT_DOCS / "provenance_log.csv")
    rounds = _rows(rounds_log)
    out = {"source": "data_deposit/provenance_logs/provenance_log.csv (round 0, round 5) and provenance_log_rounds.csv "
                     "(rounds 1-4); beambox_mae_K and meltpool_iou; seed 42",
           "round0": {t: _pick(main, ROUND0_MODEL, t) for t in TESTS}, "series": {}}
    for name, _, _, r5_model, model in SERIES:
        out["series"][name] = {t: {str(r): _pick(rounds, model.format(r=r), t, required=False) for r in range(1, 5)}
                               for t in TESTS}
        for t in TESTS:
            out["series"][name][t]["5"] = _pick(main, r5_model, t)
    DATA.write_text(json.dumps(out, indent=1))
    print("wrote", DATA)


def load_alloyflag():
    """The alloy-label runs in the shape plot() reads, with the seeds' range beside each mean."""
    rows = json.loads(DATA_ALLOYFLAG.read_text())
    d = {"round0": {}, "series": {name: {t: {} for t in TESTS} for name, *_ in SERIES}}
    for t in TESTS:
        r0 = [r for r in rows if r["half"] == "ti64_only" and r["test"] == t]
        assert len(r0) == 1, (t, len(r0))
        d["round0"][t] = {"bb_mae_k": r0[0]["bb_mae_k"], "iou": r0[0]["iou"]}
        for name, *_ in SERIES:
            for rnd in range(1, 6):
                g = [r for r in rows if r["half"] == name and r["round"] == rnd and r["test"] == t]
                assert len(g) == 3, (name, rnd, t, len(g))
                d["series"][name][t][str(rnd)] = {
                    k: float(np.mean([r[k] for r in g])) for k in ("bb_mae_k", "iou")} | {
                    f"{k}_lo": min(r[k] for r in g) for k in ("bb_mae_k", "iou")} | {
                    f"{k}_hi": max(r[k] for r in g) for k in ("bb_mae_k", "iou")}
    return d


def plot():
    figstyle.apply()
    d = json.loads(DATA.read_text()) if USE_PREVIOUS else load_alloyflag()
    out = []
    for key, ylabel, stem, ylim in (("bb_mae_k", "BB MAE (K)", "al_rounds_bbmae", (0.0, 500.0)),
                                    ("iou", "Melt-pool IoU", "al_rounds_iou", (0.0, 1.0))):
        fig, ax = figstyle.panel(frac=FRAC, aspect=ASPECT)
        for k, (name, colour, marker, _, _) in enumerate(SERIES):
            dx = (k - 1) * 0.1
            for test, ls, face in (("IN718", "-", colour), ("Ti64", "--", "white")):
                pts = d["series"][name][test]
                rounds = sorted(int(r) for r in pts)
                x = np.array(rounds, float) + dx
                # a round not yet evaluated is a gap in the line, never bridged
                y = np.array([pts[str(r)][key] if pts[str(r)] else np.nan for r in rounds])
                ax.plot([0, x[0]], [d["round0"][test][key], y[0]], color=colour, lw=1.5, ls=":")
                ax.plot(x, y, color=colour, ls=ls, lw=2.0, marker=marker, ms=8,
                        markerfacecolor=face, markeredgecolor=colour, markeredgewidth=1.8)
                if all(pts[str(r)] and f"{key}_lo" in pts[str(r)] for r in rounds):   # range over the seeds
                    lo = np.array([pts[str(r)][f"{key}_lo"] for r in rounds])
                    hi = np.array([pts[str(r)][f"{key}_hi"] for r in rounds])
                    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=colour, elinewidth=1.2,
                                capsize=3, capthick=1.2, zorder=1)
        for test, face in (("IN718", "0.25"), ("Ti64", "white")):
            ax.plot(0, d["round0"][test][key], "D", color="0.25", ms=8, markerfacecolor=face,
                    markeredgewidth=1.8, zorder=3)
        ax.set_xticks(range(0, 6))
        ax.set_xlim(-0.4, 5.4)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Acquisition round")
        ax.set_ylabel(ylabel)
        out.append(figstyle.save(fig, stem))
    out.append(legend_strip())
    return out


def legend_strip():
    """The figure's one legend (2026-10-06), a strip placed above both panels. Two
    lines since the test sets were added: colour and marker name the model on the first,
    line style and marker fill name the test set on the second. Six entries at 10 pt do not
    fit one line of the text width."""
    models = [plt.Line2D([], [], ls="", marker="D", color="0.25", ms=8, label="Ti-6Al-4V only")]
    for name, colour, marker, _, _ in SERIES:
        models.append(plt.Line2D([], [], ls="", color=colour, marker=marker, ms=8, label=name.capitalize()))
    tests = [plt.Line2D([], [], color="0.25", ls="-", lw=2.0, marker="o", ms=7, label="IN718 test set"),
             plt.Line2D([], [], color="0.25", ls="--", lw=2.0, marker="o", ms=7, markerfacecolor="white",
                        markeredgewidth=1.8, label="Ti-6Al-4V test set")]
    w = figstyle.width_in(LEGEND_FRAC)
    fig = plt.figure(figsize=(w, 0.85), constrained_layout=False)
    for handles, y in ((models, 0.76), (tests, 0.24)):
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, y), ncol=len(handles), frameon=False,
                   handlelength=2.2, columnspacing=1.6, handletextpad=0.5, borderpad=0.0)
    return figstyle.save(fig, "al_rounds_legend")


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        args = sys.argv[sys.argv.index("--refresh") + 1:]
        log = pathlib.Path(args[args.index("--rounds-log") + 1]) if "--rounds-log" in args \
            else DEPOSIT_DOCS / "provenance_log_rounds.csv"
        refresh(log)
    else:
        print(plot())
