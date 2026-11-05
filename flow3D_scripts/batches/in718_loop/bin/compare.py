#!/usr/bin/env python3
"""Head-to-head arm comparison on the frozen test set, persisted per round.

Reads roundNN_eval_pks.json (keys "<arm>_m<seed>_<slice>" -> eval pk) and prints
median [min-max] across the 3 seeds for fl IoU and beam-box T MAE in Kelvin.

Kelvin comes from recorded_kelvin_factor -- never derived (three wrong derivations
in one day). fl IoU, not alpha IoU: alpha saturates near 0.99 and ranks nothing.
Writes out/roundNN_compare.json so the trajectory accumulates instead of being
recomputed ad hoc every round.
"""
import argparse, json, pathlib, statistics as st

LOOP = pathlib.Path(__file__).resolve().parent.parent
SLICES = ["all", "extrap", "overlap"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    a = p.parse_args()

    from aiida import load_profile, orm
    load_profile()
    from aiida_pino.console.model_metrics import recorded_kelvin_factor

    evals = json.loads((LOOP / f"round{a.round:02d}_eval_pks.json").read_text())
    rows = {}
    for key, pk in evals.items():
        arm, _, sl = key.rsplit("_", 2)[0], None, key.rsplit("_", 1)[1]
        arm = key.split("_")[0]
        n = orm.load_node(pk)
        if not n.is_terminated:
            print(f"  {key} (pk {pk}) still running")
            return 2
        if n.exit_status:
            print(f"  {key} (pk {pk}) exit {n.exit_status} -- skipped")
            continue
        m = n.outputs.metrics.get_dict(); ev = n.outputs.evaluation
        kf, _ = recorded_kelvin_factor(
            ev.base.attributes.all["evaluation_metadata"], ev.base.extras.all)
        rows.setdefault((arm, sl), []).append(
            (m["fl"]["beambox_iou"], m["T"]["beambox_abs_err_mean"] * kf,
             m["fl"].get("iou_support_frames"), m["fl"]["iou_mean"]))

    out = {}
    print(f"  round {a.round}  (frozen test set, median [min-max] over seeds)")
    print(f"  {'arm':<9}{'slice':<9}{'fl beambox_iou':>22}{'fl iou_mean':>22}"
          f"{'BB MAE K':>18}{'sup':>7}")
    for (arm, sl) in sorted(rows, key=lambda k: (k[0], SLICES.index(k[1]))):
        v = rows[(arm, sl)]
        iou = [x[0] for x in v]; mae = [x[1] for x in v]
        fio = [x[3] for x in v]
        sup = {x[2] for x in v}
        if len(sup) > 1:
            print(f"  !! {arm}/{sl} mixed fl support frames {sup} -- rows not comparable")
        out[f"{arm}|{sl}"] = {"beambox_iou": iou, "iou_mean": fio, "mae_k": mae,
                              "support": sorted(map(str, sup))}
        print(f"  {arm:<9}{sl:<9}"
              f"{st.median(iou):>10.3f} [{min(iou):.3f}-{max(iou):.3f}]"
              f"{st.median(fio):>10.3f} [{min(fio):.3f}-{max(fio):.3f}]"
              f"{st.median(mae):>8.0f} [{min(mae):.0f}-{max(mae):.0f}]"
              f"{sorted(sup)[0]:>7}")

    (LOOP / "out").mkdir(exist_ok=True)
    (LOOP / "out" / f"round{a.round:02d}_compare.json").write_text(
        json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
