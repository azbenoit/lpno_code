#!/usr/bin/env python3
"""RIGNO vs GAOT on every split where both were evaluated, with the interface metrics.

The question the manuscript needs answered: does either architecture DOMINATE, or do
they trade? Whole-domain T MAE cannot tell them apart -- on split 21285 they sit 0.45 K
apart while their peak deficits differ by 600 K -- so this pairs every eval by
(dataset, architecture) and reports the region- and extreme-resolved metrics side by
side.

Pairing rule: an eval is comparable only against another eval on the SAME dataset node
with the SAME frame stride. A strided eval and a full-split eval are not comparable --
the strided sample misses the hottest frames and biases tail deficit low -- so runs
carrying `eval_frame_stride` are reported separately, never merged.

Metrics, and why each is here:
  abs_err_mean      the headline; included to show what it hides
  tail_deficit_max_K  true_max - pred_max, positive = too cold. Noise-sensitive (max
                    over frames); prefer tail_deficit_K where present
  tail_deficit_K    p99.9-based, the robust variant
  beambox_mae_T     error inside 2 beam radii of the laser, z<=0 -- the hard part
  surface_hd95_um   melt-boundary distance, 95th percentile
  fl iou            phase-boundary agreement

Usage:
    python3 scripts/compare_architectures.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

# NB: these are the ACTUAL stored key names. An earlier version of this script looked
# for `beambox_mae_T`/`surface_hd95_um`, which do not exist, and therefore reported the
# interface metrics as absent from all 148 evaluations when in fact many carry them.
# Beam-box and surface metrics are stored PER FIELD (T, alpha, fl), each with its own
# `beambox_*`/`surface_*` block, not as one flat `*_T` key.
METRICS = (
    'abs_err_mean',            # whole-domain headline
    'substrate_abs_err_mean',  # gas-excluded -- the region convention the paper uses
    'beambox_abs_err_mean',    # the hard part: within 2 beam radii, z<=0
    'beambox_rmse',
    'surface_abs_err_mean',    # the z=0 surface plane
    'rmse',
    'tail_deficit_max_K',      # true_max - pred_max, positive = too cold
    'thermal_headroom_K',      # margin between predicted peak and the melt threshold
    'pred_T_max_K', 'true_T_max_K', 'can_reach_melt',
    'rmse_over_mae',           # peak-sensitivity proxy: high = error concentrated
)
FL_METRICS = ('iou_mean', 'beambox_iou', 'iou_support_frames', 'iou_empty_gt_frames',
              'beambox_abs_err_mean')
REF_T = 3000.0
KELVIN = {'abs_err_mean', 'rmse'}  # normalised -> multiply by REF_T


def architecture_of(label: str, config: str) -> str:
    text = f'{label} {config}'.lower()
    if 'gaot' in text:
        return 'GAOT'
    if 'rigno' in text:
        return 'RIGNO'
    if 'fno' in text:
        return 'FNO'
    return '?'


def collect():
    from aiida import load_profile, orm
    load_profile()

    qb = orm.QueryBuilder().append(
        orm.Data, filters={'node_type': {'like': '%PinoModelEvaluation%'}},
        project=['id', 'attributes', 'ctime'])
    rows = []
    for pk, attrs, ctime in qb.all():
        md = (attrs or {}).get('evaluation_metadata', {}) or {}
        metrics = md.get('metrics', {}) or {}
        if not metrics.get('T'):
            continue
        model_pk = md.get('model_pk')
        label, config = '', ''
        try:
            model = orm.load_node(model_pk)
            label = model.label or ''
            creator = next((l.node for l in model.base.links.get_incoming().all()
                            if l.node.node_type.startswith('process.')), None)
            if creator is not None:
                top = creator
                while top.caller is not None:
                    top = top.caller
                for l in top.base.links.get_incoming().all():
                    if isinstance(l.node, orm.Dict) and 'config_name' in l.node.get_dict():
                        config = l.node.get_dict()['config_name']
        except Exception:
            pass
        # FRICTION: evaluation_metadata does NOT record which dataset was scored, so
        # grouping by it requires walking provenance: eval node <- creator calcjob <-
        # `dataset` input. Without this every eval groups under `None` and nothing is
        # comparable.
        dataset_pk = md.get('dataset_pk') or md.get('eval_dataset_pk')
        if dataset_pk is None:
            try:
                node = orm.load_node(pk)
                creator = next((l.node for l in node.base.links.get_incoming().all()
                                if l.node.node_type.startswith('process.')), None)
                seen = set()
                stack = [creator] if creator is not None else []
                while stack and dataset_pk is None:
                    cur = stack.pop()
                    if cur is None or cur.pk in seen:
                        continue
                    seen.add(cur.pk)
                    for l in cur.base.links.get_incoming().all():
                        cls = type(l.node).__name__
                        if 'DatasetCollection' in cls:
                            dataset_pk = l.node.pk
                            break
                    if dataset_pk is None and cur.caller is not None:
                        stack.append(cur.caller)
            except Exception:
                pass
        stride = (metrics.get('eval_frame_stride') or {}).get('stride')
        rows.append({
            'eval_pk': pk, 'model_pk': model_pk, 'label': label, 'config': config,
            'arch': architecture_of(label, config), 'dataset_pk': dataset_pk,
            'stride': stride, 'ctime': ctime.isoformat(),
            'T': {k: metrics['T'][k] for k in METRICS if k in metrics['T']},
            'fl_iou': (metrics.get('fl') or {}).get('iou')
                      or (metrics.get('fl') or {}).get('iou_mean'),
            'fl': {k: (metrics.get('fl') or {})[k]
                   for k in FL_METRICS if k in (metrics.get('fl') or {})},
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', help='also write the raw rows here')
    args = ap.parse_args()

    rows = collect()
    print(f'evaluations with T metrics: {len(rows)}')

    # group by (dataset, stride) so only comparable evals sit together
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset_pk'], r['stride'])].append(r)

    contested = {k: v for k, v in groups.items()
                 if len({r['arch'] for r in v} & {'GAOT', 'RIGNO'}) == 2}
    print(f'groups where BOTH architectures were evaluated: {len(contested)}\n')

    tally = defaultdict(lambda: defaultdict(int))
    for (dataset_pk, stride), members in sorted(contested.items(), key=lambda kv: str(kv[0])):
        tag = f'dataset {dataset_pk}' + (f' stride {stride:g}' if stride else ' (full split)')
        print(f'=== {tag} ===')
        best = {}
        for arch in ('GAOT', 'RIGNO'):
            arm = [m for m in members if m['arch'] == arch]
            if not arm:
                continue
            # best = lowest headline MAE for that architecture on this group
            pick = min(arm, key=lambda m: m['T'].get('abs_err_mean', 9e9))
            best[arch] = pick
            mae = pick['T'].get('abs_err_mean')
            print(f'  {arch:<6} eval {pick["eval_pk"]:>6} {pick["label"][:34]:<36} '
                  f'MAE {mae*REF_T:7.2f} K' if mae is not None else f'  {arch}: no MAE')
            for key in ('substrate_abs_err_mean', 'beambox_abs_err_mean',
                        'beambox_rmse', 'surface_abs_err_mean'):
                if pick['T'].get(key) is not None:
                    print(f'          T.{key:<26} {pick["T"][key]*REF_T:9.2f} K')
            for key in ('tail_deficit_max_K', 'thermal_headroom_K', 'pred_T_max_K',
                        'true_T_max_K', 'rmse_over_mae'):
                if pick['T'].get(key) is not None:
                    print(f'          T.{key:<26} {pick["T"][key]:9.2f}')
            for key in ('iou_mean', 'beambox_iou'):
                if pick.get('fl', {}).get(key) is not None:
                    print(f'          fl.{key:<25} {pick["fl"][key]:9.4f}')
        if len(best) == 2:
            for key in ('abs_err_mean', 'substrate_abs_err_mean',
                        'beambox_abs_err_mean', 'beambox_rmse',
                        'surface_abs_err_mean', 'tail_deficit_max_K'):
                a, b = best['GAOT']['T'].get(key), best['RIGNO']['T'].get(key)
                if a is None or b is None:
                    continue
                tally[f'T.{key}']['GAOT' if abs(a) < abs(b) else 'RIGNO'] += 1
            for key in ('iou_mean', 'beambox_iou'):   # higher is better
                a = best['GAOT'].get('fl', {}).get(key)
                b = best['RIGNO'].get('fl', {}).get(key)
                if a is None or b is None:
                    continue
                tally[f'fl.{key}']['GAOT' if a > b else 'RIGNO'] += 1
        print()

    print('=== dominance tally (lower is better, except fl_iou) ===')
    for key, counts in tally.items():
        print(f'  {key:<22} GAOT {counts["GAOT"]}  RIGNO {counts["RIGNO"]}')
    if not tally:
        print('  (no directly comparable pairs found)')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rows, fh, indent=1)
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
