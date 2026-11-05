#!/usr/bin/env python3
"""Provenance lineage of one reported number (fig:lineage), as stored.

Every box is a node of the deposited archive (lpno_provenance.aiida), labelled with the first eight characters of its
UUID, walked back from evaluation cc341977: the FNO on the variable-length free-form paths
(uni_fno_rand150_b2_12ch_35k_s43), whose 18.45 K substrate MAE is its row of tab:paths-accuracy. Links drawn are the
links that exist (read from the rebuilt archive on 2026-11-03):

  run config dfceef0f -> Flow3DGenerateWorkChain 96635f56 -> simulation fdd9ef08 (r150np_010)
    (the workchain calls the ingest job RawFlow3DToDatasetCalculation 8a551a23, which creates the simulation)
  simulation -> test split 4c1a4251 (through compose_dataset, with the other 21 test simulations)
  train split f1a2a76f -> TrainModelWorkChain d2c12595 -> model 81e5ec80
  model + test split -> EvaluateModelWorkChain 6e6bc5f7 -> evaluation cc341977 -> the number
    (6e6bc5f7 is called by the EvalOnlyWorkChain 2e2b9011 wrapper, which returns the same evaluation)

Colours follow fig:prov-chain: green holds field arrays, blue is metadata only, orange is a process.

Layout: a serpentine of four columns and three rows, placed at full \\textwidth and authored at
figstyle.width_in(1.0), so every label prints at body size. At that size the longer AiiDA class names do not fit four
to a row, so each wraps once at a word boundary of its CamelCase; the empty slot left of row 2 carries the test
split's link down to the evaluation and the colour key.

  LPNO_REPLACE_FIGURES=1 python3 reproducibility/figures/make_lineage_fig.py
"""
import pathlib, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle; figstyle.apply()

ARR, META, PROC = '#A8D5A2', '#AECBEB', '#F5C79A'
FACE = {'arr': ARR, 'meta': META, 'proc': PROC}
EDGE = {'arr': '#4F8A48', 'meta': '#3C6D9E', 'proc': '#C07B33'}
LINK = '0.45'
LW = 1.2

NODES = {  # key: (kind, AiiDA class name as wrapped lines, UUID prefix in the deposited archive, role note)
 'cfg':  ('meta', ['Dict'],                              'dfceef0f', 'Run config'),
 'gen':  ('proc', ['Flow3DGenerate', 'WorkChain'],       '96635f56', ''),
 'sim':  ('arr',  ['PinoSimulation', 'Data'],            'fdd9ef08', 'One of 150'),
 'test': ('meta', ['PinoDataset', 'CollectionData'],     '4c1a4251', 'Test split, 22'),
 'train':('meta', ['PinoDataset', 'CollectionData'],     'f1a2a76f', 'Train split, 120'),
 'trwc': ('proc', ['TrainModel', 'WorkChain'],           'd2c12595', ''),
 'model':('arr',  ['PinoModelData'],                     '81e5ec80', ''),
 'evwc': ('proc', ['EvaluateModel', 'WorkChain'],        '6e6bc5f7', ''),
 'eval': ('arr',  ['PinoModel', 'EvaluationData'],       'cc341977', ''),
}
PLACE = {'cfg': (0, 0), 'gen': (1, 0), 'sim': (2, 0), 'test': (3, 0),
         'train': (3, 1), 'trwc': (2, 1), 'model': (1, 1),
         'evwc': (1, 2), 'eval': (2, 2)}

# geometry in authored points (the axes span the canvas one-to-one)
WPT = figstyle.width_in(1.0) * 72.0
M = 2.0                        # outer margin
BW = 164.0                     # box width: the widest wrapped line is 151 pt at 20 pt type
COLG = (WPT - 2 * M - 4 * BW) / 3
PITCH = 1.1 * figstyle.FONT    # line pitch inside a box
BH = 4 * PITCH + 10.0          # four lines (class, class, pk, note) plus padding
GAP12, GAP23 = 34.0, 28.0      # row gaps; the first carries the return corridor
HPT = 2 * M + 3 * BH + GAP12 + GAP23
COLS = [M + BW / 2 + i * (BW + COLG) for i in range(4)]
ROWS = [HPT - M - BH / 2, HPT - M - 1.5 * BH - GAP12, HPT - M - 2.5 * BH - GAP12 - GAP23]


def node(ax, key):
    kind, cls, pk, note = NODES[key]
    x, y = COLS[PLACE[key][0]], ROWS[PLACE[key][1]]
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                                boxstyle='round,pad=0,rounding_size=6', lw=LW,
                                facecolor=FACE[kind], edgecolor=EDGE[kind], zorder=3))
    lines = [(c, dict(color='k')) for c in cls] + [(pk, dict(color='0.30'))]
    if note:
        lines.append((note, dict(color='0.40', style='italic')))
    top = y + (len(lines) - 1) / 2 * PITCH
    for i, (s, kw) in enumerate(lines):
        ax.text(x, top - i * PITCH, s, ha='center', va='center_baseline', zorder=4, **kw)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, lw=LW, color=LINK,
                                 shrinkA=0, shrinkB=1, zorder=2))


def lineage():
    fig = plt.figure(figsize=(WPT / 72.0, HPT / 72.0), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WPT); ax.set_ylim(0, HPT); ax.axis('off')
    for k in NODES:
        node(ax, k)
    P = {k: (COLS[c], ROWS[r]) for k, (c, r) in PLACE.items()}
    def h(a, b):                                   # horizontal, either direction
        (xa, ya), (xb, _) = P[a], P[b]
        s = 1 if xb > xa else -1
        arrow(ax, (xa + s * BW / 2, ya), (xb - s * BW / 2, ya))
    def v(a, b):
        (xa, ya), (_, yb) = P[a], P[b]
        arrow(ax, (xa, ya - BH / 2), (xa, yb + BH / 2))
    for a, b in (('cfg', 'gen'), ('gen', 'sim'), ('sim', 'test')): h(a, b)
    for a, b in (('train', 'trwc'), ('trwc', 'model')): h(a, b)
    v('model', 'evwc')
    h('evwc', 'eval')
    # the one return link: down out of the test split, left along the gap under row 1,
    # down the left edge of the empty slot, then right into the evaluation workflow
    ycorr = ROWS[0] - BH / 2 - GAP12 / 2
    xcorr = COLS[0] - BW / 2 + 14.0
    ax.plot([P['test'][0], P['test'][0]], [ROWS[0] - BH / 2, ycorr], color=LINK, lw=LW, zorder=2)
    ax.plot([P['test'][0], xcorr], [ycorr, ycorr], color=LINK, lw=LW, zorder=2)
    ax.plot([xcorr, xcorr], [ycorr, ROWS[2]], color=LINK, lw=LW, zorder=2)
    arrow(ax, (xcorr, ROWS[2]), (P['evwc'][0] - BW / 2, ROWS[2]))
    # the number the chain ends in, beside the evaluation that holds it
    xe = P['eval'][0] + BW / 2
    ax.plot([xe, xe + COLG], [ROWS[2], ROWS[2]], color='0.55', lw=LW, zorder=2)
    ax.text(xe + COLG + 4, ROWS[2], 'Substrate MAE\n18.45 K', ha='left', va='center',
            color='0.12', linespacing=1.1)
    # colour key in the empty slot of row 2, right of the return link
    for i, (k, lab) in enumerate((('arr', 'Field arrays'), ('meta', 'Metadata'), ('proc', 'Process'))):
        y = ROWS[1] + (1 - i) * 1.2 * PITCH
        x = xcorr + 18.0
        ax.add_patch(FancyBboxPatch((x, y - 8), 16, 16, boxstyle='round,pad=0,rounding_size=3',
                                    lw=LW, facecolor=FACE[k], edgecolor=EDGE[k], zorder=3))
        ax.text(x + 22, y, lab, ha='left', va='center_baseline', color='0.35', zorder=4)
    if figstyle.save(fig, 'lineage'):   # the PNG preview follows the guarded PDF
        fig.savefig(figstyle.FIG_DIR / 'lineage.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    lineage()
