#!/usr/bin/env python3
"""Panels for section 3.4 (surrogate against experiment), as manuscript PDFs.

Emits four files into Figs/, one LaTeX figure in a 2x2 grid, every panel placed
at 0.49 of the line and authored at figstyle.width_in(0.49) on one shared canvas:
  closing_parity_{b,c}.pdf  parity: surrogate/measured, surrogate/solver (depths only;
                            since 2026-10-12 one model for both alloys, an alloy-label joint
                            model, RIGNO by default (LPNO_CLOSING_ARCH=fno|gaot|rigno; the three
                            were compared on 2026-10-13); LPNO_CLOSING_MODEL=previous restores the earlier pair,
                            Ti-6Al-4V from the single-alloy FNO and IN718 from the joint GAOT)
  closing_regime.pdf        depth error against simulated cavity aspect ratio
  closing_xray.pdf          the NIST radiograph with both boundaries and the gap bar

Sources, per series:
  * IN718 (seven anchor conditions): data/in718_joint_gaot_depths.json, the depths of the
    joint model's evaluation.
  * Ti-6Al-4V (fourteen replicates) and the radiograph panel: computed by geometry()
    and xray_from_data() from the validation runs, model dumps and NIST frames when they
    are present; otherwise drawn from data/figure_data/closing_ti64_replicates.json,
    closing_xray.json and closing_xray_radiograph.png.

    LPNO_REPLACE_FIGURES=1 python3 reproducibility/figures/make_closing_figs.py

Regenerate with LPNO_REPLACE_FIGURES=1.
"""
import csv, glob, json, os, pathlib, re, subprocess, sys, tempfile
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as _pe
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle; figstyle.apply()

# Sibling repos: resolved beside the repo root (two levels up from this file), so the
# layout works both privately and in the public cut where figures/ sits beside them.
# LPNO_FLOW3D_ROOT / LPNO_PINO_ROOT override when the checkouts live elsewhere.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
FLOW3D = pathlib.Path(os.environ.get('LPNO_FLOW3D_ROOT', _ROOT / 'flow3D_scripts'))
PINO = pathlib.Path(os.environ.get('LPNO_PINO_ROOT', _ROOT / 'lpno'))
sys.path.insert(0, str(FLOW3D / 'python'))
sys.path.insert(0, str(PINO / 'src'))
try:   # only the from-data paths need these
    import melt_metrics as mm
    import score_anchors as _sa  # noqa: E402  (AiiDA fallback for reclaimed exports)
    from predicted_fields import predicted_case
except ImportError:
    mm = _sa = predicted_case = None

# Colour key shared by every panel (2026-09-16): the surrogate is coloured by alloy,
# amber for Ti-6Al-4V and pink for IN718, and FLOW-3D is grey wherever it appears.
C_TI, C_IN, C_SOLVER = figstyle.PALETTE[1], figstyle.PALETTE[3], '0.45'

W = FLOW3D / 'runs/anchorbox_model_validation'
BATCH = FLOW3D / 'runs/batches/ti64_anchorbox_lhs150_20260830'
ANCH = FLOW3D / 'validation/anchors'
REF_T, SOLIDUS_K, BAND, MIN_CELLS = 3000.0, 1898.0, 0.30, 5
REC = HERE / 'data' / 'figure_data'
REC_TI64 = REC / 'closing_ti64_replicates.json'
REC_XRAY = REC / 'closing_xray.json'
REC_XRAY_PNG = REC / 'closing_xray_radiograph.png'

# ---- one canvas for all four panels ------------------------------------------------
# Placed at 0.49 of the line, so authored at width_in(0.49) and printed at REDUCTION.
# Every panel gets the same axes box at the same margins (in authored points), so the
# 2x2 grid lines up: left and bottom margins hold the tick labels and the axis label,
# the top margin a two-row legend. The box height is the radiograph's at equal aspect
# (480 x 250 um), which is the one panel whose box shape is fixed by its data.
FRAC = 0.49
XLIM_XRAY, ZLIM_XRAY = (-305.0, 175.0), (-170.0, 80.0)
# One legend for the whole figure (2026-10-02): it sits in its own panel, in the
# slot the residual panel (closing_regime) held, so the other panels carry none and
# lose the top margin a legend needed. closing_regime is still drawn, with its own
# legend, should it return; set SHARED_LEGEND = False to restore per-panel legends.
SHARED_LEGEND = True
ML, MR, MB, MT = 80.0, 14.0, 56.0, (12.0 if SHARED_LEGEND else 50.0)
FIG_W_PT = figstyle.width_in(FRAC) * 72.0
AX_W_PT = FIG_W_PT - ML - MR
AX_H_PT = AX_W_PT * (ZLIM_XRAY[1] - ZLIM_XRAY[0]) / (XLIM_XRAY[1] - XLIM_XRAY[0])
FIG_H_PT = MB + AX_H_PT + MT
ASPECT = FIG_H_PT / FIG_W_PT
LEG = dict(frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
           labelspacing=0.2, borderpad=0.2, borderaxespad=0.0)


def _panel():
    fig = plt.figure(figsize=(FIG_W_PT / 72.0, FIG_H_PT / 72.0), constrained_layout=False)
    ax = fig.add_axes([ML / FIG_W_PT, MB / FIG_H_PT, AX_W_PT / FIG_W_PT, AX_H_PT / FIG_H_PT])
    return fig, ax


def _legend_above(fig, handles, ncol):
    """Legend centred on the panel, its bottom edge on the axes' top edge."""
    return fig.legend(handles=handles, loc='lower center', ncol=ncol,
                      bbox_to_anchor=(0.5, (MB + AX_H_PT + 3.0) / FIG_H_PT), **LEG)


# ---- data ------------------------------------------------------------------------
CASES = sorted(c for c in
               (pathlib.Path(p).stem for p in glob.glob(str(BATCH / 'runs/*/npz/*.npz')))
               if not c.endswith('_lhs'))

def anchors():
    gan = [e for e in json.loads((ANCH / 'gan_points.json').read_text())
           if 'Ti64' in str(e.get('group'))]
    gal = [p for p in json.loads((ANCH / 'galiegue2025_ti64_targets.json').read_text())['points']
           if p.get('material') == 'Ti64' and not p.get('defocus_mm')]
    return gan, gal

def match(P, V, r0, gan, gal):
    if abs(r0 - 47.5) < 1.0:
        d, e = min(((abs(e['P']-P)/P + abs(e['V']-V)/V, e) for e in gan
                    if abs(e['r0']*1e6 - r0) <= 1.0), key=lambda t: t[0])
        return 'depression', e['depth_um'], None
    d, e = min(((abs(e['power_w']-P)/P + abs(e['speed_m_s']-V)/V, e) for e in gal
                if abs(e['r0_m']*1e6 - r0) <= 1.0), key=lambda t: t[0])
    return 'fusion', e['depth_um'], e.get('width_um')

def geometry():
    """Raw solver and surrogate geometry on the laser-on tail, per replicate."""
    # Retrained steady FNO 86651 (edge75 set). The eval's window is the laser-on
    # window by construction now (frame_transfer.laser_on_frame_stop), so this is
    # the eval dump 86786 plus per-case axes, not a hand-built tail.
    lon = np.load(W / 'dumps/steady_86651_laseron_pred.npz')
    cl = [str(c) for c in lon['cases']]
    design = {r['case']: r for r in csv.DictReader((BATCH / 'design.csv').open())}
    gan, gal = anchors()
    rows = []
    for c in cl:
        i = cl.index(c)
        # local npz/ first, AiiDA payload when the export has been reclaimed off-box
        with _sa.open_case_arrays(BATCH / 'runs' / c) as z:
            if z is None:
                raise FileNotFoundError(f'{c}: no local npz and no single AiiDA simulation')
            ax = {k: np.asarray(z[k]) for k in ('x', 'y', 'z', 'time_s')}
            raw_g = mm.fusion_geometry_auto(z); raw_d = mm.depression(z)
        src = {'x': lon['x'][i], 'y': ax['y'], 'z': lon['z'][i], 'time_s': ax['time_s'][:1]}
        pr = predicted_case(lon['pred_T'][i][None] / REF_T, lon['pred_alpha'][i][None], src, kelvin=REF_T)
        gp = mm.fusion_geometry(pr, track_half_cells=40, solidus_k=SOLIDUS_K)
        kind, md, mw = match(float(design[c]['P_w']), float(design[c]['V_mps']),
                             float(design[c]['r0_um']), gan, gal)
        rows.append(dict(case=c, kind=kind, meas_d=md, meas_w=mw,
                         sim_d=raw_d.depth_um if kind == 'depression' else raw_g.depth_um,
                         mdl_d=float(mm.depression_series(pr)[0]) if kind == 'depression' else gp.depth_um,
                         sim_w=raw_g.width_um, mdl_w=gp.width_um,
                         aspect=raw_d.aspect, flag=(c == 'abx_010_galedge100')))
    return rows

# Which surrogate the parity panels show. 'joint_fno_alloyflag' (default, 2026-10-12): the alloy-label
# joint FNO fno_joint_r05_alloyflag_s42 on both alloys, depths from its evaluation frame dumps on the
# GPU cluster (prepare_closing_depths.py), measured and FLOW-3D depths carried over
# from the earlier files for the same cases. 'previous': the single-alloy FNO and the joint GAOT.
CLOSING_MODEL = os.environ.get('LPNO_CLOSING_MODEL', 'joint_alloyflag')
CLOSING_ARCH = os.environ.get('LPNO_CLOSING_ARCH', 'rigno')   # 2026-10-13: closest to the measurements
# Replicates not plotted: galedge100, whose tabulated depth lies well off its source's own trend, and (2026-10-30)
# lad_r60_v09, the Galiegue 210 W, 0.90 m/s condition, which the solver validation does not score.
LEFT_OUT = ('abx_010_galedge100', 'abx_005_lad_r60_v09')

def ti64_rows():
    """The Ti-6Al-4V replicates: from the runs when they are here, else the committed table."""
    if CLOSING_MODEL == 'joint_alloyflag':
        d = json.loads((HERE / 'data' / f'ti64_joint_{CLOSING_ARCH}_alloyflag_depths.json').read_text())
        old = json.loads(REC_TI64.read_text())['rows']   # cavity aspect ratio, joined by the solver's values
        aspect = {(round(r['meas_d'], 1), round(r['sim_d'], 1)): r['aspect'] for r in old}
        rows = [dict(case=r['case'], kind=r['kind'], meas_d=r['meas_d'], meas_w=None, sim_d=r['sim_d'],
                     mdl_d=r['model_d'], sim_w=float('nan'), mdl_w=float('nan'), flag=False,
                     aspect=aspect.get((round(r['meas_d'], 1), round(r['sim_d'], 1)), float('nan')))
                for r in d['rows'] if r['case'] not in LEFT_OUT]
        return rows, f"joint {CLOSING_ARCH} with the alloy label ({d['model_label']}, {d['evaluation']})"
    if mm is not None and (W / 'dumps/steady_86651_laseron_pred.npz').exists():
        return geometry(), 'computed from the runs (geometry())'
    rows = json.loads(REC_TI64.read_text())['rows']
    return [dict(r, flag=False) for r in rows], f'committed data ({REC_TI64.name})'

def load_in718():
    """The seven measured IN718 anchor conditions (tab:in718-depth), predicted by the
    joint two-alloy GAOT model: reproducibility/figures/data/in718_joint_gaot_depths.json, the depths of the
    joint model's evaluation. Fusion depth only; the IN718 tracks are
    600 um long and unsolidified when the solve ends, so no fusion width is read."""
    name = f'in718_joint_{CLOSING_ARCH}_alloyflag_depths.json' if CLOSING_MODEL == 'joint_alloyflag' \
        else 'in718_joint_gaot_depths.json'
    d = json.loads((HERE / 'data' / name).read_text())
    return [dict(case=r['case'], kind='fusion', meas_d=r['meas_depth_um'], meas_w=None,
                 sim_d=r['flow3d_depth_um'], mdl_d=r['surrogate_depth_um'], sim_w=float('nan'), mdl_w=float('nan'),
                 aspect=r['aspect'], flag=False) for r in d['rows']]


# ---- panels ----------------------------------------------------------------------
def fig_parity(rows, in718=()):
    """Two parity panels on the shared canvas. No titles: LaTeX supplies subcaptions.
    Both carry the marker key, so each reads without the other."""
    # Depths only (2026-09-16): the fusion-width markers were dropped, since the
    # anchor tracks are unsolidified when the solve ends and a width read on them
    # depends on where along the tail it is taken (result_details.md, 2026-09-16).
    pts = []
    for r in rows:
        pts.append(dict(m=r['meas_d'], sim=r['sim_d'], mdl=r['mdl_d'], series='ti64',
                        mk='+' if r['kind'] == 'depression' else 'o', mfc='full', flag=r['flag']))
    for r in in718:
        pts.append(dict(m=r['meas_d'], sim=r['sim_d'], mdl=r['mdl_d'], series='in718', mk='s', mfc='full', flag=False))
    C, CF, CI = C_TI, '0.65', C_IN
    lo, hi = 40.0, 320.0; grid = np.array([lo, hi])   # the scored lengths span 50-240 um
    counts = {}
    # The solver-against-measurement panel is not drawn (2026-09-05): the
    # solver's own agreement is the calibration figure's subject.
    for stem, (yk, xk, xl, yl) in (('closing_parity_b', ('mdl', 'm', 'Experiment', 'Surrogate')),
                                    ('closing_parity_c', ('mdl', 'sim', 'FLOW-3D', 'Surrogate'))):
        fig, ax = _panel()
        ax.fill_between(grid, grid*(1-BAND), grid*(1+BAND), color='0.6', alpha=.18, lw=0)
        ax.plot(grid, grid, 'k-', lw=.9)
        nin = ntot = 0; nin_i = ntot_i = 0
        for p_ in pts:
            if p_['flag']: continue   # the flagged source outlier is omitted from every panel (2026-09-05)
            x, y = p_[xk], p_[yk]
            if not (np.isfinite(x) and np.isfinite(y) and x > 0): continue
            if p_['series'] == 'in718':
                ax.plot(x, y, 's', color=CI, ms=6.0, zorder=3)
                ntot_i += 1; nin_i += abs(y-x)/x <= BAND
                continue
            col = CF if p_['flag'] else C
            if p_['mk'] == '+': ax.plot(x, y, '+', color=col, ms=9, mew=1.8, zorder=4)
            else: ax.plot(x, y, 'o', color=col, ms=6.5, zorder=3, mew=1.5,
                          mfc=(col if p_['mfc'] == 'full' else 'none'))
            if not p_['flag']: ntot += 1; nin += abs(y-x)/x <= BAND
        counts[stem[-1]] = dict(ti64=(nin, ntot), in718=(nin_i, ntot_i))
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ticks = [50, 100, 200, 300]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks]); ax.set_yticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(f'{xl} (µm)'); ax.set_ylabel(f'{yl} (µm)')
        band = Patch(facecolor='0.6', alpha=.18, edgecolor='none', label='±30 %')
        handles = [plt.Line2D([], [], ls='', marker='o', color=C, ms=6.5, label='Ti-6Al-4V melt'),
                   plt.Line2D([], [], ls='', marker='+', color=C, ms=9, mew=1.8, label='Ti-6Al-4V cavity'),
                   plt.Line2D([], [], ls='', marker='s', color=CI, ms=6.0, label='IN718 melt'),
                   band]
        if not SHARED_LEGEND:
            _legend_above(fig, handles, ncol=2)
        figstyle.save(fig, stem); plt.close(fig)
    print('  parity counts:', counts)
    return counts

def fig_regime(rows, in718=()):
    """Depth error against the solver's cavity aspect ratio. Solver and surrogate
    often coincide to within a cell, so the grey solver marker is drawn larger
    underneath and the filled surrogate marker on top (2026-09-16: filled,
    for consistency with (a)/(b)); a coincident pair shows as a grey ring."""
    C_S, C_M, C_I = C_SOLVER, C_TI, C_IN
    fig, ax = _panel()
    ax.axhspan(-BAND*100, BAND*100, color='0.6', alpha=.18, lw=0)
    ax.axhline(0, color='k', lw=.6)
    for r in rows:
        if not r['meas_d'] or r['flag']: continue   # the flagged source outlier is omitted here (2026-09-05)
        x = r['aspect']; es = 100*(r['sim_d']-r['meas_d'])/r['meas_d']; em = 100*(r['mdl_d']-r['meas_d'])/r['meas_d']
        a = 1.0
        ax.plot([x, x], [es, em], color='0.55', lw=1.0, alpha=a, zorder=2)
        if r['kind'] == 'depression':
            ax.plot(x, es, '+', color=C_S, ms=11, mew=2.4, alpha=a, zorder=3)
            ax.plot(x, em, 'x', color=C_M, ms=7.5, mew=2.0, alpha=a, zorder=4)
        else:
            ax.plot(x, es, 'o', color=C_S, ms=10, alpha=a, zorder=3)
            ax.plot(x, em, 'o', color=C_M, ms=6.5, alpha=a, zorder=4)
    for r in in718:   # joint GAOT on the seven measured IN718 conditions, same convention, squares
        x = r['aspect']; es = 100*(r['sim_d']-r['meas_d'])/r['meas_d']; em = 100*(r['mdl_d']-r['meas_d'])/r['meas_d']
        ax.plot([x, x], [es, em], color='0.55', lw=1.0, zorder=2)
        ax.plot(x, es, 's', color=C_S, ms=9, zorder=3)
        ax.plot(x, em, 's', color=C_I, ms=5.8, zorder=4)
    ax.set_xlabel('FLOW-3D cavity aspect ratio')
    ax.set_ylabel('Depth error (%)')   # reference named in the subcaption and caption
    ax.set_xlim(0, 1.75); ax.set_ylim(-45, 50)   # Ti-6Al-4V spans -30 to +32 %, IN718 -25 to +17 %
    ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_yticks([-25, 0, 25, 50])
    ya = 43
    st = dict(color='0.4', va='center', annotation_clip=False,
              arrowprops=dict(arrowstyle='-|>', color='0.4', lw=1.1, shrinkA=2, shrinkB=0))
    ax.annotate('Conduction', xy=(0.02, ya), xytext=(0.80, ya), ha='right', **st)
    ax.annotate('Keyhole', xy=(1.73, ya), xytext=(0.95, ya), ha='left', **st)
    # Grey (larger, underneath) = FLOW-3D; filled, coloured by alloy = surrogate; + / x = cavity depth.
    # The two long entries share a column, so two columns fit the panel width.
    _legend_above(fig, [plt.Line2D([], [], ls='', marker='o', color=C_M, ms=6.5, label='Ti-6Al-4V surrogate'),
                        plt.Line2D([], [], ls='', marker='s', color=C_I, ms=5.8, label='IN718 surrogate'),
                        plt.Line2D([], [], ls='', marker='o', color=C_S, ms=8.5, label='FLOW-3D'),
                        plt.Line2D([], [], ls='', marker='+', color=C_S, ms=11, mew=2.4, label='Cavity')], ncol=2)
    figstyle.save(fig, 'closing_regime'); plt.close(fig)


def xray_from_data():
    """Everything the radiograph panel draws, computed from the NIST frames, the
    FLOW-3D run and the retrained LAB FNO dumps. Same structure as the committed
    closing_xray.json, so fig_xray() draws either."""
    from PIL import Image
    from scipy.ndimage import rotate as _rot
    TILT, SURF, UMPX, XON = 6.50, 92, 2.01, 109.12
    TEXT_BOX = (slice(0, 30), slice(0, 200)); V_MS = 0.7
    X0 = json.loads((FLOW3D / 'runs/validation/t2_nist195/case_manifest.json')
                    .read_text())['input']['laser']['start_point_m'][0]
    _num = lambda f: int(re.search(r'G1200(\d+)\.tif$', str(f)).group(1))
    _tof = lambda f: (_num(f) - 98.634) * 20e-6
    def _frame(f):
        raw = np.asarray(Image.open(f), float).copy(); raw[TEXT_BOX] = np.median(raw)
        return _rot(raw, TILT, reshape=False, order=1, mode='nearest')
    xr = sorted(glob.glob(str(PINO / 'data/experimental/xray/*.tif')))
    ref = np.mean([_frame(g) for g in xr if _tof(g) < 0], axis=0)
    sim = np.load(pathlib.Path.home() / '.flow3d/npz_cache/t2_nist195/npz/t2_nist195.npz')
    t = np.asarray(sim['time_s']); X = np.asarray(sim['x']); Z = np.asarray(sim['z'])
    jy = np.asarray(sim['Fraction Of Fluid']).shape[2] // 2
    gs_all = 1.0 - np.asarray(sim['Fraction Of Fluid'])[:, :, jy, :]
    # Retrained LAB FNO 86677: a per-frame prediction, so it can be read at the
    # radiograph's own instant. The STEADY model 86651 gives ONE moving-frame
    # window ending at the declared path end; it is drawn at the beam position of
    # the chosen frame, which is exactly the steady assumption.
    lab = np.load(W / 'dumps/lab_86677_nist_dump.npz')
    stdy = np.load(W / 'dumps/steady_86651_nist_dump.npz')
    gm_all = 1.0 - lab['pred_alpha'][:, 0, :, jy, :]
    def surface_m(a3):
        prof = a3[0:6, :, :].mean(axis=(0, 1))
        for iz in range(len(prof)-1):
            if prof[iz] >= 0.5 > prof[iz+1]:
                f = (prof[iz]-0.5)/(prof[iz]-prof[iz+1]); return float(Z[iz]+f*(Z[iz+1]-Z[iz]))
        return float('nan')
    tgt = 627e-6
    f = min((g for g in xr if 0 < _tof(g) <= t[-1]), key=lambda g: abs(_tof(g)-tgt))
    t2 = _tof(f); a = _frame(f); bp = XON + (V_MS*t2*1e6)/UMPX
    flat = a/np.maximum(ref, 1e-6)
    k = int(np.argmin(np.abs(t-t2))); beam = X0 + V_MS*t[k]
    dz_off = surface_m(lab['pred_alpha'][k, 0]) - surface_m(lab['true_alpha'][k, 0])
    # The STEADY model's prediction on this case is not usable and is NOT drawn.
    # Eval 87490 (steady 86651 on t2_nist195): fl IoU 0.086, substrate MAE 222 K,
    # against ~0.89 / ~15 K on anchorbox cases. The NIST domain is 90x40x40 (400 um
    # in y and z) where the model trained on 90x60x55 with a 100 um gas layer; the
    # lab model tolerates that, the steady one predicts a cavity to the domain floor
    # behind the beam and temperatures from 97 K to 4204 K. Re-enable a steady
    # overlay once a steady prediction exists on an in-family grid (re-solve the NIST
    # case on the anchorbox domain), not before.
    # Grey-mapped radiograph (2nd-99.5th percentile window), as the gray cmap draws it.
    vmin, vmax = np.percentile(flat, 2), np.percentile(flat, 99.5)
    grey = np.round(255 * np.clip((flat - vmin) / (vmax - vmin), 0, 1)).astype(np.uint8)
    ext = [(0-bp)*UMPX, (a.shape[1]-bp)*UMPX, (a.shape[0]-SURF)*-UMPX, (0-SURF)*-UMPX]
    def _iso(x, z, field):   # the 0.5 iso-line as polylines, exactly what contour() draws
        f_ = plt.figure(); cs = f_.add_subplot().contour(x, z, field, levels=[0.5])
        segs = [np.asarray(s).T.tolist() for s in cs.allsegs[0]]; plt.close(f_); return segs
    solver = _iso((X-beam)*1e6, Z*1e6, gs_all[k].T)
    surrogate = _iso((X-beam)*1e6, (Z-dz_off)*1e6, gm_all[k].T)
    # STEADY model: one moving-frame window (30 steps, so x is 61 cells) with the
    # beam at window index `behind` of legacy_window_split(61). Its x axis is
    # already beam-relative; place it at the beam position of THIS frame. The
    # window's own time is the path end, not t2 -- the steady assumption is that
    # the cavity shape does not depend on that, which is what this overlay tests.
    from lpno.models.pipeline.frame_transfer import legacy_window_split
    st_a = stdy['pred_alpha'][0, 0]; st_t = stdy['true_alpha'][0, 0]
    nxw = st_a.shape[0]; behind, _ahead = legacy_window_split(nxw)
    dx = float(np.diff(X)[0]); xw = (np.arange(nxw) - behind) * dx
    jyw = st_a.shape[1] // 2
    dz_off_t = surface_m(st_a) - surface_m(st_t)
    gt_all = 1.0 - st_a[:, jyw, :]
    # gap bar at the solver's deepest column
    gsurf = np.empty_like(gm_all[k])
    for ix in range(gm_all[k].shape[0]):
        gsurf[ix] = np.interp(Z, Z-dz_off, gm_all[k][ix])
    dz_um = float(np.diff(Z)[0])*1e6
    def floor_z(gas_col, zline):
        """z of the cavity floor: lowest alpha=0.5 crossing, interpolated as contour does."""
        g = gas_col[zline < 0.0]; zz = zline[zline < 0.0]
        hit = np.where(g > 0.5)[0]
        if not hit.size: return np.nan
        i0 = int(hit.min())
        if i0 == 0: return float(zz[0])
        g1, g0 = g[i0], g[i0-1]
        f = (0.5-g0)/(g1-g0) if g1 != g0 else 0.0
        return float(zz[i0-1] + f*(zz[i0]-zz[i0-1]))
    gs_ = gs_all[k] > 0.5
    cols = np.where((gs_[:, Z < 0]).any(axis=1))[0]
    depths = [floor_z(gs_all[k][c], Z) for c in cols]
    ic = int(cols[int(np.nanargmin(depths))])
    z_sim = floor_z(gs_all[k][ic], Z)*1e6; z_mdl = floor_z(gsurf[ic], Z)*1e6
    xc = (X[ic]-beam)*1e6; gap = abs(z_sim-z_mdl)
    z_st = floor_z(np.interp(Z, Z-dz_off_t, gt_all[int(np.argmin(np.abs(xw - (X[ic]-beam))))]), Z)*1e6
    print(f'  lab gap {gap:.0f} um = {int(round(gap/dz_um))} cell(s); surface offset {dz_off*1e6:+.1f} um; '
          f'steady (NOT drawn) floor at that column {z_st:.0f} um vs FLOW-3D {z_sim:.0f} um')
    return dict(xlim=list(XLIM_XRAY), zlim=list(ZLIM_XRAY), t_us=t2*1e6,
                image=grey, image_extent_um=ext,
                solver=solver, surrogate=surrogate,
                bar=dict(x_um=xc, lo_um=min(z_sim, z_mdl), hi_um=max(z_sim, z_mdl)),
                gap_um=gap, ncell=int(round(gap/dz_um)))

def xray_inputs():
    if mm is not None and (W / 'dumps/lab_86677_nist_dump.npz').exists() \
            and glob.glob(str(PINO / 'data/experimental/xray/*.tif')):
        return xray_from_data(), 'computed from the frames and dumps (xray_from_data())'
    from PIL import Image
    d = json.loads(REC_XRAY.read_text())
    d['image'] = np.asarray(Image.open(REC_XRAY_PNG).convert('L'))
    return d, f'committed data ({REC_XRAY.name})'

def fig_xray(d):
    """The radiograph at t = 627 us, FLOW-3D (grey) and surrogate (amber) cavity
    boundaries, and the gap between them at the solver's deepest column."""
    C_S, C_M = C_SOLVER, C_TI   # same key as the other panels: grey solver, amber surrogate
    fig, ax = _panel()
    ax.imshow(d['image'], cmap='gray', vmin=0, vmax=255, aspect='equal',
              extent=d['image_extent_um'], origin='upper')
    halo = [_pe.withStroke(linewidth=3.8, foreground='white')]   # grey stays legible on the dark rim
    for s in d['solver']:
        ax.plot(s[0], s[1], color=C_S, lw=2.2, path_effects=halo)
    for s in d['surrogate']:
        ax.plot(s[0], s[1], color=C_M, lw=2.6)
    b = d['bar']; xc, lo_, hi_ = b['x_um'], b['lo_um'], b['hi_um']
    ax.plot([xc, xc], [lo_, hi_], color='k', lw=2.0, solid_capstyle='butt', zorder=6)
    for zz_ in (lo_, hi_): ax.plot([xc-6, xc+6], [zz_, zz_], color='k', lw=2.0, zorder=6)
    # label centred below the bar, in clear speckle, joined by a thin white leader
    # below the bar, but never into the t-label's corner (bottom-left ~ x<-150, z<-110)
    ylab, xlab = lo_ - 42, xc + 48
    if ylab < -105 and xlab < -100:
        xlab = max(xlab, -60)
    ax.annotate('', xy=(xc, lo_ - 1), xytext=(xlab - 30, ylab + 4),
                arrowprops=dict(arrowstyle='-', color='w', lw=0.8, shrinkA=1, shrinkB=1), zorder=7)
    n = d['ncell']
    ax.text(xlab, ylab, f"{d['gap_um']:.0f} µm ({n} cell{'s' if n != 1 else ''})",
            color='w', ha='center', va='center', zorder=8)
    ax.axhline(0, color='0.3', lw=0.5, ls=':')
    ax.set_xlim(*d['xlim']); ax.set_ylim(*d['zlim'])
    ax.set_xticks([-300, -200, -100, 0, 100]); ax.set_yticks([-100, 0])
    ax.set_xlabel('$x$ (µm)'); ax.set_ylabel('$z$ (µm)')
    ax.text(0.02, 0.06, f"$t$ = {d['t_us']:.0f} µs", transform=ax.transAxes, color='w', va='bottom')
    if not SHARED_LEGEND:
        _legend_above(fig, [plt.Line2D([], [], color=C_S, lw=2.2, label='FLOW-3D', path_effects=halo),
                            plt.Line2D([], [], color=C_M, lw=2.6, label='Surrogate')], ncol=2)
    figstyle.save(fig, 'closing_xray'); plt.close(fig)


def fig_legend():
    """The figure's one legend, on a canvas the size of a panel: the parity markers
    and band of (a, b) and the two boundaries of the radiograph panel."""
    halo = [_pe.withStroke(linewidth=3.8, foreground='white')]
    handles = [plt.Line2D([], [], ls='', marker='o', color=C_TI, ms=6.5, label='Ti-6Al-4V melt-pool depth'),
               plt.Line2D([], [], ls='', marker='+', color=C_TI, ms=9, mew=1.8, label='Ti-6Al-4V cavity depth'),
               plt.Line2D([], [], ls='', marker='s', color=C_IN, ms=6.0, label='IN718 melt-pool depth'),
               Patch(facecolor='0.6', alpha=.18, edgecolor='none', label='±30 % band'),
               plt.Line2D([], [], color=C_SOLVER, lw=2.2, label='FLOW-3D boundary', path_effects=halo),
               plt.Line2D([], [], color=C_TI, lw=2.6, label='Surrogate boundary')]
    fig = plt.figure(figsize=(FIG_W_PT / 72.0, FIG_H_PT / 72.0), constrained_layout=False)
    fig.legend(handles=handles, loc='center', ncol=1, frameon=False, handlelength=1.6,
               handletextpad=0.6, labelspacing=0.35, borderpad=0.0)
    figstyle.save(fig, 'closing_legend'); plt.close(fig)


if __name__ == '__main__':
    rows, src = ti64_rows(); print(f'{len(rows)} Ti-6Al-4V replicates, {src}')
    in718 = load_in718(); print(f'{len(in718)} IN718 anchor conditions ({CLOSING_MODEL}, {CLOSING_ARCH})')
    fig_parity(rows, in718)
    if not SHARED_LEGEND:   # the residual panel is out of fig:closing while the legend is shared
        fig_regime(rows, in718)
    xd, src = xray_inputs(); print(f'radiograph panel {src}'); fig_xray(xd)
    if SHARED_LEGEND:
        fig_legend()
