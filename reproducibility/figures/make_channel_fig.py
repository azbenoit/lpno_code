#!/usr/bin/env python3
"""Fig. 6: the trajectory channels and the two offsets, one PDF per panel.

Synthetic variable-speed S-curve identical to conditioning_channels.ipynb cell 2
(footprint -> beam_dist; time-integrated footprint -> path_dose; leaky
integrators at tau -> the bank). Fields are shown normalized to their own
maximum so one 0-1 bar serves panels (b)-(e); the offsets share a diverging bar.

Every panel file has the same size and the same axes box, so LaTeX lays them
out as a 3x3 grid of subfigures at 0.32\\textwidth that lines up, each with its
own \\subcaption below; the two colour bars are narrow files of the same
height, placed in the third column of rows 2 and 3.

  python3 reproducibility/figures/make_channel_fig.py                 -> Figs/channels_*.pdf
  python3 reproducibility/figures/make_channel_fig.py --sheet OUT.png  -> colour-map comparison sheet only
"""
import argparse, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle; figstyle.apply()

# The conduction response is the SAME channel the datasource builds, so it is
# computed with the same solver instead of being re-implemented here. Try the
# installed package first, then a sibling checkout, then the public cut.
for _cand in (None,
              pathlib.Path(__file__).resolve().parents[2] / 'lpno' / 'src',
              pathlib.Path(__file__).resolve().parents[2] / 'lpno_code'
              / 'lpno' / 'src'):
    try:
        if _cand is not None:
            sys.path.insert(0, str(_cand))
        from lpno.custom_utils import linear_response as lr
        break
    except ImportError:
        continue
else:
    raise ImportError('linear_response not importable; set PYTHONPATH to lpno/src')

SIGMA = 50e-6
TAUS = {"tau100us": 100e-6, "tau1ms": 1e-3, "tau3ms": 3e-3}
nx = ny = 220
x = np.linspace(-0.9e-3, 0.9e-3, nx); y = np.linspace(-0.6e-3, 0.6e-3, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
s = np.linspace(0, 1, 600)
px = -0.75e-3 + 1.5e-3 * s; py = 0.28e-3 * np.sin(2 * np.pi * s)
seg = np.hypot(np.diff(px), np.diff(py)); arc = np.concatenate([[0], np.cumsum(seg)]); frac = arc / arc[-1]
speed = 0.55 - 0.45 * np.exp(-((frac - 0.45) / 0.12) ** 2)
dt_seg = np.diff(arc) / ((speed[:-1] + speed[1:]) / 2)
t = np.concatenate([[0], np.cumsum(dt_seg)]); dt = np.gradient(t)
def footprint(cx, cy): return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * SIGMA ** 2))
k = 460; cx, cy = px[k], py[k]
beam_dist = footprint(cx, cy)
dose = np.zeros_like(X); banks = {n: np.zeros_like(X) for n in TAUS}
for i in range(k + 1):
    f = footprint(px[i], py[i]); dose += f * dt[i]
    for n, tau in TAUS.items():
        decay = np.exp(-dt[i] / tau) if i else 0.0
        banks[n] = banks[n] * decay + f * dt[i]
path_dose = dose / dt[: k + 1].sum()
for n, tau in TAUS.items(): banks[n] /= tau
# Conduction response: spectral stepping of the heat equation along the same
# path, mirrored in z so the free surface passes no heat, normalized to [0, 1]
# per case exactly as npz_datasource does.
NZ = 24
_sim = lr.SpectralConduction((nx, ny, NZ), (x[1] - x[0], y[1] - y[0], 10e-6),
                             alpha=3e-6, sigma0=SIGMA)
for i in range(k + 1):
    _sim.step(dt[i], cx=px[i] - x[0], cy=py[i] - y[0], q=dt[i])
_surf = _sim.field()[:, :, 0]
conduction = (_surf - _surf.min()) / (_surf.max() - _surf.min())

ext = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
# 2026-09-09 results lock: only tau = 1 ms is supplied to any tabled model, so the
# 100 us and 3 ms panels were dropped. banks[] still computes all three; `sheet` uses tau1ms.
FIELDS = [("footprint", beam_dist), ("cumulative dose", path_dose),
          (r"dose, $\tau=1$ ms", banks["tau1ms"]),
          ("conduction response", conduction)]
# Signed offsets in the frame that turns with the beam, as npz_datasource builds
# beam_s / beam_n: along the instantaneous direction of travel and to its left,
# in beam radii. Same construction as conditioning_channels.ipynb cell 2.
_vx, _vy = np.gradient(px)[k] / dt[k], np.gradient(py)[k] / dt[k]
_vhat = np.array([_vx, _vy]) / np.hypot(_vx, _vy)
beam_s = ((X - cx) * _vhat[0] + (Y - cy) * _vhat[1]) / SIGMA
beam_n = (-(X - cx) * _vhat[1] + (Y - cy) * _vhat[0]) / SIGMA

FRAC = 0.32                              # placed at 0.32\\textwidth, three across
W = figstyle.width_in(FRAC)
# One axes box for every panel, in authored inches: room on the left for the
# y tick labels and label, below for the x ones. The box has the field's aspect.
L, R, B, T = 0.98, 0.08, 0.72, 0.08
PW = W - L - R
PH = PW * (ext[3] - ext[2]) / (ext[1] - ext[0])
H = B + PH + T
CBAR_FRAC = 0.14                         # the colour bar files, same height
OFFSET_LIM = float(np.ceil(max(np.abs(beam_s).max(), np.abs(beam_n).max())))


def _panel():
    fig = plt.figure(figsize=(W, H), constrained_layout=False)
    ax = fig.add_axes([L / W, B / H, PW / W, PH / H])
    return fig, ax


def _field(name, fld, cmap, vmin, vmax, path_color):
    fig, ax = _panel()
    ax.imshow(fld.T, origin='lower', extent=ext, aspect='auto', cmap=cmap,
              vmin=vmin, vmax=vmax, interpolation=figstyle.FIELD_INTERPOLATION)
    ax.plot(px[:k + 1] * 1e3, py[:k + 1] * 1e3, color=path_color, lw=1.8)
    ax.plot(cx * 1e3, cy * 1e3, 'o', mfc='none', mec=path_color, ms=10, mew=1.8)
    ax.set_xticks([-0.5, 0.0, 0.5]); ax.set_yticks([-0.5, 0.0, 0.5])
    ax.xaxis.set_major_formatter(figstyle.tick_g()); ax.yaxis.set_major_formatter(figstyle.tick_g())
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    return figstyle.save(fig, name)


def _cbar(name, cmap, vmin, vmax, label, ticks):
    """Vertical bar spanning exactly the panels' axes box, label on its right."""
    w = figstyle.width_in(CBAR_FRAC)
    fig = plt.figure(figsize=(w, H), constrained_layout=False)
    cax = fig.add_axes([0.08 / w, B / H, 0.16 / w, PH / H])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=ticks)
    cb.ax.yaxis.set_major_formatter(figstyle.tick_g())
    cb.set_label(label)
    cb.outline.set_linewidth(figstyle.TICK_WID)
    return figstyle.save(fig, name)


def panels(cmap):
    pc_seq = 'w' if cmap in ('viridis', 'inferno', 'magma', 'plasma', 'cividis') else 'k'
    out = []
    fig, ax = _panel()
    ax.plot(arc * 1e3, speed, lw=2.0, color='C0')
    ax.plot(arc[k] * 1e3, speed[k], 'o', mfc='none', mec='k', ms=10, mew=1.8)
    ax.axvspan(arc[np.searchsorted(frac, 0.33)] * 1e3, arc[np.searchsorted(frac, 0.57)] * 1e3,
               alpha=0.15, color='C1', lw=0)
    ax.set_xlim(arc[0] * 1e3, arc[-1] * 1e3)
    ax.set_xlabel('Arc length (mm)'); ax.set_ylabel('$v$ (m/s)')
    ax.set_yticks([0.2, 0.4])
    out.append(figstyle.save(fig, 'channels_speed'))
    for name, fld in [('channels_footprint', beam_dist), ('channels_dose_inf', path_dose),
                      ('channels_dose_1ms', banks['tau1ms']), ('channels_conduction', conduction)]:
        out.append(_field(name, fld / fld.max(), cmap, 0, 1, pc_seq))
    for name, fld in [('channels_offset_s', beam_s), ('channels_offset_n', beam_n)]:
        out.append(_field(name, fld, figstyle.DIVERGING, -OFFSET_LIM, OFFSET_LIM, 'k'))
    out.append(_cbar('channels_cbar_norm', cmap, 0, 1, 'Normalized', [0, 0.5, 1]))
    out.append(_cbar('channels_cbar_offset', figstyle.DIVERGING, -OFFSET_LIM, OFFSET_LIM,
                     'Beam radii', [-20, -10, 0, 10, 20]))
    plt.close('all')
    return out


def sheet(out):
    cmaps = ['viridis', 'inferno', 'inferno_r', 'magma', 'plasma', 'cividis', 'YlOrRd', 'hot']
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2))
    fld = banks['tau1ms'] / banks['tau1ms'].max()
    for ax, cm in zip(axes.ravel(), cmaps):
        ax.imshow(fld.T, origin='lower', extent=ext, aspect='equal', cmap=cm, vmin=0, vmax=1)
        pc = 'k' if cm in ('inferno_r', 'YlOrRd') else 'w'
        ax.plot(px[:k + 1] * 1e3, py[:k + 1] * 1e3, color=pc, lw=1.5)
        ax.plot(cx * 1e3, cy * 1e3, 'o', mfc='none', mec=pc, ms=8, mew=1.5)
        ax.set_title(cm, fontsize=15); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(r'dose, $\tau=1$ ms, normalized, under eight colour maps', fontsize=16)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cmap', default=figstyle.SEQUENTIAL)
    ap.add_argument('--sheet', default=None); a = ap.parse_args()
    if a.sheet: sheet(a.sheet); print('sheet', a.sheet)
    else: print(*panels(a.cmap), sep='\n')
