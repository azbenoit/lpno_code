"""Cavity depth through the NIST mds2-2525 STATIONARY-SPOT pulse, from the radiographs.

The scanned run of the same record is already handled inline in
``experimental_validation.ipynb``; the spot needs its own reader because its
frames are distributed raw only and its geometry differs.

Scale. 2.01 um/px, transferred from the scanned run. The transfer is measured,
not assumed: the distributed "processed" images are a mirrored crop of the raw
at IDENTICAL sampling (cavity travel -7.26 px/frame in the raw against
+7.94 px/frame in the processed -- ratio 0.91, not the 0.5 a 2x rebin would
give), and 2.01 um/px is the beam-travel calibration of those processed images.
The scanned run's beam-travel derivation cannot be repeated here because
nothing moves, which is why the transfer had to be justified instead.

Timing. t(frame) = -52.48 us + (frame - 80) * 19.987 us, from the CameraTrigger
and FrameNumber columns of the supplied absorption CSV. The laser is on from
t = 0 to t = 2008.4 us.

Polarity. The cavity is missing metal, so it transmits more X-rays: bright in
the flat-fielded ratio. The dark lens that straddles the surface early in the
pulse is the opposite -- displaced melt piled ABOVE the original surface, which
adds material to the beam path.
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
from PIL import Image

UM_PER_PX = 2.01
FRAME0_US = -52.48
FRAME_DT_US = 19.987
PULSE_US = 2008.4
PRE_FRAMES = range(20, 80)      # before laser-on
THRESH = 1.10                   # flat-field ratio; see sensitivity note below
MIN_BLOB_PX = 8


def frame_time_us(n: int) -> float:
    return FRAME0_US + (n - 80) * FRAME_DT_US


def _files(root: pathlib.Path) -> dict[int, str]:
    return {int(f[-8:-4]): f for f in glob.glob(str(root / 'raw' / '*' / '*.tif'))}


def _img(files, n):
    return np.asarray(Image.open(files[n])).astype(float)


def surface_fit(ref):
    """Free-surface row per column, straight-line fitted. The sample is tilted."""
    rows = np.argmin(np.gradient(ref, axis=0), axis=0)
    c = np.arange(ref.shape[1])
    k = (rows > 5) & (rows < ref.shape[0] - 5)
    a, b = np.polyfit(c[k], rows[k], 1)
    return float(a), float(b), float(np.std(rows[k] - (a * c[k] + b)))


def geometry(root):
    """Flat-field reference, fitted surface, and a signed depth-below-surface map."""
    root = pathlib.Path(root)
    files = _files(root)
    ref = np.mean([_img(files, k) for k in PRE_FRAMES if k in files], axis=0)
    a, b, sd = surface_fit(ref)
    H, W = ref.shape
    yy, xx = np.mgrid[0:H, 0:W]
    depth_px = (yy - (a * xx + b)) / np.hypot(1.0, a)   # perpendicular, +ve below
    return files, ref, (a, b, sd), depth_px


def cavity_mask(ratio, depth_px, thresh=THRESH, min_px=MIN_BLOB_PX):
    """Bright blobs that TOUCH the original surface line.

    Connectivity to the surface is the discriminator. The vapour plume above the
    surface, the detector's bright banding and single hot pixels in the bulk are
    all bright too, and a per-column scan cannot tell them from the cavity.
    """
    from scipy import ndimage
    lab, k = ndimage.label((ratio > thresh) & (depth_px > -2))
    keep = np.zeros(lab.shape, bool)
    for i in range(1, k + 1):
        sel = lab == i
        if sel.sum() >= min_px and depth_px[sel].min() <= 6:
            keep |= sel
    return keep


def depth_series(root, thresh=THRESH, frames=range(80, 261), cache=True):
    """[(frame, t_us, depth_um)] for the pulse and its immediate aftermath."""
    root = pathlib.Path(root)
    cache_path = root / f'depth_series_thresh{thresh:.2f}.json'
    if cache and cache_path.exists():
        return [tuple(r) for r in json.loads(cache_path.read_text())['series']]
    files, ref, (a, b, sd), depth_px = geometry(root)
    out = []
    for n in sorted(files):
        if n not in frames:
            continue
        m = cavity_mask(_img(files, n) / np.maximum(ref, 1), depth_px, thresh)
        d = float(depth_px[m].max()) * UM_PER_PX if m.any() else 0.0
        out.append((n, frame_time_us(n), d))
    if cache:
        cache_path.write_text(json.dumps(
            {'tilt_deg': float(np.degrees(np.arctan(a))), 'surf_fit_sd_px': sd,
             'um_per_px': UM_PER_PX, 'thresh': thresh, 'series': out}, indent=1))
    return out


def absorptance(root, bin_us=20.0):
    """Measured GROSS absorbed fraction, binned. Windowed to the plateau.

    RelativeAbsorption in the supplied CSV reaches an unphysical 182 % during the
    power ramp (it divides by a near-zero input), so the fraction is recomputed
    here as binned absorbed power over binned input power inside the pulse only.
    """
    import csv
    root = pathlib.Path(root)
    rows = list(csv.DictReader(open(root / 'Spot_Calibrated_Absorption.csv')))
    t = np.array([float(r['Time']) for r in rows]) * 1e6
    P = np.array([float(r['InputLaser']) for r in rows])
    A = np.array([float(r['AbsoluteAbsorption']) for r in rows])
    on = (t >= 0) & (t <= PULSE_US)
    edges = np.arange(0, PULSE_US + bin_us, bin_us)
    idx = np.digitize(t[on], edges) - 1
    out = []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() == 0:
            continue
        pp, aa = P[on][m].mean(), A[on][m].mean()
        out.append((edges[k] + bin_us / 2, pp, aa, aa / max(pp, 1e-9)))
    return np.array(out)


if __name__ == '__main__':
    ROOT = pathlib.Path(__file__).resolve().parents[2] / 'data/experimental/spot'
    s = np.array([(t, d) for _, t, d in depth_series(ROOT, cache=False)])
    on = s[(s[:, 0] > 0) & (s[:, 0] < PULSE_US)]
    a = absorptance(ROOT)
    # Keyhole onset from each channel, independently.
    d_on = on[on[:, 1] > 20][0, 0] if (on[:, 1] > 20).any() else float('nan')
    lo = a[a[:, 0] < 400, 3].mean()
    jump = a[a[:, 3] > lo + 0.5 * (a[:, 3].max() - lo)]
    print(f'depth      : onset (>20 um) at {d_on:.0f} us, max {on[:, 1].max():.0f} um '
          f'at {on[np.argmax(on[:, 1]), 0]:.0f} us')
    print(f'absorptance: {lo:.3f} before 400 us, half-rise at {jump[0, 0]:.0f} us, '
          f'{a[-1, 3]:.3f} at the end')
