"""Single source of truth for manuscript figure style.

Every figure notebook imports this and calls ``apply()`` before plotting, so
figures cannot drift apart in font, size, line weight or palette. Change a
convention here, rerun the notebooks, and every figure moves together.

Conventions follow npj Computational Materials / Nature Portfolio guidance: one
sans-serif typeface (Liberation Sans, Arial metrics, shipped in reproducibility/figures/fonts) at ONE
size across all figures, RGB, white background, no boxing or decorative effects,
distinct colours of comparable visibility, and red/green never used as a
contrasting pair (colour-vision deficiency).

THE SIZE RULE (2026-10-01). Every figure is authored at width_in(frac) and
placed at frac*\\textwidth, so it prints at REDUCTION = 0.5, and all its text is
FONT = 20 pt authored = 10 pt printed, the size of the body text. No figure sets
a font size of its own; no text is smaller "to fit". If it does not fit, the
figure has too much text, or the panel is placed too small. Verify with
``python3 tools/check_figures.py`` after any figure change.

Usage in a notebook::

    import sys; sys.path.insert(0, "<path to manuscript>/notebooks")
    import figstyle
    figstyle.apply()
    fig, ax = figstyle.figure(width="single")
    ...
    figstyle.save(fig, "melt_pool_comparison")   # respects REPLACE_FIGURES
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Regeneration guard. FALSE BY DEFAULT: running a notebook end to end must not
# silently overwrite the figures currently in the manuscript. Flip it
# deliberately, per session, when you intend to replace them.
#   export LPNO_REPLACE_FIGURES=1     (or set figstyle.REPLACE_FIGURES = True)
# --------------------------------------------------------------------------- #
REPLACE_FIGURES: bool = os.environ.get("LPNO_REPLACE_FIGURES", "0") == "1"

FIG_DIR = Path(os.environ.get("LPNO_FIG_DIR") or Path(__file__).resolve().parent.parent / "Figs")

# Nature Portfolio column widths (mm -> inches). A PANEL is normally produced at
# "single" and placed by LaTeX inside a subfigure; "double" is for the rare figure
# that is genuinely one indivisible image.
WIDTHS_MM = {"single": 89.0, "onehalf": 120.0, "double": 183.0}

#: Text width of the sn-jnl class (measured: \the\textwidth = 372 pt) and the ONE
#: reduction every figure is placed at. A figure placed at frac*\textwidth is
#: authored at frac*TEXTWIDTH/REDUCTION, so LaTeX shrinks every figure in the paper
#: by exactly REDUCTION. That single factor is what makes type, line weight and
#: marker size identical across figures. tools/check_figures.py measures it.
TEXTWIDTH_PT = 372.0
REDUCTION = 0.5

#: Body text of the paper (measured: \f@size = 10.04 pt at the class default).
#: User rule, 2026-10-01: every piece of figure text prints at least this size,
#: so a figure reads without zooming. One size for ALL figure text (axis labels,
#: ticks, legends, colour bars, annotations, panel letters, diagram boxes), as
#: Nature Portfolio asks; hierarchy comes from weight and placement, never from a
#: smaller size.
PRINTED_TEXT_PT = 10.0
FONT = PRINTED_TEXT_PT / REDUCTION      # 20 pt authored -> 10 pt printed

# Kept as names because every generator already reads them; all are FONT now.
BASE_FONT_SIZE = FONT
LABEL_SIZE = FONT
TICK_SIZE = FONT
LEGEND_SIZE = FONT
TICK_LEN = 6        # authored; prints at 3 pt
TICK_WID = 1.2      # authored; prints at 0.6 pt

# Panel default: generous absolute size so the baseline point sizes above are
# correct once LaTeX scales the panel down into a subfigure.
PANEL_SIZE_IN = (5.2, 4.0)

# Field-plot conventions carried over from the same notebook.
MELT_CONTOUR_K = 1898.0        # Ti-6Al-4V solidus/liquidus midpoint
MELT_CONTOUR_COLOR = "orange"
MELT_CONTOUR_LW = 1.5
T_CLIM = (300.0, 3150.0)       # temperature colour limits
FIELD_INTERPOLATION = "nearest"  # never bilinear: it smears the keyhole edge

# Schematic diagrams follow the same rule as every other figure: authored at
# width(frac) and printed at FONT * REDUCTION. The three names survive because
# the diagram code reads them; there is one size, and sub-labels are told apart
# by italics and grey, never by being smaller.
DIAGRAM_BODY = FONT
DIAGRAM_SMALL = FONT
DIAGRAM_TINY = FONT

# 600 dpi is used for anything with filled colour (heatmaps, field slices,
# imshow panels); pure line art is vector and dpi-independent, but the value is
# set globally so a rasterised inset never lands below it.
DPI = 600

# Colour-vision-safe qualitative palette (Okabe-Ito). Deliberately contains no
# red/green contrasting pair. Sequential fields should use viridis/inferno;
# diverging (difference panels) should use a symmetric map about zero.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

SEQUENTIAL = "viridis"
FIELD_TEMPERATURE = "inferno"   # matches the demo/notebook convention for T
DIVERGING = "coolwarm"          # difference panels, always symmetric about 0


#: Liberation Sans: metric-identical to Arial (the Nature Portfolio figure face) and
#: ~12 % narrower than matplotlib's DejaVu default, which is what lets 10 pt text
#: fit a third-of-a-page panel. SIL OFL, so the TTFs ship in reproducibility/figures/fonts and
#: every machine renders identical glyphs. Arial itself cannot be redistributed.
FONT_FAMILY = "Liberation Sans"
FONT_DIR = Path(__file__).resolve().parent / "fonts"


def _register_font() -> None:
    from matplotlib import font_manager
    for ttf in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(ttf))


def apply() -> None:
    """Install the manuscript rcParams. Call once per notebook, before plotting.

    Sizes follow the "FNO Coarse Inference Plot" convention in
    the LP-FNO figures, which is the agreed
    manuscript baseline. They are large in absolute points because panels
    are produced one per file at a generous figure size and placed by LaTeX at
    column or half-column width; the reduction is what brings them to final size.
    """
    _register_font()
    mpl.rcParams.update({
        # Typeface: ONE family, pinned and shipped with the repo (see _register_font).
        # Listing "Arial, Helvetica, DejaVu Sans" as before made a figure's typeface
        # depend on which machine regenerated it.
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.sf": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "legend.title_fontsize": FONT,
        # Baseline sizes (FNO Coarse Inference Plot).
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.major.size": TICK_LEN,
        "xtick.major.width": TICK_WID,
        "ytick.major.size": TICK_LEN,
        "ytick.major.width": TICK_WID,
        "axes.titlesize": BASE_FONT_SIZE,
        # No titles on panels: LaTeX supplies subcaptions (see save()).
        "figure.titlesize": BASE_FONT_SIZE,
        # White background, minimal chrome.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "axes.linewidth": TICK_WID,
        "lines.linewidth": 1.8,
        "lines.markersize": 6.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Vector PDF, fonts embedded, tight box.
        "savefig.format": "pdf",
        "savefig.dpi": DPI,
        "figure.dpi": 150,
        # NOT "tight". A tight bbox trims each PDF to its own ink, so two panels
        # authored at the same figsize save at different page widths; LaTeX then
        # scales each by a different factor and their rendered size, type size and
        # line weight all diverge. Measured 2026-08-15: reductions across the
        # paper ran 0.27-0.74, printed tick type 4.4-11.8 pt. Constrained layout
        # fits the labels inside the authored canvas instead of cropping to them.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.h_pad": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "image.cmap": SEQUENTIAL,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def panel(size=None, frac=0.49, aspect=0.77, **kwargs):
    """Create a SINGLE panel. This is the default for manuscript figures.

    Multi-panel figures are assembled in LaTeX from one PDF per panel, so that
    each panel carries its own \\subcaption and (a)/(b)/(c) label. Panels
    therefore carry NO title; the subcaption supplies it.

    `frac` is the \\textwidth fraction the panel will be PLACED at, and it sets
    the authored width so every figure reduces by REDUCTION. Two panels of one
    figure must share `frac` -- that is what makes them identical twins on the
    page. `aspect` may differ between them; only width drives the reduction.
    Passing `size` explicitly overrides both and breaks that guarantee, so it is
    for one-off diagrams only.
    """
    if size is None:
        w_in = frac * TEXTWIDTH_PT / REDUCTION / 72.0
        size = (w_in, w_in * aspect)
    return plt.subplots(figsize=size, **kwargs)


def width_in(frac: float = 1.0) -> float:
    """Authored width, in inches, of a figure that LaTeX places at frac*\\textwidth.

    Every figure file is built at this width and included
    with ``width=<frac>\\textwidth`` (or ``\\linewidth`` inside a subfigure of that
    fraction), so it prints at exactly REDUCTION and its text at PRINTED_TEXT_PT.
    """
    return frac * TEXTWIDTH_PT / REDUCTION / 72.0


def tick_g():
    """Tick formatter: shortest number ('0.5', '20') with a true minus sign.

    Matplotlib's default formatter already prints U+2212; a format string such
    as '{x:g}' prints a hyphen, which looks different beside every other figure.
    """
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212"))


def colorbar_strip(name: str, vmin: float, vmax: float, frac: float = 0.5,
                   cmap: str = FIELD_TEMPERATURE, label: str = "Temperature (K)",
                   ticks=None, force: bool = False):
    """A stand-alone horizontal colour bar placed at frac*\\textwidth.

    The 3-D render figures carry their colour scale as a separate strip under the
    panels. Every strip comes from here so they share bar thickness, tick length,
    label wording and type size; `vmin`/`vmax` must be the clim the renders used.
    The label sits to the LEFT of the bar, so the strip is one tick-label line
    tall (about 20 pt printed).
    """
    w = width_in(frac)
    h, bar_h, bar_y = 0.66, 0.20, 0.40          # authored inches
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    txt = fig.text(0.0, (bar_y + bar_h / 2) / h, label, ha="left", va="center")
    fig.canvas.draw()
    label_w = txt.get_window_extent().width / fig.dpi
    x0 = label_w + 0.18
    cax = fig.add_axes([x0 / w, bar_y / h, (w - x0 - 0.30) / w, bar_h / h])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                      ticks=ticks if ticks is not None else
                      [t for t in (0, 1000, 2000, 3000, 4000) if vmin <= t <= vmax])
    cb.ax.tick_params(pad=2)
    cb.outline.set_linewidth(TICK_WID)
    return save(fig, name, force=force)


SCALE_BAR_LW = 4.0   # authored; prints at 2 pt


def scale_bar_strip(name: str, length_um: float, pt_per_um: float, frac: float = 0.2,
                    force: bool = False):
    """A stand-alone length scale for the 3-D renders, placed at frac*\\textwidth
    on the same line as their colour-bar strip.

    `pt_per_um` is the printed length of one micrometre in the render panels, from
    the camera projection in project_render_overlays.py. The strip has the authored
    height of `colorbar_strip` and draws its bar on the colour bar's centre line, so
    the two line up when placed side by side; the label stands right of the bar.
    """
    w = width_in(frac)
    h, mid, x0 = 0.66, 0.50, 0.10          # authored inches, as colorbar_strip
    bar_in = length_um * pt_per_um / (frac * TEXTWIDTH_PT) * w
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    fig.add_artist(mpl.lines.Line2D([x0 / w, (x0 + bar_in) / w], [mid / h] * 2, color="black",
                                    lw=SCALE_BAR_LW, solid_capstyle="butt"))
    fig.text((x0 + bar_in + 0.12) / w, mid / h, f"{length_um:g} µm", ha="left", va="center")
    return save(fig, name, force=force)


TRACK_COLOR = "white"
TRACK_LW = 1.2              # authored; prints at 0.6 pt
TRACK_DASH = (0, (4, 3))    # in units of the line width
TRACK_ALPHA = 0.5           # the path is faint so the field shows through; the chevron is opaque
TRACK_GAP = 1.5             # the line stops this many beam radii before the beam ...
TRACK_AHEAD = 1.0           # ... and an open chevron sits this many radii ahead of it,
TRACK_HEAD = 8              # so the keyhole under the beam stays clear (2026-10-28)
TRACK_HEAD_LW = 1.0         # chevron stroke, authored
TRACK_DOT = 3.5             # end-of-path dot diameter, authored points; prints at 1.75 pt


def _cut_polyline(pts, length):
    """The first `length` (arc length, same units as `pts`) of a polyline."""
    import numpy as np
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if length >= s[-1]:
        return pts
    k = int(np.searchsorted(s, length, side="right"))
    end = pts[k - 1] + (pts[k] - pts[k - 1]) * ((length - s[k - 1]) / seg[k - 1])
    return np.vstack([pts[:k], end])


def draw_track(ax, done, head=None, radius=None):
    """The laser path scanned so far on a render panel, in the panel's data coordinates.

    `done` is the polyline from the start of the path to the beam. `head` is a
    (tail, tip) pair along the direction of travel with the tip at the beam, None
    once the beam has finished its path. `radius` is the beam radius in data units.
    With a head, the dashed line stops TRACK_GAP radii short of the beam, on the end
    of a whole dash, and an open chevron points the way TRACK_AHEAD radii in front of
    it. A finished path ends in a small dot, with the line stopping just before it.
    Call after the axes limits are set: the dash snapping needs the scale.
    """
    import numpy as np
    done = np.asarray(done, float)
    pos = ax.get_position()
    pt_per_unit = pos.width * ax.figure.get_figwidth() * 72.0 / abs(np.diff(ax.get_xlim())[0])
    if head is not None:
        tail, tip = np.asarray(head, float)
        u = (tip - tail) / np.linalg.norm(tip - tail)
        at, gap = tip + TRACK_AHEAD * radius * u, TRACK_GAP * radius
    else:                                         # finished: a dot on the end of the path
        at, gap = done[-1], 0.3 * radius
    length = float(np.hypot(*np.diff(done, axis=0).T).sum()) - gap
    on, period = TRACK_DASH[1][0] * TRACK_LW, sum(TRACK_DASH[1]) * TRACK_LW      # points
    n = np.floor((length * pt_per_unit - on) / period)
    line = _cut_polyline(done, (n * period + on) / pt_per_unit) if n >= 0 else done[:0]
    if len(line) > 1:
        ax.plot(line[:, 0], line[:, 1], color=TRACK_COLOR, lw=TRACK_LW, ls=TRACK_DASH,
                alpha=TRACK_ALPHA, dash_capstyle="butt", solid_capstyle="butt")
    if head is None:
        ax.plot([at[0]], [at[1]], "o", color=TRACK_COLOR, ms=TRACK_DOT, mew=0)
        return
    ax.annotate("", xy=tuple(at), xytext=tuple(at - 0.05 * radius * u),
                arrowprops=dict(arrowstyle="->", color=TRACK_COLOR, lw=TRACK_HEAD_LW,
                                mutation_scale=TRACK_HEAD, shrinkA=0, shrinkB=0))


def track_key_strip(name: str, frac: float = 0.22, finished: bool = False, force: bool = False):
    """Key for `draw_track`, placed at frac*\\textwidth on the colour-bar line.

    The white track needs a dark ground to be seen, so each sample sits on a swatch
    of the low end of the temperature colour map. The strip stands on the same
    baseline as `colorbar_strip` and centres its rows on the bar's centre line, 0.50 in
    above that baseline: one "Laser path" row on the line, or with `finished` that row
    and a "Path finished" row for the end-of-path dot, one on either side of it.
    """
    w = width_in(frac)
    x0, sw, sh, step, mid = 0.06, 0.62, 0.30, 0.42, 0.50      # authored inches
    if finished:
        rows = [(mid - step / 2, "Laser path", False), (mid + step / 2, "Path finished", True)]
    else:
        rows = [(mid, "Laser path", False)]
    h = max(0.66, rows[-1][0] + sh / 2 + 0.06)
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    for mid, label, done in rows:
        ax = fig.add_axes([x0 / w, (mid - sh / 2) / h, sw / w, sh / h])
        ax.set_facecolor(plt.get_cmap(FIELD_TEMPERATURE)(0.08))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        if done:
            draw_track(ax, [(0.06, 0.5), (0.80, 0.5)], radius=0.1)
        else:
            draw_track(ax, [(0.06, 0.5), (0.72, 0.5)], head=[(0.62, 0.5), (0.72, 0.5)], radius=0.1)
        fig.text((x0 + sw + 0.12) / w, mid / h, label, ha="left", va="center")
    return save(fig, name, force=force)


def colorbar_column(name: str, vmin: float, vmax: float, height_in: float,
                    bottom_in: float, bar_h_in: float, frac: float = 0.14,
                    cmap: str = FIELD_TEMPERATURE, label: str = "Temperature (K)",
                    ticks=None, force: bool = False):
    """A stand-alone VERTICAL colour bar, to stand beside a panel of the same height.

    `height_in`, `bottom_in` and `bar_h_in` are the neighbouring panel's figure
    height, axes bottom margin and axes height (authored inches), so the bar spans
    exactly that panel's plot box once both are placed at the same scale. Ticks on
    the right, label rotated beside them, all at the one font size.
    """
    w = width_in(frac)
    fig = plt.figure(figsize=(w, height_in), constrained_layout=False)
    cax = fig.add_axes([0.06 / w, bottom_in / height_in, 0.20 / w, bar_h_in / height_in])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical",
                      ticks=ticks if ticks is not None else
                      [t for t in (0, 1000, 2000, 3000, 4000) if vmin <= t <= vmax])
    cb.set_label(label, labelpad=6)
    cb.outline.set_linewidth(TICK_WID)
    return save(fig, name, force=force)


def figure(width: str = "single", height: float | None = None, **kwargs):
    """Create a figure at a Nature column width.

    Use only for a figure that is genuinely one indivisible image. Anything with
    parts should be built as separate panels with ``panel()`` and combined in
    LaTeX.
    """
    if width not in WIDTHS_MM:
        raise ValueError(f"width must be one of {sorted(WIDTHS_MM)}, got {width!r}")
    w_in = WIDTHS_MM[width] / 25.4
    h_in = height if height is not None else w_in * 0.75
    return plt.subplots(figsize=(w_in, h_in), **kwargs)


def save(fig, name: str, *, force: bool = False) -> Path | None:
    """Write ``FIG_DIR/<name>.pdf``, honouring the regeneration guard.

    Returns the written path, or None when the guard blocked the write. Pass
    ``force=True`` only for a one-off deliberate overwrite; prefer flipping
    REPLACE_FIGURES for a whole regeneration pass so figures stay in step.
    """
    target = FIG_DIR / f"{name}.pdf"
    if target.exists() and not (REPLACE_FIGURES or force):
        print(f"[figstyle] kept existing {target.name} "
              f"(set LPNO_REPLACE_FIGURES=1 to replace)")
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=DPI)
    _subset_fonts(target)
    print(f"[figstyle] wrote {target}")
    return target


def _subset_fonts(target: Path) -> None:
    """Shrink a written PDF by subsetting its embedded fonts, if Ghostscript is here.

    Matplotlib only subsets embedded TrueType from 3.4 on. On an older matplotlib
    every figure carries the WHOLE typeface, which took the three provenance
    diagrams from 21 kB each to 712 kB and would have grown the manuscript by
    megabytes of duplicated font. Ghostscript subsets them back, measured at a
    ten-fold saving with a maximum pixel difference of 4/255 (anti-aliasing) on a
    110 dpi rasterisation of the same page.

    Silent no-op when Ghostscript is absent or unhappy: the unsubsetted figure is
    correct, just large, so this must never be able to lose one.
    """
    gs = shutil.which("gs")
    if gs is None:
        return
    tmp = target.with_suffix(".subset.pdf")
    try:
        r = subprocess.run(
            [gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
             "-dSubsetFonts=true", "-dEmbedAllFonts=true",
             "-dCompatibilityLevel=1.5", "-dAutoRotatePages=/None",
             f"-sOutputFile={tmp}", str(target)],
            capture_output=True, timeout=120)
        if r.returncode == 0 and tmp.exists() and 0 < tmp.stat().st_size < target.stat().st_size:
            tmp.replace(target)
        else:
            tmp.unlink(missing_ok=True)
    except (OSError, subprocess.SubprocessError):
        tmp.unlink(missing_ok=True)
