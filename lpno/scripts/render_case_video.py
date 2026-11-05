#!/usr/bin/env python3
"""Render a case as a GT-vs-prediction melt-surface video.

Takes an eval's ``eval_frames_dump.npz`` (written when ``PINO_DUMP_FRAMES=1``)
and renders every frame through the demo's PyVista melt-surface renderer, then
stitches the frames with ffmpeg. Ground truth and prediction share one camera
and one colour limit, so the two halves are comparable by eye -- a per-frame or
per-panel clim would make a cooling model look identical to an accurate one.

The renderer is `lpno_web_demo_backend/app/render.py`, deliberately: it is the
code the public demo uses, so a video made here looks like what a reader sees.

Usage:
    python scripts/render_case_video.py DUMP.npz OUTDIR --case-frames 386 367 \
        --case-names t64v_119 t64v_103 [--fps 20] [--mesh-um 10]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

# The demo renderer lives in the website server, which is not importable as a
# package; add its root so `app.render` resolves.
_WEBSITE = Path(__file__).resolve().parents[2] / "lpno_web_demo_backend"

# Bound by _load_renderer(); the demo owns this exception type.
_RenderError: type[Exception] = RuntimeError


def _load_renderer():
    """The demo's iso-surface renderer plus its colourbar composer.

    NOT the public `render_3d_surface_png`: that one derives its own colour limit
    per call, so a GT panel and a prediction panel would be scaled independently
    and a model under-predicting peak temperature by half would look correct.
    The inner `_render_iso_surface_img` accepts an explicit clim, which is the
    whole point of rendering the pair together.
    """
    if not _WEBSITE.is_dir():
        raise SystemExit(f"demo renderer not found at {_WEBSITE}")
    sys.path.insert(0, str(_WEBSITE))
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    global _RenderError
    from app.render import (  # noqa: PLC0415
        RenderError,
        _compose_3d_figure_with_colorbar,
        _render_iso_surface_img,
    )

    _RenderError = RenderError
    return _render_iso_surface_img, _compose_3d_figure_with_colorbar


def shared_clim(true_t: np.ndarray, pred_t: np.ndarray, true_a: np.ndarray, pred_a: np.ndarray) -> list[float]:
    """One colour range for both panels and every frame.

    Taken over metal cells only (alpha >= 0.5) and over the WHOLE series, so a
    frame's colour means the same thing at frame 0 and frame 386.
    """
    hi = 0.0
    for t, a in ((true_t, true_a), (pred_t, pred_a)):
        metal = a >= 0.5
        if np.any(metal):
            hi = max(hi, float(np.max(t[metal])))
    return [0.0, hi if hi > 0 else 3200.0]


def _panel(render, compose, t_vol, a_vol, *, mesh_um, width, height, plane, clim):
    """One panel at the CALLER's colour limit, so both panels share a scale."""
    img, used = render(
        t_vol, a_vol, t_vol,
        color_field="T", mesh_size_um=mesh_um, width_px=width, height_px=height,
        smoothing_iterations=12, plane=plane, clim=list(clim),
    )
    return compose(img=img, color_field="T", clim=used, width_px=width, height_px=height)


def _raw_panel(render, t_vol, a_vol, *, mesh_um, width, height, plane, clim):
    """The screenshot only -- for callers that compose the panels themselves."""
    img, _ = render(
        t_vol, a_vol, t_vol,
        color_field="T", mesh_size_um=mesh_um, width_px=width, height_px=height,
        smoothing_iterations=12, plane=plane, clim=list(clim),
    )
    return img


def detect_case_boundaries(true_t: np.ndarray, *, sigma: float = 6.0) -> list[int]:
    """Frame indices where a new case starts, found in the DATA.

    The dump concatenates cases in the loader's order, which is **alphabetical by
    case file name** (`_case_files` returns `sorted(p.glob("*.npz"))`) -- NOT the
    order the cases were passed to `collection create`. Trusting the caller's
    order split a 755-frame dump at 387 when the true segments were 368/387, so
    one video held a whole case plus 19 frames of the next and the other held the
    remainder. Both looked completely plausible.

    A case boundary is a discontinuity in frame-to-frame mean temperature: within
    a case the field evolves smoothly, between cases it jumps. Verified against
    the independent alphabetical prediction on a 4-case dump -- detected
    [191, 90, 125, 168], predicted [191, 90, 125, 168].
    """
    mean_t = np.asarray(true_t).reshape(len(true_t), -1).mean(axis=1)
    step = np.abs(np.diff(mean_t))
    if step.size == 0:
        return []
    thresh = step.mean() + sigma * step.std()
    return [int(i) + 1 for i in np.where(step > thresh)[0]]


def check_segments(true_t: np.ndarray, expected: list[int]) -> None:
    """Refuse a WRONG split, but not a boundary the detector merely cannot see.

    Two different failures, and only one of them is the caller's fault:

    * the data shows a boundary the caller does NOT have -- that is the mis-split this
      guard exists for (a 755-frame dump cut at 387 when the segments were 368/387, so
      one video held a whole case plus 19 frames of the next and both looked plausible);
    * the caller has a boundary the data does not confirm -- `detect_case_boundaries`
      thresholds a mean-temperature STEP, so two neighbouring cases at a similar level
      produce no step. Measured 2026-09-01: mb80 counts 95/95/34/90/95/95 summed to the
      dump exactly and agreed at every other junction, and were refused over the one
      faint boundary at 224.

    So the test is containment, not equality, plus the total -- which nothing checked
    before, and which is the cheap arithmetic that actually catches a wrong split.
    """
    total = len(true_t)
    if sum(expected) != total:
        raise SystemExit(
            f"[render] the frame counts sum to {sum(expected)} but the dump holds {total} "
            f"frames. A strided eval halves each case; pass the DUMPED counts, not the "
            f"source ones."
        )
    found = detect_case_boundaries(true_t)
    starts, acc = [], 0
    for n in expected[:-1]:
        acc += n
        starts.append(acc)
    unexplained = [b for b in found if b not in starts]
    if unexplained:
        raise SystemExit(
            "[render] the frame counts do not match the case boundaries in the dump.\n"
            f"  caller implies boundaries at {starts}\n"
            f"  the data has them at        {found}\n"
            f"  the data separates cases at {unexplained}, which the counts do not.\n"
            "Loader order is ALPHABETICAL by case file name, not the order passed to "
            "`collection create`. Re-order --case-names/--case-frames to match, or drop "
            "the counts and use the detected boundaries."
        )
    unconfirmed = [b for b in starts if b not in found]
    if unconfirmed:
        print(f"[render] boundaries {unconfirmed} are not visible as a temperature step "
              f"(neighbouring cases at a similar level); counts sum to {total}, and every "
              f"boundary the data DOES show is accounted for, so proceeding.")


def beam_speed(manifest: dict, t_seconds: np.ndarray) -> np.ndarray:
    """Per-frame beam speed (m/s) from the SAME path builder the channels use.

    Not from the design table: that records the sampled Vmean, while the solved
    case follows the built polyline. (They agree to ~0.2% here, which is the
    check that this is reading the right path.)
    """
    from lpno.custom_utils.npz_datasource import (  # noqa: PLC0415
        _beam_centres_over_time,
    )

    c = np.asarray(_beam_centres_over_time(manifest, np.asarray(t_seconds, dtype=float)))
    xy = c[:, 0, :2] if c.ndim == 3 else c[:, :2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(np.asarray(t_seconds, dtype=float))
    v = np.divide(step, dt, out=np.full_like(step, np.nan), where=dt > 0)
    # One value per FRAME: repeat the first interval so len(v) == len(t).
    return np.concatenate([v[:1], v])


def beam_track_drawer(beams, t_s, *, extent_m=None):
    """Side panel for a MULTI-BEAM case: plan view of the tracks.

    A multi-beam case has no single scan speed, so the varspeed curve has nothing
    to plot. What distinguishes these cases is the geometry -- how many beams,
    where they run, how their tracks interleave -- so the analogue is an x-y plan
    view with one colour per beam and a filled marker at each beam's position in
    the frame being shown.

    `beams` is the merged per-beam spec from the STAGED manifest
    (`derived.beams[i].path_xy_m` / `path_times_s`). The stored simulation
    metadata carries those lists EMPTY -- aiida-pino merges them from the prepin
    velocity tables at staging -- so this cannot be built from the node alone.
    """
    colours = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    tracks = []
    for b in beams:
        xy = np.asarray(b.get("path_xy_m") or [], dtype=float)
        tt = np.asarray(b.get("path_times_s") or [], dtype=float)
        tracks.append((xy, tt, float(b.get("power_w") or 0.0)))

    def draw(ax, frame):
        now = float(np.asarray(t_s)[frame])
        for i, (xy, tt, pw) in enumerate(tracks):
            c = colours[i % len(colours)]
            if xy.size == 0:
                continue
            ax.plot(xy[:, 0] * 1e3, xy[:, 1] * 1e3, color=c, lw=1.0, alpha=0.55)
            # Interpolate this beam's position at the displayed instant.
            if tt.size == xy.shape[0] and tt.size > 1:
                x = np.interp(now, tt, xy[:, 0]) * 1e3
                y = np.interp(now, tt, xy[:, 1]) * 1e3
                ax.plot([x], [y], "o", color=c, ms=6, mec="white", mew=0.7)
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9, labelpad=2)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{len(tracks)} beams", fontsize=10, pad=6)

    return draw


def side_panel_for(manifest, t_s):
    """The side panel a case deserves: track plan view for multi-beam, else None (speed curve).

    `beam_track_drawer` existed for multi-beam from the start and was never reachable from
    `main()` -- `side_draw` defaulted to None on every CLI path -- so multibeam videos drew
    the beam-speed curve instead. For several beams that curve is the POWER-WEIGHTED
    CENTROID of whichever beams are still on, so it jumps discontinuously each time one
    switches off: mbc40_031 renders spikes to 25 m/s on a case whose beams run near
    0.45 m/s (2026-09-04). The spikes are the centroid definition, not beam motion.

    Returning None for a single beam is deliberate: there the centroid IS the beam and the
    trace is the physical scan speed the varspeed figures are built on. Returning None when
    the per-beam paths are missing is also deliberate -- a NODE manifest carries them empty
    (aiida-pino merges them at staging), and an empty plan view is worse than the curve.
    """
    derived = (manifest or {}).get("derived") or {}
    beams = derived.get("beams") or []
    n = int(derived.get("beam_count") or len(beams) or 1)
    if n < 2 or not beams:
        return None
    if not any((b or {}).get("path_xy_m") for b in beams):
        return None
    return beam_track_drawer(beams, t_s)


def _compose_stacked(pred_img, gt_img, *, clim, speed, t_s, frame, width_px, title, side_draw=None):
    """Ground truth on top, prediction below, speed curve down the right side.

    Panel order follows the demo's own `_compose_3d_gt_pred_figure`. The speed
    panel has no demo equivalent -- it exists because these are VARSPEED cases,
    where the beam slows into turns, and the model's error tracks that.

    Both panels share one colourbar: the point of the layout is that a colour
    means the same temperature in both, so a missing keyhole peak reads as a
    deficit rather than as a rescale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dpi = 100
    img_w = 0.60          # image column
    panel_ar = pred_img.shape[0] / max(pred_img.shape[1], 1)
    fig_w_in = width_px / dpi
    ax_w_in = img_w * fig_w_in
    ax_h_in = ax_w_in * panel_ar
    sup_in, title_in, bot_in = 0.42, 0.30, 0.10
    fig_h_in = sup_in + 2 * (title_in + ax_h_in) + bot_in

    fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=12, y=1.0 - 0.45 * sup_in / fig_h_in)

    y_top = fig_h_in - sup_in
    y0s = []
    for label, img in (("Ground truth  T [K]", gt_img), ("Prediction  T [K]", pred_img)):
        y0 = y_top - title_in - ax_h_in
        ax = fig.add_axes([0.02, y0 / fig_h_in, img_w, ax_h_in / fig_h_in])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=11, pad=5)
        y0s.append(y0)
        y_top = y0

    # Shared colourbar spanning both panels.
    cb_lo = y0s[1] + 0.10 * ax_h_in
    cb_hi = y0s[0] + 0.90 * ax_h_in
    cax = fig.add_axes([0.615, cb_lo / fig_h_in, 0.020, (cb_hi - cb_lo) / fig_h_in])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=float(clim[0]), vmax=float(clim[1])),
                               cmap=plt.get_cmap("inferno"))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=8)

    # Side panel, full height on the right. Varspeed draws a speed curve; a
    # multi-beam case has no single speed, so it draws the beam tracks instead.
    sax = fig.add_axes([0.795, (cb_lo / fig_h_in), 0.175, (cb_hi - cb_lo) / fig_h_in])
    if side_draw is not None:
        side_draw(sax, frame)
    else:
        ms = np.asarray(t_s) * 1e3
        sax.plot(ms, speed, color="#3b6ea5", lw=1.2)
        sax.axvline(ms[frame], color="#d1495b", lw=1.0)
        sax.plot([ms[frame]], [speed[frame]], "o", color="#d1495b", ms=5)
        sax.set_xlabel("t [ms]", fontsize=9)
        sax.set_ylabel("beam speed [m/s]", fontsize=9, labelpad=2)
        sax.tick_params(labelsize=8)
        sax.grid(alpha=0.25, lw=0.5)
        sax.set_title(f"V = {speed[frame]:.3f} m/s", fontsize=10, pad=6)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pil_kwargs={"compress_level": 3})
    plt.close(fig)
    return buf.getvalue()


def render_case(dump: Path, out_dir: Path, *, start: int, n: int, name: str,
                mesh_um: float, fps: int, width: int, height: int, plane: str,
                t_scale: float, stacked=None, side_draw=None) -> Path:
    render, compose = _load_renderer()
    with np.load(dump) as z:
        sl = slice(start, start + n)
        pt, pa = z["pred_T"][sl], z["pred_alpha"][sl]
        tt, ta = z["true_T"][sl], z["true_alpha"][sl]

    # The dump is in normalised units (T/ref_scales.T). The renderer works in
    # Kelvin -- its melt contour and fill inversion are absolute -- so scale here.
    # Read the factor off the model's config; NEVER derive it from the data
    # (three derivations of this number were wrong on 2026-08-22).
    pt = pt * t_scale
    tt = tt * t_scale

    # (frames, 1, X, Y, Z) -> drop the channel axis
    squeeze = lambda a: a[:, 0] if a.ndim == 5 else a  # noqa: E731
    pt, pa, tt, ta = map(squeeze, (pt, pa, tt, ta))
    clim = shared_clim(tt, pt, ta, pa)
    print(f"[{name}] {len(pt)} frames, shared clim {clim[0]:.0f}-{clim[1]:.0f} K", flush=True)

    frames_dir = out_dir / f"{name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    kept: list[int] = []

    if stacked is not None:
        speed, t_s, title_stem = stacked
        if len(speed) != len(pt):
            raise SystemExit(
                f"[{name}] speed has {len(speed)} samples for {len(pt)} frames -- "
                "a mismatch here would label every frame with the wrong speed"
            )
        for i in range(len(pt)):
            try:
                gi = _raw_panel(render, tt[i], ta[i], mesh_um=mesh_um, width=width,
                                height=height, plane=plane, clim=clim)
                pi = _raw_panel(render, pt[i], pa[i], mesh_um=mesh_um, width=width,
                                height=height, plane=plane, clim=clim)
            except _RenderError:
                print(f"  frame {i}: no melt surface in one panel -- pair skipped", flush=True)
                continue
            png = _compose_stacked(pi, gi, clim=clim, speed=speed, t_s=t_s, frame=i,
                                   width_px=width, title=f"{title_stem}   t = {t_s[i]*1e3:.3f} ms",
                                   side_draw=side_draw)
            (frames_dir / f"cmp_{i:04d}.png").write_bytes(png)
            kept.append(i)
            if i % 50 == 0:
                print(f"  ...{i}/{len(pt)}", flush=True)
        if not kept:
            raise SystemExit(f"[{name}] no frame produced a melt surface in BOTH panels")
        print(f"[{name}] kept {len(kept)}/{len(pt)} frame pairs", flush=True)
        return _encode(frames_dir, out_dir, name, "cmp", kept, fps)

    for i in range(len(pt)):
        # Render the PAIR before writing either. Skipping the two panels
        # independently lets them drift: if GT has a melt surface at frame 5 and
        # the prediction does not, the two videos silently fall out of step and
        # every later frame compares the wrong pair.
        panels = {}
        for tag, (t_vol, a_vol) in (("gt", (tt[i], ta[i])), ("pred", (pt[i], pa[i]))):
            try:
                panels[tag] = _panel(render, compose, t_vol, a_vol, mesh_um=mesh_um,
                                     width=width, height=height, plane=plane, clim=clim)
            except _RenderError:
                # No alpha>=0.5 surface yet -- the beam has not melted anything.
                # Deliberately NOT `except Exception`: a blanket catch here turned a
                # TypeError in this very function into 1510 tidy "skipped" lines and
                # an empty video, which read as "the model never melts".
                break
        if len(panels) != 2:
            print(f"  frame {i}: no melt surface in one panel -- pair skipped", flush=True)
            continue
        for tag, png in panels.items():
            (frames_dir / f"{tag}_{i:04d}.png").write_bytes(png)
        kept.append(i)
        if i % 50 == 0:
            print(f"  ...{i}/{len(pt)}", flush=True)

    if not kept:
        raise SystemExit(f"[{name}] no frame produced a melt surface in BOTH panels")
    print(f"[{name}] kept {len(kept)}/{len(pt)} frame pairs", flush=True)

    outputs = [_encode(frames_dir, out_dir, name, tag, kept, fps) for tag in ("gt", "pred")]
    return outputs[0]


def _encode(frames_dir: Path, out_dir: Path, name: str, tag: str, kept: list[int], fps: int) -> Path:
    """Stitch the kept frames of one tag into an mp4.

    An explicit concat list rather than a numeric %04d pattern, which would stop
    at the first skipped frame.

    NOTE ON OUT_DIR: ffmpeg here is the snap build, which is confined and cannot
    read HIDDEN directories -- frames staged under a dot-directory fail with
    "Permission denied" on this list (and "Could not open file" on a glob), while
    the identical run under the repo's runs/ works. One more reason run outputs
    belong in the repo, not in a worktree or a job dir.
    """
    listing = frames_dir / f"{tag}_files.txt"
    listing.write_text("".join(
        f"file '{frames_dir / f'{tag}_{i:04d}.png'}'\nduration {1.0 / fps:.6f}\n" for i in kept
    ))
    mp4 = out_dir / f"{name}_{tag}.mp4"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(mp4)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        # Surface ffmpeg's own words; swallowing them cost a debugging round.
        raise SystemExit(f"ffmpeg failed for {name}/{tag}:\n{done.stderr[-1500:]}")
    print(f"[{name}] wrote {mp4.name} ({mp4.stat().st_size / 1e6:.1f} MB)", flush=True)
    return mp4


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--case-frames", type=int, nargs="+", required=True,
                    help="frame count per case, in loader order")
    ap.add_argument("--case-names", nargs="+", required=True)
    ap.add_argument("--mesh-um", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--plane", default="top", choices=("top", "xz"))
    ap.add_argument("--t-scale", type=float, default=3000.0,
                    help="ref_scales.T from the MODEL config; the dump is normalised")
    ap.add_argument("--stacked", action="store_true",
                    help="one video per case: GT over prediction, shared colourbar, speed curve")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="the eval's PINO_EVAL_FRAME_STRIDE; times are subsampled to match")
    ap.add_argument("--manifest-dir", type=pathlib.Path, default=None,
                    help="staged <case>.manifest.json dir; REQUIRED for multibeam, whose "
                         "per-beam paths are empty in the node and merged only at staging")
    ap.add_argument("--sim-pk", type=int, nargs="*", default=[],
                    help="AiiDA simulation pk per case, for the manifest the speed curve needs")
    a = ap.parse_args(argv)

    if len(a.case_frames) != len(a.case_names):
        ap.error("--case-frames and --case-names must have the same length")

    if a.stacked and len(a.sim_pk) != len(a.case_names):
        ap.error("--stacked needs one --sim-pk per case (the speed curve reads its manifest)")

    # Verify the caller's segmentation against the data BEFORE rendering 1500 frames
    # into mislabelled videos.
    with np.load(a.dump) as z:
        check_segments(z["true_T"], list(a.case_frames))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for idx, (name, n) in enumerate(zip(a.case_names, a.case_frames)):
        stacked = None
        if a.stacked:
            stacked = _stacked_inputs(a.sim_pk[idx], n, name, frame_stride=a.frame_stride,
                                  manifest_dir=a.manifest_dir)
        side = None
        if stacked is not None and a.manifest_dir is not None:
            sidecar = pathlib.Path(a.manifest_dir) / f"{name}.manifest.json"
            if sidecar.exists():
                side = side_panel_for(json.loads(sidecar.read_text()), stacked[1])
                if side is not None:
                    print(f"[{name}] multi-beam: drawing the track plan view, not the "
                          f"centroid speed curve")
        render_case(a.dump, a.out_dir, start=start, n=n, name=name, mesh_um=a.mesh_um,
                    fps=a.fps, width=a.width, height=a.height, plane=a.plane,
                    t_scale=a.t_scale, stacked=stacked, side_draw=side)
        start += n
    return 0


def _stacked_inputs(sim_pk: int, n_frames: int, name: str, *,
                    frame_stride: int = 1, manifest_dir: "pathlib.Path | None" = None):
    """(speed, t_seconds, title) for one case.

    TWO things the AiiDA node alone cannot give you:

    * `frame_stride` -- a strided eval dumps every Nth frame, so the times must be
      subsampled the SAME way. Taking `t_s[:n_frames]` instead silently plots the first
      half of the track against the whole clip: the curve and the marker then disagree
      with the field being shown, and nothing errors because the count still matches.
    * `manifest_dir` -- a multibeam node's `derived.beams` carries the per-beam specs with
      EMPTY `path_xy_m`; aiida-pino merges the real paths in at staging time. So the beam
      curve has to come from the staged sidecar, not the node, or `_multibeam_specs`
      refuses with "N beams but per-beam paths for only 0".
    """
    from aiida import load_profile, orm  # noqa: PLC0415

    load_profile()
    node = orm.load_node(int(sim_pk))
    md = node.base.attributes.all["simulation_metadata"]
    manifest = md["flow3d_case_manifest"]
    if manifest_dir is not None:
        sidecar = pathlib.Path(manifest_dir) / f"{name}.manifest.json"
        if sidecar.exists():
            manifest = json.loads(sidecar.read_text())
    with node.base.repository.open("_time_values_s.npy", "rb") as fh:
        t_s = np.load(fh)
    t_s = np.asarray(t_s, dtype=float).ravel()[:: max(1, int(frame_stride))][:n_frames]
    if len(t_s) != n_frames:
        raise SystemExit(
            f"[{name}] sim {sim_pk} yields {len(t_s)} times at stride {frame_stride} "
            f"for {n_frames} dumped frames")
    return beam_speed(manifest, t_s), t_s, name


if __name__ == "__main__":
    raise SystemExit(main())
