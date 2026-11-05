# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false
"""Inference-time prediction dumping and metric reporting utilities."""

import csv
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...custom_utils.inferencer import FNOGridInferencer
from ...custom_utils.utils import (
    boiling_fill_value,
    fill_where_mask_smooth_np,
    iou_per_batch,
    smooth_liquid_fraction_numpy,
    t_fill_field,
)

from .frame_transfer import (
    crop_batch_to_moving_ref,
    fixed_window_slice,
    steady_entry_selection,
)
from .dataclasses import InferenceMetrics, RunConfig


_ALPHA_MASKING_NOTICES: set[str] = set()

# fl and alpha are dimensionless fractions in [0, 1], so their absolute
# error already IS the relative one and there is nothing to normalize away.
# Dividing by the sample's own spatial mean instead sent a low-melt sample's
# relative error to 1e3-1e9 (mean ~1e-3, or 0), and those values were averaged
# into the fold metrics. No relative metric is reported for these keys; IoU and
# the absolute/RMSE metrics carry them.
_UNITLESS_KEYS = ("fl", "alpha")


def _has_relative_metrics(key: str) -> bool:
    return key not in _UNITLESS_KEYS


def _print_alpha_masking_notice_once(key: str) -> None:
    """Print alpha-masking behavior once per process instead of once per inference batch."""
    if key in _ALPHA_MASKING_NOTICES:
        return
    _ALPHA_MASKING_NOTICES.add(key)
    print(f"Alpha masking enabled for {key}; outside-metal values use boiling-temperature fill.")


#: How far below a model's own recorded best an eval may land before it is called
#: out. Deliberately loose: the recorded number is a VALIDATION metric on val
#: frames and an eval is a test-split full-frame score, so they differ legitimately
#: and not always in the same direction -- 68163's rescued weights recorded ~0.012
#: in validation and scored BETTER on test. 68163's broken eval was 15x off, which
#: no honest val-vs-test spread reaches.
EVAL_VS_BEST_WARN_FACTOR = 5.0


def _warn_if_far_from_recorded_best(errors, network_dir) -> None:
    """Say so when an eval lands far below what the training demonstrably reached.

    Model 68163 recorded `mae_T 0.012049` and scored 15x worse with exit 0, a
    populated metrics.json and nothing flagged -- it read as a completed eval of a
    bad model. `use_best_checkpoint` fixes one mechanism (last weights instead of
    best); this catches the class: era mismatch, wrong geometry, wrong dataset, a
    truncated checkpoint. Anything that lands an eval far from the training's own
    result.

    Warn, never fail: the two are different quantities, so a hard stop would block
    legitimate results over an unlike comparison.
    """
    if network_dir is None:
        return
    meta = Path(network_dir) / "best_checkpoint.json"
    if not meta.is_file():
        return  # models from before best-checkpoint keeping archive none
    try:
        import json

        payload = json.loads(meta.read_text(encoding="utf-8"))
        # best_checkpoint.py writes {"metric": <NAME>, "value": <number>, "step": ...}.
        # This read `float(payload["metric"])` -- float("mae_T") -- and the bare except
        # below swallowed it, so the guard never printed on any real sidecar from the
        # day it landed (2026-08-26) until the public-cut audit reproduced it against
        # the tracked runs/gaot_s43_bestckpt sidecar and a 15x-worse eval (2026-09-14).
        recorded = float(payload["value"])
        key = str(payload.get("metric", "metric"))
        step = payload.get("step")
    except Exception:  # noqa: BLE001 - a broken sidecar must never stop an eval
        return
    if not recorded > 0:
        return
    # The eval's own headline: mean absolute error on the primary output.
    scored = None
    for out_key in ("T", "Temperature"):
        stats = errors.get(out_key) if hasattr(errors, "get") else None
        if isinstance(stats, dict) and "abs_err_mean" in stats:
            scored = float(stats["abs_err_mean"])
            break
    if scored is None or not scored > 0:
        return
    if scored <= EVAL_VS_BEST_WARN_FACTOR * recorded:
        return
    print(
        f"WARNING: this eval scored {scored:g} against the model's own recorded best "
        f"{key}={recorded:g} at step {step} -- {scored / recorded:.1f}x worse. "
        f"THERE IS PROBABLY A BUG: an eval this far below what the training reached "
        f"is usually scoring the wrong weights, the wrong geometry or the wrong "
        f"dataset, not a genuinely bad model (model 68163 read exactly like this at "
        f"exit 0, and was scoring its last checkpoint instead of its best). Compare "
        f"against best_checkpoint.json before reporting this number; "
        f"hydra_overrides=['++custom.use_best_checkpoint=true'] scores the best "
        f"weights. Some val-vs-test spread is normal -- {EVAL_VS_BEST_WARN_FACTOR:g}x "
        f"is not.",
        flush=True,
    )


def _write_metric_file(error_path: Path, errors: dict[str, dict[str, float]],
                       network_dir=None, run_cfg=None) -> None:
    """Persist compact per-output metric summaries to text.

    The single funnel every eval path writes through, which is why the
    recorded-best comparison hooks here rather than at four call sites. An
    evaluation also stores the H_norm denominator its network saw, as an
    ``[h_norm]`` block (`h_norm_record.metric_block`).
    """

    _warn_if_far_from_recorded_best(errors, network_dir)
    from .h_norm_record import metric_block

    errors = {**errors, **metric_block(run_cfg)}
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with error_path.open("w", encoding="utf-8") as f:
        for key, stats in errors.items():
            f.write(f"[{key}]\n")
            for stat_name, value in stats.items():
                f.write(f"{stat_name}: {value}\n")
            f.write("\n")


def _center_crop_numpy(array: np.ndarray, target_spatial_shape: Sequence[int]) -> np.ndarray:
    """Center-crop trailing spatial axes down to ``target_spatial_shape``.

    This used to edge-pad a short array up to the target instead, which
    replicated boundary values into a band that the model never predicted. Those
    invented voxels were then differenced against real truth in every metric, so
    growing an array is now an error and callers crop both sides to the overlap.
    """

    aligned = np.asarray(array)
    current_spatial_shape = aligned.shape[-len(target_spatial_shape) :]
    if tuple(current_spatial_shape) == tuple(target_spatial_shape):
        return aligned

    slices = [slice(None)] * aligned.ndim
    for dim_idx, (current_dim, target_dim) in enumerate(zip(current_spatial_shape, target_spatial_shape, strict=True)):
        axis = aligned.ndim - len(target_spatial_shape) + dim_idx
        if current_dim < target_dim:
            raise ValueError(
                f"Spatial alignment failed: dim {current_dim} is smaller than target {target_dim}; "
                "padding would fabricate voxels."
            )
        crop_start = (current_dim - target_dim) // 2
        slices[axis] = slice(crop_start, crop_start + target_dim)
    return aligned[tuple(slices)]


def _align_preds_to_truth_shapes(
    preds: dict[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    output_keys: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Crop predictions and truth to their common spatial overlap.

    Decoder rounding (U-Net pooling, FNO grid restore) can leave a prediction a
    voxel or two off the truth grid in either direction. Both sides are
    center-cropped to the elementwise minimum so every compared voxel is real on
    both; nothing is padded (A38).
    """

    aligned_preds = dict(preds)
    aligned_truth = dict(truth)
    for key in output_keys:
        if key not in aligned_preds or key not in aligned_truth:
            continue
        target = np.asarray(aligned_truth[key])
        pred = np.asarray(aligned_preds[key])
        if pred.shape == target.shape:
            continue
        if pred.ndim < 3 or target.ndim < 3:
            continue
        if pred.shape[:2] != target.shape[:2]:
            continue
        overlap = tuple(min(p, t) for p, t in zip(pred.shape[2:], target.shape[2:], strict=True))
        aligned_preds[key] = _center_crop_numpy(pred, overlap)
        aligned_truth[key] = _center_crop_numpy(target, overlap)
    return aligned_preds, aligned_truth


def _pointwise_spatial_shape(run_cfg: RunConfig) -> tuple[int, ...] | None:
    """Return the structured spatial shape for pointwise baselines when available."""

    grid_params = getattr(run_cfg, "grid_params", None)
    if isinstance(grid_params, Mapping):
        resolution = grid_params.get("resolution")
    else:
        resolution = None
    super_res_factor = int(getattr(run_cfg, "super_res_factor", 1) or 1)
    moving_frame_steps = int(getattr(run_cfg, "moving_frame_steps", 1) or 1)
    moving_ref_frame = bool(getattr(run_cfg, "moving_ref_frame", True))
    include_time = bool(getattr(run_cfg, "include_time", False))
    time_as_dim = bool(getattr(run_cfg, "time_as_dim", False))

    nx = getattr(run_cfg, "nx_effective", None)
    ny = getattr(run_cfg, "ny_effective", None)
    nz = getattr(run_cfg, "nz_effective", None)

    if resolution is not None and len(resolution) == 3:
        nx_res, ny_res, nz_res = (int(value) for value in resolution)
        if include_time and not time_as_dim and moving_ref_frame:
            nx_res = nx_res - max(0, (moving_frame_steps - 1) * super_res_factor)
        nx = nx_res
        ny = ny_res
        nz = nz_res

    if nx is None or nz is None:
        return None
    if bool(getattr(run_cfg, "use_2d", False)) and not bool(getattr(run_cfg, "time_as_dim", False)):
        return (int(nx), int(nz))
    if ny is None:
        return None
    return (int(nx), int(ny), int(nz))


def _restore_pointwise_grid_array(
    array: np.ndarray,
    spatial_shape: Sequence[int],
    *,
    key: str,
) -> np.ndarray:
    """Invert pointwise flattening from ``(N, C)`` back to ``(B, C, ...)`` grid tensors."""

    restored = np.asarray(array)
    if restored.ndim >= 4:
        return restored
    if restored.ndim == 1:
        restored = restored[:, None]
    if restored.ndim != 2:
        raise ValueError(
            f"Pointwise grid restoration expects '{key}' to have shape (N, C); got {restored.shape}."
        )

    points_per_sample = int(np.prod(tuple(spatial_shape)))
    if points_per_sample <= 0:
        raise ValueError(f"Invalid spatial shape {tuple(spatial_shape)} for key '{key}'.")
    if restored.shape[0] % points_per_sample != 0:
        raise ValueError(
            f"Cannot restore '{key}' with flattened shape {restored.shape} to spatial shape "
            f"{tuple(spatial_shape)}; {restored.shape[0]} rows is not divisible by {points_per_sample}."
        )
    batch = restored.shape[0] // points_per_sample
    restored = restored.reshape(batch, *tuple(spatial_shape), restored.shape[1])
    return np.moveaxis(restored, -1, 1)


def _restore_pointwise_grid_layout(
    tensors: Mapping[str, np.ndarray],
    run_cfg: RunConfig,
    *,
    keys: Sequence[str],
) -> dict[str, np.ndarray]:
    """Restore flattened pointwise tensors to structured grid layout when metadata is available."""

    spatial_shape = _pointwise_spatial_shape(run_cfg)
    restored = {key: np.asarray(value) for key, value in tensors.items()}
    if spatial_shape is None:
        return restored
    for key in keys:
        if key not in restored:
            continue
        restored[key] = _restore_pointwise_grid_array(restored[key], spatial_shape, key=key)
    return restored


def z_region_indices(z_field: np.ndarray, eps: float = 1e-9):
    """Return (substrate_mask, surface_index) along z (the last axis).

    ``substrate`` is the boolean mask of z<=0 slices (excludes the gas/void
    layer above z=0); ``surf_idx`` is the z-slice nearest 0 from below. Returns
    (None, None) if no slice has z<=0. Shared by region_error_stats and the
    streaming lazy-eval region accumulator so both define the regions identically.
    """

    z1d = np.asarray(z_field, dtype=np.float64).mean(axis=tuple(range(z_field.ndim - 1)))
    substrate = z1d <= eps
    if not substrate.any():
        return None, None
    surf_idx = int(np.argmax(np.where(substrate, z1d, -np.inf)))
    return substrate, surf_idx


def _align_z_field(z_field: np.ndarray, nz: int) -> np.ndarray:
    """Crop the z coordinate grid to ``nz`` trailing slices.

    A prediction/truth pair cropped to its overlap (A38) is shorter than the
    input coordinate grid; cropping z the same way keeps the z-derived regions
    lined up with the data.
    """

    z_field = np.asarray(z_field)
    return z_field if z_field.shape[-1] == nz else _center_crop_numpy(z_field, (nz,))


def region_error_stats(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    z_field: np.ndarray,
    eps: float = 1e-9,
    relative: bool = True,
) -> dict[str, float]:
    """Substrate-only (z<=0) and z=0-surface-plane error stats.

    The random-path grids include a gas/void layer above z=0 that is masked to
    ~zero error in both prediction and truth, deflating volume-averaged
    metrics (the straight-line grids end at z=0 and never had it). Substrate
    stats exclude those slices; surface stats report the z-slice nearest 0
    from below. z is the last axis; ``z_field`` may be normalized (sign and
    ordering are all that matter). Returns {} if no slice has z<=0.
    """

    z_field = _align_z_field(z_field, np.asarray(y_true).shape[-1])
    substrate, surf_idx = z_region_indices(z_field, eps)
    if substrate is None:
        return {}
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    out: dict[str, float] = {}
    for tag, e, t in (
        ("substrate", err[..., substrate], np.asarray(y_true)[..., substrate]),
        ("surface", err[..., surf_idx], np.asarray(y_true)[..., surf_idx]),
    ):
        mae = float(np.abs(e).mean())
        out[f"{tag}_abs_err_mean"] = mae
        out[f"{tag}_rmse"] = float(np.sqrt(np.mean(e**2)))
        if relative:
            out[f"{tag}_rel_err_mean"] = float(mae / (np.mean(t) + 1e-12))
    return out


# Half-maximum is the ONE contour both beam shapes share a meaning at: for a Gaussian
# exp(-r^2/2s^2) it sits at r = 1.1774 s, for the top-hat 0.5(1-tanh(4(r-R)/R)) it sits
# at r = R exactly. Recovering the scale there makes the beam box comparable between the
# sweep's gaussian and uniform halves; recovering it at exp(-0.5), as the original code
# did, silently shrinks a top-hat case's box to ~38 % of a gaussian case's area and so
# scores the two halves over different regions.
_GAUSS_HALF_MAX_IN_SIGMA = float(np.sqrt(2.0 * np.log(2.0)))   # 1.1774


def beam_sigma_from_footprint(
    beam_field: np.ndarray, x_field: np.ndarray, one_sigma: float = float(np.exp(-0.5))
) -> float | None:
    """Beam radius in GRID units, recovered from the `beam_dist` footprint.

    `beam_dist` is built as ``exp(-r**2 / 2 sigma**2)`` on the actual grid
    (npz_datasource `_emit_beam_family`), so the contour at ``exp(-0.5)`` is
    exactly one sigma. Recovering it this way needs no config plumbing and is in
    the same normalized units as `x` -- the reason the first version of this was
    dead code is that `laser_params` carries no radius in any of 129 saved run
    configs, and a metres radius compared against `x/L` is wrong by 1e4.
    """

    beam = np.asarray(beam_field, dtype=np.float64)
    xs = np.asarray(x_field, dtype=np.float64)
    if beam.shape != xs.shape:
        try:
            beam, xs = np.broadcast_arrays(beam, xs)
        except ValueError:
            return None
    # Per SAMPLE, then the median: under time_as_batch the leading axis stacks frames with
    # the beam in a different place in each, and the x-extent of the union ring is the
    # radius plus half the travel between frames (2cbfc60: batch 2 gave a 1.71x box).
    # The radius is one constant per run, so one float is still the right answer; frames
    # with the laser off have no ring and are skipped.
    samples = (beam, xs) if beam.ndim < 4 else zip(beam, xs)
    sigmas = []
    for b, x in ([samples] if beam.ndim < 4 else samples):
        ring = b >= one_sigma
        if ring.any():
            vals = x[ring]
            sigmas.append(float((vals.max() - vals.min()) / 2.0))
    if not sigmas:
        return None
    sigma = float(np.median(sigmas))
    return sigma if sigma > 0 else None


def beam_box_threshold(n_radii: float, uniform: bool) -> float:
    """Footprint value at `n_radii` beam radii, for the footprint shape in use.

    A single hardcoded exp(-n^2/2) is only correct for the Gaussian branch. Against the
    uniform branch's 0.5*(1-tanh(4(r-R)/R)) it solves to r = 1.232 R rather than 2 R, so
    a top-hat case's beam box came out at 37.9 % of a Gaussian case's area -- 71 of the
    230 PV-LHS cases, scored over a 2.6x smaller and hotter region under the same metric
    name. Invert each profile instead.
    """
    if uniform:
        return float(0.5 * (1.0 - np.tanh(4.0 * (n_radii - 1.0))))
    return float(np.exp(-(n_radii ** 2) / 2.0))


# Fields worth keeping per frame. T and alpha are what the melt-surface renderers
# take; everything else in a batch is either an input channel (reconstructible) or
# derived (fl = f(T)), so dumping it would multiply size for nothing.
_FRAME_DUMP_KEYS = ("T", "alpha")


def frame_dump_stride() -> int:
    """0 = off; N >= 1 = keep every Nth scored batch.

    `PINO_DUMP_FRAMES=1` keeps everything, which is one volume per frame per
    field -- 2.9 GB for 755 frames of a 150x100x40 grid. A stride is the cheap
    way to render a smooth video without paying for every frame: at N=2 and half
    the frame rate the clip has the same duration and half the size.
    """
    raw = os.environ.get("PINO_DUMP_FRAMES", "").strip().lower()
    if raw in ("", "0", "false", "no"):
        return 0
    if raw in ("true", "yes"):
        return 1
    try:
        n = int(raw)
    except ValueError:
        # A typo must not silently disable the dump the caller asked for.
        print(f"[eval] PINO_DUMP_FRAMES={raw!r} is not an integer; keeping every frame", flush=True)
        return 1
    return max(n, 1)


def frame_dump_enabled() -> bool:
    """True when PINO_DUMP_FRAMES asks for a per-frame series."""
    return frame_dump_stride() > 0


def _as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def accumulate_frame_dump(acc: dict, preds: dict, truth: dict, *,
                          stride: int = 1, index: int | None = None) -> None:
    """Append this batch's T/alpha to ``acc`` in loader order.

    Order is the point: the loader runs shuffle=False so batch order IS frame
    order, and a dump that drops or reorders batches renders as a video that
    looks plausible and is wrong.

    ``stride`` keeps every Nth batch. ``index`` is the batch counter the stride
    is applied to -- passed in rather than derived from len(acc), which would
    stop striding the moment one field is absent from a batch.
    """
    if stride > 1:
        if index is None:
            raise ValueError("a strided frame dump needs the batch index")
        if index % stride:
            return
    for prefix, source in (("pred", preds), ("true", truth)):
        for key in _FRAME_DUMP_KEYS:
            if key in source:
                acc.setdefault(f"{prefix}_{key}", []).append(_as_numpy(source[key]))


def write_frame_dump(acc: dict, output_dir) -> Path | None:
    """Concatenate the accumulated batches along the frame axis and save.

    Returns the written path, or None when nothing was accumulated -- an empty
    archive would read as "the model produced no frames" rather than "the dump
    was never fed".
    """
    if not acc:
        return None
    path = Path(output_dir) / "eval_frames_dump.npz"
    np.savez_compressed(path, **{k: np.concatenate(v, axis=0) for k, v in acc.items()})
    return path


def _preview_score(truth: dict) -> float:
    """How interesting a batch is to look at: molten fraction, else T spread.

    `fl` is the melt indicator, so its mean IS the molten fraction. A run whose
    outputs carry no `fl` still gets a sensible pick from the temperature spread
    (a flat ambient frame scores 0).
    """
    fl = truth.get("fl") if truth else None
    if fl is not None:
        a = np.asarray(fl, dtype=np.float64)
        if a.size:
            return float(np.nanmean(a))
    t = (truth or {}).get("T")
    if t is not None:
        a = np.asarray(t, dtype=np.float64)
        if a.size:
            return float(np.nanmax(a) - np.nanmin(a))
    return 0.0


def preferred_preview_batch(incumbent: dict | None, candidate: dict) -> bool:
    """True to KEEP ``incumbent``, False to replace it with ``candidate``.

    Ties keep the incumbent so the choice is deterministic, and a `None`
    incumbent always loses so the first batch is always stored.
    """
    if incumbent is None:
        return False
    return _preview_score(incumbent) >= _preview_score(candidate)


def beam_box_mask(
    beam_field: np.ndarray,
    z_field: np.ndarray,
    *,
    n_radii: float = 2.0,
    sigma: float | None = None,
    uniform_footprint: bool = False,
) -> np.ndarray | None:
    """Cells within ``n_radii`` beam radii of the beam AND inside the metal.

    Built elementwise from `beam_dist`, so a per-frame beam position is handled
    by construction: each batch row gets its own box. `beam_dist` is an x-y
    footprint broadcast over z, so ``beam_dist >= exp(-n**2/2)`` is exactly the
    disc of radius ``n*sigma``. z is bounded both by the substrate (z<=0) and,
    when sigma is known, to within ``n_radii*sigma`` of the surface -- without
    the depth bound the box is a full-depth column of mostly quiescent metal,
    which is the dilution this metric exists to remove.
    """

    beam = np.asarray(beam_field, dtype=np.float64)
    thr = beam_box_threshold(float(n_radii), bool(uniform_footprint))
    mask = beam >= thr
    if not mask.any():
        return None
    z = np.asarray(z_field, dtype=np.float64)
    try:
        mask, z = np.broadcast_arrays(mask, z)
    except ValueError:
        return None
    mask = mask & (z <= 1e-9)
    if sigma:
        mask = mask & (z >= -float(n_radii) * float(sigma))
    return mask if mask.any() else None


def beam_box_error_stats(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    box: np.ndarray,
    relative: bool = True,
    iou: bool = False,
) -> dict[str, float]:
    """Error stats inside a beam box mask, reported as ``beambox_*``.

    Whole-domain and substrate-wide means are diluted by quiescent metal and
    cannot see a defect confined to the melt pool. Measured 2026-08-03: model
    29125 held T RMSE 84 K and alpha IoU 0.998 while predicting zero melt, the
    error living in a few hundred cells around the beam.

    ``iou`` adds ``beambox_iou`` for unitless fields (fl/alpha). That is the
    metric that actually catches "predicts no melt": a whole-domain alpha IoU of
    0.998 came from getting the bulk metal volume right, while every hot cell in
    the box was wrong.
    """

    pred = np.asarray(y_pred, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.float64)
    if box.shape != pred.shape:
        try:
            box = np.broadcast_to(box, pred.shape)
        except ValueError:
            return {}
    if not box.any():
        return {}
    p = pred[box]
    t = true[box]
    e = p - t
    mae = float(np.abs(e).mean())
    out = {
        "beambox_abs_err_mean": mae,
        "beambox_rmse": float(np.sqrt(np.mean(e**2))),
        "beambox_n": int(box.sum()),
    }
    if relative:
        out["beambox_rel_err_mean"] = float(mae / (np.mean(t) + 1e-12))
    if iou:
        pb = p > 0.5
        tb = t > 0.5
        union = int(np.logical_or(pb, tb).sum())
        out["beambox_iou"] = 1.0 if union == 0 else float(np.logical_and(pb, tb).sum() / union)
    return out


def surface_distance_stats(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    *,
    cell_um: float | tuple[float, float, float] = 1.0,
) -> dict[str, float]:
    """Symmetric surface distances between two binary masks, in micrometres.

    IoU and volume error are volume-dependent: on a thin melt shell wrapping a
    keyhole, a one-cell displacement can halve IoU while the interface is
    essentially right. Surface distance measures how far the predicted interface
    sits from the true one, which is nearly independent of how much volume is
    enclosed -- the property the record has twice asked for
    ("interface-aware loss/representation", 2026-07-28).

    Reports the symmetric mean surface distance (`surface_msd_um`), the 95th
    percentile Hausdorff (`surface_hd95_um`, robust to a stray voxel) and the
    max (`surface_hd_max_um`).

    An empty prediction has no surface and therefore no finite distance. Rather
    than emit NaN -- which averages away silently -- this flags
    `surface_pred_empty` and returns the domain diagonal as the sentinel, so a
    model that predicts no melt at all scores as maximally wrong. Model 29125 is
    exactly that case.
    """

    from scipy import ndimage

    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)
    if pred.shape != true.shape:
        return {}
    # A leading batch (inference_batch_size > 1) holds SEPARATE frames, so unlike
    # the coordinate fields in _axis_um it cannot be indexed down to sample 0 --
    # that would silently score one frame and drop the rest. Score each frame and
    # average, matching how the volume stats beside this call are .mean()ed.
    # Before this, batch 2 raised "cannot reshape array of size 288000 into shape
    # (90,40,40)" and killed the eval (2026-08-13).
    if pred.ndim > 3:
        frames = [
            surface_distance_stats(p, t, cell_um=cell_um)
            for p, t in zip(pred.reshape(-1, *pred.shape[-3:]),
                            true.reshape(-1, *true.shape[-3:]))
        ]
        frames = [f for f in frames if f]
        if not frames:
            return {}
        # Booleans are not averaged: float(np.mean) turned surface_pred_empty
        # into 0.0 and broke its `is False` contract, and a MEAN of a flag is
        # the wrong aggregate anyway -- "did ANY frame predict an empty
        # surface" is the question the sentinel answers.
        return {
            k: (bool(any(f[k] for f in frames)) if k == "surface_pred_empty"
                else float(np.mean([f[k] for f in frames])))
            for k in frames[0]
        }
    spatial = pred
    tspatial = true
    # Anisotropic spacing matters: a stride of [2,2,1] leaves dz=10 um but
    # dx=dy=20 um, so a single z-derived cell size under-reports x/y distances by
    # 2x (measured 2026-08-04). Accept a per-axis tuple.
    if np.isscalar(cell_um):
        sampling = [float(cell_um)] * spatial.ndim
    else:
        sampling = [float(v) for v in cell_um][: spatial.ndim]
    diag = float(np.sqrt(sum((n * sp) ** 2 for n, sp in zip(spatial.shape, sampling))))

    if not spatial.any() or not tspatial.any():
        both_empty = (not spatial.any()) and (not tspatial.any())
        return {
            "surface_msd_um": 0.0 if both_empty else diag,
            "surface_hd95_um": 0.0 if both_empty else diag,
            "surface_hd_max_um": 0.0 if both_empty else diag,
            "surface_pred_empty": bool(not spatial.any()),
        }

    def _border(m):
        return m & ~ndimage.binary_erosion(m)

    pb, tb = _border(spatial), _border(tspatial)
    # Distance from every voxel to the nearest surface voxel of the other mask.
    d_to_true = ndimage.distance_transform_edt(~tb, sampling=sampling)
    d_to_pred = ndimage.distance_transform_edt(~pb, sampling=sampling)
    d = np.concatenate([d_to_true[pb].ravel(), d_to_pred[tb].ravel()])
    if d.size == 0:
        return {}
    return {
        "surface_msd_um": float(d.mean()),
        "surface_hd95_um": float(np.percentile(d, 95)),
        "surface_hd_max_um": float(d.max()),
        "surface_pred_empty": False,
    }


def _frame_count(field: np.ndarray) -> int:
    """Frames contributing to a peak: the leading (batch/time) dimension."""
    arr = np.asarray(field)
    return int(arr.shape[0]) if arr.ndim >= 1 else 1


def _tail_deficit_scale(deficit_K: float, n_frames: int) -> dict[str, float]:
    """N-dependence of a peak-based tail deficit, made explicit.

    A model that predicts the PHASE-AVERAGE of a fluctuating field is not merely
    missing the fluctuation -- under an L2 objective the conditional mean IS the
    minimiser, so it is trained to flatten it. If truth fluctuates with std sigma
    about its trend, the max over N frames exceeds the trend by roughly
    ``sigma * sqrt(2 ln N)``. So a peak-based deficit GROWS WITH N even for an
    unchanged model.

    Measured 2026-08-05 on the FLOW-3D sets (sigma ~112.9 K, 19-35 kHz):

        301 frames  -> predicted 381 K, measured 366.9 K
        6016 frames -> predicted 471 K, measured 449.3 K

    Consequence: `tail_deficit_max_K` MUST NOT be compared across evaluations
    with different frame counts. `tail_deficit_per_sqrt2lnN_K` divides it out and
    is an estimate of the implied sigma, which IS comparable -- and which can be
    read against the dataset's own measured fluctuation to see whether a model is
    worse than phase-averaging or merely doing it.
    """

    n = max(int(n_frames), 2)
    scale = float(np.sqrt(2.0 * np.log(n)))
    return {
        "tail_deficit_n_frames": int(n),
        "tail_deficit_sqrt2lnN": scale,
        "tail_deficit_per_sqrt2lnN_K": float(deficit_K) / scale if scale else float("nan"),
    }


def metric_melt_threshold_K(run_cfg, tensors=None):
    """Melt threshold in KELVIN for the headroom metrics: an array, or a scalar.

    Precedence, widest-evidence first, mirroring `_derive_fl`: the per-case `mat_Ts`/`mat_Tl`
    CHANNELS when this batch carries them (the only correct answer for a mixed-material set);
    then the dataset's own unanimous window from `adopt_metric_melt_window`; then the config's
    `material_params`, which is what every campaign used before and is right whenever the
    config's material IS the data's.

    Returns Kelvin, not normalised units, because that is what the headroom payload reports.
    """
    if tensors is not None:
        t_s, t_l = tensors.get("mat_Ts"), tensors.get("mat_Tl")
        if t_s is not None and t_l is not None:
            rs = run_cfg.ref_scales
            a_s = np.asarray(t_s, dtype=np.float64) * float(rs.get("mat_Ts", 1.0))
            a_l = np.asarray(t_l, dtype=np.float64) * float(rs.get("mat_Tl", 1.0))
            return 0.5 * (a_s + a_l)
    window = getattr(run_cfg, "metric_melt_window", None)
    if window is not None:
        return 0.5 * (float(window[0]) + float(window[1]))
    mp = getattr(run_cfg, "material_params", None) or {}
    return 0.5 * (float(mp.get("T_s", 0.0)) + float(mp.get("T_l", 0.0)))


def thermal_headroom_stats(
    pred_T: np.ndarray,
    pred_alpha: np.ndarray | None,
    z_field: np.ndarray,
    *,
    T_s: float,
    T_l: float,
    ref_T: float,
    true_T: np.ndarray | None = None,
    true_alpha: np.ndarray | None = None,
) -> dict[str, float]:
    """Margin between the predicted peak temperature and the melt threshold.

    Melt is ~1% of the substrate, so a model can flatten its temperature tail by
    hundreds of K while barely moving an L2 objective -- and then every
    thresholded quantity collapses to exactly zero. Measured 2026-08-04:

    - varspeed model 29125 peaked at 1631 K in the substrate metal, 267 K BELOW
      the 1898 K melt threshold, so melt volume was identically 0 while
      whole-domain T RMSE stayed 84 K and alpha IoU 0.998.
    - the 1601 model 25218 peaked at 1944 K -- only 46 K of margin.

    Nothing reported that margin: melt volume shows the consequence after the
    fact. The threshold is the MIDPOINT ``(T_s+T_l)/2``, because the liquid
    fraction is a tanh centred there, not a step at ``T_l``.

    Restricted to substrate metal: ``_apply_prediction_postprocessing`` fills
    ``T = T_boil`` wherever ``alpha < 0.5``, and the gas layer carries the same
    fill, so reading the peak over the whole domain reports the fill rather than
    a prediction (that is how a 3523 K "peak" was reported for a model whose real
    ceiling was 1631 K).
    """

    z_field = _align_z_field(z_field, np.asarray(pred_T).shape[-1])
    substrate, _ = z_region_indices(z_field)
    if substrate is None or not substrate.any():
        return {}
    T = np.asarray(pred_T, dtype=np.float64)[..., substrate] * float(ref_T)
    if pred_alpha is not None:
        alpha = np.asarray(pred_alpha, dtype=np.float64)[..., substrate]
        if alpha.shape == T.shape:
            T = T[alpha > 0.5]
    T = np.asarray(T).ravel()
    if T.size == 0:
        return {}
    threshold = 0.5 * (float(T_s) + float(T_l))
    p999 = float(np.percentile(T, 99.9))
    peak = float(T.max())
    out = {
        "melt_threshold_K": float(threshold),
        "pred_T_max_K": peak,
        "pred_T_p999_K": p999,
        "thermal_headroom_K": peak - threshold,
        # Stored as a float: a Python bool is dropped by the Dict round-trip
        # (measured on eval 30164, where the key vanished entirely while the
        # floats beside it survived). 1.0 = can melt, 0.0 = cannot.
        "can_reach_melt": float(peak > threshold),
    }
    # Tail deficit against truth. Headroom alone reads healthy for a model that
    # clears the threshold yet still runs cold: measured 2026-08-04, GAOT
    # `disp_deep_v2` had +837 K headroom while its p99.9 sat 151 K BELOW truth
    # (1771 vs 1922 K), which is the "runs cold / deeper made it worse" defect.
    # The same quantity is what drives RIGNO varspeed all the way to zero melt,
    # so it spans both regimes where headroom does not.
    if true_T is not None:
        tT = np.asarray(true_T, dtype=np.float64)[..., substrate] * float(ref_T)
        ta = true_alpha if true_alpha is not None else pred_alpha
        if ta is not None:
            ta = np.asarray(ta, dtype=np.float64)[..., substrate]
            if ta.shape == tT.shape:
                tT = tT[ta > 0.5]
        tT = np.asarray(tT).ravel()
        if tT.size:
            t999 = float(np.percentile(tT, 99.9))
            out["true_T_p999_K"] = t999
            out["true_T_max_K"] = float(tT.max())
            # p99.9-based: the better statistic, available only where the whole
            # field is held in memory.
            out["tail_deficit_K"] = t999 - p999
            # Peak-based: ALSO emitted here so it is comparable with the lazy
            # streaming path, which cannot hold order statistics and can only
            # report this one. Without it, comparing a streaming eval against a
            # non-streaming one silently mixes two different quantities that
            # differ by a factor of ~2 (measured: 1085 K peak-based vs 540 K
            # p99.9-based on the same model).
            out["tail_deficit_max_K"] = float(tT.max()) - peak
            out.update(_tail_deficit_scale(out["tail_deficit_max_K"], _frame_count(pred_T)))
    return out


def _synthetic_beam_footprint(ds_input, uniform: bool, radius_fallback=None, run_cfg_hint=None):
    """The pipeline's own footprint, rebuilt at the ORIGIN from radius and grid.

    Same closed forms as `npz_datasource._build_path_channels` -- Gaussian
    ``exp(-r^2/2 sigma^2)`` (peak 1) and the uniform top-hat
    ``0.5*(1 - tanh(4*(r-sigma)/sigma))`` -- so `beam_box_mask` thresholds it exactly as
    it thresholds the real channel and there is still ONE box definition. Do not
    re-derive the disc here: both beam-box regressions (2026-08-10, 2026-08-15) were a
    second copy of the predicate drifting from the first.

    Raises rather than returning None when the radius is missing: the caller has already
    asserted a beam-following frame, so an unsized box is a defect, and a silently absent
    beam-box metric is what put an empty column in front of a reviewer.
    """

    radius = ds_input.get("beam_radius") if hasattr(ds_input, "get") else None
    if radius is None:
        # The per-case channel wins; the dataset-level value from the manifest
        # (`adopt_beam_radius`) is the fallback, and is only set when every case agrees.
        # UNITS: the caller must supply it in the SAME units as the x/y grids it will be
        # compared against. At the metrics stage those grids are already divided by
        # ref_scales["L"], so a raw-metre radius (5e-5 against x of order 1) makes r >> sigma
        # everywhere, the footprint underflows to zero, and the box comes back EMPTY --
        # measured on evals 61523/61528/61530, which produced no beambox keys at all.
        radius = radius_fallback
    if radius is None:
        # Directive 2026-08-21: the centre is known in this frame, so use a FIXED size
        # rather than reporting nothing. `ref_scales["L"]` is the grid unit; without it
        # the value cannot be converted and a box would be the wrong size, which is worse
        # than none -- that case alone still declines.
        length = 0.0
        if run_cfg_hint is not None:
            try:
                length = float((getattr(run_cfg_hint, "ref_scales", None) or {}).get("L") or 0.0)
            except (TypeError, ValueError):
                length = 0.0
        if length <= 0:
            length = DEFAULT_REF_LENGTH_M
        radius = DEFAULT_BEAM_RADIUS_M / length
        print(
            f"[beam-box] no beam_radius channel and no dataset-level radius; sizing the "
            f"box with the DEFAULT {DEFAULT_BEAM_RADIUS_M * 1e6:.1f} um. Position is exact "
            f"(beam at the frame origin); only the SIZE is assumed.",
            flush=True,
        )
    sigma = float(np.asarray(radius).ravel()[0])
    if not sigma > 0:
        raise ValueError(f"beam_box_from_input: beam_radius must be positive, got {sigma!r}.")
    xs = ds_input.get("x")
    ys = ds_input.get("y")
    if xs is not None and ys is None:
        # A 2D (middle-plane) run has no y extent, so a beam BOX is not a thing that
        # exists here -- not a misconfiguration. Returning None routes it through the
        # documented "empty dict when not computable" path instead of killing the
        # eval; raising here broke every use_2D eval from 2026-08-20 (the CI GPU
        # smoke among them). Announced rather than silent, because this metric going
        # quietly missing is its own recurring failure.
        if not _BEAM_BOX_2D_WARNED:
            _BEAM_BOX_2D_WARNED.add(True)
            print(
                "[beam-box] no 'y' grid (2D middle-plane run): the beam box has no "
                "y extent, so beambox_* metrics are skipped for this run. In a 3D "
                "run this would be a misconfiguration -- see the raise below.",
                flush=True,
            )
        # (beam, sigma) -- the caller unpacks two values; a bare None is a TypeError.
        return None, None
    if xs is None or ys is None:
        raise ValueError(
            "beam_box_from_input: synthesising the footprint needs both 'x' and 'y' grids."
        )
    # Centre on the BEAM, not on the origin. `beam_x`/`beam_y` carry the beam's position
    # directly and are present in almost every path-bearing key set, so the box no longer
    # depends on `beam_dist` -- an ablation that drops that one channel used to drop the
    # beam-box column with it, silently, which is the tenth instance of this metric going
    # missing (2026-08-22: "beam box should not need beam dist, fix it").
    # In a beam-following frame these channels are absent or zero and this reduces to the
    # origin, which is where the beam is there by construction, so one expression covers both.
    cx = ds_input.get("beam_x") if hasattr(ds_input, "get") else None
    cy = ds_input.get("beam_y") if hasattr(ds_input, "get") else None
    xs_a = np.asarray(xs, dtype=np.float64)
    ys_a = np.asarray(ys, dtype=np.float64)
    if cx is not None:
        xs_a = xs_a - np.asarray(cx, dtype=np.float64)
    if cy is not None:
        ys_a = ys_a - np.asarray(cy, dtype=np.float64)
    r2 = xs_a ** 2 + ys_a ** 2
    # float32 to match what `_build_path_channels` actually emits. Not cosmetic: the mask
    # is `footprint >= exp(-n^2/2)`, and at r = n*sigma the two sides are equal, so a
    # float64 synthesis includes the boundary ring that the float32 channel excludes --
    # 44 cells on a 41x41x20 probe, entirely at r = 2*sigma. Same dtype, same box.
    if uniform:
        r = np.sqrt(r2)
        return (0.5 * (1.0 - np.tanh(4.0 * (r - sigma) / sigma))).astype(np.float32), sigma
    return np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32), sigma


#: Beam radius assumed when a beam-following frame carries neither the per-case
#: `beam_radius` channel nor a dataset-level value from `adopt_beam_radius`. The beam
#: CENTRE is known by construction in that frame (it is the grid origin), so only the
#: SIZE is missing, and a box of assumed size beats no box at all -- three eval rounds
#: produced an empty beam-box column because the code preferred silence. 50 um is the
#: same figure `build_constant_path_channels` already falls back to, and is spkh123's
#: actual radius. Announced whenever used, never silent.
DEFAULT_BEAM_RADIUS_M = 5e-5
#: Grid unit assumed when no ref_scales are reachable. Every config in this repo sets
#: ref_scales["L"] = 1e-4, so the default radius is 0.5 in grid units.
DEFAULT_REF_LENGTH_M = 1e-4


def _beam_radius_in_grid_units(run_cfg):
    """Dataset beam radius expressed in the units the x/y metric grids use.

    `adopt_beam_radius` stores raw metres; the grids reaching the metrics have already
    been divided by ref_scales["L"]. Comparing the two directly is what made the first
    synthesized beam box come back empty on every steady eval.
    """
    radius = (getattr(run_cfg, "laser_params", None) or {}).get("beam_radius_m")
    if radius is None:
        return None
    try:
        length = float((getattr(run_cfg, "ref_scales", None) or {}).get("L") or 0.0)
    except (TypeError, ValueError):
        return None
    return float(radius) / length if length > 0 else None


_BEAM_BOX_2D_WARNED: set = set()


#: What a caller must project out of a batch for `beam_box_from_input` to work.
#: `beam_x`/`beam_y` are in this list because the builder centres the box on them when
#: `beam_dist` is absent -- omitting them silently removes every beambox_* key from any
#: arm whose channel set drops `beam_dist`, which is how the metric went missing from
#: 82336/82400 (2026-09-01) ten days after the builder was taught not to need it.
#: Callers that pass the WHOLE batch do not need this; the lazy path projects and does.
BEAM_BOX_INPUT_KEYS = (
    "beam_dist", "x", "y", "beam_x", "beam_y", "beam_radius", "beam_shape_uniform",
)


def beam_box_from_input(ds_input, z_grid, n_radii: float = 2.0, beam_at_origin: bool = False,
                        beam_radius=None, run_cfg_hint=None):
    """Beam-box mask from `beam_dist` in ``ds_input``, or None.

    None when the run has no beam channel and the beam position is unknown, or the laser
    is off for these frames (`beam_dist` is gated to zero past `laser_off_time_s`), in
    which case no box is reported rather than a wrong one.

    ``beam_at_origin`` says the caller is in a BEAM-FOLLOWING frame (moving-reference or
    steady), where the beam is pinned at the grid origin by construction and the box is a
    determined geometric region -- on a fixed-radius set like spkh123 it is the same box
    for every case. Then `beam_dist` is not needed and the footprint is rebuilt from
    `beam_radius`. Measured, not assumed: eval 60434's stored `in_beam_dist` peaks at
    x = -1.9 um, y = -5.0 um on a 10 um grid, and its >0.5 disc spans +-55 um, matching
    r < 1.177*sigma at sigma = 50 um.

    It is OFF by default and must stay that way for lab-frame runs: there the beam moves,
    so the origin is the wrong place and the box would cover quiescent metal -- which is
    not a degraded metric but the exact documented failure, a beam-box number sitting on
    the substrate number.
    """

    beam = ds_input.get("beam_dist") if hasattr(ds_input, "get") else None
    synth_sigma = None
    if beam is None:
        # The box needs a beam CENTRE, not the `beam_dist` channel. Two ways to know one:
        # a beam-following frame pins it at the origin, or `beam_x`/`beam_y` state it
        # outright. Requiring `beam_dist` is what made the metric vanish from every key set
        # that omitted it -- including the 8-channel ablation arm, whose set carries
        # beam_x and beam_y and therefore knows exactly where the beam is.
        _has_centre = (
            (ds_input.get("beam_x") is not None and ds_input.get("beam_y") is not None)
            if hasattr(ds_input, "get") else False
        )
        if not (beam_at_origin or _has_centre):
            return None
        shape_ch0 = ds_input.get("beam_shape_uniform") if hasattr(ds_input, "get") else None
        uniform0 = bool(np.asarray(shape_ch0).ravel()[0] > 0.5) if shape_ch0 is not None else False
        beam, synth_sigma = _synthetic_beam_footprint(ds_input, uniform0, beam_radius, run_cfg_hint)
        if beam is None:
            return None
        if not np.isfinite(beam).any() or float(np.nanmax(beam)) <= 0.0:
            raise ValueError(
                "beam_box_from_input: the synthesised footprint is empty. Almost always a "
                "UNIT mismatch -- beam_radius must be in the same units as the x/y grids, "
                "which are divided by ref_scales['L'] at the metrics stage."
            )
    beam = np.asarray(beam)
    xs = ds_input.get("x") if hasattr(ds_input, "get") else None
    if synth_sigma is not None:
        # Exact by construction; fitting it back out of our own footprint would only
        # add error, and the depth bound n_radii*sigma depends on it.
        sigma = synth_sigma
    else:
        sigma = beam_sigma_from_footprint(beam, np.asarray(xs)) if xs is not None else None
    # `beam_shape_uniform` is an input channel on the PV-LHS arms; absent (every earlier
    # dataset) it defaults to the Gaussian branch, so nothing previously measured moves.
    shape_ch = ds_input.get("beam_shape_uniform") if hasattr(ds_input, "get") else None
    uniform = bool(np.asarray(shape_ch).ravel()[0] > 0.5) if shape_ch is not None else False
    return beam_box_mask(beam, z_grid, n_radii=n_radii, sigma=sigma,
                         uniform_footprint=uniform)


def _extent_cells(flags: np.ndarray) -> np.ndarray:
    """Per-sample span, in cells, of the True band of a ``(batch, n)`` mask."""

    first = np.argmax(flags, axis=1)
    last = flags.shape[1] - 1 - np.argmax(flags[:, ::-1], axis=1)
    return np.where(flags.any(axis=1), last - first + 1, 0).astype(np.float64)


def _mask_geometry(
    mask: np.ndarray, cell_um: float, cell_xyz_um: tuple[float, float, float] | None = None
) -> dict[str, np.ndarray]:
    """Per-sample depth/length/width/volume of a ``(batch, ..., x, y, z)`` mask.

    z is the last axis and increases upward, so the mask must already be
    restricted to substrate slices: depth then counts cells from the surface
    slice down to the deepest hit, the ``nzm - z_min`` convention aiida-pino
    uses. Columns on the x/y boundary are dropped, as they are there — a gas
    column against the domain wall is an edge artifact, not a keyhole.
    """

    mask = np.array(mask, dtype=bool)
    if mask.ndim > 4:
        mask = mask.any(axis=tuple(range(1, mask.ndim - 3)))
    mask[:, [0, -1], :, :] = False
    mask[:, :, [0, -1], :] = False
    z_hit = mask.any(axis=(1, 2))
    nz = mask.shape[-1]
    depth = np.where(z_hit.any(axis=1), nz - np.argmax(z_hit, axis=1), 0).astype(np.float64)
    # Per-axis spacing. Using one z-derived cell size on all three axes is right
    # only on an isotropic grid: a stride of [2,2,1] leaves dz=10 um but
    # dx=dy=20 um, which under-reported x/y lengths by 2x and volumes by 4x on
    # every strided eval before 2026-08-04. `cell_um` remains the fallback so a
    # caller that cannot supply x/y behaves exactly as before.
    dx, dy, dz = cell_xyz_um if cell_xyz_um else (cell_um, cell_um, cell_um)
    return {
        "depth_um": depth * dz,
        "length_um": _extent_cells(mask.any(axis=(2, 3))) * dx,
        "width_um": _extent_cells(mask.any(axis=(1, 3))) * dy,
        "volume_um3": mask.sum(axis=(1, 2, 3)).astype(np.float64) * (dx * dy * dz),
    }


# Melt pool from fl > 0.5, keyhole cavity from alpha < 0.5, both inside the
# metal: the thresholds and mask semantics of the aiida-pino per-simulation
# stats (console/viz.py), so an eval number reads against the dataset's own.
_GEOMETRY_FIELDS = (("melt", "fl", False), ("keyhole", "alpha", True))


def melt_geometry_stats(
    preds: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    z_field: np.ndarray,
    length_scale_m: float = 1.0,
    x_field: np.ndarray | None = None,
    y_field: np.ndarray | None = None,
) -> dict[str, float]:
    """Melt-pool and keyhole geometry scalars, predicted against ground truth.

    Voxel-wise error says nothing about whether the predicted pool has the right
    shape, which is what a welding reader cares about. For each of melt pool and
    keyhole this reports ``<tag>_<quantity>_true``, ``_pred`` and ``_abs_err``
    (the per-sample absolute gap, so over- and under-prediction cannot cancel),
    batch-averaged. ``length_scale_m`` is the config ``ref_scales["L"]`` that
    normalized z; with the default of 1.0 the numbers are in grid units, not um.
    Returns {} when the grid has no substrate slice or carries neither field.
    """

    fields = [entry for entry in _GEOMETRY_FIELDS if entry[1] in preds and entry[1] in truth]
    if not fields:
        return {}
    z_field = _align_z_field(z_field, np.asarray(truth[fields[0][1]]).shape[-1])
    substrate, _ = z_region_indices(z_field)
    if substrate is None:
        return {}
    z1d = np.asarray(z_field, dtype=np.float64).mean(axis=tuple(range(z_field.ndim - 1)))
    if z1d.size < 2:
        return {}
    cell_um = float(np.mean(np.diff(z1d))) * length_scale_m * 1e6

    def _axis_um(field, axis):
        """Spacing along one axis, or None when the coordinate is unavailable."""
        if field is None:
            return None
        arr = np.asarray(field, dtype=np.float64)
        if arr.ndim < 3:
            return None
        # Take the FIRST sample rather than reshaping: with inference_batch_size
        # > 1 the leading dim is the batch, so `reshape(shape[-3:])` raised
        # "cannot reshape array of size 360000 into shape (90,40,50)" and killed
        # the whole eval (gaot_pvlhs_base_32k, 2026-08-12, batch 2). Coordinate
        # fields are identical across the batch, so sample 0 is the geometry.
        arr = arr.reshape(-1, *arr.shape[-3:])[0]
        line = arr.mean(axis=tuple(i for i in range(3) if i != axis))
        if line.size < 2:
            return None
        return float(np.mean(np.diff(line))) * length_scale_m * 1e6

    dx = _axis_um(x_field, 0)
    dy = _axis_um(y_field, 1)
    cell_xyz = (dx, dy, cell_um) if (dx and dy) else None

    out: dict[str, float] = {}
    for tag, key, is_cavity in fields:
        sides = {}
        for side, source in (("pred", preds), ("true", truth)):
            field = np.asarray(source[key], dtype=np.float64)[..., substrate]
            sides[side] = _mask_geometry(
                field < 0.5 if is_cavity else field > 0.5, cell_um, cell_xyz
            )
        for quantity, pred_values in sides["pred"].items():
            true_values = sides["true"][quantity]
            out[f"{tag}_{quantity}_pred"] = float(pred_values.mean())
            out[f"{tag}_{quantity}_true"] = float(true_values.mean())
            out[f"{tag}_{quantity}_abs_err"] = float(np.abs(pred_values - true_values).mean())
        # Surface distance beside the volume scalars: on a thin melt shell a
        # one-cell displacement halves IoU while the interface is nearly right,
        # so a distance in micrometres separates models that IoU cannot. On the
        # d12/d18 pair it gave 5x the separation of IoU (2026-08-04).
        pred_field = np.asarray(preds[key], dtype=np.float64)[..., substrate]
        true_field = np.asarray(truth[key], dtype=np.float64)[..., substrate]
        pm = pred_field < 0.5 if is_cavity else pred_field > 0.5
        tm = true_field < 0.5 if is_cavity else true_field > 0.5
        surf = surface_distance_stats(pm, tm, cell_um=cell_xyz or cell_um)
        for name, value in surf.items():
            out[f"{tag}_{name.replace('surface_', '')}"] = value
    # Self-describing scale, so a consumer never has to infer it from the stride.
    out["cell_um_z"] = cell_um
    if cell_xyz:
        out["cell_um_x"], out["cell_um_y"] = cell_xyz[0], cell_xyz[1]
        out["axes_isotropic"] = bool(abs(cell_xyz[0] - cell_um) < 1e-9 and abs(cell_xyz[1] - cell_um) < 1e-9)
    return out


def _with_fl(tensors: Mapping[str, np.ndarray], run_cfg: RunConfig) -> Mapping[str, np.ndarray]:
    """``tensors`` plus a derived ``fl`` when it carries T but no fl.

    The ground-truth T carries the same boiling fill as the prediction, so its
    fl must be derived through the same masked helper (see ``_derive_fl``).
    """

    if "fl" in tensors or "T" not in tensors:
        return tensors
    return {**tensors, "fl": _derive_fl(tensors, run_cfg)}


def _compute_error_stats(
    preds: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    metric_keys: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute aggregate error stats for a key-aligned prediction/ground-truth mapping."""

    errors: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        y_pred = np.asarray(preds[key])
        y_true = np.asarray(truth[key])
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch for {key}: pred {y_pred.shape}, true {y_true.shape}")

        err = y_pred - y_true
        abs_err = np.abs(err)
        rmse = np.sqrt(np.mean(err ** 2))

        stats: dict[str, float] = {
            "abs_err_mean": float(abs_err.mean()),
            "abs_err_std": float(abs_err.std()),
            "rmse": float(rmse),
        }
        if _has_relative_metrics(key):
            denom = np.mean(y_true) + 1e-12
            rel_err = abs_err / denom
            stats["rel_err_mean"] = float(rel_err.mean())
            stats["rel_err_std"] = float(rel_err.std())
            stats["rel_rmse"] = float(rmse / denom)
        if key in _UNITLESS_KEYS and y_pred.ndim >= 4:
            stats.update(_iou_stats(y_pred, y_true))
        errors[key] = stats
    return errors


def _track_prediction_extremes(acc: dict, y_pred, y_true, batch_idx: int, aux: dict | None = None) -> None:
    """Running min/max of a prediction, plus where the worst cell is.

    RMSE >> MAE means a handful of cells carry the error, and the aggregate alone
    cannot tell you whether that is a shape error or a numerical blow-up. Measured
    2026-07-30 on `gaot_mat_steady_20k`: T MAE 17.75 K against T RMSE 6279 K, a ratio
    of 354, while that model's own validation on 5 cases read 47.7 K. Nothing in the
    eval recorded a prediction range, so there was no way to see it without re-running
    inference by hand.

    Records the batch index and the unravelled location of the extreme cell so the
    offending case and voxel can be found directly, plus `alpha` and `z` there when
    available -- `alpha > 0.5` means the masking fill passed the value through
    untouched (by design, it is the metal region), which distinguishes a blow-up
    inside the substrate from one leaking through the gas fill.
    """
    y_pred = np.asarray(y_pred)
    if y_pred.size == 0:
        return
    amax = float(y_pred.max())
    amin = float(y_pred.min())
    if amax > acc.get("pred_max", -np.inf):
        acc["pred_max"] = amax
        idx = int(np.argmax(y_pred))
        loc = np.unravel_index(idx, y_pred.shape)
        acc["pred_max_batch"] = int(batch_idx)
        acc["pred_max_loc"] = [int(v) for v in loc]
        acc["pred_max_truth"] = float(np.asarray(y_true).reshape(-1)[idx])
        for name in ("alpha", "z"):
            arr = (aux or {}).get(name)
            if arr is not None:
                arr = np.asarray(arr)
                if arr.size == y_pred.size:
                    acc[f"pred_max_{name}"] = float(arr.reshape(-1)[idx])
    acc["pred_min"] = min(amin, acc.get("pred_min", np.inf))


def _accumulate_iou(acc: dict, y_pred, y_true, threshold: float = 0.5) -> None:
    """Accumulate IoU over frames whose GROUND TRUTH is non-empty.

    The streaming path keeps running sums instead of calling `_iou_stats`, so it needs
    its own support-aware accumulation -- fixing only the batched path left every lazy
    eval reporting the old all-frames mean (`iou_support_frames` absent). See
    `_iou_stats` for why empty-GT frames must not be averaged in.
    """
    y_true = np.asarray(y_true)
    iou = iou_per_batch(y_pred, y_true, threshold=threshold)
    support = (y_true > threshold).any(axis=tuple(range(1, y_true.ndim)))
    acc["iou_sum"] += float(iou[support].sum())
    acc["iou_sq_sum"] += float(np.square(iou[support]).sum())
    acc["iou_count"] += int(support.sum())
    acc["iou_all_sum"] = acc.get("iou_all_sum", 0.0) + float(iou.sum())
    acc["iou_all_count"] = acc.get("iou_all_count", 0) + int(iou.size)
    acc["iou_empty_gt_frames"] = acc.get("iou_empty_gt_frames", 0) + int((~support).sum())
    acc["iou_empty_gt_false_positive_frames"] = (
        acc.get("iou_empty_gt_false_positive_frames", 0) + int((~support & (iou < 1.0)).sum())
    )


def _iou_stats(y_pred, y_true, threshold: float = 0.5) -> dict[str, float]:
    """IoU averaged over frames whose GROUND TRUTH is non-empty.

    A frame with empty GT scores 1.0 for predicting nothing and exactly 0.0 for
    predicting anything, so including those frames makes the mean bimodal noise
    rather than accuracy -- see `iou_per_batch_with_support`. The legacy
    all-frames value is still reported as `iou_mean_all_frames` so older numbers
    remain comparable, and the empty-GT frames are counted rather than hidden.
    """
    iou = iou_per_batch(y_pred, y_true, threshold=threshold)
    # Support = frames whose GROUND TRUTH has anything above threshold. Computed here
    # rather than via a second helper import so existing test stubs of
    # custom_utils.utils (which provide only iou_per_batch) keep working.
    y_true = np.asarray(y_true)
    support = (y_true > threshold).any(axis=tuple(range(1, y_true.ndim)))
    n_empty = int((~support).sum())
    stats = {
        "iou_mean": float(iou[support].mean()) if support.any() else float("nan"),
        "iou_std": float(iou[support].std()) if support.any() else float("nan"),
        "iou_mean_all_frames": float(iou.mean()),
        "iou_support_frames": int(support.sum()),
        "iou_empty_gt_frames": n_empty,
        # Empty GT but the model predicted something: a false positive on nothing.
        # This is the signal the old mean was conflating with shape error.
        "iou_empty_gt_false_positive_frames": int((~support & (iou < 1.0)).sum()),
    }
    return stats


_FL_WINDOW_NOTED: set = set()


def _derive_fl(
    tensors: Mapping[str, np.ndarray],
    run_cfg: RunConfig,
    mask_key: str = "alpha",
) -> np.ndarray:
    """Liquid fraction from ``T``, zero-filled outside the metal region.

    With alpha_masking on, the T the network is trained against has its
    off-metal cells filled with the boiling temperature, so a predicted T reads
    ~T_boil across the gas layer by construction and any fl derived from it is 1
    there -- against a ground-truth fl of 0. Deriving fl before the fill does not
    help, because the fill is already inside the prediction. Every derivation
    site must go through this helper; an unmasked one put pred fl>0.5 on 24.9% of
    the random-path test points against 0.11% in GT (fl IoU 0.0002).
    """

    # Same per-case window as the TARGET (preprocessing_datasets). If prediction and
    # truth used different windows the metric would compare two different questions.
    # Channels arrive divided by their own ref_scale, so rescale to the T scale.
    _rs = run_cfg.ref_scales
    _Ts = tensors.get("mat_Ts")
    _Tl = tensors.get("mat_Tl")
    if _Ts is not None and _Tl is not None:
        _Ts = _Ts * float(_rs.get("mat_Ts", 1.0)) / _rs["T"]
        _Tl = _Tl * float(_rs.get("mat_Tl", 1.0)) / _rs["T"]
        if "per_case" not in _FL_WINDOW_NOTED:
            _FL_WINDOW_NOTED.add("per_case")
            print("[fl] predicted melt fraction: PER-CASE window from mat_Ts/mat_Tl", flush=True)
    else:
        _Ts = run_cfg.material_params["T_s"] / _rs["T"]
        _Tl = run_cfg.material_params["T_l"] / _rs["T"]
    fl = smooth_liquid_fraction_numpy(tensors["T"], _Ts, _Tl)
    if run_cfg.alpha_masking and mask_key in tensors:
        fl = fill_where_mask_smooth_np(
            fl, tensors[mask_key], threshold=0.5, d=0.0, k=20, inplace=False
        )
    return fl


def _apply_prediction_postprocessing(
    preds: dict[str, np.ndarray],
    run_cfg: RunConfig,
    target_keys: Sequence[str],
    invar: dict | None = None,
) -> dict[str, np.ndarray]:
    """Mirror standalone inference masking logic for rollout predictions.

    ``invar`` is the batch's input dict when the caller has it; the per-case mask fill
    (`custom.per_case_t_boil`) reads `mat_Tv` from it. Without it the scalar fill applies.
    """

    # Physical clamp of the fed-back state (mitigation #1): keeps a drifting
    # rollout from feeding unphysical alpha/T back into the next step. Applied
    # before fl is re-derived so the floored T flows through. Off by default.
    if getattr(run_cfg, "autoregressive_clamp_feedback", False):
        if "alpha" in preds:
            preds["alpha"] = np.clip(preds["alpha"], 0.0, 1.0)
        if "T" in preds:
            t_max = getattr(run_cfg, "autoregressive_clamp_t_max", None)
            preds["T"] = np.clip(
                preds["T"], getattr(run_cfg, "autoregressive_clamp_t_min", 0.0), t_max
            )

    if "T" in preds:
        # The per-case melt window rides in the inputs when the eval serves it
        # (custom.eval_per_case_melt_window); the target fl was built with the same pair.
        _window = {k: invar[k] for k in ("mat_Ts", "mat_Tl") if isinstance(invar, dict) and k in invar}
        preds["fl"] = _derive_fl({**preds, **_window} if len(_window) == 2 else preds, run_cfg)
    if run_cfg.alpha_masking and "alpha" in preds:
        for key in target_keys:
            if key not in preds:
                continue
            if key == "T":
                preds[key] = fill_where_mask_smooth_np(
                    preds[key],
                    preds["alpha"],
                    threshold=0.5,
                    d=t_fill_field(run_cfg, invar),
                    k=20,
                    inplace=False,
                )
            elif key not in ("x", "y", "z", "P", "V_scan", "Deposition"):
                preds[key] = fill_where_mask_smooth_np(
                    preds[key],
                    preds["alpha"],
                    threshold=0.5,
                    d=0.0,
                    k=20,
                    inplace=False,
                )
        # A derived fl is already zero-filled by _derive_fl; a predicted one goes
        # through the elif above with every other output key.
        for vel_key in ("u", "v", "w"):
            if vel_key in preds and "fl" in preds:
                preds[vel_key] = fill_where_mask_smooth_np(
                    preds[vel_key], preds["fl"], threshold=0.5, d=0.0, k=20, inplace=False
                )
    return preds


def _forward_model_chunked(
    model: torch.nn.Module,
    invar: Mapping[str, np.ndarray],
    output_names: Sequence[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run model forward on numpy inputs in chunks and return numpy outputs."""

    total_batch = next(iter(invar.values())).shape[0]
    chunk_size = max(1, min(int(batch_size), total_batch))
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_chunks: dict[str, list[np.ndarray]] = {key: [] for key in output_names}
    for start in range(0, total_batch, chunk_size):
        stop = min(start + chunk_size, total_batch)
        invar_t = {
            key: torch.from_numpy(np.asarray(value[start:stop])).to(device=device, dtype=torch.float32)
            for key, value in invar.items()
        }
        with torch.inference_mode():
            pred_out = model(invar_t)
        for key in output_names:
            if key not in pred_out:
                raise KeyError(f"Model output missing expected key '{key}'. Got keys: {list(pred_out.keys())}")
            output_chunks[key].append(pred_out[key].detach().cpu().numpy())

    return {key: np.concatenate(chunks, axis=0) for key, chunks in output_chunks.items()}


def _flatten_grid_rows(invar: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten grid tensors ``(B, C, ...)`` to pointwise ``(N, C)`` rows.

    Row order matches preprocessing's pointwise flattening: batch-major, then
    voxels in row-major order, so restored tensors align with grid ground truth.
    """

    flattened: dict[str, np.ndarray] = {}
    for key, value in invar.items():
        arr = np.asarray(value)
        if arr.ndim < 3:
            raise ValueError(
                f"Pointwise rollout flattening expects grid tensors shaped (B, C, ...); "
                f"key '{key}' has shape {arr.shape}."
            )
        flattened[key] = np.moveaxis(arr, 1, -1).reshape(-1, arr.shape[1])
    return flattened


def _restore_grid_rows(rows: np.ndarray, batch: int, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Restore pointwise ``(N, C)`` rows back to a grid tensor ``(B, C, ...)``."""

    rows = np.asarray(rows)
    restored = rows.reshape(batch, *spatial_shape, rows.shape[1])
    return np.moveaxis(restored, -1, 1)


def run_autoregressive_rollout_and_metrics(
    model: torch.nn.Module,
    trajectory_var: dict[str, np.ndarray],
    run_cfg: RunConfig,
    output_dir: Path,
    write_files: bool = True,
    collect: dict | None = None,
) -> InferenceMetrics:
    """Roll out one-step autoregressive predictions over full trajectories and compute metrics.

    ``write_files=False`` skips every on-disk artifact (the lazy per-case twin
    calls this once per case). ``collect``, when given, receives
    ``errors_full``/``errors_k``/``n_frames`` so callers can aggregate both
    horizons without re-deriving them from files.
    """

    if not run_cfg.autoregressive_enabled:
        raise ValueError("run_autoregressive_rollout_and_metrics requires autoregressive_enabled=True.")
    if not run_cfg.autoregressive_eval_use_gt_init:
        raise ValueError("autoregressive.eval_use_gt_init=False is not implemented in this rollout path.")

    context_keys = list(run_cfg.autoregressive_context_keys)
    state_keys = list(run_cfg.autoregressive_state_keys)
    prev_suffix = run_cfg.autoregressive_state_prev_suffix
    if not context_keys or not state_keys:
        raise ValueError("Autoregressive rollout requires non-empty context/state key lists.")

    required_keys = list(dict.fromkeys(context_keys + state_keys + ["fl"]))
    missing = [key for key in required_keys if key not in trajectory_var]
    if missing:
        raise KeyError(f"Autoregressive rollout missing trajectory keys: {missing}")

    ref_key = context_keys[0]
    ref = np.asarray(trajectory_var[ref_key])
    if ref.ndim not in (5, 6):
        raise ValueError(
            f"Autoregressive rollout expects temporal tensors with ndim in (5, 6); got {ref.shape} for '{ref_key}'."
        )
    n_batch, n_time = ref.shape[:2]
    if n_time < 2:
        raise ValueError(f"Autoregressive rollout requires at least two timesteps; got {n_time}.")

    for key in required_keys:
        arr = np.asarray(trajectory_var[key])
        if arr.shape[:2] != (n_batch, n_time):
            raise ValueError(
                f"Temporal shape mismatch for key '{key}': {arr.shape[:2]} != {(n_batch, n_time)}."
            )

    current_state = {key: np.asarray(trajectory_var[key][:, 0, ...]).copy() for key in state_keys}
    rollout_preds: dict[str, list[np.ndarray]] = {key: [] for key in state_keys}
    rollout_fl: list[np.ndarray] = []

    pointwise_layout = getattr(run_cfg, "data_layout", "grid") == "pointwise"
    # Temporal bundling (upstream RIGNO stepping.py:369-404): one forward emits
    # frames t+1..t+B (`{key}_bnd{i}` outputs), the state advances by the LAST
    # bundled frame, so the rollout composes ~B x fewer times. bundle_steps=1
    # keeps the original one-step loop bit for bit.
    bundle_steps = max(1, int(getattr(run_cfg, "autoregressive_bundle_steps", 1)))
    if bundle_steps > 1 and pointwise_layout:
        raise ValueError("autoregressive.bundle_steps>1 rollout is grid-only.")
    # Teacher forcing: restart every step from GROUND TRUTH instead of the model's
    # own output, so the result is the per-frame one-step error with NO accumulation.
    # That is the honest baseline for a free-running curve -- holding the step-1 error
    # flat instead misreads frame difficulty, which rises as the keyhole develops.
    teacher_forcing = bool(getattr(run_cfg, "autoregressive_teacher_forcing", False))
    if teacher_forcing and bundle_steps > 1:
        raise ValueError(
            "autoregressive.teacher_forcing is not supported with bundle_steps>1: "
            "a bundled forward advances the state by the LAST bundled frame, so "
            "'the ground truth for this step' is ambiguous."
        )
    bundle_out_names = list(state_keys) + [
        f"{key}_bnd{i}" for key in state_keys for i in range(1, bundle_steps)
    ]
    step = 0
    while step < n_time - 1:
        if bundle_steps > 1:
            take = min(bundle_steps, n_time - 1 - step)
            step_input = {key: np.asarray(trajectory_var[key][:, step + 1, ...]) for key in context_keys}
            # Per-bundled-frame exogenous context, mirroring the training
            # layout. Past the trajectory end (partial last bundle) the last
            # available frame is repeated; those outputs are discarded anyway.
            for key in context_keys:
                for i in range(1, bundle_steps):
                    src = min(step + 1 + i, n_time - 1)
                    step_input[f"{key}__ar{i}"] = np.asarray(trajectory_var[key][:, src, ...])
            for key in state_keys:
                step_input[f"{key}{prev_suffix}"] = current_state[key]
            bundle_pred = _forward_model_chunked(
                model=model,
                invar=step_input,
                output_names=bundle_out_names,
                batch_size=run_cfg.inference_batch_size,
            )
            for i in range(take):
                frame = {
                    key: bundle_pred[key if i == 0 else f"{key}_bnd{i}"] for key in state_keys
                }
                frame = _apply_prediction_postprocessing(frame, run_cfg, target_keys=state_keys)
                for key in state_keys:
                    rollout_preds[key].append(frame[key])
                if "fl" in frame:
                    rollout_fl.append(frame["fl"])
                current_state = {key: frame[key] for key in state_keys}
            step += take
            continue
        step_input = {key: np.asarray(trajectory_var[key][:, step + 1, ...]) for key in context_keys}
        for key in state_keys:
            step_input[f"{key}{prev_suffix}"] = current_state[key]

        if pointwise_layout:
            # DeepONet/FC consume flattened (N, C) rows; restore to grid for
            # postprocessing, metrics, and the next step's *_prev state.
            flat_pred = _forward_model_chunked(
                model=model,
                invar=_flatten_grid_rows(step_input),
                output_names=state_keys,
                batch_size=run_cfg.inference_batch_size,
            )
            step_pred = {
                key: _restore_grid_rows(
                    flat_pred[key],
                    batch=current_state[key].shape[0],
                    spatial_shape=current_state[key].shape[2:],
                )
                for key in state_keys
            }
        else:
            step_pred = _forward_model_chunked(
                model=model,
                invar=step_input,
                output_names=state_keys,
                batch_size=run_cfg.inference_batch_size,
            )
        step_pred = _apply_prediction_postprocessing(step_pred, run_cfg, target_keys=state_keys)
        for key in state_keys:
            rollout_preds[key].append(step_pred[key])
            current_state[key] = (
                np.asarray(trajectory_var[key][:, step + 1, ...]).copy()
                if teacher_forcing
                else step_pred[key]
            )
        if "fl" in step_pred:
            rollout_fl.append(step_pred["fl"])
        step += 1

    pred_rollout = {key: np.stack(values, axis=1) for key, values in rollout_preds.items()}
    truth_rollout = {key: np.asarray(trajectory_var[key][:, 1:, ...]) for key in state_keys}
    if rollout_fl:
        pred_rollout["fl"] = np.stack(rollout_fl, axis=1)
        truth_rollout["fl"] = np.asarray(trajectory_var["fl"][:, 1:, ...])

    metric_keys = [key for key in list(state_keys) + ["fl"] if key in pred_rollout and key in truth_rollout]

    def _frames_view(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Merge (B, T-1, ...) to (B*(T-1), ...) so IoU is per-frame like the
        # direct eval; elementwise stats (MAE/RMSE) are unchanged by this.
        return {key: np.asarray(value).reshape(-1, *np.asarray(value).shape[2:]) for key, value in tensors.items()}

    errors_full = _compute_error_stats(_frames_view(pred_rollout), _frames_view(truth_rollout), metric_keys=metric_keys)

    horizon_k = min(run_cfg.autoregressive_eval_horizon_k, n_time - 1)
    pred_k = {key: value[:, :horizon_k, ...] for key, value in pred_rollout.items()}
    truth_k = {key: value[:, :horizon_k, ...] for key, value in truth_rollout.items()}
    errors_k = _compute_error_stats(_frames_view(pred_k), _frames_view(truth_k), metric_keys=metric_keys)

    # Built unconditionally: the lazy twin runs one case per call with
    # write_files only on the first, so gating this on write_files threw away
    # the per-frame curve for every other case.
    per_step_rows: list[dict[str, Any]] = []
    for step in range(n_time - 1):
        for key in metric_keys:
            y_pred = pred_rollout[key][:, step, ...]
            y_true = truth_rollout[key][:, step, ...]
            abs_err = np.abs(y_pred - y_true)
            abs_mean = float(abs_err.mean())
            rel_mean: str | float = ""
            if _has_relative_metrics(key):
                rel_mean = float(abs_mean / (np.mean(y_true) + 1e-12))
            per_step_rows.append(
                {
                    "step": int(step + 1),
                    "output_key": key,
                    "abs_err_mean": abs_mean,
                    "rel_err_mean": rel_mean,
                }
            )

    if collect is not None:
        collect["errors_full"] = errors_full
        collect["errors_k"] = errors_k
        collect["n_frames"] = int(n_time - 1)
        collect["per_step"] = per_step_rows

    if write_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "rollout_predictions.npz", **pred_rollout)
        np.savez_compressed(output_dir / "ground_truth_rollout.npz", **truth_rollout)
        _write_metric_file(output_dir / "model_errors_rollout_full.txt", errors_full)
        _write_metric_file(output_dir / "model_errors_rollout_k.txt", errors_k)

        with (output_dir / "rollout_errors_per_step.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["case_index", "step", "output_key", "abs_err_mean", "rel_err_mean"],
            )
            writer.writeheader()
            writer.writerows(per_step_rows)

    for key, stats in errors_full.items():
        print(f"rollout_full::{key}: {stats}")
    for key, stats in errors_k.items():
        print(f"rollout_k::{key}: {stats}")

    return InferenceMetrics(errors=errors_full, val_rows=[])


def run_lazy_autoregressive_rollout_and_metrics(
    model: torch.nn.Module,
    frame_reader,
    case_indices: Sequence[int],
    case_lengths: Mapping[int, int],
    run_cfg: RunConfig,
    output_dir: Path,
) -> InferenceMetrics:
    """Per-case streamed twin of ``run_autoregressive_rollout_and_metrics``.

    The eager rollout needs the whole split's trajectories in RAM, which the
    lazy dataset path never materializes (``var_train``/``var_val`` are None —
    the AiiDA eval lane crashed on exactly that). This twin materializes ONE
    case's trajectory at a time via ``frame_reader(case_idx, time_idx)`` and
    frame-weight-combines the per-case stats: ``*_mean``/``iou_mean`` combine
    exactly (equal voxel counts per frame), ``rmse``-like stats via the sqrt of
    the weighted mean of squares; ``*_std`` stats do not combine across cases
    and live only in ``rollout_per_case.csv`` (full per-case stat rows for both
    horizons). The first case additionally writes the standard eager artifacts
    (predictions/GT npz, per-step csv) for debugging; the aggregate
    ``model_errors_rollout_full.txt``/``_k.txt`` overwrite the first case's
    after the loop.
    """
    context_keys = list(run_cfg.autoregressive_context_keys)
    state_keys = list(run_cfg.autoregressive_state_keys)
    needed = list(dict.fromkeys(context_keys + state_keys + ["fl"]))

    def _combine_rule(stat_name: str) -> str:
        if stat_name.endswith("_std"):
            return "skip"
        if stat_name == "rmse" or stat_name.endswith("_rmse"):
            return "rms"
        return "mean"

    # agg[scope][key][stat] = [weighted_sum, weight]
    agg: dict[str, dict[str, dict[str, list[float]]]] = {"full": {}, "k": {}}
    per_case_rows: list[dict[str, Any]] = []
    per_step_rows_all: list[dict[str, Any]] = []
    n_done = 0
    for pos, case_idx in enumerate(int(c) for c in case_indices):
        n_time = int(case_lengths[case_idx])
        if n_time < 2:
            print(f"[lazy rollout] skipping case {case_idx}: only {n_time} frame(s)")
            continue
        frames = [frame_reader(case_idx=case_idx, time_idx=t) for t in range(n_time)]
        trajectory = {
            key: np.stack([np.asarray(f[key])[0] for f in frames], axis=0)[None]
            for key in needed
        }
        del frames
        collected: dict = {}
        run_autoregressive_rollout_and_metrics(
            model=model,
            trajectory_var=trajectory,
            run_cfg=run_cfg,
            output_dir=output_dir,
            write_files=(n_done == 0),
            collect=collected,
        )
        for row in collected.get("per_step") or []:
            per_step_rows_all.append({"case_index": case_idx, **row})
        weight = float(collected["n_frames"])
        for scope, errs in (("full", collected["errors_full"]), ("k", collected["errors_k"])):
            for key, stats in errs.items():
                row: dict[str, Any] = {
                    "case_index": case_idx, "scope": scope,
                    "output_key": key, "n_frames": int(weight),
                }
                for stat, value in stats.items():
                    row[stat] = float(value)
                    rule = _combine_rule(stat)
                    if rule == "skip":
                        continue
                    slot = agg[scope].setdefault(key, {}).setdefault(stat, [0.0, 0.0])
                    slot[0] += weight * (float(value) ** 2 if rule == "rms" else float(value))
                    slot[1] += weight
                per_case_rows.append(row)
        n_done += 1
    if n_done == 0:
        raise ValueError("Lazy rollout eval found no case with >= 2 frames.")

    def _finalize(scope: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, stats in agg[scope].items():
            out[key] = {}
            for stat, (wsum, w) in stats.items():
                value = wsum / w
                out[key][stat] = float(np.sqrt(value)) if _combine_rule(stat) == "rms" else float(value)
        return out

    errors_full = _finalize("full")
    errors_k = _finalize("k")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metric_file(output_dir / "model_errors_rollout_full.txt", errors_full)
    _write_metric_file(output_dir / "model_errors_rollout_k.txt", errors_k)
    fieldnames = ["case_index", "scope", "output_key", "n_frames"]
    fieldnames += sorted({k for row in per_case_rows for k in row} - set(fieldnames))
    with (output_dir / "rollout_per_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(per_case_rows)

    # Overwrite the first case's single-case per-step csv with every case's rows.
    # Reading one case as the split's behaviour is actively misleading: on ds 55017
    # the first case scored 53.2 K against a 9.1 K per-case median.
    if per_step_rows_all:
        with (output_dir / "rollout_errors_per_step.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["case_index", "step", "output_key", "abs_err_mean", "rel_err_mean"],
                restval="",
            )
            writer.writeheader()
            writer.writerows(per_step_rows_all)
    print(f"[lazy rollout] {n_done} cases aggregated (frame-weighted)")
    for key, stats in errors_full.items():
        print(f"lazy_rollout_full::{key}: {stats}")
    return InferenceMetrics(errors=errors_full, val_rows=[])


def run_inference_and_metrics(
    model: torch.nn.Module,
    ds_input: dict[str, np.ndarray],
    ds_output: dict[str, np.ndarray],
    run_cfg: RunConfig,
    output_dir: Path,
    relative_reference: Optional[dict[str, np.ndarray]] = None,
    fold: Optional[int] = None,
    val_idx: Optional[np.ndarray] = None,
    p_values: Optional[np.ndarray] = None,
    v_values: Optional[np.ndarray] = None,
) -> InferenceMetrics:
    """Run inference, compute aggregate metrics, and optionally record per-sample rows."""

    infer = FNOGridInferencer(
        model=model,
        input_keys=run_cfg.keys.str_fno_input_keys,
        output_names=run_cfg.keys.str_fno_output_keys,
        results_dir=output_dir / "vti",
        device=None,  # CUDA when present, else the CPU (PINO_ALLOW_CPU=1 evaluations)
        requires_grad=False,
        mask_fn=None,
        log_iter=False,
        bounds=run_cfg.bounds,
    )
    preds = infer.run(
        ds_input,
        name="preds",
        step=0,
        batch_size=run_cfg.inference_batch_size,
    )
    preds, ds_output = _align_preds_to_truth_shapes(preds, ds_output, run_cfg.keys.str_fno_output_keys)

    preds["fl"] = _derive_fl(preds, run_cfg)
    # Volume fractions are physically confined to [0, 1], but nothing upstream
    # enforces it: the final projection of every family is an unbounded linear
    # layer. That matters here specifically because the loop below masks each key
    # with the PREDICTED alpha -- alpha included, i.e. alpha masked by itself. An
    # out-of-range alpha saturates its own k=20 sigmoid to "keep", survives
    # unchanged, and then dominates the global metric. Measured 2026-07-30 on
    # gaot_mat_steady_20k: eval rmse_alpha 1.425 on a [0,1] quantity and rmse_T
    # 6278 K, against 0.016 and 47.7 K from that same model's own validation.
    # Clamping before the mask fixes both the mask and the metric, for every family.
    for _frac_key in ("alpha", "fl", "VOF"):
        if _frac_key in preds:
            preds[_frac_key] = np.clip(preds[_frac_key], 0.0, 1.0)
    if run_cfg.alpha_masking:
        for key in ds_output.keys():
            if key == "T":
                _print_alpha_masking_notice_once(key)
                preds[key] = fill_where_mask_smooth_np(
                    preds[key],
                    preds["alpha"],
                    threshold=0.5,
                    d=t_fill_field(run_cfg, ds_input),
                    k=20,
                    inplace=False,
                )
            elif key not in ("x", "y", "z", "P", "V_scan", "Deposition"):
                preds[key] = fill_where_mask_smooth_np(
                    preds[key], preds["alpha"], threshold=0.5, d=0.0, k=20, inplace=False
                )

        # Mirror node-graph velocity_fl_mask for standalone inferencer evaluation.
        for vel_key in ("u", "v", "w"):
            if vel_key in preds:
                preds[vel_key] = fill_where_mask_smooth_np(
                    preds[vel_key], preds["fl"], threshold=0.5, d=0.0, k=20, inplace=False
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "predictions.npz", **preds)
    np.savez_compressed(output_dir / "ground_truth.npz", **ds_output)

    errors: dict[str, dict[str, float]] = {}
    val_rows: list[dict[str, Any]] = []
    for key in ds_output.keys():
        y_pred = np.asarray(preds[key])
        y_true = np.asarray(ds_output[key])
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch for {key}: pred {y_pred.shape}, true {y_true.shape}")

        err = y_pred - y_true
        abs_err = np.abs(err)

        rmse = np.sqrt(np.mean(err ** 2))
        relative = _has_relative_metrics(key)

        stats: dict[str, float] = {
            "abs_err_mean": float(abs_err.mean()),
            "abs_err_std": float(abs_err.std()),
            "rmse": float(rmse),
        }
        if relative:
            denom_source = y_true
            if relative_reference is not None and key in relative_reference:
                # K-fold aggregation can normalize relative errors against full-data stats.
                denom_source = np.asarray(relative_reference[key])
            denom = np.mean(denom_source) + 1e-12
            rel_err = abs_err / denom
            stats["rel_err_mean"] = float(rel_err.mean())
            stats["rel_err_std"] = float(rel_err.std())
            stats["rel_rmse"] = float(rmse / denom)
        iou_vals = None
        if key in _UNITLESS_KEYS:
            iou_vals = iou_per_batch(y_pred, y_true, threshold=0.5)
            stats.update(_iou_stats(y_pred, y_true))

        if "z" in ds_input:
            stats.update(
                region_error_stats(y_pred, y_true, np.asarray(ds_input["z"]), relative=relative)
            )
            # Beam box: the region that actually matters. A substrate-wide mean
            # is diluted by quiescent metal and cannot see a melt-pool defect.
            # In a beam-following frame the beam sits at the origin, so the box is
            # constructible without the trajectory channel -- which is what every
            # steady arm lacks. Off in the lab frame, where the beam moves.
            box = beam_box_from_input(
                ds_input, np.asarray(ds_input["z"]),
                beam_at_origin=bool(getattr(run_cfg, "moving_ref_frame", False)),
                beam_radius=_beam_radius_in_grid_units(run_cfg),
            )
            if box is not None:
                stats.update(
                    beam_box_error_stats(
                        y_pred, y_true, box=box, relative=relative,
                        iou=key in _UNITLESS_KEYS,
                    )
                )
            # Peak and headroom (`pred_T_max_K`, `true_T_max_K`, `thermal_headroom_K`,
            # tail deficits), as the lazy streaming path reports them. `thermal_headroom_stats`
            # was written for this path and never called from it, so an eager evaluation
            # lacked the keys the paper's peak ratio is read from (lpno_repro.metrics).
            # Lowest per-case threshold for a mixed-material batch, as the lazy path does.
            if key == "T":
                threshold_K = float(np.min(np.asarray(metric_melt_threshold_K(run_cfg, ds_input))))
                if np.isfinite(threshold_K) and threshold_K > 0:
                    stats.update(
                        thermal_headroom_stats(
                            y_pred, preds.get("alpha"), np.asarray(ds_input["z"]),
                            T_s=threshold_K, T_l=threshold_K,
                            ref_T=float(run_cfg.ref_scales.get("T", 1.0)),
                            true_T=y_true, true_alpha=ds_output.get("alpha"),
                        )
                    )

        errors[key] = stats
        print(f"{key}: {stats}")

        if fold is None or val_idx is None or p_values is None or v_values is None:
            continue

        axis = tuple(range(1, y_true.ndim))
        abs_err_mean_per_sample = abs_err.mean(axis=axis)
        rel_err_mean_per_sample: list[str | float] = [""] * len(abs_err_mean_per_sample)
        if _has_relative_metrics(key):
            y_true_mean_per_sample = y_true.mean(axis=axis)
            rel_err_mean_per_sample = [
                float(value) for value in abs_err_mean_per_sample / (y_true_mean_per_sample + 1e-12)
            ]
        for i, (abs_mean, rel_mean) in enumerate(zip(abs_err_mean_per_sample, rel_err_mean_per_sample)):
            iou_value: str | float = ""
            if iou_vals is not None:
                iou_value = float(iou_vals[i])
            val_rows.append(
                {
                    "fold": int(fold),
                    "global_index": int(val_idx[i]),
                    "local_val_index": int(i),
                    "P": float(p_values[i]),
                    "V_scan": float(v_values[i]),
                    "output_key": key,
                    "abs_err_mean": float(abs_mean),
                    "rel_err_mean": rel_mean,
                    "iou": iou_value,
                }
            )

    if "z" in ds_input:
        geometry = melt_geometry_stats(
            _with_fl(preds, run_cfg),
            _with_fl(ds_output, run_cfg),
            np.asarray(ds_input["z"]),
            run_cfg.ref_scales.get("L", 1.0),
            x_field=np.asarray(ds_input["x"]) if "x" in ds_input else None,
            y_field=np.asarray(ds_input["y"]) if "y" in ds_input else None,
        )
        if geometry:
            errors["geometry"] = geometry
            print(f"geometry: {geometry}")

    _write_metric_file(output_dir / "model_errors.txt", errors,
                       network_dir=getattr(run_cfg, "network_base_dir", None), run_cfg=run_cfg)
    return InferenceMetrics(errors=errors, val_rows=val_rows)


def run_pointwise_inference_and_metrics(
    model: torch.nn.Module,
    ds_input: dict[str, np.ndarray],
    ds_output: dict[str, np.ndarray],
    run_cfg: RunConfig,
    output_dir: Path,
    fold: Optional[int] = None,
    val_idx: Optional[np.ndarray] = None,
    p_values: Optional[np.ndarray] = None,
    v_values: Optional[np.ndarray] = None,
) -> InferenceMetrics:
    """Run generic batched inference/metrics for pointwise baselines."""

    preds = _forward_model_chunked(
        model=model,
        invar=ds_input,
        output_names=run_cfg.keys.str_fno_output_keys,
        batch_size=run_cfg.inference_batch_size,
    )
    layout_keys = list(dict.fromkeys(list(run_cfg.keys.str_fno_output_keys) + list(ds_output.keys())))
    preds = _restore_pointwise_grid_layout(preds, run_cfg, keys=layout_keys)
    truth = _restore_pointwise_grid_layout(ds_output, run_cfg, keys=layout_keys)
    preds = _apply_prediction_postprocessing(preds, run_cfg, target_keys=run_cfg.keys.str_fno_output_keys)

    if "T" in preds and "fl" not in preds:
        preds["fl"] = _derive_fl(preds, run_cfg)
    if "T" in truth and "fl" not in truth:
        # The ground-truth T carries the same boiling fill, so its derived fl
        # needs the same mask -- otherwise both sides read fl=1 in the gas and
        # the melt IoU is measured against a layer neither model nor data melts.
        truth["fl"] = _derive_fl(truth, run_cfg)

    metric_keys = list(run_cfg.keys.str_fno_output_keys)
    if "fl" in preds and "fl" in truth:
        metric_keys.append("fl")
    errors = _compute_error_stats(preds, truth, metric_keys=metric_keys)

    if "z" in ds_input:
        z_grid = _restore_pointwise_grid_layout(
            {"z": np.asarray(ds_input["z"])}, run_cfg, keys=["z"]
        )["z"]
        for key in metric_keys:
            errors[key].update(
                region_error_stats(
                    np.asarray(preds[key]),
                    np.asarray(truth[key]),
                    z_grid,
                    relative=_has_relative_metrics(key),
                )
            )
            box = beam_box_from_input(
                ds_input, z_grid,
                beam_at_origin=bool(getattr(run_cfg, "moving_ref_frame", False)),
                beam_radius=_beam_radius_in_grid_units(run_cfg),
            )
            if box is not None:
                errors[key].update(
                    beam_box_error_stats(
                        np.asarray(preds[key]),
                        np.asarray(truth[key]),
                        box=box,
                        relative=_has_relative_metrics(key),
                        iou=key in _UNITLESS_KEYS,
                    )
                )
        _xy = _restore_pointwise_grid_layout(
            {k: np.asarray(ds_input[k]) for k in ("x", "y") if k in ds_input}, run_cfg,
            keys=[k for k in ("x", "y") if k in ds_input],
        ) if ("x" in ds_input and "y" in ds_input) else {}
        geometry = melt_geometry_stats(
            preds, truth, z_grid, run_cfg.ref_scales.get("L", 1.0),
            x_field=_xy.get("x"), y_field=_xy.get("y"),
        )
        if geometry:
            errors["geometry"] = geometry

    val_rows: list[dict[str, Any]] = []
    for key in metric_keys:
        y_pred = np.asarray(preds[key])
        y_true = np.asarray(truth[key])
        abs_err = np.abs(y_pred - y_true)
        iou_vals = None
        if key in ("fl", "alpha") and y_pred.ndim >= 4:
            iou_vals = iou_per_batch(y_pred, y_true, threshold=0.5)

        if fold is None or val_idx is None or p_values is None or v_values is None:
            continue

        axis = tuple(range(1, y_true.ndim))
        abs_err_mean_per_sample = abs_err.mean(axis=axis)
        rel_err_mean_per_sample: list[str | float] = [""] * len(abs_err_mean_per_sample)
        if _has_relative_metrics(key):
            y_true_mean_per_sample = y_true.mean(axis=axis)
            rel_err_mean_per_sample = [
                float(value) for value in abs_err_mean_per_sample / (y_true_mean_per_sample + 1e-12)
            ]
        for i, (abs_mean, rel_mean) in enumerate(zip(abs_err_mean_per_sample, rel_err_mean_per_sample)):
            iou_value: str | float = ""
            if iou_vals is not None:
                iou_value = float(iou_vals[i])
            val_rows.append(
                {
                    "fold": int(fold),
                    "global_index": int(val_idx[i]),
                    "local_val_index": int(i),
                    "P": float(p_values[i]),
                    "V_scan": float(v_values[i]),
                    "output_key": key,
                    "abs_err_mean": float(abs_mean),
                    "rel_err_mean": rel_mean,
                    "iou": iou_value,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "predictions.npz", **preds)
    np.savez_compressed(output_dir / "ground_truth.npz", **truth)
    _write_metric_file(output_dir / "model_errors.txt", errors,
                       network_dir=getattr(run_cfg, "network_base_dir", None), run_cfg=run_cfg)

    for key, stats in errors.items():
        print(f"pointwise::{key}: {stats}")
    return InferenceMetrics(errors=errors, val_rows=val_rows)


def run_masked_patch_metrics(
    model: torch.nn.Module,
    dataset,
    run_cfg: RunConfig,
    output_dir: Path,
) -> InferenceMetrics:
    """Evaluate masked autoregressive patch predictions without materializing full trajectories."""
    metric_keys = list(run_cfg.autoregressive_state_keys) + ["fl"]
    accumulators = {
        key: {
            "count": 0,
            "abs_err_sum": 0.0,
            "sq_err_sum": 0.0,
            "truth_sum": 0.0,
            "iou_sum": 0.0,
            "iou_sq_sum": 0.0,
            "iou_count": 0,
        }
        for key in metric_keys
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=run_cfg.inference_batch_size, shuffle=False, drop_last=False)
    preview_written = False
    for batch_idx, (invar, true_outvar, _) in enumerate(loader):
        invar_t = {key: value.to(device=device, dtype=torch.float32) for key, value in invar.items()}
        with torch.inference_mode():
            pred_t = model(invar_t)
        preds = {key: value.detach().cpu().numpy() for key, value in pred_t.items()}
        truth = {key: value.detach().cpu().numpy() for key, value in true_outvar.items()}
        preds = _apply_prediction_postprocessing(preds, run_cfg, target_keys=run_cfg.autoregressive_state_keys)
        if "fl" in truth and "fl" not in preds and "T" in preds:
            preds["fl"] = _derive_fl(preds, run_cfg)

        if not preview_written:
            np.savez_compressed(
                output_dir / "temporal_preview_batch0.npz",
                # `invar` is torch on the plain path and numpy after a frame transfer,
                # so this cannot assume either. Every other reader here already guards.
                **{f"in_{key}": (value.detach().cpu().numpy() if hasattr(value, "detach")
                                 else np.asarray(value))
                   for key, value in invar.items()},
                **{f"pred_{key}": value for key, value in preds.items()},
                **{f"true_{key}": value for key, value in truth.items()},
            )
            preview_written = True


        for key in metric_keys:
            if key not in preds or key not in truth:
                continue
            y_pred = np.asarray(preds[key])
            y_true = np.asarray(truth[key])
            err = y_pred - y_true
            abs_err = np.abs(err)
            accumulators[key]["count"] += int(y_true.size)
            accumulators[key]["abs_err_sum"] += float(abs_err.sum())
            accumulators[key]["sq_err_sum"] += float(np.square(err).sum())
            accumulators[key]["truth_sum"] += float(y_true.sum())
            if key in ("fl", "alpha"):
                _accumulate_iou(accumulators[key], y_pred, y_true)

    errors: dict[str, dict[str, float]] = {}
    for key, stats in accumulators.items():
        if stats["count"] <= 0:
            continue
        denom = (stats["truth_sum"] / stats["count"]) + 1e-12
        rmse = np.sqrt(stats["sq_err_sum"] / stats["count"])
        abs_err_mean = stats["abs_err_sum"] / stats["count"]
        # abs_err**2 == err**2, so sq_err_sum is already the sum of squared
        # absolute errors and the population std needs no extra accumulator:
        # Var = E[x^2] - E[x]^2. The materialized path reports abs_err.std()
        # over every voxel, which is exactly this. Clamped at 0 because the
        # subtraction can go slightly negative on nearly-constant error fields.
        abs_err_std = float(np.sqrt(max(0.0, stats["sq_err_sum"] / stats["count"] - abs_err_mean ** 2)))
        payload = {
            "abs_err_mean": float(abs_err_mean),
            "abs_err_std": abs_err_std,
            "rmse": float(rmse),
        }
        if _has_relative_metrics(key):
            # denom is a scalar, so rel_err = abs_err / denom and the std scales with it.
            payload["rel_err_mean"] = float(abs_err_mean / denom)
            payload["rel_err_std"] = float(abs_err_std / denom)
            payload["rel_rmse"] = float(rmse / denom)
        if stats["iou_count"] > 0:
            iou_mean = stats["iou_sum"] / stats["iou_count"]
            payload["iou_mean"] = float(iou_mean)
            payload["iou_std"] = float(
                np.sqrt(max(0.0, stats["iou_sq_sum"] / stats["iou_count"] - iou_mean ** 2))
            )
            payload["iou_support_frames"] = int(stats["iou_count"])
        if stats.get("iou_all_count"):
            # Legacy all-frames value, so older numbers stay comparable.
            payload["iou_mean_all_frames"] = float(stats["iou_all_sum"] / stats["iou_all_count"])
            payload["iou_empty_gt_frames"] = int(stats.get("iou_empty_gt_frames", 0))
            payload["iou_empty_gt_false_positive_frames"] = int(
                stats.get("iou_empty_gt_false_positive_frames", 0)
            )
        errors[key] = payload

    _write_metric_file(output_dir / "model_errors.txt", errors,
                       network_dir=getattr(run_cfg, "network_base_dir", None), run_cfg=run_cfg)
    for key, stats in errors.items():
        print(f"masked_patch::{key}: {stats}")
    return InferenceMetrics(errors=errors, val_rows=[])


def _substrate_metal_peak(pred_T, pred_alpha, substrate, run_cfg) -> float:
    """Peak predicted T (K) over substrate METAL cells, for the streaming path.

    Metal only: `_apply_prediction_postprocessing` fills `T = T_boil` wherever
    `alpha < 0.5`, so a whole-domain peak reports the fill, not a prediction.
    """

    T = np.asarray(pred_T, dtype=np.float64)[..., substrate]
    if pred_alpha is not None:
        a = np.asarray(pred_alpha, dtype=np.float64)[..., substrate]
        if a.shape == T.shape:
            T = T[a > 0.5]
    T = np.asarray(T).ravel()
    if T.size == 0:
        return float("-inf")
    return float(T.max()) * float(run_cfg.ref_scales.get("T", 1.0))


def run_lazy_dataset_metrics(
    model: torch.nn.Module,
    dataset,
    run_cfg: RunConfig,
    output_dir: Path,
    metric_keys: Sequence[str],
) -> InferenceMetrics:
    """Evaluate a lazy dataset object directly without materializing full tensors."""
    accumulators = {
        key: {
            "count": 0,
            "abs_err_sum": 0.0,
            "sq_err_sum": 0.0,
            "truth_sum": 0.0,
            "iou_sum": 0.0,
            "iou_sq_sum": 0.0,
            "iou_count": 0,
        }
        for key in metric_keys
    }
    # Streaming substrate/surface accumulators (raw sums, finalized after the
    # loop) so the memory-safe lazy eval reports the same z-region breakdown as
    # region_error_stats without holding whole fields. Populated only when the
    # batch carries a z coordinate; grids with no z<=0 slice leave it empty.
    region_acc = {
        key: {
            tag: {"abs": 0.0, "sq": 0.0, "n": 0, "truth": 0.0, "inter": 0, "union": 0}
            for tag in ("substrate", "surface", "beambox")
        }
        for key in metric_keys
    }
    # Peak substrate-METAL temperature, streamed as a running max. Melt is ~1% of
    # the substrate, so a clipped temperature tail costs almost nothing in L2 and
    # sends every thresholded quantity to exactly zero (model 29125: peak 1631 K
    # against an 1898 K threshold -> melt volume 0, T RMSE still 84 K).
    head_acc = {"max": float("-inf"), "true_max": float("-inf"), "frames": 0,
                "margin": float("-inf"), "thr_min": float("inf"), "thr_max": float("-inf")}
    # Melt-pool/keyhole geometry, sample-weighted across batches (each batch
    # contributes its own mean, so the weight is its sample count).
    geom_acc: dict[str, list[float]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PINO_EVAL_FRAME_STRIDE=N scores every Nth frame. An eval deliberately ignores
    # custom.val_max_cases (see _effective_val_max_cases) so a scored split is never a
    # subset -- so this is the only way to get a fast read on a peak statistic like
    # tail_deficit_K, which is a percentile difference and converges long before the
    # full 5-11k frames. Unset (=1) keeps the full split. Every subsampled result is
    # stamped below and printed loudly, because a subset score must never be mistaken
    # for a split score.
    frame_stride = max(1, int(os.environ.get("PINO_EVAL_FRAME_STRIDE", "1")))
    if frame_stride > 1:
        from torch.utils.data import Subset

        n_all = len(dataset)
        dataset = Subset(dataset, range(0, n_all, frame_stride))
        print(f"[eval] SUBSAMPLED: every {frame_stride}th frame, {len(dataset)}/{n_all} "
              f"frames -- NOT a full-split score", flush=True)

    transfer_mode = getattr(run_cfg, "eval_frame_transfer", "") or ""
    steady_case_ids: list[int] = []
    steady_state: dict = {"case": None, "preds": None, "truth": None, "invar": None, "n": 0,
                          "scored": 0}
    if transfer_mode == "steady":
        if frame_stride > 1:
            raise ValueError(
                "eval_frame_transfer='steady' averages CONSECUTIVE frames per case; "
                "PINO_EVAL_FRAME_STRIDE would subsample the window. Unset the stride."
            )
        entries = getattr(dataset, "entries", None)
        if entries is None:
            raise ValueError(
                "eval_frame_transfer='steady' needs the dataset's (case, frame) entries "
                "to window per case; this dataset does not expose them."
            )
        from torch.utils.data import Subset

        avg_w = int(getattr(run_cfg, "eval_frame_avg", 30))
        # Where each case's tail ends: the last frame on the declared path for a laser
        # that overruns it, so the scored frames are the ones the steady target was
        # built from and the window is not pinned against the +x wall. Stores without
        # the accessor (arraydir runtimes) keep the export tail.
        stop_fn = getattr(getattr(dataset, "store", None), "laser_on_frame_stop", None)
        stop_by_case = None
        if stop_fn is not None:
            stop_by_case = {int(c): int(stop_fn(int(c))) for c in sorted({int(c) for c, _ in entries})}
        sel, steady_case_ids = steady_entry_selection(entries, avg_window=avg_w, stop_by_case=stop_by_case)
        dataset = Subset(dataset, sel)
        print(f"[eval] steady transfer: last {avg_w} frames of "
              f"{len(sel) // avg_w} cases, averaged in the beam window", flush=True)
    loader = DataLoader(dataset, batch_size=run_cfg.inference_batch_size, shuffle=False, drop_last=False)
    # Deferred so the stored batch can be the most-molten one, not the first.
    preview_payload: dict | None = None
    preview_truth: dict | None = None
    preview_frozen = False
    # ponytail: opt-in full-series dump. The preview above keeps ONE batch, which
    # shows whether a model melts but cannot show HOW it evolves -- rendering a case
    # as video needs every frame. Off by default because it is O(frames) on disk.
    dump_stride = frame_dump_stride()
    dump_frames = dump_stride > 0
    dump_acc: dict[str, list] = {}
    dump_seen = 0
    # ponytail: three counters, logged once. Evals here run 5-25 h against ~1.65 s/case
    # of compute and nobody could say which phase owned it; field pruning, cache size and
    # surface distances each measured innocent (the last is 13.6 min of a 25 h run), so
    # the gap is still unattributed. Measure it instead of guessing a fourth time.
    phase_s = {"load": 0.0, "fwd": 0.0, "metrics": 0.0}
    batch_idx = -1  # an empty loader must not NameError in the summary below
    mark = time.perf_counter()
    for batch_idx, (invar, true_outvar, _) in enumerate(loader):
        phase_s["load"] += time.perf_counter() - mark
        mark = time.perf_counter()
        invar_t = {key: value.to(device=device, dtype=torch.float32) for key, value in invar.items()}
        with torch.inference_mode():
            pred_t = model(invar_t)
        if device.type == "cuda":
            # Without this the async launch returns instantly, the forward looks free,
            # and its cost lands on whichever phase next touches the tensor.
            torch.cuda.synchronize(device)
        phase_s["fwd"] += time.perf_counter() - mark
        mark = time.perf_counter()
        preds = {key: value.detach().cpu().numpy() for key, value in pred_t.items()}
        truth = {key: value.detach().cpu().numpy() for key, value in true_outvar.items()}
        preds, truth = _align_preds_to_truth_shapes(preds, truth, metric_keys)
        preds = _apply_prediction_postprocessing(
            preds, run_cfg, target_keys=metric_keys,
            invar={k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
                   for k, v in invar.items()} if isinstance(invar, dict) else None,
        )
        if "fl" in truth and "fl" not in preds and "T" in preds:
            preds["fl"] = _derive_fl(preds, run_cfg)

        # Cross-frame scoring. Crop predictions, truth AND inputs into the beam-following
        # window before anything is measured, so the beam box, the z masks and the melt
        # geometry all report in the target frame without a second implementation of any
        # of them. One eval is one (model, target frame) pair; the eval node records which.
        _nx_batch = int(next(iter(preds.values())).shape[-3])
        _win = int(getattr(run_cfg, "eval_frame_behind", 0)) + int(getattr(run_cfg, "eval_frame_ahead", 0))
        if transfer_mode in ("moving_ref", "steady") and _nx_batch == _win:
            # Already windowed by the loader (a moving-ref-native eval): the crop would
            # degenerate to slice(0, nx) anyway, but its beam-relative x rewrite would
            # double-shift. Skip; 'steady' still averages below.
            invar = {
                k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                for k, v in invar.items()
            }
        elif transfer_mode in ("moving_ref", "steady"):
            invar_np = {
                k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                for k, v in invar.items()
            }
            merged, invar_np = crop_batch_to_moving_ref(
                {**{f"__p__{k}": v for k, v in preds.items()},
                 **{f"__t__{k}": v for k, v in truth.items()}},
                invar_np,
                behind=int(run_cfg.eval_frame_behind),
                ahead=int(run_cfg.eval_frame_ahead),
            )
            preds = {k[len("__p__"):]: v for k, v in merged.items() if k.startswith("__p__")}
            truth = {k[len("__t__"):]: v for k, v in merged.items() if k.startswith("__t__")}
            print(f"[eval] frame transfer: cropped to the beam-following window "
                  f"{run_cfg.eval_frame_behind}+{run_cfg.eval_frame_ahead} cells "
                  f"-> x {invar_np['x'].shape[-3]}", flush=True)
            invar = invar_np

        def score_batch(preds, truth, invar):
            """Everything measured for one batch: preview, regions, beam box, geometry."""
            nonlocal preview_payload, preview_truth, dump_seen

            # Keep the most-molten batch rather than the first one: the loader is
            # case-ordered with shuffle=False, so batch 0 is frame 0 -- beam just on,
            # `fl` identically zero, `T` flat ambient. The preview is the only
            # per-voxel artifact an eval leaves, so it has to show the melt.
            if not preview_frozen and not preferred_preview_batch(preview_truth, truth):
                preview_payload = {
                    # `invar` is torch on the plain path and numpy after a frame transfer,
                    # so this cannot assume either. Every other reader here already guards.
                    **{f"in_{key}": (value.detach().cpu().numpy() if hasattr(value, "detach")
                                     else np.asarray(value))
                       for key, value in invar.items()},
                    **{f"pred_{key}": value for key, value in preds.items()},
                    **{f"true_{key}": value for key, value in truth.items()},
                }
                preview_truth = truth

            if dump_frames:
                accumulate_frame_dump(dump_acc, preds, truth, stride=dump_stride, index=dump_seen)
                dump_seen += 1

            # z-region masks for this batch (constant across batches, but cheap to
            # recompute and robust to an absent z channel).
            substrate = surf_idx = None
            if "z" in invar:
                z_np = invar["z"].detach().cpu().numpy() if hasattr(invar["z"], "detach") else np.asarray(invar["z"])
                substrate, surf_idx = z_region_indices(z_np)

            # Beam box for this batch: elementwise from `beam_dist`, so a per-frame
            # beam position is handled by construction.
            # Route through the SHARED builder. This site used to construct the mask
            # itself, gated on `"beam_dist" in invar` -- a fourth copy of the predicate,
            # and the reason every steady eval returned no beambox_* keys at all while
            # three rounds of fixes to beam_box_from_input went unreached. The two prior
            # beam-box regressions (2026-08-10, 2026-08-15) were also duplicate copies
            # drifting; there is now one.
            #
            # `beam_at_origin` follows moving_ref_frame: in a beam-following frame the
            # beam sits at the grid origin BY CONSTRUCTION, so the synthesis about that
            # origin is exact, not approximate. In the LAB frame the beam moves and the
            # builder declines rather than centring a box where no beam is -- an
            # ungated fallback would emit a plausible wrong number, which is precisely
            # how this metric was wrong before.
            box_mask = None
            if "z" in invar:
                np_invar = {}
                for _k in BEAM_BOX_INPUT_KEYS:
                    _v = invar.get(_k)
                    if _v is None:
                        continue
                    np_invar[_k] = _v.detach().cpu().numpy() if hasattr(_v, "detach") else np.asarray(_v)
                box_mask = beam_box_from_input(
                    np_invar, z_np, n_radii=2.0,
                    beam_at_origin=bool(getattr(run_cfg, "moving_ref_frame", False)),
                    beam_radius=_beam_radius_in_grid_units(run_cfg),
                    run_cfg_hint=run_cfg,
                )
            if "T" in preds and substrate is not None:
                _peak = _substrate_metal_peak(preds["T"], preds.get("alpha"), substrate, run_cfg)
                head_acc["max"] = max(head_acc["max"], _peak)
                # MARGIN, accumulated per batch against THIS batch's threshold. Comparing a
                # global peak to a global threshold is only right when every case shares a
                # material; with per-case mat_Ts/mat_Tl it would subtract one material's
                # threshold from another material's peak. On a single-material set the
                # threshold is constant and this is arithmetically identical to the old
                # `max(peak) - threshold`, so nothing previously measured moves.
                _thr = metric_melt_threshold_K(run_cfg, invar if isinstance(invar, dict) else None)
                _thr_arr = np.asarray(_thr, dtype=np.float64)
                if np.isfinite(_peak) and _thr_arr.size:
                    head_acc["margin"] = max(head_acc["margin"], _peak - float(_thr_arr.min()))
                    head_acc["thr_min"] = min(head_acc["thr_min"], float(_thr_arr.min()))
                    head_acc["thr_max"] = max(head_acc["thr_max"], float(_thr_arr.max()))
                head_acc["frames"] += int(np.asarray(preds["T"]).shape[0])
                if "T" in truth:
                    head_acc["true_max"] = max(
                        head_acc["true_max"],
                        _substrate_metal_peak(
                            truth["T"], truth.get("alpha", preds.get("alpha")), substrate, run_cfg
                        ),
                    )

            for key in metric_keys:
                if key not in preds or key not in truth:
                    continue
                y_pred = np.asarray(preds[key])
                y_true = np.asarray(truth[key])
                err = y_pred - y_true
                abs_err = np.abs(err)
                accumulators[key]["count"] += int(y_true.size)
                accumulators[key]["abs_err_sum"] += float(abs_err.sum())
                accumulators[key]["sq_err_sum"] += float(np.square(err).sum())
                accumulators[key]["truth_sum"] += float(y_true.sum())
                _track_prediction_extremes(
                    accumulators[key], y_pred, y_true, batch_idx,
                    aux={"alpha": preds.get("alpha"), "z": z_np if "z" in invar else None},
                )
                if key in ("fl", "alpha"):
                    _accumulate_iou(accumulators[key], y_pred, y_true)
                if substrate is not None and y_pred.shape[-1] == substrate.size:
                    for tag, sel in (("substrate", substrate), ("surface", surf_idx)):
                        e = err[..., sel]
                        t = y_true[..., sel]
                        ra = region_acc[key][tag]
                        ra["abs"] += float(np.abs(e).sum())
                        ra["sq"] += float(np.square(e).sum())
                        ra["n"] += int(e.size)
                        ra["truth"] += float(t.sum())
                    if box_mask is not None:
                        try:
                            bm = np.broadcast_to(box_mask, y_pred.shape)
                        except ValueError:
                            bm = None
                        if bm is not None and bm.any():
                            ra = region_acc[key]["beambox"]
                            eb = err[bm]
                            ra["abs"] += float(np.abs(eb).sum())
                            ra["sq"] += float(np.square(eb).sum())
                            ra["n"] += int(eb.size)
                            ra["truth"] += float(y_true[bm].sum())
                            if key in ("fl", "alpha"):
                                pb = y_pred[bm] > 0.5
                                tb = y_true[bm] > 0.5
                                ra["inter"] += int(np.logical_and(pb, tb).sum())
                                ra["union"] += int(np.logical_or(pb, tb).sum())

            if substrate is not None:
                weight = float(next(iter(truth.values())).shape[0])
                geometry = melt_geometry_stats(
                    _with_fl(preds, run_cfg),
                    _with_fl(truth, run_cfg),
                    z_np,
                    run_cfg.ref_scales.get("L", 1.0),
                    x_field=(invar["x"].detach().cpu().numpy() if hasattr(invar.get("x"), "detach")
                             else (np.asarray(invar["x"]) if "x" in invar else None)),
                    y_field=(invar["y"].detach().cpu().numpy() if hasattr(invar.get("y"), "detach")
                             else (np.asarray(invar["y"]) if "y" in invar else None)),
                )
                for name, value in geometry.items():
                    slot = geom_acc.setdefault(name, [0.0, 0.0])
                    slot[0] += value * weight
                    slot[1] += weight

        if transfer_mode == "steady":
            # Accumulate per-case running means; emit ONE averaged frame per case --
            # the same construction as the steady dataset, applied to predictions.
            def _steady_emit():
                n = float(steady_state["n"])
                score_batch({k: v / n for k, v in steady_state["preds"].items()},
                            {k: v / n for k, v in steady_state["truth"].items()},
                            {k: v / n for k, v in steady_state["invar"].items()})
                steady_state["scored"] += 1

            nb = next(iter(preds.values())).shape[0]
            invar_np_rows = {
                k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                for k, v in invar.items()
            }
            for row in range(nb):
                case = steady_case_ids[steady_state.get("cursor", 0)]
                steady_state["cursor"] = steady_state.get("cursor", 0) + 1
                if steady_state["case"] is not None and case != steady_state["case"]:
                    _steady_emit()
                    steady_state.update(case=None, preds=None, truth=None, invar=None, n=0)
                if steady_state["case"] is None:
                    steady_state.update(
                        case=case,
                        preds={k: np.array(v[row:row + 1], dtype=np.float64) for k, v in preds.items()},
                        truth={k: np.array(v[row:row + 1], dtype=np.float64) for k, v in truth.items()},
                        invar={k: np.array(v[row:row + 1], dtype=np.float64) for k, v in invar_np_rows.items()},
                        n=1,
                    )
                else:
                    for k, v in preds.items():
                        steady_state["preds"][k] += v[row:row + 1]
                    for k, v in truth.items():
                        steady_state["truth"][k] += v[row:row + 1]
                    for k, v in invar_np_rows.items():
                        steady_state["invar"][k] += v[row:row + 1]
                    steady_state["n"] += 1
        else:
            if transfer_mode == "fixed_window":
                # A steady/moving-ref NATIVE grid: beam pinned by construction, so the
                # window is one fixed x-slice shared by predictions, truth and inputs.
                nx_native = next(iter(preds.values())).shape[-3]
                sl_fixed = fixed_window_slice(
                    nx=nx_native,
                    behind=int(run_cfg.eval_frame_behind),
                    ahead=int(run_cfg.eval_frame_ahead),
                )
                def _crop_fixed(d):
                    return {
                        k: (np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v)
                            [..., sl_fixed, :, :]
                            if np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v).ndim >= 3
                            and np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v).shape[-3] == nx_native
                            else np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v))
                        for k, v in d.items()
                    }
                preds, truth, invar = _crop_fixed(preds), _crop_fixed(truth), _crop_fixed(invar)
            score_batch(preds, truth, invar)

        phase_s["metrics"] += time.perf_counter() - mark
        mark = time.perf_counter()

    if transfer_mode == "steady" and steady_state["case"] is not None:
        n = float(steady_state["n"])
        score_batch({k: v / n for k, v in steady_state["preds"].items()},
                    {k: v / n for k, v in steady_state["truth"].items()},
                    {k: v / n for k, v in steady_state["invar"].items()})
        steady_state["scored"] += 1

    # Name kept for every consumer that globs it (notebooks, aiida eval_artifacts).
    if preview_payload is not None:
        np.savez_compressed(output_dir / "lazy_preview_batch0.npz", **preview_payload)
        print(f"[eval] preview: kept the batch with melt fraction "
              f"{_preview_score(preview_truth):.4g}", flush=True)

    if dump_frames:
        dumped = write_frame_dump(dump_acc, output_dir)
        if dumped is None:
            print("[eval] PINO_DUMP_FRAMES was set but no frames were scored", flush=True)
        else:
            n = sum(len(v) for v in dump_acc.values()) // max(len(dump_acc), 1)
            mb = dumped.stat().st_size / 1e6
            print(f"[eval] frame dump: kept {n} of {dump_seen} batches "
                  f"(stride {dump_stride}) -> {dumped.name}, {mb:.0f} MB", flush=True)

    total_s = sum(phase_s.values()) or 1.0
    print(
        "eval phase timing over {} batches: loader {:.1f}s ({:.0f}%), "
        "forward {:.1f}s ({:.0f}%), metrics {:.1f}s ({:.0f}%)".format(
            batch_idx + 1,
            phase_s["load"], 100 * phase_s["load"] / total_s,
            phase_s["fwd"], 100 * phase_s["fwd"] / total_s,
            phase_s["metrics"], 100 * phase_s["metrics"] / total_s,
        ),
        flush=True,
    )

    errors: dict[str, dict[str, float]] = {}
    # Fallback for an eval that produced no batch (the accumulators stay empty); the live
    # value comes from `head_acc`, which is per batch and therefore per material.
    threshold_K = float(np.asarray(metric_melt_threshold_K(run_cfg)).min())
    if np.isfinite(head_acc["thr_min"]):
        threshold_K = head_acc["thr_min"]
    for key, stats in accumulators.items():
        if stats["count"] <= 0:
            continue
        denom = (stats["truth_sum"] / stats["count"]) + 1e-12
        rmse = np.sqrt(stats["sq_err_sum"] / stats["count"])
        abs_err_mean = stats["abs_err_sum"] / stats["count"]
        # abs_err**2 == err**2, so sq_err_sum is already the sum of squared
        # absolute errors and the population std needs no extra accumulator:
        # Var = E[x^2] - E[x]^2. The materialized path reports abs_err.std()
        # over every voxel, which is exactly this. Clamped at 0 because the
        # subtraction can go slightly negative on nearly-constant error fields.
        abs_err_std = float(np.sqrt(max(0.0, stats["sq_err_sum"] / stats["count"] - abs_err_mean ** 2)))
        payload = {
            "abs_err_mean": float(abs_err_mean),
            "abs_err_std": abs_err_std,
            "rmse": float(rmse),
        }
        if _has_relative_metrics(key):
            # denom is a scalar, so rel_err = abs_err / denom and the std scales with it.
            payload["rel_err_mean"] = float(abs_err_mean / denom)
            payload["rel_err_std"] = float(abs_err_std / denom)
            payload["rel_rmse"] = float(rmse / denom)
        if stats["iou_count"] > 0:
            iou_mean = stats["iou_sum"] / stats["iou_count"]
            payload["iou_mean"] = float(iou_mean)
            payload["iou_std"] = float(
                np.sqrt(max(0.0, stats["iou_sq_sum"] / stats["iou_count"] - iou_mean ** 2))
            )
            payload["iou_support_frames"] = int(stats["iou_count"])
        if stats.get("iou_all_count"):
            # Legacy all-frames value, so older numbers stay comparable.
            payload["iou_mean_all_frames"] = float(stats["iou_all_sum"] / stats["iou_all_count"])
            payload["iou_empty_gt_frames"] = int(stats.get("iou_empty_gt_frames", 0))
            payload["iou_empty_gt_false_positive_frames"] = int(
                stats.get("iou_empty_gt_false_positive_frames", 0)
            )
        for tag in ("substrate", "surface"):
            ra = region_acc[key][tag]
            if ra["n"] > 0:
                r_mae = ra["abs"] / ra["n"]
                r_rmse = np.sqrt(ra["sq"] / ra["n"])
                r_denom = (ra["truth"] / ra["n"]) + 1e-12
                payload[f"{tag}_abs_err_mean"] = float(r_mae)
                payload[f"{tag}_rmse"] = float(r_rmse)
                if _has_relative_metrics(key):
                    payload[f"{tag}_rel_err_mean"] = float(r_mae / r_denom)
        # Prediction range and the worst cell's location. Cheap, and it is what
        # distinguishes "the shape is a bit off" from "a few cells blew up": a
        # ratio rmse/abs_err_mean far above ~5 with a huge pred_max is the latter.
        for extra in ("pred_max", "pred_min", "pred_max_batch", "pred_max_loc",
                      "pred_max_truth", "pred_max_alpha", "pred_max_z"):
            if extra in stats:
                payload[extra] = stats[extra]
        if abs_err_mean > 0:
            payload["rmse_over_mae"] = float(rmse / abs_err_mean)
        # Gas-gap accounting. The gas above the material is ~25% of a FLOW-3D domain
        # and, with alpha masking, carries the SAME constant fill in prediction and
        # truth, so it contributes ~zero error and dilutes every mean by ~1/0.75
        # (measured on split 21285: T MAE ratio 1.33x GAOT, 1.31x RIGNO).
        #
        # The gas-excluded values are ALREADY emitted as `substrate_abs_err_mean` /
        # `substrate_rmse` above -- they always were. An earlier version of this block
        # overwrote `abs_err_mean`/`rmse` in place with the substrate values, which was
        # a silent redefinition: all 120 stored evals carry whole-domain numbers under
        # those keys, and the console ranks models on `T.abs_err_mean`
        # (aiida-pino console/model_metrics.py), so mixing the two populations under one
        # key would have produced a meaningless leaderboard. Do NOT reintroduce that.
        #
        # What is recorded here is the fraction of the domain that is substrate, so the
        # dilution is visible on every eval, plus the flag itself for provenance.
        ra_sub = region_acc[key]["substrate"]
        if ra_sub["n"] > 0 and stats["count"] > 0:
            payload["substrate_fraction"] = float(ra_sub["n"] / stats["count"])
        if getattr(run_cfg, "metrics_exclude_gas", False):
            # Marker only: `abs_err_mean`/`rmse` stay whole-domain and comparable with
            # every historical eval; read `substrate_*` for the gas-excluded figure.
            payload["metrics_exclude_gas"] = True
        # Beam box: the only region where the hard physics lives. A substrate mean
        # is still ~99% quiescent metal and cannot see a melt-pool failure.
        ra_box = region_acc[key]["beambox"]
        if ra_box["n"] > 0:
            box_mae = ra_box["abs"] / ra_box["n"]
            payload["beambox_abs_err_mean"] = float(box_mae)
            payload["beambox_rmse"] = float(np.sqrt(ra_box["sq"] / ra_box["n"]))
            payload["beambox_n"] = int(ra_box["n"])
            if _has_relative_metrics(key):
                payload["beambox_rel_err_mean"] = float(
                    box_mae / ((ra_box["truth"] / ra_box["n"]) + 1e-12)
                )
            if ra_box["union"] > 0:
                payload["beambox_iou"] = float(ra_box["inter"] / ra_box["union"])
            elif key in ("fl", "alpha"):
                payload["beambox_iou"] = 1.0
        # Thermal headroom, on the temperature key only: the margin that decides
        # whether ANY melt can be produced.
        if key == "T" and np.isfinite(head_acc["max"]) and threshold_K > 0:
            payload["melt_threshold_K"] = float(threshold_K)
            payload["pred_T_max_K"] = float(head_acc["max"])
            _margin = head_acc["margin"]
            if not np.isfinite(_margin):
                _margin = head_acc["max"] - threshold_K
            payload["thermal_headroom_K"] = float(_margin)
            payload["can_reach_melt"] = float(_margin > 0.0)
            # A mixed-material eval has no single threshold, and reporting one as if it had
            # is how a metric comes to describe the wrong material. Say the spread instead
            # of hiding it; the keys appear only when it is real.
            if np.isfinite(head_acc["thr_max"]) and head_acc["thr_max"] - head_acc["thr_min"] > 1.0:
                payload["melt_threshold_K_min"] = float(head_acc["thr_min"])
                payload["melt_threshold_K_max"] = float(head_acc["thr_max"])
            # Streaming variant of the tail deficit: the lazy path cannot hold the
            # order statistics needed for p99.9, so it compares peaks. Same signal
            # -- a model that clears the threshold can still run measurably colder
            # than truth (GAOT disp_deep_v2: +837 K headroom, 151 K colder tail).
            if np.isfinite(head_acc["true_max"]):
                payload["true_T_max_K"] = float(head_acc["true_max"])
                payload["tail_deficit_max_K"] = float(head_acc["true_max"] - head_acc["max"])
                # A peak-based deficit grows as sqrt(2 ln N) even for an unchanged
                # model, so record N and the normalised value beside it.
                payload.update(
                    _tail_deficit_scale(payload["tail_deficit_max_K"], head_acc["frames"])
                )
        # What the store actually served, onto the node. See NpzDirStore.served_grid:
        # the derivation profile and the crop env var both misreport it, so this is the
        # only auditable record of the grid a number was measured on.
        _sg = getattr(getattr(dataset, "store", None), "served_grid", None)
        if callable(_sg):
            _sg = _sg()
            if _sg:
                payload["served_grid"] = _sg
        errors[key] = payload

    if geom_acc:
        errors["geometry"] = {name: total / w for name, (total, w) in geom_acc.items()}

    if frame_stride > 1:
        # Carried into the stored metrics so a subsampled number can never be read
        # back as a full-split score months later.
        errors["eval_frame_stride"] = {
            "stride": float(frame_stride),
            "frames_scored": float(batch_idx + 1),
        }

    if transfer_mode:
        stamp = {"target": transfer_mode,
                 "behind": float(getattr(run_cfg, "eval_frame_behind", 0)),
                 "ahead": float(getattr(run_cfg, "eval_frame_ahead", 0))}
        if transfer_mode == "steady":
            stamp["cases_scored"] = float(steady_state["scored"])
            stamp["avg_window"] = float(getattr(run_cfg, "eval_frame_avg", 30))
        errors["eval_frame_transfer"] = stamp

    _write_metric_file(output_dir / "model_errors.txt", errors,
                       network_dir=getattr(run_cfg, "network_base_dir", None), run_cfg=run_cfg)
    return InferenceMetrics(errors=errors, val_rows=[])


def write_val_error_csv(rows: Sequence[dict[str, Any]], csv_path: Path) -> None:
    """Write per-sample validation error rows to CSV."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "global_index",
        "local_val_index",
        "P",
        "V_scan",
        "output_key",
        "abs_err_mean",
        "rel_err_mean",
        "iou",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
