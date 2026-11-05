import os
import warnings
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.spatial import Delaunay
import matplotlib.tri as mtri
import pyvista as pv
from pathlib import Path


# Ti-6Al-4V boiling point [K]. With alpha_masking on, this is the value the
# off-metal cells of T are filled with, in preprocessing (raw, before the
# ref_scales division) and in the model's own MaskNode (already normalized).
# MaskNode used to carry its own literal 3123/3000, so the two agreed
# only while ref_scales.T was 3000 -- a different T reference silently trained
# the network against a target its mask node could not reproduce. Override per
# material with material_params.T_boil.
T_BOIL_K = 3123.0


def boiling_temperature(material_params) -> float:
    """Configured boiling point in K, falling back to the Ti-6Al-4V default."""

    try:
        value = (material_params or {})["T_boil"]
    except KeyError:
        return T_BOIL_K
    return T_BOIL_K if value is None else float(value)


def t_boil_from_manifest(manifest) -> float | None:
    """`material.properties.vaporization_temperature_k` from a case manifest, if present.

    Accepts both the flow3d case manifest itself and the arraydir member wrapper that
    nests it under `simulation_metadata.flow3d_case_manifest`. Returns None rather than
    guessing -- the caller decides the fallback.
    """
    if not isinstance(manifest, dict):
        return None
    inner = manifest
    meta = manifest.get("simulation_metadata")
    if isinstance(meta, dict) and isinstance(meta.get("flow3d_case_manifest"), dict):
        inner = meta["flow3d_case_manifest"]
    props = ((inner.get("material") or {}).get("properties") or {})
    value = props.get("vaporization_temperature_k")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def melt_window_from_manifest(manifest) -> tuple[float, float] | None:
    """`(solidus, liquidus)` in K from a case manifest, if both are present.

    The melt-window twin of `t_boil_from_manifest`, same nesting rules and the same
    refusal to guess: returns None unless BOTH temperatures are readable and ordered,
    because half a window is not a window and a caller that fell back for one and not
    the other would threshold against a mixture of two materials.
    """
    if not isinstance(manifest, dict):
        return None
    inner = manifest
    meta = manifest.get("simulation_metadata")
    if isinstance(meta, dict) and isinstance(meta.get("flow3d_case_manifest"), dict):
        inner = meta["flow3d_case_manifest"]
    props = ((inner.get("material") or {}).get("properties") or {})
    try:
        t_s = float(props["solidus_temperature_k"])
        t_l = float(props["liquidus_temperature_k"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (t_s > 0 and t_l > 0 and t_l >= t_s):
        return None
    return t_s, t_l


_MISSING = object()


def custom_flag(run_cfg, key: str, default=None):
    """`custom.<key>` as the run sees it: a `run_cfg` attribute first, else the Hydra cfg.

    `RunConfig` carries explicit fields only, so a new `custom.*` key does not reach it by
    itself -- `getattr(run_cfg, key, default)` then returns the default forever, and an
    opt-out that is never read is an opt-out in name only. `run_cfg.cfg` is the resolved
    Hydra config and always has the key when the user set it.
    """
    value = getattr(run_cfg, key, _MISSING)
    if value is not _MISSING and value is not None:
        return value
    cfg = getattr(run_cfg, "cfg", None)
    if cfg is not None:
        try:
            from omegaconf import OmegaConf

            value = OmegaConf.select(cfg, f"custom.{key}", default=_MISSING)
            if value is not _MISSING and value is not None:
                return value
        except Exception:  # noqa: BLE001 - a cfg that is not an OmegaConf tree has no custom.*
            pass
    return default


_PER_CASE_FILL_ANNOUNCED: set = set()


def boiling_fill_from_channel(mat_tv, ref_scales):
    """Per-case off-metal T fill in normalized units from the `mat_Tv` channel.

    The channel arrives divided by its own `ref_scales['mat_Tv']` when the config declares
    one and raw Kelvin otherwise (`_preprocess_var` scales what it has a scale for); either
    way `channel * scale / ref_T` is the fill on the T axis. Works on numpy and torch alike
    -- plain arithmetic -- so the training MaskNode and the eval fills share one formula.
    """
    return mat_tv * (float(ref_scales.get("mat_Tv", 1.0)) / float(ref_scales["T"]))


def t_fill_field(run_cfg, tensors=None):
    """The off-metal T fill for this batch: per case when asked for and served, else the
    dataset/config scalar.

    `custom.per_case_t_boil` (default OFF) turns the fill into `mat_Tv` per case. Off by
    default because the fill is part of the TARGET the model learns: a Ti64+IN718 set
    trained with one fill (3520 or 3186 K) puts the wrong ceiling on half its cases, but
    switching an existing campaign silently would change the task it was scored on.
    `adopt_material_t_boil` already handles a single-material set of any alloy; this is
    for a MIXED set, where one fill is wrong either way. Requires the channel: with the
    flag on and no `mat_Tv` in the batch, the scalar is used and that is announced.
    """
    material_params = getattr(run_cfg, "material_params", None) or {}
    ref_scales = run_cfg.ref_scales
    if not bool(custom_flag(run_cfg, "per_case_t_boil", False)):
        return boiling_fill_value(material_params, ref_scales)
    mat_tv = tensors.get("mat_Tv") if isinstance(tensors, dict) else None
    key = ("on", mat_tv is not None)
    if key not in _PER_CASE_FILL_ANNOUNCED:
        _PER_CASE_FILL_ANNOUNCED.add(key)
        if mat_tv is not None:
            print("[t_boil] PER-CASE mask fill from the mat_Tv channel (custom.per_case_t_boil)",
                  flush=True)
        else:
            print("[t_boil] custom.per_case_t_boil is set but this batch carries no mat_Tv "
                  f"channel; using the scalar {boiling_temperature(material_params)} K. The "
                  "dataset must serve mat_Tv (older manifests lack vaporization_temperature_k).",
                  flush=True)
    if mat_tv is None:
        return boiling_fill_value(material_params, ref_scales)
    return boiling_fill_from_channel(mat_tv, ref_scales)


def boiling_fill_value(material_params, ref_scales) -> float:
    """Off-metal T fill in normalized units: ``T_boil / ref_scales['T']``."""

    return boiling_temperature(material_params) / float(ref_scales["T"])


def aggregate_fold_stats(fold_stats, fold_counts=None, output_path=None):
    if not fold_stats:
        raise ValueError("fold_stats cannot be empty.")
    if fold_counts is not None and len(fold_counts) != len(fold_stats):
        raise ValueError("fold_counts must match fold_stats length.")

    key_order = []
    for fold in fold_stats:
        for key in fold:
            if key not in key_order:
                key_order.append(key)

    aggregated = {}
    for key in key_order:
        stat_names = []
        for fold in fold_stats:
            if key not in fold:
                continue
            for stat_name in fold[key]:
                if stat_name not in stat_names:
                    stat_names.append(stat_name)

        key_stats = {}
        for stat_name in stat_names:
            values = [
                fold[key][stat_name]
                for fold in fold_stats
                if key in fold and stat_name in fold[key]
            ]
            if not values:
                continue
            values_arr = np.asarray(values, dtype=float)
            key_stats[stat_name] = {
                "mean": float(values_arr.mean()),
                "std": float(values_arr.std()),
                "min": float(values_arr.min()),
                "max": float(values_arr.max()),
                "n": int(values_arr.size),
            }

        if fold_counts is not None:
            for stat_name in stat_names:
                if not stat_name.endswith("_mean"):
                    continue
                base = stat_name[:-5]
                std_name = f"{base}_std"
                means = []
                stds = []
                counts = []
                for fold_idx, fold in enumerate(fold_stats):
                    if key not in fold:
                        continue
                    if stat_name not in fold[key] or std_name not in fold[key]:
                        continue
                    means.append(float(fold[key][stat_name]))
                    stds.append(float(fold[key][std_name]))
                    counts.append(int(fold_counts[fold_idx]))
                if not means:
                    continue
                counts_arr = np.asarray(counts, dtype=float)
                means_arr = np.asarray(means, dtype=float)
                stds_arr = np.asarray(stds, dtype=float)
                total_n = int(np.sum(counts_arr))
                weighted_mean = float(np.sum(counts_arr * means_arr) / total_n)
                if total_n > 1:
                    within = np.sum((counts_arr - 1) * (stds_arr ** 2))
                    between = np.sum(counts_arr * ((means_arr - weighted_mean) ** 2))
                    pooled_var = float((within + between) / (total_n - 1))
                else:
                    pooled_var = 0.0
                key_stats[f"{base}_pooled_var"] = {"value": pooled_var, "n": total_n}
                key_stats[f"{base}_pooled_std"] = {"value": float(np.sqrt(pooled_var)), "n": total_n}
        aggregated[key] = key_stats

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for key in key_order:
                if key not in aggregated:
                    continue
                f.write(f"[{key}]\n")
                for stat_name, summary in aggregated[key].items():
                    f.write(f"{stat_name}:\n")
                    for metric_name, metric_value in summary.items():
                        f.write(f"  {metric_name}: {metric_value}\n")
                f.write("\n")

    return aggregated

def split_dict(d, idx):
    return {k: v[idx] for k, v in d.items()}

def make_interface_sharp(alpha, threshold=0.5):
    """
    Make the liquid-solid interface sharp by thresholding alpha.

    Inputs:
        alpha: Liquid fraction field (numpy array)
        threshold: Threshold value to determine solid/liquid
    Outputs:
        alpha_sharp: Sharp liquid fraction field (numpy array)
    """
    alpha_sharp = np.where(alpha >= threshold, 1.0, 0.0)
    return alpha_sharp

def iou_per_batch(a, b, threshold=0.5):
    """
    Compute IoU per batch for channel-first arrays [B, C, ...] using a threshold.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim < 4:
        raise ValueError(f"expected at least 4D array [B, C, ...], got {a.ndim}D")

    a_mask = a > threshold
    b_mask = b > threshold

    reduce_axes = tuple(range(1, a.ndim))
    intersection = np.logical_and(a_mask, b_mask).sum(axis=reduce_axes)
    union = np.logical_or(a_mask, b_mask).sum(axis=reduce_axes)

    # Use masked division so NumPy does not evaluate intersection / union where union is zero.
    iou = np.ones_like(intersection, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=union != 0)
    return iou

def iou_per_batch_with_support(a, b, threshold=0.5):
    """Per-batch IoU plus a mask of which entries have NON-EMPTY ground truth.

    `iou_per_batch` returns 1.0 when both masks are empty, which is the right
    convention for a single comparison but makes a plain mean over frames
    meaningless on sparse targets. With `time_as_batch=true` each batch entry is one
    time frame, and early frames have no melt at all, so a frame whose GT is empty
    scores either 1.0 (predicted nothing) or EXACTLY 0.0 (predicted anything at all).
    The mean of that is bimodal noise, not accuracy.

    Measured 2026-07-30 on a real frame from eval 22352 (`gaot_cond150_fullres_20k`):
    GT `fl` was identically zero everywhere (range [0.0, 0.0]) while the prediction
    had 4 cells above 0.5 out of 600,000 -- 0.0007% -- giving IoU 0/4 = 0.0, the worst
    possible score, for a rounding-level false positive against no melt at all. Across
    frames this produced `fl.iou_mean` 0.539 +/- 0.413 and was misread as GAOT having
    an interface problem. It is a melt-VOLUME regime effect: datasets with big
    keyholes (steady straight-path, `fl` IoU 0.89-0.93) have ample support, while a
    weak-keyhole material sweep or a per-frame conduction case often has none.

    Returns `(iou, gt_nonempty)` so callers can average over frames that actually
    have something to measure, and report the empty-GT frames separately instead of
    silently folding them into the mean.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    iou = iou_per_batch(a, b, threshold=threshold)
    reduce_axes = tuple(range(1, b.ndim))
    gt_nonempty = (b > threshold).any(axis=reduce_axes)
    return iou, gt_nonempty


#: Liquid-fraction sharpness. Kept at the historical value: changing it would
#: move the melt threshold, which every stored metric is defined against.
LIQUID_FRACTION_BETA = 2.5


def smooth_liquid_fraction_numpy(T, T_s, T_l):
    """Smooth liquid fraction in [0,1], sigmoid form.

    `0.5*(tanh(x)+1)` and `sigmoid(2x)` are the SAME function algebraically, but
    not numerically: float32 `tanh` returns exactly -1.0 for x <= -9.011, so the
    tanh form is bit-zero (and bit-zero in its derivative) for every T below
    1717.8 K with the Ti64 solidus/liquidus. Model 29125 peaks at 1631 K, well
    inside that dead zone, so any loss routed through `fl` measured
    `max|dL/dT| = 0.0` -- silent on exactly the models that need it (2026-08-04).

    The sigmoid form computes the small value directly instead of cancelling two
    near-equal numbers, so the gradient survives far further out (7.6e-10 at
    1631 K, 7.6e-7 at 1700 K, versus exactly 0). This is the same fix
    `fill_where_mask_smooth_torch` already carries for the alpha gate.

    Forward values are unchanged to float32 precision: worst difference 5.96e-08
    over 300-4000 K, and ZERO of 200001 sampled temperatures change their
    `fl > 0.5` decision -- so every stored metric and every prior ablation stays
    comparable. See tests/test_liquid_fraction_sigmoid.py.
    """
    # T_s/T_l may be ARRAYS: ck300 spans liquidus 868-1898 K, and one Ti64 window
    # applied to all 300 materials made the fl target near-empty for every
    # low-melting one (ckc_072: 1080 Melt Region cells, 4 under Ti64's window).
    T, T_s, T_l = np.asarray(T), np.asarray(T_s, dtype=np.float64), np.asarray(T_l, dtype=np.float64)
    if np.any(T_s >= T_l):
        raise ValueError("Require T_s < T_l")
    phi = (T - T_s) / (T_l - T_s)
    # sigmoid(2*beta*(phi-0.5)) == 0.5*(tanh(beta*(phi-0.5))+1)
    return 1.0 / (1.0 + np.exp(-2.0 * LIQUID_FRACTION_BETA * (phi - 0.5)))

def smooth_liquid_fraction_torch(T, T_s, T_l):
    """Torch twin of `smooth_liquid_fraction_numpy`; see it for why sigmoid.

    Both must use the same form or the melt mask the model trains against
    differs from the one the metrics score it with.
    """
    import torch as _torch
    if bool(_torch.as_tensor(T_s >= T_l).any()):
        raise ValueError("Require T_s < T_l")
    phi = (T - T_s) / (T_l - T_s)
    return torch.sigmoid(2.0 * LIQUID_FRACTION_BETA * (phi - 0.5))

def liquid_fraction(T, T_s, T_l):
    """
    Returns 0 for T<T_s, 1 for T>T_l, and linear between T_s and T_l.
    Works on numpy arrays or scalars.
    """
    T = np.asarray(T, dtype=float)
    if T_s >= T_l:
        raise ValueError("Require T_s < T_l")
    frac = (T - T_s) / (T_l - T_s)
    return np.clip(frac, 0.0, 1.0)


#: Smallest gate value the smooth mask may take. Keeps the masked field's
#: gradient alive when the mask is confidently wrong, so a run can always climb
#: back out; 1e-6 is far below any metric's precision in the healthy regime.
GATE_FLOOR = 1.0e-6


def king_material_factor(rho, cp, k, latent, T_l, T_0=293.0, A=1.0):
    """The material half of King's normalized enthalpy, as a DIVISOR.

    King: ``dH/h_s = A P / (rho (cp dT + L_f) sqrt(pi D v r0^3))`` with ``D = k/(rho cp)``.
    `h_norm_raw` supplies ``P / sqrt(|v| r0^3)``; this returns everything else, so
    ``h_norm_raw(P, v, r0, material_factor=f)`` is the full King number.

    Returned as a divisor (not a multiplier) so the numerator keeps the shape the
    split-max denominator was always fitted against — a material with a LOWER factor
    melts more readily and reads a HIGHER normalized enthalpy.

    Kept here, beside `h_norm_raw`, because both the eager numerator and the lazy
    denominator must compute it identically. Splitting that across two files is what
    broke H_norm on 2026-08-20.
    """
    rho = np.asarray(rho, dtype=np.float64)
    cp = np.asarray(cp, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    latent = np.asarray(latent, dtype=np.float64)
    T_l = np.asarray(T_l, dtype=np.float64)
    diffusivity = np.maximum(k / np.maximum(rho * cp, 1e-12), 1e-12)
    sensible = cp * np.maximum(T_l - float(T_0), 1.0) + latent
    return np.maximum(rho * sensible * np.sqrt(np.pi * diffusivity), 1e-30) / np.maximum(A, 1e-6)


def h_norm_raw(power, v_scan, beam_radius=None, material_factor=None):
    """The UNNORMALIZED King normalized-enthalpy numerator: ``P / sqrt(|V| * r0^3)``.

    King's form is ``dH/h_s = A P / (rho h_s sqrt(pi D v r0^3))``; v and r0^3 sit under
    ONE root, which is what this returns. The material constants (A, rho, c_p, L_f, D,
    pi) are absorbed by the split-max denominator applied downstream -- exact while a
    split shares one material, which is the case for every single-material sweep. A
    material-VARYING split needs them carried explicitly (`h_norm_local` does that).

    One definition, because two places must agree exactly: the eager builder in
    `preprocessing_datasets._preprocess_var` and the split-level denominator fit in
    `lazy_runtime.compute_h_norm_denom`. They diverged the moment the r0 term was added
    to the numerator only (2026-08-20). The eager path self-cancels -- it divides by the
    max of the very array it just built -- so nothing looked wrong there, while every
    steady arm fed the network H_norm = 2.5e6 instead of 0.886: the correct value times
    1/sqrt(r0^3) for r0 = 50 um. mref and lab read exactly 1.0, which is why a whole
    campaign passed over it.

    `beam_radius` is raw metres; None keeps the classic radius-free form.
    """
    value = np.asarray(power, dtype=np.float64) / np.sqrt(
        np.abs(np.asarray(v_scan, dtype=np.float64))
    )
    if beam_radius is not None:
        r0 = np.maximum(np.abs(np.asarray(beam_radius, dtype=np.float64)), 1e-12)
        value = value / np.sqrt(r0 ** 3)
    # OPT-IN material term. Off by default: `h_norm_include_radius` taught us that
    # imposing a King grouping can LOSE decisively (radiusON was 130 % worse on RIGNO
    # steady), so this ships as an ablatable flag, never as a silent default. Whoever
    # sets it must set it on BOTH sides -- see compute_h_norm_denom.
    if material_factor is not None:
        value = value / np.maximum(np.asarray(material_factor, dtype=np.float64), 1e-30)
    return value


def fill_where_mask_smooth_torch(data, mask, threshold=0.5, d=0.0, channel_axis=1, k=20.0, inplace=False):
    """
    Torch version: smoothly blend data toward d using a tanh gate on (mask - threshold).
    mask is expected in [0, 1]; k controls sharpness (larger = sharper).
    Shapes: broadcast mask to data; channel_axis must be 1-sized if mask has fewer channels.
    """
    data_t = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    mask_t = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=data_t.device, dtype=data_t.dtype)

    # Broadcast mask to data shape if needed
    if mask_t.shape != data_t.shape:
        if mask_t.ndim == data_t.ndim - 1:
            mask_t = mask_t.unsqueeze(channel_axis)
        elif mask_t.shape[channel_axis] != 1:
            raise ValueError("mask channel dimension must be 1 to broadcast")
        mask_t = mask_t.expand_as(data_t)

    # `0.5*(tanh(x)+1)` is mathematically `sigmoid(2x)`, but for a confidently
    # wrong mask (x very negative) tanh saturates to -1.0 and the `+1` cancels
    # it *exactly*: the gate becomes bit-zero, so `d out/d data` AND
    # `d gate/d mask` are both exactly 0 and nothing downstream can ever
    # recover. MaskNode fills `alpha` with d=0.0 wherever the gate is ~0, so a
    # model whose alpha dips below threshold has alpha overwritten to 0, which
    # pins the gate at 0 for good — an absorbing dead state. That froze a
    # latent-128 RIGNO bit-identically from step 500 (when the mask warmup
    # engages) at two different learning rates, 2026-07-24.
    #
    # sigmoid computes the small value instead of cancelling two near-equal
    # numbers, so it stays finite and differentiable far further out; fp32
    # because the same expression underflows to 0 in fp16 under AMP. GATE_FLOOR
    # keeps the data path alive even past sigmoid's own underflow (~alpha < -1.4
    # here). Both are negligible in the healthy regime: at alpha=1 the gate is
    # still 1.0 to float precision.
    gate = GATE_FLOOR + (1.0 - GATE_FLOOR) * torch.sigmoid(
        2.0 * k * (mask_t.float() - threshold)
    ).to(data_t.dtype)
    fill_val = torch.as_tensor(d, device=data_t.device, dtype=data_t.dtype)
    blended = fill_val + gate * (data_t - fill_val)
    if inplace:
        data_t.copy_(blended)
        return data_t
    return blended

def fill_where_mask_smooth_np(data, mask, threshold, d, channel_axis=1, k=20.0, inplace=False):
    """
    Smoothly blend data toward d using a tanh gate on (mask - threshold).
    mask is expected in [0, 1]; k controls sharpness (larger = sharper).
    """
    data = np.asarray(data)
    mask = np.asarray(mask)

    if mask.shape != data.shape:
        if mask.ndim == data.ndim - 1:
            mask = np.expand_dims(mask, axis=channel_axis)
        elif mask.shape[channel_axis] != 1:
            raise ValueError("mask channel dimension must be 1 to broadcast")
        mask = np.broadcast_to(mask, data.shape)

    gate = 0.5 * (np.tanh(k * (mask - threshold)) + 1.0)  # in (0,1)
    arr = data if inplace else np.array(data, copy=True)
    blended = d + gate * (arr - d)
    if inplace:
        arr[...] = blended
        return arr
    return blended

def fill_where_mask(data, mask, threshold, d, channel_axis=1, inplace=False):
    """
    Like fill_where_mask, but allows mask to have a singleton channel axis (size 1)
    while data has multiple channels along channel_axis.
    """
    data = np.asarray(data)
    mask = np.asarray(mask)

    if mask.shape != data.shape:
        # insert/ensure singleton at channel_axis
        if mask.ndim == data.ndim - 1:
            mask = np.expand_dims(mask, axis=channel_axis)
        elif mask.shape[channel_axis] != 1:
            raise ValueError("mask channel dimension must be 1 to broadcast")
        # now broadcast to data shape
        mask = np.broadcast_to(mask, data.shape)

    arr = data if inplace else np.array(data, copy=True)
    arr[mask < threshold] = d
    return arr

def make_poly_expr(coeffs, dep_var, dep_var_name=None, ref_scales=None):
    # T is dimensionless if ref_scales is not None (your script scales T)
    if ref_scales is not None:
        dep_var = ref_scales[dep_var_name] * dep_var
    poly = 0
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        poly += c * dep_var**(deg - i)
    return poly

def coerce_numbers(obj):
    if isinstance(obj, dict):
        return {k: coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return obj
    return obj

def reshape_to_grid(data_dict, grid_shape):
    """
    Reshape flattened grid data back to 3D structured grid format.

    Args:
        data_dict (dict): Dictionary containing flattened grid data
        grid_shape (tuple): Shape of the grid (nx, ny, nz)

    Returns:
        dict: Dictionary with reshaped grid data
    """
    reshaped_dict = {}
    for k, v in data_dict.items():
        if k == 'grid_shape':
            continue
        # Reshape from (n_points, 1) to (nx, ny, nz, 1)
        reshaped_dict[k] = v.reshape(*grid_shape, 1)
    return reshaped_dict

def scale_min_max(data, old_min=None, old_max=None, new_min=0, new_max=2):
    """
    Scale data to a new range [new_min, new_max].

    Args:
        data (np.ndarray): Input data array
        new_min (float): New minimum value
        new_max (float): New maximum value

    Returns:
        np.ndarray: Scaled data array
    """
    if old_min is None or old_max is None:
        old_min = np.min(data)
        old_max = np.max(data)
    assert old_max > old_min, "old_max must be greater than old_min"
    scaled_data = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_data, old_min, old_max

def revert_min_max(scaled_data, original_min, original_max, new_min=0, new_max=2):
    """
    Revert scaled data back to its original range.

    Args:
        scaled_data (np.ndarray): Scaled data array
        original_min (float): Original minimum value
        original_max (float): Original maximum value
        new_min (float): New minimum value used in scaling
        new_max (float): New maximum value used in scaling

    Returns:
        np.ndarray: Data array reverted to original range
    """
    reverted_data = (scaled_data - new_min) / (new_max - new_min) * (original_max - original_min) + original_min
    return reverted_data


def fd_first_second_3d(F, dx, dim):
    """
    F: [B,1,Nx,Ny,Nz], dim: 0->x, 1->y, 2->z
    returns (F_d, F_dd) on the *interior* volume: [B,1,Nx-2,Ny-2,Nz-2]
    """
    pad = (1,1, 1,1, 1,1)  # (z,y,x)
    Fp = torch.nn.functional.pad(F, pad, mode="replicate")

    # slice helpers for the chosen dimension
    def take(Fp, s0,s1,s2):
        # s0->x, s1->y, s2->z slices in the padded tensor
        return Fp[:,:, s0, s1, s2]

    if dim == 0:  # x
        Fp_m1 = take(Fp, slice(0,-2), slice(1,-1), slice(1,-1))
        Fp_0  = take(Fp, slice(1,-1), slice(1,-1), slice(1,-1))
        Fp_p1 = take(Fp, slice(2, None), slice(1,-1), slice(1,-1))
    elif dim == 1:  # y
        Fp_m1 = take(Fp, slice(1,-1), slice(0,-2), slice(1,-1))
        Fp_0  = take(Fp, slice(1,-1), slice(1,-1), slice(1,-1))
        Fp_p1 = take(Fp, slice(1,-1), slice(2, None), slice(1,-1))
    else:          # z
        Fp_m1 = take(Fp, slice(1,-1), slice(1,-1), slice(0,-2))
        Fp_0  = take(Fp, slice(1,-1), slice(1,-1), slice(1,-1))
        Fp_p1 = take(Fp, slice(1,-1), slice(1,-1), slice(2, None))

    F_d  = (Fp_p1 - Fp_m1) / (2.0 * dx)
    F_dd = (Fp_p1 - 2.0*Fp_0 + Fp_m1) / (dx*dx)
    return F_d, F_dd


def ensure_late_param_group_scheduler_keys(optimizer) -> None:
    """Make param groups added after scheduler init schedulable.

    physicsnemo builds the LR scheduler, then adds the loss aggregator's
    parameters as a NEW param group when the aggregator has any
    (trainer.py:501 vs :525). Schedulers that stamp per-group bookkeeping at
    __init__ (OneCycleLR: `initial_lr`/`max_lr`/`min_lr`) never see the late
    group, and their first `.step()` dies with `KeyError: 'initial_lr'`.
    Sum/ReLoBRaLo are parameter-free so this stayed latent until the
    homoscedastic aggregator (2026-08-10, gaot_spcal arms, dead at step 0).

    Copies every key a scheduled group has onto groups missing them (params
    and lr excepted), so the late group follows the same curve. No-op when
    all groups already match or no scheduler stamped anything.
    """
    groups = getattr(optimizer, "param_groups", None)
    if not groups or len(groups) < 2:
        return
    donor = next((g for g in groups if "initial_lr" in g), None)
    if donor is None:
        return
    for group in groups:
        for key, value in donor.items():
            if key in ("params",) or key in group:
                continue
            group[key] = value


def attach_late_param_group_repair(solver) -> None:
    """Repair late param groups at the one seam that exists inside solve().

    physicsnemo creates optimizer, scheduler AND the aggregator's late param
    group inside `_train_loop` itself (trainer.py:499/501/525), so there is no
    attribute to fix after `Solver()` construction -- the first version of this
    fix read `solver.optimizer` there and died with AttributeError, costing the
    2026-08-10 rigno_spcal resubmits their startup. `load_network()` (line 563)
    is called after the group is added and before any `scheduler.step()`, so
    wrap it on the instance and repair once the optimizer actually exists.
    """
    original = solver.load_network

    def _load_then_repair(*args, **kwargs):
        out = original(*args, **kwargs)
        ensure_late_param_group_scheduler_keys(solver.optimizer)
        return out

    solver.load_network = _load_then_repair
