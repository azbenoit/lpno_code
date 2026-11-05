"""Ground-truth helpers shared by the three GT loaders (steady, straight, path).

Numpy-only, so every loader stays cheap and safe in stub mode. Before this
module the tanh-gate masking was written four times and the (P, V) matcher
twice, each a copy that could drift from the others (and from the served
prediction's own postprocess in the adapters).
"""

from __future__ import annotations

from typing import Any

import numpy as np

T_SOLIDUS_K = 1873.0
T_LIQUIDUS_K = 1923.0
T_FILL_K = 3123.0

# A requested (P, V) is served with GT when the nearest case is this close.
# The frontend's catalog-point clicks send the exact case P/V (exact match);
# the slack absorbs the slider snap when a user lands on a point by dragging.
MATCH_DP_W = 1.0
MATCH_DV_MPS = 5.0e-3


def fill(arr: np.ndarray, mask: np.ndarray, d: float) -> np.ndarray:
    """Same gate math as the submodule's ``fill_where_mask_smooth_np(k=20,
    threshold=0.5)`` without importing its torch-heavy module: ``arr`` where
    ``mask`` >= 0.5, ``d`` where it is < 0.5, blended by a tanh of width 1/20."""
    gate = 0.5 * (np.tanh(20.0 * (mask - 0.5)) + 1.0)
    return d + gate * (arr - d)


def mask_fields(
    t_raw: np.ndarray, fof: np.ndarray, t_s: float = T_SOLIDUS_K, t_l: float = T_LIQUIDUS_K
) -> dict[str, np.ndarray]:
    """``{T, alpha, fl}`` with the served prediction's postprocess applied to a
    raw FLOW-3D temperature and fraction-of-fluid pair (any shape, same shape).

    T is filled toward ``T_FILL_K`` in the gas (fof < 0.5); alpha is gated on
    itself; fl is the smooth liquid fraction of the RAW T (tanh(2.5*(phi-0.5))
    with phi = (T - T_s)/(T_l - T_s)), gated by the already-masked alpha --
    the eval pipeline's order (T, alpha, then fl).
    """
    phi = (t_raw - t_s) / (t_l - t_s)
    fl_raw = 0.5 * (np.tanh(2.5 * (phi - 0.5)) + 1.0)
    alpha = fill(fof, fof, 0.0)
    return {"T": fill(t_raw, fof, T_FILL_K), "alpha": alpha, "fl": fill(fl_raw, alpha, 0.0)}


def nearest_pv(entries: list[dict[str, Any]], p_watts: float, v_scan_mps: float) -> dict[str, Any] | None:
    """The entry (with ``p_watts``/``v_scan_mps`` keys) nearest to the request,
    or None when none is within ``MATCH_DP_W`` AND ``MATCH_DV_MPS`` (both
    inclusive). Ties inside the box break on the weighted distance
    dP^2 + (200 dV)^2, i.e. 5 mm/s counts as 1 W."""
    best = None
    best_d = None
    for e in entries:
        dp = float(e["p_watts"]) - p_watts
        dv = float(e["v_scan_mps"]) - v_scan_mps
        if abs(dp) > MATCH_DP_W or abs(dv) > MATCH_DV_MPS:
            continue
        d = dp * dp + (dv * 200.0) ** 2
        if best_d is None or d < best_d:
            best, best_d = e, d
    return best


def load_axes(data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, z)`` float32 1-D axes from an opened NPZ (or any mapping)."""
    return (
        np.asarray(data["x"], dtype=np.float32),
        np.asarray(data["y"], dtype=np.float32),
        np.asarray(data["z"], dtype=np.float32),
    )
