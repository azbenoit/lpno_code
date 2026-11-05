from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InferInputs(StrictModel):
    p_watts: float
    v_scan_mps: float


class InferView(StrictModel):
    field: Literal["T", "alpha", "fl"] = "T"
    plot_mode: Literal["2d", "3d"] = "3d"
    y_index: int = -1
    show_t_contour: bool = False
    t_contour_k: float = 1898.0
    # "xz" cross-section (2D slice / 3D cut-away, default) or "xy" top view.
    plane: Literal["xy", "xz"] = "xz"


class InferRender(StrictModel):
    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)


class InferRequest(StrictModel):
    client_request_id: str
    client_seq: int = Field(ge=0)
    mode: Literal["preview", "final"]
    # Steady model id (folder under data/steady-state/models). Empty means the
    # model loaded at startup; a non-empty id is validated against discovery.
    model: str = Field(default="", max_length=64)
    inputs: InferInputs
    view: InferView
    render: InferRender


class ClampedInputs(StrictModel):
    p_watts: float
    v_scan_mps: float


class DerivedValues(StrictModel):
    h_model: float
    h_star: float
    h_norm_model: float | None = None


class PlotPayload(StrictModel):
    mime_type: Literal["image/png"] = "image/png"
    width_px: int
    height_px: int
    png_base64: str


class ResultMeta(StrictModel):
    field: str
    plot_mode_used: Literal["2d", "3d"]
    grid_shape_xyz: tuple[int, int, int]
    y_index_used: int


class TimingMs(StrictModel):
    infer: float
    render: float
    total: float


class InferResponse(StrictModel):
    status: Literal["ok"] = "ok"
    client_request_id: str
    client_seq: int
    mode: Literal["preview", "final"]
    server_request_id: str
    clamped_inputs: ClampedInputs
    derived: DerivedValues
    result_meta: ResultMeta
    plot: PlotPayload
    timing_ms: TimingMs


# --------------------------------------------------------------------------- #
# /infer_path (time-dependent random-path inference)
# --------------------------------------------------------------------------- #

MAX_PATH_POINTS = 4096


class PathSpec(StrictModel):
    points_xy_m: list[list[float]] | None = None
    preset: str | None = None

    @model_validator(mode="after")
    def _validate_exactly_one(self) -> "PathSpec":
        if (self.points_xy_m is None) == (self.preset is None):
            raise ValueError("path must contain exactly one of 'points_xy_m' or 'preset'")
        if self.preset is not None and not self.preset.strip():
            raise ValueError("preset name must be non-empty")
        if self.points_xy_m is not None:
            pts = self.points_xy_m
            if not (2 <= len(pts) <= MAX_PATH_POINTS):
                raise ValueError(f"points_xy_m must contain 2..{MAX_PATH_POINTS} points, got {len(pts)}")
            for i, pt in enumerate(pts):
                if len(pt) != 2:
                    raise ValueError(f"points_xy_m[{i}] must be an [x, y] pair")
                if not all(math.isfinite(c) for c in pt):
                    raise ValueError(f"points_xy_m[{i}] contains a non-finite coordinate")
        return self


class PathView(StrictModel):
    field: Literal["T", "fl", "alpha"] = "T"
    z_index: int = -1
    # "xy" = top view at a z-slice (default); "xz" = cross-section at a y-slice.
    plane: Literal["xy", "xz"] = "xy"
    y_index: int = -1  # y-slice for the xz cross-section (-1 = domain centre)


class InferPathRequest(StrictModel):
    client_request_id: str = ""
    model: str = Field(min_length=1, max_length=64)
    path: PathSpec
    view: PathView = PathView()
    render: InferRender
    # Laser power / scan speed for variable-physics models (meta.json "physics").
    # None = the model's default. Sending a value for a fixed-physics model
    # (e.g. rp3) is a 422; out-of-range values are clamped to the trained range
    # and reported back in meta (same convention as /infer's ClampedInputs).
    p_watts: float | None = Field(default=None, gt=0.0, lt=1.0e4)
    v_scan_mps: float | None = Field(default=None, gt=0.0, lt=1.0e2)
    # Frame spacing in seconds. Frames span the weld duration (or the fixed
    # window for fixed-duration models) at THIS step, so the frame count
    # follows from the physics instead of being pinned at 20 — a 1.5 mm weld
    # and a 6 mm weld no longer animate at different, unstated time scales.
    # None = DEFAULT_FRAME_DT_S (1.3e-4 s: 20 frames for 1.5 mm at 0.6 m/s).
    # The resulting count is clamped to MIN_FRAMES..MAX_FRAMES (5..100) and to
    # the samples that exist, and the effective step is echoed in path_meta.
    frame_dt_s: float | None = Field(default=None, gt=0.0, le=1.0)


class PathMeta(StrictModel):
    n_points_used: int
    path_length_m: float
    laser_off_time_s: float
    clamped: bool
    preset: str | None
    gt_available: bool
    points_xy_m: list[list[float]]
    # MAE / RMSE of the shown field vs GT over the whole space-time volume
    # (Kelvin for T, 0..1 for alpha/fl); None when no GT is available.
    gt_mae: float | None = None
    gt_rmse: float | None = None
    # Applied laser physics (after clamping to the model's trained range);
    # None for fixed-physics models where the values are implicit constants.
    p_watts: float | None = None
    v_scan_mps: float | None = None
    physics_clamped: bool = False
    # Rendered cadence. frame_dt_s is the EFFECTIVE spacing (frames snap to the
    # training-dt dose grid, and the count is capped), so a client can show the
    # step it actually got beside the frame count rather than what it asked for.
    n_frames: int | None = None
    frame_dt_s: float | None = None


class PathResultMeta(StrictModel):
    grid_shape_xyz: tuple[int, int, int]
    z_index_used: int
    field: Literal["T", "fl", "alpha"]
    panel: Literal["single", "triple"]


class InferPathResponse(StrictModel):
    status: Literal["ok"] = "ok"
    model: str = Field(min_length=1, max_length=64)
    times_s: list[float]
    frames: list[str]  # 20 base64 PNGs of the requested field only
    path_meta: PathMeta
    result_meta: PathResultMeta
    timing_ms: TimingMs


class InferPathRender3DRequest(StrictModel):
    """Like InferPathRequest but with no view block: the 3D render has a fixed camera."""
    # Frame spacing in seconds. Frames span the weld duration (or the fixed
    # window for fixed-duration models) at THIS step, so the frame count
    # follows from the physics instead of being pinned at 20 — a 1.5 mm weld
    # and a 6 mm weld no longer animate at different, unstated time scales.
    # None = DEFAULT_FRAME_DT_S (1.3e-4 s: 20 frames for 1.5 mm at 0.6 m/s).
    # The resulting count is clamped to MIN_FRAMES..MAX_FRAMES (5..100) and to
    # the samples that exist, and the effective step is echoed in path_meta.
    frame_dt_s: float | None = Field(default=None, gt=0.0, le=1.0)

    client_request_id: str = ""
    model: str = Field(min_length=1, max_length=64)
    path: PathSpec
    render: InferRender
    plane: Literal["xy", "xz"] = "xy"  # top view of the block, or a cut-away cross-section
    # Same optional variable-physics inputs as InferPathRequest (shared cache key).
    p_watts: float | None = Field(default=None, gt=0.0, lt=1.0e4)
    v_scan_mps: float | None = Field(default=None, gt=0.0, lt=1.0e2)


class PathRender3DResultMeta(StrictModel):
    grid_shape_xyz: tuple[int, int, int]
    panel: Literal["3d"] = "3d"


class InferPathRender3DResponse(StrictModel):
    status: Literal["ok"] = "ok"
    model: str = Field(min_length=1, max_length=64)
    times_s: list[float]
    frames: list[str]  # base64 PNGs, one per time step; same slider contract as /infer_path
    path_meta: PathMeta
    result_meta: PathRender3DResultMeta
    timing_ms: TimingMs


class PresetEntry(StrictModel):
    name: str
    points_xy_m: list[list[float]]
    # The stored case's own laser physics (variable-path GT sets vary these);
    # None for legacy sets whose manifests predate the fields.
    p_watts: float | None = None
    v_scan_mps: float | None = None


class PresetsResponse(StrictModel):
    status: Literal["ok"] = "ok"
    presets: list[PresetEntry]


# --------------------------------------------------------------------------- #
# /infer_straight (time-dependent straight-line inference, free P/V)
# --------------------------------------------------------------------------- #


class StraightInputs(StrictModel):
    p_watts: float
    v_scan_mps: float


class InferStraightRequest(StrictModel):
    client_request_id: str = ""
    model: str = Field(min_length=1, max_length=64)
    inputs: StraightInputs
    view: PathView = PathView()
    render: InferRender


class InferStraightRender3DRequest(StrictModel):
    client_request_id: str = ""
    model: str = Field(min_length=1, max_length=64)
    inputs: StraightInputs
    render: InferRender
    plane: Literal["xy", "xz"] = "xy"  # top view of the block, or a cut-away cross-section


class StraightMeta(StrictModel):
    p_watts: float
    v_scan_mps: float
    path_length_m: float
    laser_off_time_s: float
    gt_available: bool
    gt_name: str | None
    split: str | None
    points_xy_m: list[list[float]]
    gt_mae: float | None = None
    gt_rmse: float | None = None


class InferStraightResponse(StrictModel):
    status: Literal["ok"] = "ok"
    model: str = Field(min_length=1, max_length=64)
    times_s: list[float]
    frames: list[str]
    straight_meta: StraightMeta
    result_meta: PathResultMeta
    timing_ms: TimingMs


class InferStraightRender3DResponse(StrictModel):
    status: Literal["ok"] = "ok"
    model: str = Field(min_length=1, max_length=64)
    times_s: list[float]
    frames: list[str]
    straight_meta: StraightMeta
    result_meta: PathRender3DResultMeta
    timing_ms: TimingMs


# --------------------------------------------------------------------------- #
# /gt_catalog (P-V ground-truth data points, colored by split client-side)
# --------------------------------------------------------------------------- #


class GtCatalogEntry(StrictModel):
    name: str
    p_watts: float
    v_scan_mps: float
    split: Literal["train", "val", "test"]


class GtCatalogResponse(StrictModel):
    status: Literal["ok"] = "ok"
    dataset: Literal["steady", "straight"]
    entries: list[GtCatalogEntry]


class ModelInfo(StrictModel):
    id: str
    label: str
    type: str = ""
    default: bool = False
    # Per-model variable-physics capabilities from meta.json's "physics" block
    # (p_watts/v_scan_mps ranges + defaults, start_anywhere, ...). None for
    # fixed-physics models — the frontend greys out the controls then.
    physics: dict[str, Any] | None = None


class ModelsResponse(StrictModel):
    status: Literal["ok"] = "ok"
    page: Literal["steady", "straight", "path"]
    models: list[ModelInfo]


# --------------------------------------------------------------------------- #
# /infer/gt_compare (steady-state prediction vs FLOW-3D ground truth)
# --------------------------------------------------------------------------- #


class InferGtCompareRequest(StrictModel):
    client_request_id: str = ""
    model: str = Field(default="", max_length=64)
    inputs: InferInputs
    field: Literal["T", "alpha", "fl"] = "T"
    y_index: int = -1
    # "2d" stacked XZ panels (default) or "3d" stacked iso-surface blocks; plane
    # selects cross-section vs top view for the 3D comparison.
    plot_mode: Literal["2d", "3d"] = "2d"
    plane: Literal["xy", "xz"] = "xz"
    render: InferRender


class GtCompareResultMeta(StrictModel):
    field: Literal["T", "alpha", "fl"]
    gt_available: bool
    gt_p_watts: float | None
    gt_v_scan_mps: float | None
    gt_split: Literal["train", "val", "test"] | None = None
    grid_shape_xyz: tuple[int, int, int]
    y_index_used: int
    panel: Literal["single", "gt_compare", "gt_compare_3d", "single_3d"]
    plot_mode_used: Literal["2d", "3d"] = "2d"
    # MAE / RMSE of the shown field vs GT (steady pred is x-resampled onto the
    # GT grid over their overlap); None when no GT case matched.
    gt_mae: float | None = None
    gt_rmse: float | None = None


class InferGtCompareResponse(StrictModel):
    status: Literal["ok"] = "ok"
    server_request_id: str
    clamped_inputs: ClampedInputs
    result_meta: GtCompareResultMeta
    plot: PlotPayload
    timing_ms: TimingMs


class ErrorDetail(StrictModel):
    code: str
    message: str
    retryable: bool
    retry_after_ms: int | None = None


class ErrorResponse(StrictModel):
    status: Literal["error"] = "error"
    error: ErrorDetail
    server_request_id: str


def make_error_response(
    *,
    server_request_id: str,
    code: str,
    message: str,
    retryable: bool,
    retry_after_ms: int | None = None,
) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            retry_after_ms=retry_after_ms,
        ),
        server_request_id=server_request_id,
    )
