# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false
"""Typed containers shared across the training/evaluation pipeline."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from physicsnemo.sym.dataset import Dataset, DictGridDataset
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node


@dataclass
class KeyConfig:
    """Grouped key definitions for FNO and optional PDE branches."""

    fno_input_keys: list[Key]
    str_fno_input_keys: list[str]
    fno_output_keys: list[Key]
    str_fno_output_keys: list[str]
    energy_input_keys: list[Key]
    str_energy_input_keys: list[str]
    energy_output_keys: list[Key]
    str_energy_output_keys: list[str]
    vof_input_keys: list[Key]
    str_vof_input_keys: list[str]
    vof_output_keys: list[Key]
    str_vof_output_keys: list[str]
    diff_input_keys: list[Key]
    str_diff_input_keys: list[str]
    diff_output_keys: list[Key]
    str_diff_output_keys: list[str]


@dataclass
class RunConfig:
    """Resolved runtime configuration after Hydra ``custom`` processing."""

    cfg: PhysicsNeMoConfig
    seed: int
    model_family: str
    data_layout: str
    eval_only: bool
    split_mode: str
    pde_loss: bool
    vof_pde_loss: bool
    diff_loss: bool
    diff_weight: float
    data_directory_path: str
    val_directory_path: str
    param_dir: str
    run_name: str
    pde_weight: float
    super_res_factor: int
    debug: bool
    sharp_alpha: bool
    alpha_masking: bool
    include_time: bool
    use_2d: bool
    time_as_dim: bool
    fast_training: bool
    time_avg_window: int
    time_as_batch: bool
    autoregressive_enabled: bool
    autoregressive_mode: str
    autoregressive_state_prev_suffix: str
    autoregressive_mask_mode: str
    autoregressive_mask_ratio: float
    autoregressive_patch_size: tuple[int, int, int]
    autoregressive_context_steps: int
    autoregressive_train_patches_per_epoch: int
    autoregressive_val_patches_per_epoch: int
    autoregressive_state_mask_keys: list[str]
    autoregressive_eval_horizon_k: int
    autoregressive_eval_use_gt_init: bool
    autoregressive_context_keys: list[str]
    autoregressive_state_keys: list[str]
    autoregressive_prev_noise_std: float
    deeponet_branch_keys: list[str]
    deeponet_trunk_keys: list[str]
    moving_ref_frame: bool
    moving_frame_steps: int
    inference_batch_size: int
    k_folds: int
    fold_index: int
    grid_params: dict[str, Any]
    ref_scales: dict[str, Any]
    laser_params: dict[str, Any]
    material_params: dict[str, Any]
    software: str
    vtk_fields: Any
    bounds: list[list[float]]
    dx: float
    dy: float
    dz: float
    nx_effective: int
    ny_effective: int
    nz_effective: int
    keys: KeyConfig
    wb_project: str
    network_base_dir: str
    # Autoregressive rollout-stability knobs (all default to current behaviour).
    autoregressive_clamp_feedback: bool = False  # clamp fed-back state at rollout
    autoregressive_clamp_t_min: float = 0.0  # normalized T floor (ambient) when clamping
    autoregressive_clamp_t_max: float | None = None  # normalized T cap; None = no cap
    autoregressive_prev_noise_resample: bool = False  # resample prev-noise per batch
    autoregressive_unroll_steps: int = 1  # k training-time rollout steps (1 = one-step)
    autoregressive_bundle_steps: int = 1  # B frames per forward (temporal bundling; 1 = off)
    val_max_cases: int = 5  # in-training validation on a seeded subset of N val cases (0 = all)
    val_frames_per_case: int = 10  # in-training validation on N evenly spaced frames per case (0 = all)
    autoregressive_unroll_detach_grad: bool = False  # truncate grad between unroll steps
    autoregressive_predict_delta: bool = False  # reparametrize as u_{t+1} = u_t + f(x)
    # Refuse to KNN-resample a source grid onto a different resolution (guards
    # against a misexported case silently interpolated to the wrong size). Set
    # true to opt back into interpolation. See openFoam_datasource.
    allow_grid_interpolation: bool = False
    # Explicit H_norm denominator (custom.h_norm_denom). None = fit the max over
    # the train split (BUG-04 behaviour). Set it when train sets differ between
    # runs that must share a normalization (active learning, candidate grids).
    h_norm_denom: float | None = None
    # Carry r0 in H_norm (King form P/sqrt(V*r0^3)). Default OFF: see the long note in
    # preprocessing_datasets._preprocess_var. Only meaningful where beam radius VARIES,
    # and only safe once every store type can serve beam_radius to the denominator.
    h_norm_include_radius: bool = False
    # OPT-IN, default OFF. Carries King's material half (rho, cp, k, L_f, T_l, A) in
    # H_norm, which is otherwise P/sqrt(V r0^3) with the material constants absorbed by
    # a split-max denominator -- exact only while a split shares ONE material. On ck300
    # the channel measures r = +1.000 against P/sqrt(V), i.e. no material information.
    # Default OFF because the radius flag taught us a King grouping can LOSE (radiusON
    # was 130 % worse on RIGNO steady): this ships ablatable, never as a silent default.
    h_norm_include_material: bool = False
    # Where an EVALUATION's H_norm denominator came from (`h_norm_record.SOURCE_*`): an
    # explicit value, the model's training record, a refit on its training split, or the
    # scored split. None outside eval. Stored in the metrics as the `[h_norm]` block.
    h_norm_denom_source: str | None = None
    # True when `h_norm_denom` was fitted WITH the r0 term (a training record or refit of a
    # radius-aware run), so it may meet a radius-carrying numerator; a user-supplied
    # custom.h_norm_denom is radius-blind by convention and stays refused there.
    h_norm_denom_radius_aware: bool = False
    # The value an eval fitted on the SCORED split when it had no training denominator
    # (source "scored split"), kept only so its `[h_norm]` metrics block can name it.
    h_norm_denom_scored_fit: float | None = None

    # Restrict the headline metrics to the substrate (z <= 0), excluding the gas gap
    # above the material. The gas is ~25% of these domains and is filled with the same
    # constant in prediction and truth, so it contributes ~zero error and dilutes every
    # mean by ~1/0.75: measured 2026-07-30, substrate/global MAE ratio 1.26-1.33 across
    # T, alpha and fl on both `rigno_ti64ds3` and `gaot_cond150_fullres_20k`.
    # Self-disabling: a dataset with no z > 0 slices has no gas gap, and the substrate
    # mask then covers everything, so this is a no-op there.
    metrics_exclude_gas: bool = False
    # Scale every dose channel by power_w/100 W. Only power-varying datasets need
    # it; left off by default so an already-trained fixed-power model is not fed
    # rescaled inputs at eval time. Declared here because `power_weighted_dose_enabled`
    # reads it off RunConfig -- without the field the yaml key is silently ignored.
    power_weighted_dose: bool = False
    # Scale `conduction_field` by the same power_w/100 W factor. A SEPARATE flag from
    # power_weighted_dose on purpose: every model in the manuscript results tables sets
    # that one True, so widening its meaning would silently rescale their eval-time
    # inputs and invalidate published rows. Same default-off reasoning otherwise.
    power_weighted_conduction: bool = False
    # Standard deviation of the conduction_field source as a multiple of the beam radius. 1.0 is the original
    # channel (a source twice as wide as the solver's Gaussian beam, whose 1/e^2 radius is the beam radius);
    # 0.5 matches the solver. Default 1.0 so every existing model keeps the input it trained on.
    conduction_source_sigma_factor: float = 1.0
    # Two-level shuffle for lazy training data: cases shuffled, frames shuffled within
    # a case, so the store LRU is hit for all but one frame per case block. Fixes the
    # measured ~99% miss rate (whole-case decompression per frame) that starved
    # full-resolution varspeed training to 14 s/iter and OOM-killed 4-worker loaders.
    # Changes sample ORDER only, never distribution; default off so existing runs keep
    # byte-identical behaviour.
    case_coherent_sampling: bool = False
    autoregressive_teacher_forcing: bool = False
    # Score this model in a frame OTHER than the one it was trained in, by cropping each
    # eval batch (predictions, truth and inputs alike) into the beam-following window
    # before any metric is computed. "" scores natively; "moving_ref" pushes a lab-frame
    # model into the moving frame, which is the only direction that loses nothing and is
    # what makes a lab arm and a moving-ref arm comparable on one yardstick. One eval is
    # one (model, target frame) pair, so the eval node itself carries which frame it is.
    eval_frame_transfer: str = ""
    eval_frame_behind: int = 32
    eval_frame_ahead: int = 16
    # frames averaged per case by eval_frame_transfer='steady' (the steady dataset's 30)
    eval_frame_avg: int = 30


@dataclass
class SplitSpec:
    """A train/validation index split for one fold."""

    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass
class DatasetBundle:
    """All per-fold datasets used by constraints, validation, and inference."""

    var_train: Optional[dict[str, np.ndarray]]
    var_val: Optional[dict[str, np.ndarray]]
    ds_input: Optional[dict[str, np.ndarray]]
    ds_output: Optional[dict[str, np.ndarray]]
    ds_input_val: Optional[dict[str, np.ndarray]]
    ds_output_val: Optional[dict[str, np.ndarray]]
    data_ds: Dataset
    val_data_ds: Dataset
    energy_ds: Optional[Dataset] = None
    val_energy_ds: Optional[Dataset] = None
    vof_ds: Optional[Dataset] = None
    val_vof_ds: Optional[Dataset] = None
    diff_ds: Optional[Dataset] = None
    val_diff_ds: Optional[Dataset] = None
    # Lazy AR rollout eval: stream one case's trajectory at a time instead of
    # the eager var_val the lazy path never materializes.
    val_frame_reader: Optional[Any] = None  # (case_idx, time_idx) -> processed frame dict
    val_case_indices: Optional[list[int]] = None
    val_case_lengths: Optional[dict[int, int]] = None


@dataclass
class ModelBundle:
    """Instantiated model and node lists for active training branches."""

    model: torch.nn.Module
    nn_nodes: list[Node]
    energy_nodes: Optional[list[Node]] = None
    vof_nodes: Optional[list[Node]] = None
    diff_nodes: Optional[list[Node]] = None


@dataclass
class InferenceMetrics:
    """Aggregate metric payload returned by inference/postprocessing."""

    errors: dict[str, dict[str, float]]
    val_rows: list[dict[str, Any]]
