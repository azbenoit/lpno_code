# pyright: reportAttributeAccessIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOptionalSubscript=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalMemberAccess=false
"""Configuration, runtime setup, and wandb bootstrap helpers for the pipeline."""

import copy
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from ...custom_utils.npz_datasource import derive_geometry_from_data, is_consolidated_npz
from ...custom_utils.utils import coerce_numbers
from ..cuda_bootstrap import SUPPORTED_CUDA_DEVICES

from .dataclasses import KeyConfig, RunConfig


# OneCycleLR raises "Tried to step N+1 times" if stepped past total_steps, so the
# cycle is sized a little beyond the run rather than exactly at it.
ONE_CYCLE_STEP_MARGIN = 200


def validate_recording_cadence(cfg, run_mode: str = "train") -> None:
    """Fail at startup for cadence mistakes that otherwise surface hours in or never.

    TRAINING only. `load_cfg_and_paths` is on every entry to the pipeline, eval
    included, and an eval never trains: it writes no checkpoints, records no
    validation rows and never reaches a stop criterion, so it cannot violate any
    of these invariants -- but it was refused by them anyway. Eval 68075
    (model 63797 `fno_rp2`) exited 300 on the stop_criterion rule after the model
    had loaded, and BOTH fixed-length random-path models (63795, 63797) archive
    the offending pair, so every evaluation of either was blocked over a setting
    that cannot affect a score. It presents as a broken model rather than a scope
    bug, which is the part that misleads. Defaults to "train" so the guard stays
    on when the caller says nothing.

    All three cases were hit in the 2026-08-11 PV-LHS launches:
      * rec_results_freq > max_steps: the constraint, monitor and inference records it
        drives (and the validation records, where rec_validation_freq is left to inherit
        it) are never written. It is NOT the checkpoint interval: physicsnemo writes the
        weights every `training.save_network_freq` steps, which
        `cap_save_network_freq` keeps within the run.
      * rec_validation_freq > max_steps: no validation rows, so checkpoint selection has
        nothing to select on.
      * stop_criterion.freq not a multiple of rec_validation_freq: physicsnemo raises
        for this, but only at the FIRST criterion evaluation -- a wave-1 arm trained
        ~2 hours before dying on it. The same check here costs nothing and fires in
        seconds.
    """
    if str(run_mode) != "train":
        return
    training = OmegaConf.select(cfg, "training") or {}
    max_steps = int(OmegaConf.select(cfg, "training.max_steps", default=0) or 0)
    if not max_steps:
        return
    results = OmegaConf.select(cfg, "training.rec_results_freq", default=None)
    validation = OmegaConf.select(cfg, "training.rec_validation_freq", default=None)
    if results is not None and int(results) > max_steps:
        raise ValueError(
            f"training.rec_results_freq={int(results)} > max_steps={max_steps}: the run "
            f"would never write its constraint, monitor or inference records (nor its "
            f"validation records where rec_validation_freq inherits this value). Set it "
            f"to at most max_steps."
        )
    if validation is not None and int(validation) > max_steps:
        raise ValueError(
            f"training.rec_validation_freq={int(validation)} > max_steps={max_steps}: "
            f"no validation would ever be recorded, so there is nothing to select a "
            f"checkpoint on."
        )
    stop_freq = OmegaConf.select(cfg, "stop_criterion.freq", default=None)
    if stop_freq is not None and validation is not None and int(validation) > 0:
        if int(stop_freq) % int(validation) != 0:
            raise ValueError(
                f"stop_criterion.freq={int(stop_freq)} is not a multiple of "
                f"training.rec_validation_freq={int(validation)}. physicsnemo enforces "
                f"this only at the first criterion evaluation -- hours into the run. "
                f"Align the two (a wave-1 arm died ~2 h in on exactly this)."
            )


def cap_save_network_freq(cfg, run_mode: str = "train") -> int | None:
    """Keep the weights-saving interval within the run, so the final weights are written.

    physicsnemo writes `u_network.0.pth` only on steps that are a multiple of
    `training.save_network_freq` (default 1000), whatever `rec_results_freq` says. A
    run shorter than that interval therefore ends with the weights of step 0 on disk,
    and its evaluation scores an untrained model without any warning (found with
    reproducibility/train_evaluate.ipynb: 200-step runs, default interval). Capped to
    `max_steps`, as `reproducibility/rerun/replay_overrides.py` and aiida-pino's train adapter cap it.

    An interval that does not divide `max_steps` still leaves the last steps unsaved;
    that is announced, not changed, since the interval is the caller's choice.
    Training only, like `validate_recording_cadence`. Returns the interval in force,
    or None when the config sets none.
    """
    if str(run_mode) != "train":
        return None
    max_steps = int(OmegaConf.select(cfg, "training.max_steps", default=0) or 0)
    freq = OmegaConf.select(cfg, "training.save_network_freq", default=None)
    if not max_steps or freq is None:
        return None
    freq = int(freq)
    if freq > max_steps:
        with open_dict(cfg):
            cfg.training.save_network_freq = max_steps
        print(
            f"training.save_network_freq: {freq} -> {max_steps} (capped to max_steps: "
            f"physicsnemo saves the weights only every save_network_freq steps, so a "
            f"longer interval would leave the step-0 weights on disk)",
            flush=True,
        )
        return max_steps
    if freq > 0 and max_steps % freq:
        print(
            f"WARNING: training.save_network_freq={freq} does not divide "
            f"max_steps={max_steps}; the last weights saved are step "
            f"{max_steps - max_steps % freq}'s, not the final step's "
            f"(u_network.best.pth is unaffected).",
            flush=True,
        )
    return freq


def fall_back_from_unbuildable_beambox_stop_metric(
    cfg, input_keys, moving_ref_frame: bool, run_mode: str = "train"
) -> str | None:
    """Stop on `mae_T` when the configured `beambox_*` stop metric can never be reported.

    The random-path configs, and the steady ones that inherit from them, stop on
    `beambox_mae_T`. Validation reports it only when a beam box is buildable from the
    input channels: a `beam_dist` footprint, the beam centre (`beam_x`/`beam_y`), or a
    beam-following frame (`moving_ref_frame=true`, beam at the origin). A steady run on
    `[V_scan, P, H_norm, x, y, z]` with `moving_ref_frame=false` has none of these, so
    physicsnemo's stop criterion found no such metric and the run died at its first
    validation (found with reproducibility/train_evaluate.ipynb). The best-checkpoint
    keeper already falls back to `mae_T` in that case; the stop criterion now does too,
    and says so. Returns the metric in force when it changed it, else None.
    """
    if str(run_mode) != "train":
        return None
    metric = OmegaConf.select(cfg, "stop_criterion.metric", default=None)
    if not isinstance(metric, str) or not metric.startswith("beambox_"):
        return None
    from .validation_beambox import beam_box_inputs_available

    keys = set(input_keys or [])
    if bool(_get_cfg_value(cfg, "fast_training", False)):
        return None                       # no validators at all: not this function's case
    if not bool(OmegaConf.select(cfg, "custom.validation_beambox", default=True)):
        return _set_stop_metric(cfg, metric, "custom.validation_beambox=false")
    if beam_box_inputs_available(lambda k: k in keys, at_origin=bool(moving_ref_frame)):
        return None
    return _set_stop_metric(
        cfg, metric,
        "no beam box is buildable from the input channels in this frame (needs beam_dist, "
        "beam_x/beam_y with x/y, or moving_ref_frame=true with x/y)",
    )


def _set_stop_metric(cfg, metric: str, why: str) -> str:
    fallback = metric[len("beambox_"):]
    with open_dict(cfg):
        cfg.stop_criterion.metric = fallback
    print(
        f"WARNING: stop_criterion.metric {metric} -> {fallback}: {why}, so validation never "
        f"reports {metric} and the stop criterion would fail at the first validation.",
        flush=True,
    )
    return fallback


def _refuse_fno_modes_beyond_served_grid(modes, grid, padding) -> None:
    """Refuse an FNO training whose modes the SERVED grid cannot hold.

    The Nyquist cap in `load_cfg_and_paths` reads the dataset's stored grid, while `PINO_LAZY_STRIDE` (and
    `PINO_LAZY_CROP_Z_TOP`) shrink the grid the model actually trains on at read time. physicsnemo's SpectralConv3d
    writes its low ([:m]) and high ([-m:]) mode blocks one after the other, so on a padded axis shorter than 2m they
    overlap and the later write wins: the overwritten weights never receive a gradient. That is how the Table 3 FNOs
    of 2026-10-31 (modes [20, 18, 8], padding 9, stride 2,2,1: y padded to 29) kept y-modes 11-17 of two blocks at
    their initial values, which then broke their full-resolution scoring. `PINO_ALLOW_FNO_MODE_OVERLAP=1` overrides.
    """
    if os.environ.get("PINO_ALLOW_FNO_MODE_OVERLAP", "").strip() == "1":
        return
    stride = (1, 1, 1)
    raw = os.environ.get("PINO_LAZY_STRIDE", "").strip()
    if raw:
        try:
            stride = tuple(max(1, int(v)) for v in raw.split(","))
        except ValueError:
            return  # malformed: the lazy store refuses it with its own message
        if len(stride) != 3:
            return
    crop = 0
    try:
        crop = max(0, int(os.environ.get("PINO_LAZY_CROP_Z_TOP", "0") or 0))
    except ValueError:
        pass
    served = [-(-int(n) // s) for n, s in zip(grid, stride)]
    served[2] = max(1, served[2] - crop)
    pads = [int(padding)] * 3 if isinstance(padding, (int, float)) else [int(p) for p in list(padding)[:3]]
    pads = (pads + [0, 0, 0])[:3]
    padded = [n + p for n, p in zip(served, pads)]
    bad = []
    for axis, (n, m) in enumerate(zip(padded[:2], modes[:2])):
        if n < 2 * int(m):
            bad.append(f"{'xy'[axis]}: {int(m)} modes on {n} padded points (at most {n // 2})")
    if int(modes[2]) > padded[2] // 2 + 1:
        bad.append(f"z: {int(modes[2])} modes on {padded[2]} padded points (at most {padded[2] // 2 + 1})")
    if bad:
        raise ValueError(
            f"[fno-modes] the served grid {served[0]}x{served[1]}x{served[2]} (stored {'x'.join(map(str, grid))}, "
            f"read-time stride {stride}, z-crop {crop}) cannot hold the requested FNO modes {list(modes)}: "
            + "; ".join(bad) + ". The low and high mode blocks would overlap and part of the weights would never "
            "train. Lower arch.fno.fno_modes on that axis or train without the stride "
            "(PINO_ALLOW_FNO_MODE_OVERLAP=1 overrides)."
        )


def _sync_one_cycle_total_steps(cfg) -> int | None:
    """Derive ``scheduler.total_steps`` from ``training.max_steps``.

    One-cycle only helps if the cosine actually finishes inside the run: the
    whole benefit is the deep anneal at the end, which is what pins the final
    checkpoint to the run's best instead of leaving it mid-oscillation
    (straight-path GAOT, 2026-07-28: last-10 validation spread 2.52 K under
    ``tf_exponential_lr`` versus 0.28 K under a completed one-cycle).

    A hand-set ``total_steps`` silently breaks that whenever ``max_steps``
    changes -- reusing a config written for 32000 steps in a 20000-step run
    leaves the schedule only ~60% down the cosine, at a learning rate *higher*
    than the exponential schedule would have reached, which makes the run worse
    than not using one-cycle at all. Deriving it here means the two can never
    disagree, whatever sets ``max_steps`` (config file, CLI override, or the
    aiida training profile).

    Returns the value written, or ``None`` when the scheduler is not one-cycle.
    """
    target = str(OmegaConf.select(cfg, "scheduler._target_", default="") or "")
    if not target.endswith("OneCycleLR"):
        return None
    max_steps = OmegaConf.select(cfg, "training.max_steps", default=None)
    if max_steps is None:
        return None
    derived = int(max_steps) + ONE_CYCLE_STEP_MARGIN
    previous = OmegaConf.select(cfg, "scheduler.total_steps", default=None)
    with open_dict(cfg):
        cfg.scheduler.total_steps = derived
    if previous is not None and int(previous) != derived:
        print(
            f"scheduler.total_steps: {previous} -> {derived} "
            f"(derived from training.max_steps={int(max_steps)} "
            f"+ {ONE_CYCLE_STEP_MARGIN} margin; a one-cycle schedule must finish "
            "inside the run or its anneal never completes)"
        )
    else:
        print(f"scheduler.total_steps = {derived} (derived from max_steps={int(max_steps)})")
    return derived


# Source fields that only exist to back specific model keys. A field listed here is
# loaded ONLY when something actually consumes it; anything not listed is always
# kept, so an unrecognised field is never dropped by accident.
_OPTIONAL_VTK_FIELDS = {
    "Velocity": {"u", "v", "w", "U"},
    "X-velocity": {"u", "U"},
    "Y-velocity": {"v", "U"},
    "Z-velocity": {"w", "U"},
    "Pressure": {"p"},
}


def _prune_unused_vtk_fields(vtk_fields, str_input_keys, str_output_keys,
                             *, pde_loss: bool, vof_pde_loss: bool,
                             diff_loss: bool, infer_velocities: bool):
    """Load only the source fields something consumes.

    The lazy per-frame path decompresses EVERY configured field for the whole case
    on each cache miss (`npz_lazy_store.py`), so a field no key reads is pure cost:
    measured 2026-07-28, Velocity (5.96 s) and Pressure (1.76 s) were 58% of a
    14.69 s per-case inflate on the straight-path data, built and then discarded.

    Deliberately conservative -- a field is dropped only when ALL of these hold:
      * it is in `_OPTIONAL_VTK_FIELDS` (an unknown field is always kept);
      * none of the model keys it backs appear in the input or output keys;
      * no PDE/VOF/Diff residual loss is on (those consume u/v/w through
        `Energy_input_keys`/`VOF_input_keys`);
      * `custom.infer_velocities` is off.
    Anything pruned is logged, so a surprising drop is visible in the run output
    rather than silent.
    """
    if not vtk_fields:
        return vtk_fields, []
    if pde_loss or vof_pde_loss or diff_loss or infer_velocities:
        return list(vtk_fields), []
    consumed = set(str_input_keys) | set(str_output_keys)
    kept, dropped = [], []
    for field in vtk_fields:
        backs = _OPTIONAL_VTK_FIELDS.get(str(field))
        if backs is None or (backs & consumed):
            kept.append(field)
        else:
            dropped.append(str(field))
    return kept, dropped


def _warn_if_amp_with_rigno_edge_masking(cfg, model_family: str, run_mode: str) -> bool:
    """Warn when a RIGNO-family training runs under fp16 autocast.

    On this data family fp16 autocast can overflow the gradients so that the
    grad scaler backs off on every step and physicsnemo skips each optimizer
    update; the run then completes with its initial weights. The same training
    is finite in fp32. Edge masking was first suspected as the trigger, but the
    overflow also occurs without it, so the warning fires for any rigno-family
    training with amp on. A warning, so a deliberate experiment on other data
    stays possible. Returns whether it warned (for tests). See
    potential_bugs.md, BUG-43a.
    """
    if model_family not in ("rigno", "rigno_refined") or run_mode != "train":
        return False
    if not bool(OmegaConf.select(cfg, "amp.enabled", default=False)):
        return False
    # Same default as model_domain: an omitted p_edge_masking means 0.5.
    p_mask = float(OmegaConf.select(cfg, "arch.rigno.p_edge_masking", default=0.5))
    if p_mask > 0.0:
        detail = f"with arch.rigno.p_edge_masking={p_mask}.\n"
    else:
        detail = "on a rigno-family training.\n"
    detail += (
        "fp16 autocast can overflow the gradients on this data family, with or "
        "without edge\nmasking: the grad scaler then backs off on every step, "
        "physicsnemo skips each\noptimizer update, and the run completes "
        "without training anything."
    )
    print(
        "=" * 78 + "\n"
        f"WARNING (fp16 autocast on RIGNO): amp.enabled=true {detail}\n"
        "Set amp.enabled=false (fp32 is verified finite on this family). See\n"
        "potential_bugs.md, BUG-43a.\n"
        + "=" * 78
    )
    return True


def _warn_if_one_cycle_can_be_truncated(cfg, run_mode: str) -> bool:
    """A one-cycle run that early-stops leaves the LR anneal unfinished.

    One-cycle only produces its trained model if the schedule completes: stop
    it early and the weights sit mid-schedule at an elevated LR. On 2026-08-08
    `rigno2579_linall_oc` stopped at step 21500 of 32000 on patience 4000 and
    registered as "done" with a partially annealed model. Exponential decay
    degrades gracefully, so this is specific to one-cycle. Warning, not an
    error: a deliberately short exploratory run is legitimate.
    """
    if run_mode != "train":
        return False
    # same detection as _sync_one_cycle_total_steps: the composed config carries
    # _target_, not a friendly name (checking _name_ alone never fires in practice)
    target = str(OmegaConf.select(cfg, "scheduler._target_", default="") or "")
    name = str(OmegaConf.select(cfg, "scheduler._name_", default="") or "")
    if not (target.endswith("OneCycleLR") or "one_cycle" in name):
        return False
    patience = OmegaConf.select(cfg, "stop_criterion.patience", default=None)
    max_steps = OmegaConf.select(cfg, "training.max_steps", default=None)
    if patience is None or max_steps is None or float(patience) >= float(max_steps):
        return False
    print(
        "=" * 78 + "\n"
        f"WARNING: one-cycle schedule with stop_criterion.patience={patience} < "
        f"training.max_steps={max_steps}.\n"
        "Early stopping can fire before the anneal completes, leaving the weights "
        "mid-\nschedule at an elevated LR (rigno2579_linall_oc stopped at 21500 of "
        "32000,\n2026-08-08, and registered as done). Raise stop_criterion.patience "
        "above\nmax_steps, or accept a partially annealed model deliberately.\n"
        + "=" * 78
    )
    return True


def _geometry_wants_derivation(grid_params) -> bool:
    """Whether the config asks to derive the domain from data (``geometry``/``resolution: auto``)."""
    if grid_params is None:
        return True
    if isinstance(grid_params, str):
        return grid_params.strip().lower() == "auto"
    try:
        resolution = grid_params.get("resolution")
    except AttributeError:
        return False
    return isinstance(resolution, str) and resolution.strip().lower() == "auto"


def _grid_params_for_data(grid_params, data_directory_path, software: str):
    """The grid `run_cfg` should describe for this data, and whether the moving-frame
    x-window is already baked into it.

    Returns ``(grid_params, window_baked)``. Derives from the data for a consolidated
    NPZ, a staged arraydir runtime, or a config that asks (`geometry`/`resolution: auto`);
    keeps the config box otherwise (VTK: a curated crop). The arraydir case is the one
    that used to fall through: the loader serves those arrays untouched, so the config's
    `[90, 40, 30]` reached `run_cfg` while the tensors were 61x60x55 (anchorbox steady
    arms, 2026-09-01) and every reader of `n{x,y,z}_effective` -- validator slice
    indices, the FNO Nyquist cap, bounds/dx -- described a grid the data did not have.
    """
    from lpno.models.pipeline.lazy_runtime import (
        is_arraydir_runtime,
        runtime_window_is_baked,
    )

    is_arraydir = bool(data_directory_path) and is_arraydir_runtime(data_directory_path)
    if _geometry_wants_derivation(grid_params) or is_consolidated_npz(data_directory_path) or is_arraydir:
        grid_params = derive_geometry_from_data(data_directory_path, software)
        print(f"[geometry] Derived rectangular domain from data: {grid_params}")
    window_baked = is_arraydir and runtime_window_is_baked(data_directory_path)
    if window_baked:
        print("[geometry] Staged runtime already carries the moving-frame x-window; "
              "no x shift applied at config time.")
    return grid_params, window_baked


_DATA_ROOT_ENV_VAR = "PINO_SIMULATIONS_DATA_ROOT"
_DEFAULT_PATH_SENTINEL = "DEFAULT"
# An optional default path whose data root could not be resolved. Deliberately falsy
# and a str (RunConfig.param_dir is typed str), so it flows to the read site and is
# refused there rather than at config load. See _resolve_default_data_path.
_UNRESOLVED_DATA_PATH = ""
_DEFAULT_TRAIN_DATA_SUBDIR = Path("flow3D/original_straight_path_data/train")
_DEFAULT_VAL_DATA_SUBDIR = Path("flow3D/original_straight_path_data/val")
_DEFAULT_TEST_DATA_SUBDIR = Path("flow3D/original_straight_path_data/test")
_DEFAULT_PARAM_DATA_SUBDIR = Path("abaqus_data/Ti64_params")
_LEGACY_TIME_CONFIG_PATHS = (
    "infer_time",
    "custom.infer_time",
    "time_sampling_mode",
    "custom.time_sampling_mode",
)


def _discover_repo_data_root() -> Path | None:
    """Return the checkout-local ``data`` directory when running from a source tree."""

    repo_root = Path(__file__).resolve().parents[4]
    data_root = repo_root / "data"
    if data_root.is_dir():
        return data_root
    return None


def _resolve_default_data_root() -> Path:
    """Resolve the default data root from env var or an adjacent source checkout."""

    env_value = os.environ.get(_DATA_ROOT_ENV_VAR)
    if env_value:
        data_root = Path(env_value).expanduser()
        if not data_root.is_dir():
            raise ValueError(
                f"{_DATA_ROOT_ENV_VAR} is set to '{data_root}', but that directory does not exist."
            )
        return data_root

    repo_data_root = _discover_repo_data_root()
    if repo_data_root is not None:
        return repo_data_root

    raise ValueError(
        "Default dataset paths could not be resolved. Set custom.data_dir/custom.val_data_dir/custom.param_dir "
        f"explicitly, or set {_DATA_ROOT_ENV_VAR} to the repo data directory."
    )


def _resolve_default_data_path(relative_path: Path, *, label: str, require_exists: bool = True) -> str:
    """Resolve a default dataset path under the configured data root.

    ``require_exists=False`` resolves the path without asserting it exists — used
    for optional inputs (``param_dir`` is only read for piecewise-tabulated
    ``c(T)`` materials; existence is validated at read time instead).
    """

    try:
        data_root = _resolve_default_data_root()
    except ValueError:
        if not require_exists:
            # OPTIONAL path on a machine with no data root -- defer, do not kill the
            # run. A compute box receives its datasets through AiiDA staging and has
            # no repo `data/` tree; requiring the root here killed runs that pass
            # custom.data_dir and custom.val_data_dir explicitly and never read this
            # path at all (reported from a compute box, 2026-08-19). The read site raises if
            # anything actually reads it.
            return _UNRESOLVED_DATA_PATH
        raise

    resolved_path = data_root / relative_path
    if require_exists and not resolved_path.exists():
        raise ValueError(
            f"Resolved default {label} path does not exist: '{resolved_path}'. "
            f"Set the path explicitly or update {_DATA_ROOT_ENV_VAR}."
        )
    return str(resolved_path)


def _uses_default_data_path(value: Any) -> bool:
    """Return whether a config path value should defer to default resolution."""

    if value is None:
        return True
    if isinstance(value, str) and value.strip().upper() == _DEFAULT_PATH_SENTINEL:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _resolve_configured_data_path(value: Any, *, relative_path: Path, label: str,
                                  require_exists: bool = True) -> str:
    """Resolve a configured path, falling back to the default resolver when requested."""

    if _uses_default_data_path(value):
        return _resolve_default_data_path(relative_path, label=label, require_exists=require_exists)
    return str(value)


def configure_device(cfg: PhysicsNeMoConfig, set_device: bool = True) -> None:
    """Apply CUDA visibility and optionally select the active CUDA device."""

    requested_device = int(_get_cfg_value(cfg, "cuda_device", 0))
    if requested_device not in SUPPORTED_CUDA_DEVICES:
        supported = " or ".join(f"cuda_device={d}" for d in SUPPORTED_CUDA_DEVICES)
        raise SystemExit(
            f"Unsupported cuda_device={requested_device}. Use {supported}."
        )

    # If cuda_visible_devices is not explicitly set, map requested physical GPU
    # to a single visible device and use logical device 0.
    visible_devices = _get_cfg_value(cfg, "cuda_visible_devices", None)
    if visible_devices is None:
        # An externally-set CUDA_VISIBLE_DEVICES is a decision already
        # made -- a Slurm allocation, or gpu_spool pinning the seat it leased --
        # and cuda_bootstrap preserves it on purpose. Overwriting it with
        # str(cuda_device) sent every seat to GPU 0, since every shipped config
        # carries cuda_device: 0. An explicit config value below still wins.
        pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
        if pinned:
            print(f"[device] honouring CUDA_VISIBLE_DEVICES={pinned} set by the caller")
            logical_device = int(_get_cfg_value(cfg, "cuda_local_device", 0))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_device)
            logical_device = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        logical_device = int(_get_cfg_value(cfg, "cuda_local_device", requested_device))

    if not set_device:
        return

    import torch

    if not torch.cuda.is_available():
        # Refused by default: a training silently on the CPU takes days and reads as a hang.
        # An evaluation is minutes of CPU, and PINO_ALLOW_CPU=1 is how a reader without a GPU
        # re-scores a deposited model (2026-10-08, reproduction from the deposit on a CPU cluster).
        if os.environ.get("PINO_ALLOW_CPU", "").strip().lower() in ("1", "true", "yes"):
            print("[device] no CUDA device; running on the CPU (PINO_ALLOW_CPU=1). Fine for an "
                  "evaluation or a short training; a full training run would take days.", flush=True)
            # A CPU-only torch has no NVTX, which physicsnemo calls on every step.
            from .nvtx_guard import make_nvtx_noop_on_cpu_build

            make_nvtx_noop_on_cpu_build()
            return
        raise SystemExit(
            "CUDA is not available. Aborting instead of running on CPU. "
            "Check driver/runtime installation and CUDA_VISIBLE_DEVICES; "
            "set PINO_ALLOW_CPU=1 to run on the CPU anyway (an evaluation, or a short training)."
        )

    device_count = torch.cuda.device_count()
    if logical_device < 0 or logical_device >= device_count:
        raise SystemExit(
            f"Configured CUDA device index {logical_device} is out of range for "
            f"{device_count} visible CUDA device(s)."
        )
    torch.cuda.set_device(logical_device)



def _reject_lab_frame_keys_in_moving_frame(input_keys, moving_ref_frame: bool) -> None:
    """Fail early when lab-frame trajectory channels are requested in the moving frame.

    The random-path recipe passes the beam trajectory explicitly -- `beam_x`, `beam_y`,
    `beam_dist`, `path_dose*`, `beam_s`, `beam_n`, `beam_speed`. Those exist only in the
    LAB frame. With `moving_ref_frame=true` the grid is re-cropped and shifted to follow
    the beam, so the beam is always in the same place by construction and the datasource
    never builds them.

    Requesting both is silently inconsistent until the training is well underway, where
    it surfaces as a bare `KeyError: 'beam_x'` from
    `preprocessing_datasets._builder`'s `{key: processed[key] for key in ...}` -- far
    from the setting that caused it. Measured 2026-08-05: the first varspeed GAOT
    submission died this way after a full dataset stage and graph build, because the
    profile listed the channels but inherited `moving_ref_frame=true` from the
    straight-path config it was based on.

    Two settings that must agree, with nothing enforcing it. This enforces it, at config
    load, naming the fix.
    """
    from ...custom_utils.npz_datasource import PATH_CHANNEL_NAMES

    if not moving_ref_frame:
        return
    offenders = sorted(set(input_keys) & set(PATH_CHANNEL_NAMES))
    if offenders:
        # Downgraded from a hard failure 2026-08-19, on measurement plus a user
        # decision that the steady/moving-frame arms should inherit the SAME input
        # keys as their lab-frame partners even where a channel goes near-constant,
        # so the three frame treatments stay comparable and warm-startable.
        #
        # The docstring above is stale for the consolidated-NPZ path, which is what
        # every current dataset uses. Verified by running it: with
        # moving_ref_frame=True, avg_window_size=30 and moving_frame_steps=30,
        # `load_consolidated_npz` returns beam_x, beam_y, beam_dist, path_dose and
        # path_dose_recent at the SAME shape as Temperature. They are built at
        # npz_datasource.py:1230 and then cropped and shifted with every other key
        # by `_apply_moving_frame` at :1249, so they stay consistent with the field
        # data rather than being dropped.
        #
        # There are THREE paths, not two, and the old text named only two of them
        # (2026-08-21). It told the reader to ask "does this run read VTK?" -- so
        # `fno_spkh123_steady_32k_added14ch` on ds55011, a FLOW-3D NPZ dataset,
        # answered no, proceeded, and died with `KeyError: 'beam_dist'` exactly as
        # predicted but for a reason the warning had excluded:
        #
        #   load_consolidated_npz               BUILDS them (see the note above)
        #   legacy VTK / OpenFOAM               does not build them
        #   NpzDirStore / staged derived member does not, BY DESIGN
        #
        # The third is deliberate, not a gap: `lazy_runtime.py:68-70` and :413 --
        # the trajectory channels "need a per-frame beam path a staged derived
        # member does not carry, and must fail by name rather than be
        # approximated". So the KeyError is correct behaviour and only the
        # diagnostic was wrong.
        #
        # The condition is therefore NOT the storage format; it is whether the
        # dataset's members carry a per-frame beam path. This function is given
        # only the key list and the flag, so it cannot test that here -- it names
        # the condition instead of guessing at a path.
        print(
            f"WARNING: custom.FNO_input_keys requests trajectory channels {offenders} "
            "with custom.moving_ref_frame=true. Whether that works depends on the "
            "PER-FRAME BEAM PATH, not on the file format:\n"
            "  - consolidated NPZ with a laser path: built and cropped with the "
            "fields; some go near-constant because the beam is fixed by "
            "construction -- redundant, not wrong.\n"
            "  - a STAGED/derived member or any dataset whose manifests carry no "
            "laser path (e.g. a var_1_time_noref collection): NOT built, by design "
            "(lazy_runtime.py:68-70). Training dies later with a bare KeyError on "
            "one of these channels, and the moving window itself fails with "
            "'[NpzDirStore] moving_window needs per-frame beam centres; the case "
            "manifest has no laser path'.\n"
            "  - legacy VTK/OpenFOAM: NOT built either.\n"
            "Check the dataset's manifests for a per-frame beam path; if they have "
            "none, drop these channels from FNO_input_keys or use a dataset that "
            "carries the path.",
            flush=True,
        )

def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""

    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_cfg_value(cfg: PhysicsNeMoConfig, key: str, default: Any = None) -> Any:
    """Fetch ``custom.<key>`` first, then top-level ``<key>``, then ``default``."""

    missing = object()
    custom_value = OmegaConf.select(cfg, f"custom.{key}", default=missing)
    if custom_value is not missing:
        return custom_value
    top_level_value = OmegaConf.select(cfg, key, default=missing)
    if top_level_value is not missing:
        return top_level_value
    return default


def _build_key_config(
    yaml_params: dict[str, Any],
    add_t_hat: bool = False,
    add_velocity_outputs: bool = False,
    drop_y_input: bool = False,
    state_prev_suffix: str | None = None,
    extra_input_keys: list[str] | None = None,
) -> KeyConfig:
    """Build typed input/output key lists from resolved simulation config."""

    str_fno_input_keys = list(yaml_params.get("FNO_input_keys", []))
    if drop_y_input:
        str_fno_input_keys = [key for key in str_fno_input_keys if key != "y"]
    if add_t_hat and "t_hat" not in str_fno_input_keys:
        str_fno_input_keys.append("t_hat")
    str_fno_output_keys = list(yaml_params.get("FNO_output_keys", []))
    if add_velocity_outputs:
        for velocity_key in ("u", "v", "w"):
            if velocity_key not in str_fno_output_keys:
                str_fno_output_keys.append(velocity_key)
    if state_prev_suffix:
        for state_key in str_fno_output_keys:
            state_prev_key = f"{state_key}{state_prev_suffix}"
            if state_prev_key not in str_fno_input_keys:
                str_fno_input_keys.append(state_prev_key)
    for input_key in extra_input_keys or []:
        if input_key not in str_fno_input_keys:
            str_fno_input_keys.append(input_key)
    str_energy_input_keys = yaml_params.get("Energy_input_keys", []) + str_fno_input_keys
    str_energy_output_keys = yaml_params.get("Energy_output_keys", [])
    str_vof_input_keys = yaml_params.get("VOF_input_keys", []) + str_fno_input_keys
    str_vof_output_keys = yaml_params.get("VOF_output_keys", [])
    str_diff_input_keys = yaml_params.get("Diff_input_keys", []) + str_fno_input_keys
    str_diff_output_keys = yaml_params.get("Diff_output_keys", [])

    return KeyConfig(
        fno_input_keys=[Key(k) for k in str_fno_input_keys],
        str_fno_input_keys=str_fno_input_keys,
        fno_output_keys=[Key(k) for k in str_fno_output_keys],
        str_fno_output_keys=str_fno_output_keys,
        energy_input_keys=[Key(k) for k in str_energy_input_keys],
        str_energy_input_keys=str_energy_input_keys,
        energy_output_keys=[Key(k) for k in str_energy_output_keys],
        str_energy_output_keys=str_energy_output_keys,
        vof_input_keys=[Key(k) for k in str_vof_input_keys],
        str_vof_input_keys=str_vof_input_keys,
        vof_output_keys=[Key(k) for k in str_vof_output_keys],
        str_vof_output_keys=str_vof_output_keys,
        diff_input_keys=[Key(k) for k in str_diff_input_keys],
        str_diff_input_keys=str_diff_input_keys,
        diff_output_keys=[Key(k) for k in str_diff_output_keys],
        str_diff_output_keys=str_diff_output_keys,
    )


def _get_wandb_project(software: str) -> str:
    """Map software backend name to a wandb project."""

    if software == "openFoam":
        return "NS_PINO"
    if software == "flow3D":
        return "NS_PINO_flow3D"
    if software == "abaqus":
        return "heat_diffusion_PINO"
    return "NS_PINO"


_SIM_PARAM_OVERRIDE_KEYS = (
    "software",
    "geometry",
    "ref_scales",
    "laser",
    "material_params",
    "vtk_fields",
    "FNO_input_keys",
    "FNO_output_keys",
    "Energy_input_keys",
    "Energy_output_keys",
    "VOF_input_keys",
    "VOF_output_keys",
    "Diff_input_keys",
    "Diff_output_keys",
)


def _collect_sim_param_overrides(cfg: PhysicsNeMoConfig) -> dict[str, Any]:
    """Collect simulation fields from ``custom`` (with top-level compatibility fallback)."""

    overrides: dict[str, Any] = {}
    missing = object()
    for key in _SIM_PARAM_OVERRIDE_KEYS:
        value = OmegaConf.select(cfg, f"custom.{key}", default=missing)
        if value is missing:
            value = OmegaConf.select(cfg, key, default=missing)
        if value is missing:
            continue
        if OmegaConf.is_config(value):
            overrides[key] = OmegaConf.to_container(value, resolve=True)
        else:
            overrides[key] = copy.deepcopy(value)
    return overrides


def _resolve_sim_config(cfg: PhysicsNeMoConfig) -> dict[str, Any]:
    """Resolve simulation config from ``custom`` and reject deprecated sim_params."""

    sim_cfg = _collect_sim_param_overrides(cfg)
    if OmegaConf.select(cfg, "sim_params", default=None) is not None or OmegaConf.select(
        cfg, "custom.sim_params", default=None
    ) is not None:
        raise ValueError(
            "The 'sim_params' override is deprecated and no longer supported. "
            "Move simulation fields into Hydra custom.* instead."
        )
    return coerce_numbers(sim_cfg)


def _parse_explicit_cli_override_paths(argv: list[str] | None = None) -> tuple[set[str], set[str]]:
    """Collect explicit Hydra override paths from CLI tokens."""

    args = list(sys.argv[1:] if argv is None else argv)
    explicit_set_paths: set[str] = set()
    explicit_delete_paths: set[str] = set()

    for arg in args:
        if not arg or arg.startswith("--"):
            continue
        if arg.startswith("~"):
            key = arg[1:].strip()
            if key:
                explicit_delete_paths.add(key)
            continue
        if "=" not in arg:
            continue
        key = arg.split("=", 1)[0].strip()
        if not key:
            continue
        key = key.lstrip("+")
        if key.startswith("~"):
            key = key[1:]
            if key:
                explicit_delete_paths.add(key)
            continue
        if key:
            explicit_set_paths.add(key)

    return explicit_set_paths, explicit_delete_paths


def _delete_cfg_path(cfg: DictConfig | ListConfig, path: str) -> None:
    """Delete a dotted path from DictConfig/ListConfig when present."""

    if not path:
        return

    tokens = [token for token in path.split(".") if token]
    if not tokens:
        return

    node: Any = cfg
    for token in tokens[:-1]:
        if isinstance(node, DictConfig):
            if token not in node:
                return
            node = node[token]
            continue
        if isinstance(node, ListConfig):
            try:
                index = int(token)
            except ValueError:
                return
            if index < 0 or index >= len(node):
                return
            node = node[index]
            continue
        return

    leaf = tokens[-1]
    if isinstance(node, DictConfig):
        if leaf in node:
            with open_dict(node):
                del node[leaf]
    elif isinstance(node, ListConfig):
        try:
            index = int(leaf)
        except ValueError:
            return
        if 0 <= index < len(node):
            del node[index]


def _coerce_bool(value: Any) -> bool:
    """Convert common CLI/config representations to boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
    return bool(value)


def _legacy_time_mode_to_batch_flag(mode_value: Any) -> bool:
    """Map legacy time_sampling_mode to time_as_batch boolean."""

    mode = str(mode_value).strip().lower()
    if mode not in ("avg_last", "per_timestep"):
        raise ValueError(
            f"Unsupported legacy time_sampling_mode: {mode}. Expected one of: avg_last, per_timestep."
        )
    return mode == "per_timestep"


def _assert_no_legacy_time_config_keys(
    cfg: PhysicsNeMoConfig,
    explicit_set_paths: set[str],
    explicit_delete_paths: set[str],
) -> None:
    """Reject legacy temporal keys in active user config/CLI."""

    missing = object()
    offenders: list[str] = []

    for path in _LEGACY_TIME_CONFIG_PATHS:
        if OmegaConf.select(cfg, path, default=missing) is not missing:
            offenders.append(path)

    for path in sorted(explicit_set_paths | explicit_delete_paths):
        if path in _LEGACY_TIME_CONFIG_PATHS:
            offenders.append(f"CLI:{path}")

    if offenders:
        deduped = ", ".join(dict.fromkeys(offenders))
        raise ValueError(
            "Legacy temporal config keys are no longer supported: "
            f"{deduped}. Use custom.time_as_batch instead."
        )


def _migrate_legacy_time_keys_in_model_cfg(model_cfg: DictConfig) -> bool:
    """Migrate legacy model .hydra temporal keys to custom.time_as_batch."""

    missing = object()
    migrated = False

    time_as_batch_value = OmegaConf.select(model_cfg, "custom.time_as_batch", default=missing)
    if time_as_batch_value is missing:
        for path, converter in (
            ("custom.infer_time", _coerce_bool),
            ("infer_time", _coerce_bool),
            ("custom.time_sampling_mode", _legacy_time_mode_to_batch_flag),
            ("time_sampling_mode", _legacy_time_mode_to_batch_flag),
        ):
            legacy_value = OmegaConf.select(model_cfg, path, default=missing)
            if legacy_value is missing:
                continue
            OmegaConf.update(
                model_cfg,
                "custom.time_as_batch",
                converter(legacy_value),
                merge=False,
                force_add=True,
            )
            migrated = True
            break

    for path in _LEGACY_TIME_CONFIG_PATHS:
        if OmegaConf.select(model_cfg, path, default=missing) is not missing:
            _delete_cfg_path(model_cfg, path)
            migrated = True

    return migrated


def _find_eval_hydra_config_path(network_dir: str | Path) -> Path | None:
    """Find composed Hydra config in eval model directory, parent, or grandparent."""

    run_dir = Path(network_dir)
    candidates = [
        run_dir / ".hydra" / "config.yaml",
        run_dir.parent.parent / ".hydra" / "config.yaml",
        run_dir.parent / ".hydra" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _has_explicit_cli_override(explicit_set_paths: set[str], key: str) -> bool:
    """Check whether ``key`` was explicitly overridden on CLI (top-level or custom)."""

    return key in explicit_set_paths or f"custom.{key}" in explicit_set_paths


def _apply_masked_patch_runtime_defaults(
    cfg: PhysicsNeMoConfig,
    explicit_set_paths: set[str],
) -> None:
    """Apply safer defaults for masked-patch autoregressive runs unless explicitly overridden."""

    with open_dict(cfg):
        if "training.grad_clip_max_norm" not in explicit_set_paths:
            current_clip = OmegaConf.select(cfg, "training.grad_clip_max_norm", default=0.5)
            OmegaConf.update(
                cfg,
                "training.grad_clip_max_norm",
                min(float(current_clip), 0.1),
                force_add=True,
            )
        if "custom.alpha_masking" not in explicit_set_paths:
            OmegaConf.update(cfg, "custom.alpha_masking", False, force_add=True)


def _apply_eval_model_hydra_defaults(
    cfg: PhysicsNeMoConfig,
    explicit_set_paths: set[str],
    explicit_delete_paths: set[str],
) -> PhysicsNeMoConfig:
    """Apply trained-run Hydra config as eval defaults while preserving explicit CLI overrides."""

    original_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    eval_model_dir = cfg.get("network_dir", "PATH")
    hydra_cfg_path = _find_eval_hydra_config_path(eval_model_dir)
    if hydra_cfg_path is None:
        print(
            "Warning: Eval defaults not loaded from model .hydra config; "
            f"no config.yaml found under '{eval_model_dir}', its parent, or its grandparent."
        )
        return original_cfg

    try:
        model_cfg = OmegaConf.load(hydra_cfg_path)
    except Exception as exc:  # noqa: BLE001 - keep load error details visible.
        print(f"Warning: Failed to load eval defaults from '{hydra_cfg_path}': {exc}")
        return original_cfg
    print(f"Eval mode: loading defaults from model Hydra config: {hydra_cfg_path}")

    model_cfg_sanitized = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=False))
    try:
        migrated_legacy = _migrate_legacy_time_keys_in_model_cfg(model_cfg_sanitized)
    except ValueError as exc:
        print(f"Warning: Failed to migrate legacy temporal keys from '{hydra_cfg_path}': {exc}")
        return original_cfg
    if migrated_legacy:
        print(
            "Warning: Migrated legacy model temporal keys to custom.time_as_batch "
            f"while loading eval defaults from '{hydra_cfg_path}'."
        )
    for path in ("network_dir", "run_mode", "hydra", "custom.data_dir", "custom.val_data_dir"):
        _delete_cfg_path(model_cfg_sanitized, path)

    merged_cfg = OmegaConf.merge(original_cfg, model_cfg_sanitized)

    missing = object()
    for path in explicit_set_paths:
        value = OmegaConf.select(original_cfg, path, default=missing)
        if value is missing:
            continue
        if OmegaConf.is_config(value):
            value_to_apply: Any = OmegaConf.to_container(value, resolve=False)
        else:
            value_to_apply = copy.deepcopy(value)
        OmegaConf.update(merged_cfg, path, value_to_apply, merge=False, force_add=True)

    for path in explicit_delete_paths:
        if OmegaConf.select(original_cfg, path, default=missing) is missing:
            _delete_cfg_path(merged_cfg, path)

    return merged_cfg



def apply_t_boil_override(material_params: dict, t_boil_k) -> None:
    """Put `custom.T_boil_k` into material_params["T_boil"], the door everything reads.

    The mask fill, the MaskNode and the thermal-headroom metrics all go through
    `boiling_temperature(material_params)`; injecting here keeps them coherent
    (a second literal is how they once disagreed). Absent -> unchanged
    (legacy 3123.0 fallback), so existing arms are untouched.
    """
    if t_boil_k is None:
        return
    value = float(t_boil_k)
    if not value > 0:
        raise ValueError(f"custom.T_boil_k must be a positive temperature in K, got {t_boil_k!r}.")
    material_params["T_boil"] = value

def load_cfg_and_paths(cfg: PhysicsNeMoConfig, default_split_mode: str = "standard") -> RunConfig:
    """Resolve Hydra ``custom`` runtime config into a fully prepared ``RunConfig``."""

    explicit_set_paths, explicit_delete_paths = _parse_explicit_cli_override_paths()
    _assert_no_legacy_time_config_keys(cfg, explicit_set_paths, explicit_delete_paths)
    eval_only = cfg.get("run_mode", "train") == "eval"
    if eval_only:
        cfg = _apply_eval_model_hydra_defaults(cfg, explicit_set_paths, explicit_delete_paths)
    _assert_no_legacy_time_config_keys(cfg, set(), set())

    yaml_params = _resolve_sim_config(cfg)
    model_family = str(_get_cfg_value(cfg, "model_family", "fno")).strip().lower()
    if model_family not in ("fno", "unet", "fully_connected", "deeponet", "rigno", "rigno_refined", "gaot"):
        raise ValueError(
            f"Unsupported model_family='{model_family}'. Expected one of: "
            "fno, unet, fully_connected, deeponet, rigno, rigno_refined, gaot."
        )
    _warn_if_amp_with_rigno_edge_masking(cfg, model_family, str(cfg.get("run_mode", "train")))
    _warn_if_one_cycle_can_be_truncated(cfg, str(cfg.get("run_mode", "train")))
    data_layout = "grid" if model_family in ("fno", "unet", "rigno", "rigno_refined", "gaot") else "pointwise"

    grid_params = copy.deepcopy(yaml_params.get("geometry"))
    ref_scales = copy.deepcopy(yaml_params.get("ref_scales"))
    laser_params = copy.deepcopy(yaml_params.get("laser", {}))
    material_params = copy.deepcopy(yaml_params.get("material_params", {}))
    apply_t_boil_override(material_params, _get_cfg_value(cfg, "T_boil_k", None))
    software = yaml_params.get("software", "openFoam")
    vtk_fields = yaml_params.get("vtk_fields", None)
    include_time = bool(_get_cfg_value(cfg, "include_time", software == "flow3D"))
    use_2d = bool(_get_cfg_value(cfg, "use_2D", _get_cfg_value(cfg, "2D", False)))
    time_as_dim = bool(_get_cfg_value(cfg, "time_as_dim", False))
    fast_training = bool(_get_cfg_value(cfg, "fast_training", False))
    if time_as_dim:
        # PhysicsNeMo currently supports up to 3D FNO, so time-as-dimension requires slicing out y.
        use_2d = True
    if time_as_dim and not include_time:
        raise ValueError("time_as_dim=True requires include_time=True.")
    time_as_batch_cfg = OmegaConf.select(cfg, "custom.time_as_batch", default=None)
    if time_as_batch_cfg is None:
        time_as_batch_cfg = _get_cfg_value(cfg, "time_as_batch", False)
    time_as_batch = _coerce_bool(time_as_batch_cfg)
    if time_as_batch and not include_time:
        raise ValueError("time_as_batch=True requires include_time=True.")
    if time_as_batch and time_as_dim:
        raise ValueError("time_as_batch=True is incompatible with time_as_dim=True.")

    seed = int(_get_cfg_value(cfg, "seed", 42))
    h_norm_denom_value = _get_cfg_value(cfg, "h_norm_denom", None)
    h_norm_denom = float(h_norm_denom_value) if h_norm_denom_value is not None else None
    eval_only = cfg.get("run_mode", "train") == "eval"
    pde_loss = bool(_get_cfg_value(cfg, "pde_loss", False))
    vof_pde_loss = bool(_get_cfg_value(cfg, "VOF_pde_loss", False))
    diff_loss = bool(_get_cfg_value(cfg, "Diff_loss", False))
    if model_family != "fno" and (pde_loss or vof_pde_loss or diff_loss):
        raise ValueError(
            "Only model_family='fno' currently supports pde_loss, VOF_pde_loss, or Diff_loss."
        )
    if use_2d and (pde_loss or vof_pde_loss or diff_loss):
        raise ValueError("2D mode currently supports data-only training; PDE branches are not implemented for 2D.")
    if model_family != "fno" and time_as_dim:
        raise ValueError("time_as_dim=True is currently only supported for model_family='fno'.")
    infer_velocities = bool(OmegaConf.select(cfg, "custom.infer_velocities", default=False))
    autoregressive_enabled = bool(_get_cfg_value(cfg, "autoregressive.enabled", False))
    autoregressive_mode = str(_get_cfg_value(cfg, "autoregressive.mode", "masked_patch")).strip().lower()
    autoregressive_state_prev_suffix = str(_get_cfg_value(cfg, "autoregressive.state_prev_suffix", "_prev"))
    if not autoregressive_state_prev_suffix:
        raise ValueError("autoregressive.state_prev_suffix must be a non-empty string.")
    autoregressive_mask_mode = str(_get_cfg_value(cfg, "autoregressive.mask_mode", "block")).strip().lower()
    autoregressive_mask_ratio = float(_get_cfg_value(cfg, "autoregressive.mask_ratio", 0.15))
    autoregressive_patch_size = tuple(int(value) for value in _get_cfg_value(cfg, "autoregressive.patch_size", [32, 16, 16]))
    autoregressive_context_steps = int(_get_cfg_value(cfg, "autoregressive.context_steps", 1))
    autoregressive_train_patches_per_epoch = int(
        _get_cfg_value(cfg, "autoregressive.train_patches_per_epoch", 256)
    )
    autoregressive_val_patches_per_epoch = int(
        _get_cfg_value(cfg, "autoregressive.val_patches_per_epoch", 64)
    )
    autoregressive_eval_horizon_k = int(_get_cfg_value(cfg, "autoregressive.eval_horizon_k", 10))
    if autoregressive_eval_horizon_k <= 0:
        raise ValueError(
            f"autoregressive.eval_horizon_k must be positive; got {autoregressive_eval_horizon_k}."
        )
    autoregressive_eval_use_gt_init = bool(_get_cfg_value(cfg, "autoregressive.eval_use_gt_init", True))
    autoregressive_teacher_forcing = bool(_get_cfg_value(cfg, "autoregressive.teacher_forcing", False))
    autoregressive_require_time = bool(_get_cfg_value(cfg, "autoregressive.require_time", True))
    autoregressive_prev_noise_std = float(_get_cfg_value(cfg, "autoregressive.prev_noise_std", 0.0))
    if autoregressive_prev_noise_std < 0.0:
        raise ValueError(
            f"autoregressive.prev_noise_std must be >= 0; got {autoregressive_prev_noise_std}."
        )
    allow_grid_interpolation = bool(_get_cfg_value(cfg, "allow_grid_interpolation", False))
    # Rollout-stability knobs (docs/training_and_evaluation.md, "Roadmap").
    autoregressive_prev_noise_resample = bool(
        _get_cfg_value(cfg, "autoregressive.prev_noise_resample", False)
    )
    autoregressive_clamp_feedback = bool(_get_cfg_value(cfg, "autoregressive.clamp_feedback", False))
    _clamp_t_min_cfg = _get_cfg_value(cfg, "autoregressive.clamp_t_min", None)
    _clamp_t_max_cfg = _get_cfg_value(cfg, "autoregressive.clamp_t_max", None)
    autoregressive_clamp_t_max = None if _clamp_t_max_cfg is None else float(_clamp_t_max_cfg)
    autoregressive_unroll_steps = int(_get_cfg_value(cfg, "autoregressive.unroll_steps", 1))
    if autoregressive_unroll_steps < 1:
        raise ValueError(
            f"autoregressive.unroll_steps must be >= 1; got {autoregressive_unroll_steps}."
        )
    # Temporal bundling (upstream RIGNO): one forward predicts B future frames
    # via widened `{key}_bnd{i}` outputs; rollouts jump B frames per forward.
    autoregressive_bundle_steps = int(_get_cfg_value(cfg, "autoregressive.bundle_steps", 1))
    if autoregressive_bundle_steps < 1:
        raise ValueError(
            f"autoregressive.bundle_steps must be >= 1; got {autoregressive_bundle_steps}."
        )
    autoregressive_unroll_detach_grad = bool(
        _get_cfg_value(cfg, "autoregressive.unroll_detach_grad", False)
    )
    # Residual-in-time reparametrization: u_{t+1} = u_t + f(x). Needs the *_prev
    # state inputs that autoregressive mode adds, so it requires enabled=True.
    autoregressive_predict_delta = bool(_get_cfg_value(cfg, "autoregressive.predict_delta", False))
    if autoregressive_predict_delta and not autoregressive_enabled:
        raise ValueError(
            "autoregressive.predict_delta=True requires autoregressive.enabled=True "
            "(the delta add needs the *_prev state inputs, which only exist in AR mode)."
        )

    base_keys = _build_key_config(
        yaml_params,
        add_t_hat=time_as_batch,
        add_velocity_outputs=infer_velocities,
        drop_y_input=use_2d,
    )
    keys = base_keys
    autoregressive_context_keys: list[str] = []
    autoregressive_state_keys: list[str] = []
    autoregressive_state_mask_keys: list[str] = []
    if autoregressive_enabled:
        # gaot rides the same grid path as rigno: every stage below the gate branches
        # on data_layout, not model_family, and GAOTArch takes the *_prev feedback as
        # ordinary input channels (tests/test_gaot_autoregressive.py).
        if model_family not in ("fno", "rigno", "gaot", "deeponet"):
            raise ValueError(
                "autoregressive.enabled=True is currently only supported for model_family "
                f"'fno', 'rigno', 'gaot', or 'deeponet'; got '{model_family}'."
            )
        if autoregressive_mode not in ("masked_patch", "rollout"):
            raise ValueError(
                f"Unsupported autoregressive.mode='{autoregressive_mode}'. Use 'masked_patch' or 'rollout'."
            )
        if autoregressive_mode == "masked_patch":
            if model_family != "fno":
                raise ValueError(
                    "autoregressive.mode='masked_patch' is currently only supported for model_family='fno'; "
                    "use autoregressive.mode='rollout' for rigno/gaot/deeponet."
                )
            _apply_masked_patch_runtime_defaults(cfg, explicit_set_paths)
        elif not autoregressive_eval_use_gt_init:
            raise ValueError(
                "autoregressive.eval_use_gt_init=false is not implemented for autoregressive.mode='rollout'; "
                "the rollout eval always seeds from the ground-truth initial state."
            )
        if autoregressive_require_time and not include_time:
            raise ValueError("autoregressive.enabled=True requires include_time=True when require_time=True.")
        if not time_as_batch:
            raise ValueError("autoregressive.enabled=True requires time_as_batch=True.")
        if pde_loss or vof_pde_loss or diff_loss:
            raise ValueError(
                "autoregressive.enabled=True currently supports data-only mode; set pde_loss, "
                "VOF_pde_loss, and Diff_loss to false."
            )
        if len(base_keys.str_fno_output_keys) == 0:
            raise ValueError("autoregressive.enabled=True requires at least one FNO_output_keys entry.")
        autoregressive_context_keys = list(base_keys.str_fno_input_keys)
        autoregressive_state_keys = list(base_keys.str_fno_output_keys)
        if autoregressive_mode == "masked_patch":
            autoregressive_state_mask_keys = [
                f"{state_key}{autoregressive_state_prev_suffix}_mask" for state_key in autoregressive_state_keys
            ]
        conflicting_prev_keys = [
            f"{state_key}{autoregressive_state_prev_suffix}"
            for state_key in autoregressive_state_keys
            if f"{state_key}{autoregressive_state_prev_suffix}" in autoregressive_context_keys
        ]
        if conflicting_prev_keys:
            raise ValueError(
                "autoregressive input key collision: the following prev-state keys already exist in "
                f"FNO_input_keys: {conflicting_prev_keys}. Choose a different autoregressive.state_prev_suffix."
            )
        keys = _build_key_config(
            yaml_params,
            add_t_hat=time_as_batch,
            add_velocity_outputs=infer_velocities,
            drop_y_input=use_2d,
            state_prev_suffix=autoregressive_state_prev_suffix,
            extra_input_keys=autoregressive_state_mask_keys,
        )
        if autoregressive_bundle_steps > 1:
            if autoregressive_mode != "rollout":
                raise ValueError(
                    "autoregressive.bundle_steps>1 requires autoregressive.mode='rollout'."
                )
            if autoregressive_unroll_steps != 1:
                raise ValueError(
                    "autoregressive.bundle_steps>1 and autoregressive.unroll_steps>1 are "
                    "mutually exclusive (bundling replaces unrolled feedback)."
                )
            if autoregressive_predict_delta:
                raise ValueError(
                    "autoregressive.bundle_steps>1 does not support predict_delta."
                )
            # Widen the model head: bundled future frames become real output keys.
            bundle_output_keys = [
                f"{state_key}_bnd{i}"
                for state_key in autoregressive_state_keys
                for i in range(1, autoregressive_bundle_steps)
            ]
            keys.str_fno_output_keys.extend(bundle_output_keys)
            keys.fno_output_keys.extend(Key(k) for k in bundle_output_keys)
            # ...and widen the input: each bundled frame needs the exogenous
            # context (beam position/dose) at ITS timestep. Without this the
            # far frames are unpredictable on a random path — measured 0.57
            # beam radii of travel across a 4-frame bundle. Double underscore
            # is safe on INPUTS (DerivScalers only parses requested outputs).
            bundle_input_keys = [
                f"{ctx}__ar{i}"
                for ctx in autoregressive_context_keys
                for i in range(1, autoregressive_bundle_steps)
            ]
            keys.str_fno_input_keys.extend(bundle_input_keys)
            keys.fno_input_keys.extend(Key(k) for k in bundle_input_keys)
        if autoregressive_context_steps != 1:
            raise ValueError("autoregressive.context_steps currently only supports 1.")
        if autoregressive_mode == "masked_patch":
            if len(autoregressive_patch_size) != 3:
                raise ValueError(
                    f"autoregressive.patch_size must contain 3 integers; got {autoregressive_patch_size}."
                )
            if autoregressive_mask_mode not in ("voxel", "block"):
                raise ValueError(
                    f"Unsupported autoregressive.mask_mode='{autoregressive_mask_mode}'. Use 'voxel' or 'block'."
                )
            if not 0.0 <= autoregressive_mask_ratio <= 1.0:
                raise ValueError(
                    f"autoregressive.mask_ratio must be in [0, 1]; got {autoregressive_mask_ratio}."
                )
            if autoregressive_train_patches_per_epoch <= 0 or autoregressive_val_patches_per_epoch <= 0:
                raise ValueError("autoregressive patch counts must be positive.")

    if grid_params is None or ref_scales is None:
        raise ValueError("custom config must define geometry and ref_scales.")
    # Default the fed-back T floor to ambient (273.15 K) in normalized units so the
    # clamp keeps a physical minimum rather than 0 K; overridable via clamp_t_min.
    autoregressive_clamp_t_min = (
        273.15 / float(ref_scales["T"]) if _clamp_t_min_cfg is None else float(_clamp_t_min_cfg)
    )
    diff_weight = _get_cfg_value(cfg, "Diff_weight", 1e14)
    eval_model_dir = cfg.get("network_dir", "PATH")
    if eval_model_dir == "PATH" and eval_only:
        raise ValueError("Please provide a valid eval_model_dir in the config file for evaluation mode.")

    data_dir_value = _get_cfg_value(cfg, "data_dir", _DEFAULT_PATH_SENTINEL)
    val_data_dir_value = _get_cfg_value(cfg, "val_data_dir", _DEFAULT_PATH_SENTINEL)
    if eval_only and not _has_explicit_cli_override(explicit_set_paths, "data_dir"):
        data_directory_path = _resolve_default_data_path(_DEFAULT_TEST_DATA_SUBDIR, label="test data")
    else:
        data_directory_path = _resolve_configured_data_path(
            data_dir_value,
            relative_path=_DEFAULT_TRAIN_DATA_SUBDIR,
            label="train data",
        )
    if eval_only:
        # Eval validates against the eval set itself, so the resolved VAL path is
        # discarded. Resolving it first meant a missing default val/ subdir raised
        # and killed the run over a value that was about to be overwritten.
        val_directory_path = data_directory_path
    else:
        val_directory_path = _resolve_configured_data_path(
            val_data_dir_value,
            relative_path=_DEFAULT_VAL_DATA_SUBDIR,
            label="validation data",
        )

    param_dir_value = _get_cfg_value(cfg, "param_dir", _DEFAULT_PATH_SENTINEL)
    # param_dir is only read for piecewise-tabulated c(T) materials
    # (_define_material_properties reads <param_dir>/specific_heat.txt only when
    # material_params.c == "c"); random-path materials use polynomial c(T), so
    # do not require the default path to exist here — validate at read time.
    param_dir = _resolve_configured_data_path(
        param_dir_value,
        relative_path=_DEFAULT_PARAM_DATA_SUBDIR,
        label="parameter data",
        require_exists=False,
    )
    run_name = _get_cfg_value(cfg, "name", "unnamed")
    pde_weight = _get_cfg_value(cfg, "pde_weight", 1.0e-1)
    super_res_factor = int(_get_cfg_value(cfg, "super_res", 1))
    debug = bool(_get_cfg_value(cfg, "detect_anomaly", False))
    sharp_alpha = bool(_get_cfg_value(cfg, "sharp_alpha", False))
    alpha_masking = bool(_get_cfg_value(cfg, "alpha_masking", software == "flow3D"))
    # Default ON for FLOW-3D, which is the data that has a gas gap above the substrate.
    # Recorded in the run config so it reaches AiiDA provenance: a stored metric is
    # otherwise ambiguous about whether the free gas region was counted.
    metrics_exclude_gas = bool(
        _get_cfg_value(cfg, "metrics_exclude_gas", software == "flow3D")
    )
    power_weighted_dose = bool(_get_cfg_value(cfg, "power_weighted_dose", False))
    power_weighted_conduction = bool(
        _get_cfg_value(cfg, "power_weighted_conduction", False)
    )
    conduction_source_sigma_factor = float(_get_cfg_value(cfg, "conduction_source_sigma_factor", 1.0))
    case_coherent_sampling = bool(_get_cfg_value(cfg, "case_coherent_sampling", False))
    eval_frame_transfer = str(_get_cfg_value(cfg, "eval_frame_transfer", "") or "").strip()
    if eval_frame_transfer not in ("", "moving_ref", "steady", "fixed_window"):
        raise ValueError(
            f"custom.eval_frame_transfer must be '' (native), 'moving_ref' (crop each "
            f"frame to the beam window), 'steady' (crop then average the last "
            f"custom.eval_frame_avg frames per case) or 'fixed_window' (x-slice an "
            f"already beam-centred grid), got {eval_frame_transfer!r}. Scoring a "
            "moving-frame model in the lab frame is not possible -- the crop discarded "
            "everything outside the window."
        )
    eval_frame_behind = int(_get_cfg_value(cfg, "eval_frame_behind", 32))
    eval_frame_ahead = int(_get_cfg_value(cfg, "eval_frame_ahead", 16))
    eval_frame_avg = int(_get_cfg_value(cfg, "eval_frame_avg", 30))
    time_avg_window = int(_get_cfg_value(cfg, "time_avg_window", 29 * (not time_as_batch) + 1))
    # In-training validation subset (lazy dataset path): a full-split pass at
    # batch 1 costs tens of minutes cold and recurs every rec_validation_freq
    # (measured 2520 s per validation over 52 val cases, ~30% of a 32k run).
    # Bounded by default; set 0 to validate on the whole split. Scoring runs
    # ignore it -- see _effective_val_max_cases.
    val_max_cases = int(_get_cfg_value(cfg, "val_max_cases", 5))
    if val_max_cases < 0:
        raise ValueError(f"custom.val_max_cases must be >= 0; got {val_max_cases}.")
    # Frames-per-case twin of val_max_cases: per-frame val metrics are heavily
    # autocorrelated within a case, so ~10 spread frames answer the same health
    # question at a fraction of the cost. The pvlhs wave-1 arms set val_max_cases=0
    # and spent 10.8 h of an 11.6 h run validating (17 passes x 38 min over 12
    # cases x ~500 frames); the same failure had already been recorded on
    # 2026-07-22. Scoring runs ignore it -- see _effective_val_frames_per_case.
    val_frames_per_case = int(_get_cfg_value(cfg, "val_frames_per_case", 10))
    if val_frames_per_case < 0:
        raise ValueError(f"custom.val_frames_per_case must be >= 0; got {val_frames_per_case}.")
    configured_moving_frame_steps = int(_get_cfg_value(cfg, "moving_frame_steps", 30))
    if configured_moving_frame_steps <= 0:
        raise ValueError(f"moving_frame_steps must be positive; got {configured_moving_frame_steps}.")
    moving_ref_frame = _coerce_bool(_get_cfg_value(cfg, "moving_ref_frame", True))
    moving_frame_steps = configured_moving_frame_steps if moving_ref_frame else 1
    _reject_lab_frame_keys_in_moving_frame(keys.str_fno_input_keys, moving_ref_frame)
    if "t_hat" in keys.str_fno_input_keys and not include_time:
        raise ValueError("FNO_input_keys includes 't_hat' but include_time=False.")

    # x,y,z always route to the DeepONet trunk (query coordinates); the branch takes
    # spatially-constant conditioning. A spatially-varying path field (e.g. path_dose)
    # must go to the trunk too, else it degenerates the branch factorization -- allow
    # extra trunk keys via custom.deeponet_trunk_extra_keys.
    extra_trunk = _get_cfg_value(cfg, "deeponet_trunk_extra_keys", None) or []
    coordinate_key_order = ("x", "y", "z", *list(extra_trunk))
    deeponet_trunk_keys = [key for key in keys.str_fno_input_keys if key in coordinate_key_order]
    deeponet_branch_keys = [key for key in keys.str_fno_input_keys if key not in deeponet_trunk_keys]
    if model_family == "deeponet":
        if not deeponet_branch_keys:
            raise ValueError("DeepONet requires at least one non-coordinate branch input key.")
        if not deeponet_trunk_keys:
            raise ValueError("DeepONet requires at least one coordinate trunk input key.")

    validation_batch_size = int(OmegaConf.select(cfg, "batch_size.validation", default=1))
    inference_batch_size_cfg = OmegaConf.select(cfg, "custom.inference_batch_size", default=None)
    if inference_batch_size_cfg is None:
        # Backward-compatible fallback for older override style.
        inference_batch_size_cfg = _get_cfg_value(cfg, "inference_batch_size", validation_batch_size)
    inference_batch_size = int(inference_batch_size_cfg)
    if inference_batch_size <= 0:
        raise ValueError(f"inference_batch_size must be positive; got {inference_batch_size}.")
    split_mode = str(_get_cfg_value(cfg, "split_mode", default_split_mode)).lower()
    if split_mode not in ("standard", "kfold"):
        raise ValueError(f"Unsupported split_mode: {split_mode}. Expected one of: standard, kfold.")
    k_folds = int(_get_cfg_value(cfg, "k_folds", 8))
    fold_index = int(_get_cfg_value(cfg, "fold_index", -1))

    os.environ["WANDB_MODE"] = "offline" if bool(_get_cfg_value(cfg, "wandb_offline", False)) else "online"

    if eval_model_dir in ("PATH", ".") and not eval_only:
        cfg.network_dir = f"outputs/{run_name}_{datetime.now().strftime('%m%d_%H%M')}"
    elif eval_model_dir in ("PATH", ".") and eval_only:
        raise ValueError("Please provide a valid eval_model_dir (network_dir) for evaluation mode.")
    else:
        cfg.network_dir = eval_model_dir
    os.makedirs(cfg.network_dir, exist_ok=True)
    print(f"Eval dir: {eval_model_dir}")

    # Derive the rectangular domain from the data when it carries its own grid
    # (consolidated NPZ, or a staged arraydir runtime -- which the arraydir loader
    # serves at native shape, so the config box would describe nothing) or when the
    # config opts in with geometry/resolution: auto. VTK data keeps its curated config
    # box (a deliberate crop, not the raw extent).
    grid_params, window_baked = _grid_params_for_data(grid_params, data_directory_path, software)

    (xmin, xmax) = grid_params["x_range"]
    (ymin, ymax) = grid_params["y_range"]
    (zmin, zmax) = grid_params["z_range"]
    (nx, ny, nz) = grid_params["resolution"]
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz
    bounds = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

    nx_sr, ny_sr, nz_sr = (nx * super_res_factor, ny * super_res_factor, nz * super_res_factor)
    grid_params["resolution"] = (nx_sr, ny_sr, nz_sr)
    effective_shift = 0
    if include_time and not time_as_dim and moving_ref_frame and not window_baked:
        # crop_shift_moving_frame keeps the last N steps and shifts each by its
        # timestep index, so x is reduced by (Nsteps - 1) * super_res_factor. A staged
        # moving-frame variant was exported with that window already applied and its
        # derived resolution IS the served x extent, so the shift is not applied again.
        effective_shift = max(0, (moving_frame_steps - 1) * super_res_factor)
    nx_effective = nx_sr - effective_shift
    if nx_effective <= 0:
        raise ValueError(
            f"Effective x-resolution must be positive after temporal shift: nx_sr={nx_sr}, "
            f"effective_shift={effective_shift}."
        )
    if use_2d and not time_as_dim:
        y_mid = 0.5 * (ymin + ymax)
        bounds = [[xmin, xmax], [y_mid, y_mid], [zmin, zmax]]
    elif time_as_dim:
        # VTK export remains 3D; map the second axis to time-index coordinates.
        t_max = float(max(1, moving_frame_steps) - 1)
        bounds = [[xmin, xmax], [0.0, t_max], [zmin, zmax]]

    print(f"Using grid resolution: {grid_params['resolution']}, with super res factor: {super_res_factor}")
    if model_family == "fno":
        cfg.arch.fno.padding = cfg.arch.fno.padding * super_res_factor

        if use_2d and not time_as_dim:
            cfg.arch.fno.dimension = 2
        else:
            cfg.arch.fno.dimension = 3

        current_modes = list(cfg.arch.fno.fno_modes)
        patch_resolution = None
        if autoregressive_enabled and autoregressive_mode == "masked_patch":
            patch_resolution = [int(value) for value in autoregressive_patch_size]

        if cfg.arch.fno.dimension == 2:
            if len(current_modes) >= 3:
                # 2D xz mode: reuse x and z entries from the prior xyz configuration.
                current_modes = [current_modes[0], current_modes[2]]
            elif len(current_modes) != 2:
                raise ValueError(f"Expected 2 or 3 FNO modes for 2D setup; got {current_modes}.")
            axis_names = ("x", "z")
            if patch_resolution is not None:
                nyquist_limits = [max(1, patch_resolution[0] // 2), max(1, patch_resolution[-1] // 2)]
            else:
                nyquist_limits = [max(1, nx_effective // 2), max(1, nz_sr // 2)]
            ny_effective = 1
        else:
            if len(current_modes) == 2:
                current_modes = [current_modes[0], current_modes[0], current_modes[1]]
            elif len(current_modes) < 3:
                raise ValueError(f"Expected at least 3 FNO modes for 3D setup; got {current_modes}.")
            elif len(current_modes) > 3:
                current_modes = current_modes[:3]
            axis_names = ("x", "y", "z")
            if patch_resolution is not None:
                nyquist_limits = [max(1, int(value) // 2) for value in patch_resolution]
                ny_effective = int(patch_resolution[1])
            elif time_as_dim:
                time_extent_hint = max(1, moving_frame_steps)
                nyquist_limits = [
                    max(1, nx_effective // 2),
                    max(1, time_extent_hint // 2),
                    max(1, nz_sr // 2),
                ]
                ny_effective = time_extent_hint
            else:
                nyquist_limits = [max(1, nx_effective // 2), max(1, ny_sr // 2), max(1, nz_sr // 2)]
                ny_effective = ny_sr

        print(f"Nyquist smallest modes: {tuple(nyquist_limits)}, requested FNO modes: {tuple(current_modes)}")
        for i, (axis, limit) in enumerate(zip(axis_names, nyquist_limits)):
            if current_modes[i] > limit:
                print(
                    f"Warning: Requested FNO modes {current_modes[i]} exceed Nyquist limit in the {axis} direction {limit}. "
                    "Capping to Nyquist limit."
                )
                current_modes[i] = limit
            if current_modes[i] <= 0:
                current_modes[i] = 1
        cfg.arch.fno.fno_modes = tuple(current_modes)
        print(f"Using FNO dimension: {cfg.arch.fno.dimension}, modes: {cfg.arch.fno.fno_modes}")
        if not eval_only and cfg.arch.fno.dimension == 3 and patch_resolution is None and not time_as_dim:
            _refuse_fno_modes_beyond_served_grid(
                list(current_modes), (nx_effective, ny_sr, nz_sr), cfg.arch.fno.padding
            )
        if eval_only:
            print(f"Evaluation mode: Make sure FNO modes {cfg.arch.fno.fno_modes} is compatible with trained model.")
    else:
        ny_effective = 1 if (use_2d and not time_as_dim) else ny_sr
        if model_family == "rigno":
            # RIGNO builds its graph lazily from the true grid shape; it only needs
            # the spatial dimensionality to size the coordinate/edge features.
            cfg.arch.rigno.dimension = 2 if (use_2d and not time_as_dim) else 3
            print(f"Using RIGNO dimension: {cfg.arch.rigno.dimension}")
        if model_family == "gaot" and OmegaConf.select(cfg, "arch.gaot") is not None:
            # Same deal: GAOT builds its phys<->latent graphs lazily from the true
            # grid shape and only needs the spatial dimensionality up front. The
            # latent lattice must carry one axis per dimension, so drop the y axis
            # of the configured lattice in 2D mode. Guarded on the block existing so
            # a config that leaves every GAOT knob at its model_domain default still
            # loads (the arch then falls back to its own dimension default).
            dim = 2 if (use_2d and not time_as_dim) else 3
            cfg.arch.gaot.dimension = dim
            latent = [int(v) for v in cfg.arch.gaot.latent_tokens]
            if len(latent) != dim:
                cfg.arch.gaot.latent_tokens = (
                    [latent[0], latent[2]] if dim == 2 and len(latent) == 3 else latent[:dim]
                )
            print(
                f"Using GAOT dimension: {dim}, latent tokens: "
                f"{list(cfg.arch.gaot.latent_tokens)}"
            )
        print(f"Using model family: {model_family} with {data_layout} data layout.")
    if fast_training:
        print("fast_training=True: validators disabled.")

    _sync_one_cycle_total_steps(cfg)
    cap_save_network_freq(cfg, str(cfg.get("run_mode", "train")))
    fall_back_from_unbuildable_beambox_stop_metric(
        cfg, keys.str_fno_input_keys, moving_ref_frame, str(cfg.get("run_mode", "train"))
    )
    validate_recording_cadence(cfg, str(cfg.get("run_mode", "train")))

    # Load only the source fields something consumes. Applies to EVERY model
    # family, not just one config: an unused field is decompressed for the whole
    # case on each lazy cache miss and then discarded.
    vtk_fields, _pruned_fields = _prune_unused_vtk_fields(
        vtk_fields,
        keys.str_fno_input_keys,
        keys.str_fno_output_keys,
        pde_loss=pde_loss,
        vof_pde_loss=vof_pde_loss,
        diff_loss=diff_loss,
        infer_velocities=infer_velocities,
    )
    if _pruned_fields:
        print(
            f"vtk_fields: not loading {_pruned_fields} -- no configured input or "
            "output key consumes them and no PDE/VOF/Diff loss is on. They are "
            f"decompressed per case on the lazy path, so this is pure saving. "
            f"Loading: {list(vtk_fields)}"
        )

    h_norm_include_radius = bool(_get_cfg_value(cfg, "h_norm_include_radius", False))
    h_norm_include_material = bool(_get_cfg_value(cfg, "h_norm_include_material", False))
    h_norm_denom_source = None
    h_norm_denom_radius_aware = False
    if eval_only and split_mode == "standard":
        # The denominator the model TRAINED with, never a refit on the scored split
        # (h_norm_record.py; every standalone eval before 2026-10-30 refitted it).
        from .h_norm_record import resolve_eval_denominator

        resolution = resolve_eval_denominator(
            network_dir=cfg.network_dir, explicit=h_norm_denom,
            include_radius=h_norm_include_radius, include_material=h_norm_include_material,
            time_as_batch=time_as_batch,
        )
        h_norm_denom = resolution.denom
        h_norm_denom_source = resolution.source
        h_norm_denom_radius_aware = resolution.radius_aware

    return RunConfig(
        cfg=cfg,
        seed=seed,
        h_norm_denom=h_norm_denom,
        h_norm_include_radius=h_norm_include_radius,
        h_norm_include_material=h_norm_include_material,
        h_norm_denom_source=h_norm_denom_source,
        h_norm_denom_radius_aware=h_norm_denom_radius_aware,
        model_family=model_family,
        data_layout=data_layout,
        eval_only=eval_only,
        split_mode=split_mode,
        pde_loss=pde_loss,
        vof_pde_loss=vof_pde_loss,
        diff_loss=diff_loss,
        diff_weight=diff_weight,
        data_directory_path=data_directory_path,
        val_directory_path=val_directory_path,
        param_dir=param_dir,
        run_name=run_name,
        pde_weight=pde_weight,
        super_res_factor=super_res_factor,
        debug=debug,
        sharp_alpha=sharp_alpha,
        alpha_masking=alpha_masking,
        metrics_exclude_gas=metrics_exclude_gas,
        power_weighted_dose=power_weighted_dose,
        power_weighted_conduction=power_weighted_conduction,
        conduction_source_sigma_factor=conduction_source_sigma_factor,
        case_coherent_sampling=case_coherent_sampling,
        eval_frame_transfer=eval_frame_transfer,
        eval_frame_behind=eval_frame_behind,
        eval_frame_ahead=eval_frame_ahead,
        eval_frame_avg=eval_frame_avg,
        include_time=include_time,
        use_2d=use_2d,
        time_as_dim=time_as_dim,
        fast_training=fast_training,
        time_avg_window=time_avg_window,
        val_max_cases=val_max_cases,
        val_frames_per_case=val_frames_per_case,
        time_as_batch=time_as_batch,
        autoregressive_enabled=autoregressive_enabled,
        autoregressive_mode=autoregressive_mode,
        autoregressive_state_prev_suffix=autoregressive_state_prev_suffix,
        autoregressive_mask_mode=autoregressive_mask_mode,
        autoregressive_mask_ratio=autoregressive_mask_ratio,
        autoregressive_patch_size=tuple(autoregressive_patch_size),
        autoregressive_context_steps=autoregressive_context_steps,
        autoregressive_train_patches_per_epoch=autoregressive_train_patches_per_epoch,
        autoregressive_val_patches_per_epoch=autoregressive_val_patches_per_epoch,
        autoregressive_state_mask_keys=autoregressive_state_mask_keys,
        autoregressive_eval_horizon_k=autoregressive_eval_horizon_k,
        autoregressive_eval_use_gt_init=autoregressive_eval_use_gt_init,
        autoregressive_teacher_forcing=autoregressive_teacher_forcing,
        autoregressive_context_keys=autoregressive_context_keys,
        autoregressive_state_keys=autoregressive_state_keys,
        autoregressive_prev_noise_std=autoregressive_prev_noise_std,
        autoregressive_prev_noise_resample=autoregressive_prev_noise_resample,
        autoregressive_clamp_feedback=autoregressive_clamp_feedback,
        autoregressive_clamp_t_min=autoregressive_clamp_t_min,
        autoregressive_clamp_t_max=autoregressive_clamp_t_max,
        autoregressive_unroll_steps=autoregressive_unroll_steps,
        autoregressive_bundle_steps=autoregressive_bundle_steps,
        autoregressive_unroll_detach_grad=autoregressive_unroll_detach_grad,
        autoregressive_predict_delta=autoregressive_predict_delta,
        allow_grid_interpolation=allow_grid_interpolation,
        deeponet_branch_keys=deeponet_branch_keys,
        deeponet_trunk_keys=deeponet_trunk_keys,
        moving_ref_frame=moving_ref_frame,
        moving_frame_steps=moving_frame_steps,
        inference_batch_size=inference_batch_size,
        k_folds=k_folds,
        fold_index=fold_index,
        grid_params=grid_params,
        ref_scales=ref_scales,
        laser_params=laser_params,
        material_params=material_params,
        software=software,
        vtk_fields=vtk_fields,
        bounds=bounds,
        dx=dx,
        dy=dy,
        dz=dz,
        nx_effective=nx_effective,
        ny_effective=ny_effective,
        nz_effective=nz_sr,
        keys=keys,
        wb_project=_get_wandb_project(software),
        network_base_dir=cfg.network_dir,
    )


def setup_wandb_run(cfg: PhysicsNeMoConfig, wb_project: str, run_name: str) -> None:
    """Initialize wandb and tensorboard patching for one training/eval run."""

    os.makedirs(cfg.network_dir, exist_ok=True)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    initialize_wandb(
        project=wb_project,
        name=run_name,
        # From the environment, never a literal: a hard-coded org made `wandb login` fail
        # for every outside user and put a lab slug in a public tree. None = your default.
        entity=os.environ.get("WANDB_ENTITY"),
        mode=os.environ.get("WANDB_MODE"),
        sync_tensorboard=True,
        save_code=True,
        config=wandb_config,
    )
    wandb.tensorboard.unpatch()
    wandb.tensorboard.patch(root_logdir=str(cfg.network_dir))
    os.environ["WANDB_TENSORBOARD_LOGDIR"] = str(cfg.network_dir)
    LaunchLogger.initialize(use_wandb=True)
