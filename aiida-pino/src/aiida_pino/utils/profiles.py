"""Profile validation, default loading, and metrics helpers."""

from __future__ import annotations

import ast
import re
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled through fallback defaults
    yaml = None


CONVERSION_PROFILE_KEYS = {"case_regex", "num_timesteps", "output_basename_mode"}
# `flip_axes` is deliberately optional: when absent the adapters resolve it
# source-aware (OpenFOAM flips, FLOW-3D does not; see BUG-11).
DATASET_PROFILE_KEYS = {
    "geometry",
    "vtk_fields",
}
# Direct FLOW-3D NPZ import profile: every key is optional, so validation
# rejects unknown keys and bad types instead of requiring anything.
FLOW3D_NPZ_PROFILE_KEYS = frozenset(
    {
        "npz_fields",
        "time_target_frames",
        "description",
        # The dataset's own keywords. `batch_keyword` is NOT this: it NAMES the
        # dataset (and tags its simulations), which is exactly why it reads like
        # a keyword field and why a sweep could not set keywords at all before
        # (2026-08-10 friction).
        "keywords",
        "batch_keyword",
        "usable_for_training",
        "exclude_globs",
        "trust_npz_payload",
    }
)
# FLOW-3D run execution profile (flow3D_scripts runner): every key optional.
FLOW3D_RUN_EXECUTION_PROFILE_KEYS = frozenset(
    {
        "flow3d_scripts_root",
        "out_root",
        "case_name",
        # Higher wins the FLOW-3D seat when several runs are queued. Without it
        # "run this next" had one implementation: pause the other sweeps by
        # hand, which stops them entirely and leaves work nothing is scheduled
        # Ordering among runs waiting at the same moment is strict: a lower-priority waiter stands
    # aside while any live, fresh, higher-priority run is queued, so a steady stream of
    # urgent work does keep it waiting; the only bound is ticket validity
    # (TICKET_MAX_AGE_S in flow3d_run_adapter). A solving run holds no ticket, so it cannot
    # reserve the seat against a batch that keeps one case in flight
    # (potential_bugs.md, PB-26).
        "seat_priority",
        "cores",
        "threads_per_rank",
        "prepare_only",
        "overwrite",
        "export",
        "flow3d_env",
        "post_home",
        "resource_scope",
        "memory_max",
        "timeout_s",
        "watchdog",
        "stall_s",
        "keep_flsgrf",
    }
)
FLOW3D_RUN_EXPORT_MODES = frozenset({"npz", "vtk", "vtk-surface", "vtk-legacy60", "both", "none"})


def validate_flow3d_run_execution_profile(values: dict) -> None:
    """Validate a FLOW-3D run execution profile (all keys optional, no license material)."""
    if not isinstance(values, dict):
        raise ValueError("execution_profile must be a mapping.")
    license_keys = sorted(key for key in values if "license" in str(key).lower())
    if license_keys:
        raise ValueError(
            f"execution_profile must not carry license configuration ({', '.join(license_keys)}). "
            "FLOW-3D license endpoints come from the runner environment (F3D_LICENSE / "
            "F3DTKNUX_LICENSE_FILE / LM_LICENSE_FILE or the runner repo's ignored license.env) "
            "and are never stored in AiiDA nodes."
        )
    unknown = sorted(set(values) - FLOW3D_RUN_EXECUTION_PROFILE_KEYS)
    if unknown:
        raise ValueError(
            f"execution_profile contains unknown keys: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(FLOW3D_RUN_EXECUTION_PROFILE_KEYS))}."
        )
    for key in ("flow3d_scripts_root", "out_root", "case_name", "flow3d_env", "post_home"):
        value = values.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"execution_profile.{key} must be a non-empty string.")
    for key in ("cores", "threads_per_rank"):
        value = values.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"execution_profile.{key} must be a positive integer.")
    seat_priority = values.get("seat_priority")
    if seat_priority is not None and (isinstance(seat_priority, bool) or not isinstance(seat_priority, int)):
        raise ValueError(
            "execution_profile.seat_priority must be an integer (higher wins the FLOW-3D seat; "
            "default 0)."
        )
    resource_scope = values.get("resource_scope")
    if resource_scope is not None and not isinstance(resource_scope, bool):
        raise ValueError("execution_profile.resource_scope must be a boolean.")
    watchdog = values.get("watchdog")
    if watchdog is not None and not isinstance(watchdog, bool):
        raise ValueError("execution_profile.watchdog must be a boolean.")
    keep_flsgrf = values.get("keep_flsgrf")
    if keep_flsgrf is not None and not isinstance(keep_flsgrf, bool):
        raise ValueError("execution_profile.keep_flsgrf must be a boolean.")
    for key in ("timeout_s", "stall_s"):
        value = values.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        ):
            raise ValueError(f"execution_profile.{key} must be a positive number of seconds.")
    export = values.get("export")
    if export is not None and export not in FLOW3D_RUN_EXPORT_MODES:
        raise ValueError(
            f"execution_profile.export must be one of: {', '.join(sorted(FLOW3D_RUN_EXPORT_MODES))}."
        )


def dataset_profile_metadata_gaps(values: dict) -> str | None:
    """Warning naming the findability fields a `dataset_profile` omits, or None.

    REPORT, never refuse. An untagged dataset is unfindable by topic among 1300+
    base datasets and unusable by anyone who did not build it, and because inputs
    freeze at submit the only repair is a manual post-hoc write — six batches
    needed one before this existed. But a refusing gate is the wrong cure: on
    2026-07-21 a gate that rejected instead of waiting killed 300 queued cases in
    five minutes, and a sweep can cost days of seat time. A report line costs
    nothing, shows in `verdi process report`, and cannot destroy a batch.
    """
    if not isinstance(values, dict):
        return None

    def _empty(key: str) -> bool:
        value = values.get(key)
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set)):
            return not [v for v in value if str(v).strip()]
        return False

    missing = [key for key in ("keywords", "description") if _empty(key)]
    if not missing:
        return None
    named = " and ".join(missing)
    return (
        f"dataset_profile omits {named}: the resulting dataset will be untagged and "
        "unfindable by topic among 1300+ base datasets. Inputs freeze at submit, so "
        "this can only be repaired afterwards with `aiida-pino tag` / a description "
        "write. Note `batch_keyword` NAMES the dataset and does not set keywords."
    )


def validate_flow3d_npz_profile(values: dict) -> None:
    """Validate a FLOW-3D NPZ import profile (all keys optional)."""
    if not isinstance(values, dict):
        raise ValueError("dataset_profile must be a mapping.")
    unknown = sorted(set(values) - FLOW3D_NPZ_PROFILE_KEYS)
    if unknown:
        raise ValueError(
            f"dataset_profile contains unknown keys: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(FLOW3D_NPZ_PROFILE_KEYS))}."
        )
    npz_fields = values.get("npz_fields")
    if npz_fields is not None and (
        not isinstance(npz_fields, list) or not all(isinstance(field, str) and field.strip() for field in npz_fields)
    ):
        raise ValueError("dataset_profile.npz_fields must be a list of non-empty field-name strings.")
    exclude_globs = values.get("exclude_globs")
    if exclude_globs is not None and (
        not isinstance(exclude_globs, list)
        or not all(isinstance(pattern, str) and pattern.strip() for pattern in exclude_globs)
    ):
        raise ValueError(
            "dataset_profile.exclude_globs must be a list of non-empty relative glob patterns "
            "(e.g. \"smoke/*\" or \"*_twin.npz\")."
        )
    target = values.get("time_target_frames")
    if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target < 2):
        raise ValueError("dataset_profile.time_target_frames must be an integer >= 2.")
    for key in ("description", "batch_keyword"):
        value = values.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"dataset_profile.{key} must be a string.")
    usable = values.get("usable_for_training")
    if usable is not None and not isinstance(usable, bool):
        raise ValueError("dataset_profile.usable_for_training must be a boolean.")
    trust = values.get("trust_npz_payload")
    if trust is not None and not isinstance(trust, bool):
        raise ValueError("dataset_profile.trust_npz_payload must be a boolean.")

DEFAULT_TRAINING_PROFILE = {
    "config_name": "config_NS_energy_PINO_flow3D",
    "run_name": "unnamed",
    "hydra_overrides": [],
    # "auto", not "0" (2026-08-23). The console form PRE-FILLS this field from the
    # default and submits whatever it holds, so a shipped "0" arrives as an explicit
    # choice that `resolve_cuda_device_for_target` is right to honour -- and every
    # console submission pinned card 0. Measured: 25 of the last 120 local trainings
    # pinned there, with six arms queued on card 0 while card 1 sat at 0/22528 MiB.
    # `gpu_spool` resolves "auto" by least-committed card and exports
    # CUDA_VISIBLE_DEVICES, which the training honours over the packaged config's own
    # `custom.cuda_device` -- the adapter emits no cuda_device token at all for "auto".
    "cuda_device": "auto",
    # Which machine to run on, by AiiDA code label. None means this one.
    # `TRAINING_PROFILE_KEYS` is built from these keys and is the only gate on
    # what a profile may contain -- `merge_training_profile` carries unknown
    # keys through untouched -- so a key that is not here silently does nothing.
    "code_label": None,
    "wandb_offline": False,
    "split_mode": "standard",
    "k_folds": 8,
    "fold_index": -1,
    "pde_loss": False,
    "vof_pde_loss": False,
    "diff_loss": False,
    "diff_weight": 1.0e14,
    "pde_weight": 1.0e-1,
    "include_time": True,
    "time_as_batch": False,
    "time_as_dim": False,
    "use_2d": False,
    "time_avg_window": 30,
    "moving_ref_frame": True,
    "moving_frame_steps": 30,
    "fast_training": False,
    "detect_anomaly": False,
    "sharp_alpha": False,
    "alpha_masking": True,
    "max_steps": 15000,
    "learning_rate": 6.0e-5,
    "weight_decay": 0.0,
    "batch_size_grid": 20,
    "batch_size_validation": 20,
    "scheduler_decay_rate": 0.98,
    "scheduler_decay_steps": 100,
    "inference_batch_size": 2,
    "infer_velocities": False,
    # Measured peaks, so gpu_spool can pack this run beside others instead of
    # booking the whole card. None = undeclared = exclusive booking (safe:
    # under-declaring VRAM OOMs whatever it was packed with). Size from an
    # observed peak, never a guess.
    "gpu_vram_mb": None,
    # The footprint key `gpu_vram_mb` was measured under. Written by the VRAM
    # automation, carried along when a profile is cloned, and compared on the
    # next run: a booking is only meaningful for the footprint that produced it.
    # Absent means "hand-set", which is honoured as a deliberate reservation.
    "vram_key": None,
    # Deliberate whole-card booking that also opts out of automatic reservation.
    "gpu_exclusive": False,
    "host_ram_mb": None,
    # Node count of the grid this run trains on (nx*ny*nz). Read by
    # `_check_host_ram_against_loader_settings` to scale the full-resolution
    # host-RAM floor via `full_res_host_ram_floor_mb`, whose 40960 anchor is a
    # 600k-node GAOT build peaking at ~57 GB. Undeclared keeps that flat anchor.
    # It was READ but never DECLARED, so `TRAINING_PROFILE_KEYS` refused every
    # submission that set it and the grid-aware branch was unreachable -- the same
    # defect recorded above for `gpu_seat_priority`. A cropped 90x40x30 RIGNO arm
    # (108k nodes, measured RSS 8188 MiB incl. loader children) was therefore made
    # to book 40960, which caps a 128 GB box at two concurrent runs because host
    # RAM is booked machine-wide.
    "graph_nodes": None,
    # Jump the gpu_spool queue for the GPU seat: it parks a `.want` marker so spool
    # workers hold off starting new runs, and this run goes in ahead of the queue.
    # It never preempts a run already holding the card. `train_adapter.wrap_with_gpu_seat`
    # has always read this and appended `--priority`, but the key was only ever
    # declared on the EVALUATION profile, so the console's unknown-key check
    # refused it on a training and the working code was unreachable. Boolean,
    # not an integer, unlike the FLOW-3D `execution_profile.seat_priority`.
    #
    # It does NOT buy a deadline. Bare `--priority` also swaps gpu_spool's unbounded
    # wait for DEFAULT_PRIORITY_MAX_WAIT_S = 3600 s, which killed training 99083 at
    # 3602 s (2026-09-09) and two evals at 3601 s (2026-07-29); the adapters now pass
    # `--max-wait-s 0` alongside it, so the jump and the deadline are separable and
    # only CI takes the deadline. The cost that remains is the `.want` marker: a
    # stuck priority job makes workers hold off, which can starve an idle GPU.
    "gpu_seat_priority": False,
    # Tri-state per-model marker for power-scaled dose channels
    # (2026-08-14). None = "not stamped": `merge_training_profile` resolves it to
    # True for a NEW training (dose channels carry power by default), while
    # `merge_model_training_with_evaluation_profile` resolves a pre-stamp model's
    # None to False so its evals reproduce the power-blind inputs it trained on.
    # The resolved value is frozen into the model's training profile — that frozen
    # boolean IS the marker; the adapters emit it as `custom.power_weighted_dose`
    # only when it is not None.
    "power_weighted_dose": None,
    # Extra environment variables for the training/eval subprocess
    # (PINO_LAZY_NPZ, PYTORCH_CUDA_ALLOC_CONF, dose switches, ...).
    "env": {},
    "autoregressive": {
        "enabled": False,
        "mode": "masked_patch",
        "state_prev_suffix": "_prev",
        "patch_size": [32, 16, 16],
        "context_steps": 1,
        "mask_mode": "block",
        "mask_ratio": 0.15,
        "train_patches_per_epoch": 256,
        "val_patches_per_epoch": 64,
        "eval_horizon_k": 10,
        "eval_use_gt_init": True,
        "require_time": True,
    },
}
# Spatial stride applied automatically to RIGNO runs, as "sx,sy,sz".
#
# Deliberately EMPTY: nothing is downsampled unless a caller asks for it, so a
# dataset that is already coarse (the ti64_random150_ds_* splits are 75x50x40 on
# disk) cannot be silently downsampled twice. Striding still exists as a feature
# -- pass env={"PINO_LAZY_STRIDE": "2,2,1"} on the training profile -- and it is
# what keeps full-resolution RIGNO inside the 24 GB card (22.2 -> 14.7 GB
# measured on the 300-case random-path set).
#
# Set this to "2,2,1" to make it the default for every RIGNO config again.
RIGNO_DEFAULT_SPATIAL_STRIDE = ""

TRAINING_PROFILE_KEYS = frozenset(DEFAULT_TRAINING_PROFILE)
TRAINING_PROFILE_REQUIRED_KEYS = frozenset({"config_name", "run_name", "hydra_overrides", "cuda_device", "wandb_offline"})
#: What a *finished* run must declare. `cuda_device` and `wandb_offline`
#: configure a launch that already happened, so demanding them when registering
#: an existing run forces the caller to invent values describing nothing --
#: fabricated provenance inside a provenance tool (2026-08-10 friction, item 2).
IMPORT_PROFILE_REQUIRED_KEYS = frozenset({"config_name", "run_name", "hydra_overrides"})
AUTOREGRESSIVE_KEYS = frozenset(DEFAULT_TRAINING_PROFILE["autoregressive"])


#: Matches the one Hydra key that sets the step budget, with the optional `+`
#: (append) and `++` (force) prefixes, anchored so `training.max_steps_warmup`
#: and `custom.autoregressive.max_steps` cannot be mistaken for it.
_MAX_STEPS_OVERRIDE = re.compile(r"^\+{0,2}training\.max_steps=(.*)$")


#: Structured profile fields that `train_adapter` also emits as a Hydra key.
#: When a profile ALSO carries that key in `hydra_overrides`, the override is
#: appended later and wins, leaving the structured field dead but stored.
STRUCTURED_TO_HYDRA = {
    "max_steps": "training.max_steps",
    "learning_rate": "optimizer.lr",
    "weight_decay": "optimizer.weight_decay",
    "batch_size_grid": "batch_size.grid",
    "batch_size_validation": "batch_size.validation",
    "inference_batch_size": "custom.inference_batch_size",
}


def override_conflicts(profile: dict[str, Any]) -> dict[str, tuple[Any, str]]:
    """Structured fields their own overrides silently overrule.

    `effective_max_steps` solved this for ONE field; the same shape applies to
    every key `train_adapter` emits structurally. Measured 2026-08-13: 7 models
    record `weight_decay: 0.0` while their own override sets 0.01 or 0.1 -- so
    a reader (or the ablation referee) comparing the stored field compares a
    value that never trained anything.

    Returns {field: (stored_value, winning_override_value)} for fields that
    disagree.
    """
    overrides: dict[str, str] = {}
    for item in profile.get("hydra_overrides") or []:
        key, sep, value = str(item).partition("=")
        if sep:
            overrides[key.strip().lstrip("+~")] = value.strip()
    out: dict[str, tuple[Any, str]] = {}
    for field, hydra_key in STRUCTURED_TO_HYDRA.items():
        if hydra_key in overrides and str(profile.get(field)) != overrides[hydra_key]:
            out[field] = (profile.get(field), overrides[hydra_key])
    return out


def effective_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """The profile as Hydra will actually apply it: overrides folded in.

    Read comparisons and displays through this. It does NOT change what is
    stored or submitted -- resolving overrides at submission time would change
    behaviour and is left open (potential_bugs.md, PB-07).
    """
    resolved = dict(profile or {})
    for field, (_stored, winner) in override_conflicts(resolved).items():
        current = resolved.get(field)
        try:
            resolved[field] = type(current)(winner) if current is not None else winner
        except (TypeError, ValueError):
            resolved[field] = winner
    return resolved


def effective_max_steps(profile: dict[str, Any]) -> int | None:
    """The step budget Hydra will actually use, not the one the profile asked for.

    `train_adapter` emits `training.max_steps=<profile['max_steps']>` and then
    appends `hydra_overrides` verbatim, so a profile carrying its own
    `training.max_steps=` override lands the key on the command line twice.
    Hydra applies overrides in order, so the later one wins and the structured
    field is dead -- one run trained to 35000 while `max_steps` said 20000,
    and the console progress bar read `step 19900 / 20000`.

    Read the budget through this, never `profile["max_steps"]` directly. It
    matters beyond the display: `_sync_one_cycle_total_steps` sizes the
    one-cycle LR schedule from whichever value wins, so a requeue that trusts
    the structured field builds a cosine for the wrong length and a resumed
    scheduler state lands in the wrong place on it.
    """
    if not isinstance(profile, dict):
        return None
    declared = profile.get("max_steps")
    overrides = profile.get("hydra_overrides")
    if isinstance(overrides, list):
        # Last match wins, exactly as hydra applies them.
        for item in reversed(overrides):
            if not isinstance(item, str):
                continue
            match = _MAX_STEPS_OVERRIDE.match(item.strip())
            if match is None:
                continue
            try:
                return int(str(match.group(1)).strip())
            except (TypeError, ValueError):
                # A value hydra itself would reject; fall back to the declared
                # field rather than guessing or breaking the Runs list.
                break
    try:
        return int(declared) if declared is not None else None
    except (TypeError, ValueError):
        return None


def effective_stride(
    dataset_attributes: dict[str, Any], training_profile: dict[str, Any]
) -> tuple[int, int, int]:
    """The grid decimation a run actually saw: dataset stride x loader stride.

    Two independent sources compose by MULTIPLICATION: `derivation_profile.stride`
    is baked into the served arrays, and `env.PINO_LAZY_STRIDE` is applied by the
    loader on top of whatever it is handed. A pre-strided [2,2,1] copy served with
    the env stride still set really does train on [4,4,1] (the recorded
    double-stride bug), so a resolver that lets one source "win" hides exactly
    the misconfiguration this exists to expose.

    Loud on anything it cannot interpret (C of the 2026-08-09 post-mortem:
    absence must never silently read as a value):

    - malformed stride values raise ``ValueError`` rather than degrading to
      full resolution;
    - a stride-bearing sibling key without `derivation_profile` raises
      ``KeyError`` — that shape means the schema moved (or the caller misread
      it, which is how a correctly-configured run got killed), not "no stride".
    """

    def _parse3(values, source: str) -> tuple[int, int, int]:
        try:
            parts = [int(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} stride {values!r} is not 3 integers") from exc
        if len(parts) != 3 or any(p < 1 for p in parts):
            raise ValueError(f"{source} stride {values!r} is not 3 integers >= 1")
        return tuple(parts)

    attrs = dataset_attributes or {}
    derivation = attrs.get("derivation_profile")
    if derivation is None:
        drifted = [k for k in attrs if "derivation" in str(k).lower()]
        if drifted:
            raise KeyError(
                f"dataset attributes carry {drifted} but no 'derivation_profile'; "
                "refusing to guess the stride from an unexpected schema"
            )
        ds_stride = (1, 1, 1)
    else:
        recorded = derivation.get("stride")
        ds_stride = (1, 1, 1) if recorded is None else _parse3(recorded, "derivation_profile")

    env = (training_profile or {}).get("env") or {}
    raw = str(env.get("PINO_LAZY_STRIDE", "") or "").strip()
    env_stride = (
        _parse3([p for p in raw.replace(" ", "").split(",") if p], "env.PINO_LAZY_STRIDE")
        if raw
        else (1, 1, 1)
    )
    return tuple(d * e for d, e in zip(ds_stride, env_stride))


def validate_profile_keys(profile_name: str, values: dict, required: set[str]) -> None:
    """Validate that a profile dict contains the minimum required keys."""
    if not isinstance(values, dict):
        raise ValueError(f"{profile_name} must be a mapping.")

    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"{profile_name} is missing required keys: {', '.join(missing)}")


def resolve_artifacts_dir(path: str | Path) -> str:
    """The directory that actually holds the checkpoints.

    physicsnemo nests its run output (`outputs/<run>/outputs/<run>/`), and
    pointing the import at the OUTER directory used to succeed silently while
    ingesting wandb logs and the graph cache instead of the model (2026-08-10
    friction, item 5). Descend at most two levels looking for `u_network*.pth`;
    refuse a directory that has no checkpoint anywhere below, since importing
    such a folder can only produce a model node with no weights.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ValueError(f"artifacts path is not a directory: {root}")

    def _has_ckpt(d: Path) -> bool:
        return any(d.glob("u_network*.pth")) or any(d.glob("*.ckpt"))

    if _has_ckpt(root):
        return str(root)
    candidates = []
    for depth1 in sorted(p for p in root.iterdir() if p.is_dir()):
        if _has_ckpt(depth1):
            candidates.append(depth1)
            continue
        for depth2 in sorted(p for p in depth1.iterdir() if p.is_dir()):
            if _has_ckpt(depth2):
                candidates.append(depth2)
    if len(candidates) == 1:
        return str(candidates[0])
    if not candidates:
        raise ValueError(
            f"no checkpoint (u_network*.pth or *.ckpt) in {root} or two levels below; "
            "importing it would register a model with no weights"
        )
    raise ValueError(
        f"{len(candidates)} checkpoint directories below {root} "
        f"({', '.join(str(c.relative_to(root)) for c in candidates[:4])}...); "
        "point --artifacts-path at exactly one run"
    )


def metrics_from_artifacts(folder) -> dict:
    """Metrics the run already wrote, lifted out of its artifact files.

    An import used to land `model_errors.txt` and `validation_metrics.csv` as
    opaque blobs inside the artifacts folder, so after a successful back-fill
    "which seed had the best beam-box error" still meant digging through files
    -- precisely the question the back-fill existed to answer (2026-08-10
    friction, item 6).

    Returns {} when the run wrote nothing parseable; never raises, because a
    missing metrics file is not a reason to lose a model registration.
    """
    import csv as _csv
    import io as _io
    import tempfile as _tempfile
    from pathlib import Path as _Path

    out: dict = {}
    try:
        names = folder.base.repository.list_object_names()
    except Exception:  # noqa: BLE001 - not a folder-shaped node
        return out

    if "model_errors.txt" in names:
        try:
            with _tempfile.TemporaryDirectory() as tmp:
                target = _Path(tmp) / "model_errors.txt"
                target.write_text(
                    folder.base.repository.get_object_content("model_errors.txt"),
                    encoding="utf-8",
                )
                parsed = parse_model_errors(target)
            if parsed:
                out["model_errors"] = parsed
        except Exception:  # noqa: BLE001
            pass

    if "validation_metrics.csv" in names:
        try:
            text = folder.base.repository.get_object_content("validation_metrics.csv")
            rows = list(_csv.DictReader(_io.StringIO(text)))
            if rows:
                # The last row is the final validation; keep it flat and
                # queryable, and keep the row count so a truncated run is visible.
                final = {
                    key: float(value)
                    for key, value in rows[-1].items()
                    if key and value not in (None, "")
                    and _is_number(value)
                }
                if final:
                    out["validation_final"] = final
                    out["validation_rows"] = len(rows)
        except Exception:  # noqa: BLE001
            pass
    return out


def _is_number(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def parse_model_errors(path: str | Path) -> dict[str, dict[str, float]]:
    """Parse the current PINO metric-file format into a nested mapping."""
    metrics: dict[str, dict[str, float]] = {}
    current_key: str | None = None

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_key = line[1:-1]
                metrics[current_key] = {}
                continue
            if current_key is None or ":" not in line:
                continue
            stat_name, value = [part.strip() for part in line.split(":", 1)]
            try:
                metrics[current_key][stat_name] = float(value)
            except ValueError:
                continue

    return metrics


# Moved to `aiida_pino.scripts.live_tail`, which is the ONE module staged beside
# every adapter on a compute box. Re-exported because four call sites import it
# from here (`utils/__init__.py`, `parsers/common.py`, `eval_adapter.py`, and
# the tests), and because a compute box has no `aiida_pino.utils` to reach.
from aiida_pino.scripts.live_tail import parse_stdout_metric_dicts  # noqa: F401,E402

def resolve_cuda_device_for_target(supplied_profile: dict, merged_value: str, *, remote: bool = False) -> str:
    """The card a submission should ask for: the caller's choice, else `"auto"`.

    `cuda_device` ships as `"0"` -- in `DEFAULT_TRAINING_PROFILE`, in every packaged hydra
    config, and in the console form -- so a submission that never mentions the key still
    carries a card. On a two-card box that pins the run to card 0 and serialises it:
    nothing errors, the box just runs at half capacity, which for a repo whose purpose is
    adding GPUs is exactly backwards.

    Whether it IS the default is read from `supplied_profile` -- the dict the caller
    actually passed -- not by comparing the merged value against `"0"`. An explicit `"0"`
    and a defaulted `"0"` are the same string, so the merged profile cannot tell them
    apart, and guessing would quietly overrule someone who asked for card 0 on purpose.
    Present-but-blank is how a form says "the user did not choose": the console assigns
    `cuda_device` on EVERY submission, so a key-presence test alone made this dead code on
    the path that matters most.

    2026-08-23: the promotion used to apply only to REMOTE submissions, on the reasoning
    that "locally there is nothing to promote to". That is false on a multi-GPU
    workstation, and it was measurably costing throughput: of the last 120
    `PinoTrainCalculation`s, all 8 remote ones carried `auto` while **25 local ones were
    pinned to card 0**, and six arms were observed queued on card 0 with card 1 at
    0/22528 MiB reserved. `gpu_spool` resolves `"auto"` by least-committed card locally
    just as it does remotely -- 87 local runs in that same window prove it -- and the
    adapter emits no `custom.cuda_device` token at all for `auto`, so nothing downstream
    sees the literal string. `remote` is kept only so existing callers need not change; it
    no longer affects the result.
    """
    supplied = (supplied_profile or {})
    if "cuda_device" in supplied and str(supplied["cuda_device"] or "").strip():
        return merged_value
    return "auto"


def available_config_names() -> list[str]:
    """Return packaged PINO config names when available."""
    try:
        config_dir = resources.files("lpno.conf.hydra")
    except ModuleNotFoundError:
        return [DEFAULT_TRAINING_PROFILE["config_name"]]

    # All packaged training configs, not just the NS_energy_PINO family — the
    # old prefix filter hid every RIGNO/baseline config from the console and
    # CLI even though submission accepted them.
    names = sorted(
        path.name[:-5]
        for path in config_dir.iterdir()
        if path.name.startswith("config_") and path.suffix == ".yaml"
    )
    return names or [DEFAULT_TRAINING_PROFILE["config_name"]]


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _load_config_yaml(config_name: str) -> dict[str, Any]:
    if yaml is None:
        return {}
    try:
        config_path = resources.files("lpno.conf.hydra").joinpath(f"{config_name}.yaml")
    except ModuleNotFoundError:
        return {}
    if not config_path.is_file():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _load_group_yaml(group: str, choice: Any) -> dict[str, Any]:
    """A config-group file (`conf/hydra/<group>/<choice>.yaml`), or {} if not packaged."""
    if yaml is None:
        return {}
    try:
        path = resources.files("lpno.conf.hydra").joinpath(str(group)).joinpath(f"{choice}.yaml")
    except ModuleNotFoundError:
        return {}
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config_chain(config_name: str, _seen: set[str] | None = None) -> dict[str, Any]:
    """`config_name` merged with the packaged configs it inherits, as hydra composes them.

    `_load_config_yaml` reads one file, but most configs set little of their own: the
    `*_pvlhs_steady` configs inherit `training.max_steps: 32000` and `batch_size: 1`
    from `config_flow3D_*_pvlhs` and its parents, so a default read from the leaf alone
    fell back to this module's 15000 steps and batch 20, and since the train adapter
    emits these keys on every run, a profile that left them out trained at those values.

    Walks the `defaults` list in order: a parent config is merged where it is listed,
    the file's own body at `_self_` (at the end when `_self_` is absent), and a group
    entry (`{loss: relobralo_tuned}`) under its group key when the group file is
    packaged; `override /group` replaces that group. Entries resolved only by
    physicsnemo's config store (`physicsnemo_default`, `/arch/...@...`, `scheduler:
    one_cycle`) contribute nothing here, so their keys keep this module's defaults.
    """
    seen = set() if _seen is None else _seen
    if not config_name or config_name in seen:
        return {}
    seen.add(config_name)
    own = _load_config_yaml(config_name)
    if not own:
        return {}
    body = {key: value for key, value in own.items() if key != "defaults"}
    merged: dict[str, Any] = {}
    self_merged = False
    for entry in own.get("defaults") or []:
        if entry == "_self_":
            merged = _deep_merge(merged, body)
            self_merged = True
        elif isinstance(entry, str):
            if "/" not in entry and "@" not in entry:
                merged = _deep_merge(merged, _load_config_chain(entry, seen))
        elif isinstance(entry, dict):
            for group, choice in entry.items():
                text = str(group)
                replace = text.startswith("override ")
                name = text.replace("override ", "", 1).strip().lstrip("/")
                if replace:
                    merged.pop(name, None)
                group_cfg = _load_group_yaml(name, choice)
                if group_cfg:
                    merged = _deep_merge(merged, {name: group_cfg})
    if not self_merged:
        merged = _deep_merge(merged, body)
    return merged


_ONE_CYCLE_SCHEDULERS = frozenset({"one_cycle"})


def resolve_effective_lr(config_name: str | None) -> float | None:
    """The peak LR `config_name` will actually train at, or None if unreadable.

    `_load_config_yaml` reads ONE yaml, so an inherited `optimizer:`/`scheduler:`
    block is invisible and the default falls back to the packaged constant --
    which is how the pvlhs chain advertised 6e-5 while running 1e-4. Worse, on a
    one-cycle config the binding key is `scheduler.max_lr`: OneCycleLR overwrites
    each param group's lr from it, so `optimizer.lr` is decoration (measured
    2026-08-15 across four campaign arms, all recording 3e-4 and all training at
    1e-4).

    Walks the `defaults` chain child-first (nearest definition wins, matching
    hydra's own precedence) and returns `scheduler.max_lr` when the chain selects
    a one-cycle scheduler, else `optimizer.lr`.
    """
    if not config_name:
        return None
    seen: set[str] = set()
    stack = [str(config_name)]
    max_lr: float | None = None
    opt_lr: float | None = None
    one_cycle: bool | None = None
    while stack:
        name = stack.pop(0)
        if not name or name in seen:
            continue
        seen.add(name)
        cfg = _load_config_yaml(name)
        if not cfg:
            continue
        if max_lr is None:
            max_lr = _as_float((cfg.get("scheduler") or {}).get("max_lr"), None)
        if opt_lr is None:
            opt_lr = _as_float((cfg.get("optimizer") or {}).get("lr"), None)
        for entry in cfg.get("defaults", []) or []:
            if isinstance(entry, dict):
                if "scheduler" in entry and one_cycle is None:
                    one_cycle = str(entry["scheduler"]) in _ONE_CYCLE_SCHEDULERS
            elif isinstance(entry, str) and entry != "_self_":
                stack.append(entry)
    if one_cycle and max_lr is not None:
        return max_lr
    return opt_lr


def load_training_profile_defaults(config_name: str | None = None) -> dict[str, Any]:
    """Load runtime defaults from packaged PINO configs, with a local fallback.

    Read from the config as composed with the configs it inherits
    (`_load_config_chain`), so a key a parent sets is the default here too: the
    `*_pvlhs_steady` configs report their inherited 32000 steps and batch 1.

    Two groups stay read from the named file alone, as before:

    * the layout keys (`include_time`, `time_as_batch`, `time_as_dim`, `use_2d`,
      `moving_ref_frame`, `moving_frame_steps`, `time_avg_window`), over this module's
      steady layout. A layout is stated by every run that needs one, in the training and
      the derivation profile, and selects the dataset variant: inheriting a parent's
      lab-frame layout would let a steady config train on lab-frame data with no layout
      guard firing;
    * the identity and launch keys (`run_name`, `cuda_device`, `wandb_offline`), which
      every submission states (they are required): a parent's `custom.name` would name
      an FNO run after its RIGNO parent, and a parent's `cuda_device: 0` would turn the
      blank default, which `resolve_cuda_device_for_target` reads as "auto", into a pin.
    """
    selected_name = config_name or DEFAULT_TRAINING_PROFILE["config_name"]
    loaded = _load_config_chain(selected_name)
    custom = loaded.get("custom", {})
    leaf = _load_config_yaml(selected_name).get("custom", {}) or {}   # the named file's own
    autoregressive = custom.get("autoregressive", {})

    defaults = deepcopy(DEFAULT_TRAINING_PROFILE)
    defaults["config_name"] = selected_name
    defaults["run_name"] = str(leaf.get("name", defaults["run_name"]))
    defaults["cuda_device"] = "" if leaf.get("cuda_device") is None else str(leaf.get("cuda_device"))
    defaults["wandb_offline"] = _as_bool(leaf.get("wandb_offline"), defaults["wandb_offline"])
    defaults["split_mode"] = str(custom.get("split_mode", defaults["split_mode"])).lower()
    defaults["k_folds"] = _as_int(custom.get("k_folds"), defaults["k_folds"])
    defaults["fold_index"] = _as_int(custom.get("fold_index"), defaults["fold_index"])
    defaults["pde_loss"] = _as_bool(custom.get("pde_loss"), defaults["pde_loss"])
    defaults["vof_pde_loss"] = _as_bool(custom.get("VOF_pde_loss"), defaults["vof_pde_loss"])
    defaults["diff_loss"] = _as_bool(custom.get("Diff_loss"), defaults["diff_loss"])
    defaults["diff_weight"] = _as_float(custom.get("Diff_weight"), defaults["diff_weight"])
    defaults["pde_weight"] = _as_float(custom.get("pde_weight"), defaults["pde_weight"])
    defaults["include_time"] = _as_bool(leaf.get("include_time"), defaults["include_time"])
    defaults["time_as_batch"] = _as_bool(leaf.get("time_as_batch"), defaults["time_as_batch"])
    defaults["time_as_dim"] = _as_bool(leaf.get("time_as_dim"), defaults["time_as_dim"])
    defaults["use_2d"] = _as_bool(leaf.get("use_2D"), defaults["use_2d"])
    defaults["time_avg_window"] = _as_int(leaf.get("time_avg_window"), defaults["time_avg_window"])
    defaults["moving_ref_frame"] = _as_bool(leaf.get("moving_ref_frame"), defaults["moving_ref_frame"])
    defaults["moving_frame_steps"] = _as_int(leaf.get("moving_frame_steps"), defaults["moving_frame_steps"])
    defaults["fast_training"] = _as_bool(custom.get("fast_training"), defaults["fast_training"])
    defaults["detect_anomaly"] = _as_bool(custom.get("detect_anomaly"), defaults["detect_anomaly"])
    defaults["sharp_alpha"] = _as_bool(custom.get("sharp_alpha"), defaults["sharp_alpha"])
    defaults["alpha_masking"] = _as_bool(custom.get("alpha_masking"), defaults["alpha_masking"])
    training = loaded.get("training", {})
    optimizer = loaded.get("optimizer", {})
    scheduler = loaded.get("scheduler", {})
    batch_size = loaded.get("batch_size", {})
    defaults["max_steps"] = _as_int(training.get("max_steps"), defaults["max_steps"])
    # Chain-resolved and scheduler-aware: `optimizer.lr` from this one yaml is
    # neither inherited nor the binding key under one-cycle (see
    # `resolve_effective_lr`). Falls back to the old local read when the chain
    # cannot be resolved, so an unreadable config degrades rather than lies.
    defaults["learning_rate"] = _as_float(
        resolve_effective_lr(selected_name),
        _as_float(optimizer.get("lr"), defaults["learning_rate"]),
    )
    defaults["weight_decay"] = _as_float(optimizer.get("weight_decay"), defaults["weight_decay"])
    defaults["batch_size_grid"] = _as_int(batch_size.get("grid"), defaults["batch_size_grid"])
    defaults["batch_size_validation"] = _as_int(batch_size.get("validation"), defaults["batch_size_validation"])
    defaults["scheduler_decay_rate"] = _as_float(
        scheduler.get("decay_rate"), defaults["scheduler_decay_rate"]
    )
    defaults["scheduler_decay_steps"] = _as_int(
        scheduler.get("decay_steps"), defaults["scheduler_decay_steps"]
    )
    defaults["inference_batch_size"] = _as_int(
        custom.get("inference_batch_size"), defaults["inference_batch_size"]
    )
    defaults["infer_velocities"] = _as_bool(custom.get("infer_velocities"), defaults["infer_velocities"])
    defaults["autoregressive"] = {
        "enabled": _as_bool(autoregressive.get("enabled"), defaults["autoregressive"]["enabled"]),
        "mode": str(autoregressive.get("mode", defaults["autoregressive"]["mode"])),
        "state_prev_suffix": str(
            autoregressive.get("state_prev_suffix", defaults["autoregressive"]["state_prev_suffix"])
        ),
        "patch_size": [
            int(value)
            for value in autoregressive.get("patch_size", defaults["autoregressive"]["patch_size"])
        ],
        "context_steps": _as_int(
            autoregressive.get("context_steps"), defaults["autoregressive"]["context_steps"]
        ),
        "mask_mode": str(autoregressive.get("mask_mode", defaults["autoregressive"]["mask_mode"])),
        "mask_ratio": _as_float(
            autoregressive.get("mask_ratio"), defaults["autoregressive"]["mask_ratio"]
        ),
        "train_patches_per_epoch": _as_int(
            autoregressive.get("train_patches_per_epoch"),
            defaults["autoregressive"]["train_patches_per_epoch"],
        ),
        "val_patches_per_epoch": _as_int(
            autoregressive.get("val_patches_per_epoch"),
            defaults["autoregressive"]["val_patches_per_epoch"],
        ),
        "eval_horizon_k": _as_int(
            autoregressive.get("eval_horizon_k"), defaults["autoregressive"]["eval_horizon_k"]
        ),
        "eval_use_gt_init": _as_bool(
            autoregressive.get("eval_use_gt_init"), defaults["autoregressive"]["eval_use_gt_init"]
        ),
        "require_time": _as_bool(autoregressive.get("require_time"), defaults["autoregressive"]["require_time"]),
    }
    # Detect RIGNO from the config's own arch block rather than the config name, so
    # a renamed or user-added RIGNO config still gets the downsampled default.
    if RIGNO_DEFAULT_SPATIAL_STRIDE and "rigno" in (loaded.get("arch") or {}):
        defaults["env"] = {**defaults["env"], "PINO_LAZY_STRIDE": RIGNO_DEFAULT_SPATIAL_STRIDE}
    return defaults


# The full-resolution host-RAM floor, scaled by grid size.
#
# 40960 came from ONE measurement -- a 600k-node GAOT graph build peaking at ~57 GB
# against gpu_spool's 1.5x cgroup enforcement -- and was then applied to every
# full-resolution run regardless of size. On a 180k-node grid that books 40 GB for a
# ~10 GB peak, and two such arms then cannot share a 125 GB box.
#
# Two measured anchors (peak host RAM of two training runs):
#     97.6k nodes -> 2.4 GB peak (straight-path moving frame, flat, no transient)
#     600k nodes  -> 57.2 GB peak (150x100x40 full)
# The ratio 6.1x nodes -> 24x memory gives an exponent of ~1.77; peak tracks EDGES, not
# cells, which is why it is superlinear. Anchored at 600k so the historical floor is
# reproduced exactly and larger grids demand MORE.
_FLOOR_ANCHOR_NODES = 600_000
_FLOOR_ANCHOR_MB = 40960
_FLOOR_EXPONENT = 1.77
_FLOOR_MIN_MB = 8192


def full_res_host_ram_floor_mb(graph_nodes: int | None) -> int:
    """Minimum full-resolution `host_ram_mb`, scaled by the run's node count.

    `graph_nodes` absent or zero returns the historical flat floor, so every profile
    that does not declare a grid keeps exactly its previous behaviour.
    """
    if not graph_nodes or graph_nodes <= 0:
        return _FLOOR_ANCHOR_MB
    scaled = _FLOOR_ANCHOR_MB * (float(graph_nodes) / _FLOOR_ANCHOR_NODES) ** _FLOOR_EXPONENT
    return int(max(_FLOOR_MIN_MB, min(scaled, _FLOOR_ANCHOR_MB * 4)))


def merge_training_profile(values: dict[str, Any], *, for_import: bool = False) -> dict[str, Any]:
    """Merge a partial profile with packaged defaults."""
    if not isinstance(values, dict):
        raise ValueError("training_profile must be a mapping.")

    aliased = dict(values)
    if "VOF_pde_loss" in aliased and "vof_pde_loss" not in aliased:
        aliased["vof_pde_loss"] = aliased["VOF_pde_loss"]
    if "Diff_loss" in aliased and "diff_loss" not in aliased:
        aliased["diff_loss"] = aliased["Diff_loss"]
    if "Diff_weight" in aliased and "diff_weight" not in aliased:
        aliased["diff_weight"] = aliased["Diff_weight"]
    if "use_2D" in aliased and "use_2d" not in aliased:
        aliased["use_2d"] = aliased["use_2D"]

    validate_profile_keys(
        "training_profile",
        aliased,
        set(IMPORT_PROFILE_REQUIRED_KEYS if for_import else TRAINING_PROFILE_REQUIRED_KEYS),
    )
    defaults = load_training_profile_defaults(str(aliased.get("config_name", DEFAULT_TRAINING_PROFILE["config_name"])))
    merged = deepcopy(defaults)
    merged.update({key: value for key, value in aliased.items() if key != "autoregressive"})
    merged["hydra_overrides"] = list(merged.get("hydra_overrides", []))

    # env merges per key, like `autoregressive`: a caller adding one variable must
    # not silently drop a packaged default (e.g. RIGNO's PINO_LAZY_STRIDE).
    provided_env = aliased.get("env")
    if provided_env is not None and not isinstance(provided_env, dict):
        raise ValueError("training_profile.env must be a mapping.")
    merged["env"] = {**deepcopy(defaults.get("env") or {}), **(provided_env or {})}

    merged_autoregressive = deepcopy(defaults["autoregressive"])
    provided_autoregressive = aliased.get("autoregressive", {})
    if provided_autoregressive is None:
        provided_autoregressive = {}
    if not isinstance(provided_autoregressive, dict):
        raise ValueError("training_profile.autoregressive must be a mapping.")
    merged_autoregressive.update(provided_autoregressive)
    merged["autoregressive"] = merged_autoregressive
    # Stamp the power-weighted-dose marker for NEW trainings: unset -> ON. Import
    # of a finished run must NOT be stamped — inventing True for a run that
    # trained power-blind would be fabricated provenance; its evals rely on the
    # eval-merge pinning None -> False instead.
    if not for_import and merged.get("power_weighted_dose") is None:
        merged["power_weighted_dose"] = True
    return merged


def validate_training_profile(
    values: dict[str, Any],
    expected_split_mode: str | None = None,
    *,
    for_import: bool = False,
    for_launch: bool = True,
) -> dict[str, Any]:
    """Normalize and validate the PINO runtime profile.

    `for_launch=False` skips the host-RAM/loader-worker resource check. That check asks
    "can this profile safely START a training", so it is only meaningful before a launch.
    Applied to a FROZEN profile of a training that already finished, it can only ever
    block a read: the training is history and cannot be re-declared, and an eval takes
    its resources from the evaluation profile, not from this one. It blocked the
    beam-box eval of a random-path model (one that trained fine at
    host_ram_mb=32768 with 4 workers) behind a floor calibrated for a 600k-node GAOT
    graph build it never performs.

    `for_import` drops the launch-only keys (see IMPORT_PROFILE_REQUIRED_KEYS).
    """
    profile = merge_training_profile(values, for_import=for_import)

    if not isinstance(profile["hydra_overrides"], list) or not all(
        isinstance(item, str) for item in profile["hydra_overrides"]
    ):
        raise ValueError("training_profile.hydra_overrides must be a list of strings.")

    split_mode = str(profile["split_mode"]).lower()
    if split_mode not in {"standard", "kfold"}:
        raise ValueError("training_profile.split_mode must be one of: standard, kfold.")
    profile["split_mode"] = split_mode

    if expected_split_mode is not None and split_mode != expected_split_mode:
        raise ValueError(f"training_profile.split_mode must be `{expected_split_mode}` for this workflow.")

    if int(profile["k_folds"]) <= 1:
        raise ValueError("training_profile.k_folds must be greater than 1.")
    if int(profile["max_steps"]) <= 0:
        raise ValueError("training_profile.max_steps must be positive.")
    if float(profile["learning_rate"]) <= 0:
        raise ValueError("training_profile.learning_rate must be positive.")
    if float(profile["weight_decay"]) < 0:
        raise ValueError("training_profile.weight_decay must be non-negative.")
    if int(profile["batch_size_grid"]) <= 0:
        raise ValueError("training_profile.batch_size_grid must be positive.")
    if int(profile["batch_size_validation"]) <= 0:
        raise ValueError("training_profile.batch_size_validation must be positive.")
    # None is an explicit opt-out: the adapters then emit no scheduler.decay_*
    # override, which a config whose scheduler group has no such keys (e.g. the
    # one_cycle scheduler) needs — hydra struct mode rejects unknown keys.
    if profile["scheduler_decay_rate"] is not None and float(profile["scheduler_decay_rate"]) <= 0:
        raise ValueError("training_profile.scheduler_decay_rate must be positive.")
    if profile["scheduler_decay_steps"] is not None and int(profile["scheduler_decay_steps"]) <= 0:
        raise ValueError("training_profile.scheduler_decay_steps must be positive.")
    if int(profile["inference_batch_size"]) <= 0:
        raise ValueError("training_profile.inference_batch_size must be positive.")
    if int(profile["time_avg_window"]) <= 0:
        raise ValueError("training_profile.time_avg_window must be positive.")
    if int(profile["moving_frame_steps"]) <= 0:
        raise ValueError("training_profile.moving_frame_steps must be positive.")

    if profile["time_as_dim"] and not profile["include_time"]:
        raise ValueError("training_profile.time_as_dim=True requires include_time=True.")
    if profile["time_as_batch"] and profile["time_as_dim"]:
        raise ValueError("training_profile.time_as_batch=True is incompatible with time_as_dim=True.")

    autoregressive = profile["autoregressive"]
    if bool(autoregressive["enabled"]):
        if not profile["time_as_batch"]:
            raise ValueError("training_profile.autoregressive.enabled=True requires time_as_batch=True.")
        if bool(autoregressive["require_time"]) and not profile["include_time"]:
            raise ValueError("training_profile.autoregressive.enabled=True requires include_time=True.")
        if profile["pde_loss"] or profile["vof_pde_loss"] or profile["diff_loss"]:
            raise ValueError("Autoregressive mode currently supports data-only training.")
        # Both modes the trainer implements: `rollout`
        # (fno/rigno/deeponet, next-state feedback -- the k-step unrolled recipe)
        # and `masked_patch` (fno, lazy-store only). Allowing only masked_patch
        # here made every rollout training unreachable through AiiDA.
        if str(autoregressive["mode"]).strip().lower() not in {"masked_patch", "rollout"}:
            raise ValueError(
                "training_profile.autoregressive.mode must be one of: masked_patch, rollout."
            )
        if int(autoregressive["context_steps"]) != 1:
            raise ValueError("training_profile.autoregressive.context_steps currently only supports 1.")
        patch_size = list(autoregressive["patch_size"])
        if len(patch_size) != 3 or any(int(value) <= 0 for value in patch_size):
            raise ValueError("training_profile.autoregressive.patch_size must be three positive integers.")
        if str(autoregressive["mask_mode"]).strip().lower() not in {"voxel", "block"}:
            raise ValueError("training_profile.autoregressive.mask_mode must be one of: voxel, block.")
        if not 0.0 <= float(autoregressive["mask_ratio"]) <= 1.0:
            raise ValueError("training_profile.autoregressive.mask_ratio must be in [0, 1].")
        if int(autoregressive["train_patches_per_epoch"]) <= 0:
            raise ValueError("training_profile.autoregressive.train_patches_per_epoch must be positive.")
        if int(autoregressive["val_patches_per_epoch"]) <= 0:
            raise ValueError("training_profile.autoregressive.val_patches_per_epoch must be positive.")

    if (profile["use_2d"] or profile["time_as_dim"]) and (
        profile["pde_loss"] or profile["vof_pde_loss"] or profile["diff_loss"]
    ):
        raise ValueError("2D and time_as_dim modes currently support data-only training.")

    # An IMPORT is not a launch, and both of these are launch-only questions.
    # `import-model` registers a model from artifacts of a training that already ran:
    # it starts nothing, takes no GPU and no host RAM. Running them anyway meant an
    # import printed "cuda_device='0' pins this run to ONE gpu" about a run that does
    # not exist (14 of 61 packaged configs declare a card, and the profile pre-fill
    # comes from the config), and -- the part that is not cosmetic -- the host-RAM
    # check RAISES, so an import could be REFUSED over loader settings of a training
    # that had already finished. Exactly the failure the `for_launch` docstring above
    # describes for the frozen profile of a random-path model.
    if for_launch and not for_import:
        _check_host_ram_against_loader_settings(profile)
        _warn_if_host_ram_is_over_declared(profile)
        _warn_if_gpu_vram_is_under_declared(profile)
        _warn_if_gpu_is_pinned(profile)

    return profile


def _warn_if_gpu_is_pinned(profile: dict[str, Any]) -> None:
    """Say so when a profile targets ONE card instead of `auto`.

    A pin costs nothing until the box is busy, and then it costs everything: the job can only
    wait for its own card, so it sits out every card that frees first. Three arms lost a combined
    ~7 hours to this on 2026-08-21 -- rigno_1601_35k_varspeed_recipe_v5 and
    gaot_1601_20k_varspeed_recipe_v2 (both cuda_device="0"), and
    fno_spkh123_mref_32k_deadchannels_dropped (cuda_device="1"), which logged ZERO steps in
    3h20m while its A/B sibling ran to completion on the card it was waiting for.

    None of the three meant to pin: each inherited the field from the profile it was cloned from.
    That is what makes it worth announcing at submission rather than leaving to monitoring -- the
    symptom is a job that looks alive in every process listing and simply never starts, and it
    was misread as healthy contention three separate times before the pattern was spotted.

    A warning, not a refusal: pinning is legitimate when you deliberately want one card. It just
    should never happen by accident.
    """
    device = str(profile.get("cuda_device") or "auto").strip().lower()
    if device in {"auto", "none", ""}:
        return
    print(
        f"[profile] WARNING cuda_device={device!r} pins this run to ONE gpu; it will not take "
        f"another card that frees first. Use 'auto' unless the pin is deliberate.",
        flush=True,
    )


HOST_RAM_PEAK_EXTRA = "host_ram_peak_mb"
GPU_VRAM_PEAK_EXTRA = "gpu_vram_peak_mb"
# How many recent peaks to consider, matching `vram.LOOKUP_RECENT`.
HOST_RAM_LOOKUP_RECENT = 5


def _recorded_peak_mb(extra: str, config_name, full_resolution=None) -> int | None:
    """Largest peak recorded under `extra` for this config at this resolution.

    Host RAM and VRAM are the same query against the same model extras -- every finished
    training already writes both -- so this needs no new bookkeeping. `max` over a short
    recent window for the reason given in `vram.lookup_vram_peak_mb`: within one config
    the peaks estimate the same number, and the window ages out a stale one.

    Returns None when nothing comparable is recorded, which is the common case for a new
    config and MUST leave the submission alone.
    """
    if not config_name:
        return None
    try:
        from aiida import orm

        filters: dict[str, Any] = {
            "extras.config_name": str(config_name),
            f"extras.{extra}": {">": 0},
        }
        if full_resolution is not None:
            filters["extras.full_resolution"] = bool(full_resolution)
        query = (
            orm.QueryBuilder()
            .append(orm.Node, filters=filters, project=[f"extras.{extra}"])
            .order_by({orm.Node: {"ctime": "desc"}})
            .limit(HOST_RAM_LOOKUP_RECENT)
        )
        peaks = [int(r[0]) for r in query.all() if r[0]]
    except Exception:  # noqa: BLE001 - no profile loaded, no DB, nothing to advise on
        return None
    return max(peaks) if peaks else None


def recorded_host_ram_peak_mb(config_name, full_resolution=None) -> int | None:
    """Largest host-RAM peak recently recorded for this config at this resolution."""
    return _recorded_peak_mb(HOST_RAM_PEAK_EXTRA, config_name, full_resolution)


def recorded_gpu_vram_peak_mb(config_name, full_resolution=None) -> int | None:
    """Largest GPU VRAM peak recently recorded for this config at this resolution."""
    return _recorded_peak_mb(GPU_VRAM_PEAK_EXTRA, config_name, full_resolution)


def _warn_if_gpu_vram_is_under_declared(profile: dict[str, Any]) -> None:
    """Say so when a manual VRAM booking is below what this config has measured.

    An under-declared booking is not a tight budget, it is an invisible one: the ledger
    admits the job against the number it claimed and the allocator then exceeds it, so the
    run dies holding a seat and returns nothing. Six eval jobs failed this way on
    2026-08-22 with a booking about a quarter of what the architecture needs.

    This is only reachable for a MANUAL reservation. A profile that leaves `gpu_vram_mb`
    unset is sized automatically from the footprint key (`vram.lookup_vram_peak_mb`, plus
    `vram_reserve_mb`'s margin) and never lands here -- which is why the advice below is
    "delete the field", not "raise it to N".

    Config-level rather than footprint-level on purpose: at validation time the dataset
    layout that completes a footprint key is not resolved yet, and a coarse comparison that
    catches a 4x gap is worth more than a precise one that cannot run.

    Coarse means the threshold has to be generous. One config name covers runs of very
    different size -- `config_flow3D_RIGNO_randompath` has recorded peaks from 10634 to
    19574 MiB -- so comparing a booking against the config's LARGEST peak flags every
    smaller variant. Measured over the 40 most recent models with a recorded peak: firing
    on `declared < peak` gave 6 hits of which 5 were the same near-miss (10000 booked, own
    peak 10634, config max 19574), while firing at half the peak leaves all five quiet and
    still catches the quarter-sized booking this exists for.
    """
    declared = profile.get("gpu_vram_mb")
    if not declared:
        return  # unset means auto-sized from the measured footprint, which is the good path
    peak = recorded_gpu_vram_peak_mb(profile.get("config_name"))
    if not peak or int(declared) * 2 >= peak:
        return
    from aiida_pino.utils.vram import vram_reserve_mb

    print(
        f"[profile] WARNING gpu_vram_mb={int(declared)} is BELOW the largest peak recorded "
        f"for config_name={profile.get('config_name')!r} ({peak} MiB over the last "
        f"{HOST_RAM_LOOKUP_RECENT} runs). The ledger admits a job against what it declares, "
        f"so this is admitted and then killed by the allocator, losing the seat and the "
        f"result. Legal band for this config: >= {peak} MiB, and {vram_reserve_mb(peak)} MiB "
        f"is what the automatic sizing would book. Deleting gpu_vram_mb entirely is usually "
        f"right -- it is then measured, not guessed.",
        flush=True,
    )


def _warn_if_host_ram_is_over_declared(profile: dict[str, Any]) -> None:
    """Say so when a booking is far above what this config has ever measured.

    The floor check below is the dangerous direction and refuses. This is the other one,
    and it only warns: host RAM is booked machine-wide, so an over-declaration does not
    fail anything -- it silently halves the box. Measured (recorded in
    `_check_host_ram_against_loader_settings`): arms serving a cropped 90x40x30 booked
    40960 MiB while peaking at 8188 and 9673, and against ~88542 MiB usable that admitted
    TWO concurrent arms on a four-card box. Both machines sat half-idle with work queued
    while VRAM looked like the binding constraint.

    Deliberately NOT a flat ceiling. The obvious 20480 would be wrong: GAOT_randompath has
    a recorded peak of 56861 MiB, so a flat cap refuses the largest legitimate run on the
    box. The comparison is against THIS config's own recorded peaks, and with none
    recorded it says nothing at all.

    A warning, not a refusal, and the asymmetry is deliberate: an under-declaration
    OOM-kills co-tenants, while an over-declaration only wastes a slot the submitter can
    reclaim by re-reading this line. A refusing gate would turn a queue inefficiency into
    a dead submission, which is the more expensive failure.
    """
    declared = profile.get("host_ram_mb")
    if not declared:
        return
    env = profile.get("env") or {}
    reduced = str(env.get("PINO_LAZY_STRIDE", "") or "").strip() not in {"", "1,1,1"}
    try:
        reduced = reduced or int(str(env.get("PINO_LAZY_CROP_Z_TOP", "") or 0) or 0) > 0
    except ValueError:
        pass
    peak = recorded_host_ram_peak_mb(profile.get("config_name"), full_resolution=not reduced)
    if not peak:
        return  # nothing measured for this config: no basis to advise, so do not
    # Twice the measured peak plus 4 GiB of slack. Peaks grow late in training and a
    # booking should sit comfortably above the last one; this fires only on the
    # order-of-magnitude gap the incident above was.
    ceiling = peak * 2 + 4096
    if int(declared) <= ceiling:
        return
    print(
        f"[profile] WARNING host_ram_mb={int(declared)} is far above the largest peak "
        f"recorded for config_name={profile.get('config_name')!r} at this resolution "
        f"({peak} MiB over the last {HOST_RAM_LOOKUP_RECENT} runs). Host RAM is booked "
        f"machine-wide, so this reserves capacity nothing will use and can leave a box "
        f"half-idle with work queued. Consider <= {ceiling} MiB unless this run is known "
        f"to be bigger than its predecessors.",
        flush=True,
    )


#: Cached cases charged nothing, so the eval adapter's default of 3 keeps passing.
CACHE_FREE_CASES = 3
#: A LOWER bound on one cached case, and MEASURED to be one: a controlled pair of runs
#: (cache 24, peak 20612 MiB / cache 96, peak 40018 MiB) -- same dataset, workers 1,
#: stride 2,2,1, crop 10 -- gives a marginal 269.5 MiB per cached case, so 256 sits
#: ~5 % under. Deliberately NOT raised to the measurement: one pair calibrates that dataset,
#: stride and crop, and the per-case cost scales with frames per case and field count. At
#: cache 96 the term is ~63 % of the run's whole footprint, which is why it was invisible.
CACHED_CASE_FLOOR_MB = 256


def _check_host_ram_against_loader_settings(profile: dict[str, Any]) -> None:
    """Reject a host-RAM declaration too small for this profile's loader settings.

    The gate that admits a run books DECLARED host RAM, so an under-declaration is not
    a tight budget -- it is an invisible one: the job is admitted and then exceeds what
    it claimed, taking co-tenants down with it.

    Measured incident (2026-08-05, `gaot_varspeed_multitau_v2`): a profile cloned from a
    working RIGNO run kept `PINO_LOADER_WORKERS=4` and `host_ram_mb=32768` while
    correctly DROPPING that run's `PINO_LAZY_STRIDE=2,2,1`. The workers and the RAM
    figure were only safe because of the stride -- removing it multiplies every cached
    case ~4x -- and a DataLoader worker was OOM-killed by the kernel at the first
    validation pass with the box at 85 GB.

    Each loader worker holds its own copy of the case cache, so the requirement scales
    with worker count and shrinks with the stride.

    This used to advise `PINO_LOADER_WORKERS=0` as the cheap fix, citing 144.6 s against
    129.2 s over 301 frames (aiida-pino `7414653`). **That advice inverted the standing
    recipe and is withdrawn.** `7414653` is dated 2026-08-05, a day BEFORE
    `case_coherent_sampling` existed, so its "workers are slower" measurement is workers
    WITHOUT coherence -- the mode that duplicates whole-case decompression per worker, and
    the same mode as the OOM this guard refuses. It was quoting the failure mode as
    evidence against the fix.

    The measured recipe is `case_coherent_sampling: true` + `PINO_LOADER_WORKERS=2` +
    `host_ram_mb=40960`: 0.67 s/step against 4.70 s/step, 7x on identical hardware.
    Neither knob works alone. A `time_avg_window: 30` config reads ~30 frames per sample
    and is MORE loader-bound, not less.

    Found 2026-08-26, after this text steered a submission to
    workers=0 and then to coherence-without-workers -- the two documented failure modes in
    sequence, costing two restarts.

    And worker count is NOT only a throughput knob. Under `case_coherent_sampling=true` it
    sets how many case-streams interleave, so `PINO_LOADER_WORKERS=0` collapses to ONE
    stream and maximally correlated consecutive steps; we measured 3x on
    final substrate error at seed 42 with an otherwise identical resolved config.
    Treat it as a hyperparameter: carry the
    donor's value when cloning, and never vary it silently between A/B arms. That is why
    `env.PINO_LOADER_WORKERS` scores as a `confound` in `compare_models` rather than as
    bookkeeping.
    """
    env = profile.get("env") or {}
    declared = profile.get("host_ram_mb")
    if not declared:
        return  # nothing booked, nothing to check

    try:
        workers = int(env.get("PINO_LOADER_WORKERS", 0) or 0)
    except (TypeError, ValueError):
        workers = 0
    if workers <= 0:
        return  # serial loading: the main process holds one cache, base is enough

    stride = str(env.get("PINO_LAZY_STRIDE", "") or "").strip()
    strided = bool(stride and stride not in {"1,1,1"})
    # A read-time CROP shrinks the cached case exactly as a stride does, and this check
    # could not see it: arms serving 90x40x30 via `PINO_LAZY_CROP_Z_TOP=10` were charged
    # the full-resolution floor and booked 40960 MiB while measurably using 8188 and
    # 9673 MiB (two runs). The largest `host_ram_peak_mb` ever recorded on any
    # RIGNO run is 16641. Host RAM is booked machine-wide against ~88542 MiB usable, so the
    # over-declaration admitted TWO concurrent arms on a four-card box -- both machines sat
    # half-idle with work queued while VRAM looked like the binding constraint.
    crop = str(env.get("PINO_LAZY_CROP_Z_TOP", "") or "").strip()
    try:
        cropped = int(crop) > 0 if crop else False
    except ValueError:
        cropped = False
    reduced = strided or cropped
    # `PINO_LAZY_CACHE` is a case COUNT, and the cases stay resident. Nothing here read it,
    # while `vram.py` already lists it in MEMORY_RELEVANT_ENV_KEYS and `eval_adapter`
    # defaults evals to 3 -- one module calling it memory-relevant while admission ignored
    # it. A staged kh40 case measured ~470 MiB post-stride, so
    # `PINO_LAZY_CACHE=96` is ~45 GB, several times the whole modelled requirement, so
    # booking the modelled floor while caching 96 cases under-declares by tens of GB. What
    # that costs today is ADMISSION: gpu_spool packs runs by the declared figure, so an
    # under-declared run is placed beside co-tenants the box cannot actually hold. It is NOT
    # the cgroup kill this comment first described -- the 1.5x MemoryMax (BUG-43, fb6dd91)
    # became opt-in via PINO_SPOOL_ENFORCE_RAM in lpno f90778e (2026-08-06),
    # default off, and nothing sets it on either box (checked the compute box's checkout and login
    # env, 2026-09-04). One run peaked at 38542 MiB against a 24576 declaration and finished
    # exit 0 for exactly that reason.
    #
    # 256 MiB per case is a deliberate LOWER BOUND, not a model of the real footprint: it
    # is well under the one measurement available, so it catches gross under-declaration
    # without refusing a booking that is merely tight. It is NOT calibrated: 391 of 590
    # stored models set the variable, but every one at 1 or 2 (372 with a measured peak,
    # cache-1 median 10536 MiB), so they calibrate the FLOOR above, not this term. The first
    # run that caches heavily and records a peak should replace this.
    #
    # The first CACHE_FREE_CASES are charged nothing so the eval adapter's default of 3 does
    # not start refusing evals that run today.
    try:
        cached_cases = int(str(env.get("PINO_LAZY_CACHE", "") or 0) or 0)
    except (TypeError, ValueError):
        cached_cases = 0  # unparseable is unknown, and unknown must not crash a booking check
    # Calibrated against a run that is KNOWN to work rather than against the OOM this
    # check was written for. gaot_cond_disp_tail (calc 30444) trained 20k steps on the
    # full 150x100x40 grid with PINO_LOADER_WORKERS=4 and host_ram_mb=32768, at
    # 1911 ms/iter -- so 4 workers at full resolution need AT MOST 32768, i.e. 4096 per
    # worker over a 16384 base. An earlier version of this function demanded 8192 per
    # worker and therefore rejected that proven configuration outright.
    #
    # What this check does NOT protect against, and must not pretend to: the OOM that
    # motivated it was two CONCURRENT full-resolution graph builds (~57 GB each), not
    # the worker count -- 30444 used 4 workers and was fine. Build concurrency is the
    # build stagger's job.
    graph_nodes = profile.get('graph_nodes')
    per_worker_mb = 1024 if reduced else 4096
    required = 16384 + workers * per_worker_mb
    if cached_cases > CACHE_FREE_CASES:
        required += (cached_cases - CACHE_FREE_CASES) * CACHED_CASE_FLOOR_MB

    # A full-resolution GAOT graph build peaks at ~57 GB (measured VmHWM, 600k nodes /
    # 7.2M edges), and gpu_spool ENFORCES the declaration as a cgroup ceiling at 1.5x
    # (memory_scope_prefix, BUG-43). A declaration below peak/1.5 = 39 GB therefore does
    # not merely under-book -- it guarantees the kernel kills the run at graph
    # construction. That killed the varspeed multi-tau arm four times (16384, 32768,
    # 32768 -> 24, 48, 48 GB ceilings against a 57 GB peak) while presenting as a
    # mysterious SIGTERM: gpu_spool's seat propagates the child's -15 and Python masks
    # it to exit status 241, which names neither memory nor the cap.
    if workers > 0 and (not reduced or graph_nodes):
        # Grid-aware: the 40960 anchor is the 600k-node build. A profile that declares
        # `graph_nodes` gets the floor its own grid implies; one that does not keeps the
        # historical flat value.
        #
        # The flat anchor was measured on the uncompressed 150x100x40 grid, so it does not
        # describe a grid that has been reduced at read time -- that is the crop fix above.
        # But `graph_nodes` DOES describe the real graph however the grid was reduced, so a
        # profile that declares one keeps its floor either way: a 600k-node GAOT build peaks
        # near 57 GB whether or not the loader crops z, and that is what this floor exists
        # for.
        #
        # A measurement made here and later withdrawn: "of 446 stored profiles, ZERO
        # declare both a stride and `graph_nodes`, so this refuses nothing that already
        # runs". The count was right and the inference was wrong -- zero profiles declared
        # `graph_nodes` AT ALL, because the key was missing from `DEFAULT_TRAINING_PROFILE`
        # and `TRAINING_PROFILE_KEYS` is built from it, so every submission naming it was
        # refused as an unknown key (fixed in `ec16e8f`). The query answered "how many got
        # past a check that rejected all of them", not "how rare is this combination".
        # The rule stands on its own reasoning instead: `graph_nodes` describes the real
        # graph however the grid was reduced, so a profile that declares one should be held
        # to the floor that node count implies.
        required = max(required, full_res_host_ram_floor_mb(graph_nodes))

    if int(declared) < required:
        from aiida_pino.utils.guards import resource_guard

        resource_guard(exc=ValueError, message=(
            f"training_profile.host_ram_mb={declared} is too small for "
            f"PINO_LOADER_WORKERS={workers} at "
            f"{'reduced' if reduced else 'FULL'} resolution "
            f"(PINO_LAZY_STRIDE={stride or 'unset'}, "
            f"PINO_LAZY_CROP_Z_TOP={crop or 'unset'}, "
            f"PINO_LAZY_CACHE={cached_cases or 'unset'}); need >= {required} MiB. "
            "Each loader worker holds its own case cache, and dropping a stride "
            "inherited from another profile multiplies every cached case ~4x -- that "
            "combination OOM-killed gaot_varspeed_multitau_v2 on 2026-08-05. The fix is "
            "the standing recipe, NOT fewer workers: raise host_ram_mb to >= 40960 AND "
            "set `+custom.case_coherent_sampling=true`. Workers WITHOUT coherence "
            "duplicate whole-case decompression per worker, which is both the OOM above "
            "and the slowness; coherence without workers leaves the GPU idle behind a "
            "single-threaded loader. Measured 2026-08-08: 0.67 s/step with the recipe "
            "against 4.70 s/step without."
        ))
