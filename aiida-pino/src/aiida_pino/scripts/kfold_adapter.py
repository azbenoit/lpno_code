"""Standalone adapter for the packaged PINO k-fold entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path

try:
    from aiida_pino.scripts.live_tail import profile_environment, run_streamed_subprocess, run_with_live_tail
except ImportError:
    _LIVE_TAIL_SPEC = importlib.util.spec_from_file_location("aiida_pino_live_tail", Path(__file__).with_name("live_tail.py"))
    if _LIVE_TAIL_SPEC is None or _LIVE_TAIL_SPEC.loader is None:
        raise
    _LIVE_TAIL_MODULE = importlib.util.module_from_spec(_LIVE_TAIL_SPEC)
    _LIVE_TAIL_SPEC.loader.exec_module(_LIVE_TAIL_MODULE)
    profile_environment = _LIVE_TAIL_MODULE.profile_environment
    run_streamed_subprocess = _LIVE_TAIL_MODULE.run_streamed_subprocess
    run_with_live_tail = _LIVE_TAIL_MODULE.run_with_live_tail

try:
    from aiida_pino.scripts.staged_dataset import (
        runtime_repair_command as _shared_runtime_repair_command,
        validate_dataset_input_dir as _shared_validate_dataset_input_dir,
    )
except ImportError:  # a compute box has no aiida_pino; the file is staged beside us
    import importlib.util as _ilu

    _SPEC = _ilu.spec_from_file_location("aiida_pino_staged_dataset", Path(__file__).with_name("staged_dataset.py"))
    _MODULE = _ilu.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    _shared_validate_dataset_input_dir = _MODULE.validate_dataset_input_dir
    _shared_runtime_repair_command = _MODULE.runtime_repair_command


PACKAGE_INSTALL_HINT = (
    "`lpno` is not installed in this Code environment. "
    "Install the sibling checkout with `pip install -e /path/to/lpno`."
)
KFOLD_MODULE = "lpno.models.NS_Energy_laser_PINO_Kfolds"
GPU_SPOOL_MODULE = "lpno.gpu_spool"


def wrap_with_gpu_seat(command: list[str], profile: dict, cwd: Path | None = None) -> list[str]:
    """Serialise the run on the lpno gpu_spool seat lock.

    Without this, an AiiDA k-fold run and a gpu_spool training can land on the
    same GPU at once. ``gpu_spool seat --gpu N -- <cmd>`` blocks until the card
    has room in the shared reservation ledger (``/var/tmp/pino_gpu_leases``).
    ``--vram-mb``/``--ram-mb`` ARE passed when the profile declares
    ``gpu_vram_mb``/``host_ram_mb`` (``_declared_memory_args`` below). With no
    ``gpu_vram_mb`` the run books the card *exclusively*, and with no
    ``host_ram_mb`` it takes gpu_spool's default share
    (``PINO_SPOOL_DEFAULT_RAM_MB``, 32 GB), so an undeclared submission is the
    one least likely to start on a busy box. The profile's
    ``custom.cuda_device`` override is kept as-is: the entrypoint routes by
    physical GPU index (``configure_device`` re-exports CUDA_VISIBLE_DEVICES
    from it before the first CUDA init), so the override must stay equal to
    the seat's ``--gpu`` value rather than be rewritten to 0.

    Falls back to the bare command with a loud warning when the spooler module
    is missing or ``cuda_device`` is not a single GPU index (multi-GPU values
    pass through unwrapped).
    """
    cwd = Path.cwd() if cwd is None else cwd
    raw_device = str(profile.get("cuda_device") or "0").strip()
    try:
        gpu = int(raw_device)
    except ValueError:
        print(
            f"[gpu-seat] WARNING: cuda_device={raw_device!r} is not a single GPU index; "
            "running without the gpu_spool seat lock.",
            flush=True,
        )
        return command
    if importlib.util.find_spec(GPU_SPOOL_MODULE) is None:
        print(
            f"[gpu-seat] WARNING: {GPU_SPOOL_MODULE} is not importable; running without the "
            f"GPU seat lock (a concurrent spooled run on GPU {gpu} will not be serialised).",
            flush=True,
        )
        return command
    print(f"[gpu-seat] waiting for gpu {gpu} (shared pino spool seat lock)", flush=True)
    # `seat` runs its command from the lpno repo root; the sh shim
    # logs the acquired seat and restores this job's working directory.
    # After the seat wait (can be hours), rebuild any staged dataset symlink the
    # cache lost in the meantime (disk-pressure cleanup evicts it), THEN exec the
    # training. The adapter-time repair cannot cover this window. `|| true` so a
    # repair miss falls through to the training's own missing-data error.
    repair = _shared_runtime_repair_command(sys.executable)
    shim = f'echo "[gpu-seat] acquired gpu {gpu}"; cd {shlex.quote(str(cwd))} && {repair}; exec "$0" "$@"'
    seat_args = ["--gpu", str(gpu), *_declared_memory_args(profile)]
    return [sys.executable, "-m", GPU_SPOOL_MODULE, "seat", *seat_args, "--", "sh", "-c", shim, *command]


def _declared_memory_args(profile: dict) -> list[str]:
    """Turn measured `gpu_vram_mb`/`host_ram_mb` into `seat` reservation flags.

    Omitting them books the whole card exclusively plus gpu_spool's blind 32 GB
    host default, which is why an AiiDA job could never sit beside a run that
    had declared 8 GB on a card with 22 GB free. A value that is not a usable
    positive int is ignored rather than passed through: `--vram-mb 0` reserves
    nothing and would let unbounded runs pile onto one card, so the safe
    direction for a typo is the exclusive booking.
    """
    args: list[str] = []
    for key, flag in (("gpu_vram_mb", "--vram-mb"), ("host_ram_mb", "--ram-mb")):
        value = profile.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        args += [flag, str(value)]
    return args


def parse_model_errors(path: Path) -> dict[str, dict[str, float]]:
    """Parse a PINO metric file into a nested mapping."""
    metrics: dict[str, dict[str, float]] = {}
    current_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_key = line[1:-1]
            metrics[current_key] = {}
            continue
        if current_key is None or ":" not in line:
            continue
        name, value = [part.strip() for part in line.split(":", 1)]
        try:
            metrics[current_key][name] = float(value)
        except ValueError:
            continue

    return metrics


def ensure_module_available(module_name: str) -> None:
    """Fail early with an actionable message if the package is missing."""
    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(PACKAGE_INSTALL_HINT)


def validate_dataset_input_dir(path: Path, *, label: str) -> None:
    """Validate staged dataset input; see `staged_dataset` for the reasoning."""
    _shared_validate_dataset_input_dir(path, label=label)

def hydra_scalar(value) -> str:
    """Serialize simple Python values into Hydra CLI scalars."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def append_runtime_overrides(command: list[str], profile: dict, *, split_mode: str) -> None:
    """Append the supported README-level runtime options to a Hydra command."""
    autoregressive = profile.get("autoregressive", {})
    command.extend(
        [
            f"custom.split_mode={split_mode}",
            f"custom.k_folds={hydra_scalar(profile['k_folds'])}",
            f"custom.fold_index={hydra_scalar(profile['fold_index'])}",
            f"custom.pde_loss={hydra_scalar(profile['pde_loss'])}",
            f"custom.VOF_pde_loss={hydra_scalar(profile['vof_pde_loss'])}",
            f"custom.Diff_loss={hydra_scalar(profile['diff_loss'])}",
            f"custom.Diff_weight={hydra_scalar(profile['diff_weight'])}",
            f"custom.pde_weight={hydra_scalar(profile['pde_weight'])}",
            f"custom.include_time={hydra_scalar(profile['include_time'])}",
            f"custom.time_as_batch={hydra_scalar(profile['time_as_batch'])}",
            f"custom.time_as_dim={hydra_scalar(profile['time_as_dim'])}",
            f"custom.use_2D={hydra_scalar(profile['use_2d'])}",
            f"custom.time_avg_window={hydra_scalar(profile['time_avg_window'])}",
            f"custom.moving_ref_frame={hydra_scalar(profile['moving_ref_frame'])}",
            f"custom.moving_frame_steps={hydra_scalar(profile['moving_frame_steps'])}",
            f"custom.fast_training={hydra_scalar(profile['fast_training'])}",
            f"custom.detect_anomaly={hydra_scalar(profile['detect_anomaly'])}",
            f"custom.sharp_alpha={hydra_scalar(profile['sharp_alpha'])}",
            f"custom.alpha_masking={hydra_scalar(profile['alpha_masking'])}",
            # Restrict headline metrics to the substrate, excluding the free gas gap
            # above the material. Emitted only when explicitly set, so existing
            # profiles keep the pipeline default and old evals stay comparable; when
            # set it lands in the profile Dict and is therefore provenance-tracked.
            *(
                [f"custom.metrics_exclude_gas={hydra_scalar(profile['metrics_exclude_gas'])}"]
                if profile.get("metrics_exclude_gas") is not None
                else []
            ),
            # Per-model marker (see profiles.DEFAULT_TRAINING_PROFILE).
            *(
                # ++ (force-add): only pvlhs-era configs declare custom.power_weighted_dose,
                # so a plain override dies in struct mode on every other config -- it killed
                # the first mb fine-tune's test eval (46475). Force-add is valid either way.
                [f"++custom.power_weighted_dose={hydra_scalar(profile['power_weighted_dose'])}"]
                if profile.get("power_weighted_dose") is not None
                else []
            ),
            f"training.max_steps={hydra_scalar(profile['max_steps'])}",
            f"optimizer.lr={hydra_scalar(profile['learning_rate'])}",
            f"optimizer.weight_decay={hydra_scalar(profile['weight_decay'])}",
            f"batch_size.grid={hydra_scalar(profile['batch_size_grid'])}",
            f"batch_size.validation={hydra_scalar(profile['batch_size_validation'])}",
            f"custom.inference_batch_size={hydra_scalar(profile['inference_batch_size'])}",
            f"custom.infer_velocities={hydra_scalar(profile['infer_velocities'])}",
            f"custom.autoregressive.enabled={hydra_scalar(autoregressive.get('enabled', False))}",
            f"custom.autoregressive.state_prev_suffix={autoregressive.get('state_prev_suffix', '_prev')}",
            f"custom.autoregressive.eval_horizon_k={hydra_scalar(autoregressive.get('eval_horizon_k', 10))}",
            "custom.autoregressive.eval_use_gt_init="
            f"{hydra_scalar(autoregressive.get('eval_use_gt_init', True))}",
            f"custom.autoregressive.require_time={hydra_scalar(autoregressive.get('require_time', True))}",
        ]
    )
    # None opts out of the scheduler token: configs whose scheduler group has
    # no decay keys (e.g. one_cycle) reject the unknown key in hydra struct mode.
    if profile.get("scheduler_decay_rate") is not None:
        command.append(f"scheduler.decay_rate={hydra_scalar(profile['scheduler_decay_rate'])}")
    if profile.get("scheduler_decay_steps") is not None:
        command.append(f"scheduler.decay_steps={hydra_scalar(profile['scheduler_decay_steps'])}")
    cuda_device = profile.get("cuda_device")
    if cuda_device not in (None, ""):
        command.append(f"custom.cuda_device={cuda_device}")


def build_kfold_command(profile: dict, dataset_dir: Path, output_dir: Path) -> list[str]:
    """Build the packaged k-fold command."""
    command = [
        sys.executable,
        "-u",
        "-m",
        KFOLD_MODULE,
        f"--config-name={profile['config_name']}",
        f"custom.name={profile['run_name']}",
        f"custom.data_dir={dataset_dir}",
        f"network_dir={output_dir}",
        f"custom.wandb_offline={hydra_scalar(profile['wandb_offline'])}",
    ]
    append_runtime_overrides(command, profile, split_mode="kfold")
    command.extend(profile.get("hydra_overrides", []))
    return command


def collect_kfold_metrics(output_dir: Path) -> dict:
    """Collect aggregated and per-fold metric files."""
    payload: dict = {}
    aggregated = output_dir / "fno_eval" / "aggregated_errors.txt"
    if aggregated.exists():
        payload["aggregated"] = parse_model_errors(aggregated)

    folds: dict[str, dict] = {}
    for fold_dir in sorted(path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        fold_metrics: dict = {}
        model_errors = fold_dir / "model_errors.txt"
        rollout_full = fold_dir / "model_errors_rollout_full.txt"
        rollout_k = fold_dir / "model_errors_rollout_k.txt"
        if model_errors.exists():
            fold_metrics.update(parse_model_errors(model_errors))
        if rollout_full.exists():
            fold_metrics["rollout_full"] = parse_model_errors(rollout_full)
        if rollout_k.exists():
            fold_metrics["rollout_k"] = parse_model_errors(rollout_k)
        if fold_metrics:
            folds[fold_dir.name] = fold_metrics

    if folds:
        payload["folds"] = folds
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-file", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    profile = json.loads(Path(args.profile_file).read_text(encoding="utf-8"))
    validate_dataset_input_dir(dataset_dir, label="K-fold dataset directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_module_available(KFOLD_MODULE)
    command = build_kfold_command(profile, dataset_dir, output_dir)
    command = wrap_with_gpu_seat(command, profile)

    run_streamed_subprocess(command, cwd=Path.cwd(), env=profile_environment(profile))

    metrics = collect_kfold_metrics(output_dir)
    Path("metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "profile": profile,
        "aggregated_errors_present": (output_dir / "fno_eval" / "aggregated_errors.txt").is_file(),
        "fold_directories": sorted(path.name for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")),
    }
    Path("kfold_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run_with_live_tail(_main))
