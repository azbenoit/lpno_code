"""Standalone adapter for the packaged PINO training entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

try:
    from aiida_pino.scripts.live_tail import RuntimeMeter, collect_environment, profile_environment, run_streamed_subprocess, run_with_live_tail
except ImportError:
    _LIVE_TAIL_SPEC = importlib.util.spec_from_file_location("aiida_pino_live_tail", Path(__file__).with_name("live_tail.py"))
    if _LIVE_TAIL_SPEC is None or _LIVE_TAIL_SPEC.loader is None:
        raise
    _LIVE_TAIL_MODULE = importlib.util.module_from_spec(_LIVE_TAIL_SPEC)
    _LIVE_TAIL_SPEC.loader.exec_module(_LIVE_TAIL_MODULE)
    profile_environment = _LIVE_TAIL_MODULE.profile_environment
    run_streamed_subprocess = _LIVE_TAIL_MODULE.run_streamed_subprocess
    run_with_live_tail = _LIVE_TAIL_MODULE.run_with_live_tail
    collect_environment = _LIVE_TAIL_MODULE.collect_environment
    RuntimeMeter = _LIVE_TAIL_MODULE.RuntimeMeter

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
TRAIN_MODULE = "lpno.models.NS_Energy_laser_PINO"
GPU_SPOOL_MODULE = "lpno.gpu_spool"


def pick_free_gpu(default: int = 0) -> int:
    """Index of the card with the most free VRAM right now.

    `gpu_spool seat` needs an explicit index, so a submitter that pins
    `cuda_device` hours in advance guesses which card will be free when the job
    finally starts — and guesses wrong. On 2026-07-28 twelve evals pinned to
    GPU 1 queued behind a 21 GiB training while GPU 0 sat completely idle.
    Choosing here, at job start, is the same decision made with current
    information. Falls back to `default` when nvidia-smi is unavailable.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return default
        best, best_free = default, -1
        for line in out.stdout.splitlines():
            index, _, free = line.partition(",")
            try:
                index, free = int(index.strip()), int(free.strip())
            except ValueError:
                continue
            if free > best_free:
                best, best_free = index, free
        return best
    except Exception:  # noqa: BLE001 - no nvidia-smi: keep the caller's default
        return default


def _cuda_device_is_auto(value) -> bool:
    raw = str(value or "").strip()
    return raw == "" or raw.lower() == "auto"


def resolve_cuda_device(profile: dict) -> None:
    """Keep unpinned `cuda_device` unresolved until the gpu_spool seat admits it.

    `""`/absent/`"auto"` mean "decide when a seat is actually available". The
    seat then injects the selected physical GPU into the payload as
    `custom.cuda_device=N`, so the reservation and CUDA routing still agree.
    Explicit indices are always honoured.
    """
    raw = str(profile.get("cuda_device") or "").strip()
    if raw and raw.lower() != "auto":
        return
    profile["cuda_device"] = "auto"


def wrap_with_gpu_seat(command: list[str], profile: dict, cwd: Path | None = None) -> list[str]:
    """Serialise the run on the lpno gpu_spool seat lock.

    Without this, an AiiDA training and a gpu_spool training can land on the
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
    raw_device = str(profile.get("cuda_device") or "").strip()
    auto_gpu = _cuda_device_is_auto(raw_device)
    gpu_arg = "auto" if auto_gpu else raw_device
    if not auto_gpu:
        try:
            int(raw_device)
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
            f"GPU seat lock (a concurrent spooled run may not be serialised).",
            flush=True,
        )
        return command
    print(f"[gpu-seat] waiting for gpu {gpu_arg} (shared pino spool seat lock)", flush=True)
    # `seat` runs its command from the lpno repo root; the sh shim
    # logs the acquired seat and restores this job's working directory.
    # After the seat wait (can be hours), rebuild any staged dataset symlink the
    # cache lost in the meantime (disk-pressure cleanup evicts it), THEN exec the
    # training. The adapter-time repair cannot cover this window. `|| true` so a
    # repair miss falls through to the training's own missing-data error.
    repair = _shared_runtime_repair_command(sys.executable)
    shim = f'echo "[gpu-seat] acquired gpu $CUDA_VISIBLE_DEVICES"; cd {shlex.quote(str(cwd))} && {repair}; exec "$0" "$@"'
    seat_args = ["--gpu", gpu_arg, *_declared_memory_args(profile)]
    if profile.get("gpu_seat_priority"):
        # Admission is a poll race, not a queue: a restarted long run otherwise
        # loses its own card to whichever queued job polls first. Opt-in per
        # profile -- it makes spool workers hold off starting new runs.
        seat_args.append("--priority")
        # ...but NOT gpu_spool's CI deadline. `--priority` alone swaps the unbounded
        # wait for DEFAULT_PRIORITY_MAX_WAIT_S = 3600 s "so CI reports a cause",
        # and on a saturated box that is what kills the job rather than slowing it:
        # training 99083 died at 3602 s without ever seating, waiting on a seat that
        # was ~33 h out, while two sibling arms without the flag were still alive
        # 12 h later. 0 is falsy at gpu_spool's deadline check, i.e. wait forever.
        seat_args += ["--max-wait-s", "0"]
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


# GPU peak sampling. Reservations (`gpu_vram_mb`) are sized from measured
# peaks (see utils/vram.py); the sampler is best-effort, so a host without
# nvidia-smi degrades to no measurement, never an error. Duplicated in
# eval_adapter.py because these scripts are copied standalone into calcjob
# workdirs, same as the seat wrapper above.
_SMI_QUERY = ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]


def parse_smi_used_mib(csv_text: str, pids: set[int]) -> int:
    """Sum used MiB over `pid, used_mib` lines whose pid is in `pids`.

    One run may hold several CUDA processes (dataloader workers); malformed
    lines (`[N/A]`, headers) are skipped.
    """
    total = 0
    for line in csv_text.splitlines():
        pid_str, _, used_str = line.partition(",")
        try:
            if int(pid_str.strip()) in pids:
                total += int(used_str.strip())
        except ValueError:
            continue
    return total


def _descendant_pids(root: int) -> set[int]:
    """`root` plus all its descendants, from /proc/*/stat ppid links."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path("/proc", entry, "stat").read_text(encoding="ascii", errors="replace")
            # Split after the last ')' to survive comm names with spaces or
            # parens; in the remainder, field 0 is state and field 1 is ppid.
            ppid = int(stat.rpartition(")")[2].split()[1])
        except (OSError, ValueError):
            continue  # process vanished mid-walk or unparsable stat
        children.setdefault(ppid, []).append(int(entry))
    pids = {root}
    queue = [root]
    while queue:
        for child in children.get(queue.pop(), ()):
            if child not in pids:
                pids.add(child)
                queue.append(child)
    return pids


class GpuPeakSampler(threading.Thread):
    """Daemon thread tracking this run's peak GPU memory via nvidia-smi."""

    def __init__(self, interval_s: float = 10.0) -> None:
        super().__init__(daemon=True)
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self.peak_mb = 0

    def run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                result = subprocess.run(_SMI_QUERY, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    sample = parse_smi_used_mib(result.stdout, _descendant_pids(os.getpid()))
                    self.peak_mb = max(self.peak_mb, sample)
            except Exception:  # noqa: BLE001 - nvidia-smi may be absent; sampling is best-effort
                pass

    def stop(self) -> int | None:
        """Stop sampling and return the peak in MiB, or None if never measured."""
        self._stop_event.set()
        self.join(timeout=5)
        return self.peak_mb or None


def ensure_module_available(module_name: str) -> None:
    """Fail early with an actionable message if the package is missing."""
    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(PACKAGE_INSTALL_HINT)


def has_consolidated_case_npz(path: Path) -> bool:
    """Return whether ``path`` holds top-level consolidated per-case NPZ files.

    Mirrors the lpno datasource signature check (``power_w`` /
    ``scan_speed_m_per_s`` / ``case_name`` members distinguish a case NPZ from
    a ``var_*.npz`` resample cache) without importing it: this adapter must
    stay runnable when only numpy is installed. npz_file-mode dataset staging
    hardlinks the original case NPZs flat into the dataset directory, which
    the datasource then reads natively.
    """
    import zipfile

    required = {"power_w.npy", "scan_speed_m_per_s.npy", "case_name.npy"}
    for candidate in sorted(path.glob("*.npz")):
        if candidate.name.startswith("var_"):
            continue
        try:
            with zipfile.ZipFile(candidate) as archive:
                names = set(archive.namelist())
        except (OSError, zipfile.BadZipFile):
            continue
        if required.issubset(names):
            return True
    return False


def validate_dataset_input_dir(path: Path, *, label: str) -> None:
    """Validate staged dataset input; see `staged_dataset` for the reasoning."""
    _shared_validate_dataset_input_dir(
        path,
        label=label,
        accept_consolidated_npz=True,
        consolidated_npz_check=has_consolidated_case_npz,
    )

def hydra_scalar(value) -> str:
    """Serialize simple Python values into Hydra CLI scalars."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_scalar(item) for item in value) + "]"
    return str(value)



# Scheduler groups that expose `decay_rate`/`decay_steps`. Anything else (notably
# `one_cycle`, whose OneCycleLRConf has max_lr/total_steps/pct_start instead)
# rejects those keys under hydra struct mode.
_DECAY_SCHEDULERS = {"tf_exponential_lr", "exponential_lr"}


def _scheduler_accepts_decay_keys(config_name) -> bool:
    """Whether `config_name`'s scheduler group takes decay_rate/decay_steps.

    Walks the packaged hydra config's `defaults` list, following inheritance from
    sibling configs, to find the selected `scheduler:` group. Unknown or
    unreadable configs return True, preserving the previous behaviour rather than
    silently dropping a token the run may need.
    """
    if not config_name:
        return True
    try:
        import yaml  # local import: the adapter runs in the pino env
        from pathlib import Path as _Path
        import lpno

        conf_dir = _Path(lpno.__file__).parent / "conf" / "hydra"
    except Exception:
        return True

    seen = set()
    stack = [str(config_name)]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        path = conf_dir / f"{name}.yaml"
        if not path.is_file():
            continue
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except Exception:
            return True
        for entry in cfg.get("defaults", []) or []:
            if isinstance(entry, dict):
                if "scheduler" in entry:
                    return str(entry["scheduler"]) in _DECAY_SCHEDULERS
            elif isinstance(entry, str) and entry not in ("_self_",):
                stack.append(entry)
    return True

#: Override keys that carry the training seed (Hydra +/~ prefixes stripped).
_SEED_KEYS = ("seed", "training.seed", "custom.seed", "trainer.seed")


def resolve_training_seed(profile: dict, config_name=None) -> tuple[int | None, str]:
    """The seed this run will use, and where it comes from.

    Chain measured in the sibling repo: `custom.seed` -> `seed` -> the literal
    42 (`config_loading.py: int(_get_cfg_value(cfg, "seed", 42))`), applied via
    `seed_all` (python/numpy/torch/cuda) at `ns_energy_pipeline.py:374`. So an
    omitted seed is NOT unseeded -- but it was invisible: the archived
    `.hydra/config.yaml` of a default run carries no seed key, and a training
    that never becomes a model node has no seed record anywhere (2026-08-10).

    Returns (None, "unresolved") only when the packaged config cannot be read
    -- the one case where injecting a value could CHANGE the run.
    """
    if profile.get("seed") is not None:
        return int(profile["seed"]), "training_profile"
    for override in profile.get("hydra_overrides") or []:
        text = str(override)
        key, sep, raw = text.partition("=")
        if sep and key.strip().lstrip("+~") in _SEED_KEYS:
            try:
                return int(str(raw).strip()), "hydra_override"
            except ValueError:
                return None, "unresolved"
    # No explicit seed: the run uses the packaged config's value if it sets
    # one, else the literal 42. Walk the config chain the same way
    # `_scheduler_accepts_decay_keys` does.
    try:
        import yaml
        from pathlib import Path as _Path

        import lpno

        conf_dir = _Path(lpno.__file__).parent / "conf" / "hydra"
        seen: set[str] = set()
        stack = [str(config_name or "")]
        while stack:
            name = stack.pop()
            if not name or name in seen:
                continue
            seen.add(name)
            path = conf_dir / f"{name}.yaml"
            if not path.is_file():
                continue
            cfg = yaml.safe_load(path.read_text()) or {}
            custom = cfg.get("custom") or {}
            if isinstance(custom, dict) and custom.get("seed") is not None:
                return int(custom["seed"]), "config_default"
            if cfg.get("seed") is not None:
                return int(cfg["seed"]), "config_default"
            for entry in cfg.get("defaults", []) or []:
                if isinstance(entry, str) and entry != "_self_":
                    stack.append(entry)
    except Exception:  # pragma: no cover - unreadable config: do not guess
        return None, "unresolved"
    return 42, "lpno_default"


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
            # Per-model marker (see profiles.DEFAULT_TRAINING_PROFILE): emitted only
            # when stamped, so a pre-stamp profile keeps the pipeline default.
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
            f"custom.autoregressive.mode={autoregressive.get('mode', 'masked_patch')}",
            f"custom.autoregressive.state_prev_suffix={autoregressive.get('state_prev_suffix', '_prev')}",
            f"custom.autoregressive.patch_size={hydra_scalar(autoregressive.get('patch_size', [32, 16, 16]))}",
            f"custom.autoregressive.context_steps={hydra_scalar(autoregressive.get('context_steps', 1))}",
            f"custom.autoregressive.mask_mode={autoregressive.get('mask_mode', 'block')}",
            f"custom.autoregressive.mask_ratio={hydra_scalar(autoregressive.get('mask_ratio', 0.15))}",
            "custom.autoregressive.train_patches_per_epoch="
            f"{hydra_scalar(autoregressive.get('train_patches_per_epoch', 256))}",
            "custom.autoregressive.val_patches_per_epoch="
            f"{hydra_scalar(autoregressive.get('val_patches_per_epoch', 64))}",
            f"custom.autoregressive.eval_horizon_k={hydra_scalar(autoregressive.get('eval_horizon_k', 10))}",
            "custom.autoregressive.eval_use_gt_init="
            f"{hydra_scalar(autoregressive.get('eval_use_gt_init', True))}",
            f"custom.autoregressive.require_time={hydra_scalar(autoregressive.get('require_time', True))}",
        ]
    )
    # None opts out of the scheduler token: configs whose scheduler group has
    # no decay keys (e.g. one_cycle) reject the unknown key in hydra struct mode.
    # Beyond the manual opt-out, drop the tokens automatically when the target
    # config selects such a scheduler -- otherwise a stored profile carrying the
    # old exponential defaults fails at hydra compose time the moment a config
    # switches to one_cycle, which is exactly what happened when the straight-path
    # configs adopted it as their default.
    decay_ok = _scheduler_accepts_decay_keys(profile.get("config_name"))
    # One-cycle: `optimizer.lr` is NOT the LR that runs. OneCycleLR sets every
    # param group's lr from `scheduler.max_lr` at construction and on each step,
    # so a profile's learning_rate was silently dead -- four campaign arms
    # recorded 3e-4 and all trained at the config's 1e-4 (2026-08-15), which let
    # a no-op "LR fix" take credit for other changes' gains. Emit the peak too,
    # force-add so a config without the key cannot crash compose.
    if not decay_ok and profile.get("learning_rate") is not None:
        # PLAIN override, NOT `++`. pino's `remap_plus_args_to_custom` rewrites
        # every arg starting with `+`/`++` into the `custom.` namespace
        # (`target_key = key if key.startswith("custom.") else f"custom.{key}"`),
        # so `++scheduler.max_lr=3e-4` silently became
        # `++custom.scheduler.max_lr=3e-4` -- a dead key nothing reads, while the
        # real scheduler kept the config default. Measured 2026-08-17 from the
        # composed configs of every arm: `scheduler.max_lr=0.0001` with
        # `custom.scheduler={'max_lr': 0.0003}` beside it. A plain override is
        # untouched by the remapper, and it is safe here because this branch only
        # runs for one-cycle configs, which declare `scheduler.max_lr`.
        command.append(f"scheduler.max_lr={hydra_scalar(profile['learning_rate'])}")
        print(
            f"[lr] one-cycle config: peak LR set to {profile['learning_rate']} via "
            "scheduler.max_lr (optimizer.lr alone would be overwritten)",
            flush=True,
        )
    if profile.get("scheduler_decay_rate") is not None and decay_ok:
        command.append(f"scheduler.decay_rate={hydra_scalar(profile['scheduler_decay_rate'])}")
    if profile.get("scheduler_decay_steps") is not None and decay_ok:
        command.append(f"scheduler.decay_steps={hydra_scalar(profile['scheduler_decay_steps'])}")
    if not decay_ok and (
        profile.get("scheduler_decay_rate") is not None
        or profile.get("scheduler_decay_steps") is not None
    ):
        print(
            f"[scheduler] {profile.get('config_name')} selects a scheduler with no "
            "decay knobs; dropping scheduler_decay_rate/steps from the overrides",
            flush=True,
        )
    cuda_device = profile.get("cuda_device")
    if cuda_device not in (None, "") and not _cuda_device_is_auto(cuda_device):
        command.append(f"custom.cuda_device={cuda_device}")


#: Cadence keys `validate_recording_cadence` (pino config_loading.py) checks against
#: `training.max_steps`. A run whose config declares any of these COARSER than its own
#: length is refused at startup, which is right for a real run and fatal for a VRAM
#: smoke, whose whole point is to be 100 steps long.
_CADENCE_KEYS = (
    "training.rec_results_freq",
    "training.rec_validation_freq",
    "stop_criterion.freq",
    # Not checked by the validator, and no shipped config sets it: physicsnemo's own
    # default (1000) applies. It is the interval at which physicsnemo writes the
    # weights, so a run shorter than it ends with its step-0 weights on disk and its
    # evaluation scores an untrained model (found with reproducibility/train_evaluate.ipynb).
    "training.save_network_freq",
)

#: Cadence keys physicsnemo's base schema (`DefaultTraining`) declares with a default
#: of its own, so a config that leaves them out still runs with that value, and a
#: plain override of them composes on every config.
_PHYSICSNEMO_CADENCE_DEFAULTS = {"training.save_network_freq": 1000}


def _override_value(command: list[str], dotted_key: str):
    """The value the LAST override of `dotted_key` in `command` sets, or None.

    Hydra applies overrides left to right, so the last one wins. `+key=`/`++key=` set
    the key like a plain override; `~key` deletes it, which falls back to the config.
    """
    found = None
    for item in command:
        text = str(item)
        if text.startswith("~"):
            if text[1:].partition("=")[0].strip() == dotted_key:
                found = None
            continue
        key, sep, value = text.partition("=")
        if sep and key.strip().lstrip("+") == dotted_key:
            found = value.strip()
    return found


def _config_declares(config_name, dotted_key):
    """(declared, value) for `dotted_key` in `config_name`'s packaged config chain.

    Returns (False, None) when the chain is walked without finding the key, and for an
    unknown or unreadable config. Both mean "do not emit an override for it", which is
    the safe direction: omitting a clamp leaves the clear validator error, while
    emitting a plain override for a key the config does not declare dies in hydra's
    struct mode -- a worse failure than the one being fixed.

    Follows the same `defaults` inheritance as `_scheduler_accepts_decay_keys`, plus
    group entries (`{scheduler: one_cycle}` -> `conf/hydra/scheduler/one_cycle.yaml`),
    since a cadence key can be declared by an included group rather than the leaf.
    """
    if not config_name:
        return False, None
    try:
        import yaml  # local import: the adapter runs in the pino env
        from pathlib import Path as _Path

        import lpno

        conf_dir = _Path(lpno.__file__).parent / "conf" / "hydra"
    except Exception:
        return False, None

    def _select(mapping, dotted):
        cur = mapping
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    seen: set[str] = set()
    stack = [str(config_name)]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        path = conf_dir / f"{name}.yaml"
        if not path.is_file():
            continue
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except Exception:
            return False, None
        found = _select(cfg, dotted_key)
        if found is not None:
            return True, found
        for entry in cfg.get("defaults", []) or []:
            if isinstance(entry, dict):
                for group, choice in entry.items():
                    group_path = conf_dir / str(group) / f"{choice}.yaml"
                    if not group_path.is_file():
                        continue
                    try:
                        group_cfg = yaml.safe_load(group_path.read_text()) or {}
                    except Exception:
                        continue
                    # A group file's body lives UNDER the group name once composed.
                    found = _select(group_cfg, dotted_key)
                    if found is None:
                        found = _select({str(group): group_cfg}, dotted_key)
                    if found is not None:
                        return True, found
            elif isinstance(entry, str) and entry != "_self_":
                stack.append(entry)
    return False, None


def _config_known(config_name) -> bool:
    """Whether `config_name` is a packaged lpno config this host can read."""
    if not config_name:
        return False
    try:
        from pathlib import Path as _Path

        import lpno

        conf_dir = _Path(lpno.__file__).parent / "conf" / "hydra"
    except Exception:
        return False
    return (conf_dir / f"{config_name}.yaml").is_file()


def append_recording_cadence_clamp(command: list[str], profile: dict) -> None:
    """Clamp a config cadence that is coarser than this run's `max_steps`.

    Without it such a run is REFUSED at startup by `validate_recording_cadence`, which
    made every VRAM smoke of a coarse-cadence config impossible -- the smoke is 100
    steps and `config_flow3D_RIGNO_pvlhs` declares a 2000-step cadence.

    **PLAIN overrides, never `++`.** pino's `remap_plus_args_to_custom` rewrites every
    arg starting with `+`/`++` into the `custom.` namespace
    (`target_key = key if key.startswith("custom.") else f"custom.{key}"`), so an
    earlier `++training.rec_results_freq=100` became `++custom.training.rec_results_freq`
    -- a dead key nothing reads, while the real cadence kept the config's 2000 and the
    run was refused anyway (2026-08-22, three RIGNO seed replicates). Same trap that
    killed `++scheduler.max_lr` on 2026-08-15.

    A plain override is untouched by the remapper but dies in struct mode if the config
    does not declare the key -- which is why each key is emitted ONLY when
    `_config_declares` finds it, and why nothing is emitted when the config is unknown.
    `validate_recording_cadence` skips absent keys too, so a key the config does not
    declare needs no clamp. The exception is `training.save_network_freq`: physicsnemo's
    base schema declares it (default 1000), so it applies, and composes, on every
    config. It is the interval at which the weights are written, so a run shorter than
    it would end with its step-0 weights on disk.

    The value checked is the one the run will use: the LAST override of the key already
    in `command` (the profile's `hydra_overrides` included), else the config's. Only a
    value coarser than the run is lowered; a finer one a caller chose (a 50-step
    validation cadence in a 200-step run) is left alone. When a clamp leaves
    `stop_criterion.freq` off a multiple of the validation cadence, it follows the
    validation cadence, as `reproducibility/rerun/replay_overrides.py` does.

    Appended AFTER the profile's `hydra_overrides` so the clamp WINS: a donor profile
    can carry a plain `training.rec_results_freq=2000`, and inheriting it into a
    100-step smoke reinstates exactly the refusal this exists to prevent.
    """
    try:
        max_steps = int(profile.get("max_steps") or 0)
    except (TypeError, ValueError):
        return
    if max_steps <= 0:
        return
    config_name = profile.get("config_name")
    effective: dict[str, int] = {}
    clamped: set[str] = set()
    for key in _CADENCE_KEYS:
        value = _override_value(command, key)
        source = "an override sets"
        if value is None:
            declared, value = _config_declares(config_name, key)
            source = f"{config_name} declares"
            if not declared:
                if key not in _PHYSICSNEMO_CADENCE_DEFAULTS or not _config_known(config_name):
                    continue
                value = _PHYSICSNEMO_CADENCE_DEFAULTS[key]
                source = "physicsnemo defaults to"
        try:
            current = int(value)
        except (TypeError, ValueError):
            continue
        effective[key] = current
        if current <= max_steps:
            continue
        effective[key] = max_steps
        clamped.add(key)
        command.append(f"{key}={max_steps}")
        print(
            f"[cadence] {source} {key}={current} but this run is {max_steps} steps; "
            f"clamped to {max_steps}",
            flush=True,
        )
    validation = effective.get("training.rec_validation_freq",
                               effective.get("training.rec_results_freq"))
    stop = effective.get("stop_criterion.freq")
    if (validation and stop and stop % validation
            and clamped & {"training.rec_validation_freq", "training.rec_results_freq",
                           "stop_criterion.freq"}):
        command.append(f"stop_criterion.freq={validation}")
        print(
            f"[cadence] stop_criterion.freq={stop} is not a multiple of the validation "
            f"cadence {validation} after clamping; set to {validation}",
            flush=True,
        )



# FORCED TWIN of `eval_adapter.promote_replayed_overrides`: adapters are uploaded as
# single standalone files and cannot import each other on the compute box. A cloned
# profile carrying era `+custom.seed=42` overrides hits the same Hydra refusal at
# TRAINING time once the packaged config declares the key. `test_adapter_override_twins`
# pins the two copies byte-identical so they cannot drift (the duplicated-builder rule).
def promote_replayed_overrides(overrides: list) -> list[str]:
    """Stored `+key=value` overrides become `++key=value` for replay.

    A stored profile's overrides were legal against the packaged config OF THEIR DAY:
    `+custom.seed=42` appended a key the config did not declare. Replayed onto the model's
    own ARCHIVED `.hydra/config.yaml` -- which bakes in everything the run resolved,
    including that very key -- Hydra refuses the append (`An item is already at
    'custom.seed'`). So every model whose era stored single-plus overrides was unevaluable
    through `archived_config_dir`, the path that exists precisely to keep old checkpoints
    evaluable as the packaged schema drifts: the two safety mechanisms defeated each other.
    Found on an eval whose model carried `+custom.seed=42` + `+custom.case_coherent_sampling`,
    verified on the node.

    `++` is append-or-override. On the archived path the stored override IS the archived
    value, so it is a no-op; where the two ever differ, the stored profile is what the run
    actually used, which is the value replay wants. And against a NEWER packaged config
    that has since declared the key, plain `+` would hit the same refusal -- `++` is right
    on both paths. The failure mode worth checking -- `++` silently creating an undeclared
    key -- is the same permissiveness the original `+` already had, so nothing new is
    masked. Only a single leading `+` is promoted: `++` stays, `~key` (delete) and plain
    overrides are untouched.
    """
    promoted = []
    for item in overrides or []:
        text = str(item)
        if text.startswith("+") and not text.startswith("++"):
            text = "+" + text
        promoted.append(text)
    return promoted


def build_train_command(
    profile: dict,
    train_dir: Path,
    val_dir: Path,
    output_dir: Path,
    init_dir: Path | None = None,
) -> list[str]:
    """Build the packaged training command.

    ``init_dir`` warm-starts through physicsnemo's own
    ``initialization_network_dir``: weights are loaded from another run's
    directory and a fresh optimizer starts at step 0, so the LR schedule
    restarts at warmup and any step-indexed curriculum replays.

    Callers that want the optimizer carried over instead seed ``network_dir``
    (see ``plan_warm_start``) and pass no ``init_dir`` at all -- physicsnemo
    then reads its own checkpoints and keeps the step counter. ``_main`` picks
    between the two; this function only renders what it is told.
    """
    command = [
        sys.executable,
        "-u",
        "-m",
        TRAIN_MODULE,
        f"--config-name={profile['config_name']}",
        f"custom.name={profile['run_name']}",
        f"custom.data_dir={train_dir}",
        f"custom.val_data_dir={val_dir}",
        f"network_dir={output_dir}",
        f"custom.wandb_offline={hydra_scalar(profile['wandb_offline'])}",
    ]
    if init_dir is not None:
        command.append(f"initialization_network_dir={init_dir}")
    append_runtime_overrides(command, profile, split_mode=str(profile.get("split_mode", "standard")))
    # Make the seed explicit in the launched config whenever it is implicit.
    # `++` (force-add) is safe for both a config that defines custom.seed and
    # one that does not; the injected value IS the value the run would have
    # used, so behaviour is identical -- the archived .hydra/config.yaml just
    # stops being silent about it. When the seed is already explicit (profile
    # or override) nothing is added; when the packaged config is unreadable,
    # nothing is added rather than risk changing the run. Injected BEFORE the
    # user's hydra_overrides: they stay last, so under Hydra's last-wins rule
    # any explicit user seed beats the injection even in a spelling the
    # resolver does not recognise.
    seed, source = resolve_training_seed(profile, profile.get("config_name"))
    if seed is not None and source in ("lpno_default", "config_default"):
        command.append(f"++custom.seed={seed}")
    command.extend(promote_replayed_overrides(profile.get("hydra_overrides", [])))
    # LAST, so it beats an inherited cadence override too -- see the docstring.
    append_recording_cadence_clamp(command, profile)
    return command


#: Written by `Trainer._save_checkpoint`; the five state dicts
#: `Trainer._load_optimizer` restores in one bare `except:`.
OPTIMIZER_CHECKPOINT = "optim_checkpoint.0.pth"


def amp_enabled_in_profile(profile: dict) -> bool:
    """Whether this run will build *enabled* grad scalers.

    An override wins, because that is where a caller states intent. With no
    override the answer comes from the run's OWN packaged config, not from an
    assumed default: `config_flow3D_GAOT_randompath.yaml` sets
    `amp.enabled: false` deliberately -- GAOT's segment softmax over ~1.2M edges
    is an fp16 overflow site -- and assuming true made `plan_warm_start` reject
    a restorable step-21000 checkpoint with "amp.enabled is true but
    scaler_state_dict is empty" (2026-08-09). That failure is silent in the
    damaging direction: the resume downgrades to weights-only, the run trains,
    and only the step counter and the one-cycle anneal position are wrong.
    """
    for override in reversed(list(profile.get("hydra_overrides", []) or [])):
        key, sep, value = str(override).partition("=")
        if sep and key.strip().lstrip("+~") == "amp.enabled":
            return value.strip().lower() not in {"false", "0", "no", "off"}
    resolved = _config_amp_enabled(profile.get("config_name"))
    # True only when the config is unreadable: that preserves the historical
    # behaviour rather than silently flipping every run whose config we cannot
    # see, and an unreadable config is already reported elsewhere.
    return True if resolved is None else resolved


def _config_amp_enabled(config_name) -> bool | None:
    """`amp.enabled` from the packaged config chain, or None if unreadable.

    Walks `defaults` the same way `_scheduler_accepts_decay_keys` does, so an
    inherited setting is found rather than only a top-level one.
    """
    if not config_name:
        return None
    try:
        import yaml
        from pathlib import Path as _Path

        import lpno

        conf_dir = _Path(lpno.__file__).parent / "conf" / "hydra"
        seen: set[str] = set()
        stack = [str(config_name)]
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            path = conf_dir / f"{name}.yaml"
            if not path.is_file():
                continue
            cfg = yaml.safe_load(path.read_text()) or {}
            amp = cfg.get("amp")
            if isinstance(amp, dict) and amp.get("enabled") is not None:
                return bool(amp["enabled"])
            for entry in cfg.get("defaults", []) or []:
                if isinstance(entry, str) and entry != "_self_":
                    stack.append(entry)
    except Exception:  # noqa: BLE001 - unreadable config: say so, do not guess
        return None
    return None


def resume_blockers(checkpoint: dict, amp_enabled: bool, max_steps: int | None = None) -> str | None:
    """Why `checkpoint` cannot be resumed, or None when it can be.

    physicsnemo restores the optimizer, the scheduler and the step counter only
    if all five `load_state_dict` calls survive. An enabled `GradScaler` or
    `DerivScalers` raises on an empty state dict, and a disabled one writes
    exactly that -- so an AMP run cannot continue an AMP-free checkpoint. The
    bare `except:` turns that into a silent `step = 0` plus a half-loaded
    optimizer, which is worse than an honest weights-only warm start.

    `init_model` serves BOTH same-run resume (a killed run's own checkpoints,
    2026-08-09) and cross-run transfer (warm start from a donor).
    The two checks on `max_steps` tell them apart mechanically:

    - a restored scheduler keeps its own horizon, so a donor from a different
      run length crashes the new run at the donor's `total_steps` boundary
      (measured 2026-08-14: a 20k donor into a 32k run died at step
      20201 with "Tried to step 20201 times") or silently anneals on the wrong
      schedule. A horizon that does not fit this run's `max_steps` is a
      transfer, and transfers are weights-only.
    - a checkpoint at or past this run's `max_steps` has nothing to continue:
      the trainer would exit immediately, "producing" the donor unchanged.

    A cross-run transfer with the SAME max_steps and step < max is not
    mechanically distinguishable from a resume; that case resumes. If it is
    ever needed, add an explicit `warm_start: weights_only` profile key.
    """
    step = checkpoint.get("step")
    if not isinstance(step, int) or step <= 0:
        return f"step is {step!r}; nothing to continue"
    if max_steps is not None:
        if step >= int(max_steps):
            return f"checkpoint step {step} >= this run's max_steps {max_steps}; nothing to continue"
        horizon = (checkpoint.get("scheduler_state_dict") or {}).get("total_steps")
        # A same-run resume's horizon is max_steps plus a small pino-side margin
        # (currently +200; the band tolerates margin changes up to 1000). Below
        # the band the restored scheduler steps past its own horizon and crashes;
        # above it (a longer donor) it silently anneals on the wrong schedule.
        if horizon is not None and not (int(max_steps) <= int(horizon) <= int(max_steps) + 1000):
            return (
                f"checkpoint scheduler total_steps {horizon} does not fit this run's "
                f"max_steps {max_steps} (+margin): cross-run transfer -> weights-only"
            )
    if amp_enabled:
        for key in ("scaler_state_dict", "deriv_scalers_state_dict"):
            if not checkpoint.get(key):
                return (
                    f"amp.enabled is true but {key} is empty; physicsnemo would "
                    "raise and silently restart at step 0"
                )
    return None


def plan_warm_start(init_dir: Path, profile: dict) -> tuple[bool, str]:
    """Decide between resuming (seed network_dir) and weights-only warm start.

    Returns (resume, reason). Any failure to read the checkpoint falls back to
    the weights-only path, which works with no optimizer state at all.
    """
    checkpoint_file = Path(init_dir) / OPTIMIZER_CHECKPOINT
    if not checkpoint_file.is_file():
        return False, f"no {OPTIMIZER_CHECKPOINT} in {init_dir}"
    try:
        import torch

        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any unreadable checkpoint degrades the same way
        return False, f"could not read {checkpoint_file}: {exc}"
    try:
        from aiida_pino.utils.profiles import effective_max_steps

        max_steps = effective_max_steps(profile)
    except Exception:  # noqa: BLE001 - standalone adapter run without the package
        max_steps = profile.get("max_steps")
    blocker = resume_blockers(checkpoint, amp_enabled_in_profile(profile), max_steps=max_steps)
    if blocker is not None:
        return False, blocker
    return True, f"continuing from step {checkpoint['step']}"


def seed_network_dir(init_dir: Path, output_dir: Path) -> list[str]:
    """Copy checkpoint files from a previous model into the fresh network_dir.

    The physicsnemo solver resumes from any checkpoints it finds in
    network_dir at startup, so this carries the optimizer and the step counter
    as well as the weights -- gate it on `plan_warm_start`, because a checkpoint
    physicsnemo cannot restore costs the step counter silently. Only checkpoint
    files are seeded; hydra/output clutter stays behind.
    """
    seeded: list[str] = []
    for path in sorted(Path(init_dir).iterdir()):
        if path.is_file() and path.suffix in (".pth", ".pt", ".mdlus"):
            shutil.copy2(path, output_dir / path.name)
            seeded.append(path.name)
    if not seeded:
        print(f"WARNING: --init-dir {init_dir} contains no checkpoint files; training starts from scratch.", flush=True)
    return seeded


def archive_hydra_config(output_dir: Path, cwd: Path | None = None) -> bool:
    """Copy the run's `.hydra` dir into `network_dir` so it ships with the model.

    Hydra writes its config into its own run dir (`outputs/<name>_<stamp>/.hydra/`),
    not into `network_dir`, so without this the retrieved artifacts hold only
    `*.pth` and the model node cannot describe how it was built. Eval then falls
    back to package defaults and silently produces no metrics (exit 502), and
    serving the checkpoint elsewhere means reconstructing the config by hand.

    Returns whether a config is present in `output_dir` afterwards.
    """
    target = Path(output_dir) / ".hydra"
    if (target / "config.yaml").is_file():
        return True  # the trainer already placed one; never clobber it
    root = Path(cwd) if cwd is not None else Path.cwd()
    candidates = sorted(
        root.glob("outputs/*/.hydra/config.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        # A warm start or a nested layout can push it one level deeper.
        candidates = sorted(
            root.glob("outputs/*/*/.hydra/config.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        print(
            "WARNING: no hydra config.yaml found under outputs/; the model node will "
            "not describe its own architecture (eval will fall back to defaults).",
            flush=True,
        )
        return False
    source = candidates[0].parent
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.is_file() and item.suffix in (".yaml", ".yml"):
            shutil.copy2(item, target / item.name)
    print(f"[hydra] archived {source} -> {target}", flush=True)
    return (target / "config.yaml").is_file()



def _report_declaration_overrun(profile: dict, key: str, measured_mb: int, label: str) -> None:
    """Warn when a run exceeded the resource it declared to the admission gate.

    The gate admits on DECLARED numbers, so an under-declaration is not a tight budget
    but an invisible one: the job is admitted and then exceeds what it claimed, taking
    co-tenants with it. Measured 2026-08-05, twice in one evening -- a varspeed GAOT
    declared 11218 MiB of VRAM against a proven need of 20000 and died, and another
    declared 32768 MiB of host RAM while peaking at 57 GB during graph construction.

    Reporting only. Failing here would throw away a completed training over a
    bookkeeping mismatch, which is exactly the trade this function exists to avoid.
    """
    declared = profile.get(key)
    if not declared or not measured_mb:
        return
    try:
        declared = int(declared)
    except (TypeError, ValueError):
        return
    if measured_mb > declared:
        print(f"[resource] WARNING {label} peak {measured_mb} MiB exceeded the declared "
              f"{key}={declared} MiB by {measured_mb - declared} MiB. The admission gate "
              f"booked the declared figure, so co-tenants were under-protected. Declare "
              f">= {measured_mb} MiB next time.", flush=True)
    else:
        print(f"[resource] {label} peak {measured_mb} MiB within declared "
              f"{key}={declared} MiB", flush=True)

def steps_trained(output_dir: Path) -> int | None:
    """Steps the run actually completed, from the checkpoint it just wrote.

    `max_steps` is a ceiling, and the trainer's stopping criterion routinely
    stops short of it: 31 of 158 models trained fewer steps than their record
    claims, worst 4000 of a declared 48000. Nothing on any node said so, while
    the console labelled the ceiling "Training steps".

    The optimizer checkpoint carries the counter physicsnemo restores from, so
    it is the same number a resume would continue at.
    """
    checkpoint_file = Path(output_dir) / OPTIMIZER_CHECKPOINT
    if not checkpoint_file.is_file():
        return None
    try:
        import torch

        step = torch.load(checkpoint_file, map_location="cpu", weights_only=False).get("step")
    except Exception:  # noqa: BLE001 - an unreadable checkpoint is not worth failing a finished run
        return None
    return int(step) if isinstance(step, int) and step > 0 else None


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-file", required=True)
    parser.add_argument("--init-dir", default=None, help="previous model checkpoint dir seeding network_dir (warm start)")
    args = parser.parse_args()

    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    profile = json.loads(Path(args.profile_file).read_text(encoding="utf-8"))
    validate_dataset_input_dir(train_dir, label="Training dataset directory")
    validate_dataset_input_dir(val_dir, label="Validation dataset directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    seeded_checkpoints: list[str] = []
    init_dir: Path | None = None
    resumed = False
    if args.init_dir:
        init_dir = Path(args.init_dir).resolve()
        available = sorted(
            p.name for p in init_dir.iterdir() if p.is_file() and p.suffix in (".pth", ".pt", ".mdlus")
        )
        if not available:
            print(
                f"WARNING: --init-dir {init_dir} contains no checkpoint files; "
                "training starts from scratch.",
                flush=True,
            )
        resumed, reason = plan_warm_start(init_dir, profile)
        if resumed:
            # Copying into network_dir is what makes physicsnemo treat this as a
            # resume of its own run: optimizer, LR schedule and step counter all
            # carry over. `initialization_network_dir` deliberately does not --
            # it leaves network_dir empty, which forces step 0.
            seeded_checkpoints = seed_network_dir(init_dir, output_dir)
            init_dir = None
            print(f"[warm-start] resuming optimizer and step: {reason}", flush=True)
        else:
            seeded_checkpoints = available
            print(f"[warm-start] weights only, fresh optimizer at step 0: {reason}", flush=True)
    ensure_module_available(TRAIN_MODULE)
    resolve_cuda_device(profile)
    command = build_train_command(profile, train_dir, val_dir, output_dir, init_dir=init_dir)
    time_binary = Path("/usr/bin/time")
    if time_binary.is_file():
        command = [str(time_binary), "-v", "-o", str(output_dir / "resource_usage.txt"), *command]
    command = wrap_with_gpu_seat(command, profile)

    # Measured peaks feed the automatic reservations in utils/vram.py; a
    # failing payload must still stop the thread, hence the try/finally.
    # RuntimeMeter counts reaped children, so it must bracket the payload run;
    # its wall clock includes any gpu-seat queue wait (cpu_seconds does not).
    sampler = GpuPeakSampler()
    sampler.start()
    meter = RuntimeMeter()
    # BEFORE the payload: this is the code about to run. Captured afterwards it
    # named HEAD at run end, which for a run that waited hours in the GPU queue
    # and then trained for a day is a commit that did not exist when the code
    # was loaded (measured worst drift: +51.1 h, nine commits).
    environment = collect_environment()
    try:
        run_streamed_subprocess(command, cwd=Path.cwd(), env=profile_environment(profile))
    finally:
        runtime = meter.stop()
        peak_mb = sampler.stop()

    # Before the summary: the config must be inside network_dir to be retrieved
    # with the checkpoints, otherwise the model node cannot describe itself.
    archive_hydra_config(output_dir)

    resolved_seed, seed_source = resolve_training_seed(profile, profile.get("config_name"))
    summary = {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "output_dir": str(output_dir),
        "profile": profile,
        # Seed swings peak-T by ~1700 K across replicates, so the resolved seed
        # is part of the result, not a config detail.
        "resolved_seed": resolved_seed,
        "resolved_seed_source": seed_source,
        "contains_hydra": (output_dir / ".hydra" / "config.yaml").is_file(),
        "seeded_checkpoints": seeded_checkpoints,
        "resumed_optimizer": resumed,
    }
    completed = steps_trained(output_dir)
    if completed is not None:
        summary["steps_trained"] = completed
        requested = profile.get("max_steps")
        if isinstance(requested, int) and completed < requested:
            print(f"[steps] stopped at {completed} of a requested {requested} steps "
                  f"({completed / requested:.0%}); `max_steps` is a ceiling, not a record",
                  flush=True)
    if isinstance(peak_mb, int) and peak_mb > 0:
        summary["gpu_vram_peak_mb"] = peak_mb
        _report_declaration_overrun(profile, "gpu_vram_mb", peak_mb, "GPU VRAM")
    host_peak_mb = runtime.get("max_rss_mb") or runtime.get("peak_rss_mb")
    if isinstance(host_peak_mb, (int, float)) and host_peak_mb > 0:
        summary["host_ram_peak_mb"] = int(host_peak_mb)
        _report_declaration_overrun(profile, "host_ram_mb", int(host_peak_mb), "host RAM")
    summary["runtime"] = {**runtime, "environment": environment}
    Path("train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run_with_live_tail(_main))
