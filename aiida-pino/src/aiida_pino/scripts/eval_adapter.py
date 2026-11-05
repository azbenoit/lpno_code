"""Standalone adapter for the packaged PINO evaluation entrypoint."""

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
    from aiida_pino.scripts.live_tail import (
        RuntimeMeter,
        collect_environment,
        parse_stdout_metric_dicts,
        profile_environment,
        run_streamed_subprocess,
        run_with_live_tail,
    )
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
    # From live_tail, which is staged beside this file. It used to be loaded
    # from `../utils/profiles.py`, a path that does not exist in a compute
    # box's sandbox -- and `exec_module` on a missing file raises
    # FileNotFoundError, which this `except ImportError` does not catch, so
    # every remote eval died at import.
    parse_stdout_metric_dicts = _LIVE_TAIL_MODULE.parse_stdout_metric_dicts

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
EVAL_MODULE = "lpno.models.NS_Energy_laser_PINO_eval"
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

    Without this, an AiiDA eval and a gpu_spool training can land on the same
    GPU at once. ``gpu_spool seat --gpu N -- <cmd>`` blocks until the card
    has room in the shared reservation ledger (``/var/tmp/pino_gpu_leases``).
    ``--vram-mb``/``--ram-mb`` ARE passed when the profile declares them
    (``_declared_memory_args`` below); this docstring said the opposite until
    2026-08-25, and a reader of it reported exclusive booking as unconditional.
    What remains true is the consequence of NOT declaring: with no ``gpu_vram_mb`` the
    run books the card *exclusively*, and with no ``host_ram_mb`` it takes gpu_spool's
    default share (``PINO_SPOOL_DEFAULT_RAM_MB``, 32 GB). So an honest "let it size
    itself" submission is the one least likely to start on a busy box -- which is worth
    knowing, and is not the same claim as "no memory flags are ever sent". The profile's
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
    # logs the acquired seat and restores this job's working directory (the
    # eval collects `fno_eval/` relative to it).
    # After the seat wait (can be hours), rebuild any staged dataset symlink the
    # cache lost in the meantime (disk-pressure cleanup evicts it), THEN exec the
    # training. The adapter-time repair cannot cover this window. `|| true` so a
    # repair miss falls through to the training's own missing-data error.
    repair = _shared_runtime_repair_command(sys.executable)
    shim = f'echo "[gpu-seat] acquired gpu $CUDA_VISIBLE_DEVICES"; cd {shlex.quote(str(cwd))} && {repair}; exec "$0" "$@"'
    # Evals are short and small; --priority parks a `.want` marker so queued
    # trainings hold off and this eval goes in ahead of them (it never preempts
    # a run already holding the card). Only the eval adapter is priority.
    #
    # Priority USED to cost a deadline: gpu_spool waits forever for a plain seat
    # (DEFAULT_SEAT_MAX_WAIT_S = 0) but gives up after
    # DEFAULT_PRIORITY_MAX_WAIT_S = 3600s with --priority, "so CI reports a
    # cause" -- and two evals of a finished model died at 3601s on 2026-07-29
    # while both cards stayed booked. That is no longer the trade: `--max-wait-s 0`
    # below keeps the jump and restores the unbounded wait, so the two are
    # separable and this comment's old "worth less than the deadline costs" no
    # longer applies.
    # Still default OFF for evals, now for the one remaining reason: priority parks
    # a `.want` marker that makes workers hold off starting new runs, which is how
    # a stuck priority job starved an otherwise idle GPU for hours.
    # Set `gpu_seat_priority: true` for the queue-jump.
    priority = profile.get("gpu_seat_priority", False)
    seat_args = ["--gpu", gpu_arg]
    if priority:
        seat_args.append("--priority")
        # ...but NOT gpu_spool's CI deadline. `--priority` alone swaps the unbounded
        # wait for DEFAULT_PRIORITY_MAX_WAIT_S = 3600 s "so CI reports a cause",
        # and on a saturated box that is what kills the job rather than slowing it:
        # training 99083 died at 3602 s without ever seating, waiting on a seat that
        # was ~33 h out, while two sibling arms without the flag were still alive
        # 12 h later. 0 is falsy at gpu_spool's deadline check, i.e. wait forever.
        seat_args += ["--max-wait-s", "0"]
    seat_args += _declared_memory_args(profile)
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
# train_adapter.py because these scripts are copied standalone into calcjob
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
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_scalar(item) for item in value) + "]"
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
            # Per-model marker: the eval merge always resolves it (stamped models
            # carry their value, pre-stamp models are pinned False), so an eval
            # reproduces the dose scaling its model trained on.
            *(
                # ++ (force-add): only pvlhs-era configs declare custom.power_weighted_dose,
                # so a plain override dies in struct mode on every other config -- it killed
                # the first mb fine-tune's test eval (46475). Force-add is valid either way.
                [f"++custom.power_weighted_dose={hydra_scalar(profile['power_weighted_dose'])}"]
                if profile.get("power_weighted_dose") is not None
                else []
            ),
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
    cuda_device = profile.get("cuda_device")
    if cuda_device not in (None, "") and not _cuda_device_is_auto(cuda_device):
        command.append(f"custom.cuda_device={cuda_device}")


#: Where an eval is told to write. Any fixed name works -- each job has its own working
#: directory -- and stating it removes the guesswork that lost 67015 and 67034.
EVAL_RUN_DIR = "outputs/aiida_eval"


def archived_config_dir(training_dir: Path) -> Path | None:
    """The model's own `.hydra` directory, when it archived one.

    A model node stores the resolved config its run actually used, and the
    staged `training_dir` carries it. Rebuilding the architecture from TODAY's
    packaged config instead is why one GAOT eval failed with "Current model architecture/config likely differs from the
    training run" -- the GAOT arch gained fields on 2026-08-05 and the intact
    `u_network.0.pth` became unloadable. Every stored model degrades that way as
    the schema moves, which is exactly what a provenance store must not do.
    """
    candidate = Path(training_dir) / ".hydra"
    return candidate if (candidate / "config.yaml").is_file() else None


#: Written by `lpno`' `npz_lazy_store` during training, and the ONLY record of
#: the grid a model actually served. The archived `.hydra/config.yaml` is not: an
#: NPZ-trained model's config keeps whatever `custom.geometry` said while the run replaced
#: it with a derived box (one model archives `[90, 40, 30]` and trained on 40 layers), so
#: reading geometry from the config is reading an intention, not a fact.
GRID_STAMP_NAME = "trained_grid.json"


def warn_if_trained_grid_is_unverifiable(training_dir: Path) -> str | None:
    """Say so when the model carries no grid stamp, instead of scoring in silence.

    With a stamp, lpno compares trained-vs-served and REPORTS the difference
    (standing policy, 2026-08-19: a model trained at reduced resolution is still
    scored at full resolution). Without one that comparison never runs, and a model scored
    on a domain it was never trained for returns plausible numbers -- one eval scored a
    30-layer model on 40 layers and was read as a model pathology before anyone checked the
    geometry. The absence of a check is not the same as a check that passed.
    """
    for candidate in (training_dir, training_dir.parent, training_dir.parent.parent):
        try:
            if (candidate / GRID_STAMP_NAME).is_file():
                return None
        except OSError:
            continue
    message = (
        f"[grid] NOTE: this model carries no {GRID_STAMP_NAME}, so the grid it TRAINED on "
        f"cannot be verified against the grid being served. A model scored on a domain it "
        f"was not trained for returns plausible numbers rather than an error -- if it was "
        f"trained on a curated crop, set `crop_z_top` on the evaluation dataset's "
        f"derivation profile to restore that domain."
    )
    print(message, flush=True)
    return message


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


def build_eval_command(profile: dict, dataset_dir: Path, training_dir: Path) -> list[str]:
    """Build the packaged evaluation command.

    Prefers the model's ARCHIVED config over the packaged one (since
    2026-08-14): an old checkpoint is evaluated under the config it
    was trained with, so it stays evaluable as the packaged schema moves. Falls
    back to the packaged config when the model archived none -- 29 of 243 models
    predate the archiving -- which is exactly today's behaviour.
    """
    warn_if_trained_grid_is_unverifiable(training_dir)
    archived = None if profile.get("use_packaged_config") else archived_config_dir(training_dir)
    config_flags = (
        # `.hydra/config.yaml` is the fully-composed config with no `defaults:`
        # list, so naming it directly reproduces the run rather than
        # recomposing today's groups.
        [f"--config-dir={archived}", "--config-name=config"]
        if archived is not None
        else [f"--config-name={profile['config_name']}"]
    )
    # Pin the output directory rather than inferring it afterwards. The packaged configs
    # set `hydra.run.dir` themselves, but a model's archived `config.yaml` carries no
    # `hydra:` block (hydra writes that to a separate `hydra.yaml`), so an archived-config
    # eval silently reverted to hydra's date-based default and its artifacts landed
    # somewhere discovery did not look. Stating it makes the location the same either way
    # and independent of which config won. PLAIN override: a `+` prefix would be rewritten
    # into the `custom.` namespace by `remap_plus_args_to_custom` and set nothing.
    config_flags = [*config_flags, f"hydra.run.dir={EVAL_RUN_DIR}"]
    command = [
        sys.executable,
        "-u",
        "-m",
        EVAL_MODULE,
        *config_flags,
        f"custom.name={profile['run_name']}_eval",
        f"network_dir={training_dir}",
        f"custom.data_dir={dataset_dir}",
        f"custom.val_data_dir={dataset_dir}",
        f"custom.wandb_offline={hydra_scalar(profile['wandb_offline'])}",
    ]
    append_runtime_overrides(command, profile, split_mode=str(profile.get("split_mode", "standard")))
    command.extend(promote_replayed_overrides(profile.get("hydra_overrides", [])))
    return command


def discover_raw_eval_dir(cwd: Path) -> Path:
    """Locate the pipeline's ``fno_eval`` artifact directory.

    Hydra runs chdir into ``outputs/<custom.name>_<timestamp>/`` (the config's
    ``hydra.run.dir``), so the eval artifacts land there rather than in the job
    CWD — which made every lazy-path eval report exit 502 with the metrics
    sitting on disk. Prefer ``cwd/fno_eval`` (legacy layout), else the newest
    ``outputs/*/fno_eval``.
    """
    raw = cwd / "fno_eval"
    if raw.exists():
        return raw
    # `outputs/*/fno_eval` matched ONE level, which is the run-named form the packaged
    # configs set (`hydra.run.dir: outputs/${custom.name}_${now:...}`). Hydra's DEFAULT is
    # `outputs/<date>/<time>/` -- TWO levels -- and an eval built from a model's archived
    # `.hydra/config.yaml` gets exactly that, because hydra keeps its own settings in a
    # separate `hydra.yaml` that the archived config does not carry. So 67015 and 67034
    # ran to completion, wrote their artifacts, and were failed for missing metrics with
    # the files on disk (2026-08-25). Search at any depth: the cost of a stray match is a
    # wrong number, the cost of no match is a discarded run, and `fno_eval` is specific.
    candidates = sorted(cwd.glob("outputs/**/fno_eval"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else raw


def collect_metric_payload(output_dir: Path, stdout_path: Path | None = None) -> dict:
    """Collect standard and autoregressive metric files into one JSON payload."""
    model_errors = output_dir / "model_errors.txt"
    rollout_full = output_dir / "model_errors_rollout_full.txt"
    rollout_k = output_dir / "model_errors_rollout_k.txt"

    payload: dict = {}
    if model_errors.exists():
        payload.update(parse_model_errors(model_errors))
    if rollout_full.exists():
        payload["rollout_full"] = parse_model_errors(rollout_full)
    if rollout_k.exists():
        payload["rollout_k"] = parse_model_errors(rollout_k)
    if not payload and stdout_path is not None and stdout_path.exists():
        payload.update(parse_stdout_metric_dicts(stdout_path.read_text(encoding="utf-8")))
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-file", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    training_dir = Path(args.training_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    profile = json.loads(Path(args.profile_file).read_text(encoding="utf-8"))
    validate_dataset_input_dir(dataset_dir, label="Evaluation dataset directory")
    ensure_module_available(EVAL_MODULE)
    resolve_cuda_device(profile)
    command = build_eval_command(profile, dataset_dir, training_dir)
    command = wrap_with_gpu_seat(command, profile)

    # Measured peaks feed the automatic reservations in utils/vram.py; a
    # failing payload must still stop the thread, hence the try/finally.
    sampler = GpuPeakSampler()
    sampler.start()
    meter = RuntimeMeter()
    # Before the payload: HEAD at run END is a different commit (see F5).
    environment = collect_environment()
    try:
        # Evaluation reads cases serially on purpose -- MEASURED, not assumed.
        #
        # This block previously forced PINO_LOADER_WORKERS=4 on the theory that eval
        # silently serialised while training did not, and that serial decompression
        # (13.56 s/case against ~1.65 s of compute) explained the multi-hour evals.
        # An A/B on the same model and the same 301 frames says otherwise:
        #
        #   4 workers: loader 144.6s (46%), forward  97.5s (31%), metrics 71.1s (23%)
        #   0 workers: loader 129.2s (41%), forward 113.7s (36%), metrics 71.9s (23%)
        #
        # Workers made loading SLOWER -- IPC and tensor serialisation cost more than
        # the parallel decompression recovers, because the eval loader reads cases in
        # order (shuffle=False) and a cache of 3 already gives full within-case reuse.
        # Totals matched within noise, so the original theory was wrong in mechanism
        # and in direction.
        #
        # There is no single hot spot: a full 6016-frame eval is ~2 h because a GAOT
        # forward over 600k grid nodes with 2.4M encoder edges costs that, times 6016.
        # The lever that works is fewer frames -- PINO_EVAL_FRAME_STRIDE (stride 20
        # scores 301 frames in 5.9 min). An explicit env in the profile still wins.
        eval_env = profile_environment(profile) or dict(os.environ)
        eval_env.setdefault("PINO_LAZY_CACHE", "3")
        run_streamed_subprocess(command, cwd=Path.cwd(), env=eval_env)
    finally:
        runtime = meter.stop()
        peak_mb = sampler.stop()

    raw_eval_dir = discover_raw_eval_dir(Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw_eval_dir.exists():
        for path in raw_eval_dir.iterdir():
            target = output_dir / path.name
            if path.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)

    metrics = collect_metric_payload(output_dir, Path("aiida.out"))
    Path("metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    # Which config produced this score. A number computed under the model's own
    # archived config and one computed under today's packaged config are not the
    # same measurement, and the difference is invisible in the metrics.
    _archived = archived_config_dir(training_dir)
    _config_source = ("archived"
                      if (_archived is not None and not profile.get("use_packaged_config"))
                      else "packaged")
    summary = {
        "dataset_dir": str(dataset_dir),
        "training_dir": str(training_dir),
        "output_dir": str(output_dir),
        "profile": profile,
        "config_source": _config_source,
        "archived_config_available": _archived is not None,
        "metrics_file_present": bool(metrics),
        "metrics_source": "stdout" if metrics and not any((output_dir / name).exists() for name in (
            "model_errors.txt",
            "model_errors_rollout_full.txt",
            "model_errors_rollout_k.txt",
        )) else "files",
    }
    if isinstance(peak_mb, int) and peak_mb > 0:
        summary["gpu_vram_peak_mb"] = peak_mb
    summary["runtime"] = {**runtime, "environment": environment}
    Path("eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run_with_live_tail(_main))
