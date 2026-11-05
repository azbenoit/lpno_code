"""Helpers for remote live-tail capture in CalcJob adapter scripts."""

from __future__ import annotations

import ast
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback
from typing import Callable, Iterator


def _lines(command: list[str]) -> list[str]:
    """Every output line, for queries that report one row per device."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001 - environment capture is best-effort
        pass
    return []


def _first_line(command: list[str]) -> str:
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001 - environment capture is best-effort
        pass
    return ""


def _module_version(name: str) -> str:
    try:
        import importlib

        return str(getattr(importlib.import_module(name), "__version__", ""))
    except Exception:  # noqa: BLE001
        return ""


def _repo_identity(module_name: str) -> tuple[str, bool, str]:
    """(git SHA, dirty, src tree hash) of the checkout providing ``module_name``.

    The SHA alone overstates what it knows, in two different ways.

    First, these are editable checkouts that get edited while runs are queued,
    so an uncommitted working tree means the code that ran is not the code that
    SHA names — hence ``dirty``.

    Second, and this is what the third field is for: **a commit hash is not a
    content address.** It is a name in one history, and a history rewrite
    destroys it. On 2026-08-21 a fast-import purging `Important_models/`
    re-hashed every commit after `4b25162`; of 196 distinct
    `lpno_sha` values recorded across 1834 nodes, **137 stopped
    being ancestors of HEAD** and resolve in no fresh clone. Recording the sha
    EARLIER would not have helped — it was already recorded correctly, at run
    time, on the box that ran. Nor can it be corrected afterwards: AiiDA seals a
    stored node's attributes.

    The `src/` TREE hash is a content address, and it survived that rewrite
    exactly: `git rev-parse <old>:src == <new>:src` held for **all 138**
    rewritten pairs, 0 differing. A run that records its src tree can be tied to
    its code with no remap, no rescue branch, and no reliance on objects that
    only exist on one machine.

    Looking a tree hash up yields a SET of commits, not one, whenever
    consecutive commits changed nothing under `src/`. That is the right
    granularity -- the question is which code ran, not which commit named it,
    and every commit in the set is byte-identical there. Do not narrow it.
    """
    try:
        import importlib

        module = importlib.import_module(module_name)
        root = Path(module.__file__).resolve()
        for parent in root.parents:
            if (parent / ".git").exists():
                repo = str(parent)
                sha = _first_line(["git", "-C", repo, "rev-parse", "--short=12", "HEAD"])
                dirty = bool(_first_line(["git", "-C", repo, "status", "--porcelain"]))
                # `HEAD:<src dir>`, not the whole tree: the purge stripped a sibling
                # directory, so full-tree hashes legitimately differ across the
                # rewrite while the code that ran is byte-identical. The src directory is
                # the package's parent, addressed RELATIVE TO THE REPO ROOT: `src` in this
                # checkout, `aiida-pino/src` in the combined public repository where the
                # package sits one level down (a literal `HEAD:src` recorded '' there).
                src_dir = root.parent.parent if root.name == "__init__.py" else root.parent
                rel = src_dir.relative_to(parent).as_posix()
                src_tree = _first_line(["git", "-C", repo, "rev-parse", "--short=12", f"HEAD:{rel}"])
                return sha, dirty, src_tree
    except Exception:  # noqa: BLE001
        pass
    return "", False, ""


def _repo_sha(module_name: str) -> str:
    return _repo_identity(module_name)[0]


#: Regime labels. A LABEL, not a schema: by agreement nothing may
#: parse these, so the value is free to become e.g. `tf32-conv+bf16` later
#: without a migration. Greppable on purpose.
FP32_CONV_REGIME = "fp32-conv"
TF32_CONV_REGIME = "tf32-conv"


def numerical_regime_fields(
    *,
    cudnn_allow_tf32: bool,
    capability: tuple[int, int] | None,
    matmul_allow_tf32: bool | None = None,
    cudnn_version=None,
    cuda_runtime=None,
) -> dict:
    """Raw numerical settings plus the one derived field that carries meaning.

    `cudnn.allow_tf32` defaults to True everywhere, and Turing (sm_75) has no
    TF32 units — so the flag is a no-op on the primary and honoured on Ada, and
    recording it alone would assert the two boxes AGREE when that flag is the
    entire source of their disagreement. `tf32_conv_active` is the discriminator,
    and it is derived HERE, at capture time, on the box that ran the job: a
    capability query can lie (`torch.cuda.is_bf16_supported()` returns True on
    this TITAN RTX, where bf16 is emulated), so a capability-derived fact must
    never be reconstructed later from a `gpu_name` string.

    Returns `{}` when there is no capability to reason from — a CPU-only or
    unreadable device gets NO regime claim rather than a default one.
    """
    if not capability:
        return {}
    major, minor = int(capability[0]), int(capability[1])
    active = bool(cudnn_allow_tf32) and (major, minor) >= (8, 0)
    fields = {
        "cudnn_allow_tf32": bool(cudnn_allow_tf32),
        "compute_capability": f"{major}.{minor}",
        "tf32_conv_active": active,
        "numerical_regime": TF32_CONV_REGIME if active else FP32_CONV_REGIME,
    }
    if matmul_allow_tf32 is not None:
        fields["matmul_allow_tf32"] = bool(matmul_allow_tf32)
    if cudnn_version:
        fields["cudnn_version"] = str(cudnn_version)
    if cuda_runtime:
        fields["cuda_runtime"] = str(cuda_runtime)
    return fields


def _capture_numerics() -> dict:
    """Read the numerical settings of the device ACTUALLY in use.

    `torch.cuda.current_device()`, not device 0 and not the config's index:
    `gpu_spool` pins `CUDA_VISIBLE_DEVICES` to the booked card and
    `configure_device` re-exports it from the physical index before first CUDA
    init, so which index maps to which card depends on the layer that won. On a
    homogeneous box it makes no difference; on a mixed-GPU box reading the wrong
    device silently records the wrong regime.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        return numerical_regime_fields(
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            capability=torch.cuda.get_device_capability(device),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_version=getattr(torch.backends.cudnn, "version", lambda: None)(),
            cuda_runtime=getattr(torch, "version", None) and getattr(torch.version, "cuda", None),
        )
    except Exception:  # noqa: BLE001 - environment capture is best-effort
        return {}


def collect_environment() -> dict:
    """Software + hardware identity of this job, for the run summary.

    A result that cannot be tied to the code and machine that produced it is
    only half provenance: the profile records *what was asked*, this records
    *what actually ran*. Everything is best-effort — a missing tool yields an
    empty field, never an error.
    """
    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    except Exception:  # noqa: BLE001
        pass
    env = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "python": platform.python_version(),
        "gpu_name": _first_line(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "gpu_driver": _first_line(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    for module in ("torch", "physicsnemo", "numpy"):
        version = _module_version(module)
        if version:
            env[f"{module}_version"] = version
    for module in ("aiida_pino", "lpno"):
        sha, dirty, src_tree = _repo_identity(module)
        if sha:
            env[f"{module}_sha"] = sha
            if dirty:
                env[f"{module}_dirty"] = True
        # Rewrite-proof source identity. Recorded beside the sha, never instead
        # of it: the sha is what a human greps and what links to a commit
        # message, the tree is what still resolves when that commit no longer
        # exists under that name.
        if src_tree:
            env[f"{module}_src_tree"] = src_tree
    # `captured_at` says WHEN this snapshot was taken. Called after the payload,
    # it recorded HEAD at run end: 21 of 21 models with an environment block
    # matched HEAD at `started_utc + wall_seconds`, worst drift +51.1 h, so a
    # field that looked authoritative named commits made during the run.
    env["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # NESTED, deliberately. The comprehension below drops falsy values, and
    # `tf32_conv_active: False` is the PRIMARY's value — the one a comparison
    # depends on. Dropped, it becomes absence, indistinguishable from a run that
    # never captured it, which is exactly the ambiguity this field exists to
    # end. A non-empty dict is truthy, so everything inside it survives.
    numerics = _capture_numerics()
    if numerics:
        env["numerics"] = numerics
    return {k: v for k, v in env.items() if v}


class RuntimeMeter:
    """Wall/CPU/RSS of everything this adapter ran, for the run summary.

    ``resource.getrusage(RUSAGE_CHILDREN)`` accumulates over *reaped* children,
    which is exactly the payload subprocesses `run_streamed_subprocess` waits
    on. Sampled as start/stop deltas so the meter only counts this run.
    """

    def __init__(self) -> None:
        self._wall0 = time.monotonic()
        self._t0 = time.time()
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._utime0, self._stime0 = r.ru_utime, r.ru_stime
        self._contention0 = contention_snapshot()

    def stop(self) -> dict:
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.monotonic() - self._wall0
        payload = {
            "wall_seconds": round(wall, 2),
            "cpu_user_seconds": round(r.ru_utime - self._utime0, 2),
            "cpu_system_seconds": round(r.ru_stime - self._stime0, 2),
            # peak RSS of the largest reaped child, in MiB (ru_maxrss is KiB on Linux)
            "max_rss_mb": int(r.ru_maxrss / 1024),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._t0)),
            # What else the box was doing, at both ends. A wall clock from a
            # shared box is not comparable with one from an idle box, and there
            # is no way to make it so after the fact -- but a run that RECORDS
            # what it was competing with can be compared with like.
            "contention_start": self._contention0,
            "contention_end": contention_snapshot(),
        }
        return payload


def contention_snapshot() -> dict:
    """What else was running, cheaply enough to call at both ends of a run.

    Timing on a shared machine is never contention-free: two GPUs have been seen
    at 100%/81% simultaneously. Rather than pretend otherwise, every run records
    the load it ran under, so a later comparison can restrict itself to runs
    that were actually alone (`solo`) instead of averaging over a mixture.

    Cost budget: `getloadavg` and `/proc` reads are free; the one `nvidia-smi`
    call is ~50 ms, taken twice for a run that lasts minutes to days. Everything
    is best effort -- a missing tool yields a missing field, never an error and
    never a delay that matters.
    """
    snapshot: dict = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        one, five, fifteen = os.getloadavg()
        snapshot["loadavg"] = [round(one, 2), round(five, 2), round(fifteen, 2)]
        snapshot["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    # GPU utilisation and the SM clock, so throttling is visible rather than
    # showing up as unexplained slowness.
    fields = "utilization.gpu,clocks.sm,memory.used,temperature.gpu"
    gpus = []
    for row in _lines(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]):
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append({"util_pct": int(parts[0]), "sm_mhz": int(parts[1]),
                         "mem_used_mb": int(parts[2]), "temp_c": int(parts[3])})
        except ValueError:
            continue
    if gpus:
        snapshot["gpus"] = gpus
        snapshot["gpu_busy"] = sum(1 for gpu in gpus if gpu["util_pct"] >= 10)
    # FLOW-3D seat: held by someone else, and how many are queued behind it.
    try:
        spool = Path(os.environ.get("F3D_SPOOL_DIR") or (Path.home() / ".flow3d" / "spool"))
        waiting = spool / "waiting"
        snapshot["seat_waiters"] = len(list(waiting.glob("*.json"))) if waiting.is_dir() else 0
    except Exception:  # noqa: BLE001
        pass
    return snapshot


def profile_environment(profile: dict) -> dict[str, str] | None:
    """Process env for an adapter subprocess: os.environ overlaid with the
    training profile's `env` mapping (PINO_LAZY_NPZ, PYTORCH_CUDA_ALLOC_CONF,
    …). None when the profile sets nothing, so Popen inherits as before."""
    overlay = profile.get("env") or {}
    if not isinstance(overlay, dict) or not overlay:
        return None
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in overlay.items()})
    return env

LIVE_TAIL_FILENAME = ".aiida_pino_live_tail.log"


class _TeeStream:
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def writelines(self, lines) -> None:
        # run_streamed_subprocess calls this to echo a failed child's tail. Without
        # it the AttributeError replaced the child's real error with a traceback
        # into this module, so every subprocess failure looked like a live_tail
        # bug and the actual cause was lost (2026-07-22).
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def live_tail_context(filename: str = LIVE_TAIL_FILENAME) -> Iterator[Path]:
    """Mirror adapter stdout/stderr to a temporary live-tail file."""
    path = Path(filename).resolve()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    success = False

    with path.open("w", encoding="utf-8", buffering=1) as handle:
        stdout_tee = _TeeStream(original_stdout, handle)
        stderr_tee = _TeeStream(original_stderr, handle)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            try:
                yield path
            except BaseException:
                raise
            else:
                success = True

    if success:
        path.unlink(missing_ok=True)


def single_process_environment(env: dict[str, str] | None) -> dict[str, str]:
    """`env` (or this process's) without the Slurm variable that makes physicsnemo go distributed.

    physicsnemo's `DistributedManager` reads SLURM_PROCID as an `srun` launch and then needs
    SLURM_LAUNCH_NODE_IPADDR, which only `srun` sets. A job run from a Slurm batch script (the
    AiiDA scheduler's own job, or `verdi`/`aiida-pino --foreground` inside one) has the first
    and not the second, and every training and evaluation died on
    `os.environ["MASTER_ADDR"] = None` (2026-10-08, reproduction from the deposit on a CPU
    cluster). A genuine `srun` launch keeps both and is left alone.
    """
    out = dict(os.environ if env is None else env)
    if "SLURM_PROCID" in out and "SLURM_LAUNCH_NODE_IPADDR" not in out:
        out.pop("SLURM_PROCID")
    return out


def run_streamed_subprocess(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a subprocess and stream merged stdout/stderr through the current stdout."""
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=single_process_environment(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=80)
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            tail.append(line)
        return_code = process.wait()
    finally:
        process.stdout.close()

    if return_code != 0:
        # Echo the end of the child's output to STDERR so the scheduler
        # stderr file carries the real traceback, not just the argv.
        if tail:
            sys.stderr.write(f"--- last {len(tail)} line(s) of child output ---\n")
            sys.stderr.writelines(tail)
            sys.stderr.flush()
        raise subprocess.CalledProcessError(return_code, command)


def run_with_live_tail(main: Callable[[], int | None]) -> int:
    """Execute a main function with stdout/stderr mirrored to the live-tail file."""
    with live_tail_context():
        try:
            result = main()
        except SystemExit as exc:
            if exc.code in (None, 0):
                return 0
            if isinstance(exc.code, str):
                print(exc.code, file=sys.stderr)
                raise SystemExit(1) from None
            raise
        except BaseException:
            traceback.print_exc()
            raise SystemExit(1) from None
        if isinstance(result, int) and result != 0:
            raise SystemExit(result)

    if isinstance(result, int):
        return result
    return 0


def parse_stdout_metric_dicts(text: str) -> dict[str, dict[str, float]]:
    """Parse field metric dictionaries printed to stdout by packaged PINO eval runs."""
    payload: dict[str, dict[str, float]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        field_name, raw_metrics = [part.strip() for part in line.split(":", 1)]
        if not field_name or not raw_metrics.startswith("{") or not raw_metrics.endswith("}"):
            continue

        try:
            parsed = ast.literal_eval(raw_metrics)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue

        metrics = {}
        for key, value in parsed.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        if metrics:
            payload[field_name] = metrics

    return payload
