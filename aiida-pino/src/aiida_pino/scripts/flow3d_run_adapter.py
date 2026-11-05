"""Standalone adapter that launches a `flow3D_scripts` FLOW-3D run as a tracked job.

The runner repo's ``python/run_rectangular_case.py`` is invoked in place (it
imports sibling modules from its own directory, so it must never be copied).
The runner root is resolved, in order, from the execution profile's
``flow3d_scripts_root``, the ``F3D_SCRIPTS_ROOT`` environment variable, or the
``external/flow3D_scripts`` submodule of the aiida-pino checkout.

License handling: the FLOW-3D license endpoint comes exclusively from the
runner environment (``F3D_LICENSE`` / ``F3DTKNUX_LICENSE_FILE`` /
``LM_LICENSE_FILE`` or the runner repo's git-ignored ``license.env``). This
adapter never reads, prints, or forwards license values; the inherited process
environment is passed through untouched and is never echoed.

The compact NPZ export stays on disk under the run's ``out_root``; only small
curated files are copied into ``run_artifacts/`` for AiiDA retrieval, and the
summary records absolute paths. The runner itself deletes the multi-GB
``flsgrf*`` solver output once that export completes, so nothing here should
expect it to survive; pass ``keep_flsgrf`` in the execution profile to keep it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    from aiida_pino.scripts.live_tail import RuntimeMeter, collect_environment, run_streamed_subprocess, run_with_live_tail
except ImportError:
    _LIVE_TAIL_SPEC = importlib.util.spec_from_file_location("aiida_pino_live_tail", Path(__file__).with_name("live_tail.py"))
    if _LIVE_TAIL_SPEC is None or _LIVE_TAIL_SPEC.loader is None:
        raise
    _LIVE_TAIL_MODULE = importlib.util.module_from_spec(_LIVE_TAIL_SPEC)
    _LIVE_TAIL_SPEC.loader.exec_module(_LIVE_TAIL_MODULE)
    run_streamed_subprocess = _LIVE_TAIL_MODULE.run_streamed_subprocess
    run_with_live_tail = _LIVE_TAIL_MODULE.run_with_live_tail
    collect_environment = _LIVE_TAIL_MODULE.collect_environment
    RuntimeMeter = _LIVE_TAIL_MODULE.RuntimeMeter

RUNNER_RELATIVE = Path("python") / "run_rectangular_case.py"
SUMMARY_FILENAME = "flow3d_run_summary.json"
ARTIFACTS_DIRNAME = "run_artifacts"
RUNHYD_TAIL_LINES = 200
# Small per-run files worth keeping in AiiDA provenance; never flsgrf*/NPZ.
SMALL_ARTIFACT_NAMES = ("case_manifest.json", "summary.md", "path.png", "machines")
# Machine budget for a FLOW-3D solve: never more than 10 cores.
DEFAULT_SCOPE_CORES = 10
# Grace between TERM and KILL for the coreutils `timeout` wall-clock wrapper.
WATCHDOG_KILL_GRACE_S = 60
# Default wall seconds without sim_time progress before the monitor aborts a solve.
DEFAULT_STALL_S = 900.0


def systemd_scope_available() -> bool:
    """Probe once whether a systemd user scope with resource properties can start."""
    if shutil.which("systemd-run") is None:
        return False
    try:
        probe = subprocess.run(
            ["systemd-run", "--user", "--scope", "-p", "CPUQuota=100%", "/bin/true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def apply_resource_scope(command: list[str], profile: dict, *, probe=None) -> tuple[list[str], str]:
    """Wrap the runner in the systemd resource caps of the solve machine budget.

    Returns ``(command, state)`` with state one of ``active``/``disabled``/
    ``unavailable``. A missing or broken systemd never fails the job; the run
    then proceeds uncapped and the summary records ``"resource_scope":
    "unavailable"``.
    """
    if probe is None:
        probe = systemd_scope_available  # resolved at call time so tests can monkeypatch it
    if profile.get("resource_scope") is False:
        return command, "disabled"
    if not probe():
        print(
            "[resource-scope] WARNING: systemd-run user scope unavailable; running the solver uncapped.",
            flush=True,
        )
        return command, "unavailable"
    cores = int(profile.get("cores") or DEFAULT_SCOPE_CORES)
    quota = cores * 100
    # No hard memory cap by default. flow3D_scripts' batch wrappers drop it on
    # purpose: measured 2026-07-14, FLOW-3D's ~13 GB flsgrf write-back page cache is
    # charged to the cgroup while anon RSS stays ~1.2 GB, so a 16G ceiling OOM-kills
    # long solves that are behaving perfectly. CPUQuota is what keeps interactive SSH
    # responsive; `memory_max` is there for callers who have measured their own.
    memory_max = str(profile.get("memory_max") or "").strip()
    caps = ["-p", f"CPUQuota={quota}%"]
    if memory_max:
        caps += ["-p", f"MemoryMax={memory_max}"]
    print(
        f"[resource-scope] systemd user scope: nice 10, CPUQuota={quota}%"
        + (f", MemoryMax={memory_max}" if memory_max else ", no memory cap"),
        flush=True,
    )
    wrapped = [
        "systemd-run",
        "--user",
        "--scope",
        "--nice=10",
        *caps,
        "--",
        *command,
    ]
    return wrapped, "active"


#: Which candidate `resolve_runner_root` actually used, for error messages and
#: for the run summary — a result nobody can tie to a runner checkout is only
#: half provenance.
_RESOLVED_FROM: dict[str, str] = {}


def _default_repo_root() -> Path | None:
    """Derive the aiida-pino checkout root from the installed package (editable installs)."""
    try:
        import aiida_pino
    except ImportError:
        return None
    package_file = getattr(aiida_pino, "__file__", None)
    if not package_file:
        return None
    # <repo>/src/aiida_pino/__init__.py -> <repo>
    return Path(package_file).resolve().parents[2]


#: Config keys the runner must *understand*, and where they live. A runner that
#: predates one of these does not reject it — it ignores it — so a config asking
#: for calibrated physics renders pre-calibration physics with no error anywhere.
#: Every entry arrived with the 2026-07-29 calibration campaign.
RUNNER_REQUIRED_SUPPORT = {
    ("simulation", "recoil_model"): "recoil_model",
    ("simulation", "void_vapor_pressure_pa"): "void_vapor_pressure_pa",
    ("laser", "fresnel_from_material"): "fresnel_from_material",
}


def runner_identity(runner_root: Path, source: str = "") -> dict:
    """The runner checkout that rendered this case: path, source, git sha, dirty.

    `dirty` for the same reason `live_tail._repo_identity` reports it on the
    python repos: a SHA from an edited working tree does not name the code that
    ran. This was the only code-identity field without one, and it is not
    hypothetical — of the 81 run summaries carrying a runner block, 31 record
    `lpno_dirty` and 6 `aiida_pino_dirty`, so 37 of 81 ran from a
    dirty editable checkout on the python side alone.
    """
    sha = ""
    dirty = False
    try:
        import subprocess

        sha = subprocess.run(
            ["git", "-C", str(runner_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(runner_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip())
    except Exception:  # noqa: BLE001 - identity is best effort, never fatal
        pass
    if dirty and source and "flow3d_scripts_root" not in source:
        # A dirty checkout nobody named is the worst of both: the physics came
        # from uncommitted edits in a root that was merely guessed.
        print(f"[runner] WARNING: {runner_root} has uncommitted changes and was resolved from "
              f"{source!r} rather than named explicitly; `runner.sha` {sha} does not describe "
              "what actually rendered this case.", flush=True)
    return {
        "root": str(runner_root),
        "resolved_from": source,
        "sha": sha,
        "dirty": dirty,
        "supports": {token: runner_supports(runner_root, token) for token in sorted(set(RUNNER_REQUIRED_SUPPORT.values()))},
    }


def runner_supports(runner_root: Path, token: str) -> bool:
    """Does this checkout's runner know `token` at all?

    Crude on purpose: the runner has no capability API, and the alternative is
    trusting a checkout whose age nobody recorded. A literal occurrence in the
    source is exactly the evidence that distinguishes the two checkouts here
    (7 occurrences of `recoil_model` in the current runner, 0 in the pinned
    submodule).
    """
    try:
        return token in (runner_root / RUNNER_RELATIVE).read_text(encoding="utf-8")
    except OSError:
        return False


def assert_runner_supports_config(runner_root: Path, config: dict, *, source: str = "") -> None:
    """Refuse a config whose physics this runner would silently ignore.

    The failure this prevents is the worst kind available here: the run
    succeeds, the data looks fine, and it is pre-calibration physics filed
    beside calibrated data. Measured 2026-08-05: the console left
    `flow3d_scripts_root` unset, so its runs fell through to
    `external/flow3D_scripts` — pinned at 2026-07-01, a month before the
    calibration campaign — and rendered `pv0=50650` where the config asked for
    `void_vapor_pressure_pa=101325`.
    """
    missing = []
    for (section, key), token in RUNNER_REQUIRED_SUPPORT.items():
        block = config.get(section)
        if isinstance(block, dict) and key in block and not runner_supports(runner_root, token):
            missing.append(f"{section}.{key}")
    if not missing:
        return
    raise ValueError(
        f"the flow3D_scripts checkout at `{runner_root}`"
        + (f" (resolved from {source})" if source else "")
        + f" does not support: {', '.join(sorted(missing))}. It would IGNORE these keys and "
        "render different physics than the config asks for, with no error. Point "
        "`flow3d_scripts_root` (or F3D_SCRIPTS_ROOT) at a current checkout, or update the "
        "`external/flow3D_scripts` submodule."
    )


def resolve_runner_root(profile: dict) -> Path:
    """Resolve the flow3D_scripts checkout that carries python/run_rectangular_case.py."""
    candidates: list[tuple[str, Path | None]] = []
    profile_root = str(profile.get("flow3d_scripts_root") or "").strip()
    if profile_root:
        candidates.append(("execution_profile.flow3d_scripts_root", Path(profile_root)))
    env_root = os.environ.get("F3D_SCRIPTS_ROOT", "").strip()
    if env_root:
        candidates.append(("F3D_SCRIPTS_ROOT", Path(env_root)))
    repo_root = _default_repo_root()
    if repo_root is not None:
        # The sibling checkout BEFORE the vendored submodule. 1175 of 1178
        # Flow3DRunCalculations pin `<...>/flow3D_scripts`, and the three that
        # did not were the console submits that fell through to the submodule --
        # pinned a month behind master, silently rendering pre-calibration
        # physics. Omitting `flow3d_scripts_root` should land on the checkout
        # everything else uses, not on the stalest copy in the tree.
        candidates.append(("sibling flow3D_scripts checkout", repo_root.parent / "flow3D_scripts"))
        candidates.append(("external/flow3D_scripts submodule", repo_root / "external" / "flow3D_scripts"))

    for label, root in candidates:
        resolved = root.expanduser().resolve()
        runner = resolved / RUNNER_RELATIVE
        if runner.is_file():
            _RESOLVED_FROM["label"] = label
            if label == "external/flow3D_scripts submodule":
                print(
                    "[runner] WARNING: falling back to the vendored external/flow3D_scripts "
                    f"submodule at {resolved}. It is pinned and can be months behind the "
                    "checkout every other run uses, which is how a config asking for "
                    "calibrated physics rendered pre-calibration physics. Set "
                    "`flow3d_scripts_root` in the execution profile or F3D_SCRIPTS_ROOT.",
                    flush=True,
                )
            return resolved
        # An explicitly-configured root that lacks the runner is a hard error,
        # not something to silently fall through. The two derived candidates
        # (sibling checkout, vendored submodule) are guesses, so a missing
        # runner there just moves on to the next guess.
        if label not in ("external/flow3D_scripts submodule", "sibling flow3D_scripts checkout"):
            raise ValueError(f"{label} is `{resolved}` but `{runner}` does not exist.")

    raise ValueError(
        "Could not resolve the flow3D_scripts runner root. Provide one of: "
        "`flow3d_scripts_root` in the execution profile, the `F3D_SCRIPTS_ROOT` "
        "environment variable, a `flow3D_scripts` checkout beside this one, or "
        "`external/flow3D_scripts` inside it "
        "(each must contain `python/run_rectangular_case.py`)."
    )


# Holds the seat for this adapter's lifetime; released when the process exits.
_SEAT = contextlib.ExitStack()


def spool_dir() -> Path:
    override = os.environ.get("F3D_SPOOL_DIR")
    return Path(override) if override else Path.home() / ".flow3d" / "spool"


@contextlib.contextmanager
def hold_seat(priority: int = 0):
    """Take the machine-wide FLOW-3D seat before the solve clock starts.

    The runner takes this same lock itself, but it does so *inside* the `timeout`
    wrapper, so a case submitted alongside others sits in the queue burning its
    own solve budget -- submit N at once and everything past the first few dies.
    Acquiring here, before the wrapper is built, makes a simultaneous batch
    serialize on its own: each job blocks here (no clock running) and only starts
    its timeout once it actually owns the seat.

    The seat is handed to the child via `F3D_SEAT_HELD=1` rather than released,
    because a release/re-acquire gap would let another job slip in between. The
    lock drops when this process exits, including on a kill.
    """
    lock_dir = spool_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_dir / ".seat.lock", os.O_CREAT | os.O_WRONLY, 0o644)
    ticket = None
    try:
        # The fast path used to grab a free seat without ever reading `waiting/`,
        # so a job ARRIVING during a handoff jumped every queued higher-priority
        # waiter -- measured deterministically inside the free window. Consulting
        # the queue first costs one directory listing and only bites when
        # somebody is actually waiting for more.
        queued = bool(_better_waiters(lock_dir, priority))
        if queued:
            print("[seat] a higher-priority run is queued; joining the queue rather than "
                  "taking the free seat", flush=True)
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                queued = True
                print("[seat] waiting for the FLOW-3D seat (no solve clock running yet) ...",
                      flush=True)
        if queued:
            ticket = _register_waiter(lock_dir, priority)
            _wait_for_seat(fd, lock_dir, priority)
            print("[seat] acquired", flush=True)
        yield
    finally:
        if ticket is not None:
            ticket.unlink(missing_ok=True)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


SEAT_POLL_S = 5.0

#: How long a registered ticket stays believable, and the ONLY bound on
#: deference.
#:
#: There used to be a second one, `MAX_DEFER_S = 3600`, after which a waiter
#: took its turn regardless. It was removed: two bounds guarded the same failure
#: and the time cap was the worse of them, because it made priority mean
#: something surprising. You set `seat_priority=9` and a priority-0 run that had
#: been queuing an hour took the seat in front of it — the opposite of what the
#: field says, and invisible unless you read the log.
#:
#: Starvation is what a bound is for, and this one covers it. The FLOW-3D seat
#: has no admission criteria beyond the lock itself: a ticket-holder is sitting
#: in `_wait_for_seat` blocked on the flock, so if it can take the lock it can
#: run. That is unlike `gpu_spool`, where a waiter can be admitted-but-blocked
#: on host RAM and genuinely never start (its own docs record a CI seat holding
#: GPU 0 empty for 2.7 h). The only way a higher-priority ticket blocks anything
#: forever here is if its owner is dead, zombied or SIGSTOPped — which the pid
#: check and this age cap already handle.
#:
#: So ordering is now strict: a lower-priority run waits while any live, fresh,
#: higher-priority run is queued. A continuous stream of urgent work does starve
#: it, and that is the honest meaning of "priority" — it is visible in
#: `aiida-pino paused` and `flow3d-queue`, and it is a queue the operator
#: created.
TICKET_MAX_AGE_S = 6 * 3600.0


def _waiting_dir(lock_dir: Path) -> Path:
    path = lock_dir / "waiting"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _register_waiter(lock_dir: Path, priority: int) -> Path | None:
    """Announce this waiter and what it is worth. Best effort."""
    try:
        ticket = _waiting_dir(lock_dir) / f"{os.getpid()}.json"
        ticket.write_text(
            json.dumps({"pid": os.getpid(), "priority": int(priority), "since": time.time()}),
            encoding="utf-8",
        )
        return ticket
    except Exception:  # noqa: BLE001 - priority is an optimisation, never a gate
        return None


def _live_waiters(lock_dir: Path) -> list[dict]:
    """Registered waiters that still plausibly want the seat; stale tickets pruned.

    `os.kill(pid, 0)` alone is not liveness. A zombie (`ZN`) and a SIGSTOPped
    process (`TN`) both pass it, and each held the seat idle for the whole
    deference budget in testing; pids also recycle, measured at 75.8/s against
    `pid_max` 4194304, wrapping about every 0.6 days. So the `since` field --
    written since the first version and never read -- caps a ticket's life at
    the deference budget. A ticket cannot outlive the longest wait it could
    possibly justify.
    """
    out = []
    now = time.time()
    try:
        entries = list(_waiting_dir(lock_dir).glob("*.json"))
    except Exception:  # noqa: BLE001
        return out
    for path in entries:
        try:
            row = json.loads(path.read_text())
            os.kill(int(row["pid"]), 0)
            age = now - float(row.get("since") or 0)
        except (ProcessLookupError, ValueError, KeyError, json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)
            continue
        if age > TICKET_MAX_AGE_S:
            # Nothing should defer to this any longer, whatever its pid says.
            path.unlink(missing_ok=True)
            continue
        out.append(row)
    return out


def _better_waiters(lock_dir: Path, priority: int) -> list[dict]:
    """Live waiters worth strictly more than `priority`, excluding this process."""
    try:
        return [
            waiter for waiter in _live_waiters(lock_dir)
            if int(waiter.get("priority", 0)) > int(priority)
            and int(waiter.get("pid", 0)) != os.getpid()
        ]
    except Exception:  # noqa: BLE001 - priority is an optimisation, never a gate
        return []


def _wait_for_seat(fd: int, lock_dir: Path, priority: int) -> None:
    """Take the seat, yielding first to any waiter that is worth more.

    A plain blocking `flock` hands the seat to whoever the kernel wakes, so
    "run this next" had exactly one implementation: pause the other sweeps by
    hand. That is coarse -- it stops a whole sweep rather than deprioritising
    it -- and it leaves paused work that nothing is scheduled to resume.

    Ordering is strict and cooperative: stand aside while any live, fresh,
    higher-priority run is queued, for as long as that stays true. Any failure
    in the bookkeeping falls straight through to a plain blocking `flock`, and
    a ticket stops counting once its owner dies or it exceeds
    `TICKET_MAX_AGE_S` — which is the one bound against starvation.
    """
    # Deference is "decline to compete", not "take it then hand it back".
    # Taking the lock in order to release it wasted up to SEAT_POLL_S of seat on
    # EVERY handoff (measured: median 3.5 s against 0.0001 s for a plain
    # blocking flock), made the console's seat monitor blind because a waiter
    # asleep in time.sleep never appears as a `->` line in /proc/locks, and lost
    # 5/5 races to the runner's own blocking flock even at priority 9.
    #
    # So: block on LOCK_EX -- zero-latency handoff, visible to /proc/locks --
    # unless someone better is queued, in which case simply do not compete.
    #
    # Ordering is STRICT: stand aside while any live, fresh, higher-priority
    # run is queued, for as long as that is true. The old one-hour deference
    # budget is gone -- it duplicated the ticket age cap, and it made
    # `seat_priority=9` lose to a priority-0 run that had been queuing an hour,
    # which is the opposite of what the field says.
    waited_since = None
    while True:
        # BLOCKING acquire, not a poll. Deciding before the lock instead cannot
        # work: a low-priority waiter that evaluates the queue before a
        # higher-priority one has written its ticket commits to the kernel queue
        # ahead of it and can no longer reconsider (measured: p1 took the seat
        # before p9). Blocking also restores the zero-latency handoff and the
        # `->` lines in /proc/locks that the console's seat monitor reads.
        fcntl.flock(fd, fcntl.LOCK_EX)
        better = _better_waiters(lock_dir, priority)
        if not better:
            return
        if waited_since is None:
            waited_since = time.time()
        top = max(int(w.get("priority", 0)) for w in better)
        # Say how long this has been going on. A wait with no explanation is
        # indistinguishable from a hang, and this one can now legitimately last
        # as long as the higher-priority queue does.
        print(f"[seat] standing aside for {len(better)} higher-priority waiter(s) "
              f"(top priority {top}); waiting {int(time.time() - waited_since)}s so far",
              flush=True)
        fcntl.flock(fd, fcntl.LOCK_UN)
        time.sleep(SEAT_POLL_S)


def wait_while_paused(poll_s: float = 30.0) -> None:
    """Block while the shared FLOW-3D pause sentinel exists.

    The seat lock is fed by the spool worker, AiiDA and direct runs, so
    `<spool_dir>/PAUSED` is the single switch that holds them all. Mirrors
    `run_rectangular_case.py::wait_while_paused`, duplicated rather than
    imported because this adapter is copied to a scratch dir at submission and
    cannot rely on the runner being importable there.
    """
    spool_dir = os.environ.get("F3D_SPOOL_DIR")
    sentinel = (Path(spool_dir) if spool_dir else Path.home() / ".flow3d" / "spool") / "PAUSED"
    announced = False
    while sentinel.exists():
        if not announced:
            try:
                reason = sentinel.read_text(encoding="utf-8").strip()
            except OSError:
                reason = ""
            print(f"[paused] {sentinel} present; waiting before the solve clock starts. {reason}", flush=True)
            announced = True
        time.sleep(poll_s)
    if announced:
        print("[paused] sentinel cleared; continuing", flush=True)


def build_runner_command(config_path: Path, profile: dict, runner_root: Path) -> tuple[list[str], dict]:
    """Build the run_rectangular_case.py command and the resolved run parameters."""
    case_name = str(profile.get("case_name") or "").strip()
    if not case_name:
        case_name = f"aiida_flow3d_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_root = Path(str(profile.get("out_root") or (runner_root / "runs"))).expanduser().resolve()
    export = str(profile.get("export") or "npz")

    command = [
        sys.executable,
        str(runner_root / RUNNER_RELATIVE),
        str(config_path),
        "--case-name",
        case_name,
        "--out-root",
        str(out_root),
        "--export",
        export,
    ]
    if profile.get("prepare_only"):
        command.append("--prepare-only")
    if profile.get("overwrite"):
        command.append("--overwrite")
    # The runner drops the multi-GB flsgrf once the NPZ is written; only a planned
    # re-export (e.g. the legacy flow3d_to_vtk ingest) needs it kept.
    if profile.get("keep_flsgrf"):
        command.append("--keep-flsgrf")
    for key, flag in (("cores", "--cores"), ("threads_per_rank", "--threads-per-rank")):
        if profile.get(key) is not None:
            command.extend([flag, str(int(profile[key]))])
    for key, flag in (("flow3d_env", "--flow3d-env"), ("post_home", "--post-home")):
        if profile.get(key):
            command.extend([flag, str(profile[key])])

    resolved = {
        "case_name": case_name,
        "case_dir": str(out_root / case_name),
        "prepare_only": bool(profile.get("prepare_only")),
        "export": export,
        # Higher wins the FLOW-3D seat when several runs are queued (see hold_seat).
        "seat_priority": int(profile.get("seat_priority") or 0),
    }
    return command, resolved


# Interpreter-poisoning variables stripped from the runner's environment. A
# daemon launched from a conda shell leaks PYTHONPATH (the conda stdlib +
# site-packages) and LD_LIBRARY_PATH into the job; FLOW-3D POST's embedded
# pvpython then imports the conda vtk bindings against ParaView's native libs
# and aborts (`GetDimensions() takes exactly 1 argument`, then
# `free(): invalid pointer` — alsi_c000, 2026-07-20). The spool path always
# worked because its worker runs from a clean login shell; this makes the
# AiiDA path match it. License variables are never touched.
POISON_ENV_VARS = ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD")


def sanitized_runner_env() -> dict:
    """The inherited environment minus interpreter-poisoning variables.

    Carries `F3D_SEAT_HELD` when set by `hold_seat()`, telling the runner not to
    re-acquire a seat this process already owns (it would block on itself).
    """
    return {key: value for key, value in os.environ.items() if key not in POISON_ENV_VARS}


def build_monitor_args(profile: dict, runner_root: Path, case_dir: Path, case_name: str):
    """Argv factory for the runner repo's stall/divergence monitor sidecar.

    Returns ``None`` when the sidecar should not run: prepare-only jobs,
    explicit ``watchdog: false`` opt-out, or a runner root without
    ``python/monitor_run.py``. The monitor watches the solver's ``hd3msg.<case>``
    (not the block-buffered ``runhyd.log``) and SIGKILLs only the solve subtree
    rooted at the runner PID; it exits on its own when that PID dies. Waiting on
    the machine-wide seat lock produces no progress rows and is not a stall.
    """
    if profile.get("prepare_only") or profile.get("watchdog") is False:
        return None
    monitor_script = runner_root / "python" / "monitor_run.py"
    if not monitor_script.is_file():
        print(
            "[watchdog] WARNING: monitor_run.py missing from the runner root; "
            "the solve runs without the stall/divergence monitor.",
            flush=True,
        )
        return None
    hd3msg = case_dir / f"hd3msg.{case_name}"
    stall_s = float(profile.get("stall_s") or DEFAULT_STALL_S)

    def monitor_args(pid: int) -> list[str]:
        return [
            sys.executable,
            str(monitor_script),
            str(hd3msg),
            "--until-pid",
            str(pid),
            "--stall-s",
            str(stall_s),
        ]

    return monitor_args


def _remove_stale_progress(case_dir: Path, case_name: str) -> None:
    """Drop a previous attempt's solver progress files before a watched solve.

    A leftover ``hd3msg.<case>``/``runhyd.log`` from an earlier attempt anchors
    the monitor's stall baseline on stale progress (the requeued-case false
    abort the batch wrappers guard against the same way).
    """
    for stale in (case_dir / f"hd3msg.{case_name}", case_dir / "runhyd.log"):
        stale.unlink(missing_ok=True)


def _launch_with_watchdog(command: list[str], monitor_args) -> None:
    """Run the solver command with the monitor sidecar attached to its PID.

    The runner's merged output streams through this process's stdout (matching
    ``run_streamed_subprocess``); the monitor inherits stdout so its
    ``[monitor] ABORT`` line lands in the job output. The monitor exits by
    itself when the solve ends, and is reaped here regardless.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=sanitized_runner_env(),
    )
    monitor = subprocess.Popen(monitor_args(process.pid))
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
        return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        monitor.terminate()
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def parse_solver_version(text: str) -> dict:
    """FLOW-3D's own version banner, from `runhyd.log` text.

    Reads the solver's `program version :` line — deliberately NOT the F3D_HOME
    banner a dozen lines above it, which is an install path: machine-specific,
    and nothing outside the machine should record it. The line looks like

        program version :   hydr3d   version     12.3.4.20240101  lnx64  WELD  1.0.0
        version id : double

    Returns {} for a prepare-only run, a run that died before the solver
    started, or any format this does not recognise — an absent version is
    better than a guessed one.
    """
    out: dict = {}
    match = re.search(
        r"^\s*program version\s*:\s*(\S+)\s+version\s+(\S+)\s*(.*)$", text, re.MULTILINE
    )
    if match:
        out["program"] = match.group(1)
        out["version"] = match.group(2)
        build = " ".join(match.group(3).split())
        if build:
            out["build"] = build
    precision = re.search(r"^\s*version id\s*:\s*(\S+)", text, re.MULTILINE)
    if precision:
        out["precision"] = precision.group(1)
    return out


def read_solver_version(case_dir: Path) -> dict:
    """`parse_solver_version` over this case's runhyd.log, {} if unreadable.

    Reads the *whole* log on purpose: only the last 200 lines are retrieved into
    provenance (`runhyd_tail.log`) and the banner sits around line 57, so a long
    solve's version would be truncated away if this waited for the parser.
    """
    log = case_dir / "runhyd.log"
    if not log.is_file():
        return {}
    with contextlib.suppress(OSError):
        return parse_solver_version(log.read_text(encoding="utf-8", errors="replace"))
    return {}


def curate_run_artifacts(case_dir: Path, case_name: str, artifacts_dir: Path) -> list[str]:
    """Copy the small per-run files into the retrieved artifacts directory."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in (*SMALL_ARTIFACT_NAMES, f"prepin.{case_name}"):
        source = case_dir / name
        if source.is_file():
            shutil.copy2(source, artifacts_dir / name)
            copied.append(name)
    runhyd_log = case_dir / "runhyd.log"
    if runhyd_log.is_file():
        tail = runhyd_log.read_text(encoding="utf-8", errors="replace").splitlines()[-RUNHYD_TAIL_LINES:]
        (artifacts_dir / "runhyd_tail.log").write_text("\n".join(tail) + "\n", encoding="utf-8")
        copied.append("runhyd_tail.log")
    return copied


def _write_summary(payload: dict) -> None:
    Path(SUMMARY_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--profile-file", required=True)
    args = parser.parse_args()

    config_path = Path(args.config_file).resolve()
    profile = json.loads(Path(args.profile_file).read_text(encoding="utf-8"))

    runner_root = resolve_runner_root(profile)
    # Before anything is rendered or queued: a runner that cannot honour this
    # config's physics must say so, not silently render something else.
    assert_runner_supports_config(
        runner_root,
        json.loads(config_path.read_text(encoding="utf-8")),
        source=_RESOLVED_FROM.get("label", ""),
    )
    command, resolved = build_runner_command(config_path, profile, runner_root)
    case_dir = Path(resolved["case_dir"])
    # Sit out a machine-wide pause BEFORE the timeout clock starts. The runner
    # waits on the same sentinel itself, but that wait happens inside `timeout`,
    # so parked time burns the solve budget: a 40 h disk-guard pause killed the
    # whole mck_275..mck_294 block on 2026-07-27, each case dying minutes after
    # it finally got to solve. Unbounded on purpose -- a long pause is visible.
    wait_while_paused()

    # Take the seat before the timeout wrapper exists, so a batch submitted all at
    # once serializes here with no clock running instead of every job starting its
    # budget in the queue and the backlog dying. Held until this process exits
    # (one adapter process per CalcJob), and handed to the child so the runner does
    # not block re-acquiring it. Prepare-only never solves, so it takes no seat.
    # `F3D_SEAT_HELD` already means "this process tree owns the seat, do not
    # re-acquire" for the runner (see sanitized_runner_env). Honour it here too:
    # taking the lock a second time from the same process blocks on itself,
    # forever, with no clock and no timeout. That is not hypothetical — it is
    # what made the test suite unfinishable, and it would hit production the
    # first time an adapter ran under a parent that already held the seat.
    if not resolved["prepare_only"] and os.environ.get("F3D_SEAT_HELD") != "1":
        _SEAT.enter_context(hold_seat(int(resolved.get("seat_priority") or 0)))
        os.environ["F3D_SEAT_HELD"] = "1"

    timeout_s = profile.get("timeout_s")
    if timeout_s:
        # Prefixed inside the (optional) systemd scope so the runner is
        # timeout's direct child and TERM/KILL reach the real solve chain. The
        # clock now starts only once this job owns the seat, so it measures the
        # solve, not the queue.
        command = ["timeout", f"--kill-after={WATCHDOG_KILL_GRACE_S}", str(int(timeout_s)), *command]
    command, scope_state = apply_resource_scope(command, profile)
    monitor_args = build_monitor_args(profile, runner_root, case_dir, resolved["case_name"])

    summary = {
        "status": "error",
        "case_name": resolved["case_name"],
        "case_dir": str(case_dir),
        "prepare_only": resolved["prepare_only"],
        "export": resolved["export"],
        "resource_scope": scope_state,
        "npz_file": None,
        "manifest": None,
        "returncode": None,
    }
    meter = RuntimeMeter()
    # Before the solve: HEAD at run END is a different commit (see F5).
    environment = collect_environment()
    try:
        print(f"Launching flow3D_scripts run `{resolved['case_name']}` -> {case_dir}", flush=True)
        # Note: a concurrent solve just blocks on the machine-wide FLOW-3D seat
        # lock (~/.flow3d/spool/.seat.lock) until the seat frees up.
        if monitor_args is None:
            run_streamed_subprocess(command, cwd=Path.cwd(), env=sanitized_runner_env())
        else:
            if case_dir.is_dir():
                _remove_stale_progress(case_dir, resolved["case_name"])
            _launch_with_watchdog(command, monitor_args)
    except subprocess.CalledProcessError as exc:
        summary["returncode"] = exc.returncode
        message = f"flow3D_scripts runner exited with code {exc.returncode}."
        if exc.returncode == 124 and timeout_s:
            message += (
                f" wall timeout exceeded (timeout_s={int(timeout_s)}s). The clock starts "
                "only once this job owns the FLOW-3D seat, so this is solve time, not "
                "queueing."
            )
        elif exc.returncode in (-9, 137) and monitor_args is not None:
            message += (
                " SIGKILL: likely the stall/divergence monitor (see `[monitor] ABORT` "
                "in the job output) or the resource-scope MemoryMax."
            )
        summary["error_message"] = message
        summary["runtime"] = {**meter.stop(), "environment": environment}
        summary["solver"] = read_solver_version(case_dir)  # a failed solve still names its solver
        _write_summary(summary)
        raise
    except Exception as exc:
        summary["error_message"] = str(exc)
        summary["runtime"] = {**meter.stop(), "environment": environment}
        summary["solver"] = read_solver_version(case_dir)  # a failed solve still names its solver
        _write_summary(summary)
        raise

    manifest_path = case_dir / "case_manifest.json"
    if manifest_path.is_file():
        summary["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz_file = case_dir / "npz" / f"{resolved['case_name']}.npz"
    if npz_file.is_file():
        summary["npz_file"] = str(npz_file)

    copied = curate_run_artifacts(case_dir, resolved["case_name"], Path(ARTIFACTS_DIRNAME))
    # Adapter-level wall/CPU (includes seat wait; the solver's own numbers are
    # in manifest.run) plus the software/hardware identity of this job.
    summary["runtime"] = {**meter.stop(), "environment": environment}
    summary["solver"] = read_solver_version(case_dir)
    # Which runner checkout produced this. The summary already records the
    # aiida_pino and lpno shas; without this one a result cannot be
    # tied to the code that rendered its physics, which is how a month-old
    # runner produced pre-calibration data unnoticed.
    summary["runner"] = runner_identity(runner_root, _RESOLVED_FROM.get("label", ""))
    summary.update({"status": "ok", "returncode": 0, "artifact_files": copied})
    _write_summary(summary)


if __name__ == "__main__":
    raise SystemExit(run_with_live_tail(_main))
