"""System/queue snapshot helpers for the console Monitor tab.

Pure text/procfs parsers kept free of AiiDA imports so they test without a
profile or real hardware; :meth:`AiidaConsoleService.monitor_status` glues
them to process info.
"""

from __future__ import annotations

import json
import os
import fcntl
import re
import subprocess
from pathlib import Path
from typing import Any

_NVSMI_TIMEOUT_S = 5


def _int_or_none(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):  # "[N/A]" on some boards
        return None


def parse_gpu_csv(gpu_txt: str, apps_txt: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu/--query-compute-apps`` csv,noheader,nounits output."""

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.split(",")]

    gpus: list[dict[str, Any]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    for line in gpu_txt.strip().splitlines():
        if not line.strip():
            continue
        idx, uuid, name, util, used, total = cells(line)[:6]
        gpu: dict[str, Any] = {
            "index": _int_or_none(idx),
            "name": name,
            "util_pct": _int_or_none(util),
            "mem_used_mib": _int_or_none(used),
            "mem_total_mib": _int_or_none(total),
            "procs": [],
        }
        gpus.append(gpu)
        by_uuid[uuid] = gpu
    for line in apps_txt.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = cells(line)[:3]
        gpu = by_uuid.get(uuid)
        if gpu is not None:
            gpu["procs"].append({"pid": _int_or_none(pid), "mem_mib": _int_or_none(mem)})
    return gpus


def proc_info(pid: int) -> dict[str, str]:
    """Command line and cwd of a live pid (empty strings when unreadable)."""
    base = Path(f"/proc/{pid}")
    try:
        cmd = base.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = ""
    try:
        cwd = os.readlink(base / "cwd")
    except OSError:
        cwd = ""
    return {"cmd": cmd, "cwd": cwd}


# One argv for the local and the remote query, so a column edit reaches both readers of
# `parse_gpu_csv` (which unpacks positionally); the local path kept literal copies until
# 2026-09-14.
GPU_QUERY_ARGS = [
    "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]
APPS_QUERY_ARGS = ["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]

def gpu_snapshot() -> tuple[list[dict[str, Any]], str | None]:
    """Live GPU utilisation + compute processes, or ([], reason) without nvidia-smi."""

    def query(args: list[str]) -> str:
        return subprocess.run(  # noqa: S603,S607 - fixed binary, fixed args
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=_NVSMI_TIMEOUT_S, check=True
        ).stdout

    try:
        gpu_txt = query(list(GPU_QUERY_ARGS))
        apps_txt = query(list(APPS_QUERY_ARGS))
    except Exception as exc:  # noqa: BLE001 - missing binary, driver error, timeout
        return [], f"nvidia-smi unavailable: {type(exc).__name__}"
    gpus = parse_gpu_csv(gpu_txt, apps_txt)
    for gpu in gpus:
        for proc in gpu["procs"]:
            proc.update(proc_info(proc["pid"]))
    return gpus, None


#: The two queries, shared by the local and remote paths so they cannot drift.


def remote_gpu_snapshot(transport, *, hostname: str) -> tuple[list[dict[str, Any]], str | None]:
    """GPU utilisation on ANOTHER machine, through an open transport.

    The console's GPU panel reads a local `nvidia-smi`, so a job on a compute box
    shows no card at all — the panel describes one machine while the work
    happens on another. Every row is labelled with the host it describes, for
    the same reason a log tail is: a reading with no machine attached invites
    the reader to assume this one.

    Compute processes are deliberately NOT enriched with `proc_info`: that reads
    `/proc` HERE, so a remote pid would be decorated with a local process's
    command line — a wrong answer wearing the shape of a right one.
    """
    try:
        retval, gpu_txt, why = transport.exec_command_wait("nvidia-smi " + " ".join(GPU_QUERY_ARGS))
        if retval:
            # 255 is ssh's own failure exit, so it USUALLY means the box was never
            # reached. Any other code came FROM the box and is a different fact.
            #
            # But `SshCommandRunner._subprocess_run` also FABRICATES 255 for a timeout or
            # an OSError, so this branch covers two very different things and the message
            # used to name only one of them. On 2026-08-31 a remote box vanished from the panel
            # for two days reading "ssh exited 255" while the box answered that exact
            # command in 2.2 s from three other contexts -- the reason was in the
            # discarded stderr, and without it the panel accused a healthy machine and
            # left nothing to diagnose from. Carry it: the string is the only artifact a
            # reader gets.
            detail = " ".join(str(why or "").split())[:160]
            if retval == 255:
                return [], (
                    f"GPUs on {hostname} unreachable: ssh exited 255"
                    + (f" -- {detail}" if detail else " (no stderr; may be a console-side"
                                                       " timeout rather than ssh)")
                )
            return [], (f"nvidia-smi on {hostname} exited {retval}"
                        + (f" -- {detail}" if detail else ""))
        retval, apps_txt, _ = transport.exec_command_wait("nvidia-smi " + " ".join(APPS_QUERY_ARGS))
        # A FAILED process query is not an empty one. Setting `apps_txt = ""`
        # and moving on renders a busy card with no processes, indistinguishable
        # from an idle one. Not observed in production -- it was found by reading
        # this code while chasing a misreading of the payload -- but it is the
        # same class as the `PID -> ?` bug fixed the same day, where a failure
        # rendered as an absence and two readers called a training box idle. The
        # utilisation reading is independent and still good, so the card is kept
        # and its process list is marked UNKNOWN rather than reported as none.
        apps_failed = bool(retval)
        if apps_failed:
            apps_txt = ""
    except Exception as exc:  # noqa: BLE001 - unreachable box must not raise into a request
        return [], f"GPUs on {hostname} unavailable: {type(exc).__name__}"
    gpus = parse_gpu_csv(gpu_txt, apps_txt)
    for gpu in gpus:
        gpu["hostname"] = hostname
        if apps_failed:
            gpu["processes_unknown"] = True
    if apps_failed:
        return gpus, (
            f"process list on {hostname} unavailable (nvidia-smi compute-apps exited "
            f"{retval}); utilisation is still accurate"
        )
    _identify_remote_procs(transport, gpus)
    return gpus, None


def _identify_remote_procs(transport, gpus: list[dict[str, Any]]) -> None:
    """Name each GPU process by asking the box that owns the pid.

    Without this a busy remote card shows `PID 169217 · WHAT: ?`, and beside the
    local cards — which name their run — the `?` reads as "nothing running".
    That is not hypothetical: two readers concluded a box was free while a
    32k-step training on it was at step 7100.

    The lookup goes over the SAME transport, never to local `/proc`: a remote
    pid resolved here would be decorated with whatever this machine happens to
    be running under that number, which is a wrong answer wearing the shape of a
    right one. A failed lookup leaves the fields empty, degrading to "unknown"
    rather than to "somebody else's process".
    """
    pids = [p["pid"] for gpu in gpus for p in gpu.get("procs") or [] if p.get("pid")]
    if not pids:
        return
    listed = " ".join(str(int(pid)) for pid in pids)
    script = (
        f"for p in {listed}; do "
        "echo \"$p|$(tr '\\0' ' ' </proc/$p/cmdline 2>/dev/null)\"; "
        "echo \"$p|$(readlink /proc/$p/cwd 2>/dev/null)\"; "
        "done"
    )
    try:
        retval, out, _ = transport.exec_command_wait(script)
    except Exception:  # noqa: BLE001 - unknown is the correct degradation
        return
    if retval:
        return
    cmds: dict[int, str] = {}
    cwds: dict[int, str] = {}
    for line in (out or "").splitlines():
        pid_txt, sep, value = line.partition("|")
        if not sep or not pid_txt.strip().isdigit():
            continue
        pid = int(pid_txt.strip())
        value = value.strip()
        if not value:
            continue
        # cmdline first, cwd second, per the loop above.
        (cwds if pid in cmds else cmds)[pid] = value
    for gpu in gpus:
        for proc in gpu.get("procs") or []:
            pid = proc.get("pid")
            proc.setdefault("cmd", cmds.get(pid, ""))
            proc.setdefault("cwd", cwds.get(pid, ""))


def parse_meminfo(text: str) -> dict[str, float]:
    kb: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"(MemTotal|MemAvailable):\s+(\d+)\s*kB", line)
        if m:
            kb[m.group(1)] = int(m.group(2))
    out: dict[str, float] = {}
    if "MemTotal" in kb:
        out["mem_total_gib"] = round(kb["MemTotal"] / 1048576, 1)
    if "MemAvailable" in kb:
        out["mem_available_gib"] = round(kb["MemAvailable"] / 1048576, 1)
    return out


def cpu_snapshot() -> dict[str, Any]:
    load1, load5, load15 = os.getloadavg()
    out: dict[str, Any] = {
        "load_1m": round(load1, 2),
        "load_5m": round(load5, 2),
        "load_15m": round(load15, 2),
        "cores": os.cpu_count(),
    }
    try:
        out.update(parse_meminfo(Path("/proc/meminfo").read_text()))
    except OSError:  # pragma: no cover - /proc always exists on linux
        pass
    return out


def task_names(subdir: Path) -> list[str]:
    """Task names in one spool subdir (flow3D and gpu spool share the format)."""
    names: list[str] = []
    for task in sorted(subdir.glob("*.task")):
        try:
            names.append(json.loads(task.read_text()).get("name") or task.stem)
        except Exception:  # noqa: BLE001 - unreadable task file still counts
            names.append(task.stem)
    return names


def gpu_spool_snapshot(root: Path | None = None) -> list[dict[str, Any]]:
    """Queued/running training tasks per GPU in the pino gpu spool."""
    root = root or Path.home() / ".pino" / "spool"
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.glob("gpu*")):
        if not d.is_dir():
            continue
        queued = task_names(d / "queue")
        out.append({"gpu": d.name, "running": task_names(d / "running"), "queued_count": len(queued), "queued": queued[:10]})
    return out


def _lease_root() -> Path:
    """gpu_spool's shared VRAM ledger (same env override it honours)."""
    env = os.environ.get("PINO_GPU_LEASE_DIR")
    return Path(env) if env else Path("/var/tmp/pino_gpu_leases")


def _lease_is_live(path: Path, payload: dict[str, Any]) -> bool:
    """Whether a ledger entry still has an owner.

    gpu_spool decides this by flock -- the kernel drops the lock when the owning
    fd closes, crash or not -- so an entry we can lock exclusively is abandoned.
    A dead pid is stale too, which covers a lease written by a process that has
    since exited without the file being cleaned up.
    """
    pid = payload.get("pid")
    if isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False  # owner is gone: stale whatever the lock says
        except PermissionError:
            return True  # another user's live process
        except OSError:
            pass
        else:
            return True  # owner alive; do not under-report a held card
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return True  # unreadable: assume live rather than under-report load
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True  # someone holds it
    except OSError:
        return True
    else:
        return False
    finally:
        os.close(fd)


def gpu_seat_snapshot(root: Path | None = None) -> list[dict[str, Any]]:
    """Seat holders per GPU, read from gpu_spool's lease ledger.

    `gpu_spool_snapshot` only sees the spooler worker's own queue. An
    AiiDA training calls `gpu_spool seat`, which takes a lease instead, so a
    card fully booked by trainings reported "idle, 0 queued" (observed
    2026-07-23 with three holders on 21 GB and two jobs waiting for a seat).
    """
    root = root or _lease_root()
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.glob("gpu*")):
        if not d.is_dir():
            continue
        def _read(suffix: str) -> list[dict[str, Any]]:
            found: list[dict[str, Any]] = []
            for path in sorted(d.glob(f"*{suffix}")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - a torn write must not break the panel
                    continue
                if not _lease_is_live(path, payload):
                    continue
                found.append(
                    {
                        "name": str(payload.get("name", "?")),
                        "vram_mb": int(payload.get("vram_mb", 0) or 0),
                        "ram_mb": int(payload.get("ram_mb", 0) or 0),
                        "user": str(payload.get("user", "")),
                        "since": str(payload.get("since", "")),
                        "exclusive": bool(payload.get("exclusive", False)),
                    }
                )
            return found

        seats = _read(".lease")
        # gpu_spool publishes a `.waiting` entry while a seat request is blocked;
        # without it a card with pending work reads as idle (2026-07-23).
        waiting = _read(".waiting")
        out.append(
            {
                "gpu": d.name,
                "seats": seats,
                "seat_count": len(seats),
                "seat_vram_mb": sum(s["vram_mb"] for s in seats),
                "waiting": waiting,
                "waiting_count": len(waiting),
                "waiting_vram_mb": sum(w["vram_mb"] for w in waiting),
            }
        )
    return out


def attach_seats_to_spool(
    rows: list[dict[str, Any]], root: Path | None = None
) -> list[dict[str, Any]]:
    """Merge seat-holder info into the spool rows and set an honest `busy` flag.

    A card is busy if the spooler is running something OR anything holds a seat
    on it. Without this the panel calls a fully-booked card idle.
    """
    by_gpu = {row["gpu"]: row for row in gpu_seat_snapshot(root)}
    for row in rows:
        seat = by_gpu.get(row.get("gpu"), {})
        row["seats"] = seat.get("seats", [])
        row["seat_count"] = seat.get("seat_count", 0)
        row["seat_vram_mb"] = seat.get("seat_vram_mb", 0)
        row["waiting"] = seat.get("waiting", [])
        row["waiting_count"] = seat.get("waiting_count", 0)
        row["waiting_vram_mb"] = seat.get("waiting_vram_mb", 0)
        row["busy"] = bool(row.get("running")) or row["seat_count"] > 0
        # Anything at all on this card: running, holding a seat, blocked on one,
        # or sitting in the spool queue. `busy` alone called a card with only
        # blocked work idle, which is how two trainings went unseen for hours.
        row["pending"] = (
            row["busy"] or row["waiting_count"] > 0 or int(row.get("queued_count", 0)) > 0
        )
    return rows


def attach_aiida_to_spool(
    rows: list[dict[str, Any]], gpus: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Tag each gpu_spool row with the AiiDA compute processes on that card.

    The spool queue is per-GPU and only holds work someone queued there, so a
    card running an AiiDA training reads "idle, 0 queued" without this. Counts
    the procs `monitor_status` already matched to a run (`run_pk`).
    """
    per_card = {
        f"gpu{gpu.get('index')}": sum(1 for p in gpu.get("procs", []) if p.get("run_pk"))
        for gpu in gpus
    }
    for row in rows:
        row["aiida_procs"] = per_card.get(row.get("gpu"), 0)
    return rows


# ponytail: a batch is the task name minus its trailing case number — matches
# spool names like alsi_ctrl_alsi_c060 / ti64_rk150_case_007; revisit if a
# batch ever encodes the case some other way.
_BATCH_SUFFIX = re.compile(r"[_-]?(?:c(?:ase)?[_-]?)?\d+$")


def batch_key(name: str) -> str:
    return _BATCH_SUFFIX.sub("", name) or name


def group_batches(names: list[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for name in names:
        key = batch_key(name)
        counts[key] = counts.get(key, 0) + 1
    return [{"batch": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


_STEP_RE = re.compile(r"\[step:\s*(\d+)\]")


def last_step(tail_text: str) -> int | None:
    """Highest training step mentioned in a live-tail excerpt."""
    matches = _STEP_RE.findall(tail_text)
    return int(matches[-1]) if matches else None


_BEAMBOX_RE = re.compile(r"beambox_mae_T=([0-9.eE+-]+)\s+at step\s+(\d+)")


def last_beambox_mae(tail_text: str) -> dict[str, Any] | None:
    """Most recent beam-box MAE in a live-tail excerpt, with the step it belongs to.

    The trainer prints `best checkpoint: beambox_mae_T=... at step N` to the same log the
    step counter comes from, so the monitor can show whether a run is advancing WELL rather
    than only that it is advancing.

    Returns the LAST reading, not the lowest: those agree while the model improves and part
    company the moment it stops, and a stale best shown as the current number makes a
    plateau look like progress.

    The value is NORMALISED, exactly as logged. Converting it is the caller's job, with the
    factor recorded in the model's own config -- this repo's rule is to report Kelvin and
    never derive the factor, so a raw value is returned rather than a guessed one.
    """
    matches = _BEAMBOX_RE.findall(tail_text or "")
    if not matches:
        return None
    value, step = matches[-1]
    try:
        return {"value": float(value), "step": int(step), "normalised": True}
    except ValueError:
        return None


def waiting_for_gpu(tail_text: str) -> bool:
    last = tail_text.strip().splitlines()[-1:]
    return bool(last) and ("[gpu-spool] waiting" in last[0] or "[gpu-seat] waiting" in last[0])


def tail_file(path: Path, max_bytes: int = 16384) -> str:
    """Last chunk of a local file, empty when missing."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def parse_proc_locks(text: str, inode: int) -> tuple[int | None, list[int]]:
    """(holder pid, waiter pids) for one file, from /proc/locks.

    A line whose second field is ``->`` is a process *blocked* on the lock, not
    one holding it::

        38: FLOCK  ADVISORY  WRITE 660466 103:02:73800397 0 EOF
        38: -> FLOCK  ADVISORY  WRITE 675609 103:02:73800397 0 EOF

    The inode is the last colon-separated part of field 6, in decimal.
    """
    holder: int | None = None
    waiters: list[int] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        blocked = parts[1] == "->"
        fields = parts[2:] if blocked else parts[1:]
        if len(fields) < 5 or fields[0] != "FLOCK":
            continue
        try:
            pid = int(fields[3])
            if int(fields[4].split(":")[-1]) != inode:
                continue
        except ValueError:
            continue
        if blocked:
            waiters.append(pid)
        elif holder is None:
            holder = pid
    return holder, waiters


def seat_holder(spool: Path) -> dict[str, Any] | None:
    """Who actually holds the FLOW-3D seat lock, and what they are working in.

    The spool's ``running/`` directory says which task the *worker started*, not
    which one is on the seat: on 2026-08-04 it named a spool task queued since
    08-02 while the flock was held by an AiiDA `flow3d_run_adapter.py`, with the
    spool task and two more AiiDA runs blocked behind it. Only the kernel knows,
    so ask it — the working directory identifies the calcjob.
    """
    lock = spool / ".seat.lock"
    try:
        inode = lock.stat().st_ino
        holder, waiters = parse_proc_locks(Path("/proc/locks").read_text(), inode)
    except OSError:
        return None
    if holder is None:
        return None
    out: dict[str, Any] = {"pid": holder, "waiting_pids": waiters}
    try:
        out["workdir"] = os.readlink(f"/proc/{holder}/cwd")
    except OSError:
        out["workdir"] = None
    return out


def gpu_rows_for_computer(label, *, load_computer, snapshot=None, log=None, runner=None):
    """GPU rows for one remote computer, or `(rows, note)` explaining why not.

    The computer is resolved BY LABEL on every call rather than captured from a
    caller's listing. On 2026-08-21 the Monitor tab reported a demonstrably
    healthy compute box as "unavailable: DetachedInstanceError":
    the caller had built a closure holding a live `orm.Computer`, and in a
    long-lived console the ORM session behind it expires between the listing and
    `get_transport()` -- which queries authinfo, so it is the call that notices.
    Re-loading is nearly free and removes the class.

    The note distinguishes THREE outcomes, because they call for opposite
    reactions from whoever reads the panel and one string used to cover all of
    them:

    * the console could not resolve or use the computer object -- a fault HERE,
      and the box has not been contacted at all. Must never read as "down".
    * the box could not be reached -- a real connectivity or host problem.
    * the box answered and its reply could not be read -- reached, so
      connectivity is fine and the parsing or `nvidia-smi` is at fault.

    `log` receives the full traceback. The panel string stays short, but
    discarding the traceback is why the original failure could not be located
    from inside the process.
    """
    import traceback

    snapshot = snapshot or remote_gpu_snapshot

    def _record(exc):
        if log:
            log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    try:
        computer = load_computer(label)
        host = ssh_host_for(computer, log=log)
    except Exception as exc:  # noqa: BLE001
        _record(exc)
        return [], (
            f"GPUs on {label}: console could not resolve the host "
            f"({type(exc).__name__}); the box was not contacted"
        )
    if not host:
        return [], f"GPUs on {label}: no ssh host recorded for this computer"

    try:
        return snapshot((runner or SshCommandRunner)(host), hostname=label)
    except Exception as exc:  # noqa: BLE001
        _record(exc)
        return [], f"GPUs on {label}: could not read its cards ({type(exc).__name__})"


class SshCommandRunner:
    """`exec_command_wait` over a plain `ssh` subprocess.

    A drop-in for the transport interface `remote_gpu_snapshot` uses, so the
    parsing and every test above it are untouched.

    Why not an AiiDA transport: `core.ssh_async` bridges sync to async through
    greenback, and `run_command_blocking` asks `is_running()` of AiiDA's
    PROCESS-GLOBAL runner loop. Whenever anything else in the process has that
    loop spinning, opening a transport raises and the panel loses the box --
    measured at 4 failures in 23 polls (~17 %), clustered when the process is
    busy, which is exactly when someone is reading the panel. Changing threads
    cannot escape a process-global object.

    This panel runs two fixed read-only commands and parses CSV. It needs a
    pipe, not a transport abstraction, and a subprocess has no runner loop and
    no bridge, so that failure becomes structurally impossible rather than
    rarer.

    The host is supplied by the caller from authinfo at runtime -- never a
    literal here. `BatchMode=yes` matters: a console with no tty must fail
    rather than block on an auth prompt. No ssh config path is named; ssh's own
    default lookup already resolves the alias authinfo stores.
    """

    #: Read-only, short, and bounded, because a hung poll is worse than a gap.
    CONNECT_TIMEOUT_S = 15
    COMMAND_TIMEOUT_S = 60

    def __init__(self, host: str, *, run=None):
        self._host = str(host)
        self._run = run or self._subprocess_run

    def _subprocess_run(self, argv):
        import subprocess

        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.COMMAND_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001 - a timeout must read as a failure
            # Reported as 255 so callers need no second failure convention, but the
            # reason travels in stderr: this is NOT ssh's 255 and a reader who cannot
            # tell them apart will blame the network for a console-side timeout.
            return 255, "", f"console-side {type(exc).__name__}: {exc}"
        return done.returncode, done.stdout, done.stderr

    def exec_command_wait(self, command: str):
        argv = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.CONNECT_TIMEOUT_S}",
            self._host,
            command,
        ]
        return self._run(argv)


def ssh_host_for(computer, *, log=None) -> str:
    """The host `ssh` should be given for this computer, from its authinfo.

    Read at runtime so no machine name is ever committed. Falls back to the
    computer's own hostname and then its label, which is what the panel labels
    rows with anyway.

    The fallback is not equivalent: the authinfo host is what the user's ssh
    config knows (a `Host` alias carrying the key), the computer's hostname is
    often a bare IP that matches no alias, so the same box then answers
    "Permission denied (publickey)". A lookup failure is therefore reported
    through `log` (the full traceback) rather than swallowed -- on 2026-09-14 the
    console fell back silently and the panel blamed the box.
    """
    import traceback

    try:
        from aiida.orm import User

        params = computer.get_authinfo(User.collection.get_default()).get_auth_params()
        host = str(params.get("host") or "").strip()
        if host:
            return host
    except Exception as exc:  # noqa: BLE001 - fall through to the plainer answers, but say so
        if log:
            log("ssh_host_for: authinfo lookup failed, falling back to the computer hostname\n"
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return str(getattr(computer, "hostname", "") or getattr(computer, "label", "") or "")

