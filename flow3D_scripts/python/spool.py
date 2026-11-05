#!/usr/bin/env python3
"""Persistent task spooler for FLOW-3D runs on one machine.

A tiny, dependency-free job queue. Enqueue runs at any time with ``add``; a
single ``start`` worker drains the queue one run at a time. Solver runs that
share a licence are serialised by a lock, so a multi-core run gets the whole
machine and the next one waits its turn.

The queue defaults to a shared per-user directory (``~/.flow3d/spool``, override
with ``F3D_SPOOL_DIR`` or ``--dir``) so every repo on the machine running the
solver feeds one queue and one worker. Both the welding and the cutting repo ship this
same default, so their runs interleave through a single worker. The worker lock
lives in the shared directory, so a second ``start`` -- from either repo --
refuses to run while one is draining; that is what serializes solves across
repos. The queue is just files on disk, so it survives a reboot or a
killed worker: the next ``start`` re-scans ``queue/`` and re-runs anything left
mid-flight.

The spooler only controls *when* runs happen; each FLOW-3D run holds the solver
lock while it solves, so the worker runs them one at a time. There is no cap on
the total number of runs. A run
is enqueued with an absolute
config path and its own repo's runner, so whichever worker drains it runs the
right case no matter which repo it was added from.

Usage:
  spool add <config.json> [run args...]   # enqueue; args pass to the runner
  spool add --cmd "<command>"             # enqueue an arbitrary shell command
  spool start [--drain]                   # run the worker (one run at a time)
  spool status                            # queued / running / recent

Background it with your shell to keep adding jobs while it drains:
  nohup bash bash/spool.sh start >worker.out 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "python" / "run_rectangular_case.py"
SUBDIRS = ("queue", "running", "done", "failed", "logs")


def default_spool_dir() -> Path:
    """Shared queue location so every repo on the machine running the solver feeds one
    worker. The welding and cutting repos ship the same default, so their runs
    serialize through a single worker and a single lock. Override with
    ``F3D_SPOOL_DIR`` or ``--dir``."""
    env = os.environ.get("F3D_SPOOL_DIR")
    return Path(env) if env else Path.home() / ".flow3d" / "spool"


def ensure_dirs(spool_dir: Path) -> None:
    for sub in SUBDIRS:
        (spool_dir / sub).mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def enqueue(spool_dir: Path, name: str, cmd: str) -> Path:
    """Atomically add one task. Time-ordered, unique filename via O_EXCL."""
    ensure_dirs(spool_dir)
    payload = json.dumps({"name": name, "cmd": cmd, "added": _now()})
    ts = time.time_ns()
    while True:
        path = spool_dir / "queue" / f"{ts:020d}-{os.getpid()}.task"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            ts += 1
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    return path


def build_command(config: str, passthrough: list[str]) -> str:
    # Absolute config path so whichever worker drains the shared queue opens the
    # right file regardless of its cwd; the runner path is already absolute.
    # ponytail: only the config positional is absolutized here; --cmd tasks and
    # any paths in passthrough are the caller's responsibility.
    config_abs = Path(config).resolve()
    parts = ["python3", shlex.quote(str(RUNNER)), shlex.quote(str(config_abs))]
    parts += [shlex.quote(arg) for arg in passthrough]
    return " ".join(parts)


def _pending(spool_dir: Path) -> list[Path]:
    return sorted((spool_dir / "queue").glob("*.task"))


def run_worker(spool_dir: Path, drain: bool = False, poll: float = 1.0) -> None:
    """Run queued tasks one at a time until stopped (or queue empty if ``drain``).

    One run at a time: solver runs that share a licence are serialised by a lock,
    so they never overlap. ``drain`` exits once the queue empties; otherwise the worker
    keeps watching for new tasks until a signal stops it.
    """
    ensure_dirs(spool_dir)
    lock_fd = os.open(spool_dir / ".worker.lock", os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"A spooler worker is already running for {spool_dir}")

    # A previous worker died mid-run: re-queue whatever it left in running/.
    for leftover in sorted((spool_dir / "running").glob("*.task")):
        leftover.rename(spool_dir / "queue" / leftover.name)

    current: tuple | None = None  # (proc, task_path, name, logfile)
    stop = {"flag": False}

    def _handle(signum, frame):
        if stop["flag"] and current is not None:  # second signal: kill the in-flight run
            current[0].terminate()
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    poollog = spool_dir / "pool.log"

    while True:
        if current is not None:
            proc, task_path, name, logf = current
            if proc.poll() is not None:
                logf.close()
                status_name = "done" if proc.returncode == 0 else "failed"
                try:
                    task_path.rename(spool_dir / status_name / task_path.name)
                except FileNotFoundError:
                    # The task file vanished mid-run (external cleanup, another
                    # session). Bookkeeping must not kill the worker: on
                    # 2026-07-28 this exact crash stranded 7 queued tasks.
                    status_name = f"{status_name}-vanished"
                with poollog.open("a", encoding="utf-8") as handle:
                    handle.write(f"{_now()}\t{status_name}\t{name}\t{proc.returncode}\n")
                current = None

        if current is None and not stop["flag"] and _pending(spool_dir):
            task_path = _pending(spool_dir)[0]
            running_path = spool_dir / "running" / task_path.name
            task_path.rename(running_path)
            try:
                task = json.loads(running_path.read_text(encoding="utf-8"))
                name, cmd = task["name"], task["cmd"]
            except Exception:  # corrupt task file: shelve it, try the next one now
                running_path.rename(spool_dir / "failed" / running_path.name)
                continue
            logf = (spool_dir / "logs" / f"{name}.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
            current = (proc, running_path, name, logf)

        if current is None and (stop["flag"] or (drain and not _pending(spool_dir))):
            break
        time.sleep(poll)

    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)


def status(spool_dir: Path) -> None:
    ensure_dirs(spool_dir)

    def names(sub: str) -> list[str]:
        out = []
        for path in sorted((spool_dir / sub).glob("*.task")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8"))["name"])
            except Exception:
                out.append(path.name)
        return out

    queued, running = names("queue"), names("running")
    done = len(list((spool_dir / "done").glob("*.task")))
    failed = len(list((spool_dir / "failed").glob("*.task")))
    print(f"queue   {len(queued)}: {', '.join(queued) or '-'}")
    print(f"running {len(running)}: {', '.join(running) or '-'}")
    print(f"done {done}   failed {failed}")
    poollog = spool_dir / "pool.log"
    if poollog.exists():
        tail = poollog.read_text(encoding="utf-8").splitlines()[-5:]
        if tail:
            print("recent:")
            for line in tail:
                print("  " + line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=default_spool_dir(), help="Spool directory (default: ~/.flow3d/spool or $F3D_SPOOL_DIR; shared across repos)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Enqueue a run (config + args) or an arbitrary --cmd")
    p_add.add_argument("--name", help="Label for the task (default: config stem)")
    p_add.add_argument("--cmd", help="Enqueue this exact shell command instead of a config run")
    p_add.add_argument("rest", nargs=argparse.REMAINDER, help="<config.json> [run args...]")

    p_start = sub.add_parser("start", help="Run the worker until stopped (Ctrl-C)")
    p_start.add_argument("--drain", action="store_true", help="Exit when the queue empties instead of watching")
    p_start.add_argument("--poll", type=float, default=1.0, help="Worker poll interval in seconds")

    sub.add_parser("status", help="Show queued / running / recent tasks")
    args = parser.parse_args(argv)

    if args.command == "add":
        if args.cmd:
            name = args.name or "cmd"
            cmd = args.cmd
        else:
            if not args.rest:
                raise SystemExit("add needs a <config.json> (and optional run args) or --cmd")
            config, passthrough = args.rest[0], args.rest[1:]
            name = args.name or Path(config).stem
            cmd = build_command(config, passthrough)
        enqueue(args.dir, name, cmd)
        print(f"queued {name}: {cmd}")
    elif args.command == "start":
        run_worker(args.dir, drain=args.drain, poll=args.poll)
    elif args.command == "status":
        status(args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
