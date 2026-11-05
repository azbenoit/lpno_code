#!/usr/bin/env python3
"""Per-GPU training spooler for lpno on one workstation.

A tiny, dependency-free job queue, one queue and one worker *per GPU*. Multiple
different processes enqueue training runs at any time with ``add``; a
single worker per GPU drains that GPU's queue one run at a time. The queue is
just files on disk (default ``~/.pino/spool``, override with ``PINO_SPOOL_DIR``
or ``--dir``), so it is shared across sessions and survives a reboot or a killed
worker: the next ``start`` re-scans ``running/`` and re-runs anything left
mid-flight.

Why declared memory and not a utilisation check: a GPU's utilisation dips to
near-zero between steps of a running job, and PyTorch's caching allocator hands
blocks back and forth all through training, so neither "GPU looks idle" nor
"memory.used is low right now" means "the card is free". Nothing here ever
samples live utilisation, and it never reads live memory as evidence of *room*.
Admission is a *ledger*: every run declares the VRAM it needs, holds that
reservation for its whole lifetime, and a new run starts only if the declared
reservations still fit the card's budget. Free memory is read for one purpose
only — refusing a reservation the ledger would have allowed, when something
outside the ledger (a notebook, another user's process) is holding the card.

- ``add`` routes a task to a specific GPU (``--gpu N``) or the least-loaded queue
  (``--gpu auto``, the default). The queue you land in *is* the GPU assignment;
  the worker pins it with ``CUDA_VISIBLE_DEVICES`` (which ``cuda_bootstrap``
  preserves), so a task must not carry its own ``cuda_device=`` override.
- ``--vram-mb N`` declares a run's reservation. Without it a run reserves the
  whole card (exclusive, the old one-at-a-time behaviour); set
  ``PINO_SPOOL_DEFAULT_VRAM_MB`` for a house default. Two 8 GB runs pack onto one
  24 GB card; a 20 GB run still gets it to itself.
- ``start [--gpu N]`` runs the worker. With no ``--gpu`` it launches one worker
  per managed GPU. A second worker for the same GPU refuses to start.
- ``--ram-mb N`` books *host* RAM as well, machine-wide across both cards. GPU
  memory is not the only way a run dies: a 25-step CI smoke that peaks at 86 GB
  RSS is SIGKILLed when a 27 GB training already holds the box, and no VRAM
  budget would have caught it. A run that declares nothing books
  ``PINO_SPOOL_DEFAULT_RAM_MB`` (32 GB) rather than nothing, so it is still
  visible to the next admission; ``--ram-mb 0`` opts a small run out.
- ``seat --gpu N|auto [--vram-mb N] [--ram-mb N] [--priority] -- <cmd>`` runs a command
  directly under a reservation, so a manual run outside the spool still books its
  share of the box. ``--priority`` (used by CI) blocks the workers from starting
  anything new, so it goes in ahead of the queue instead of behind it.

The reservation ledger lives in a shared, world-writable directory (default
``/var/tmp/pino_gpu_leases``, override ``PINO_GPU_LEASE_DIR``) so runs by
*different users* on this box — the developer and a CI runner — book
against the same budget. The task queue stays private per user (``~/.pino/spool``):
the ledger only carries reservations, never commands, so a shared directory
cannot be used to get a job executed as someone else.

Usage:
  gpu_spool add [--gpu N|auto] [--name X] [--module M] [--vram-mb N] [--ram-mb N] <overrides...>
  gpu_spool add --cmd "<command>" [--gpu N|auto] [--vram-mb N] [--ram-mb N]
  gpu_spool start [--gpu N] [--drain]
  gpu_spool status
  gpu_spool seat --gpu N|auto [--vram-mb N] [--ram-mb N] [--priority] -- <command>

Background it to keep adding while it drains:
  nohup python -m lpno.gpu_spool start >spool.out 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import getpass
import json
import os
import re
import shlex
import sys
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence

# repo root: src/lpno/gpu_spool.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULE = "NS_Energy_laser_PINO"
SUBDIRS = ("queue", "running", "done", "failed", "logs")
# The spool owns GPU assignment; a task carrying its own selector would fight it.
_GPU_OVERRIDE = re.compile(r"(?:^|[\s'\"])\+?(?:custom\.)?cuda_(?:device|visible_devices)\s*=")


def default_spool_dir() -> Path:
    env = os.environ.get("PINO_SPOOL_DIR")
    return Path(env) if env else Path.home() / ".pino" / "spool"


def lease_root() -> Path:
    """Shared VRAM-reservation ledger, readable and writable by every user on the
    box (the developer and the CI runner book against one budget)."""
    env = os.environ.get("PINO_GPU_LEASE_DIR")
    return Path(env) if env else Path("/var/tmp/pino_gpu_leases")


def lease_dir(gpu: int) -> Path:
    """Per-GPU ledger directory, created 1777 (like /tmp): any user may add a
    reservation, none may delete another's."""
    path = lease_root() / f"gpu{gpu}"
    path.mkdir(parents=True, exist_ok=True)
    for part in (lease_root(), path):
        try:
            os.chmod(part, 0o1777)  # umask strips the bits mkdir() was given
        except OSError:
            pass  # someone else owns it and already set it up
    return path


def _gpu_total_mb(gpu: int) -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
             "-i", str(gpu)],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return int(float(out.splitlines()[0]))
    except Exception:
        return 24576  # this workstation's TITAN RTXs; only affects packing maths


# One switch for every hardware-dependent admission gate (VRAM ledger, host RAM, CPU
# headroom, the --ram-mb > MemTotal refusal). "enforce" refuses or waits as this workstation
# needs; "warn" prints and admits, which is what the public copy ships (2026-09-14:
# memory limits are hardware dependent, so a guard tuned to one box is a warning elsewhere).
# The public assembler rewrites this ONE literal; `PINO_RESOURCE_GUARDS` overrides it either way.
RESOURCE_GUARDS_DEFAULT = "warn"


def resource_guard(condition: bool, message: str) -> bool:
    """True when ``condition`` should block admission. Under ``warn`` a true condition
    prints ``[resource-guard] WARNING`` and returns False, so the caller proceeds."""
    if not condition:
        return False
    if os.environ.get("PINO_RESOURCE_GUARDS", RESOURCE_GUARDS_DEFAULT).strip().lower() == "warn":
        print(f"[resource-guard] WARNING {message} -- not enforced (PINO_RESOURCE_GUARDS=warn)",
              flush=True)
        return False
    return True


def vram_budget_mb(gpu: int) -> int:
    """VRAM the spooler is willing to hand out on this card, in MiB.

    Deliberately below the physical total: the headroom absorbs the allocator's
    fragmentation and whatever sits on the card outside the ledger (a display, a
    stray notebook). Conservative on purpose — an over-tight budget costs
    parallelism, an over-loose one costs an OOM in someone else's run.
    """
    env = os.environ.get("PINO_SPOOL_VRAM_MB")
    if env:
        return max(1, int(env))
    headroom = int(os.environ.get("PINO_SPOOL_VRAM_HEADROOM_MB", "2048"))
    return max(1, _gpu_total_mb(gpu) - headroom)


def free_mb(gpu: int) -> int | None:
    """What the driver reports free on the card right now, or ``None`` if it can't
    be read.

    Used only to *refuse* a reservation the ledger would have allowed, never to
    grant one. A training that has handed blocks back to the caching allocator
    looks free-er than it really is, so live memory is untrustworthy as evidence
    of room — but it is solid evidence of the opposite, and it is the only way to
    see memory the ledger doesn't know about (a notebook, another user's job).
    """
    if os.environ.get("PINO_SPOOL_CHECK_FREE", "1") == "0":
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu)],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return int(float(out.splitlines()[0]))
    except Exception:
        return None


def default_vram_mb(gpu: int) -> int | None:
    """Reservation for a run that declares none. ``None`` means *exclusive*: the
    whole card, the old one-at-a-time behaviour. ``PINO_SPOOL_DEFAULT_VRAM_MB``
    replaces that with a house size, so undeclared runs pack too."""
    env = os.environ.get("PINO_SPOOL_DEFAULT_VRAM_MB")
    return max(1, int(env)) if env else None


def _load_1min() -> float | None:
    """1-minute load average, or None where the platform cannot report one."""
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


def _cpu_count() -> int | None:
    return os.cpu_count()


def min_free_cores() -> int:
    """Cores admission leaves for the OS and interactive login. 0 disables the gate.

    Default 4: on a shared machine, leave at least 4 cores (and about 20 GB of RAM) for
    the OS and interactive logins. Override with PINO_SPOOL_MIN_FREE_CORES.
    """
    try:
        return max(0, int(os.environ.get("PINO_SPOOL_MIN_FREE_CORES", "4")))
    except ValueError:
        return 4


_UNSET = object()


def cpu_headroom_ok(load=_UNSET, cores=_UNSET) -> bool:
    """True when admitting one more run still leaves `min_free_cores()` free.

    The memory ledgers never covered CPU: concurrency stayed safe only because an
    over-booked host-RAM ledger capped it as a side effect. Right-sizing those bookings
    (2026-08-20) removed that accident -- measured the same evening, a FLOW-3D solve at
    9.9 cores plus four trainings at ~1.05 each reached 14.1 of 20 with the queue still
    admitting. Refusal here is a poll-loop delay, not a failure: `seat` retries, so a busy
    box postpones a run rather than killing it.

    Unknown load or core count admits, so a platform that cannot report either never
    becomes permanently unschedulable.
    """
    reserve = min_free_cores()
    if reserve <= 0:
        return True
    # `_UNSET` means "measure it"; an explicit None means "this box cannot report it".
    if load is _UNSET:
        load = _load_1min()
    if cores is _UNSET:
        cores = _cpu_count()
    if load is None or not cores:
        return True
    return (cores - load) >= reserve


def ram_available_mb() -> int | None:
    """``MemAvailable`` in MiB — what the kernel thinks can be used without
    swapping. Unlike GPU VRAM this *is* a fair reading: it already discounts the
    page cache and every process on the box, including ones the ledger never
    hears about (an AiiDA training, another user's job)."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def ram_headroom_mb() -> int:
    return int(os.environ.get("PINO_SPOOL_RAM_HEADROOM_MB", "8192"))


def ram_total_mb() -> int | None:
    """``MemTotal`` in MiB -- the machine's hard ceiling.

    Distinct from ``ram_available_mb``: available fluctuates, so waiting on it is
    legitimate (another run finishing frees it). Total does not, so a booking
    above it can never be satisfied by anything and is a deadlock, not a queue.
    """
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def default_ram_mb() -> int:
    """Host RAM booked by a run that declares none. VRAM has *exclusive* to fall
    back on; RAM has no equivalent (booking the whole box would serialise
    everything), and booking zero left an undeclared run invisible to the next
    admission -- which is how a 27 GB training and an 86 GiB CI stage ended up
    sharing 125 GB. Sized above the largest training measured here; re-measure
    with ``/usr/bin/time -v`` if they grow. ``--ram-mb 0`` opts out."""
    return int(os.environ.get("PINO_SPOOL_DEFAULT_RAM_MB", "32768"))


def memory_scope_prefix(declared_ram_mb: int | None) -> list[str]:
    """argv prefix putting a run in a memory-capped systemd user scope, or [].

    BUG-43: the booking is a promise, and nothing capped the process. A run that
    grew past its declaration took the RAM anyway, and the kernel's OOM killer
    picks the largest RSS rather than whoever overran -- so the victim was
    usually the *honest* big job (the 86 GiB CI stage), killed by a SIGKILL that
    leaves no traceback. A cgroup cap makes the over-running run die as itself.

    DEFAULT OFF since 2026-08-06 (f90778e): opt in with PINO_SPOOL_ENFORCE_RAM=1.
    When enabled it applies only to a run that *declared* its RAM. The house
    default is a guess, and capping a training at a guessed number kills it at
    hour three; a number someone typed came from a measurement. The ceiling is
    50% above the declaration, so a run that uses every MiB it asked for is
    untouched and only a real runaway dies. A box without a systemd user manager
    runs uncapped rather than not at all. (The 2026-08-20 spool audit found nine
    doc sites still describing the cap as default-on; reasoning about
    under-declaration risk from those was reasoning wrong.)

    Deliberately no MemoryHigh: measured here, a scope that reaches MemoryHigh
    with unreclaimable anon memory is throttled into a crawl rather than killed,
    which wedges the run and holds the seat. MemoryMax is the honest signal.
    """

    # Default OFF as of 2026-08-06, and the reason is measured, twice, in this
    # workspace. cgroup v2 charges the PAGE CACHE of files a job reads to the job's
    # cgroup, so MemoryMax is the wrong tool for IO-heavy work: flow3D_scripts
    # documented it first ("a MemoryMax charges FLOW-3D's ~13 GB flsgrf write-back
    # page cache to the cgroup and OOM-kills healthy long solves"), and on 2026-08-06
    # a training that streams ~42 GB of NPZ per epoch died at step ~900 -- healthy,
    # 440-670 ms/iter, loss falling -- when its step-1000 validation pass pushed
    # parent RSS + worker RSS + charged page cache over the 1.5x ceiling. The kill
    # surfaces as the child's SIGTERM masked to exit 241, naming nothing.
    #
    # BUG-43's need (an overrunning run should die as itself, not take the honest big
    # job with it) is now served by measurement instead of enforcement: the train
    # adapter records peak RSS against the declaration in train_summary.json and
    # warns on overrun, and admission still books the declared figure. Set
    # PINO_SPOOL_ENFORCE_RAM=1 to re-enable the hard cap for a workload whose page
    # cache is known-small.
    if not declared_ram_mb or os.environ.get("PINO_SPOOL_ENFORCE_RAM", "0") == "0":
        return []
    if shutil.which("systemd-run") is None:
        return []
    declared = int(declared_ram_mb)
    return [
        "systemd-run", "--user", "--scope", "-q", "--collect",
        "-p", f"MemoryMax={int(declared * 1.5)}M",
        # Without this the cap does not bite: memory.max reclaims to swap before
        # it OOM-kills, so an overrunning run quietly eats the shared 8 GB of
        # swap instead of dying -- which is the thrashing that locked SSH out of
        # this box in the first place. Measured: a 2 GB toy under MemoryMax=300M
        # ran to completion until swap was capped, then died at the ceiling.
        "-p", "MemorySwapMax=0",
        "--",
    ]


def booked_ram_mb() -> int:
    """Host RAM booked by live runs on *every* card — RAM is machine-wide."""
    total = 0
    for gdir in sorted(lease_root().glob("gpu*")):
        try:
            gpu = int(gdir.name[3:])
        except ValueError:
            continue
        total += sum(int(e.get("ram_mb", 0) or 0) for e in _entries(gpu, ".lease"))
    return total


def _is_stale(path: Path) -> bool:
    """True if nobody holds this ledger entry any more, i.e. its owner exited (the
    kernel drops an ``flock`` when the fd closes, crash or not). Opens read-only so
    another user's 0644 entry can still be probed."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False  # unreadable: assume live rather than hand out its memory
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


def _entries(gpu: int, suffix: str, prune: bool = False) -> list[dict]:
    """Live ledger entries of one kind. Stale ones are never counted, and are
    deleted only when ``prune`` — i.e. only from inside the admit lock, so a
    read-only caller like ``status`` cannot unlink an entry another process is
    at that moment creating. Under the 1777 sticky bit another user's file is
    not ours to unlink anyway; an unheld entry just stops counting."""
    out = []
    for path in sorted(lease_dir(gpu).glob(f"*{suffix}")):
        if _is_stale(path):
            if prune:
                try:
                    path.unlink()
                except OSError:
                    pass
            continue
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            out.append({"name": path.name, "vram_mb": 0})
    return out


def waiting_entries(gpu: int, prune: bool = False) -> list[dict]:
    """Runs currently blocked waiting for a seat on ``gpu``.

    A blocked run is otherwise invisible: it never entered the spooler queue, it
    holds no `.lease`, and it has no GPU process yet because it blocks before
    CUDA init. Two AiiDA trainings sat unseen for hours that way on 2026-07-23
    while the console reported both cards idle.

    `.waiting` is deliberately inert -- admission reads `.lease` (budget) and
    `.want` (priority yielding) only. Consulting it here would hand every
    ordinary waiter the priority semantics reserved for ``--priority``.
    """
    return _entries(gpu, ".waiting", prune=prune)


def reserved_mb(gpu: int, prune: bool = False) -> int:
    return sum(int(e.get("vram_mb", 0)) for e in _entries(gpu, ".lease", prune=prune))


def _hold(path: Path, payload: dict) -> int:
    """Create a ledger entry and hold its lock; the fd must stay open for as long
    as the reservation is meant to last."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.write(fd, (json.dumps(payload) + "\n").encode("utf-8"))
    return fd


def _drop(fd: int, path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
    _release_lock(fd)


def _entry_path(gpu: int, suffix: str) -> Path:
    return lease_dir(gpu) / f"{os.getpid()}-{time.time_ns()}{suffix}"


def _admit_lock(gpu: int) -> int:
    """Serialise read-then-reserve across processes *and users*, so two admissions
    cannot both see the same free memory. 0666: every user must be able to take it."""
    path = lease_dir(gpu) / ".admit.lock"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o666)
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _legacy_seat_held(spool_dir: Path, gpu: int) -> bool:
    """A pre-ledger ``seat`` process (older code, still running) holds the whole
    card. ponytail: drop this once no long run predates the ledger."""
    seat = gpu_dir(spool_dir, gpu) / ".seat.lock"
    return seat.exists() and not _is_stale(seat)


def _want_fits(gpu: int, want: dict, budget: int) -> bool:
    """Could this waiting ``--priority`` run be admitted right now?

    Same admission arithmetic as ``reserve``, minus the ``.want`` gate itself
    (a priority run never yields to another). Entries written before the
    payload carried sizes have no numbers to check, so they are treated as
    admissible -- the old blocking behaviour, which is the safe direction.
    """
    if "vram_mb" not in want:
        return True
    exclusive = bool(want.get("exclusive", want.get("vram_mb") is None))
    need = budget if exclusive else max(1, min(int(want["vram_mb"]), budget))
    if reserved_mb(gpu) + need > budget:
        return False
    free = None if exclusive else free_mb(gpu)
    if free is not None and need > free:
        return False
    ram = int(want.get("ram_mb") or 0)
    if ram:
        # Same split as reserve(); see the comment there.
        headroom = ram_headroom_mb()
        avail = ram_available_mb()
        if avail is not None and ram > avail - headroom:
            return False
        total = ram_total_mb()
        if total is not None and ram + booked_ram_mb() > total - headroom:
            return False
    return True


def _blocking_want(gpu: int, budget: int) -> bool:
    """True when a priority run is waiting *and* could actually start.

    A ``--priority`` marker exists to let CI jump the queue, not to idle the
    card. A priority run blocked on something else -- host RAM, most often --
    gains nothing from freezing everyone else, and costs the whole queue: on
    2026-07-28 a CI seat wanting 92 GB of host RAM held GPU 0 empty for 2.7 h
    while three runs that fitted waited behind it. So the marker only blocks
    while the priority run is itself admissible; the moment it is not, the
    queue flows past it and it takes the next slot it can genuinely use.
    """
    return any(_want_fits(gpu, want, budget) for want in _entries(gpu, ".want", prune=True))


def reserve(spool_dir: Path, gpu: int, name: str, vram_mb: int | None = None,
            priority: bool = False, ram_mb: int | None = None) -> tuple[int, Path] | None:
    """Book memory on ``gpu``. ``vram_mb=None`` books the card exclusively;
    ``ram_mb`` books host RAM (machine-wide, across both cards), falling back to
    ``default_ram_mb()`` when ``None`` -- pass ``0`` to book none.
    Returns the held entry, or ``None`` if the card is already committed. Release
    with ``_drop`` when the run ends."""
    budget = vram_budget_mb(gpu)
    exclusive = vram_mb is None
    # A request bigger than the card is clamped, not refused, or it would queue for ever.
    need = budget if exclusive else max(1, min(int(vram_mb), budget))
    admit = _admit_lock(gpu)
    try:
        if not priority and _blocking_want(gpu, budget):
            return None  # a priority run that *could* start is waiting; yield to it
        if _legacy_seat_held(spool_dir, gpu):
            return None
        # Same admit-lock prune for .waiting: they are inert for admission, but
        # dead ones otherwise accumulate for ever (235 on this box by 2026-08-20,
        # oldest 07-27) because nothing else may safely unlink them.
        waiting_entries(gpu, prune=True)
        held_mb = reserved_mb(gpu, prune=True)
        if resource_guard(held_mb + need > budget,
                          f"GPU {gpu}: {need} MiB asked with {held_mb}/{budget} MiB booked"):
            return None
        # An exclusive run states "nobody else joins me", not a size, so there is
        # no size to check against the card. Only a declared reservation is held
        # to the physical floor -- otherwise a stray 2 GB notebook would block
        # every whole-card run for ever, which is worse than the old behaviour.
        free = None if exclusive else free_mb(gpu)
        if resource_guard(free is not None and need > free,
                          f"GPU {gpu}: {need} MiB asked, {free} MiB physically free"):
            return None  # something outside the ledger is holding the card
        # Host RAM, if declared. TWO questions, each asked against the figure that
        # answers it -- mixing them deadlocked admission on 2026-09-07.
        #   1. Can this request fit right now? MemAvailable, which ALREADY discounts
        #      every running process (see ram_available_mb).
        #   2. Do all commitments fit the machine at all? MemTotal against the ledger,
        #      which is what covers a run that has booked but not yet allocated -- the
        #      2026-07-20 OOM, where two runs on different cards passed the "fits now"
        #      test in the window before either allocated.
        # Adding booked_ram_mb() to the MemAvailable test charged a running run twice,
        # once inside avail and once as its booking: measured avail 77821, booked 65536,
        # headroom 8192 left a maximum admissible request of 4093 MiB while 78 GB was
        # free, so nothing could start behind a live booking regardless of size.
        ram = default_ram_mb() if ram_mb is None else int(ram_mb)
        if ram:
            headroom = ram_headroom_mb()
            avail = ram_available_mb()
            if resource_guard(avail is not None and ram > avail - headroom,
                              f"host RAM: {ram} MiB asked, {avail} MiB available, {headroom} MiB headroom"):
                return None
            total = ram_total_mb()
            booked = booked_ram_mb()
            if resource_guard(total is not None and ram + booked > total - headroom,
                              f"host RAM: {ram} MiB asked with {booked} MiB booked of {total} MiB"):
                return None
        # CPU headroom last: it is the cheapest check but the one most likely to change
        # between polls, so it should not shadow a permanent VRAM/RAM refusal.
        if resource_guard(not cpu_headroom_ok(),
                          f"CPU: admitting would leave fewer than {min_free_cores()} cores free"):
            return None
        path = _entry_path(gpu, ".lease")
        return _hold(path, {"name": name, "vram_mb": need, "exclusive": exclusive,
                            "ram_mb": ram,
                            "pid": os.getpid(), "user": getpass.getuser(),
                            "since": _now()}), path
    finally:
        _release_lock(admit)


def managed_gpus() -> list[int]:
    """GPU indices this spooler manages: ``PINO_SPOOL_GPUS`` csv, else the GPUs
    ``nvidia-smi`` reports, else ``[0, 1]`` (this workstation's two cards)."""
    env = os.environ.get("PINO_SPOOL_GPUS")
    if env:
        return [int(x) for x in env.replace(",", " ").split()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        ).stdout
        gpus = [int(m) for m in re.findall(r"^GPU (\d+):", out, re.MULTILINE)]
        if gpus:
            return gpus
    except Exception:
        pass
    return [0, 1]


def gpu_dir(spool_dir: Path, gpu: int) -> Path:
    return Path(spool_dir) / f"gpu{gpu}"


def ensure_dirs(spool_dir: Path, gpu: int) -> Path:
    gdir = gpu_dir(spool_dir, gpu)
    for sub in SUBDIRS:
        (gdir / sub).mkdir(parents=True, exist_ok=True)
    return gdir


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _acquire_lock(path: Path, blocking: bool = False) -> int | None:
    """flock ``path`` exclusively. Returns the fd, or ``None`` if held (non-block).
    The lock is released when the fd is closed, so a crashed holder frees it."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
    flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        fcntl.flock(fd, flags)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _release_lock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def enqueue(spool_dir: Path, gpu: int, name: str, cmd: str,
            vram_mb: int | None = None, ram_mb: int | None = None) -> Path:
    """Atomically add one task to a GPU's queue. Time-ordered, unique filename.
    ``vram_mb=None`` means "reserve the whole card", resolved when it starts;
    ``ram_mb`` optionally books host RAM too."""
    ensure_dirs(spool_dir, gpu)
    payload = json.dumps({"name": name, "cmd": cmd, "gpu": gpu, "added": _now(),
                          "vram_mb": vram_mb, "ram_mb": ram_mb})
    ts = time.time_ns()
    while True:
        path = gpu_dir(spool_dir, gpu) / "queue" / f"{ts:020d}-{os.getpid()}.task"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            ts += 1
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    return path


def _load(spool_dir: Path, gpu: int) -> int:
    """Pending + running task count for a GPU (its current backlog)."""
    gdir = gpu_dir(spool_dir, gpu)
    return len(list((gdir / "queue").glob("*.task"))) + len(
        list((gdir / "running").glob("*.task"))
    )


def select_gpu_auto(spool_dir: Path, gpus: list[int]) -> int:
    """Least-loaded GPU by backlog; ties break to the lowest index."""
    return min(gpus, key=lambda g: (_load(spool_dir, g), g))


def build_command(module: str, overrides: list[str]) -> str:
    """Command for `add --module`, pinned to the interpreter that enqueued it.

    Never a bare "python": the worker launches through `bash -lc`, whose login
    profile need not have the conda env active, so a bare name resolves to the
    system interpreter and the run dies importing physicsnemo.
    """
    parts = [sys.executable, "-u", "-m", f"lpno.models.{module}"]
    parts += [shlex.quote(o) for o in overrides]
    return " ".join(parts)


#: Entry points whose results belong in the provenance store.
_REGISTERED_MODULES = re.compile(
    r"lpno\.models\.NS_Energy_laser_PINO(_eval|_Kfolds)?\b")
#: A throwaway smoke is never registered, so it is never refused.
_SMOKE_STEPS = re.compile(r"training\.max_steps\s*=\s*(\d+)")
_SMOKE_MAX_STEPS = 100


def validate_aiida_or_optout(cmd: str, no_aiida: bool = False) -> None:
    """Refuse to queue a real training/eval unless AiiDA is explicitly opted out.

    The spool is the GPU seat ledger, not a submission path. A run launched
    through it produces weights on disk and nothing else: no model node, no
    dataset linkage, invisible to the console -- and a training that was never
    ingested looks exactly like one that was never run. On 2026-08-08..10 four
    trainings and three evals were orphaned this way, on the false assumption
    that the workchain could not carry env vars and Hydra overrides (it takes
    both: `training_profile` accepts `env` and `hydra_overrides`).

    The escape hatch is deliberate and explicit: `--no-aiida` for a user or a
    machine without a configured profile. Short smokes
    (`training.max_steps <= 100`) pass unguarded, because they must never be
    registered in the first place.
    """
    if no_aiida or not _REGISTERED_MODULES.search(cmd):
        return
    # Hydra takes the LAST occurrence of a repeated override, so must we:
    # reading the first would let `max_steps=32000 ... max_steps=2` look long
    # (and, worse, `max_steps=2 ... max_steps=32000` look like a smoke).
    steps = _SMOKE_STEPS.findall(cmd)
    if steps and int(steps[-1]) <= _SMOKE_MAX_STEPS:
        return
    raise SystemExit(
        "gpu_spool: refusing to queue a training/eval that would bypass AiiDA.\n"
        "  Results launched here are not registered as model nodes: no provenance,\n"
        "  no dataset linkage, not selectable by name. Submit it through aiida-pino\n"
        "  (see docs/training_and_evaluation.md).\n"
        "  The workchain carries what you came here for: training_profile takes\n"
        "  `env` and `hydra_overrides`.\n"
        "  If you genuinely have no AiiDA profile, re-run with --no-aiida.\n"
        f"  Short smokes (training.max_steps <= {_SMOKE_MAX_STEPS}) are never blocked."
    )


def validate_no_gpu_override(cmd: str) -> None:
    if _GPU_OVERRIDE.search(cmd):
        raise SystemExit(
            "task carries a cuda_device/cuda_visible_devices override; the spool "
            "pins the GPU from the queue you add to. Drop the override and pick "
            "the GPU with --gpu."
        )


def _name_from_overrides(overrides: list[str], default: str) -> str:
    for arg in overrides:
        key, _, value = arg.partition("=")
        if key.strip().lstrip("+") in {"custom.name", "name"} and value:
            return value.strip().strip("'\"")
    return default


def _pending(gdir: Path) -> list[Path]:
    return sorted((gdir / "queue").glob("*.task"))


def run_worker(spool_dir: Path, gpu: int, drain: bool = False, poll: float = 1.0) -> None:
    """Drain one GPU's queue until stopped (or empty if drain), running as many
    jobs at once as their declared VRAM fits in the card's budget.

    Serialises on ``.worker.lock`` (one worker per GPU) and books every job in the
    shared ledger, so manual ``seat`` runs and other users' runs count against the
    same budget. Strictly FIFO: a head task too big for what is left waits rather
    than letting smaller tasks behind it jump the queue.
    """
    gdir = ensure_dirs(spool_dir, gpu)
    worker_lock = _acquire_lock(gdir / ".worker.lock")
    if worker_lock is None:
        raise SystemExit(f"A spooler worker is already running for GPU {gpu} ({gdir})")

    # A previous worker died mid-run: re-queue whatever it left in running/.
    for leftover in sorted((gdir / "running").glob("*.task")):
        leftover.rename(gdir / "queue" / leftover.name)

    running: list[dict] = []
    stop = {"flag": False}

    def _handle(signum, frame):
        if stop["flag"]:  # second signal: kill the runs
            for job in running:
                job["proc"].terminate()
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    poollog = gdir / "pool.log"

    try:
        while True:
            for job in list(running):
                if job["proc"].poll() is None:
                    continue
                rc = job["proc"].returncode
                if rc in (-9, 137) and job.get("ram_cap"):
                    # A cgroup OOM kill is a bare SIGKILL: the log would show
                    # "Killed" and a stage that stopped, with no traceback and
                    # nothing naming the cause. Say it in the run's own log.
                    job["logf"].write(
                        f"\n[gpu-spool] killed by SIGKILL at the {job['ram_cap']} MiB "
                        f"host-RAM ceiling declared with --ram-mb "
                        f"(cap {int(job['ram_cap'] * 1.5)} MiB, swap disabled). "
                        f"Re-measure the peak with /usr/bin/time -v and redeclare.\n"
                    )
                job["logf"].close()
                status_name = "done" if rc == 0 else "failed"
                job["path"].rename(gdir / status_name / job["path"].name)
                with poollog.open("a", encoding="utf-8") as handle:
                    handle.write(f"{_now()}\t{status_name}\t{job['name']}\t{rc}\n")
                _drop(job["lease"], job["lease_path"])
                running.remove(job)

            while not stop["flag"] and _pending(gdir):
                task_path = _pending(gdir)[0]
                try:
                    task = json.loads(task_path.read_text(encoding="utf-8"))
                    name, cmd = task["name"], task["cmd"]
                except Exception:  # corrupt task: shelve it and look at the next one
                    task_path.rename(gdir / "failed" / task_path.name)
                    continue
                vram = task.get("vram_mb") or default_vram_mb(gpu)
                booked = reserve(spool_dir, gpu, name, vram, ram_mb=task.get("ram_mb"))
                if booked is None:
                    break  # card is committed; hold the head of the queue
                lease, lease_path = booked
                running_path = gdir / "running" / task_path.name
                task_path.rename(running_path)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                logf = (gdir / "logs" / f"{name}.log").open("w", encoding="utf-8")
                scope = memory_scope_prefix(task.get("ram_mb"))
                proc = subprocess.Popen(
                    (scope + ["bash", "-lc", cmd]) if scope else cmd,
                    shell=not scope, cwd=ROOT, env=env,
                    stdout=logf, stderr=subprocess.STDOUT,
                )
                running.append({"proc": proc, "path": running_path, "name": name,
                                "logf": logf, "lease": lease, "lease_path": lease_path,
                                "ram_cap": task.get("ram_mb") if scope else None})

            if not running and (stop["flag"] or (drain and not _pending(gdir))):
                break
            time.sleep(poll)
    finally:
        _release_lock(worker_lock)


def _start_all(spool_dir: Path, gpus: list[int], drain: bool, poll: float) -> int:
    """Launch one worker subprocess per managed GPU and wait on them."""
    procs = [
        subprocess.Popen(
            # sys.executable, never a bare "python": see build_command above for why a
            # login shell's PATH is not the interpreter this process runs in (2026-08-08).
            [sys.executable, "-m", "lpno.gpu_spool", "--dir", str(spool_dir),
             "start", "--gpu", str(g), "--poll", str(poll)] + (["--drain"] if drain else []),
        )
        for g in gpus
    ]
    print(f"started {len(procs)} worker(s) for GPU(s): {', '.join(map(str, gpus))}")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
    return 0


def _proc_ppid(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _compute_apps(gpu: int) -> list[tuple[int, str, int]]:
    """(pid, name, used MiB) per compute app on the card; [] when unreadable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            capture_output=True, text=True, timeout=10,
        ).stdout
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                rows.append((int(parts[0]), parts[1], int(float(parts[2]))))
        return rows
    except Exception:
        return []


def unbooked_tenants(apps: list[tuple[int, str, int]], lease_pids: set[int],
                     ppid_of=_proc_ppid) -> list[tuple[int, str, int]]:
    """Compute apps whose process tree holds no lease.

    The lease records the *reserving* pid (worker or seat holder); the CUDA
    process is usually its child, so each app climbs its PPid chain looking for
    a lease holder. Anything left is memory the ledger cannot see — the
    inference website server held 1642 MiB for 7 days this way (2026-08-07) and
    only the `free_mb` driver guard stood between that and an OOM.
    """
    out = []
    for pid, name, used in apps:
        p, hops = pid, 0
        while p > 1 and p not in lease_pids and hops < 64:
            p, hops = ppid_of(p), hops + 1
        if p not in lease_pids:
            out.append((pid, name, used))
    return out


def status(spool_dir: Path, gpus: list[int]) -> None:
    for gpu in gpus:
        gdir = ensure_dirs(spool_dir, gpu)

        def names(sub: str) -> list[str]:
            out = []
            for path in sorted((gdir / sub).glob("*.task")):
                try:
                    out.append(json.loads(path.read_text(encoding="utf-8"))["name"])
                except Exception:
                    out.append(path.name)
            return out

        queued, running = names("queue"), names("running")
        done = len(list((gdir / "done").glob("*.task")))
        failed = len(list((gdir / "failed").glob("*.task")))
        print(f"GPU {gpu}: queue {len(queued)} [{', '.join(queued) or '-'}]  "
              f"running [{', '.join(running) or '-'}]  done {done}  failed {failed}")
        leases = _entries(gpu, ".lease")
        held = [f"{e.get('name', '?')} {e.get('vram_mb', 0)}MiB"
                + (f"+{e.get('ram_mb')}MiB RAM" if e.get("ram_mb") else "")
                + f" ({e.get('user', '?')})"
                for e in leases]
        free = free_mb(gpu)
        if gpu == gpus[0]:
            print(f"    host ram {booked_ram_mb()} MiB booked, "
                  f"{ram_available_mb()} MiB available (machine-wide)")
        print(f"    vram {reserved_mb(gpu)}/{vram_budget_mb(gpu)} MiB reserved"
              + (f", {free} MiB free on the card" if free is not None else "")
              + f" [{', '.join(held) or '-'}]"
              + ("  priority run waiting" if _entries(gpu, ".want") else ""))
        blocked = waiting_entries(gpu)
        if blocked:
            names_w = ", ".join(str(w.get("name", "?")) for w in blocked[:6])
            more = f" (+{len(blocked) - 6} more)" if len(blocked) > 6 else ""
            print(f"    waiting for a seat: {names_w}{more}")
        lease_pids = {int(e["pid"]) for e in leases if e.get("pid")}
        ghosts = unbooked_tenants(_compute_apps(gpu), lease_pids)
        if ghosts:
            print("    unbooked on the card (no lease; only the free-VRAM guard "
                  "accounts for these): "
                  + ", ".join(f"pid {p} {n} {m} MiB" for p, n, m in ghosts))
        poollog = gdir / "pool.log"
        if poollog.exists():
            for line in poollog.read_text(encoding="utf-8").splitlines()[-3:]:
                print("    " + line)


#: An ordinary seat waits indefinitely: queueing behind a long run is normal,
#: not a deadlock. Measured over the last 40 runs of each kind (2026-07-28),
#: a training holds its seat for a median of 0.62 h, p90 3.79 h and up to
#: 23.1 h -- so any finite default in the low hours kills legitimately-queued
#: work, which is how a resource gate turns one slow run into a lost batch.
#: The genuine deadlocks are caught by cause instead: a booking above MemTotal
#: is refused outright, and `_blocking_want` stops an unadmittable priority
#: seat from freezing the card.
DEFAULT_SEAT_MAX_WAIT_S = 0.0
#: CI is the exception: it wants a verdict, not a queue slot, so a --priority
#: seat still gives up and reports the cause rather than hanging the pipeline.
DEFAULT_PRIORITY_MAX_WAIT_S = 3600.0
#: How long a --priority seat may block every other run while it cannot start.
#: Sized from the same measurements: evals hold a seat for a median of 0.02 h
#: and p90 0.19 h (11 min), so this outlasts ~90% of them and priority still
#: wins a slot that is about to free; a training's 0.62 h median is far above
#: it, so priority never freezes the queue waiting one out. `_blocking_want`
#: already releases the block the moment the priority run is unadmittable --
#: this is the backstop for the case where it *could* start but has not.
DEFAULT_PRIORITY_YIELD_S = 900.0


def _command_has_cuda_override(command: list[str]) -> bool:
    return any(_GPU_OVERRIDE.search(arg) for arg in command)


def _command_with_cuda_device(command: list[str], gpu: int) -> list[str]:
    if _command_has_cuda_override(command):
        return command
    return [*command, f"custom.cuda_device={gpu}"]


def _seat_candidates(gpu: int | str, gpus: Sequence[int] | None) -> list[int]:
    raw = str(gpu).strip().lower()
    if raw == "auto":
        candidates = list(gpus) if gpus is not None else managed_gpus()
        if not candidates:
            raise SystemExit("[gpu-spool] no managed GPUs available for --gpu auto")
        return sorted({int(g) for g in candidates})
    return [int(gpu)]


def _rank_seat_candidates(candidates: Sequence[int]) -> list[int]:
    """Try the currently least-committed card first; ties keep low-index order."""
    return sorted(candidates, key=lambda g: (reserved_mb(g, prune=True), g))


def _tree_pss_mb(root_pid: int) -> int | None:
    """Summed Pss (MiB) of ``root_pid`` and every descendant, via smaps_rollup.

    Pss apportions shared pages, so summing over the tree does not double-count
    the way RSS does (which produced an impossible 147 GB on a 125 GB box twice
    before this was learned). Returns None if nothing could be read.
    """
    try:
        ppid_of: dict[int, int] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as fh:
                    stat = fh.read().decode("utf-8", "replace")
                ppid_of[int(entry)] = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
        tree = {root_pid}
        grew = True
        while grew:
            grew = False
            for pid, ppid in ppid_of.items():
                if ppid in tree and pid not in tree:
                    tree.add(pid); grew = True
        total_kb = 0; seen_any = False
        for pid in tree:
            try:
                with open(f"/proc/{pid}/smaps_rollup", "rb") as fh:
                    for line in fh:
                        if line.startswith(b"Pss:"):
                            total_kb += int(line.split()[1]); seen_any = True
                            break
            except OSError:
                continue
        return total_kb // 1024 if seen_any else None
    except Exception:
        return None


def _reconcile_lease_ram(fd: int, path: Path, measured_mb: int,
                         margin: float = 1.15) -> bool:
    """RAISE a lease's host-RAM booking to its measured use; never lower it.

    Closes the open loop behind six worker SIGKILLs in one campaign: bookings
    were declared at submit from guesses or transferred measurements and never
    reconciled, while admission packs strictly by the declared figure -- so an
    under-declared run was admitted beside co-tenants the box could not hold and
    the kernel picked a victim. Raising the booking makes the ledger truthful
    before the next admission stacks another tenant on top.

    Deliberately raise-ONLY. Shrinking over-declarations would densify packing,
    and every extra co-tenant evicts page cache the lazy loader depends on --
    measured 2026-09-09 as a 2x throughput loss on the incumbent with every
    anonymous-RAM booking satisfied. Phantom headroom is accidentally protective
    of the resource nothing declares; reclaiming it needs a cache-aware
    admission model, not this function.

    The truth is recorded either way (``ram_mb_measured``); the original
    declaration survives as ``ram_mb_declared``.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    payload.setdefault("ram_mb_declared", payload.get("ram_mb", 0))
    payload["ram_mb_measured"] = int(measured_mb)
    current = int(payload.get("ram_mb") or 0)
    # Raise only when the MEASUREMENT exceeds the booking. Testing
    # `measured * margin > current` instead made the lease ratchet against a flat
    # measurement -- 49152 -> 54948 -> 55948 -> 56007 while real use sat at
    # 47.8-48.7 GB -- because each round then compared against the previous RAISE.
    # A booking that already covers what the run uses is correct and is left alone.
    changed = int(measured_mb) > current
    if changed:
        target = int(measured_mb * margin)
        payload["ram_mb"] = target
        print(f"[gpu-spool] booking under-declared: measured {measured_mb} MiB "
              f"against {current} booked; lease raised to {target} MiB so the "
              "next admission sees the truth.", flush=True)
    data = (json.dumps(payload) + "\n").encode("utf-8")
    os.ftruncate(fd, 0)
    os.pwrite(fd, data, 0)
    return changed


def _seat_wait_notice(*, priority: bool, max_wait_s: float,
                      priority_yield_s: float = 0.0) -> str:
    """One line stating how long this seat will wait, printed when it starts waiting.

    `--priority` buys TWO things and only one of them is usually wanted: the
    queue-jump, and a swap of the unbounded wait for DEFAULT_PRIORITY_MAX_WAIT_S
    "so CI reports a cause". On a saturated box the deadline is what kills the
    run -- two evals at 3601 s (2026-07-29) and a training arm at 3602 s
    (2026-09-10), each waiting for a seat that was hours or days out, each
    diagnosed from the corpse. The deadline stays because CI depends on it; this
    makes a run say what it bought BEFORE the hour runs out.
    """
    parts: list[str] = []
    if max_wait_s and max_wait_s > 0:
        parts.append(
            f"[gpu-spool] this seat will GIVE UP after {max_wait_s:g}s"
            f"{' (implied by --priority)' if priority else ''}. "
            "If the seat you need is further out than that, pass "
            "--max-wait-s 0 to wait indefinitely.")
    else:
        parts.append("[gpu-spool] this seat will wait indefinitely for a slot"
                     f"{' (priority)' if priority else ''}.")
    if priority and priority_yield_s and priority_yield_s > 0:
        # The queue-jump is the `.want` marker, and the yield DROPS it. So priority
        # is a window, not a standing claim: past it this is an ordinary waiter
        # racing like everything else. For a seat hours out the jump is gone for
        # nearly the whole wait, which is why priority is the wrong tool there.
        parts.append(
            f"Its queue-jump lasts only {priority_yield_s:g}s -- after that the "
            "marker is released and this becomes an ordinary waiter.")
    return " ".join(parts)


def _run_seat(spool_dir: Path, gpu: int | str, command: list[str], vram_mb: int | None = None,
              priority: bool = False, ram_mb: int | None = None, poll: float = 5.0,
              max_wait_s: float | None = None,
              priority_yield_s: float = DEFAULT_PRIORITY_YIELD_S,
              gpus: Sequence[int] | None = None) -> int:
    """Run a command directly under a VRAM reservation (blocks until it fits),
    pinning ``CUDA_VISIBLE_DEVICES`` the same way the worker does.

    ``priority`` parks a marker in the ledger first: the workers then refuse to
    start anything new, so this run goes in ahead of the queue rather than behind
    it. CI uses it so the GPU smoke checks are not stuck behind a long training.
    """
    if not command:
        raise SystemExit("seat needs a command after --: seat --gpu N -- <command>")
    if max_wait_s is None:
        # CI (--priority) wants a verdict; an ordinary queued run wants its turn.
        max_wait_s = DEFAULT_PRIORITY_MAX_WAIT_S if priority else DEFAULT_SEAT_MAX_WAIT_S
    candidates = _seat_candidates(gpu, gpus)
    auto_gpu = len(candidates) != 1 or str(gpu).strip().lower() == "auto"
    for candidate in candidates:
        ensure_dirs(spool_dir, candidate)
    ram = default_ram_mb() if ram_mb is None else int(ram_mb)
    name = f"seat:{Path(command[0]).name}"
    asked = f"{int(vram_mb)} MiB" if vram_mb else "the whole card"

    # Admission check. `reserve` clamps an oversized VRAM request so it cannot
    # queue for ever; host RAM had no equivalent, so a booking above what the box
    # can *ever* provide waited indefinitely instead of failing. MemTotal is the
    # right bar: MemAvailable fluctuates and is worth waiting on, MemTotal is not.
    total = ram_total_mb()
    if resource_guard(bool(ram and total and ram > total),
                      f"--ram-mb {ram} exceeds the machine's {total} MiB"):
        raise SystemExit(
            f"[gpu-spool] refusing --ram-mb {ram}: the machine's total RAM is "
            f"{total} MiB, so this booking can never be satisfied no matter what "
            "finishes. Lower the declaration to the run's measured peak."
        )

    wants: list[tuple[int, Path]] = []
    if priority:
        for candidate in candidates:
            candidate_vram = int(vram_mb) if vram_mb else default_vram_mb(candidate)
            want_path = _entry_path(candidate, ".want")
            # Sizes travel with the marker so other waiters can tell whether this
            # priority run could actually start (see `_blocking_want`): a marker
            # that cannot be admitted must not freeze the card.
            wants.append((_hold(want_path, {"name": name, "pid": os.getpid(),
                                            "vram_mb": candidate_vram,
                                            "exclusive": candidate_vram is None,
                                            "ram_mb": ram,
                                            "user": getpass.getuser(),
                                            "since": _now()}), want_path))
    waiting: list[tuple[int, Path]] = []
    try:
        waited = False
        began = time.monotonic()
        while True:
            booked = None
            booked_gpu = None
            for candidate in _rank_seat_candidates(candidates):
                candidate_vram = int(vram_mb) if vram_mb else default_vram_mb(candidate)
                booked = reserve(spool_dir, candidate, name, candidate_vram,
                                 priority=priority, ram_mb=ram)
                if booked is not None:
                    booked_gpu = candidate
                    break
            if booked is not None and booked_gpu is not None:
                break
            elapsed = time.monotonic() - began
            # Yield: a priority seat stops workers starting anything new. If it
            # still cannot be admitted, that blockage buys it nothing and costs
            # the whole queue, so drop the marker and keep waiting as an ordinary
            # run rather than leaving the card idle *and* the queue frozen.
            if wants and priority_yield_s and elapsed >= priority_yield_s:
                for want in wants:
                    _drop(*want)
                wants = []
                target = "any managed GPU" if auto_gpu else f"GPU {candidates[0]}"
                print(f"[gpu-spool] priority seat on {target} has waited "
                      f"{int(elapsed)}s; releasing its hold so queued runs can "
                      "start (still waiting for a slot).", flush=True)
            if max_wait_s and elapsed >= max_wait_s:
                target = "any managed GPU" if auto_gpu else f"GPU {candidates[0]}"
                raise SystemExit(
                    f"[gpu-spool] gave up after waiting {int(elapsed)}s for {asked} "
                    f"on {target} + {ram} MiB host RAM "
                    f"({ram_available_mb()} MiB available). The booking may be "
                    "larger than this box can free while other runs hold memory: "
                    "lower --ram-mb/--vram-mb to the run's measured peak, or pass "
                    "--max-wait-s 0 to wait indefinitely."
                )
            if not waiting:
                # Publish the wait so observers can see a blocked run at all.
                # Best effort: never let bookkeeping stop a run from queuing.
                for candidate in candidates:
                    try:
                        candidate_vram = int(vram_mb) if vram_mb else default_vram_mb(candidate)
                        waiting_path = _entry_path(candidate, ".waiting")
                        waiting.append((_hold(waiting_path, {
                            "name": name, "pid": os.getpid(), "user": getpass.getuser(),
                            "since": _now(), "vram_mb": candidate_vram, "ram_mb": ram,
                            "priority": bool(priority), "auto_gpu": bool(auto_gpu),
                        }), waiting_path))
                    except OSError:
                        pass
            if not waited:
                ram_note = f" + {ram} MiB host RAM ({ram_available_mb()} MiB available)" if ram else ""
                if auto_gpu:
                    pressure = ", ".join(
                        f"GPU {candidate}: {reserved_mb(candidate)}/{vram_budget_mb(candidate)} MiB"
                        for candidate in candidates
                    )
                    print(f"[gpu-spool] waiting for {asked} on any managed GPU{ram_note} "
                          f"({pressure} reserved)", flush=True)
                else:
                    only = candidates[0]
                    print(f"[gpu-spool] waiting for {asked} on GPU {only}{ram_note} "
                          f"({reserved_mb(only)}/{vram_budget_mb(only)} MiB reserved)", flush=True)
                print(_seat_wait_notice(priority=priority, max_wait_s=max_wait_s,
                                        priority_yield_s=priority_yield_s), flush=True)
                waited = True
            time.sleep(poll)
    finally:
        for want in wants:
            _drop(*want)  # booked: stop holding the workers back
        for waiter in waiting:
            _drop(*waiter)  # no longer waiting: either booked or giving up
    lease, lease_path = booked
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(booked_gpu)
        if auto_gpu:
            print(f"[gpu-spool] auto seat admitted on GPU {booked_gpu}", flush=True)
            command = _command_with_cuda_device(command, booked_gpu)
        proc = subprocess.Popen(memory_scope_prefix(ram_mb) + command, cwd=ROOT, env=env)
        # Close the booking loop while the child runs: measure the tree's Pss
        # after warmup and periodically, and raise the lease if reality exceeds
        # the declaration (never lower it -- see _reconcile_lease_ram).
        next_check = time.monotonic() + 120.0
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            if ram and time.monotonic() >= next_check:
                measured = _tree_pss_mb(proc.pid)
                if measured:
                    _reconcile_lease_ram(lease, lease_path, measured)
                next_check = time.monotonic() + 600.0
            time.sleep(5.0)
    finally:
        _drop(lease, lease_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=default_spool_dir(),
                        help="Spool directory (default: ~/.pino/spool or $PINO_SPOOL_DIR)")
    parser.add_argument("--gpus", help="Managed GPU indices, csv (default: $PINO_SPOOL_GPUS / nvidia-smi / 0,1)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Enqueue a training run or an arbitrary --cmd")
    p_add.add_argument("--gpu", default="auto", help="Target GPU index, or 'auto' (least-loaded)")
    p_add.add_argument("--name", help="Label for the task (default: custom.name= or module)")
    p_add.add_argument("--module", default=DEFAULT_MODULE, help=f"models.<module> to run (default: {DEFAULT_MODULE})")
    p_add.add_argument("--cmd", help="Enqueue this exact shell command instead of a training run")
    p_add.add_argument("--no-aiida", action="store_true",
                       help="queue a training/eval that bypasses AiiDA (no provenance); "
                            "for machines without a configured profile")
    p_add.add_argument("--vram-mb", type=int, help="VRAM this run needs (default: the whole card)")
    p_add.add_argument("--ram-mb", type=int, help="Host RAM this run needs (default: 32768; 0 books none)")
    p_add.add_argument("rest", nargs=argparse.REMAINDER, help="Hydra overrides for the training run")

    p_start = sub.add_parser("start", help="Run the worker(s); no --gpu means one per managed GPU")
    p_start.add_argument("--gpu", type=int, help="Run the worker for just this GPU")
    p_start.add_argument("--drain", action="store_true", help="Exit when the queue empties")
    p_start.add_argument("--poll", type=float, default=1.0, help="Worker poll interval (s)")

    sub.add_parser("status", help="Show per-GPU queued / running / recent")

    p_seat = sub.add_parser("seat", help="Run a command directly under a VRAM reservation")
    p_seat.add_argument("--gpu", required=True, help="GPU index to hold, or 'auto'")
    p_seat.add_argument("--vram-mb", type=int, help="VRAM to reserve (default: the whole card)")
    p_seat.add_argument("--ram-mb", type=int, help="Host RAM to reserve (default: 32768; 0 books none)")
    p_seat.add_argument("--priority", action="store_true",
                        help="Go in ahead of queued jobs (workers hold off starting new runs)")
    p_seat.add_argument("--max-wait-s", type=float, default=None,
                        help=f"Give up if not admitted within this long (default: "
                             f"{DEFAULT_PRIORITY_MAX_WAIT_S:g}s with --priority so CI reports a "
                             f"cause, otherwise {DEFAULT_SEAT_MAX_WAIT_S:g} = wait forever, since "
                             "queueing behind a long training is normal; 0 waits forever)")
    p_seat.add_argument("--priority-yield-s", type=float, default=DEFAULT_PRIORITY_YIELD_S,
                        help=f"A --priority seat releases its hold on the queue after "
                             f"this long, still waiting (default: {DEFAULT_PRIORITY_YIELD_S:g}s; 0 never yields)")
    p_seat.add_argument("rest", nargs=argparse.REMAINDER, help="-- <command to run>")

    args = parser.parse_args(argv)
    gpus = [int(x) for x in args.gpus.replace(",", " ").split()] if args.gpus else managed_gpus()

    if args.command == "add":
        if args.cmd:
            cmd = args.cmd
            name = args.name or "cmd"
        else:
            overrides = [a for a in args.rest if a != "--"]
            if not overrides:
                raise SystemExit("add needs hydra overrides (or --cmd)")
            cmd = build_command(args.module, overrides)
            name = args.name or _name_from_overrides(overrides, args.module)
        validate_no_gpu_override(cmd)
        validate_aiida_or_optout(cmd, no_aiida=bool(getattr(args, "no_aiida", False)))
        gpu = select_gpu_auto(args.dir, gpus) if str(args.gpu) == "auto" else int(args.gpu)
        enqueue(args.dir, gpu, name, cmd, vram_mb=args.vram_mb, ram_mb=args.ram_mb)
        vram = f"{args.vram_mb} MiB" if args.vram_mb else "whole card"
        print(f"queued on GPU {gpu} [{name}] ({vram}): {cmd}")
    elif args.command == "start":
        if args.gpu is None:
            return _start_all(args.dir, gpus, args.drain, args.poll)
        run_worker(args.dir, args.gpu, drain=args.drain, poll=args.poll)
    elif args.command == "status":
        status(args.dir, gpus)
    elif args.command == "seat":
        command = [a for a in args.rest if a != "--"]
        return _run_seat(args.dir, args.gpu, command, vram_mb=args.vram_mb,
                         priority=args.priority, ram_mb=args.ram_mb,
                         max_wait_s=args.max_wait_s,
                         priority_yield_s=args.priority_yield_s,
                         gpus=gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
