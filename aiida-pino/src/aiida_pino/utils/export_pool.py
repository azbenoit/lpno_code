"""Sizing and process management for the derived-member export pool.

Deliberately NOT in `data/pino_dataset_collection.py`: that file is hashed into every
runtime cache's `staging_code_version` stamp, and a worker count or a memory factor cannot
change a single staged byte -- yet every edit here used to rebuild every cache on the box
(three full 60 GB rebuilds on 2026-09-14). Nothing below touches array content.
"""

from __future__ import annotations

#: Hard ceiling on export workers regardless of box size or env override. Set to
#: 8, not 16: at 3.3 GB decompressed per member (plus float64 transform copies)
#: sixteen workers is ~53 GB of inputs alone, and the throughput of an
#: I/O-and-memory-bound job saturates far below that. This machine is shared with
#: FLOW-3D solves and ML trainings, and the machine budget leaves at least 4
#: cores and 20 GB for the OS and interactive login -- oversubscribing them has
#: driven the machine to load 24 with sshd evicted to swap.
EXPORT_WORKERS_MAX = 8
#: Cores deliberately left free for the OS, sshd and whatever else is running.
EXPORT_WORKERS_RESERVED_CORES = 4
#: RAM per concurrent member export. Measured on a 120-case moving-frame set: 3.3 GB decompressed
#: (twelve 238 MB fields + a 713 MB velocity array), and the moving average runs
#: `mean(dtype=np.float64)`, which doubles the largest field again. 6 GB is
#: deliberately generous -- under-estimating this is what turns parallelism into
#: a swap-thrash lockout.
EXPORT_PER_MEMBER_GB = 6.0
#: RAM left for the OS, interactive login and co-tenant jobs (machine budget: >=20 GB).
EXPORT_RESERVED_GB = 20.0
#: Peak RSS of one export worker as a multiple of the member's UNCOMPRESSED array bytes.
#: Three calibration points, one per regime; adjust from the next measurement, not by
#: reasoning. A plain copy (crop/stride) holds the canonical arrays once plus one derived
#: copy: ~2x, which is where the flat `EXPORT_PER_MEMBER_GB` came from (3.3 GB members).
#: A derivation that COLLAPSES the frame axis (`moving_ref_frame` + `time_avg_window` > 1,
#: the steady window) decompresses the whole payload once and writes one frame per field:
#: measured 2026-09-14 with `derived_member_worker` in a fresh process, empty conduction
#: cache -- one member (18.01 GB raw) peaked at 19.22 GB, another (11.78 GB raw) at
#: 12.56 GB, a factor of 1.07 both times; 1.3 leaves margin. That second member is the one
#: that peaked at 74 GB on 2026-09-11 (random paths) and made the factor 6.5: the export
#: path of that day, not this one. 6.5 stays for derivations that KEEP the frame axis
#: (`derived_fields` or a moving frame without averaging), which have not been re-measured.
EXPORT_PEAK_FACTOR_PLAIN = 2.0
EXPORT_PEAK_FACTOR_STEADY = 1.3
EXPORT_PEAK_FACTOR_HEAVY = 6.5


def per_member_export_peak_bytes(members, profile: dict | None, *, fields=None) -> int:
    """Bytes one export worker will hold for the LARGEST member under `profile`.

    From `array_shapes` alone, so it costs milliseconds at resolve time. Member count is not
    the risk signal -- three members took the box to 19 GB available with swap full; one
    member's derivation is. Sized on the largest member because that is the one that gets
    the worker killed. `fields` names the arrays the worker will inflate (the training's
    `runtime_fields`); None means all of them, which is also what the worker then does.
    """
    import numpy as _np

    from aiida_pino.utils.runtime_export import payload_locator

    wanted = {str(f) for f in fields} if fields else None
    builds = bool(profile.get("derived_fields"))

    profile = profile or {}
    heavy = bool(profile.get("derived_fields")) or bool(profile.get("moving_ref_frame")) \
        or int(profile.get("time_avg_window") or 1) > 1
    steady = bool(profile.get("moving_ref_frame")) and int(profile.get("time_avg_window") or 1) > 1
    factor = EXPORT_PEAK_FACTOR_STEADY if steady else (EXPORT_PEAK_FACTOR_HEAVY if heavy else EXPORT_PEAK_FACTOR_PLAIN)
    # A moving-frame derivation reads only the frames it keeps (`member_frame_window`), so the
    # inflated fields are that share of the member; the derived fields still span every frame.
    import os as _os

    window_steps = int(profile.get("moving_frame_steps") or 1) if profile.get("moving_ref_frame") \
        and _os.environ.get("AIIDA_PINO_EXPORT_FULL_FRAMES", "").strip() not in ("1", "true", "yes") else None
    largest = 0
    for member in members or []:
        try:
            shapes = member.get_array_shapes() or {}
            if wanted is not None and any(name in wanted for name in shapes):
                shapes = {name: shape for name, shape in shapes.items() if name in wanted}
            per_field = [int(_np.prod([int(d) for d in shape])) * 4 for shape in shapes.values() if shape]
            raw = sum(per_field)
        except Exception:  # noqa: BLE001 - a member without shapes contributes nothing
            continue
        # The compressed payload is read whole into memory before any field is inflated
        # (`read_locator_bytes`), and a field build (`conduction_field`) steps float64
        # temporaries over every frame of one field: measured 2026-09-14 on the larger member with two
        # fields selected, 6.75 GB = 0.67 GB payload + 2.1 GB fields + ~4 GB of build.
        try:
            locator = payload_locator(member)
            compressed = int((locator or {}).get("size") or 0)
        except Exception:  # noqa: BLE001 - no locator, nothing to add
            compressed = 0
        build = 4 * min(per_field) if builds and per_field else 0  # one scalar field's frames, float64 x2
        if window_steps:
            try:
                frames = int(member.get_num_timesteps() or 0)
            except Exception:  # noqa: BLE001 - unknown frame count: keep the full estimate
                frames = 0
            if frames > window_steps:
                raw = raw * window_steps // frames
        largest = max(largest, int(compressed + raw * factor + build))
    return largest


def _run_derived_pool(tasks: list[dict], workers: int, worker=None) -> list:
    """Run `worker` over `tasks` in a process pool; on ANY failure, kill every worker it started.

    `ProcessPoolExecutor` does not promise that. When a worker dies abruptly the pool goes
    BROKEN and CPython 3.10's `terminate_broken` walks the pending futures calling
    `set_exception` -- but `pool.map`'s result iterator has already CANCELLED them, so that
    call raises `InvalidStateError` and the loop aborts BEFORE reaching `p.terminate()`.
    The surviving workers keep running their current task to completion. On 2026-09-10
    that meant ~7 min conduction builds still writing `<case>.npz` into the parent's build
    dir after the parent had given up and retried into the same path (see
    `_fresh_build_dir`), and 14 h later 6.7 GB of leftovers still being written.

    Children are found by diffing `multiprocessing.active_children()` around the pool, so
    only processes THIS pool started are touched. SIGTERM, a short join, then SIGKILL.
    """
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor

    from aiida_pino.utils.runtime_export import derived_member_worker

    worker = worker or derived_member_worker
    before = {proc.pid for proc in _mp.active_children()}
    # SPAWN, never fork. Forked from a daemon worker -- multithreaded, an event loop, a
    # broker socket, an SQLAlchemy session -- the pool broke within a minute on four of
    # four attempts across three workers (2026-09-10/11), while the identical export of the
    # sibling val split from a plain process ran to completion with two workers in 338 s.
    # `derived_member_worker` is session-free and module-importable by design, which is
    # exactly what spawn needs; the interpreter start-up is seconds against multi-minute tasks.
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=_mp.get_context("spawn"))
    try:
        return list(pool.map(worker, tasks, chunksize=1))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        stragglers = [proc for proc in _mp.active_children() if proc.pid not in before]
        for proc in stragglers:
            proc.terminate()
        for proc in stragglers:
            proc.join(5)
            if proc.is_alive():
                proc.kill()
        raise
    finally:
        pool.shutdown(wait=False)


#: Bytes kept inside a job's own memory limit for the parent process (the AiiDA runner and
#: its imports) beside the export workers.
JOB_PARENT_RESERVE_BYTES = int(2e9)


def job_cpu_count() -> int:
    """CPUs this process may run on: a Slurm job's allocation, not the node's core count."""
    import os as _os

    try:
        return len(_os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return _os.cpu_count() or 1


def cgroup_memory_headroom(proc_cgroup: str = "/proc/self/cgroup", sys_root: str = "/sys/fs/cgroup") -> int | None:
    """Bytes the enclosing memory cgroup (a Slurm job, a container) can still take, or None.

    psutil's `available` is the NODE's: 527 GB on the cluster node where a reproduction's
    evaluation ran in a 32 GB job, so eight workers started and the kernel killed one
    (2026-10-08). Page cache is reclaimable and is not counted as used. The tightest limit on
    the path to the root wins; cgroup v2 and v1 are both read.
    """
    from pathlib import Path as _Path

    try:
        lines = _Path(proc_cgroup).read_text().splitlines()
    except OSError:
        return None
    tightest: int | None = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, relative = parts[1], parts[2]
        if controllers == "":
            base, limit_file, usage_file, cache_key = _Path(sys_root), "memory.max", "memory.current", "file"
        elif "memory" in controllers.split(","):
            base, limit_file, usage_file, cache_key = (_Path(sys_root) / "memory", "memory.limit_in_bytes",
                                                       "memory.usage_in_bytes", "total_cache")
        else:
            continue
        node = base / relative.strip("/")
        while True:
            try:
                limit = (node / limit_file).read_text().strip()
                if limit != "max" and int(limit) < (1 << 60):
                    used = int((node / usage_file).read_text().strip())
                    cache = 0
                    for stat in (node / "memory.stat").read_text().splitlines():
                        key, _, value = stat.partition(" ")
                        if key == cache_key:
                            cache = int(value)
                            break
                    room = max(0, int(limit) - max(0, used - cache))
                    tightest = room if tightest is None else min(tightest, room)
            except (OSError, ValueError):
                pass
            if node == base or node.parent == node:
                break
            node = node.parent
    return tightest


def export_worker_count(cpu_count: int | None = None, n_members: int = 2, *,
                        per_member_bytes: int | None = None, available_bytes: int | None = None) -> int:
    """`_export_worker_count_soft`, held inside the memory limit of the job it runs in.

    The soft count is the local machine-budget policy and, under the public default
    `AIIDA_PINO_RESOURCE_GUARDS=warn`, is reported rather than enforced. A job's or a
    container's memory limit is different in kind: past it the kernel kills a worker, so it is
    always applied. Applied only when the caller leaves `available_bytes` to be measured.
    """
    headroom = cgroup_memory_headroom() if available_bytes is None else None
    if headroom is not None:
        try:
            import psutil  # noqa: PLC0415

            available_bytes = min(int(psutil.virtual_memory().available), headroom)
        except Exception:  # noqa: BLE001 - psutil optional
            available_bytes = headroom
    count = _export_worker_count_soft(job_cpu_count() if cpu_count is None else cpu_count, n_members,
                                      per_member_bytes=per_member_bytes, available_bytes=available_bytes)
    if headroom is None or count <= 1:
        return count
    per_member = int(per_member_bytes or EXPORT_PER_MEMBER_GB * 1e9)
    fits = max(1, (headroom - JOB_PARENT_RESERVE_BYTES) // per_member)
    if count > fits:
        print(f"[resource-guard] {count} export workers do not fit this job's memory limit "
              f"({headroom / 1e9:.1f} GB free, about {per_member / 1e9:.1f} GB per member); using {fits}.",
              flush=True)
    return int(min(count, fits))


def _export_worker_count_soft(cpu_count: int | None = None, n_members: int = 2, *,
                              per_member_bytes: int | None = None, available_bytes: int | None = None) -> int:
    """Threads to use for a runtime-cache export: parallel, but bounded.

    The export decompresses NPZ members, which is per-case independent and was
    running strictly serially -- measured 2026-08-05 pinning ONE core at 99%
    while 19 cores and both GPUs idled and two trainings waited hours on it.

    Threads, not processes: `zlib` releases the GIL during decompression (the
    dominant cost), so threads get real parallelism without pickling AiiDA nodes
    or opening a database connection per worker.

    The bound is the point. `AIIDA_PINO_EXPORT_WORKERS` overrides the default but
    is CLAMPED, not trusted -- a careless value must not be able to swamp a box
    that is also running a solve.

    Under `AIIDA_PINO_RESOURCE_GUARDS=warn` (the public export's default) the memory
    bound is reported, not applied: the count comes from the override or the cores, and
    a line names the estimate it would have clamped to. The thresholds are this
    workstation's; on another box the reader sets `AIIDA_PINO_EXPORT_WORKERS`.
    """

    import os as _os

    from aiida_pino.utils.guards import resource_guard_mode

    if cpu_count is None:
        cpu_count = _os.cpu_count() or 1
    if n_members <= 1:
        return 1

    # How many members fit beside the reserve, from THIS collection's own peak when the
    # caller measured one. Applied to the env override too: `AIIDA_PINO_EXPORT_WORKERS=2`
    # was in force on 2026-09-11 and could not stop one 74 GB member taking the box.
    if available_bytes is None:
        try:
            import psutil  # noqa: PLC0415
            available_bytes = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001 - psutil optional; both rules fall back to conservative
            available_bytes = None
    fits: int | None = None
    if per_member_bytes and available_bytes is not None:
        budget = max(0, available_bytes - int(EXPORT_RESERVED_GB * 1e9))
        fits = max(1, budget // int(per_member_bytes))  # 1 = serialise, never refuse
    warn_only = resource_guard_mode() == "warn"

    def _warn_instead_of_clamp(wanted: int, bound: int) -> int:
        if wanted > bound:
            print(
                f"[resource-guard] WARNING: {wanted} export workers exceed what the memory estimate "
                f"admits ({bound}: {(per_member_bytes or 0) / 1e9:.0f} GB per member beside a "
                f"{EXPORT_RESERVED_GB:.0f} GB reserve out of {(available_bytes or 0) / 1e9:.0f} GB "
                f"available); not enforced. Set AIIDA_PINO_EXPORT_WORKERS to what this machine can hold.",
                flush=True,
            )
        return wanted

    override = _os.environ.get("AIIDA_PINO_EXPORT_WORKERS")
    if override:
        try:
            wanted = max(1, min(EXPORT_WORKERS_MAX, int(override)))
            if fits is None:
                return wanted
            return _warn_instead_of_clamp(wanted, fits) if warn_only else min(wanted, fits)
        except ValueError:
            pass  # unparseable override falls through to the computed default

    usable = cpu_count - EXPORT_WORKERS_RESERVED_CORES
    if usable < 2:
        return 1
    if warn_only:
        by_cores = max(1, min(EXPORT_WORKERS_MAX, usable))
        bound = by_cores
        if available_bytes:
            bound = min(bound, max(1, int(max(0.0, available_bytes / 1e9 - EXPORT_RESERVED_GB) // EXPORT_PER_MEMBER_GB)))
        if fits is not None:
            bound = min(bound, fits)
        return _warn_instead_of_clamp(by_cores, bound)

    # Bound by MEMORY, not just cores. This export decompresses a whole
    # simulation at a time: measured 2026-08-05, one member of that set is a
    # 49.8 MB NPZ that expands 66.8x to 3.3 GB (twelve 238 MB fields plus a
    # 713 MB velocity array), before the float64 moving-average copy. A
    # core-derived bound returned 16 on the development machine -- ~53 GB of inputs alone,
    # enough to swap-thrash the machine (load 24, sshd evicted to swap). Cores are the wrong axis for a 3.3 GB-per-item
    # job.
    available_gb = (available_bytes or 0) / 1e9  # one source of truth, resolved above
    if available_gb > 0:
        budget_gb = max(0.0, available_gb - EXPORT_RESERVED_GB)
        by_memory = int(budget_gb // EXPORT_PER_MEMBER_GB)
        usable = min(usable, max(1, by_memory))
    else:
        usable = min(usable, 2)  # unknown memory: stay conservative
    if fits is not None:
        usable = min(usable, fits)
    return max(1, min(EXPORT_WORKERS_MAX, usable))

