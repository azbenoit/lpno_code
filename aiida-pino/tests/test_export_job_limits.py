"""The export pool sizes itself from the job it runs in: CPU affinity and the cgroup memory limit.

2026-10-08, reproduction from the deposit: in a 32 GB, 8-CPU Slurm job on a 527 GB, 72-core
node, the pool read the node's numbers, started eight workers, and the kernel killed one.
"""
from __future__ import annotations

from aiida_pino.utils import export_pool
from aiida_pino.utils.export_pool import cgroup_memory_headroom, export_worker_count

GB = 10 ** 9


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v2_job_limit_counts_reclaimable_cache_as_free(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/system.slice/slurmstepd.scope/job_1/step_batch/user/task_0\n")
    root = tmp_path / "sys"
    job = root / "system.slice/slurmstepd.scope/job_1"
    _write(job / "memory.max", str(32 * GB))
    _write(job / "memory.current", str(10 * GB))
    _write(job / "memory.stat", f"anon {6 * GB}\nfile {4 * GB}\n")
    _write(job / "step_batch/user/task_0/memory.max", "max")
    _write(root / "memory.max", "max")
    assert cgroup_memory_headroom(str(proc), str(root)) == 26 * GB


def test_cgroup_v1_memory_controller(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("12:cpuset:/slurm/uid_1/job_2\n4:memory:/slurm/uid_1/job_2/step_batch\n0::/\n")
    mem = tmp_path / "sys/memory"
    _write(mem / "slurm/uid_1/job_2/memory.limit_in_bytes", str(16 * GB))
    _write(mem / "slurm/uid_1/job_2/memory.usage_in_bytes", str(5 * GB))
    _write(mem / "slurm/uid_1/job_2/memory.stat", f"cache 1\ntotal_cache {1 * GB}\n")
    _write(mem / "memory.limit_in_bytes", str(1 << 62))   # unlimited at the root
    assert cgroup_memory_headroom(str(proc), str(tmp_path / "sys")) == 12 * GB


def test_no_limit_means_no_headroom_bound(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/user.slice/session.scope\n")
    _write(tmp_path / "sys/user.slice/session.scope/memory.max", "max")
    assert cgroup_memory_headroom(str(proc), str(tmp_path / "sys")) is None


def test_the_job_limit_holds_even_in_warn_mode(monkeypatch):
    monkeypatch.setenv("AIIDA_PINO_RESOURCE_GUARDS", "warn")
    monkeypatch.setattr(export_pool, "cgroup_memory_headroom", lambda: 20 * GB)
    monkeypatch.setattr(export_pool, "job_cpu_count", lambda: 8)
    # (20 GB - 2 GB parent reserve) // 9 GB per member = 2
    assert export_worker_count(n_members=24, per_member_bytes=9 * GB) == 2
    # A caller that states the memory itself is not second-guessed.
    assert export_worker_count(cpu_count=8, n_members=24, per_member_bytes=9 * GB,
                               available_bytes=500 * GB) > 2
