"""A run inside an sbatch script without srun must start as one process instead of crashing in PhysicsNeMo's Slurm
set-up (SLURM_PROCID present, the srun-only variables absent -> int(None))."""
from lpno.models.cuda_bootstrap import ensure_single_process_launch_env


def test_sbatch_without_srun_drops_the_slurm_rank():
    env = {"SLURM_JOB_ID": "1", "SLURM_PROCID": "0"}
    assert ensure_single_process_launch_env(env) is True
    assert "SLURM_PROCID" not in env
    assert "RANK" not in env and env["SLURM_JOB_ID"] == "1"


def test_sbatch_with_ntasks_but_no_srun_drops_the_slurm_rank():
    # sbatch --ntasks=1 exports SLURM_NPROCS without srun; PhysicsNeMo would still need SLURM_LOCALID.
    env = {"SLURM_JOB_ID": "1", "SLURM_PROCID": "0", "SLURM_NPROCS": "1"}
    assert ensure_single_process_launch_env(env) is True
    assert "SLURM_PROCID" not in env


def test_srun_launch_is_left_alone():
    env = {"SLURM_PROCID": "3", "SLURM_NPROCS": "4", "SLURM_LOCALID": "1", "SLURM_LAUNCH_NODE_IPADDR": "launch-node"}
    assert ensure_single_process_launch_env(env) is False
    assert env["SLURM_PROCID"] == "3"


def test_torchrun_launch_is_left_alone():
    env = {"RANK": "2", "WORLD_SIZE": "4", "SLURM_PROCID": "0"}
    assert ensure_single_process_launch_env(env) is False
    assert env["SLURM_PROCID"] == "0" and env["RANK"] == "2"


def test_outside_slurm_nothing_changes():
    env = {}
    assert ensure_single_process_launch_env(env) is False
    assert env == {}


def test_physicsnemo_then_takes_the_single_process_path(monkeypatch):
    # The real DistributedManager, in a bare sbatch environment, on whatever device this machine has.
    pytest = __import__("pytest")
    manager_mod = pytest.importorskip("physicsnemo.sym.distributed.manager")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_LAUNCH_NODE_IPADDR", "SLURM_NPROCS", "SLURM_LOCALID",
                "OMPI_COMM_WORLD_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("NCCL_ASYNC_ERROR_HANDLING", "0")      # initialize() writes it; restored afterwards
    monkeypatch.setattr(manager_mod.DistributedManager, "_shared_state", {})   # a fresh, restored singleton
    import os

    assert ensure_single_process_launch_env() is True
    assert "SLURM_PROCID" not in os.environ
    manager_mod.DistributedManager.initialize()
    manager = manager_mod.DistributedManager()
    assert manager.rank == 0 and manager.world_size == 1 and not manager.distributed


def test_every_entry_point_calls_it_before_physicsnemo():
    from pathlib import Path

    models = Path(__file__).resolve().parents[1] / "src" / "lpno" / "models"
    for name in ("NS_Energy_laser_PINO.py", "NS_Energy_laser_PINO_eval.py", "NS_Energy_laser_PINO_Kfolds.py"):
        src = (models / name).read_text()
        assert "ensure_single_process_launch_env()" in src, name
        assert src.index("ensure_single_process_launch_env()") < src.index("import physicsnemo.sym"), name
