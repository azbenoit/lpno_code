"""The kill command must be computed on the REMOTE box, never with local psutil.

aiida-core's DirectScheduler._get_kill_command calls `psutil.Process(int(jobid))` in
the daemon process to enumerate a job's descendants. For a remote computer that PID
belongs to a different machine, so it either raises NoSuchProcess (nothing is ever
sent, the job keeps running) or resolves to an UNRELATED local process whose children
are then listed and signalled remotely. Both leave the real trainer running while
AiiDA records the process as killed -- which is why every killed remote run left an
orphan holding a GPU.
"""
import pytest

from aiida_pino.schedulers.direct_pgid import DirectPgidScheduler


def test_kill_command_does_not_touch_the_local_process_table(monkeypatch):
    import psutil

    def explode(*args, **kwargs):
        raise AssertionError("psutil was consulted -- that reads the WRONG machine")

    monkeypatch.setattr(psutil, "Process", explode)
    cmd = DirectPgidScheduler()._get_kill_command(1024282)
    assert "1024282" in cmd


def test_kill_command_targets_the_process_group():
    cmd = DirectPgidScheduler()._get_kill_command(1024282)
    # the whole group, negative-pid form -- one signal reaches all five levels
    assert "pgid" in cmd and "-$" in cmd.replace('"', "").replace("'", "")
    assert "kill" in cmd


def test_kill_command_falls_back_when_the_pid_is_already_gone():
    """The tracked bash can die while its descendants live -- exactly 70052. The
    command must still do something sane rather than expand to `kill -TERM -`."""
    cmd = DirectPgidScheduler()._get_kill_command(999999)
    assert "999999" in cmd
    assert "||" in cmd or "else" in cmd, "no fallback when ps returns an empty pgid"


@pytest.mark.parametrize("jobid", ["1024282", 1024282])
def test_accepts_str_and_int_jobids(jobid):
    assert "1024282" in DirectPgidScheduler()._get_kill_command(jobid)


def test_rejects_a_non_numeric_jobid():
    """The jobid is interpolated into a shell command; anything but digits is a
    command-injection vector, so it must be refused rather than quoted-and-hoped."""
    with pytest.raises(ValueError):
        DirectPgidScheduler()._get_kill_command("1; rm -rf ~")
