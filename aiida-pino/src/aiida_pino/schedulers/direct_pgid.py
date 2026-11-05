"""A DirectScheduler whose kill command is evaluated on the machine running the job.

Why this exists
---------------
`aiida.schedulers.plugins.direct.DirectScheduler._get_kill_command` builds its kill
list with local psutil::

    process = Process(int(jobid))
    children = process.children(recursive=True)

That inspects the process table of whichever machine runs the *daemon*. For a remote
computer the job's PID lives on the other box, so one of two things happens:

* the PID does not exist locally -- `NoSuchProcess` propagates and no kill command is
  ever sent, while AiiDA still records the process as killed; or
* the PID happens to exist locally as something unrelated -- its children are
  enumerated and that list is signalled *on the remote box*, killing the tracked
  shell and, potentially, unrelated remote processes, while the job's real
  descendants survive.

Either way the training keeps running, holds its GPU, and writes no model node. On
2026-08-26 this had left three orphans (70037, 70052, 70203); the daemon log carries
the matching `NoSuchProcess: process PID not found`.

A PINO training is a five-process chain -- `bash _aiidasubmit.sh` -> `train_adapter.py`
-> `gpu_spool seat` -> `/usr/bin/time` -> the trainer -- and all five share one process
group. Signalling that group is a single remote command that needs no knowledge of the
tree, so this override asks the remote shell for the group and kills it there.
"""

from __future__ import annotations

from aiida.schedulers.plugins.direct import DirectScheduler

__all__ = ("DirectPgidScheduler",)


class DirectPgidScheduler(DirectScheduler):
    """`core.direct`, but the kill is resolved where the process actually lives."""

    def _get_submit_command(self, submit_script):
        """Launch the job as its own session leader, so PGID == the recorded job id.

        Upstream runs ``(bash script > /dev/null 2>&1 & echo $!) &``. The PID it
        records is the shell's, which is *not* the process-group leader -- measured
        here, job 1136625 sat in group 1136618 -- so the group cannot be recovered
        from the job id once that shell exits. ``setsid`` execs without forking when
        the caller is not already a group leader (true inside ``( ... & )``), which
        makes the recorded PID the leader of both the session and the group.
        """
        submit_command = f'(setsid bash {submit_script} > /dev/null 2>&1 & echo $!) &'
        self.logger.info(f'submitting with: {submit_command}')
        return submit_command

    def _get_kill_command(self, jobid) -> str:
        """Kill the job's whole process group, resolved on the execution host.

        The jobid is interpolated into a shell command, so it is required to be
        digits: a scheduler job id that is not a PID is not something to quote and
        hope about.
        """
        job = str(jobid).strip()
        if not job.isdigit():
            raise ValueError(f"refusing to build a kill command for a non-numeric job id: {jobid!r}")

        self.logger.info(f"killing job {job} and its process group (resolved remotely)")

        # Three cases, in order:
        #  1. leader alive -- ask it for its group and signal that. Covers jobs
        #     submitted before this plugin, whose PGID differs from the job id.
        #  2. leader gone, submitted through _get_submit_command above -- the group
        #     id still equals the job id and survives as long as any member lives.
        #     This is the case that matters: it is how an orphan is left behind.
        #  3. neither -- signal the bare pid so the command is never a no-op by
        #     construction.
        return (
            f'pgid=$(ps -o pgid= -p {job} 2>/dev/null | tr -d " "); '
            f'if [ -n "$pgid" ]; then kill -TERM -- -"$pgid"; '
            f'else kill -TERM -- -{job} 2>/dev/null || kill -TERM {job} 2>/dev/null; fi'
        )
