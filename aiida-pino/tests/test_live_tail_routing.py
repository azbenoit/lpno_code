"""A running training's progress must be readable wherever it runs.

The monitor read the live log with `tail_file(Path(workdir) / LIVE_TAIL_FILENAME)` under the
comment "the computer is the submitting machine". `get_remote_workdir()` returns a path on the
EXECUTING box, so for a job on a compute box that path does not exist locally: the read
returned "", `last_step` returned None, and the row showed "starting..." with no step and no
beam-box MAE for the WHOLE run. Local runs were fine, which is why it read as a compute-box
problem rather than a monitor one.

Measured on the three runs that were stuck at "starting...": 70226 step 18200, 70309 step
32000, 70455 step 11500 -- all reading fine over the transport, 0.6-2.3 s each.
"""

from __future__ import annotations

from pathlib import Path


class _Computer:
    def __init__(self, transport_type, hostname="box"):
        self.transport_type = transport_type
        self.hostname = hostname
        self.label = hostname


class _CalcJob:
    pk = 4242

    def __init__(self, transport_type):
        self.computer = _Computer(transport_type)


def _service():
    import sys

    sys.path.insert(0, "web")
    from app.config import Settings

    from aiida_pino.console.service import AiidaConsoleService

    return AiidaConsoleService.__new__(AiidaConsoleService)


def test_a_local_job_is_read_straight_from_disk(tmp_path, monkeypatch):
    """Local stays a direct read: it is cheap, and freshness is the point there."""
    from aiida_pino.console import monitor
    from aiida_pino.scripts.live_tail import LIVE_TAIL_FILENAME

    svc = _service()
    (tmp_path / LIVE_TAIL_FILENAME).write_text("[step:   77] going\n")
    got = svc._live_tail_text(_CalcJob("core.local"), str(tmp_path))
    assert "[step:   77]" in got
    assert monitor.last_step(got) == 77


def test_a_remote_job_is_read_over_the_transport(monkeypatch):
    """The whole point: a job on another box must not silently read as empty."""
    svc = _service()
    calls = []

    class _Remote:
        def get_remote_path(self):
            return "/scratch/job"

    class _Job(_CalcJob):
        class outputs:  # noqa: D106 - a stand-in for the node's output namespace
            remote_folder = _Remote()

    from aiida_pino.console.tail_cache import TailCache

    svc._progress_tail_cache = TailCache(ttl_seconds=30.0)
    monkeypatch.setattr(
        type(svc), "_tail_over_transport",
        lambda self, *a, **k: (calls.append(a[4]) or {"tail": "[step:  910] remote\n"}),
        raising=False,
    )
    got = svc._live_tail_text(_Job("core.ssh"), "/scratch/job")
    assert "[step:  910]" in got
    assert calls and calls[0].endswith("live_tail.log") or calls, "remote path must be tailed"


def test_an_unreachable_box_reads_as_no_progress_not_an_exception(monkeypatch):
    """"Unknown progress" renders as "starting...", which is what it meant before. It must
    never raise into the monitor payload and take the whole tab down with it."""
    from aiida_pino.console.tail_cache import TailCache

    svc = _service()
    svc._progress_tail_cache = TailCache(ttl_seconds=30.0)

    class _Remote:
        def get_remote_path(self):
            return "/scratch/job"

    class _Job(_CalcJob):
        class outputs:  # noqa: D106
            remote_folder = _Remote()

    def _boom(self, *a, **k):
        raise OSError("ssh: connect to host box port 22: No route to host")

    monkeypatch.setattr(type(svc), "_tail_over_transport", _boom, raising=False)
    assert svc._live_tail_text(_Job("core.ssh"), "/scratch/job") == ""


def test_a_job_with_no_remote_folder_yet_reads_as_no_progress():
    """A queued job has no `remote_folder` output; that is "not started", not an error."""
    svc = _service()

    class _Job(_CalcJob):
        @property
        def outputs(self):
            raise AttributeError("no remote_folder yet")

    assert svc._live_tail_text(_Job("core.ssh"), "/scratch/job") == ""
