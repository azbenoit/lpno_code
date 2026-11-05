"""The console's GPU panel reads THIS machine. A compute box has cards too.

`gpu_snapshot` shells out to a local `nvidia-smi`, so a job running on another
machine shows no GPU at all — the panel silently describes one box while the
work happens on another. Reading the remote cards goes through the same
transport the log tail uses, with the same TTL cache, because polling an
unreachable box per request is the hazard that deferred item G in the first
place.
"""

from __future__ import annotations

import pytest

from aiida_pino.console.monitor import parse_gpu_csv, remote_gpu_snapshot


def _in_git_checkout() -> bool:
    """True when this test file sits in a git working tree and `git` can read it."""
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("git") is None:
        return False
    try:
        probe = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
    except OSError:
        return False
    return probe.returncode == 0 and probe.stdout.strip() == "true"

GPU_CSV = "0, GPU-aaa, NVIDIA GeForce RTX 4090, 90, 4762, 24564\n1, GPU-bbb, NVIDIA GeForce RTX 4090, 0, 12, 24564\n"
APPS_CSV = "GPU-aaa, 12345, 4700\n"


class _Box:
    """A transport that answers the two nvidia-smi queries."""

    def __init__(self, gpu_txt=GPU_CSV, apps_txt=APPS_CSV, retval=0):
        self.gpu_txt, self.apps_txt, self.retval = gpu_txt, apps_txt, retval
        self.commands = []

    def exec_command_wait(self, command):
        self.commands.append(command)
        if "query-compute-apps" in command:
            return self.retval, self.apps_txt, ""
        return self.retval, self.gpu_txt, ""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_remote_cards_are_read_and_labelled():
    gpus, note = remote_gpu_snapshot(_Box(), hostname="box-a")
    assert note is None
    assert len(gpus) == 2
    assert {g["hostname"] for g in gpus} == {"box-a"}
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 4090"


def test_a_remote_process_is_NOT_enriched_from_local_proc():
    """`proc_info` reads /proc HERE. Using it on a remote pid would attribute
    this machine's command line to another box's GPU process — a wrong answer
    that looks like a right one."""
    gpus, _ = remote_gpu_snapshot(_Box(), hostname="box-a")
    procs = [p for g in gpus for p in g["procs"]]
    assert procs, "the compute app should still be listed"
    for proc in procs:
        assert proc.get("cmd", "") == "", "no local command line may be attached"


def test_an_unreachable_box_returns_a_note_not_an_exception():
    class _Down(_Box):
        def exec_command_wait(self, command):
            raise OSError("connection refused")

    gpus, note = remote_gpu_snapshot(_Down(), hostname="box")
    assert gpus == [] and note and "box" in note


def test_a_box_without_nvidia_smi_reports_why():
    gpus, note = remote_gpu_snapshot(_Box(gpu_txt="", apps_txt="", retval=127), hostname="box")
    assert gpus == [] and note


def test_the_local_parser_is_reused_not_reimplemented():
    """Two parsers for one CSV is how they drift apart."""
    local = parse_gpu_csv(GPU_CSV, APPS_CSV)
    remote, _ = remote_gpu_snapshot(_Box(), hostname="h")
    assert [g["index"] for g in local] == [g["index"] for g in remote]
    assert [g["mem_used_mib"] for g in local] == [g["mem_used_mib"] for g in remote]


REMOTE_PROC = "12345|python -m lpno.train --config x\n"
REMOTE_CWD = "12345|/home/user/aiida_run/ab/cd/ef-1234\n"


class _BoxWithProcs(_Box):
    """A box that can also answer the per-pid cmdline/cwd lookup."""

    def __init__(self, proc_out=REMOTE_PROC + REMOTE_CWD, proc_retval=0, **kw):
        super().__init__(**kw)
        self.proc_out, self.proc_retval = proc_out, proc_retval

    def exec_command_wait(self, command):
        self.commands.append(command)
        if "/proc/" in command:
            return self.proc_retval, self.proc_out, ""
        return super().exec_command_wait(command)


def test_a_remote_process_is_identified_from_the_BOX_that_owns_it():
    """A card showing util with an unnamed process reads as idle to a human.

    That happened: `PID 169217 · WHAT: ?` on a card at 71% util led two readers
    to conclude the box was free while a 32k-step training was at step 7100.
    The pids are already in hand, so the lookup goes to the machine that owns
    them — never to local /proc, which would name a different process entirely.
    """
    gpus, note = remote_gpu_snapshot(_BoxWithProcs(), hostname="box-a")
    procs = [p for g in gpus for p in g["procs"]]
    assert procs[0]["cmd"].startswith("python -m lpno.train")
    assert procs[0]["cwd"] == "/home/user/aiida_run/ab/cd/ef-1234"


def test_the_lookup_runs_on_the_transport_not_locally():
    box = _BoxWithProcs()
    remote_gpu_snapshot(box, hostname="h")
    assert any("/proc/" in c for c in box.commands), "the pid lookup must go over the transport"


def test_a_failed_pid_lookup_degrades_to_unknown_not_to_a_wrong_name():
    """Better no label than another machine's label."""
    gpus, _ = remote_gpu_snapshot(_BoxWithProcs(proc_retval=1, proc_out=""), hostname="h")
    procs = [p for g in gpus for p in g["procs"]]
    assert procs and procs[0].get("cmd", "") == ""


def test_a_card_with_no_processes_asks_the_box_nothing():
    """No pids, no second round trip."""
    box = _BoxWithProcs(apps_txt="")
    remote_gpu_snapshot(box, hostname="h")
    assert not any("/proc/" in c for c in box.commands)


# --- a healthy box must not read as an unavailable one --------------------
# 2026-08-21: the Monitor tab showed "GPUs on box-a unavailable:
# DetachedInstanceError" and a reader took it as the box being down. The box was
# fine -- 7 training processes alive, both arms running, and the same code path
# returned two GPU rows when run outside the web process.
#
# `_remote_gpu_snapshots` captured a live `orm.Computer` into the closure it
# handed the cache (`def _fetch(computer=computer, ...)`). In a long-lived
# console the ORM session behind that object expires between the listing and
# `get_transport()`, which queries authinfo, and the object is detached. Loading
# by LABEL per call removes the whole class -- the closure already carried it.
#
# The second half is the rendering: one message covered "cannot reach the box"
# and "reached it, could not read it", which call for opposite reactions. This
# panel has now produced two wrong impressions in one day from opposite
# directions -- a busy box reading as idle, a healthy box reading as down -- and
# both times the machine was fine and the rendering misled.


class _FakeRunner:
    """A runner whose ssh result is scripted. Replaces the transport fakes.

    The transport is gone by design, so these tests express the same intents
    through the seam that replaced it rather than asserting on a mechanism the
    panel no longer uses.
    """

    def __init__(self, retval=0, out=None, err=""):
        self.retval, self.out, self.err = retval, out, err

    def __call__(self, host):
        self.host = host
        return self

    def exec_command_wait(self, command):
        if self.out is None:
            return self.retval, GPU_CSV if "query-gpu" in command else APPS_CSV, self.err
        return self.retval, self.out, self.err


class _FreshComputer:
    label = "box-a"


def test_the_computer_is_reloaded_per_call_not_captured():
    from aiida_pino.console.monitor import gpu_rows_for_computer

    calls = []

    def loader(label):
        calls.append(label)
        return _FreshComputer()

    rows, note = gpu_rows_for_computer(
        "box-a", load_computer=loader, runner=_FakeRunner()
    )
    assert calls == ["box-a"], "the ORM object must be resolved per call, by label"
    assert note is None
    # Assert on CONTENT, not length. `len(rows) == 2` once passed against a
    # nested `((gpus, note), None)` because the tuple is also length 2, and that
    # bug reached the running console and 500'd /monitor.
    assert isinstance(rows, list), f"rows must be a list of gpu dicts, got {type(rows).__name__}"
    assert [r["index"] for r in rows] == [0, 1]
    assert rows[0]["hostname"] == "box-a"


def test_unreachable_box_and_unreadable_box_say_different_things():
    """A console fault, an unreachable box and a bad reply are three facts."""
    from aiida_pino.console.monitor import gpu_rows_for_computer

    def boom(_label):
        raise RuntimeError("no authinfo")

    _, internal = gpu_rows_for_computer("box-a", load_computer=boom, runner=_FakeRunner())
    # ssh's own failure exit: the box was never reached
    _, unreachable = gpu_rows_for_computer(
        "box-a", load_computer=lambda _l: _FreshComputer(),
        runner=_FakeRunner(retval=255, out=""),
    )
    # answered, but the reply could not be used
    _, unreadable = gpu_rows_for_computer(
        "box-a", load_computer=lambda _l: _FreshComputer(),
        runner=_FakeRunner(retval=2, out=""),
    )

    assert internal and unreachable and unreadable
    assert len({internal, unreachable, unreadable}) == 3, (
        "cannot-reach, reached-but-unreadable and a console-side fault must be "
        "distinguishable; they call for different reactions"
    )
    assert "unreachable" in unreachable.lower()
    assert "not contacted" in internal.lower()
    assert "unreachable" not in internal.lower()


def test_the_traceback_is_logged_even_though_the_panel_stays_terse():
    from aiida_pino.console.monitor import gpu_rows_for_computer

    logged = []

    def boom(_label):
        raise OSError("no authinfo for this computer")

    _, note = gpu_rows_for_computer("box-a", load_computer=boom, log=logged.append)
    assert note and len(note) < 200, "the panel string stays short"
    assert logged, "the traceback must reach a log, or nobody can name the failing line"
    assert "Traceback" in logged[0] or "no authinfo" in logged[0]


# --- a failed process query must not render as "no processes" --------------
# DEFENSIVE, and honestly labelled: this has NOT been observed in production.
# It was written 2026-08-21 after a misreading of the monitor payload (counting the
# `processes` key, which does not exist, instead of `procs`) and believed
# box-b showed 100% utilisation with an empty process list. It did not --
# all four cards report processes. But reading the code for that non-bug found a
# real one: if the compute-apps query exits nonzero, `apps_txt` was set to "" and
# the card rendered with NO processes, indistinguishable from a genuinely idle
# card. Same class as the `PID -> ?` finding earlier the same day -- a failure
# rendering as an absence -- so it is worth closing before it is observed.


def test_a_failed_apps_query_is_not_rendered_as_an_idle_card():
    class _AppsFail(_Box):
        def exec_command_wait(self, command):
            if "query-compute-apps" in command:
                return 9, "", "Function Not Found"
            return 0, self.gpu_txt, ""

    gpus, note = remote_gpu_snapshot(_AppsFail(), hostname="box-b")

    assert gpus, "the utilisation reading is still good and must survive"
    assert note, "a failed process query must be stated, not swallowed"
    assert all(g.get("processes_unknown") for g in gpus), (
        "each card must say its process list is UNKNOWN rather than empty, or a "
        "busy GPU reads as idle"
    )


def test_a_genuinely_idle_card_is_not_marked_unknown():
    """The distinction only means something if the good case stays clean."""
    gpus, note = remote_gpu_snapshot(_Box(apps_txt=""), hostname="box-a")

    assert gpus and note is None
    assert not any(g.get("processes_unknown") for g in gpus), (
        "an apps query that SUCCEEDED and returned nothing means no processes"
    )


def test_a_failure_opening_the_connection_is_not_a_read_failure():
    """`__enter__` is still connecting. Blaming the box's cards is wrong.

    2026-08-21, from the live log: `core.ssh_async` on box-a raised
    `RuntimeError: Cannot run awaitable: event loop is running but no greenback
    portal is available` inside `Transport.__enter__` -> `open()`. The box was
    never contacted, and both cards read fine from a context without a running
    loop. The panel said "reached the box, could not read its cards", which is
    the accusation this classification exists to prevent.
    """
    from aiida_pino.console.monitor import gpu_rows_for_computer

    class _OpenFails:
        label = "box-a"

        def get_transport(self):
            class _CM:
                def __enter__(inner):
                    raise RuntimeError("Cannot run awaitable: event loop is running")

                def __exit__(inner, *a):
                    return False

            return _CM()

    _, note = gpu_rows_for_computer("box-a", load_computer=lambda _l: _OpenFails())
    assert note
    assert "could not read its cards" not in note, (
        "a failure while OPENING the connection must not be reported as the box "
        "answering badly"
    )
    assert "not contacted" in note or "unreachable" in note


# --- the panel shells out; no AiiDA transport, no runner loop --------------
# 2026-08-21: `core.ssh_async` bridges sync->async through greenback, and
# `run_command_blocking` asks `is_running()` of AiiDA's PROCESS-GLOBAL runner
# loop. When anything else in the process has that loop spinning, opening a
# transport raises and the panel loses the box -- measured at 4 failures in 23
# polls (~17 %), clustered when the process is busy, which is when someone is
# most likely to be reading it.
#
# `remote_gpu_snapshot` never needed a transport: it runs two fixed nvidia-smi
# commands and parses CSV. A subprocess `ssh <host> <cmd>` has no runner loop
# and no bridge, so the failure becomes structurally impossible rather than
# rarer. The host comes from authinfo at runtime -- never a literal in the
# source, per the repo's no-machine-specifics rule.


def test_the_ssh_runner_matches_the_transport_interface():
    """It must be a drop-in for `exec_command_wait`, so parsing is untouched."""
    from aiida_pino.console.monitor import SshCommandRunner

    calls = []

    def fake_run(argv):
        calls.append(argv)
        return 0, GPU_CSV if "query-gpu" in " ".join(argv) else APPS_CSV, ""

    runner = SshCommandRunner("box-a", run=fake_run)
    gpus, note = remote_gpu_snapshot(runner, hostname="box-a")

    assert note is None
    assert [g["index"] for g in gpus] == [0, 1]
    assert calls, "it must actually shell out"
    for argv in calls:
        assert argv[0] == "ssh", "must invoke ssh directly, not a shell string"
        assert "box-a" in argv, "the host must come from the caller, not a literal"
        assert "BatchMode=yes" in " ".join(argv), (
            "batch mode, or a console with no tty hangs on an auth prompt instead "
            "of failing"
        )


def test_the_runner_reports_a_failed_command_rather_than_pretending():
    from aiida_pino.console.monitor import SshCommandRunner

    runner = SshCommandRunner("box-a", run=lambda argv: (255, "", "Connection refused"))
    gpus, note = remote_gpu_snapshot(runner, hostname="box-a")
    assert gpus == [] and note, "a failed ssh must surface, not render as an idle box"


@pytest.mark.skipif(not _in_git_checkout(), reason="not a git checkout: the sweep lists files with `git ls-files`")
def test_no_machine_specifics_are_committed_anywhere_in_this_component():
    """No hostnames, usernames or absolute home paths in the console or its tests.

    Scope is the point. The first version of this guard read only `monitor.py`,
    so it passed while the directory around it held a real hostname AND the
    operator's real username -- the latter in this very file, two lines above
    the assertion forbidding it. A guard that inspects one file will pass
    forever while its neighbours accumulate exactly what it forbids.

    The site-specific values (host names, domains, account names) come from the
    local, git-ignored site file that `tools/check_site_leaks.py` reads, so this
    file names none of them and can include itself in the sweep. The username is
    read from the environment, which catches a real leak on any machine and still
    permits deliberate placeholders like `/home/user/...` or `/home/other/...`.
    """
    import getpass
    import pathlib
    import re
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root.parent / "tools"))
    from check_site_leaks import site_literals

    ssh_config_path = ".ssh" + "/config"
    me = getpass.getuser()

    # Every tracked file of the component, no exemptions.
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    targets = [root / p for p in tracked.decode().split("\0") if p]
    assert len(targets) > 300, "the sweep must actually cover the tree"
    scratch_needle = "/tmp/aiida-" + me
    site_needles = [s.lower() for s in site_literals(root.parent) if len(s) >= 4]
    text_suffixes = {".py", ".md", ".sh", ".js", ".html", ".css", ".json", ".toml", ".cfg", ".txt",
                     ".service", ".yml", ".yaml", ".ini", ".ipynb", ".env", ".example", ""}

    offenders = []
    for path in targets:
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        hits = []
        low = text.lower()
        if any(n in low for n in site_needles):
            hits.append("a value from the local site file")
        if re.search(rf"/home/{re.escape(me)}\b", text):
            hits.append("operator username in a path")
        if scratch_needle in text:
            hits.append("operator username in the scratch path")
        if ssh_config_path in text:
            hits.append("ssh config path")
        if hits:
            offenders.append(f"{rel}: {', '.join(hits)}")
    assert not offenders, "machine specifics must not be committed:\n  " + "\n  ".join(offenders)


def test_a_255_carries_the_reason_it_was_given():
    """"ssh exited 255" alone accuses a machine that may be answering fine.

    `SshCommandRunner._subprocess_run` fabricates 255 for a console-side timeout or
    OSError as well as for ssh's own failure, so the code cannot distinguish them --
    only the stderr can. On 2026-08-31 a remote box was missing from the panel for two days
    reading "ssh exited 255" while the box served that exact query in 2.2 s from three
    other contexts, and the discarded stderr was the whole diagnosis.
    """
    from aiida_pino.console.monitor import remote_gpu_snapshot

    class _Runner:
        def exec_command_wait(self, command):
            return 255, "", "console-side TimeoutExpired: timed out after 60s"

    rows, note = remote_gpu_snapshot(_Runner(), hostname="box2")
    assert rows == []
    assert "console-side TimeoutExpired" in note, note


def test_a_255_with_no_stderr_says_it_may_not_be_ssh():
    from aiida_pino.console.monitor import remote_gpu_snapshot

    class _Runner:
        def exec_command_wait(self, command):
            return 255, "", ""

    _, note = remote_gpu_snapshot(_Runner(), hostname="box2")
    assert "console-side" in note, note


def test_a_box_error_still_names_the_box_and_its_reason():
    from aiida_pino.console.monitor import remote_gpu_snapshot

    class _Runner:
        def exec_command_wait(self, command):
            return 9, "", "nvidia-smi: command not found"

    _, note = remote_gpu_snapshot(_Runner(), hostname="box2")
    assert "exited 9" in note and "command not found" in note, note


def test_ssh_host_for_reports_a_failed_authinfo_lookup_instead_of_swallowing_it():
    """A swallowed authinfo error fell back to the computer's bare hostname -- an IP that
    matches no `Host` block in the ssh config, so no key was offered and the panel read
    "Permission denied (publickey)" with nothing in the log to say why (2026-09-14)."""
    from aiida_pino.console.monitor import gpu_rows_for_computer, ssh_host_for

    class _Computer:
        label = "box-a"
        hostname = "192.0.2.10"

        def get_authinfo(self, user):
            raise RuntimeError("session expired")

    logged = []
    host = ssh_host_for(_Computer(), log=logged.append)
    assert host == "192.0.2.10"
    assert len(logged) == 1 and "RuntimeError" in logged[0] and "session expired" in logged[0]

    # The same log callback the panel already threads through reaches the lookup.
    logged.clear()
    seen = {}

    def _runner(host):
        seen["host"] = host
        return object()

    rows, note = gpu_rows_for_computer(
        "box-a", load_computer=lambda label: _Computer(),
        snapshot=lambda transport, hostname: ([], None), log=logged.append, runner=_runner,
    )
    assert seen["host"] == "192.0.2.10" and note is None
    assert len(logged) == 1 and "session expired" in logged[0]
