"""Tests for the per-GPU training spooler (``lpno.gpu_spool``).

GPU-free and license-free: jobs are trivial shell commands (``true`` / ``exit`` /
``printenv``), never real training. Covers enqueue routing, auto GPU selection,
GPU-override rejection, worker drain (done/failed/crash-requeue), the child-env
GPU pinning, the per-GPU worker/seat lock exclusivity, and the VRAM ledger that
decides how many runs share a card.

Every test redirects the shared ledger (``PINO_GPU_LEASE_DIR``) into its own
tmp_path and pins the budget (``PINO_SPOOL_VRAM_MB``), so nothing here reads
``nvidia-smi`` or touches the real /var/tmp ledger other users book against.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from lpno import gpu_spool as gs

BUDGET_MB = 1000


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PINO_GPU_LEASE_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("PINO_SPOOL_VRAM_MB", str(BUDGET_MB))
    monkeypatch.setenv("PINO_SPOOL_CHECK_FREE", "0")  # never let a real GPU's state decide a test
    monkeypatch.setenv("PINO_SPOOL_DEFAULT_RAM_MB", "0")  # nor the box's real free RAM
    monkeypatch.delenv("PINO_SPOOL_DEFAULT_VRAM_MB", raising=False)


def _tasks(spool: Path, gpu: int, sub: str) -> list[Path]:
    return list((spool / f"gpu{gpu}" / sub).glob("*.task"))


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_enqueue_targets_named_gpu_queue(tmp_path: Path):
    gs.enqueue(tmp_path, 1, "job", "true")
    tasks = _tasks(tmp_path, 1, "queue")
    assert len(tasks) == 1
    payload = _payload(tasks[0])
    assert payload["name"] == "job"
    assert payload["cmd"] == "true"
    assert payload["gpu"] == 1
    assert not _tasks(tmp_path, 0, "queue")  # not cross-posted to gpu0


def test_select_gpu_auto_picks_least_loaded(tmp_path: Path):
    gs.enqueue(tmp_path, 0, "a", "true")
    gs.enqueue(tmp_path, 0, "b", "true")
    gs.enqueue(tmp_path, 1, "c", "true")
    assert gs.select_gpu_auto(tmp_path, [0, 1]) == 1


def test_select_gpu_auto_breaks_tie_to_lowest_index(tmp_path: Path):
    assert gs.select_gpu_auto(tmp_path, [0, 1]) == 0


def test_auto_seat_selects_an_admissible_gpu_and_injects_cuda_override(tmp_path: Path, monkeypatch):
    held = gs.reserve(tmp_path, 0, "busy", BUDGET_MB, ram_mb=0)
    assert held is not None
    calls = {}

    class FakeProc:
        pid = 4242
        def poll(self):
            return 0

    def fake_popen(command, cwd=None, env=None):
        calls["command"] = command
        calls["env"] = env
        return FakeProc()

    # the seat runner moved from subprocess.call to Popen + a booking-reconcile
    # monitor (2026-09-09); the assertions below are unchanged -- only the seam is.
    monkeypatch.setattr(gs.subprocess, "Popen", fake_popen)
    try:
        rc = gs._run_seat(tmp_path, "auto", ["python", "-m", "payload"],
                          vram_mb=BUDGET_MB, ram_mb=0, gpus=[0, 1])
    finally:
        gs._drop(*held)

    assert rc == 0
    assert calls["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert calls["command"][-1] == "custom.cuda_device=1"


def test_auto_seat_does_not_duplicate_existing_cuda_override(tmp_path: Path, monkeypatch):
    calls = {}

    class FakeProc:
        pid = 4242
        def poll(self):
            return 0

    def fake_popen(command, cwd=None, env=None):
        calls["command"] = command
        return FakeProc()

    monkeypatch.setattr(gs.subprocess, "Popen", fake_popen)  # same seam move as above

    rc = gs._run_seat(tmp_path, "auto", ["python", "-m", "payload", "custom.cuda_device=7"],
                      vram_mb=10, ram_mb=0, gpus=[0, 1])

    assert rc == 0
    assert calls["command"].count("custom.cuda_device=7") == 1
    assert not any(arg == "custom.cuda_device=0" for arg in calls["command"])


def test_validate_rejects_embedded_gpu_override():
    for bad in (
        "python -m x custom.cuda_device=1",
        "python -m x +custom.cuda_device=1",
        "foo cuda_device=0",
        "bar custom.cuda_visible_devices=1,0",
        "baz cuda_visible_devices=0",
    ):
        with pytest.raises(SystemExit):
            gs.validate_no_gpu_override(bad)
    # a command without a GPU selector is fine
    gs.validate_no_gpu_override("python -m x custom.name=ok")


def test_build_command_default_module():
    # The interpreter is pinned to sys.executable, never a bare "python": see
    # test_build_command_uses_the_running_interpreter for why.
    import sys

    cmd = gs.build_command("NS_Energy_laser_PINO", ["custom.name=run1", "custom.epochs=1"])
    assert cmd == (
        f"{sys.executable} -u -m lpno.models.NS_Energy_laser_PINO "
        "custom.name=run1 custom.epochs=1"
    )


def test_worker_drains_job_to_done_and_pins_gpu(tmp_path: Path):
    # The job echoes the GPU the worker pinned it to via CUDA_VISIBLE_DEVICES.
    gs.enqueue(tmp_path, 1, "pintest", "printenv CUDA_VISIBLE_DEVICES")
    gs.run_worker(tmp_path, 1, drain=True, poll=0.01)

    assert _tasks(tmp_path, 1, "done")
    assert not _tasks(tmp_path, 1, "queue")
    log = (tmp_path / "gpu1" / "logs" / "pintest.log").read_text(encoding="utf-8").strip()
    assert log == "1"  # pinned to the queue's GPU, not the host default
    assert "\tdone\tpintest\t0" in (tmp_path / "gpu1" / "pool.log").read_text(encoding="utf-8")


def test_worker_marks_failing_job_failed(tmp_path: Path):
    gs.enqueue(tmp_path, 0, "boom", "exit 3")
    gs.run_worker(tmp_path, 0, drain=True, poll=0.01)

    assert _tasks(tmp_path, 0, "failed")
    assert "\tfailed\tboom\t3" in (tmp_path / "gpu0" / "pool.log").read_text(encoding="utf-8")


def test_worker_requeues_crashed_running_task(tmp_path: Path):
    # A task left in running/ by a dead worker must be re-run, not lost.
    gs.ensure_dirs(tmp_path, 0)
    stale = tmp_path / "gpu0" / "running" / "00000000000000000000-1.task"
    stale.write_text(json.dumps({"name": "stale", "cmd": "true", "gpu": 0}), encoding="utf-8")

    gs.run_worker(tmp_path, 0, drain=True, poll=0.01)

    assert _tasks(tmp_path, 0, "done")
    assert not _tasks(tmp_path, 0, "running")


def test_second_worker_same_gpu_is_refused(tmp_path: Path):
    gs.ensure_dirs(tmp_path, 0)
    held = gs._acquire_lock(tmp_path / "gpu0" / ".worker.lock")
    assert held is not None
    try:
        with pytest.raises(SystemExit):
            gs.run_worker(tmp_path, 0, drain=True, poll=0.01)
    finally:
        gs._release_lock(held)


def test_budget_leaves_headroom_under_the_physical_total(monkeypatch):
    # The budget must sit below what the card physically has: the rest absorbs
    # allocator fragmentation and anything on the card outside the ledger.
    monkeypatch.delenv("PINO_SPOOL_VRAM_MB")
    monkeypatch.setenv("PINO_SPOOL_VRAM_HEADROOM_MB", "2048")
    monkeypatch.setattr(gs, "_gpu_total_mb", lambda gpu: 24576)
    assert gs.vram_budget_mb(0) == 24576 - 2048


def test_reservations_pack_until_the_budget_is_full(tmp_path: Path):
    a = gs.reserve(tmp_path, 0, "a", 400)
    b = gs.reserve(tmp_path, 0, "b", 400)
    assert a is not None and b is not None  # two small runs share the card
    assert gs.reserved_mb(0) == 800
    assert gs.reserve(tmp_path, 0, "c", 400) is None  # 1200 > 1000: refused
    gs._drop(*a)
    assert gs.reserved_mb(0) == 400
    c = gs.reserve(tmp_path, 0, "c", 400)  # room again
    assert c is not None
    gs._drop(*b)
    gs._drop(*c)


def test_undeclared_run_takes_the_card_exclusively(tmp_path: Path):
    assert gs.default_vram_mb(0) is None  # None == "the whole card"
    held = gs.reserve(tmp_path, 0, "big", gs.default_vram_mb(0))
    assert held is not None
    assert gs.reserve(tmp_path, 0, "other", 1) is None  # exclusive, as before
    gs._drop(*held)
    assert gs.reserve(tmp_path, 0, "small", 1) is not None  # ... only while it holds


def test_house_default_lets_undeclared_runs_pack(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PINO_SPOOL_DEFAULT_VRAM_MB", "400")
    assert gs.default_vram_mb(0) == 400
    a = gs.reserve(tmp_path, 0, "a", gs.default_vram_mb(0))
    b = gs.reserve(tmp_path, 0, "b", gs.default_vram_mb(0))
    assert a is not None and b is not None
    gs._drop(*a)
    gs._drop(*b)


def test_exclusive_claim_ignores_the_free_memory_floor(tmp_path: Path, monkeypatch):
    # An exclusive run asks for "no company", not for a number of MiB. Holding it
    # to the physical floor would let any stray notebook block whole-card runs
    # (and CI, which reserves exclusively) for ever.
    monkeypatch.setattr(gs, "free_mb", lambda gpu: BUDGET_MB - 500)
    held = gs.reserve(tmp_path, 0, "ci", None)
    assert held is not None
    gs._drop(*held)


def test_oversized_request_is_clamped_not_deadlocked(tmp_path: Path):
    # A run asking for more than the card has must still be admitted alone,
    # otherwise it would sit in the queue for ever.
    held = gs.reserve(tmp_path, 0, "huge", BUDGET_MB * 10)
    assert held is not None
    assert gs.reserved_mb(0) == BUDGET_MB
    gs._drop(*held)


def test_lease_of_a_dead_holder_is_not_counted(tmp_path: Path):
    # A crashed run leaves its lease file behind; nobody holds the flock, so the
    # memory must go back into the budget instead of blocking the card for ever.
    stale = gs.lease_dir(0) / "999999-1.lease"
    stale.write_text(json.dumps({"name": "ghost", "vram_mb": 900}), encoding="utf-8")
    assert gs.reserved_mb(0) == 0
    held = gs.reserve(tmp_path, 0, "next", 900)
    assert held is not None
    gs._drop(*held)


def test_unledgered_memory_on_the_card_blocks_a_reservation(tmp_path: Path, monkeypatch):
    # A notebook or another user's process holds the card but books nothing. The
    # ledger alone would happily hand the memory out, so free memory is a floor.
    monkeypatch.setattr(gs, "free_mb", lambda gpu: 200)
    assert gs.reserve(tmp_path, 0, "toobig", 400) is None
    small = gs.reserve(tmp_path, 0, "fits", 150)
    assert small is not None
    gs._drop(*small)


def test_free_memory_check_is_off_when_it_cannot_be_read(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gs, "free_mb", lambda gpu: None)  # no nvidia-smi
    held = gs.reserve(tmp_path, 0, "anyway", 400)
    assert held is not None  # ledger still governs; a missing reading is not a refusal
    gs._drop(*held)


def test_host_ram_booking_refuses_a_run_the_box_cannot_feed(tmp_path: Path, monkeypatch):
    # The 2026-07-20 CI failure: the time-as-batch smoke peaks at ~86 GB RSS and
    # was SIGKILLed because a 27 GB training already held the box. VRAM was never
    # the constraint, so only a host-RAM booking can hold that run back.
    monkeypatch.setenv("PINO_SPOOL_RAM_HEADROOM_MB", "8192")
    monkeypatch.setattr(gs, "ram_available_mb", lambda: 90000)
    assert gs.reserve(tmp_path, 0, "big", 100, ram_mb=92000) is None
    ok = gs.reserve(tmp_path, 0, "fits", 100, ram_mb=40000)
    assert ok is not None
    gs._drop(*ok)


def test_host_ram_bookings_add_up_across_cards(tmp_path: Path, monkeypatch):
    # RAM is machine-wide: a run booked on GPU 1 must count against GPU 0's next
    # admission, or two runs on different cards happily OOM the box together.
    #
    # 2026-09-07: the invariant is unchanged, but it is now enforced against MemTotal
    # rather than MemAvailable, so the test has to say how big the machine is. It never
    # did before, because the old check summed bookings against a MemAvailable this
    # fixture pins at a constant -- which is also what deadlocked the real box, where
    # MemAvailable drops as a run allocates and the booking was then counted twice.
    # Mocking total at the same 100000 keeps the original meaning: two 60000 bookings
    # do not fit one machine.
    monkeypatch.setenv("PINO_SPOOL_RAM_HEADROOM_MB", "0")
    monkeypatch.setattr(gs, "ram_available_mb", lambda: 100000)
    monkeypatch.setattr(gs, "ram_total_mb", lambda: 100000)
    first = gs.reserve(tmp_path, 1, "on_gpu1", 100, ram_mb=60000)
    assert first is not None
    assert gs.booked_ram_mb() == 60000
    assert gs.reserve(tmp_path, 0, "on_gpu0", 100, ram_mb=60000) is None
    gs._drop(*first)
    second = gs.reserve(tmp_path, 0, "on_gpu0", 100, ram_mb=60000)
    assert second is not None
    gs._drop(*second)


def test_undeclared_host_ram_books_the_house_default(tmp_path: Path, monkeypatch):
    # BUG-43: booking nothing left an undeclared run invisible to the next
    # admission, so a 27 GB training read as free RAM. A run that says nothing
    # now books the house default and shows up in everyone else's arithmetic.
    monkeypatch.setenv("PINO_SPOOL_DEFAULT_RAM_MB", "32768")
    monkeypatch.setattr(gs, "ram_available_mb", lambda: 100000)
    held = gs.reserve(tmp_path, 0, "quiet", 100)
    assert held is not None
    assert gs.booked_ram_mb() == 32768
    gs._drop(*held)


def test_ram_mb_zero_opts_a_small_run_out(tmp_path: Path, monkeypatch):
    # The escape hatch for a genuinely small run, and the reason 0 and None
    # cannot be the same value.
    monkeypatch.setenv("PINO_SPOOL_DEFAULT_RAM_MB", "32768")
    monkeypatch.setattr(gs, "ram_available_mb", lambda: 1)
    held = gs.reserve(tmp_path, 0, "tiny", 100, ram_mb=0)
    assert held is not None
    assert gs.booked_ram_mb() == 0
    gs._drop(*held)


def test_priority_marker_holds_back_new_runs(tmp_path: Path):
    # What CI relies on: park a want marker, and the workers stop starting runs
    # so the priority job goes in ahead of the queue, not behind it.
    want_path = gs._entry_path(0, ".want")
    want = gs._hold(want_path, {"name": "ci", "vram_mb": 0})
    assert gs.reserve(tmp_path, 0, "queued", 100) is None
    mine = gs.reserve(tmp_path, 0, "ci", 100, priority=True)
    assert mine is not None
    gs._drop(want, want_path)
    assert gs.reserve(tmp_path, 0, "queued", 100) is not None  # marker gone: back to normal
    gs._drop(*mine)


def test_legacy_seat_lock_still_blocks_the_card(tmp_path: Path):
    # A seat process started before the ledger existed holds .seat.lock only.
    gs.ensure_dirs(tmp_path, 0)
    held = gs._acquire_lock(tmp_path / "gpu0" / ".seat.lock")
    assert held is not None
    try:
        assert gs.reserve(tmp_path, 0, "new", 1) is None
    finally:
        gs._release_lock(held)
    assert gs.reserve(tmp_path, 0, "new", 1) is not None


def test_worker_runs_two_small_jobs_at_once(tmp_path: Path, monkeypatch):
    # The end-to-end claim: with declared VRAM that fits, both jobs are in flight
    # together. Each job blocks until the gate file appears, so this proves
    # overlap without depending on timing.
    monkeypatch.setattr(gs.signal, "signal", lambda *a, **k: None)  # worker runs off-thread
    gate = tmp_path / "gate"
    started = tmp_path / "started"
    job = f"echo x >> {started}; while [ ! -f {gate} ]; do sleep 0.01; done"
    gs.enqueue(tmp_path, 0, "j1", job, vram_mb=400)
    gs.enqueue(tmp_path, 0, "j2", job, vram_mb=400)

    worker = threading.Thread(target=gs.run_worker, args=(tmp_path, 0),
                              kwargs={"drain": True, "poll": 0.01}, daemon=True)
    worker.start()
    deadline = time.time() + 30
    while time.time() < deadline:
        if started.exists() and len(started.read_text().split()) == 2:
            break
        time.sleep(0.05)
    both_up = started.exists() and len(started.read_text().split()) == 2
    gate.write_text("go")
    worker.join(timeout=30)

    assert both_up, "second job never started while the first was running"
    assert len(_tasks(tmp_path, 0, "done")) == 2
    assert gs.reserved_mb(0) == 0  # leases released when the jobs finished


def test_worker_keeps_a_big_job_alone(tmp_path: Path, monkeypatch):
    # Same setup, but the first job books the whole card: the second must wait.
    monkeypatch.setattr(gs.signal, "signal", lambda *a, **k: None)
    gate = tmp_path / "gate"
    started = tmp_path / "started"
    job = f"echo x >> {started}; while [ ! -f {gate} ]; do sleep 0.01; done"
    gs.enqueue(tmp_path, 0, "big", job, vram_mb=BUDGET_MB)
    gs.enqueue(tmp_path, 0, "small", job, vram_mb=100)

    worker = threading.Thread(target=gs.run_worker, args=(tmp_path, 0),
                              kwargs={"drain": True, "poll": 0.01}, daemon=True)
    worker.start()
    deadline = time.time() + 10
    while time.time() < deadline and not started.exists():
        time.sleep(0.05)
    time.sleep(0.5)  # give a wrongly-admitted second job time to show up
    running_alone = started.read_text().split() == ["x"]
    gate.write_text("go")
    worker.join(timeout=30)

    assert running_alone, "a second job started on a fully reserved card"
    assert len(_tasks(tmp_path, 0, "done")) == 2  # the waiter still ran, after


def test_seat_lock_is_exclusive(tmp_path: Path):
    gs.ensure_dirs(tmp_path, 0)
    seat = tmp_path / "gpu0" / ".seat.lock"
    first = gs._acquire_lock(seat)
    assert first is not None
    assert gs._acquire_lock(seat) is None  # non-blocking second acquire fails
    gs._release_lock(first)
    again = gs._acquire_lock(seat)
    assert again is not None  # released → free again
    gs._release_lock(again)


def test_build_command_uses_the_running_interpreter():
    """A bare 'python' resolves to whatever is first on the worker's PATH.

    The worker runs the command through `bash -lc`, whose login profile need not
    have the conda env active, so a bare `python` imported the system
    interpreter and every `add --module` task died at
    `ModuleNotFoundError: No module named 'physicsnemo'` (2026-08-08, both
    rigno2579 arms). The queued command must name the interpreter that enqueued
    it.
    """
    import sys

    from lpno import gpu_spool

    cmd = gpu_spool.build_command("NS_Energy_laser_PINO", ["custom.name=x"])
    assert cmd.startswith(sys.executable + " "), cmd
    assert " -m lpno.models.NS_Energy_laser_PINO " in cmd


def _training_cmd(name="arm1"):
    return ("cd /repo && python -u -m lpno.models.NS_Energy_laser_PINO "
            f"--config-name=cfg custom.name={name} training.max_steps=32000")


def test_training_submission_is_refused_without_an_explicit_opt_out():
    """Trainings belong in AiiDA; the spool must not be the easy way around it.

    2026-08-08..10: four trainings and three evals were launched straight
    through the spool and are absent from the provenance store. The compliant
    path could have carried them (training_profile takes `env` and
    `hydra_overrides`); nothing stopped the shortcut.
    """
    with pytest.raises(SystemExit) as exc:
        gs.validate_aiida_or_optout(_training_cmd(), no_aiida=False)
    msg = str(exc.value)
    assert "AiiDA" in msg and "--no-aiida" in msg


def test_explicit_opt_out_allows_it():
    # the escape hatch for a user without AiiDA configured
    gs.validate_aiida_or_optout(_training_cmd(), no_aiida=True)


def test_short_smokes_and_non_training_commands_are_never_blocked():
    # a 2-step smoke must never be registered, so it must never be refused
    gs.validate_aiida_or_optout(_training_cmd() + " training.max_steps=2", no_aiida=False)
    # and the reverse order must NOT read as a smoke (Hydra takes the last)
    with pytest.raises(SystemExit):
        gs.validate_aiida_or_optout(
            _training_cmd().replace("max_steps=32000", "max_steps=2")
            + " training.max_steps=32000", no_aiida=False)
    gs.validate_aiida_or_optout("python -m lpno.gpu_spool status", no_aiida=False)
    gs.validate_aiida_or_optout("nvidia-smi", no_aiida=False)


def test_eval_submissions_are_guarded_too():
    cmd = ("cd /repo && python -u -m lpno.models.NS_Energy_laser_PINO_eval "
           "--config-name=cfg custom.name=e1 network_dir=/x")
    with pytest.raises(SystemExit):
        gs.validate_aiida_or_optout(cmd, no_aiida=False)


def test_a_live_booking_does_not_deadlock_admission(tmp_path: Path, monkeypatch):
    """The 2026-09-07 deadlock: nothing at all could be admitted behind a live booking.

    `ram_available_mb` is MemAvailable, which ALREADY discounts every running process --
    its own docstring says so. Adding the full `booked_ram_mb()` on top charged a running
    run's memory twice, once inside `avail` and once as its booking. Measured on the box:
    avail 77821, booked 65536, headroom 8192 -> the largest admissible request was 4093 MiB
    while 78 GB was genuinely free and only ~47 GB in use. Every queued arm wanted 45-64 GB,
    so the queue could not drain until the booking exited, whatever its size.

    The booking still has to count for the window between booking and allocating -- that is
    what `test_host_ram_bookings_add_up_across_cards` protects and it must keep passing --
    so the two questions are now asked separately against the figure each one suits:
    can this request fit right now (MemAvailable), and do all commitments fit at all (MemTotal).
    """
    monkeypatch.setenv("PINO_SPOOL_RAM_HEADROOM_MB", "8192")
    monkeypatch.setattr(gs, "ram_available_mb", lambda: 77821)
    monkeypatch.setattr(gs, "ram_total_mb", lambda: 128000)
    live = gs.reserve(tmp_path, 0, "already_running", 100, ram_mb=65536)
    assert live is not None
    assert gs.booked_ram_mb() == 65536
    # 40960 fits in MemAvailable (40960 <= 77821 - 8192) and total commitments
    # (40960 + 65536 = 106496 <= 128000 - 8192), so it must be admitted.
    second = gs.reserve(tmp_path, 0, "fits_alongside", 100, ram_mb=40960)
    assert second is not None, "a request that fits both MemAvailable and MemTotal was refused"
    gs._drop(*second)
    gs._drop(*live)


def test_total_commitments_still_bounded_by_physical_ram(tmp_path: Path, monkeypatch):
    """The half that must NOT be lost: bookings still cannot exceed the machine.

    Without this, splitting the check would let two runs each pass the MemAvailable test
    in the window before either allocates -- the 2026-07-20 OOM.
    """
    monkeypatch.setenv("PINO_SPOOL_RAM_HEADROOM_MB", "8192")
    monkeypatch.setattr(gs, "ram_available_mb", lambda: 120000)  # nothing allocated yet
    monkeypatch.setattr(gs, "ram_total_mb", lambda: 128000)
    first = gs.reserve(tmp_path, 0, "big_a", 100, ram_mb=70000)
    assert first is not None
    # 70000 + 70000 = 140000 > 128000 - 8192, so the second must be refused even though
    # MemAvailable alone would still allow it (70000 <= 120000 - 8192).
    assert gs.reserve(tmp_path, 1, "big_b", 100, ram_mb=70000) is None
    gs._drop(*first)
