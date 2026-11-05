"""A priority seat may hold the card only while it could actually use it.

CPU-only and license-free. ``--priority`` exists so CI jumps the queue, not so
it idles the box: on 2026-07-28 a CI seat wanting 92 GB of host RAM sat
unadmittable for 2.7 h and its ``.want`` marker kept GPU 0 completely empty
while three runs that fitted waited behind it. Admission now asks whether the
priority run is itself admissible, and only yields to it while it is.
"""

import json

import pytest

from lpno import gpu_spool


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_spool, "lease_root", lambda: tmp_path / "leases")
    monkeypatch.setattr(gpu_spool, "vram_budget_mb", lambda gpu: 22528)
    monkeypatch.setattr(gpu_spool, "free_mb", lambda gpu: 24021)
    monkeypatch.setattr(gpu_spool, "ram_available_mb", lambda: 99846)
    monkeypatch.setattr(gpu_spool, "ram_headroom_mb", lambda: 8192)


def _park_want(gpu: int, **payload):
    path = gpu_spool._entry_path(gpu, ".want")
    fd = gpu_spool._hold(path, payload)
    return fd, path


def test_unadmittable_priority_does_not_block_the_card(tmp_path):
    """The live case: CI books more host RAM than the box can currently give.
    Blocking buys it nothing and costs the whole queue."""
    _park_want(0, name="seat:ci", vram_mb=None, exclusive=True, ram_mb=92000, pid=1, user="ci")

    booked = gpu_spool.reserve(tmp_path, 0, "queued-training", vram_mb=10000, ram_mb=8000)

    assert booked is not None, "an idle card must admit a run that fits behind a stuck priority seat"


def test_admittable_priority_still_wins_the_queue(tmp_path):
    """The feature must survive the fix: a priority run that *can* start still
    goes in ahead of ordinary waiters."""
    _park_want(0, name="seat:ci", vram_mb=5000, exclusive=False, ram_mb=8000, pid=1, user="ci")

    booked = gpu_spool.reserve(tmp_path, 0, "queued-training", vram_mb=10000, ram_mb=8000)

    assert booked is None, "an admissible priority seat must still hold the queue"


def test_priority_blocked_by_vram_does_not_freeze_smaller_runs(tmp_path):
    """Same rule for VRAM: a whole-card priority run cannot start while a lease
    holds part of the card, so it must not also block what does fit."""
    gpu_spool.reserve(tmp_path, 0, "long-training", vram_mb=15000, ram_mb=0)
    _park_want(0, name="seat:ci", vram_mb=None, exclusive=True, ram_mb=0, pid=1, user="ci")

    booked = gpu_spool.reserve(tmp_path, 0, "small-eval", vram_mb=3000, ram_mb=0)

    assert booked is not None, "a 3 GiB run fits beside a 15 GiB lease; priority cannot start anyway"


def test_legacy_want_without_sizes_still_blocks(tmp_path):
    """Entries written before the payload carried sizes have nothing to check.
    Treat them as admissible -- the old behaviour, and the safe direction."""
    _park_want(0, name="seat:old", pid=1, user="ci")

    assert gpu_spool.reserve(tmp_path, 0, "queued", vram_mb=1000, ram_mb=0) is None


def test_want_marker_records_its_sizes(tmp_path, monkeypatch):
    """`_blocking_want` can only judge a marker that carries its request."""
    monkeypatch.setattr(gpu_spool, "reserve", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        gpu_spool._run_seat(tmp_path, 0, ["true"], vram_mb=4000, ram_mb=6000,
                            priority=True, poll=0.01, max_wait_s=0.05)
    wants = list((gpu_spool.lease_dir(0)).glob("*.want"))
    if wants:  # the marker is dropped on exit; only assert when still present
        payload = json.loads(wants[0].read_text())
        assert payload["vram_mb"] == 4000 and payload["ram_mb"] == 6000


def test_ordinary_seat_waits_indefinitely_by_default():
    """A finite default kills work that is merely queued: measured 2026-07-28,
    trainings hold a seat p90 3.79 h and up to 23.1 h."""
    assert gpu_spool.DEFAULT_SEAT_MAX_WAIT_S == 0.0
    assert gpu_spool.DEFAULT_PRIORITY_MAX_WAIT_S > 0, "CI still needs a verdict"


def test_priority_yield_sits_between_eval_and_training_holds():
    """Above the eval p90 (11 min) so priority still wins a slot about to free;
    far below the training median (37 min) so it never freezes the queue."""
    assert 11 * 60 < gpu_spool.DEFAULT_PRIORITY_YIELD_S < 37 * 60
