"""Automatic GPU VRAM reservation: footprint keys, reserve margin, auto-smoke.

A training profile without a manual `gpu_vram_mb` must resolve one from the
database of measured peaks (`gpu_vram_peak_mb` extras keyed by `vram_key`),
running a short hidden smoke first when no peak is recorded. The fake python
executable in conftest reports a measured peak of 4200 MB in
`train_summary.json`, so the expected reservation is
`vram_reserve_mb(4200) == 5224`. The manual-reservation path stays covered by
tests/test_workflows.py.
"""

from __future__ import annotations

import pytest

from aiida import orm
from aiida.engine import run_get_node

from aiida_pino.utils import load_training_profile_defaults
from aiida_pino.utils.vram import (
    SMOKE_KEYWORD,
    VRAM_KEY_EXTRA,
    VRAM_PEAK_EXTRA,
    vram_footprint_key,
    vram_reserve_mb,
)
from aiida_pino.workflows.train_model import TrainModelWorkChain


def _auto_training_profile(**overrides):
    """The test_workflows helper profile, minus the manual reservation."""
    profile = load_training_profile_defaults("config_NS_energy_PINO_flow3D")
    profile["run_name"] = "vram-auto"
    profile["hydra_overrides"] = []
    profile["split_mode"] = "standard"
    profile["include_time"] = True
    profile["time_as_batch"] = False
    profile["time_as_dim"] = True
    profile["moving_ref_frame"] = False
    profile["moving_frame_steps"] = 1
    profile["time_avg_window"] = 1
    profile["autoregressive"]["enabled"] = False
    profile.pop("gpu_vram_mb", None)
    profile.update(overrides)
    return profile


def _train_jobs(node):
    """PinoTrainCalculation descendants in submission order."""
    return sorted(
        (child for child in node.called_descendants if child.process_label == "PinoTrainCalculation"),
        key=lambda child: child.pk,
    )


def test_footprint_key_sensitivity_and_stability() -> None:
    profile = {
        "config_name": "config_NS_energy_PINO_flow3D",
        "batch_size_grid": 20,
        "batch_size_validation": 20,
        "inference_batch_size": 2,
        "include_time": True,
        "time_as_batch": False,
        "time_as_dim": True,
        "run_name": "run-a",
        "learning_rate": 6.0e-5,
        "wandb_offline": True,
        "env": {},
    }
    metadata = {
        "compatibility_signature": "sig-a",
        "canonical_layout": "time_dim_noavg_noref",
        "time_steps_max": 10,
    }
    key = vram_footprint_key(profile, metadata)
    assert key == vram_footprint_key(dict(profile), dict(metadata)), "key must be deterministic"

    # Memory-relevant changes move the key.
    assert vram_footprint_key({**profile, "batch_size_grid": 21}, metadata) != key
    assert vram_footprint_key({**profile, "config_name": "config_RIGNO_flow3D"}, metadata) != key
    assert vram_footprint_key({**profile, "env": {"PINO_LAZY_STRIDE": "2,2,1"}}, metadata) != key
    assert vram_footprint_key(profile, {**metadata, "compatibility_signature": "sig-b"}) != key
    # hydra_overrides carry amp/arch switches, so they must move the key too.
    assert vram_footprint_key({**profile, "hydra_overrides": ["amp.enabled=false"]}, metadata) != key

    # Bookkeeping knobs must not: the estimate database has to hit across reruns.
    assert vram_footprint_key({**profile, "run_name": "run-b"}, metadata) == key
    assert vram_footprint_key({**profile, "learning_rate": 1.0e-3}, metadata) == key
    assert vram_footprint_key({**profile, "wandb_offline": False}, metadata) == key


def test_reserve_margin() -> None:
    assert vram_reserve_mb(4200) == 5224
    assert vram_reserve_mb(20000) == 22000
    assert vram_reserve_mb(100) == 1124


def test_auto_smoke_then_reserved_real_run(
    fake_python_executable, local_code_factory, pino_dataset_factory
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-vram-auto")
    dataset = pino_dataset_factory("vram-auto")
    profile = _auto_training_profile(gpu_seat_priority=True)

    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=profile),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.is_finished_ok

    jobs = _train_jobs(node)
    assert len(jobs) == 2, "expected a smoke run followed by the real run"
    smoke, real = jobs

    smoke_profile = smoke.inputs.training_profile.get_dict()
    assert smoke_profile["max_steps"] == 100
    assert smoke_profile["run_name"].endswith("_vramsmoke")
    assert smoke_profile.get("gpu_vram_mb") is None
    assert "gpu_seat_priority" not in smoke_profile, "a priority smoke would block all spool admissions on the card"
    assert smoke.base.extras.get("keywords", None) == [SMOKE_KEYWORD]
    smoke_key = smoke.base.extras.get(VRAM_KEY_EXTRA, None)
    assert smoke_key
    assert smoke.base.extras.get(VRAM_PEAK_EXTRA, None) == 4200

    real_profile = real.inputs.training_profile.get_dict()
    assert real_profile["max_steps"] == profile["max_steps"], "the real run must keep the requested max_steps"
    assert real_profile["gpu_vram_mb"] == 5224
    assert real.base.extras.get(VRAM_KEY_EXTRA, None) == smoke_key
    assert real.base.extras.get(VRAM_PEAK_EXTRA, None) == 4200


def test_second_run_skips_smoke(fake_python_executable, local_code_factory, pino_dataset_factory) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-vram-second")
    dataset = pino_dataset_factory("vram-second")
    inputs = {
        "train_dataset": dataset,
        "val_dataset": dataset,
        # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        "allow_split_violation": orm.Bool(True),
        "training_profile": orm.Dict(dict=_auto_training_profile()),
        "python_code": python,
    }

    # First run records a peak for this footprint key (through a smoke if no
    # earlier test in this run already recorded one).
    _, first = run_get_node(TrainModelWorkChain, **inputs)
    assert first.is_finished_ok

    _, second = run_get_node(TrainModelWorkChain, **inputs)
    assert second.is_finished_ok
    jobs = _train_jobs(second)
    assert len(jobs) == 1, "a recorded peak must skip the smoke"
    assert SMOKE_KEYWORD not in (jobs[0].base.extras.get("keywords", None) or [])
    assert jobs[0].inputs.training_profile.get_dict()["gpu_vram_mb"] == 5224


def test_queue_shows_eval_bookings_and_auto_sizing() -> None:
    """The Monitor queue must not call a declared eval booking 'exclusive card'.

    Regression: the helper read only `training_profile`, so every queued
    EvaluateModelWorkChain — which books through `evaluation_profile` — fell
    into the exception branch and rendered as a whole-card booking (seen live
    2026-07-28: twelve 3000 MiB evals all shown as exclusive).
    """
    from aiida_pino.console.service import queued_vram_reservation

    class _Dict:
        def __init__(self, payload):
            self._payload = payload

        def get_dict(self):
            return self._payload

    class _Inputs:
        def __init__(self, **profiles):
            for name, payload in profiles.items():
                setattr(self, name, _Dict(payload))

    class _Node:
        def __init__(self, **profiles):
            self.inputs = _Inputs(**profiles)

    # an eval books through evaluation_profile
    assert queued_vram_reservation(_Node(evaluation_profile={"gpu_vram_mb": 3000})) == (3000, False)
    # a manual training booking still reads as booked
    assert queued_vram_reservation(_Node(training_profile={"gpu_vram_mb": 8000})) == (8000, False)
    # training with nothing declared is auto-sized, NOT exclusive
    assert queued_vram_reservation(_Node(training_profile={"gpu_vram_mb": None})) == (None, True)
    # a train+eval pair reports the larger of the two bookings
    assert queued_vram_reservation(
        _Node(training_profile={"gpu_vram_mb": 10000}, evaluation_profile={"gpu_vram_mb": 3000})
    ) == (10000, False)
    # a run with no GPU profile at all is genuinely exclusive
    assert queued_vram_reservation(_Node()) == (None, False)


def test_manual_reservation_bypasses_automation(
    fake_python_executable, local_code_factory, pino_dataset_factory
) -> None:
    python = local_code_factory(str(fake_python_executable), label="fake-python-vram-manual")
    dataset = pino_dataset_factory("vram-manual")

    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=_auto_training_profile(gpu_vram_mb=8000)),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.is_finished_ok
    jobs = _train_jobs(node)
    assert len(jobs) == 1
    assert jobs[0].inputs.training_profile.get_dict()["gpu_vram_mb"] == 8000
    assert jobs[0].base.extras.get(VRAM_KEY_EXTRA, None) is None, "manual reservation must not join the estimate database"


def test_smoke_hidden_from_process_list(
    fake_python_executable, local_code_factory, pino_dataset_factory
) -> None:
    try:
        from web.app.aiida_service import AiidaConsoleService
        from web.app.config import Settings
    except ImportError:
        pytest.skip("console service import pattern changed; see tests/test_data_eviction.py")

    python = local_code_factory(str(fake_python_executable), label="fake-python-vram-console")
    dataset = pino_dataset_factory("vram-console")
    # A batch size no other test in this file uses, so the footprint key is
    # fresh and a smoke runs even when earlier tests already recorded peaks.
    _, node = run_get_node(
        TrainModelWorkChain,
        train_dataset=dataset,
        val_dataset=dataset,
        training_profile=orm.Dict(dict=_auto_training_profile(batch_size_grid=7)),
        allow_split_violation=orm.Bool(True),  # fixture trains and validates on ONE dataset by design; the 5d25435
        # guard (exit 306) rightly refuses that without an explicit flag
        python_code=python,
    )
    assert node.is_finished_ok
    jobs = _train_jobs(node)
    assert len(jobs) == 2, "fresh footprint key should have forced a smoke"
    smoke_pk = jobs[0].pk

    svc = AiidaConsoleService(Settings())
    assert smoke_pk not in {row["pk"] for row in svc.list_processes(limit=50)}
    assert smoke_pk in {row["pk"] for row in svc.list_processes(limit=50, include_smokes=True)}


def test_overrun_of_the_booking_is_reported_not_silent(
    fake_python_executable, local_code_factory, pino_dataset_factory, tmp_path
) -> None:
    """A run whose measured peak exceeds its booking must say so.

    gpu_spool admits on the booked figure, so an overrun endangered co-tenants
    invisibly for the whole run (one run booked 15761 MiB, resident 18150).
    The recorded peak already self-corrects the NEXT booking; this pins the
    report that makes the current one visible.
    """
    dataset = pino_dataset_factory("vram-overrun")
    inputs = {
        "train_dataset": dataset,
        "val_dataset": dataset,
        # fixture trains and validates on ONE dataset by design (5d25435 guard)
        "allow_split_violation": orm.Bool(True),
        "training_profile": orm.Dict(dict=_auto_training_profile()),
    }

    # First run records peak 4200 for this footprint -> later bookings are 5224.
    python = local_code_factory(str(fake_python_executable), label="fake-python-overrun-a")
    _, first = run_get_node(TrainModelWorkChain, python_code=python, **inputs)
    assert first.is_finished_ok

    # Same footprint, but the workload now peaks at 9000 MiB > the 5224 booking.
    grown = tmp_path / "fake_python_grown_peak"
    grown.write_text(fake_python_executable.read_text().replace("4200", "9000"))
    grown.chmod(0o755)
    python2 = local_code_factory(str(grown), label="fake-python-overrun-b")
    _, second = run_get_node(TrainModelWorkChain, python_code=python2, **inputs)
    assert second.is_finished_ok
    assert _train_jobs(second)[0].inputs.training_profile.get_dict()["gpu_vram_mb"] == 5224

    messages = "\n".join(log.message for log in orm.Log.collection.get_logs_for(second))
    assert "VRAM overrun" in messages
    assert "9000" in messages and "5224" in messages
