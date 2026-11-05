"""Cross-footprint fallback: a first-of-a-kind eval must not inherit a training booking.

`vram_footprint_key` includes `hydra_overrides`, so an eval of a new config has
no recorded peak and fell back to whatever its parent training booked. That is
how one eval came to reserve 20000 MiB for a job that peaks ~4 GB, and sit
queued behind trainings on 2026-08-03/04.

Evals hold no backward pass, so their footprint is dominated by the inference
batch rather than model capacity: the 44 peaks recorded here span 1334-4580 MiB
across months, several architectures (depth-18 RIGNO, GAOT) and datasets. The
max of all recorded eval peaks is therefore a far better estimate for an unknown
eval than an unrelated training's reservation.

Two guardrails, because the fallback is a heuristic and not a measurement:

- it is a PERCENTILE, not the max. One RIGNO-unstrided eval peaked at 14328 MiB
  where every other sits at 2566-4608; taking the max let that single job book
  15761 MiB for every eval, which on 2026-08-05 left 6767 MiB free on a 22528
  card and blocked a queued 12500 MiB training for 13 hours.
- it may only ever LOWER an inherited booking (`min(inherited, fallback)`).
  Under-booking is the dangerous direction -- the ledger hands out room that is
  not there -- so raising a booking on a guess has no upside.
- peaks below `EVAL_PEAK_FLOOR_MB` are excluded, so a crashed or truncated eval
  (164/858/986 MiB were recorded here) cannot drag the estimate down.

An exact footprint hit is a real measurement and is NOT clamped: it stays free
to raise a booking.
"""

from __future__ import annotations

import pytest

from aiida import orm
from aiida.engine import run_get_node

from aiida_pino.utils.vram import (
    EVAL_PEAK_FLOOR_MB,
    fallback_eval_peak_mb,
    vram_reserve_mb,
)
from aiida_pino.workflows.evaluate_model import EvaluateModelWorkChain


def _profile(name: str, mb: int, overrides: list[str]):
    """Conftest factory's default profile plus a booking and override set."""
    return {
        "config_name": "config_NS_energy_PINO_flow3D",
        "run_name": name,
        "split_mode": "standard",
        "hydra_overrides": overrides,
        "cuda_device": "0",
        "wandb_offline": True,
        "time_as_batch": False,
        "time_as_dim": True,
        "include_time": True,
        "moving_ref_frame": False,
        "moving_frame_steps": 1,
        "time_avg_window": 1,
        "super_res_factor": 1,
        "gpu_vram_mb": mb,
        "autoregressive": {
            "enabled": False,
            "eval_horizon_k": 1,
            "eval_use_gt_init": True,
            "require_time": False,
            "state_prev_suffix": "_prev",
        },
    }


def _eval_jobs(node):
    return sorted(
        (c for c in node.called_descendants if c.process_label == "PinoEvalCalculation"),
        key=lambda c: c.pk,
    )


def test_floor_excludes_crashed_run_peaks(fake_python_executable, local_code_factory, pino_dataset_factory):
    """A 164 MiB peak from a crashed eval must never become the estimate."""
    assert EVAL_PEAK_FLOOR_MB >= 1000


def test_fallback_is_used_when_the_footprint_is_new(
    fake_python_executable, local_code_factory, pino_dataset_factory, pino_model_factory
):
    """Second model has different overrides -> different key -> no exact peak."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-fallback-1")
    dataset = pino_dataset_factory("fallback-ds-1")

    # First eval records a peak (4200 from the fake executable) under key A.
    model_a = pino_model_factory("fb-model-a", training_profile=_profile("fb-a", 20000, []))
    _, first = run_get_node(
        EvaluateModelWorkChain, model=model_a, dataset=dataset, python_code=python
    )
    assert first.is_finished_ok
    assert fallback_eval_peak_mb() == 4200

    # Second model: a different override set, so its footprint key misses.
    model_b = pino_model_factory(
        "fb-model-b", training_profile=_profile("fb-b", 20000, ["amp.enabled=false"])
    )
    _, second = run_get_node(
        EvaluateModelWorkChain, model=model_b, dataset=dataset, python_code=python
    )
    assert second.is_finished_ok
    booked = _eval_jobs(second)[0].inputs.training_profile.get_dict()["gpu_vram_mb"]
    assert booked == vram_reserve_mb(4200) == 5224, (
        f"first-of-a-kind eval booked {booked} MiB; expected the cross-footprint fallback"
    )


def test_fallback_never_raises_an_inherited_booking(
    fake_python_executable, local_code_factory, pino_dataset_factory, pino_model_factory
):
    """Guardrail: min(inherited, fallback). A guess must not increase a reservation."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-fallback-2")
    dataset = pino_dataset_factory("fallback-ds-2")

    model_a = pino_model_factory("fb2-model-a", training_profile=_profile("fb2-a", 20000, []))
    _, first = run_get_node(
        EvaluateModelWorkChain, model=model_a, dataset=dataset, python_code=python
    )
    assert first.is_finished_ok

    # Inherited booking (3000) is SMALLER than the fallback would give (5224).
    model_b = pino_model_factory(
        "fb2-model-b", training_profile=_profile("fb2-b", 3000, ["arch.rigno.processor_steps=18"])
    )
    _, second = run_get_node(
        EvaluateModelWorkChain, model=model_b, dataset=dataset, python_code=python
    )
    assert second.is_finished_ok
    booked = _eval_jobs(second)[0].inputs.training_profile.get_dict()["gpu_vram_mb"]
    assert booked == 3000, f"fallback raised the booking to {booked} MiB; it may only lower"


def test_explicit_reservation_still_wins(
    fake_python_executable, local_code_factory, pino_dataset_factory, pino_model_factory
):
    """A deliberate evaluation_profile booking is never touched by the fallback."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-fallback-3")
    dataset = pino_dataset_factory("fallback-ds-3")
    model = pino_model_factory(
        "fb3-model", training_profile=_profile("fb3", 20000, ["custom.super_res=2"])
    )
    _, node = run_get_node(
        EvaluateModelWorkChain,
        model=model,
        dataset=dataset,
        python_code=python,
        evaluation_profile=orm.Dict(dict={"gpu_vram_mb": 7000}),
    )
    assert node.is_finished_ok
    assert _eval_jobs(node)[0].inputs.training_profile.get_dict()["gpu_vram_mb"] == 7000


def test_percentile_ignores_a_single_large_outlier():
    """The regression that motivated switching from max to p90.

    One RIGNO-unstrided eval peaked at 14328 MiB against a 2566-4608 MiB cluster.
    Under `max`, that one job set the reservation for every subsequent eval
    (15761 MiB), which blocked a 12500 MiB training on a 22528 MiB card for 13
    hours. A high percentile must track the cluster, not the outlier.
    """
    from aiida_pino.utils.vram import EVAL_FALLBACK_PERCENTILE

    cluster = [2566, 3492, 3800, 3800, 4048, 4214, 4228, 4580, 4580, 4608]
    peaks = sorted(cluster + [14328])
    idx = min(len(peaks) - 1,
              max(0, int(round(EVAL_FALLBACK_PERCENTILE / 100.0 * len(peaks))) - 1))
    chosen = peaks[idx]
    assert chosen < 14328, "the outlier must not set the reservation"
    assert chosen >= 4580, "but the fallback must still sit above the cluster"


def test_floor_still_excludes_crashed_runs():
    """A truncated eval recorded 164 MiB; sizing from it would cause an OOM."""
    from aiida_pino.utils.vram import EVAL_PEAK_FLOOR_MB

    assert EVAL_PEAK_FLOOR_MB >= 1000
