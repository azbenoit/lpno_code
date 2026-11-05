"""An eval sizes its own VRAM reservation instead of inheriting the training one.

`merge_model_training_with_evaluation_profile` deep-copies the model's frozen
training profile, so an eval booked whatever its training booked. A RIGNO run
reserved at 20000 MiB made every eval of it reserve 20000 MiB too, while the
measured eval peaks on the original machine sit at 3.5-4.6 GB. Against a 22528 MiB per-card
budget that is effectively a whole card per eval: on 2026-08-03 two 20000 MiB
trainings sat blocked behind such bookings with 33 GB of the 49 GB physically
free.

The peak needed to size it was already being recorded for free -- the parser
writes `gpu_vram_peak_mb` on every calcjob (parsers/common.py). Only the
footprint key was missing, so `lookup_vram_peak_mb` could never hit; all 129
recorded peaks carried `vram_key=None`.

An eval's footprint is not its training's (no backward pass), so the key is
role-discriminated. The `train` role keeps the legacy hash, or every peak
recorded so far would be stranded under a key nothing looks up.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import run_get_node

from aiida_pino.utils.vram import (
    VRAM_KEY_EXTRA,
    VRAM_PEAK_EXTRA,
    vram_footprint_key,
    vram_reserve_mb,
)
from aiida_pino.workflows.evaluate_model import EvaluateModelWorkChain

_PROFILE = {
    "config_name": "config_NS_energy_PINO_flow3D",
    "batch_size_grid": 20,
    "include_time": True,
    "env": {},
}
_METADATA = {"compatibility_signature": "sig-a", "canonical_layout": "time_dim_noavg_noref"}


def test_train_role_keeps_the_legacy_key() -> None:
    """Back-compat: peaks already recorded must stay reachable."""
    assert vram_footprint_key(_PROFILE, _METADATA) == vram_footprint_key(_PROFILE, _METADATA, role="train")


def test_eval_role_moves_the_key() -> None:
    """An eval has no backward pass; sizing it from a training peak over-books."""
    train_key = vram_footprint_key(_PROFILE, _METADATA, role="train")
    eval_key = vram_footprint_key(_PROFILE, _METADATA, role="eval")
    assert eval_key != train_key
    assert eval_key == vram_footprint_key(dict(_PROFILE), dict(_METADATA), role="eval"), "must be deterministic"


def _eval_jobs(node):
    return sorted(
        (child for child in node.called_descendants if child.process_label == "PinoEvalCalculation"),
        key=lambda child: child.pk,
    )


def _model_with_training_reservation(pino_model_factory, name: str, mb: int):
    """A model whose frozen training profile carries a large manual booking.

    Mirrors the conftest factory's own default profile: the raw config defaults
    carry a `moving_frame_steps` the 2-step test simulations cannot satisfy.
    """
    return pino_model_factory(
        name,
        training_profile={
            "config_name": "config_NS_energy_PINO_flow3D",
            "run_name": name,
            "split_mode": "standard",
            "hydra_overrides": [],
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
        },
    )


def test_first_eval_records_a_keyed_peak(
    fake_python_executable, local_code_factory, pino_dataset_factory, pino_model_factory
) -> None:
    """Without a key on the calcjob no later eval can ever look the peak up."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-eval-vram-1")
    dataset = pino_dataset_factory("eval-vram-1")
    MODEL = "eval-vram-model-1"
    model = _model_with_training_reservation(pino_model_factory, MODEL, 20000)

    _, node = run_get_node(
        EvaluateModelWorkChain,
        model=model,
        dataset=dataset,
        python_code=python,
    )
    assert node.is_finished_ok

    job = _eval_jobs(node)[0]
    assert job.base.extras.get(VRAM_KEY_EXTRA, None), "eval calcjob must carry a footprint key"
    assert job.base.extras.get(VRAM_PEAK_EXTRA, None) == 4200


def test_second_eval_sizes_from_the_recorded_peak_not_the_training_booking(
    fake_python_executable, local_code_factory, pino_dataset_factory, pino_model_factory
) -> None:
    """The regression: 20000 MiB inherited, 4200 MiB actually used."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-eval-vram-2")
    dataset = pino_dataset_factory("eval-vram-2")
    MODEL = "eval-vram-model-2"
    model = _model_with_training_reservation(pino_model_factory, MODEL, 20000)
    inputs = {"model": model, "dataset": dataset, "python_code": python}

    _, first = run_get_node(EvaluateModelWorkChain, **inputs)
    assert first.is_finished_ok

    _, second = run_get_node(EvaluateModelWorkChain, **inputs)
    assert second.is_finished_ok
    booked = _eval_jobs(second)[0].inputs.training_profile.get_dict()["gpu_vram_mb"]
    assert booked == vram_reserve_mb(4200) == 5224, f"eval booked {booked} MiB, expected 5224"


def test_explicit_eval_reservation_is_respected(
    fake_python_executable, local_code_factory, pino_dataset_factory, pino_model_factory
) -> None:
    """Auto-sizing replaces an *inherited* booking, never a deliberate one."""
    python = local_code_factory(str(fake_python_executable), label="fake-python-eval-vram-3")
    dataset = pino_dataset_factory("eval-vram-3")
    MODEL = "eval-vram-model-3"
    model = _model_with_training_reservation(pino_model_factory, MODEL, 20000)

    _, node = run_get_node(
        EvaluateModelWorkChain,
        model=model,
        dataset=dataset,
        python_code=python,
        evaluation_profile=orm.Dict(dict={"gpu_vram_mb": 7000}),
    )
    assert node.is_finished_ok
    assert _eval_jobs(node)[0].inputs.training_profile.get_dict()["gpu_vram_mb"] == 7000
