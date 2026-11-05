"""The training-data guard: evals on training data refuse without an explicit bypass.

Background (2026-08-09 census): 4 of 208 stored evaluations were scored on their
model's own training data -- three at 100% (the alsi_rigno_r00 ensemble on its
own 10-case training set) and one at 78% (rigno_sp_c0_20k scored on
83 straight-path cases, 65 of them its training set). Nothing refused the run
and nothing marked the result.
"""

from __future__ import annotations

from aiida import orm

from aiida_pino.workflows.evaluate_model import EvaluateModelWorkChain


def test_the_guard_exists_and_must_be_bypassed_explicitly():
    spec = EvaluateModelWorkChain.spec()
    gate = spec.inputs["allow_training_data"]
    assert gate.valid_type is orm.Bool or orm.Bool in (
        gate.valid_type if isinstance(gate.valid_type, tuple) else (gate.valid_type,)
    )
    assert not gate.required, "the bypass is opt-in, never demanded"
    assert "ERROR_TRAINING_DATA_OVERLAP" in spec.exit_codes
    assert spec.exit_codes.ERROR_TRAINING_DATA_OVERLAP.status == 305


def test_overlap_detection_is_twin_collapsed(aiida_profile_clean):
    """A training case present through a payload twin still counts as overlap.

    68 of 70 duplicated payloads have their twins split across datasets, so a
    raw-uuid comparison lets the same simulation through under a different node.
    """
    from aiida_pino.data import PinoSimulationData
    from aiida_pino.utils.model_registry import payload_keys

    a, b = PinoSimulationData(), PinoSimulationData()
    for node in (a, b):
        node.base.extras.set("content_hash", "same-payload")
        node.store()
    other = PinoSimulationData()
    other.base.extras.set("content_hash", "different")
    other.store()

    assert payload_keys({a.uuid}) & payload_keys({b.uuid}), "twins must collide"
    assert not (payload_keys({a.uuid}) & payload_keys({other.uuid}))
    # a uuid that resolves to no node still counts as itself (deleted sims)
    assert payload_keys({"0" * 36}) == {"0" * 36}


def test_contaminated_rows_cannot_rank_and_carry_the_warning():
    from aiida_pino.console.model_metrics import rank_evaluations

    rows = [
        {"eval_pk": 1, "model_uuid": "m1", "dataset_key": "d", "dataset_display": "d",
         "t_mae_k": 5.0, "t_rmse_k": 7.0, "held_out": True},
        # Contaminated: the service strips its Kelvin numbers before ranking, so
        # a better-looking training-fit score must not take the badge.
        {"eval_pk": 2, "model_uuid": "m2", "dataset_key": "d", "dataset_display": "d",
         "t_mae_k": None, "t_rmse_k": None, "held_out": False, "train_overlap_n": 65},
    ]
    ranked = rank_evaluations(rows)
    assert any("best T MAE" in b for b in ranked[1]["badges"])
    assert ranked[2]["badges"] == []
    assert "TRAINED ON 65" in ranked[2]["warning"]
    assert "warning" not in ranked[1]
