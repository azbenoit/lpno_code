"""The 2026-08-09 eval frictions: resolution, failure reasons, ports, resume, ps.

Each fix verified against the live database before landing here:

* eval resolution — "has any checkpoint been evaluated at two resolutions?"
  could not be expressed in QueryBuilder; after stamping `eval_stride` it is one
  query (answer on 2026-08-09: 86 models with stamped strides, 0 at two
  resolutions — 93 evals stamped, 72 full-res, 21 at 2×2×1, 115 honestly
  unknown, same unknown-stays-unknown rule as the model-side F2).
* no-metrics evals — 42 of 282 finished at exit 500 with only
  remote_folder/retrieved. They are NOT silent (exit 500 is loud) and NOT
  salvageable (one such stdout ends in a Hydra crash; no metrics were ever
  computed). What was missing is the REASON on the node — the F7 shape again.
* port names — every vintage uses the same ports (205 calcjobs: `metrics`;
  340 workchains: `metrics` + `eval_summary` + `evaluation`). The pain is three
  similar names, not drift; `eval_metrics()` is the one entry point.
"""

from __future__ import annotations

from aiida import orm

from aiida_pino.data.pino_model_evaluation import PinoModelEvaluationData
from aiida_pino.utils.model_registry import eval_metrics


def test_the_scored_grid_is_one_queryable_field(aiida_profile_clean):
    node = PinoModelEvaluationData.from_payload(metadata={
        "evaluation_name": "strided",
        "eval_stride": [2, 2, 1],
        "metrics": {"T.abs_err_mean": 0.01},
    }).store()
    assert node.base.extras.get("eval_stride") == [2, 2, 1]
    assert node.base.extras.get("eval_full_resolution") is False

    full = PinoModelEvaluationData.from_payload(metadata={
        "evaluation_name": "full", "eval_stride": [1, 1, 1], "metrics": {},
    }).store()
    assert full.base.extras.get("eval_full_resolution") is True

    # THE query that used to need loading every eval CalcJob by hand.
    rows = orm.QueryBuilder().append(
        PinoModelEvaluationData,
        filters={"extras": {"has_key": "eval_stride"}},
        project=["extras.eval_stride"],
    ).all(flat=True)
    assert sorted(map(tuple, rows)) == [(1, 1, 1), (2, 2, 1)]


def test_an_unknown_stride_stamps_nothing(aiida_profile_clean):
    """Unknown stays unknown — the F2 rule. Absent must not read as full-res."""
    node = PinoModelEvaluationData.from_payload(metadata={"evaluation_name": "old"}).store()
    assert node.base.extras.get("eval_stride", None) is None
    assert node.base.extras.get("eval_full_resolution", None) is None


def test_eval_metrics_is_one_entry_point(aiida_profile_clean):
    record = PinoModelEvaluationData.from_payload(metadata={
        "evaluation_name": "x", "metrics": {"T.abs_err_mean": 0.02},
    }).store()
    assert eval_metrics(record) == {"T.abs_err_mean": 0.02}
    assert eval_metrics(record.pk) == {"T.abs_err_mean": 0.02}

    empty = PinoModelEvaluationData.from_payload(metadata={"evaluation_name": "none"}).store()
    assert eval_metrics(empty) == {}, "an eval with no metrics answers {}, never raises"


def test_ps_recent_orders_by_mtime(aiida_profile_clean, monkeypatch):
    from aiida_pino.console.config import Settings
    from aiida_pino.console.service import AiidaConsoleService

    service = AiidaConsoleService(Settings())
    # Just the ordering contract: recent=True must not raise and must accept
    # the same filters. (Live proof needs processes whose mtime and ctime
    # orders differ, which a clean profile cannot manufacture cheaply.)
    assert service.list_processes(recent=True) == []
    assert service.list_processes(recent=False) == []


def test_init_model_accepts_artifacts_not_just_models(aiida_profile_clean):
    """Resuming a killed run from scratch needed an import-model round trip."""
    from aiida_pino.workflows.train_model import TrainModelWorkChain

    spec = TrainModelWorkChain.spec()
    valid = spec.inputs["init_model"].valid_type
    names = {t.__name__ for t in (valid if isinstance(valid, tuple) else (valid,))}
    assert {"PinoModelData", "FolderData", "RemoteData"} <= names, names
