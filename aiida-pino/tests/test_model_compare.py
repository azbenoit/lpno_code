"""The ablation referee: every difference between two models, classified.

Ground truth from the live DB, 2026-08-10:
* seed replicates (alsi_fno_r01_m1 vs m2) read
  SINGLE-FACTOR: seed -- after factor grouping; counting FIELDS made the same
  pair read "CONFOUNDED: 2 confounds" because the seed appears both as the
  stamped extra and as override.seed.
* the rigno2579 e8-vs-linall pair (39779 vs 40397) reads SINGLE-FACTOR with
  unknowns: FNO_input_keys differs and one side has no stride stamp -- the
  honest answer, and itself a finding about that pair.
* unrelated models (2945 vs 7674) read CONFOUNDED with 17 factors.
"""

from __future__ import annotations

from aiida_pino.utils.model_compare import (
    _parse_overrides,
    _severity,
    compare_models,
)


import pytest


@pytest.fixture(autouse=True)
def _archived_config_defaults(request, monkeypatch):
    """Fixture models archive an (equal, empty-ish) resolved config by default.

    The in-memory stand-ins cannot carry a real repository, and since the
    config-content diff landed, an unreadable archive is an UNKNOWN row --
    correct for real nodes, noise for these fixtures. Tests that probe the
    config path override this patch explicitly.
    """
    if "no_config_default" in request.keywords:
        return
    monkeypatch.setattr("aiida_pino.utils.model_compare._archived_config_flat",
                        lambda model: {"custom.x": 1})


def _model(_unused=None, **kw):
    """A minimal in-memory stand-in carrying the metadata shape _facts reads."""
    profile = {"hydra_overrides": [], "env": {}, "learning_rate": 1e-3,
               "max_steps": 1000, **kw.pop("profile", {})}
    metadata = {"training_profile": profile, "config_name": "cfg",
                "training_source_uuid": "ds-1",
                "training_source_summary": {"members": [{"uuid": "s1"}, {"uuid": "s2"}]},
                **kw.pop("metadata", {})}
    extras = {"model_name": kw.pop("name", "m"),
              "training_seed": kw.pop("seed", 42),
              # The unknown-eligible facts, recorded: since the adversarial
              # blocker fix, a fact missing on BOTH sides emits an unknown row
              # (double-absence must not read as equality), so a fixture that
              # wants CLEAN has to actually record these.
              "training_stride": [1, 1, 1], "steps_trained": 1000,
              "code_version_inferred": "abc123", **kw.pop("extras", {})}
    pk = kw.pop("pk", 1)
    assert not kw, f"unused fixture args: {kw}"

    class _Extras:
        all = extras

        @staticmethod
        def get(key, default=None):
            return extras.get(key, default)

    class _Base:
        pass

    _Base.extras = _Extras

    class _Model:
        def get_model_metadata(self):
            return metadata

    node = _Model()
    node.pk = pk
    node.uuid = f"uuid-{pk}"
    node.label = f"model-{pk}"
    node.ctime = None
    node.base = _Base()
    return node


def test_identical_models_are_clean(monkeypatch):
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    a, b = _model(None, pk=1), _model(None, pk=2)
    report = compare_models(a, b)
    assert report["verdict"].startswith("CLEAN")
    assert report["training_data"] == "same dataset node"


def test_one_factor_in_two_fields_is_still_single_factor(monkeypatch):
    """The live defect: seed as extra AND override.seed is ONE knob, not two."""
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    a = _model(None, pk=1, seed=1001, profile={"hydra_overrides": ["seed=1001"]})
    b = _model(None, pk=2, seed=1002, profile={"hydra_overrides": ["seed=1002"]})
    report = compare_models(a, b)
    assert report["confound_factors"] == ["seed"]
    assert report["verdict"].startswith("SINGLE-FACTOR: only 'seed'")


def test_different_data_is_a_confound_with_the_overlap_counted(monkeypatch):
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    a = _model(None, pk=1)
    b = _model(None, pk=2, metadata={
        "training_source_uuid": "ds-2",
        "training_source_summary": {"members": [{"uuid": "s2"}, {"uuid": "s3"}]},
    })
    report = compare_models(a, b)
    assert "1 shared of 2 (a) / 2 (b)" in report["training_data"]
    assert "training data" in report["confound_factors"]


def test_unknown_never_reads_as_same(monkeypatch):
    """One side has no stride stamp: the verdict must say unverifiable, not clean."""
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    a = _model(None, pk=1, extras={"training_stride": [1, 1, 1]})
    b = _model(None, pk=2, extras={"training_stride": None})
    report = compare_models(a, b)
    assert report["unknown_factors"] == ["resolution"]
    assert "cannot be verified equal" in report["verdict"] or "unknown" in report["verdict"]


def test_shared_evals_flag_contamination_and_resolution(monkeypatch):
    evals = {
        "uuid-1": {"ds": {"eval_pk": 10, "held_out": True, "eval_stride": [1, 1, 1],
                          "t_abs_err_mean": 0.01, "dataset_label": "test set"}},
        "uuid-2": {"ds": {"eval_pk": 11, "held_out": False, "eval_stride": [2, 2, 1],
                          "t_abs_err_mean": 0.005, "dataset_label": "test set"}},
    }
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows",
                        lambda uuid: evals[uuid])
    report = compare_models(_model(None, pk=1), _model(None, pk=2))
    (shared,) = report["shared_evaluations"]
    assert shared["comparable"] is False
    assert "scored on training data" in shared["problems"]
    assert any("resolution" in p for p in shared["problems"])


def test_bookkeeping_never_pollutes_the_verdict(monkeypatch):
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    a = _model(None, pk=1, name="arm0",
               profile={"cuda_device": "0", "gpu_vram_mb": 3000})
    b = _model(None, pk=2, name="arm1",
               profile={"cuda_device": "1", "gpu_vram_mb": 9000})
    report = compare_models(a, b)
    assert report["verdict"].startswith("CLEAN")
    assert all(d["severity"] == "benign" for d in report["differences"])


def test_override_parsing_strips_prefixes_and_last_wins():
    parsed = _parse_overrides(["+custom.seed=1", "~lr=0.1", "custom.seed=2"])
    assert parsed["custom.seed"] == "2"
    assert parsed["lr"] == "0.1"


def test_unknown_keys_are_confounds_not_benign():
    """A key nobody classified must fail loud: unknown is not safe."""
    assert _severity("custom.some_new_knob") == "confound"
    assert _severity("run_name") == "benign"
    assert _severity("batch_size_validation") == "caution"


def test_both_sides_unrecorded_is_an_unknown_not_an_equality(monkeypatch):
    """The adversarial BLOCKER: va == vb short-circuited the row when a fact was
    missing on both models, so two stride-less models read "valid ablation"
    while resolution was unknowable on either. 127 of 243 models lack
    training_stride -- double absence is the common case, not a corner."""
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    a = _model(None, pk=1, extras={"training_stride": None, "steps_trained": None,
                                   "code_version_inferred": None})
    b = _model(None, pk=2, extras={"training_stride": None, "steps_trained": None,
                                   "code_version_inferred": None})
    report = compare_models(a, b)
    assert set(report["unknown_factors"]) == {"resolution", "training length", "training code"}
    assert not report["verdict"].startswith("CLEAN")


def test_twin_payloads_read_as_identical_data(monkeypatch):
    """1232 model pairs disagreed between raw-uuid and twin-collapsed overlap;
    compare_models(195, 10237) read 'DIFFERENT DATA: 0 shared' for two
    byte-identical training sets. Same identity rule as the held-out path now."""
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    monkeypatch.setattr("aiida_pino.utils.model_registry.payload_keys",
                        lambda uuids: {u.split("-")[0] for u in uuids})
    a = _model(None, pk=1, metadata={
        "training_source_uuid": "ds-1",
        "training_source_summary": {"members": [{"uuid": "h1-a"}, {"uuid": "h2-a"}]}})
    b = _model(None, pk=2, metadata={
        "training_source_uuid": "ds-2",
        "training_source_summary": {"members": [{"uuid": "h1-b"}, {"uuid": "h2-b"}]}})
    report = compare_models(a, b)
    assert "identical PAYLOADS (twin-collapsed)" in report["training_data"]
    assert "training data" not in report["confound_factors"]


def test_deletion_overrides_are_visible():
    """`~key` with no '=' is hydra's deletion; dropping it made a deleted config
    key read as SAME -- the one direction the module promises never to fail."""
    parsed = _parse_overrides(["~scheduler"])
    assert parsed["scheduler"] == "<deleted>"


def test_list_values_get_a_token_delta(monkeypatch):
    """One changed channel in a 12-vs-22 list is displayed, not eyeball-diffed."""
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    a = _model(None, pk=1, profile={"hydra_overrides": ["custom.keys=[x,y,z]"]})
    b = _model(None, pk=2, profile={"hydra_overrides": ["custom.keys=[x,y,z,beam_s]"]})
    report = compare_models(a, b)
    (row,) = [d for d in report["differences"] if d["field"] == "override.custom.keys"]
    assert row["delta"] == {"added": ["beam_s"], "removed": []}


def test_gpu_seat_priority_is_bookkeeping(monkeypatch):
    """The benign list named 'seat_priority' but the real key is
    'gpu_seat_priority', so GPU-queue booking confounded every cross-era pair."""
    assert _severity("gpu_seat_priority") == "benign"


def test_same_name_config_drift_is_visible(monkeypatch):
    """`config_name` is a pointer, not content (2026-08-13 friction).

    Same name, different resolved yaml -- commit c0cc754 flipped amp.enabled
    with the name unchanged -- must surface as config.* rows, not read as same.
    Measured: 14 of 25 config names in the DB resolve to more than one content
    state.
    """
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    configs = {"uuid-1": {"amp": {"enabled": True}, "custom": {"x": 1}},
               "uuid-2": {"amp": {"enabled": False}, "custom": {"x": 1}}}
    monkeypatch.setattr("aiida_pino.utils.model_compare._archived_config_flat",
                        lambda model: __import__("aiida_pino.utils.model_compare",
                                                 fromlist=["_flat_config"])._flat_config(
                                                     configs[model.uuid]))
    report = compare_models(_model(None, pk=1), _model(None, pk=2))
    (row,) = [d for d in report["differences"] if d["field"] == "config.amp.enabled"]
    assert row["severity"] == "confound"
    assert row["a"] is True and row["b"] is False


def test_override_covered_config_keys_are_not_double_reported(monkeypatch):
    """The resolved config CONTAINS applied overrides; the override row tells
    that story once."""
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    configs = {"uuid-1": {"custom": {"seed": 1001}}, "uuid-2": {"custom": {"seed": 1002}}}
    from aiida_pino.utils import model_compare as mc
    monkeypatch.setattr("aiida_pino.utils.model_compare._archived_config_flat",
                        lambda model: mc._flat_config(configs[model.uuid]))
    a = _model(None, pk=1, seed=1001, profile={"hydra_overrides": ["custom.seed=1001"]})
    b = _model(None, pk=2, seed=1002, profile={"hydra_overrides": ["custom.seed=1002"]})
    report = compare_models(a, b)
    # Reported ONCE -- still the contract -- but now by the MATERIALIZED value rather than
    # by the flag text (2026-08-27: the diff should show what Hydra actually saw).
    # When both sides archive a config, the override row is dropped and the config row
    # carries the difference; the factor is the same either way, which is what keeps a knob
    # from being counted twice.
    rows = [d for d in report["differences"] if d["field"].endswith("custom.seed")]
    assert len(rows) == 1, rows
    assert rows[0]["field"] == "config.custom.seed", rows
    assert report["confound_factors"] == ["seed"]


def test_an_unarchived_config_is_an_unknown(monkeypatch):
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    monkeypatch.setattr("aiida_pino.utils.model_compare._archived_config_flat",
                        lambda model: None)
    report = compare_models(_model(None, pk=1), _model(None, pk=2))
    (row,) = [d for d in report["differences"] if d["field"] == "config_content"]
    assert row["severity"] == "unknown"
    assert "config content" in report["unknown_factors"]


def test_cli_render_prints_a_lone_field_once():
    """'override.custom.FNO_input_keys override.custom.FNO_input_keys' -- the
    factor column earns its place only when it differs from the field."""
    from aiida_pino.cli import _render_compare

    report = {
        "a": {"pk": 1, "name": "a"}, "b": {"pk": 2, "name": "b"},
        "verdict": "x", "training_data": "same dataset node",
        "differences": [
            {"field": "override.custom.keys", "factor": "override.custom.keys",
             "a": "1", "b": "2", "severity": "confound"},
            {"field": "stride", "factor": "resolution", "a": "1", "b": "2",
             "severity": "unknown"},
        ],
        "shared_evaluations": [],
    }
    text = _render_compare(report)
    assert text.count("override.custom.keys") == 1
    assert "resolution" in text and "stride" in text


def test_the_referee_compares_what_hydra_actually_applied(monkeypatch):
    """A structured field its own override overrules is dead but still stored.

    7 models record weight_decay 0.0 while training with 0.01-0.1, so comparing
    the stored value compares a number that never trained anything.
    """
    monkeypatch.setattr("aiida_pino.utils.model_compare._eval_rows", lambda uuid: {})
    a = _model(None, pk=1, profile={"weight_decay": 0.0,
                                    "hydra_overrides": ["optimizer.weight_decay=0.01"]})
    b = _model(None, pk=2, profile={"weight_decay": 0.01, "hydra_overrides": []})
    report = compare_models(a, b)
    # identical in effect: one says it in the field, the other in an override
    assert not [d for d in report["differences"] if d["field"] == "profile.weight_decay"]
