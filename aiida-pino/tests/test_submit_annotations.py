"""A run says who launched it and why, at submit time rather than afterwards.

Requirement (2026-08-22): a record of who launched a run and what the training is for.

What existed: `aiida-pino describe PK "what and why" --title "..."` after the
fact, and `--title/--description` on `rerun`. The developer notes ask for a
description right after every submission. Measured on the last 12
training processes: **12 carry a label, 0 carry a description.** A rule that
depends on a second step after the interesting part is done is a rule that gets
skipped, and several submitters can share ONE OS user, so AiiDA's own `user`
cannot say which of them it was.

So the fields travel with the submission, and `_submit` is where they land --
the single chokepoint every path already goes through for labelling.

`submitted_by` is free text and deliberately not auto-detected: submitters can
run as the same OS user, so an automatic value would be uniformly true and
useless. What is wanted is which person or tool, which only the caller knows.
"""

from __future__ import annotations

from aiida_pino.console.service import _annotate_submitted_process


class _Node:
    def __init__(self, label="", description=""):
        self.label = label
        self.description = description
        self.extras = {}

    class _Base:
        def __init__(self, outer):
            self._outer = outer

        class _Extras:
            def __init__(self, outer):
                self._outer = outer

            def set(self, key, value):
                self._outer.extras[key] = value

        @property
        def extras(self):
            return self._Extras(self._outer)

    @property
    def base(self):
        return self._Base(self)


def test_description_and_submitter_are_recorded_at_submit():
    node = _Node()
    _annotate_submitted_process(
        node, description="does conduction_field help?", submitted_by="material-vars"
    )
    assert node.description == "does conduction_field help?"
    assert node.extras["submitted_by"] == "material-vars"


def test_a_caller_supplied_description_is_never_overwritten():
    """Same rule the label already follows: a deliberate value wins."""
    node = _Node(description="set by hand")
    _annotate_submitted_process(node, description="from the request", submitted_by=None)
    assert node.description == "set by hand"


def test_absent_annotations_change_nothing():
    """Submitting without them must stay legal -- this is a nudge, not a gate."""
    node = _Node()
    _annotate_submitted_process(node, description=None, submitted_by=None)
    assert node.description == ""
    assert node.extras == {}


def test_blank_strings_are_treated_as_absent():
    node = _Node()
    _annotate_submitted_process(node, description="   ", submitted_by="")
    assert node.description == ""
    assert node.extras == {}


def test_annotation_failure_never_breaks_a_submission():
    """The run has already launched; bookkeeping must not raise over it."""

    class _Hostile(_Node):
        @property
        def base(self):
            raise RuntimeError("no extras here")

    _annotate_submitted_process(_Hostile(), description="x", submitted_by="y")


# --- the eval path accepted `--description` / `--submitted-by` and dropped both (2026-09-14) ---
#
# `EvalOnlyRequest` had neither field, so `_submit_eval` had nowhere to put them and the
# CLI's help text promised what did not happen. Measured: the same
# `--submitted-by` string on the same afternoon is on a training workchain (21 nodes walked)
# and on none of an eval workchain's 18 nodes. Six evals were killed that day because
# every eval workchain in the range had an empty description. An accepted-and-
# ignored flag is worse than an unsupported one: the submitter believes the record is signed.


def test_the_eval_request_carries_description_and_submitter():
    from aiida_pino.console.schemas import EvalOnlyRequest

    req = EvalOnlyRequest(dataset_pk=1, model_pk=2, python_code="pino-python",
                          description="does the fix hold on ds1?", submitted_by="random paths / a3fcfb")
    assert req.description == "does the fix hold on ds1?"
    assert req.submitted_by == "random paths / a3fcfb"


def test_the_eval_cli_passes_both_flags_into_the_request():
    from types import SimpleNamespace

    from aiida_pino import cli

    seen = {}

    class _Svc:
        def submit_eval_only(self, req):
            seen["req"] = req
            return {"pk": 1}

    args = SimpleNamespace(dataset=1, model=2, python_code="pino-python", evaluation_profile=None,
                           derivation_profile=None, full_frames=False, allow_domain_change=False,
                           allow_training_data=False, description="why", submitted_by="who")
    cli._submit_eval(_Svc(), args)
    assert seen["req"].description == "why"
    assert seen["req"].submitted_by == "who"


def test_submit_eval_only_forwards_both_to_the_submit_step(monkeypatch, aiida_profile_clean, local_code_factory,
                                                            pino_dataset_factory, pino_model_factory):
    from aiida_pino.console.config import Settings
    from aiida_pino.console.schemas import EvalOnlyRequest
    from aiida_pino.console.service import AiidaConsoleService

    service = AiidaConsoleService(Settings())
    code = local_code_factory("/bin/bash")
    dataset = pino_dataset_factory()
    model = pino_model_factory()
    captured = {}

    def fake_submit(entry_point, *, description=None, submitted_by=None, **inputs):
        captured.update(entry_point=entry_point, description=description, submitted_by=submitted_by)
        return {"pk": 0}

    monkeypatch.setattr(service, "_submit", fake_submit)
    service.submit_eval_only(EvalOnlyRequest(dataset_pk=dataset.pk, model_pk=model.pk, python_code=code.full_label,
                                             description="why", submitted_by="who"))
    assert captured["entry_point"] == "pino.eval_only"
    assert captured["description"] == "why" and captured["submitted_by"] == "who"
