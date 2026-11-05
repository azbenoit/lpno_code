"""`submit train-eval --init-model PK` -- restart training from a checkpoint.

The workchain, the service layer and the request schema all supported warm
starts already (`TrainModelWorkChain.init_model`, `submit_train_eval`'s
`init_model_pk`), but the CLI never exposed the flag, so the only way to
continue a run was a hand-written `engine.submit` script. That is the shortcut
this closes: a warm start is a first-class CLI operation.

Why it matters: the 2026-07-22 keyhole model saw 0.80 epochs of its data in
32000 steps and lost to a baseline that saw 1.70. Continuing it to 2 epochs is
~47.8k more steps rather than a full 79.8k rerun, so the flag is the difference
between an 18 h and a 30 h job.
"""
from __future__ import annotations

import pytest

from aiida_pino.cli import build_parser
from aiida_pino.console.schemas import TrainEvalRequest


def _parse(*argv: str):
    return build_parser().parse_args(list(argv))


BASE = (
    "submit", "train-eval",
    "--train-dataset", "575",
    "--python-code", "pino-python@local",
)


def test_init_model_is_parsed():
    args = _parse(*BASE, "--init-model", "5594")
    assert args.init_model == 5594


def test_init_model_defaults_to_none():
    """Omitting it must keep the cold-start behaviour unchanged."""
    args = _parse(*BASE)
    assert getattr(args, "init_model", None) is None


def test_init_model_must_be_an_integer_pk():
    with pytest.raises(SystemExit):
        _parse(*BASE, "--init-model", "not-a-pk")


def test_handler_forwards_init_model_to_the_request():
    """The CLI handler must put the pk on the request the service reads."""
    from aiida_pino import cli

    captured = {}

    class _Svc:
        def submit_train_eval(self, req):
            captured["req"] = req
            return {"pk": 1}

    args = _parse(*BASE, "--init-model", "5594", "--mode", "train")
    cli._submit_train_eval(_Svc(), args)
    assert captured["req"].init_model_pk == 5594


def test_handler_omits_init_model_when_absent():
    from aiida_pino import cli

    captured = {}

    class _Svc:
        def submit_train_eval(self, req):
            captured["req"] = req
            return {"pk": 1}

    cli._submit_train_eval(_Svc(), _parse(*BASE))
    assert captured["req"].init_model_pk is None


def test_request_schema_accepts_the_field():
    req = TrainEvalRequest(
        train_dataset_pk=575, python_code="pino-python@local", init_model_pk=5594
    )
    assert req.init_model_pk == 5594


def test_help_mentions_the_warm_start_flag(capsys):
    """Discoverability is the point -- the flag must be documented in --help."""
    with pytest.raises(SystemExit):
        _parse("submit", "train-eval", "--help")
    out = capsys.readouterr().out
    assert "--init-model" in out
    assert "warm start" in out.lower() or "checkpoint" in out.lower()
