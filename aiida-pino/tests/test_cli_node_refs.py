"""Every node argument of the CLI takes a pk or a UUID.

2026-10-07, reproduction test from the deposit: pks are reassigned by `verdi archive import`, so
a reader holding the published UUIDs could not name a node to most subcommands, which took
`type=int`.
"""
from __future__ import annotations

import argparse

import pytest

from aiida_pino.cli import _node_ref, _resolve_node_refs, build_parser


def test_node_ref_parses_pks_and_uuids_and_refuses_labels():
    assert _node_ref("12") == 12 and isinstance(_node_ref("12"), int)
    assert _node_ref("6033e8cd") == "6033e8cd" and not isinstance(_node_ref("6033e8cd"), int)
    with pytest.raises(argparse.ArgumentTypeError):
        _node_ref("fno_twoalloy_combined_r05_m1")


def test_uuids_resolve_to_pks_for_single_and_repeated_arguments(aiida_profile_clean):
    from aiida import orm

    from aiida_pino.console.errors import NodeNotFoundError

    a, b = orm.Data().store(), orm.Data().store()
    args = build_parser().parse_args(["model", a.uuid[:12]])
    _resolve_node_refs(args)
    assert args.pk == a.pk

    args = build_parser().parse_args(["delete", a.uuid, str(b.pk)])
    _resolve_node_refs(args)
    assert args.pks == [a.pk, b.pk]

    args = build_parser().parse_args(["model", "0123456789abcdef"])
    with pytest.raises(NodeNotFoundError):
        _resolve_node_refs(args)


@pytest.mark.parametrize("argv, profile, as_json, foreground", [
    (["--foreground", "--json", "--profile", "p", "models"], "p", True, True),
    (["models", "--foreground", "--json", "--profile", "p"], "p", True, True),
    (["--profile", "p", "models", "--json"], "p", True, False),
    (["models"], "", False, False),
])
def test_global_options_hold_before_and_after_the_subcommand(argv, profile, as_json, foreground):
    # 2026-10-08: the subcommand's own defaults overwrote options given before it, so
    # `aiida-pino --foreground submit eval ...` still asked for a daemon.
    args = build_parser().parse_args(argv)
    assert (args.profile, args.json, args.foreground) == (profile, as_json, foreground)


def test_a_failed_foreground_run_is_an_error(monkeypatch):
    # 2026-10-08: `aiida-pino --foreground submit eval` printed a pk and exited 0 after the
    # evaluation workflow had finished with exit status 300.
    from types import SimpleNamespace

    import aiida.engine
    import aiida.plugins

    from aiida_pino.console import service as service_module
    from aiida_pino.console.errors import OperationFailedError

    failed = SimpleNamespace(pk=7, uuid="28a1d8d6", process_label="EvalOnlyWorkChain", process_type="x",
                             is_finished_ok=False, exit_status=300, exit_message="The evaluate-model step failed.")
    monkeypatch.setenv("AIIDA_PINO_FOREGROUND", "1")
    monkeypatch.setattr(aiida.engine, "run_get_node", lambda workflow, **inputs: (None, failed))
    monkeypatch.setattr(aiida.plugins, "WorkflowFactory", lambda entry_point: object())
    monkeypatch.setattr(service_module, "_label_submitted_process", lambda *a, **k: None)
    monkeypatch.setattr(service_module, "_annotate_submitted_process", lambda *a, **k: None)
    svc = service_module.AiidaConsoleService.__new__(service_module.AiidaConsoleService)
    with pytest.raises(OperationFailedError, match="exit status 300"):
        svc._submit("pino.eval_only")
    ok = SimpleNamespace(**{**vars(failed), "is_finished_ok": True, "exit_status": 0})
    monkeypatch.setattr(aiida.engine, "run_get_node", lambda workflow, **inputs: (None, ok))
    assert svc._submit("pino.eval_only")["pk"] == 7


def test_srun_variables_stay_and_a_batch_jobs_procid_goes():
    from aiida_pino.scripts.live_tail import single_process_environment

    batch = single_process_environment({"SLURM_PROCID": "0", "SLURM_JOB_ID": "1", "PATH": "/bin"})
    assert "SLURM_PROCID" not in batch and batch["SLURM_JOB_ID"] == "1"
    launched = {"SLURM_PROCID": "3", "SLURM_LAUNCH_NODE_IPADDR": "launch-node"}
    assert single_process_environment(launched) == launched
