"""Six unambiguous defects the pre-release audit confirmed (2026-09-14), each pinned here.

Every one was adversarially checked before it was fixed: real, the fix obvious, nothing a run
or a caller relies on changes. The audit brief lists them with the refuted candidates beside them.
"""

from __future__ import annotations

from types import SimpleNamespace


def test_models_all_reaches_the_service_as_limit_zero():
    """`aiida-pino models --all` was a no-op: the flag existed (74890cf) and was never wired,
    so `--all` sent limit=None and the page cap applied. `datasets --all` had the wiring."""
    from aiida_pino import cli

    calls = {}

    class _Svc:
        def list_models(self, **kw):
            calls.update(kw); return []

    args = cli.build_parser().parse_args(["models", "--all"])
    args.run(_Svc(), args)
    assert calls["limit"] == 0


def test_the_adapter_docstrings_say_the_memory_flags_are_passed():
    """train_adapter and kfold_adapter said "No `--vram-mb` is passed" while building
    `seat_args` with `_declared_memory_args`; only eval_adapter was corrected on 2026-08-25."""
    import inspect

    from aiida_pino.scripts import kfold_adapter, train_adapter

    for mod in (train_adapter, kfold_adapter):
        src = inspect.getsource(mod.wrap_with_gpu_seat)
        assert "No ``--vram-mb`` is passed" not in src and "No\n    ``--ram-mb`` either" not in src, mod.__name__
        assert "ARE passed" in src, mod.__name__


def test_seat_priority_comment_states_strict_ordering():
    """profiles.py described the pre-2026-08-07 bounded deference that 2da80a3 removed."""
    import inspect

    from aiida_pino.utils import profiles

    src = inspect.getsource(profiles)
    assert "never starves" not in src
    assert "strict" in src.split('"seat_priority"')[0][-1500:]


def test_the_val_split_caveat_names_a_command_that_parses():
    from aiida_pino import cli
    from aiida_pino.console import service

    src = __import__("inspect").getsource(service.AiidaConsoleService._test_split_caveat)
    assert "aiida-pino submit eval --dataset" in src and "aiida-pino eval --dataset" not in src
    # and the named form really parses
    args = cli.build_parser().parse_args(["submit", "eval", "--dataset", "1", "--model", "2", "--python-code", "c"])
    assert args.kind == "eval"


def test_local_and_remote_gpu_queries_share_one_argv(monkeypatch):
    """`GPU_QUERY_ARGS`/`APPS_QUERY_ARGS` were 'shared so they cannot drift' and used only by
    the remote path; `gpu_snapshot` kept literal copies."""
    from aiida_pino.console import monitor

    seen = []

    def fake_run(argv, **kw):
        seen.append(list(argv)); return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)
    # A one-sided edit to the shared constants must reach the LOCAL path too; the literal
    # copies were byte-identical, which is exactly what let the drift stay invisible.
    monkeypatch.setattr(monitor, "GPU_QUERY_ARGS", ["--query-gpu=index,uuid", "--format=csv"])
    monkeypatch.setattr(monitor, "APPS_QUERY_ARGS", ["--query-compute-apps=pid", "--format=csv"])
    monitor.gpu_snapshot()
    assert seen[0] == ["nvidia-smi", *monitor.GPU_QUERY_ARGS]
    assert seen[1] == ["nvidia-smi", *monitor.APPS_QUERY_ARGS]


def test_warm_is_a_cli_command_because_presubmit_tells_you_to_run_it():
    """`local_build_report` says "or `aiida-pino warm <pk>` first"; until now no such
    subcommand existed (refuted-as-not-mechanical in the audit; made mechanical here)."""
    from aiida_pino import cli

    args = cli.build_parser().parse_args(["warm", "42"])
    assert args.pk == 42
    seen = {}

    class _Svc:
        def warm_runtime_cache(self, pk, **kw):
            seen["pk"] = pk; return {"pk": pk}

    args.run(_Svc(), args)
    assert seen["pk"] == 42
