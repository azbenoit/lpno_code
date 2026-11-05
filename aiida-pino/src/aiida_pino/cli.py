"""``aiida-pino`` CLI: an argparse front-end over :class:`AiidaConsoleService`.

Talks to the service layer directly (no HTTP). Heavy imports (aiida, pydantic,
pydantic-settings) happen only after argument parsing, so ``aiida-pino --help``
is instant and works without an AiiDA profile or the ``web`` extra installed.

Every ``*-profile`` flag is repeatable and accepts either ``@file.json`` (a JSON
object file) or ``key=value`` tokens, where the value is parsed as JSON with a
plain-string fallback (``lr=0.001``, ``wandb_offline=true``, ``folds=[1,2]``,
``run_name=my-run``). Later tokens override earlier ones.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from aiida_pino.split_values import SPLIT_VALUES


# --------------------------------------------------------------------- profiles
def parse_profile(tokens: list[str] | None) -> dict[str, Any] | None:
    """Merge profile tokens into one dict; None in => None out (pydantic default)."""
    if tokens is None:
        return None
    out: dict[str, Any] = {}
    for tok in tokens:
        if tok.startswith("@"):
            data = json.loads(Path(tok[1:]).expanduser().read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{tok[1:]}: profile file must contain a JSON object")
            out.update(data)
        elif tok.lstrip().startswith("{"):
            # A whole JSON document pasted where a token was expected. Without
            # this it partitions on the first "=" inside the document and the
            # error names a key from the user's own profile, so it reads as a
            # complaint about that key rather than about the shape (2026-08-10
            # friction, item 3: two failed attempts, ~20 minutes).
            raise ValueError(
                "this looks like a whole JSON document, not a profile token. "
                "Write it to a file and pass @file.json, or use key=value tokens. "
                "`aiida-pino profile-keys training` lists every key with its default."
            )
        elif "=" in tok:
            key, _, raw = tok.partition("=")
            try:
                out[key] = json.loads(raw)
            except json.JSONDecodeError:
                out[key] = raw
        else:
            raise ValueError(f"bad profile token {tok!r}: expected @file.json or key=value")
    return out


def _schemas():
    from aiida_pino.console import schemas

    return schemas


# ------------------------------------------------------------- submit handlers
def _submit_train_eval(svc, a):
    req = _schemas().TrainEvalRequest(
        mode=a.mode,
        train_dataset_pk=a.train_dataset,
        val_dataset_pk=a.val_dataset,
        python_code=a.python_code,
        training_profile=parse_profile(a.training_profile) or {},
        derivation_profile=parse_profile(a.derivation_profile),
        evaluation_profile=parse_profile(a.evaluation_profile),
        init_model_pk=getattr(a, "init_model", None),
        description=getattr(a, "description", None),
        submitted_by=getattr(a, "submitted_by", None),
    )
    return svc.submit_train_eval(req)


def _node_pk(value) -> int:
    """A node given by PK or by UUID (or a unique UUID prefix). PKs are reassigned by
    `verdi archive import`, so a reader holding the deposit's UUIDs needs the second form."""
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    from aiida import orm

    return orm.load_node(text).pk


class _NodeRef(str):
    """A node named on the command line by UUID; `_resolve_node_refs` turns it into its PK once
    the profile is loaded (argparse runs before that, so it cannot look the node up itself)."""


def _node_ref(value: str):
    """argparse type for a node argument: a PK, or a UUID or unique UUID prefix."""
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if not re.fullmatch(r"[0-9a-fA-F][0-9a-fA-F-]{3,35}", text):
        raise argparse.ArgumentTypeError(f"{text!r} is neither a pk nor a uuid")
    return _NodeRef(text)


def _resolve_node_refs(args) -> None:
    """Replace every UUID given for a node argument with that node's PK, in place."""
    from aiida import orm
    from aiida.common.exceptions import MultipleObjectsError, NotExistent

    from aiida_pino.console.errors import NodeNotFoundError

    def pk(ref):
        if not isinstance(ref, _NodeRef):
            return ref
        try:
            return orm.load_node(str(ref)).pk
        except (NotExistent, MultipleObjectsError) as exc:
            raise NodeNotFoundError(f"no single node with uuid {ref}: {exc}") from exc

    for key, value in list(vars(args).items()):
        if isinstance(value, list):
            setattr(args, key, [pk(item) for item in value])
        else:
            setattr(args, key, pk(value))


def _submit_eval(svc, a):
    # `full_frames` was reachable only by hand-writing it into an evaluation profile,
    # so the documented escape hatch from the auto-subset had no command-line form.
    evaluation = parse_profile(a.evaluation_profile) or {}
    if getattr(a, "full_frames", False):
        evaluation["full_frames"] = True
    req = _schemas().EvalOnlyRequest(
        dataset_pk=_node_pk(a.dataset),
        model_pk=_node_pk(a.model),
        python_code=a.python_code,
        evaluation_profile=evaluation or None,
        derivation_profile=parse_profile(a.derivation_profile),
        allow_domain_change=bool(getattr(a, "allow_domain_change", False)),
        allow_training_data=bool(getattr(a, "allow_training_data", False)),
        description=getattr(a, "description", None),
        submitted_by=getattr(a, "submitted_by", None),
    )
    return svc.submit_eval_only(req)


def _submit_kfold(svc, a):
    req = _schemas().KfoldRequest(
        dataset_pk=a.dataset,
        python_code=a.python_code,
        training_profile=parse_profile(a.training_profile) or {},
        derivation_profile=parse_profile(a.derivation_profile),
    )
    return svc.submit_kfold(req)


def _submit_raw_flow3d(svc, a):
    req = _schemas().RawFlow3DImportRequest(
        source_path=a.source_path,
        python_code=a.python_code,
        dataset_profile=parse_profile(a.dataset_profile) or {},
        materials_root=a.materials_root,
    )
    return svc.submit_raw_flow3d(req)


def _submit_flow3d(svc, a, batch=False):
    sch = _schemas()
    common = dict(
        python_code=a.python_code,
        execution_profile=parse_profile(a.execution_profile) or {},
        dataset_profile=parse_profile(a.dataset_profile) or {},
        materials_root=a.materials_root,
        target_dataset_pk=getattr(a, "target_dataset", None),
    )
    if batch:
        configs = [parse_profile([tok]) or {} for tok in (a.config or [])]
        return svc.submit_flow3d_batch(
            sch.Flow3DBatchRequest(run_configs=configs, max_concurrent_runs=a.max_concurrent, **common)
        )
    return svc.submit_flow3d_generate(
        sch.Flow3DGenerateRequest(run_config=parse_profile(a.run_config) or {}, **common)
    )


def _submit_import_model(svc, a):
    req = _schemas().ImportModelRequest(
        python_code=a.python_code,
        training_source_pk=_node_pk(a.training_source),
        validation_source_pk=_node_pk(a.validation_source) if getattr(a, "validation_source", None) else None,
        artifacts_pk=a.artifacts_pk,
        artifacts_path=a.artifacts_path,
        training_profile=parse_profile(a.training_profile) or {},
        derivation_profile=parse_profile(a.derivation_profile),
        model_name=a.model_name,
        dry_run=bool(getattr(a, "dry_run", False)),
    )
    return svc.submit_import_model(req)


def _submit_derive(svc, a):
    req = _schemas().DeriveDatasetRequest(
        dataset_pk=a.dataset,
        derivation_profile=parse_profile(a.derivation_profile) or {},
    )
    return svc.submit_derive(req)


# ---------------------------------------------------------------------- parser
def _kw(p: argparse.ArgumentParser) -> None:
    p.add_argument("--keyword", action="append", default=None, metavar="K", help="filter keyword (repeatable, AND)")


def _profile_arg(p: argparse.ArgumentParser, flag: str) -> None:
    # The key list is the thing that unblocks a first-time caller, and it used to
    # be discoverable only by triggering a validation error (2026-08-10, item 4).
    kind = flag.replace("--", "").replace("-profile", "")
    p.add_argument(
        flag, action="append", default=None, metavar="@FILE|KEY=VAL",
        help="repeatable; @file.json or key=value (value parsed as JSON, string "
             f"fallback). `aiida-pino profile-keys {kind}` lists every key with its default",
    )


def _common_options(*, defaults: bool) -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    unset = {} if defaults else {"default": argparse.SUPPRESS}
    common.add_argument("--profile", help="AiiDA profile name (default: configured default profile)",
                        **({"default": ""} if defaults else unset))
    common.add_argument("--json", action="store_true", help="print raw JSON instead of tables", **unset)
    common.add_argument("--foreground", action="store_true",
                        help="run a submission in this terminal until it finishes; needs no daemon or RabbitMQ",
                        **unset)
    return common


def build_parser() -> argparse.ArgumentParser:
    # The options are accepted before and after the subcommand. The subcommands' copies carry no
    # default: argparse lets a subparser's default overwrite what was parsed before it, so
    # `aiida-pino --foreground submit eval ...` used to run with foreground off, and
    # `--profile`/`--json` before the subcommand were ignored the same way (2026-10-08).
    parser = argparse.ArgumentParser(prog="aiida-pino", description="Browse, annotate, and submit aiida-pino workflows.",
                                     parents=[_common_options(defaults=True)])
    top = parser.add_subparsers(dest="command", required=True, metavar="command")
    common = _common_options(defaults=False)

    def add(sub, name, run=None, help=None):
        p = sub.add_parser(name, help=help, description=help, parents=[common])
        if run is not None:
            p.set_defaults(run=run)
        return p

    # --- environment / browse
    add(top, "env", lambda s, a: s.env_status(), "environment status (version, profile, daemon, codes, configs)")
    add(top, "health", lambda s, a: s.health(), "service health and daemon reachability")
    add(top, "codes", lambda s, a: s.list_codes(), "installed AiiDA codes by role")
    add(top, "configs", lambda s, a: s.config_names(), "available training config names")
    p = add(top, "training-defaults", lambda s, a: s.training_defaults(a.config), "default training profile values")
    p.add_argument("--config", default=None, help="config name (default: package default)")

    p = add(top, "datasets", lambda s, a: s.list_datasets(search=a.search, role=a.role, keywords=a.keyword, limit=0 if a.all else a.limit, include_superseded=a.superseded), "list dataset nodes")
    p.add_argument("--search", default=None)
    p.add_argument("--role", default=None)
    _kw(p)
    p.add_argument("--limit", type=int, default=None, help="page size; 0 lists everything")
    p.add_argument("--superseded", action="store_true", help="also list datasets that were replaced by a newer node")
    p.add_argument("--all", action="store_true", help="list every dataset (no page cap; same as --limit 0)")

    p = add(top, "dataset", lambda s, a: s.dataset_detail(a.pk), "dataset detail")
    p.add_argument("pk", type=_node_ref)

    p = add(top, "models", lambda s, a: s.models_trained_on(a.trained_on) if a.trained_on else s.list_models(search=a.search, evaluated_only=a.evaluated_only, keywords=a.keyword, limit=0 if a.all else a.limit), "list model nodes")
    p.add_argument("--trained-on", type=_node_ref, default=None, metavar="PK",
                   help="everything that trained on dataset PK (follows splits and layout variants; includes in-flight runs and evals, which are listed separately)")
    p.add_argument("--search", default=None)
    p.add_argument("--evaluated-only", action="store_true")
    p.add_argument("--all", action="store_true", help="list every model (same as --limit 0)")
    _kw(p)
    p.add_argument("--limit", type=int, default=None)

    p = add(top, "model", lambda s, a: s.model_detail(a.pk), "model detail")
    p.add_argument("pk", type=_node_ref)

    p = add(top, "compare-models", lambda s, a: s.compare_models(a.a, a.b),
            "every difference between two models, classified confound/caution/benign "
            "-- the ablation referee")
    p.add_argument("a", type=_node_ref, metavar="PK_A")
    p.add_argument("b", type=_node_ref, metavar="PK_B")

    add(top, "kfolds", lambda s, a: s.list_kfold_runs(), "list k-fold run nodes")
    p = add(top, "kfold", lambda s, a: s.kfold_detail(a.pk), "k-fold run detail")
    p.add_argument("pk", type=_node_ref)

    p = add(top, "sims", lambda s, a: s.list_simulations(source=a.source, folder=a.folder, split=a.split, keywords=a.keyword, limit=a.limit), "list simulation nodes")
    p.add_argument("--source", default=None)
    p.add_argument("--folder", default=None)
    p.add_argument("--split", default=None)
    _kw(p)
    p.add_argument("--limit", type=int, default=None)

    p = add(top, "case-stats", lambda s, a: s.simulation_stats(a.pk, recompute=a.recompute), "melt-pool statistics for one simulation (cached in extras)")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--recompute", action="store_true", help="ignore the cached value and recompute")

    def _dataset_stats(s, a):
        if a.compute:
            remaining = None
            while remaining is None or remaining > 0:
                r = s.dataset_stats_compute(a.pk, limit=25)
                remaining = r["remaining"]
                print(f"computed {r['total'] - remaining}/{r['total']}\r", end="", flush=True)
            print()
        return s.dataset_stats(a.pk)["summary"]

    p = add(top, "dataset-stats", _dataset_stats, "aggregate melt-pool statistics for a dataset")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--compute", action="store_true", help="compute missing per-case stats first (batched)")

    def _plot(s, a):
        png = s.simulation_plot(a.pk, field=a.field, view=a.view, frame=a.frame, contour=not a.no_contour)
        out = a.output or f"sim{a.pk}_{a.field.replace(' ', '_')}_{a.view}_f{a.frame}.png"
        with open(out, "wb") as fh:
            fh.write(png)
        return {"written": out, "bytes": len(png)}

    p = add(top, "plot", _plot, "render one case slice frame to a PNG file")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--field", default="Temperature")
    p.add_argument("--view", default="long", choices=["long", "surface", "cross"])
    p.add_argument("--frame", type=int, default=-1, help="frame index (-1 = last)")
    p.add_argument("--no-contour", action="store_true")
    p.add_argument("-o", "--output", default=None)

    p = add(top, "ps", lambda s, a: s.list_processes(limit=a.limit, label_filter=a.label, only_pino=not a.all, include_smokes=a.smokes, state=a.state, recent=a.recent), "list processes (pino-only by default)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--recent", action="store_true",
                   help="order by mtime (most recently CHANGED first) instead of ctime")
    p.add_argument("--label", default=None, help="substring filter on the process label/type")
    p.add_argument("--all-processes", "--all", dest="all", action="store_true",
                   help="include non-pino processes (this does NOT lift the page cap; use --limit 0 for that)")
    p.add_argument("--smokes", action="store_true", help="include VRAM smoke runs (hidden by default)")
    p.add_argument(
        "--state",
        default=None,
        choices=["all", "queued", "finished", "failed"],
        help="filter in the query, not on the fetched page: failed = excepted, killed, or a non-zero exit",
    )

    p = add(top, "show", lambda s, a: s.process_detail(a.pk), "process detail with outputs and metrics")
    p.add_argument("pk", type=_node_ref)
    p = add(top, "logs", lambda s, a: s.process_logs(a.pk), "process report and captured console output")
    p.add_argument("pk", type=_node_ref)
    p = add(top, "tail", lambda s, a: s.process_tail(a.pk, lines=a.lines, path=a.path), "one-shot tail of a running CalcJob's live log")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--lines", type=int, default=200)
    p.add_argument("--path", default=None, help="remote file relative to the job workdir")

    # --- annotations
    p = add(top, "tag", lambda s, a: s.update_annotations(a.pk, keywords_set=a.set, keywords_add=a.add, keywords_remove=a.remove), "edit node keywords (no flags: show current)")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--add", nargs="+", action="extend", default=None, metavar="K")
    p.add_argument("--remove", nargs="+", action="extend", default=None, metavar="K")
    p.add_argument("--set", nargs="+", action="extend", default=None, metavar="K")

    p = add(top, "split", lambda s, a: s.update_annotations(a.pk, split=a.value, split_reason=a.reason), f"assign a simulation to one of {'/'.join(SPLIT_VALUES)}")
    p.add_argument("pk", type=_node_ref)
    # Naming the legal values fixes a real confusion: `eval` is legal and
    # held-out, and a reader who could not see that from the CLI concluded it
    # was corrupt data and overwrote ten nodes (2026-08-19). Deliberately NOT an
    # argparse `choices=`: this CLI's contract is a single-line "error:" and exit
    # 1, which argparse's usage dump and exit 2 would break.
    p.add_argument("value", metavar=f"{{{','.join(SPLIT_VALUES)}}}")
    p.add_argument("--reason", default=None, help="why, recorded in the node's split_audit")

    p = add(
        top,
        "presubmit",
        lambda s, a: s.presubmit_train(_schemas().PresubmitRequest(
            train_dataset_pk=a.train_dataset,
            val_dataset_pk=a.val_dataset,
            python_code=a.python_code,
            training_profile=parse_profile(a.training_profile) or {},
            derivation_profile=parse_profile(a.derivation_profile),
            allow_split_violation=a.allow_split_violation,
        )),
        "dry-run a training submission: resolve the derived collection, check remote staging, report every refusal -- submit nothing",
    )
    p.add_argument("--train-dataset", type=_node_ref, required=True)
    p.add_argument("--val-dataset", type=_node_ref, default=None)
    p.add_argument("--python-code", default=None, help="code label; the profile's code_label still wins")
    # `extend`, not a bare nargs="*": repeating the flag, as `submit` requires, used to
    # keep only the LAST occurrence, so a presubmit of the exact submit command checked
    # a different profile. Several tokens after one flag still work.
    p.add_argument("--training-profile", nargs="*", action="extend", default=None,
                   metavar="@FILE|KEY=VAL",
                   help="repeatable, like submit; several tokens per flag also accepted")
    p.add_argument("--derivation-profile", nargs="*", action="extend", default=None,
                   metavar="@FILE|KEY=VAL",
                   help="repeatable, like submit; several tokens per flag also accepted")
    p.add_argument("--allow-split-violation", action="store_true")

    def _staged(s, a):
        if a.check:
            derivation = None
            if a.derivation:
                import json as _json
                from pathlib import Path as _Path

                raw = a.derivation
                if raw.startswith("@"):
                    raw = _Path(raw[1:]).read_text()
                derivation = _json.loads(raw)
            return s.staged_check(
                a.check, computer=a.computer, derivation=derivation, fields=a.fields
            )
        return s.staged_datasets(computer=a.computer, verify=a.verify)

    p = add(
        top,
        "staged",
        _staged,
        "LIST the runtime caches on a computer, or --check whether the collection a "
        "SUBMISSION would consume is staged. The bare listing shows BASE datasets and cannot "
        "answer the submission question: a remote job consumes whatever the derivation "
        "resolves to, so seeing your uuid listed is not confirmation.",
    )
    p.add_argument("--computer", required=True, help="label of the computer to inspect")
    p.add_argument("--verify", action="store_true", help="also ask the box whether each path is still there")
    p.add_argument("--check", metavar="DATASET", help="pk or uuid: answer the SUBMISSION question for it")
    p.add_argument("--derivation", metavar="JSON|@FILE", help="derivation profile to resolve before checking")
    p.add_argument("--fields", nargs="+", metavar="FIELD", help="runtime fields the job will request")

    def _unstage(s, a):
        derivation = None
        if a.derivation:
            import json as _json
            from pathlib import Path as _Path

            raw = a.derivation
            if raw.startswith("@"):
                raw = _Path(raw[1:]).read_text()
            derivation = _json.loads(raw)
        return s.unstage_remote_dataset(
            a.dataset, computer=a.computer, derivation=derivation,
            fields=a.fields, dry_run=not a.yes,
        )

    p = add(
        top,
        "unstage",
        _unstage,
        "DELETE a staged runtime cache from a computer and retract its record (dry run "
        "unless --yes). Nothing else reclaims that disk: retraction alone only marks the "
        "record superseded and leaves the bytes on the remote.",
    )
    p.add_argument("dataset", help="pk or uuid of the dataset whose cache should go")
    p.add_argument("--computer", required=True, help="label of the computer to clear it from")
    p.add_argument("--derivation", metavar="JSON|@FILE", help="derivation profile, to target the derived collection")
    p.add_argument("--fields", nargs="+", metavar="FIELD", help="runtime fields the cache was staged with")
    p.add_argument("--yes", action="store_true", help="actually delete; without it this is a dry run")

    p = add(
        top,
        "compare",
        lambda s, a: s.compare_evaluations(a.pks),
        "check whether evaluations measure the same thing (dataset, split, derivation, grid, strides)",
    )
    p.add_argument("pks", type=_node_ref, nargs="+", metavar="PK")

    p = add(
        top,
        "stage-remote",
        lambda s, a: s.stage_remote_dataset(
            a.pk,
            computer=a.computer,
            fields=a.fields,
            remote_root=a.remote_root,
            dry_run=a.dry_run,
            for_training=a.for_training,
        ),
        "copy a dataset's runtime cache to a remote computer (needed before any job there)",
    )
    p.add_argument("pk", metavar="PK_OR_UUID", help="dataset collection pk, or its uuid")
    p.add_argument("--computer", required=True, help="label of the computer to stage onto")
    p.add_argument("--fields", nargs="+", action="extend", default=None, metavar="F",
                   help="runtime fields; must match the training profile, or the job looks for a different cache")
    p.add_argument("--remote-root", default=None,
                   help="where caches live on that box (default: its own $HOME/.cache/aiida-pino/runtime)")
    p.add_argument("--dry-run", action="store_true", help="estimate size and member count; build nothing, transfer nothing")
    p.add_argument("--for-training", action="store_true",
                   help="pk is a past training: stage every collection its jobs actually consumed "
                        "(a job reads a DERIVED collection, not the dataset you passed)")

    p = add(top, "warm", lambda s, a: s.warm_runtime_cache(a.pk, runtime_fields=a.field or None),
            "build a collection's local runtime cache now, before a job needs it (the staging note names this)")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--field", action="append", default=None, metavar="NAME",
                   help="runtime field to keep (repeatable); must match what the training passes, it is part of the cache name")

    p = add(top, "describe", lambda s, a: s.update_annotations(a.pk, description=a.text, title=a.title), "set a node's description (and optionally its short title)")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("text")
    p.add_argument("--title", default=None, help="short display name (AiiDA node label; shown bold in the console Runs list)")

    p = add(top, "folder", lambda s, a: s.set_folder(a.pks, a.path), "file datasets under a folder (empty path unfiles them)")
    p.add_argument("path", help="slash-separated folder, e.g. reference/keyhole; '' to unfile")
    p.add_argument("pks", type=_node_ref, nargs="+")

    add(top, "folders", lambda s, a: s.list_folders(include_computed=True),
        "list dataset folders (computed ones are marked and cannot be renamed or removed)")

    p = add(top, "profile-keys", lambda s, a: s.profile_keys(a.which, config_name=a.config),
            "list every accepted key of a profile, with its default")
    p.add_argument("which", nargs="?", default="training",
                   choices=["training", "evaluation", "derivation", "flow3d_run"])
    p.add_argument("--config", default=None,
                   help="training config to read defaults from (else the packaged default's, "
                        "which is another config's answer to your question)")

    p = add(top, "resubmit", lambda s, a: s.resubmit_training(
                _node_pk(a.pk),
                overrides=parse_profile(a.override),
                run_name=a.run_name,
                mode=a.mode,
                title=a.title,
                description=a.description,
            ),
            "re-run a past training with its own inputs, optionally overridden")
    p.add_argument("pk", metavar="PK_OR_UUID", help="the training run to clone (its workflow node)")
    p.add_argument("--override", action="append",
                   help="`derivation.<key>=<value>` reaches the cloned derivation profile (e.g. derivation.derived_fields=[]); anything else the training profile. key=value applied to the cloned training_profile (repeatable)")
    p.add_argument("--run-name", help="run_name for the new run (default: <original>_rerun)")
    p.add_argument("--mode", choices=["train", "train_eval"], default=None)
    p.add_argument("--title", help="short title for the Runs list "
                   "(REPLACES the node label, including one set at submit)")
    p.add_argument("--description", help="what and why")

    p = add(top, "folder-rename", lambda s, a: s.rename_folder(a.label, a.new), "rename a folder (merges when the target exists)")
    p.add_argument("label")
    p.add_argument("new")

    p = add(top, "folder-remove", lambda s, a: s.remove_folder(a.label), "delete a folder; its datasets are unfiled, not deleted")
    p.add_argument("label")

    def _file_orphans(_service, a):
        """File the completed cases of a killed batch into a named base dataset.

        A kill skips `finalize`, so a batch killed mid-sweep leaves every case it
        had finished ingested but unfiled -- invisible in the dataset views and
        indistinguishable from cases that never ran (one killed batch: 20
        ingested, 0 filed). This is a RECOVERY command, not an on-kill hook: after a kill you
        usually want to split valid from invalid before filing, which a hook
        would bundle.
        """
        from aiida import orm

        from aiida_pino.data.dataset_hierarchy import create_base_dataset, iter_base_datasets
        from aiida_pino.utils.orphan_recovery import (
            batch_dataset_profile,
            completed_children_of_batch,
            plan_orphan_filing,
        )

        completed = completed_children_of_batch(a.pk)
        owned = set()
        for node in iter_base_datasets():
            owned |= {str(u) for u in (node.base.attributes.all.get("member_simulation_uuids") or [])}
        profile = batch_dataset_profile(a.pk)
        name = a.name or str(profile.get("batch_keyword") or "") or f"batch_{a.pk}_recovered"
        plan = plan_orphan_filing(
            completed=completed,
            owned_uuids=owned,
            dataset_name=name,
            only_cases=(a.cases.split(",") if a.cases else None),
        )
        if not a.yes:
            plan["dry_run"] = True
            plan["hint"] = "pass --yes to file; --cases a,b to file a subset; --name to override"
            return plan
        # Loading 20 npz_file simulations and running the create calcfunction
        # takes minutes, not seconds. Report progress on stderr so a long run
        # does not look hung -- a first user wrapped this in `timeout 400` and
        # SIGTERMed it mid-operation twice.
        import sys as _sys

        total = len(plan["to_file"])
        sims = []
        for i, row in enumerate(plan["to_file"], 1):
            print(f"  loading {i}/{total} {row.get('case_name') or row['uuid'][:8]}",
                  file=_sys.stderr, flush=True)
            sims.append(orm.load_node(row["uuid"]))
        if not sims:
            return {**plan, "created": False, "note": "nothing to file"}
        print(f"  filing {total} case(s) into '{name}' (this can take several minutes)",
              file=_sys.stderr, flush=True)
        node, created = create_base_dataset(
            sims,
            name=name,
            extra_metadata={
                "description": a.description or profile.get("description"),
                "keywords": profile.get("keywords"),
                "source_batch_pk": a.pk,
                "recovered_from_killed_batch": True,
            },
        )
        return {**plan, "dataset_pk": node.pk, "created": created, "filed": len(sims)}

    p = add(top, "file-orphans", _file_orphans,
            "file a KILLED batch's completed cases into a base dataset (dry run unless --yes; a 20-case batch takes several MINUTES with --yes -- do not wrap it in a short timeout)")
    p.add_argument("pk", type=_node_ref, help="the killed Flow3DBatchWorkChain")
    p.add_argument("--yes", action="store_true", help="actually file (default is a dry-run report)")
    p.add_argument("--name", default=None,
                   help="dataset name (default: the batch's own dataset_profile.batch_keyword)")
    p.add_argument("--cases", default=None,
                   help="comma-separated case names to file, for splitting valid from invalid")
    p.add_argument("--description", default=None, help="override the batch's description")

    p = add(top, "evict", lambda s, a: s.evict_dataset_data(a.pk, dry_run=not a.yes, reason=a.reason),
            "delete a dataset's array data, keeping metadata + provenance (dry run unless --yes)")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--yes", action="store_true", help="actually evict (default is a dry-run report)")
    p.add_argument("--reason", default=None, help="note stored on each simulation's data_evicted extra")

    p = add(top, "restore-payloads", lambda s, a: s.restore_deposit_payloads(
                a.deposit, collections=a.collection, for_nodes=a.for_node, all_evicted=a.all,
                dry_run=not a.yes),
            "put a published deposit's simulation arrays back into the imported nodes, same UUIDs "
            "(dry run unless --yes)")
    p.add_argument("--deposit", required=True,
                   help="the deposit directory: payload_manifest.csv and the payloads_<family>.tar files")
    p.add_argument("--collection", action="append", default=[], metavar="PK_OR_UUID",
                   help="restore this dataset collection's members (repeatable)")
    p.add_argument("--for", dest="for_node", action="append", default=[], metavar="PK_OR_UUID",
                   help="restore what this evaluation scored, or what this training workflow (or the "
                        "model it produced) trained and validated on (repeatable)")
    p.add_argument("--all", action="store_true", help="restore every evicted simulation in the profile")
    p.add_argument("--yes", action="store_true", help="actually restore (default is a dry-run report)")

    def _derive_fields(s, a):
        from aiida_pino.utils.derived_fields import compute_and_store, seed_local_cache

        if a.seed_only:
            return {"seeded": seed_local_cache(a.pk)}
        stride = tuple(int(v) for v in str(a.stride).split(","))
        return compute_and_store(a.pk, stride=stride, limit=a.limit)

    p = add(top, "derive-fields", _derive_fields,
            "SPEED-UP ONLY, never what makes a channel available: precompute expensive "
            "conditioning fields (conduction_field) for a dataset's members and seed the "
            "loader cache the datasource builder reads. The staged-derived path does NOT "
            "call that builder, so this cannot add a channel to a derived dataset -- for "
            "that, put the field in the derivation profile's `derived_fields` instead, "
            "which builds it in the lab frame before the crop and average. Running this "
            "first only turns the ~6.6 s/case build into a cache read.")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--stride", default="1,1,1", help="grid stride sx,sy,sz the runs will train at (default full res)")
    p.add_argument("--limit", type=int, default=None, help="only the first N members (smoke)")
    p.add_argument("--seed-only", action="store_true", help="materialise already-stored files into the local cache, compute nothing")

    p = add(top, "reclaim", lambda s, a: s.reclaim_space(dry_run=not a.yes),
            "reclaim disk from evicted/unreferenced objects, repacking only affected packs (dry run unless --yes)")
    p.add_argument("--yes", action="store_true", help="actually delete + repack (default is a dry-run report; needs the daemon and console stopped)")

    p = add(top, "compress", lambda s, a: s.compress_dataset_data(a.pk, dry_run=not a.yes),
            "convert legacy arrays-mode members to compressed npz_file, verified lossless (dry run unless --yes)")
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--yes", action="store_true", help="actually convert (default is a dry-run report)")

    # --- previews
    pv = add(top, "preview", help="broker-free previews (collection compatibility)")
    pvsub = pv.add_subparsers(dest="what", required=True, metavar="what")
    p = add(pvsub, "collection", lambda s, a: s.preview_collection(a.pks), "validate simulations as collection members")
    p.add_argument("pks", nargs="+", type=_node_ref)

    # --- collections
    c = add(top, "collection", help="dataset collection operations")
    csub = c.add_subparsers(dest="action", required=True, metavar="action")
    p = add(
        csub,
        "create",
        lambda s, a: s.create_collection(
            a.pks, a.name, description=a.description, keywords=a.keyword, dataset_pks=a.dataset_pk,
            skip_description=a.no_description,
            field_policy="intersect" if a.intersect_fields else None,
        ),
        "compose a collection from simulation pks and/or whole datasets (role: composed)",
    )
    p.add_argument("name")
    p.add_argument("pks", nargs="*", type=_node_ref)
    p.add_argument("--dataset-pk", action="append", type=_node_ref, default=[], help="union all members of this dataset (repeatable)")
    p.add_argument("--intersect-fields", action="store_true",
                   help="admit members whose field sets differ; the collection trains on the shared fields only")
    p.add_argument("--description", default=None, help="what this dataset is for and how it was selected (required)")
    p.add_argument("--no-description", action="store_true", help="create it without one, deliberately")
    _kw(p)

    p = add(csub, "eval-only", lambda s, a: s.mark_evaluation_only(a.pk),
            "reserve a dataset for evaluation: every member becomes split=test, and training refuses it")
    p.add_argument("pk", type=_node_ref)

    p = add(
        csub,
        "assign-splits",
        lambda s, a: s.assign_splits(
            a.pk, train=a.train, val=a.val, test=a.test, only_unassigned=not a.reshuffle, seed=a.seed
        ),
        "randomly assign train/val/test splits to a collection's members",
    )
    p.add_argument("pk", type=_node_ref)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--reshuffle", action="store_true", help="also reassign members that already have a split")
    p.add_argument("--seed", type=int, default=0)

    p = add(top, "flow3d-queue", lambda s, a: s.flow3d_queue_status(), "FLOW-3D spool queue and seat-lock status")
    p = add(top, "paused", lambda s, a: s.paused_work(),
            "everything deliberately parked, with how long it has been parked")

    def _timing(_service, a):
        from aiida_pino.utils.timing import report

        return report(solo_only=a.solo)

    p = add(top, "timing", _timing,
            "how long each operation takes: median and IQR, normalised by work done")
    p.add_argument("--solo", action="store_true",
                   help="only runs that were effectively alone on the box (needs a contention witness)")

    # --- destructive
    p = add(top, "delete", lambda s, a: s.delete_nodes(a.pks, dry_run=not a.force), "delete nodes (dry run unless --force)")
    p.add_argument("pks", nargs="+", type=_node_ref)
    p.add_argument("--force", action="store_true", help="actually delete (default is a dry run)")

    p = add(top, "kill", lambda s, a: s.kill_process(a.pk), "kill a running process (needs the daemon)")
    p.add_argument("pk", type=_node_ref)

    # --- submits
    sm = add(top, "submit", help="submit a workflow (needs the daemon)")
    smsub = sm.add_subparsers(dest="kind", required=True, metavar="kind")

    p = add(smsub, "train-eval", _submit_train_eval, "train (and optionally evaluate) a model")
    p.add_argument("--description", default=None,
                   help="what this run is for and what question it answers (recorded on the node)")
    p.add_argument("--submitted-by", default=None,
                   help="who or what launched it; free text, since several people or "
                        "tools may submit as the same OS user")
    p.add_argument("--mode", choices=("train", "train_eval"), default="train_eval")
    p.add_argument("--train-dataset", type=_node_ref, required=True, metavar="PK")
    p.add_argument("--val-dataset", type=_node_ref, default=None, metavar="PK")
    p.add_argument("--python-code", required=True)
    p.add_argument(
        "--init-model",
        type=_node_ref,
        default=None,
        metavar="PK",
        help=(
            "warm start from this model's checkpoints. Resumes the optimizer, LR "
            "schedule and step counter when they can be restored cleanly (max_steps "
            "then counts TOTAL steps); otherwise loads weights only and restarts at "
            "step 0 (max_steps counts THIS run's steps). The adapter prints which, "
            "and records it as resumed_optimizer in train_summary.json."
        ),
    )
    _profile_arg(p, "--training-profile")
    _profile_arg(p, "--derivation-profile")
    _profile_arg(p, "--evaluation-profile")

    p = add(smsub, "eval", _submit_eval, "evaluate an existing model on a dataset")
    # `--submitted-by` and `--description` accepted here too. `train-eval` took them and
    # `eval` did not, so the same habit produced a usage error on one subcommand and not the
    # other -- and a piped invocation swallowed the message, costing a submission round to
    # discover. An asymmetry between two sibling subcommands is a trap whatever the reason.
    p.add_argument("--submitted-by", default=None,
                   help="who or what launched it; free text, since several people or "
                        "tools may submit as the same OS user")
    p.add_argument("--description", default=None,
                   help="what this evaluation is for and what question it answers")
    p.add_argument("--dataset", required=True, metavar="PK_OR_UUID")
    p.add_argument("--model", required=True, metavar="PK_OR_UUID")
    p.add_argument("--python-code", required=True)
    # The remedies the 306 and overlap refusals NAME. Both were WorkChain inputs with no
    # command-line form, so the sanctioned path could not apply the fix its own error
    # suggested -- the same defect already fixed once for `full_frames` above.
    p.add_argument("--allow-domain-change", action="store_true",
                   help="score the model on a different domain extent than it trained on "
                        "(the 306 refusal's documented remedy)")
    p.add_argument("--allow-training-data", action="store_true",
                   help="score cases the model trained on; the record is stamped "
                        "held_out=False")
    p.add_argument(
        "--full-frames", action="store_true",
        help="score EVERY stored frame. Without it an automatic eval subsamples to "
             "~25 evenly spread frames per case, which is a health readout, not a "
             "full-split score; the stride is stamped into the metrics either way.",
    )
    _profile_arg(p, "--evaluation-profile")
    _profile_arg(p, "--derivation-profile")

    p = add(smsub, "kfold", _submit_kfold,
            "[DEPRECATED] k-fold train/eval: refuses; the workchain is intact, "
            "see docs/developer_notes.md")
    p.add_argument("--dataset", type=_node_ref, required=True, metavar="PK")
    p.add_argument("--python-code", required=True)
    _profile_arg(p, "--training-profile")
    _profile_arg(p, "--derivation-profile")

    p = add(smsub, "raw-flow3d", _submit_raw_flow3d, "import raw FLOW-3D NPZ runs as simulations")
    p.add_argument("--source-path", required=True)
    p.add_argument("--python-code", required=True)
    _profile_arg(p, "--dataset-profile")
    p.add_argument("--materials-root", default=None)

    p = add(smsub, "flow3d", _submit_flow3d, "run one FLOW-3D case through flow3D_scripts, ingest its NPZ and file it")
    p.add_argument("--python-code", required=True)
    _profile_arg(p, "--run-config")
    _profile_arg(p, "--execution-profile")
    _profile_arg(p, "--dataset-profile")
    p.add_argument("--materials-root", default=None)
    p.add_argument("--target-dataset", type=_node_ref, default=None, metavar="PK",
                   help="file the finished case(s) into this existing base dataset "
                        "instead of one named after dataset_profile.batch_keyword")

    p = add(smsub, "flow3d-batch", lambda s, a: _submit_flow3d(s, a, batch=True), "run a FLOW-3D config sweep (one solve at a time)")
    p.add_argument("--python-code", required=True)
    p.add_argument("--config", action="append", default=[], metavar="@FILE", help="one run config per flag (repeatable)")
    _profile_arg(p, "--execution-profile")
    _profile_arg(p, "--dataset-profile")
    p.add_argument("--materials-root", default=None)
    p.add_argument("--max-concurrent", type=int, default=1)
    p.add_argument("--target-dataset", type=_node_ref, default=None, metavar="PK",
                   help="file the finished case(s) into this existing base dataset "
                        "instead of one named after dataset_profile.batch_keyword")

    p = add(smsub, "import-model", _submit_import_model, "register externally trained artifacts as a model")
    p.add_argument("--python-code", required=True)
    p.add_argument("--training-source", required=True, metavar="PK_OR_UUID",
                   help="pk or uuid of the DATASET the model was trained on (a "
                        "PinoDatasetCollectionData), not the training job -- passing a "
                        "CalcJob or WorkChain pk is refused")
    p.add_argument("--artifacts-pk", type=_node_ref, default=None)
    p.add_argument("--artifacts-path", default=None,
                   help="the run directory holding u_network*.pth; the doubly-nested "
                        "inner directory is found automatically, and a path with no "
                        "checkpoint below it is refused")
    p.add_argument("--validation-source", default=None, metavar="PK_OR_UUID",
                   help="pk or uuid of the val-split DATASET this run was validated on (same "
                        "type as --training-source); without it the model cannot say "
                        "what it was validated against")
    _profile_arg(p, "--training-profile")
    _profile_arg(p, "--derivation-profile")
    p.add_argument("--model-name", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve and validate everything, report what WOULD be "
                        "imported (including any existing model from the same "
                        "artifacts), and submit nothing")

    p = add(smsub, "derive", _submit_derive, "derive a dataset from a collection")
    p.add_argument("--dataset", type=_node_ref, required=True, metavar="PK")
    _profile_arg(p, "--derivation-profile")

    return parser


# -------------------------------------------------------------------- runtime
def _service(args):
    from aiida_pino.console.config import Settings
    from aiida_pino.console.service import AiidaConsoleService

    settings = Settings(aiida_profile=args.profile) if args.profile else Settings()
    return AiidaConsoleService.build_default(settings)


def _error_message(exc: Exception) -> str | None:
    """Single-line message for expected errors; None => unexpected, re-raise."""
    try:
        from aiida_pino.console.errors import ConsoleError, DaemonUnavailableError
    except ImportError:  # pragma: no cover
        ConsoleError = DaemonUnavailableError = ()  # type: ignore[assignment]
    if isinstance(exc, DaemonUnavailableError):
        return f"{exc} (start it: docker start aiida-rabbitmq && verdi daemon start)"
    if isinstance(exc, ConsoleError):
        return str(exc)
    try:
        from pydantic import ValidationError
    except ImportError:  # pragma: no cover
        ValidationError = ()  # type: ignore[assignment]
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()) or str(exc)
    if isinstance(exc, (ValueError, OSError)):
        return str(exc)
    return None


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list) and len(value) > 12:
        # 300 member UUIDs inline bury the useful fields; the full dump stays
        # available behind --json.
        return f"[{len(value)} items — use --json for the full list]"
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


# Fields that are true, useful, and ruinous in a fixed-width table. `evals`
# carries each model's full evaluation JSON, and every row is padded to the
# widest one: `aiida-pino models` printed 26,168-character lines, 2.6 MB for 100
# rows, even where the cell was `[]`. All of them stay in `--json`.
_TABLE_HIDDEN = frozenset({
    "uuid", "evals", "evaluations", "training_source_canonical",
    "process_type", "member_simulation_uuids", "component_pks", "describe",
})


def _render_compare(report: dict) -> str:
    """The referee's report as text a person can act on.

    The generic table renderer showed a confounded pair as "[29 items -- use
    --json]" -- hiding the differences exactly when there are many, which is
    exactly when the report matters (adversarial pass).
    """
    lines = [
        f"A: #{report['a']['pk']} {report['a']['name']}   "
        f"B: #{report['b']['pk']} {report['b']['name']}",
        f"VERDICT: {report['verdict']}",
        f"training data: {report['training_data']}",
    ]
    if report.get("backfilled_facts"):
        lines.append("note: reconstructed (backfilled) facts in play: "
                     + ", ".join(report["backfilled_facts"]))
    substantive = [d for d in report.get("differences", []) if d["severity"] != "benign"]
    if substantive:
        lines.append("")
        for d in substantive:
            factor = d.get("factor", d["field"])
            # the factor is only worth a column when it differs from the field;
            # printing both made "override.custom.FNO_input_keys" appear twice
            label = f"{factor:<18} {d['field']}" if factor != d["field"] else d["field"]
            lines.append(f"  [{d['severity']:>8}] {label}")
            delta = d.get("delta")
            if delta:
                if delta.get("removed"):
                    lines.append(f"{'':>13}- {', '.join(delta['removed'])}")
                if delta.get("added"):
                    lines.append(f"{'':>13}+ {', '.join(delta['added'])}")
            else:
                lines.append(f"{'':>13}A: {str(d['a'])[:90]}")
                lines.append(f"{'':>13}B: {str(d['b'])[:90]}")
    benign = [d for d in report.get("differences", []) if d["severity"] == "benign"]
    if benign:
        lines.append(f"  (+{len(benign)} bookkeeping difference(s): "
                     + ", ".join(sorted(d["field"] for d in benign)[:6]) + ")")
    lines.append("")
    evals = report.get("shared_evaluations", [])
    lines.append(f"shared evaluations: {len(evals)}"
                 + ("" if evals else "  (nothing legitimate to compare yet)"))
    for ev in evals:
        mark = "ok " if ev["comparable"] else "!! "
        lines.append(f"  {mark}{str(ev.get('dataset_label'))[:56]}: "
                     f"T abs err {ev['a']['t_abs_err_mean']} vs {ev['b']['t_abs_err_mean']}")
        for problem in ev.get("problems", []):
            lines.append(f"      problem: {problem}")
    return "\n".join(lines)


def _render(result: Any) -> str:
    if isinstance(result, dict) and "verdict" in result and "differences" in result:
        return _render_compare(result)
    if isinstance(result, list):
        if not result:
            return "(none)"
        if isinstance(result[0], dict):
            # The web layer pops these sentinels; the CLI printed them as two
            # columns, so "500 of 810 shown" arrived as a column of `True`.
            total = result[0].get("_total")
            truncated = any(row.get("_truncated") for row in result)
            cols = [c for c in result[0] if c not in _TABLE_HIDDEN and not c.startswith("_")]
            # 60 chars is enough to recognise a name; the rest of a description
            # or a metrics blob just widens every other row. `--json` is exact.
            def _clip(text: str) -> str:
                return text if len(text) <= 60 else text[:57] + "..."

            rows = [[_clip(_cell(row.get(col))) for col in cols] for row in result]
            # A column that is empty on every row is noise in every row.
            keep = [i for i in range(len(cols)) if any(row[i] not in ("", "-") for row in rows)]
            cols = [cols[i] for i in keep]
            rows = [[row[i] for i in keep] for row in rows]
            if not cols:
                return f"({len(result)} rows, no populated columns)"
            widths = [max(len(col), *(len(row[i]) for row in rows)) for i, col in enumerate(cols)]
            lines = ["  ".join(col.ljust(widths[i]) for i, col in enumerate(cols))]
            lines += ["  ".join(row[i].ljust(widths[i]) for i in range(len(cols))) for row in rows]
            if truncated and total:
                lines.append(f"({len(result)} of {total} shown — use --limit 0 for all)")
            return "\n".join(lines)
        return "\n".join(str(item) for item in result)
    if isinstance(result, dict):
        lines = []
        for key, value in result.items():
            # Notes are sentences for a person; one per line beats a JSON list on one line.
            if key == "notes" and isinstance(value, list) and value:
                lines.append("notes:")
                lines.extend(f"  - {note}" for note in value)
            else:
                lines.append(f"{key}: {_cell(value)}")
        return "\n".join(lines)
    return str(result)


def main(argv: list[str] | None = None) -> int:
    try:
        from aiida_pino.console.service import silence_rabbitmq_version_noise

        silence_rabbitmq_version_noise()
    except ImportError:  # pragma: no cover
        pass
    args = build_parser().parse_args(argv)
    if getattr(args, "foreground", False):
        import os

        os.environ["AIIDA_PINO_FOREGROUND"] = "1"
    try:
        service = _service(args)
        _resolve_node_refs(args)
        result = args.run(service, args)
    except Exception as exc:  # noqa: BLE001 - expected errors become exit 1, rest re-raise
        message = _error_message(exc)
        if message is None:
            raise
        print(f"error: {' '.join(message.split())}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str) if args.json else _render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
