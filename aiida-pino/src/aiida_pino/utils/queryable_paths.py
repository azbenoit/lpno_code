"""What can actually be queried on this node type? Ask the data, not the code.

Every "X isn't stored" report so far has been wrong in the same way: X *was*
stored, nested inside a metadata blob, and nobody could find it. Measured
2026-08-06 on the authors' database, the FLOW-3D solve wall clock was reported missing while

    attributes.simulation_metadata.flow3d_case_manifest.run.elapsed_seconds

returns **1547** values through a plain `QueryBuilder` — it always did. The cell
size, the mesh shape, the cycle count and the case name were the same story.
Nothing was unreachable. The paths were undiscoverable.

The repo's answer until now was a module per field: eight `*_backfill.py` files,
each lifting one more thing into an extra. That does not scale and it does not
help anyone asking a question the day before someone writes the ninth. This
walks the JSON that is actually stored and reports every path, with how many
nodes carry it and the exact query line to copy.

    python -m aiida_pino.utils.queryable_paths simulation
    python -m aiida_pino.utils.queryable_paths simulation --grep wall
    python -m aiida_pino.utils.queryable_paths model --min-coverage 50

Lifting a path into an extra is still worth doing for the handful you query
constantly — a short name, and one JSONB step instead of five. It is no longer
the only way to reach anything.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from aiida import orm

#: What a caller calls these, as opposed to the class name.
NODE_CLASSES = {
    "simulation": "PinoSimulationData",
    "model": "PinoModelData",
    "dataset": "PinoDatasetCollectionData",
    "evaluation": "PinoModelEvaluationData",
    "union": "PinoDatasetUnionData",
}

#: Deeper than this is almost always payload, not metadata worth querying.
MAX_DEPTH = 6

#: A list this long is data, not a field: report its shape, never walk into it.
LONG_LIST = 8


def _resolve(kind: str):
    from aiida_pino import data as data_module

    name = NODE_CLASSES.get(kind)
    if name is None:
        raise ValueError(f"unknown node kind {kind!r}; choose from {', '.join(sorted(NODE_CLASSES))}")
    try:
        return getattr(data_module, name)
    except AttributeError:
        from importlib import import_module

        # Not every class is re-exported from the package (`PinoModelEvaluationData`
        # lives in its own module), so fall back to the module that defines it.
        module = import_module(f"aiida_pino.data.pino_{_snake(name)}")
        return getattr(module, name)


def _snake(class_name: str) -> str:
    stem = class_name.removeprefix("Pino").removesuffix("Data")
    out = []
    for index, char in enumerate(stem):
        if char.isupper() and index:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _describe(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        inner = _describe(value[0]) if value else "?"
        return f"list[{inner}] x{len(value)}"
    if value is None:
        return "null"
    return type(value).__name__


def _sample(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _walk(node: Any, prefix: str, seen: dict, depth: int = 0) -> None:
    if depth > MAX_DEPTH or not isinstance(node, dict):
        return
    for key, value in node.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            # Record the container too: knowing it exists is how you find what
            # is under it when nothing below is populated.
            entry = seen.setdefault(path, {"count": 0, "types": set(), "example": None})
            entry["count"] += 1
            entry["types"].add("object")
            _walk(value, path, seen, depth + 1)
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict) and len(value) <= LONG_LIST:
            for index, item in enumerate(value):
                _walk(item, f"{path}[{index}]", seen, depth + 1)
        entry = seen.setdefault(path, {"count": 0, "types": set(), "example": None})
        entry["count"] += 1
        entry["types"].add(_describe(value))
        if entry["example"] is None and value not in (None, "", [], {}):
            entry["example"] = _sample(value)


def discover(kind: str, *, limit: int | None = None) -> dict[str, Any]:
    """Every path stored under `attributes` and `extras`, with coverage."""
    node_class = _resolve(kind)
    query = orm.QueryBuilder().append(node_class, project=["attributes", "extras"])
    total = query.count()
    seen: dict[str, dict] = {}
    scanned = 0
    for attributes, extras in query.iterall():
        if limit is not None and scanned >= limit:
            break
        scanned += 1
        _walk(attributes or {}, "attributes", seen)
        _walk(extras or {}, "extras", seen)

    rows = [
        {
            "path": path,
            "count": entry["count"],
            "coverage": 100.0 * entry["count"] / scanned if scanned else 0.0,
            "type": "/".join(sorted(entry["types"])),
            "example": entry["example"],
        }
        for path, entry in seen.items()
    ]
    rows.sort(key=lambda row: (-row["count"], row["path"]))
    return {"kind": kind, "class": node_class.__name__, "total": total, "scanned": scanned, "paths": rows}


def query_line(kind: str, path: str) -> str:
    """The exact QueryBuilder call that returns this path, ready to paste."""
    return (
        f"orm.QueryBuilder().append({_resolve(kind).__name__}, "
        f"project=['{path}'], filters={{'{path}': {{'!==': None}}}}).all(flat=True)"
    )


def live_log(node_or_pk) -> dict[str, Any]:
    """The live-tail file of a running job, from anything that names the job.

    The only route to a live signal is
    node -> calcjob -> remote workdir -> log file, and every monitor re-derives
    that chain by hand (2026-08-10 friction). Accepts a pk, a calcjob, or any
    workchain above one, walks DOWN to the calcjob when handed a workchain, and
    reports the log path with whether it exists and how large it is.

    `path` is None when the job never reached a remote workdir, which is itself
    the answer to "why is there no output" -- distinct from a workdir that
    exists and holds nothing.
    """
    from pathlib import Path as _Path

    from aiida import orm

    node = orm.load_node(node_or_pk) if isinstance(node_or_pk, int) else node_or_pk
    calcjob = node
    if not isinstance(node, orm.CalcJobNode):
        descendants = [d for d in getattr(node, "called_descendants", []) or []
                       if isinstance(d, orm.CalcJobNode)]
        if not descendants:
            return {"pk": node.pk, "calcjob_pk": None, "path": None,
                    "reason": "no calcjob below this node"}
        calcjob = sorted(descendants, key=lambda d: d.pk)[-1]

    workdir = calcjob.base.attributes.get("remote_workdir", None)
    if not workdir:
        return {"pk": node.pk, "calcjob_pk": calcjob.pk, "path": None,
                "reason": "no remote_workdir: the job never reached the scheduler"}

    root = _Path(str(workdir))
    candidates = sorted(root.glob("*.log")) + sorted(root.glob("*.out"))
    existing = [c for c in candidates if c.is_file()]
    chosen = max(existing, key=lambda c: c.stat().st_mtime) if existing else None
    return {
        "pk": node.pk,
        "calcjob_pk": calcjob.pk,
        "workdir": str(root),
        "workdir_exists": root.is_dir(),
        "path": str(chosen) if chosen else None,
        "bytes": chosen.stat().st_size if chosen else 0,
        "candidates": [str(c) for c in candidates],
        "reason": "" if chosen else (
            "workdir exists but holds no .log/.out yet" if root.is_dir()
            else "workdir is gone (cleaned up with the run)"
        ),
        "tail_command": f"tail -f {chosen}" if chosen else None,
    }


def probe(kind: str, path: str, *, limit: int | None = None) -> dict[str, Any]:
    """Does this path exist, and if not, what did you mean?

    QueryBuilder projects a path that does not exist as `None` rather than
    raising, so a typo and an empty database are indistinguishable. Four
    "X is not stored" reports in one day were that mistake: `collection_name`
    (the name is `label`), `training_source_*` on evaluations, evals leaking
    into `models --search`, and grid shapes (they are `extras.mesh_cells`).
    Each cost a person an hour of believing the data was missing.

    Answers with the count, and on a miss the closest paths that DO exist --
    fuzzy on the whole path and on the leaf name separately, because the usual
    error is the right leaf under the wrong container.
    """
    import difflib

    # Node COLUMNS are not attributes or extras, so the JSON walk never sees
    # them -- and they are exactly where people look for a name. "The name of a
    # collection" is `label`, a real column; probing `attributes.collection_name`
    # without this answers with the nearest JSON path instead of the answer.
    columns = {
        "label": "the node's name -- what listings and --search match on",
        "description": "free-text note on the node",
        "uuid": "stable identity; survives migrations, unlike pk",
        "id": "pk -- NOT stable across migrations",
        "ctime": "created",
        "mtime": "last changed (bumped by any extras write)",
        "node_type": "the class, for `like` filters",
    }
    # Fuzzy matching cannot bridge a synonym: "grid_shape" and "mesh_cells"
    # share no letters worth matching, yet they are the same question, and that
    # miss is what sent a whole-DB sweep down the slow per-node path. These are
    # the concepts actually asked for under a different word.
    synonyms = {
        ("grid", "shape", "mesh", "resolution", "dims", "dimensions", "nx", "cells"):
            ("mesh_cells", "array_shapes", "spatial_shape", "cell_size_m"),
        ("material", "alloy", "metal"): ("material_name", "material_source", "solidus_k"),
        ("step", "steps", "progress", "iteration"):
            ("max_step_reached", "steps_trained", "max_steps"),
        ("seed", "replicate"): ("training_seed", "training_seed_source"),
        ("member", "members", "size", "count", "n"): ("num_members", "member_simulation_uuids"),
        ("heldout", "held_out", "contaminated", "leak"): ("held_out", "train_overlap_n"),
        ("time", "duration", "wall", "runtime", "cost"):
            ("runtime_wall_seconds", "wall_seconds", "elapsed_seconds"),
    }
    report = discover(kind, limit=limit)
    rows = report["paths"]
    by_path = {row["path"]: row for row in rows}
    scanned = report["scanned"]

    hit = by_path.get(path)
    if hit and hit["count"]:
        return {
            "path": path, "exists": True, "count": hit["count"],
            "scanned": scanned, "coverage": hit["coverage"], "type": hit["type"],
            "example": hit["example"], "query": query_line(kind, path),
            "message": f"{path} exists on {hit['count']} of {scanned} {report['class']} nodes",
        }

    leaf = path.rsplit(".", 1)[-1]
    # A column hit answers the question outright, so it goes first.
    column_hits = [
        name for name in columns
        if name in (path, leaf)
        or difflib.get_close_matches(leaf, [name], n=1, cutoff=0.6)
        or (name == "label" and leaf.endswith(("name", "title")))
    ]
    words = set(leaf.lower().replace("-", "_").split("_"))
    wanted_leaves = {
        target
        for triggers, targets in synonyms.items()
        if words & set(triggers)
        for target in targets
    }
    by_synonym = [row["path"] for row in rows
                  if row["path"].rsplit(".", 1)[-1] in wanted_leaves]
    whole = difflib.get_close_matches(path, list(by_path), n=5, cutoff=0.6)
    same_leaf = [row["path"] for row in rows
                 if row["path"].rsplit(".", 1)[-1] == leaf and row["path"] != path]
    fuzzy_leaf = difflib.get_close_matches(
        leaf, [row["path"].rsplit(".", 1)[-1] for row in rows], n=5, cutoff=0.7)
    by_fuzzy_leaf = [row["path"] for row in rows
                     if row["path"].rsplit(".", 1)[-1] in set(fuzzy_leaf)]

    ordered: list[str] = []
    for candidate in [*same_leaf, *by_synonym, *whole, *by_fuzzy_leaf]:
        if candidate not in ordered and candidate != path and "_aiida_hash" not in candidate:
            ordered.append(candidate)
    suggestions = [
        {"path": name, "count": None, "type": "column", "example": None,
         "note": columns[name]}
        for name in column_hits
    ] + [
        {"path": c, "count": by_path[c]["count"], "type": by_path[c]["type"],
         "example": by_path[c]["example"], "note": ""}
        for c in ordered[:6] if c in by_path
    ]
    return {
        "path": path, "exists": False, "count": 0, "scanned": scanned,
        "suggestions": suggestions,
        "message": (
            f"{path} exists on 0 of {scanned} {report['class']} nodes"
            + (f" -- did you mean {suggestions[0]['path']}?" if suggestions else
               " -- and nothing similar exists either")
        ),
    }


def _duplicates(rows: list[dict]) -> dict[str, list[str]]:
    """Leaf names that appear at more than one path — the aliasing traps.

    `stride` under a derivation profile and `stride` under a training profile
    are different numbers, and reading the wrong one is how a half-resolution
    score got compared with a full-resolution one.
    """
    by_leaf: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["type"] == "object":
            continue
        by_leaf[row["path"].rsplit(".", 1)[-1]].append(row["path"])
    return {leaf: paths for leaf, paths in by_leaf.items() if len(paths) > 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=sorted(NODE_CLASSES), help="which node type to describe")
    parser.add_argument("--grep", default=None, help="only paths containing this substring")
    parser.add_argument("--min-coverage", type=float, default=0.0, help="percent of nodes carrying the path")
    parser.add_argument("--limit", type=int, default=None, help="scan at most this many nodes")
    parser.add_argument("--ambiguous", action="store_true", help="only leaf names that exist at several paths")
    parser.add_argument("--live-log", default=None, type=int, metavar="PK",
                        help="the live-tail file of a running job, from a pk at any "
                             "level (workchain or calcjob)")
    parser.add_argument("--probe", default=None, metavar="PATH",
                        help="check ONE path: how many nodes carry it, and if none, "
                             "the closest paths that do")
    parser.add_argument("--profile", default=None, help="AiiDA profile (default: the configured default profile)")
    args = parser.parse_args(argv)

    from aiida import load_profile

    load_profile(args.profile)
    if args.live_log:
        info = live_log(args.live_log)
        for key in ("pk", "calcjob_pk", "workdir", "path", "bytes", "reason", "tail_command"):
            if info.get(key) not in (None, "", 0):
                print(f"{key:>14}: {info[key]}")
        return 0 if info.get("path") else 1

    if args.probe:
        result = probe(args.kind, args.probe, limit=args.limit)
        print(result["message"])
        if result["exists"]:
            print(f"  type {result['type']}, {result['coverage']:.1f}% coverage")
            if result["example"] is not None:
                print(f"  e.g. {result['example']}")
            print(f"\n  {result['query']}")
            return 0
        for suggestion in result["suggestions"]:
            if suggestion["type"] == "column":
                print(f"  {'column':>7}  {suggestion['path']}   {suggestion['note']}")
                continue
            example = "" if suggestion["example"] is None else f"   e.g. {suggestion['example']}"
            print(f"  {suggestion['count']:>7} nodes  {suggestion['path']}{example}")
        paths = [item for item in result["suggestions"] if item["type"] != "column"]
        if paths:
            print(f"\n  {query_line(args.kind, paths[0]['path'])}")
        return 1

    report = discover(args.kind, limit=args.limit)
    rows = report["paths"]

    if args.ambiguous:
        clashes = _duplicates(rows)
        print(f"{len(clashes)} leaf name(s) exist at more than one path on {report['class']}:\n")
        for leaf, paths in sorted(clashes.items()):
            print(f"  {leaf}")
            for path in paths:
                count = next(row["count"] for row in rows if row["path"] == path)
                print(f"      {path}   ({count} nodes)")
        return 0

    if args.grep:
        needle = args.grep.lower()
        rows = [row for row in rows if needle in row["path"].lower()]
    rows = [row for row in rows if row["coverage"] >= args.min_coverage]

    print(f"{report['class']}: {report['scanned']} of {report['total']} nodes scanned, "
          f"{len(rows)} path(s) shown\n")
    print(f"{'nodes':>7}  {'cover':>6}  {'type':<20} path")
    print(f"{'-' * 7}  {'-' * 6}  {'-' * 20} {'-' * 40}")
    for row in rows:
        print(f"{row['count']:>7}  {row['coverage']:>5.1f}%  {row['type'][:20]:<20} {row['path']}")
        if row["example"] is not None and row["type"] != "object":
            print(f"{'':>7}  {'':>6}  {'':<20}   e.g. {row['example']}")
    if rows:
        print(f"\nto query the first one:\n  {query_line(args.kind, rows[0]['path'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
