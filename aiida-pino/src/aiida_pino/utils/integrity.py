"""What the database cannot tell you about itself unless you ask.

Four checks, all read-only, all reproducing a measurement from the 2026-07-30
adversarial review (`database_review.md`):

- **dangling members** — collections whose `member_simulation_uuids` name a node
  that no longer exists. `iter_member_simulations()` raises `NotExistent` on
  these; 61 collections flagged at review time, one of them live (criticism A).
- **byte-identical simulations** — distinct nodes holding the same payload under
  different fingerprints, because the fingerprint hashes the source *path*. 12
  pairs at review time, one pair inside a live training collection, which
  therefore double-weights a case (criticism J).
- **unlinked simulations** — nodes with no creating process. Everything imported
  before 2026-07-30 is in this set and cannot be fixed: provenance is immutable
  (criticism A). The number is here so it can be quoted rather than guessed.
- **missing sources** — `source_uri` files that no longer exist on disk. 269 at
  review time.

`--stamp` writes the free content hash (the object store is sha256-addressed) to
each simulation's `content_hash` extra, which is what makes the duplicate check
a query instead of a scan, and what the importer dedupes against from now on.

Nothing here deletes or edits a dataset. A duplicate inside a live collection is
reported, never silently removed: dropping a member changes what a model was
trained on, and that is a curation decision.

    python -m aiida_pino.utils.integrity                # report
    python -m aiida_pino.utils.integrity --stamp        # + write content_hash
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from aiida import orm
from aiida.orm import QueryBuilder

from aiida_pino.data.pino_dataset_collection import PinoDatasetCollectionData
from aiida_pino.data.pino_simulation import PinoSimulationData, payload_content_hash


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _source_path(metadata: dict[str, Any]) -> Path | None:
    uri = str(metadata.get("source_uri") or "")
    if not uri.startswith("file://"):
        return None
    return Path(unquote(urlparse(uri).path))


def mark_lost_members(apply: bool = False) -> dict[str, Any]:
    """Record what a collection lost, instead of quietly emptying its pk list.

    `member_pk_repair` reached "1064/1064 consistent" partly by writing an empty
    pk list for 61 collections whose members no longer exist. That is
    consistency bought by deletion: the collection still advertises member
    uuids it cannot produce, and `iter_member_simulations()` raises.

    Deleting them is *not* the answer, and the measurement says why: a dry-run
    delete of the one live shell takes **11 nodes** with it — the
    `Flow3DRunCalculation`, its `Flow3DGenerateWorkChain`, the ingest job, both
    summary Dicts and three retrieved artifact folders. The collection is empty
    of data and is the last handle on a complete solve record. So it is marked,
    not removed.
    """
    stats = {"collections": 0, "shells": 0, "partial": 0, "marked": 0}
    known: set[str] = set()
    for (uuid,) in QueryBuilder().append(PinoSimulationData, project=["uuid"]).iterall():
        known.add(str(uuid))
    for node in QueryBuilder().append(PinoDatasetCollectionData, project=["*"]).all(flat=True):
        stats["collections"] += 1
        uuids = [str(u) for u in (node.base.attributes.all.get("member_simulation_uuids") or [])]
        missing = [u for u in uuids if u not in known]
        if not missing:
            continue
        stats["shells" if len(missing) == len(uuids) else "partial"] += 1
        if node.base.extras.get("members_lost", None) is not None:
            continue
        if apply:
            node.base.extras.set("members_lost", {
                "uuids": missing,
                "of": len(uuids),
                "note": (
                    "These member simulations no longer exist. The collection is kept because "
                    "its provenance subtree (solve, ingest, summaries, artifacts) is intact; "
                    "deleting it would take that with it. iter_member_simulations() raises."
                ),
                "recorded_utc": _utc_now(),
            })
        stats["marked"] += 1
    return stats


def rehash_stale(apply: bool = False) -> dict[str, Any]:
    """Recompute `_aiida_hash` where a backend attribute write invalidated it.

    Several repairs wrote attributes through the backend to get past AiiDA's
    immutability seal. The data is correct; the stored hash still describes the
    old content, which leaves a reader unable to use `_aiida_hash` as an
    integrity fingerprint.

    Rehashing is safe and is *more* correct than leaving it: it recomputes from
    current content and writes an extra (extras are mutable, so no bypass is
    needed). It does not touch data. What it costs is the ability to detect
    "this node changed after creation" by hash mismatch — which is why
    `members_lost` and the repair notes exist to say so explicitly instead.
    """
    stats = {"checked": 0, "stale": 0, "rehashed": 0}
    for node in QueryBuilder().append(PinoDatasetCollectionData, project=["*"]).all(flat=True):
        stored = node.base.extras.get("_aiida_hash", None)
        if not stored:
            continue
        stats["checked"] += 1
        try:
            if node.base.caching.compute_hash() == stored:
                continue
        except Exception:  # noqa: BLE001
            continue
        stats["stale"] += 1
        if apply:
            node.base.caching.rehash()
            stats["rehashed"] += 1
    return stats


def align_time_axis_mode(apply: bool = False) -> dict[str, Any]:
    """Make each collection's legacy `time_axis_mode` agree with its own profile.

    The canonical layout became `time_as_batch` on 2026-07-22, but the legacy
    `dataset_metadata.time_axis_mode` stamp was never migrated: 1,103 of 1,430
    live collections still say `time_as_dim`, while their OWN
    `derivation_profile` -- the field training and reuse actually read -- says
    `time_as_batch`. That is not cosmetic. `is_canonical_metadata` reads the
    legacy stamp, so every one of those nodes reports as non-canonical against
    its own profile, and `datasource_cache_basename_from_metadata` takes the
    stamp in PREFERENCE to the profile.

    Each node is aligned to ITS OWN derivation_profile, never to a global
    constant: 22 of 600 sampled are genuinely `time_as_dim` on both, and
    rewriting those would replace a stale record with a false one.

    Refuses to touch any node whose cache basename would change -- the basename
    names a file on disk, and a migration that renames data is not a metadata
    migration.
    """
    from aiida_pino.data import PinoDatasetCollectionData
    from aiida_pino.data.pino_dataset import datasource_cache_basename_from_metadata
    from aiida_pino.data.pino_dataset import safe_attribute_get

    stats = {"checked": 0, "aligned": 0, "already": 0, "genuinely_dim": 0,
             "refused_basename_change": 0, "no_profile": 0}
    examples: list[dict[str, Any]] = []

    for node in QueryBuilder().append(PinoDatasetCollectionData, project=["*"]).all(flat=True):
        stats["checked"] += 1
        metadata = dict(safe_attribute_get(node, "dataset_metadata", {}) or {})
        profile = metadata.get("derivation_profile") or safe_attribute_get(
            node, "derivation_profile", {}) or {}
        mode = dict(metadata.get("time_axis_mode") or {})
        if not profile:
            stats["no_profile"] += 1
            continue

        wanted = {
            "include_time": bool(profile.get("include_time", mode.get("include_time", True))),
            "time_as_batch": bool(profile.get("time_as_batch", False)),
            "time_as_dim": bool(profile.get("time_as_dim", False)),
        }
        if mode == wanted:
            stats["already"] += 1
            continue
        if bool(mode.get("time_as_dim")) and bool(profile.get("time_as_dim")):
            stats["genuinely_dim"] += 1
            continue

        before = datasource_cache_basename_from_metadata(metadata)
        after = datasource_cache_basename_from_metadata({**metadata, "time_axis_mode": wanted})
        if before != after:
            stats["refused_basename_change"] += 1
            continue

        stats["aligned"] += 1
        if len(examples) < 5:
            examples.append({"pk": node.pk, "was": mode, "now": wanted})
        if apply:
            metadata["time_axis_mode"] = wanted
            node.backend_entity.set_attribute("dataset_metadata", metadata)
            from aiida_pino.utils.provenance_mode import mark_backfilled

            mark_backfilled(node, ["dataset_metadata.time_axis_mode"])

    stats["applied"] = bool(apply)
    stats["examples"] = examples
    return stats


def seal_orphaned_processes(apply: bool = False) -> dict[str, Any]:
    """Seal terminated processes the daemon never got to seal.

    A daemon killed between a process finishing and its seal being written
    leaves the node unsealed, and ONE such node aborts every full-profile
    export: "All ProcessNodes must be sealed before they can be exported"
    (hit live 2026-07-21 on a finished process, exit 303). Daemon wedges have
    happened more than once, so this is a recurring condition, not
    a one-off -- and it is discovered at export time, which is the worst moment.

    Only terminated nodes are sealed. Sealing a live one would freeze a process
    still writing to itself, which is a far worse failure than a blocked export.
    """
    from aiida.orm import ProcessNode, QueryBuilder

    rows = QueryBuilder().append(
        ProcessNode, filters={"attributes.sealed": False}, project=["*"],
    ).all(flat=True)
    terminated = [node for node in rows if node.is_terminated]
    live = len(rows) - len(terminated)
    sealed = []
    for node in terminated:
        if apply:
            node.seal()
        sealed.append(node.pk)
    return {
        "unsealed": len(rows),
        "terminated": len(terminated),
        "still_running": live,
        "sealed": sealed if apply else [],
        "would_seal": [] if apply else sealed,
        "applied": bool(apply),
    }


def check_database(stamp: bool = False, check_sources: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {}

    known: dict[str, PinoSimulationData] = {}
    by_content: dict[str, list[int]] = defaultdict(list)
    unlinked = 0
    missing_sources = 0
    stamped = 0
    total = 0
    for sim in QueryBuilder().append(PinoSimulationData, project=["*"]).all(flat=True):
        total += 1
        known[str(sim.uuid)] = sim
        if sim.creator is None:
            unlinked += 1
        content = sim.base.extras.get("content_hash", None) or payload_content_hash(sim)
        if content:
            if stamp and sim.base.extras.get("content_hash", None) is None:
                sim.base.extras.set("content_hash", content)
                stamped += 1
            if not sim.base.extras.get("superseded", False):
                by_content[content].append(int(sim.pk))
        if check_sources:
            path = _source_path(sim.get_simulation_metadata() or {})
            if path is not None and not path.exists():
                missing_sources += 1

    duplicates = {c: pks for c, pks in by_content.items() if len(pks) > 1}
    # Coverage is a *snapshot*, never a property: a backfill fills what exists
    # when it runs, and nothing stamps solver_version at ingest, so every
    # simulation created afterwards is uncovered again. Reporting the gap here
    # is what makes "2,339 of 3,093" a number anyone can re-derive instead of a
    # claim in a lab notebook that quietly went stale.
    report["coverage"] = {
        "simulations": total,
        "solver_version": sum(
            1 for s in known.values() if s.base.extras.get("solver_version", None)
        ),
        "content_hash": sum(1 for s in known.values() if s.base.extras.get("content_hash", None)),
    }
    report["simulations"] = {
        "total": total,
        "without_creating_process": unlinked,
        "missing_source_file": missing_sources if check_sources else None,
        "content_hash_stamped": stamped,
        "byte_identical_groups": len(duplicates),
        "byte_identical_pks": sorted(sorted(pks) for pks in duplicates.values())[:20],
    }

    dangling: list[dict[str, Any]] = []
    doubled: list[dict[str, Any]] = []
    dup_pk_of_content = {pk: content for content, pks in duplicates.items() for pk in pks}
    for node in QueryBuilder().append(PinoDatasetCollectionData, project=["*"]).all(flat=True):
        uuids = [str(u) for u in (node.base.attributes.all.get("member_simulation_uuids") or [])]
        missing = [u for u in uuids if u not in known]
        live = not bool(node.base.extras.get("superseded", False))
        if missing:
            dangling.append({
                "pk": node.pk, "label": str(node.label)[:50],
                "missing": len(missing), "of": len(uuids), "live": live,
            })
        # A collection holding two members with the same content counts one case
        # twice -- in a training set that is a silently double-weighted sample.
        seen: dict[str, int] = {}
        repeats = 0
        for uuid in uuids:
            sim = known.get(uuid)
            if sim is None:
                continue
            content = dup_pk_of_content.get(int(sim.pk))
            if content is None:
                continue
            if content in seen:
                repeats += 1
            seen[content] = int(sim.pk)
        if repeats:
            doubled.append({
                "pk": node.pk, "label": str(node.label)[:50],
                "duplicate_members": repeats, "members": len(uuids), "live": live,
            })
    from aiida_pino.data.pino_model_evaluation import PinoModelEvaluationData

    with_units = 0
    eval_total = 0
    for ev in QueryBuilder().append(PinoModelEvaluationData, project=["*"]).all(flat=True):
        eval_total += 1
        attrs = ev.get_evaluation_metadata() or {}
        if isinstance(attrs.get("metric_units"), dict) or isinstance(
            ev.base.extras.get("metric_units", None), dict
        ):
            with_units += 1
    report["evaluations"] = {"total": eval_total, "with_units": with_units}

    # Checks the earlier version did not make, and said nothing about not
    # making. An integrity report that is silent on a class of defect reads as
    # "clean" for it.
    evicted = sum(1 for s in known.values() if s.base.extras.get("evicted", False))
    chain_breaks: list[dict[str, Any]] = []
    stale_hashes = 0
    hashed = 0
    by_uuid = {}
    for node in QueryBuilder().append(PinoDatasetCollectionData, project=["*"]).all(flat=True):
        by_uuid[str(node.uuid)] = node
    for node in by_uuid.values():
        stored = node.base.extras.get("_aiida_hash", None)
        if stored:
            hashed += 1
            try:
                if node.base.caching.compute_hash() != stored:
                    stale_hashes += 1
            except Exception:  # noqa: BLE001
                pass
        if not node.base.extras.get("superseded", False):
            continue
        target = node.base.extras.get("replaced_by_uuid", None)
        if not target:
            chain_breaks.append({"pk": node.pk, "why": "superseded with no replaced_by_uuid"})
        elif str(target) not in by_uuid:
            chain_breaks.append({"pk": node.pk, "why": f"replaced_by_uuid {str(target)[:8]} does not resolve"})
    report["also_checked"] = {
        "evicted_simulations": evicted,
        "collections_with_stored_hash": hashed,
        "stale_hashes": stale_hashes,
        "supersession_chain_breaks": len(chain_breaks),
        "chain_break_examples": chain_breaks[:5],
        # Byte-identical detection can only see what it has a hash for, and an
        # evicted node has no payload to hash. Saying so keeps the count from
        # being read as exhaustive.
        "duplicate_check_is_a_lower_bound_because": (
            f"{sum(1 for s in known.values() if not s.base.extras.get('content_hash'))} simulations "
            "have no content_hash (evicted or never stamped) and cannot participate"
        ),
    }
    report["collections"] = {
        "dangling_members": dangling,
        "dangling_live": sum(1 for d in dangling if d["live"]),
        "holding_duplicate_content": doubled,
    }

    # Authinfo is per-user and does not live in the node tables, so a
    # node-level migration (SQLite -> PostgreSQL, 2026-07-21) leaves every
    # computer unconfigured while the codes are intact — and the first symptom
    # is `verdi code list` printing "No codes found", the one thing that is
    # fine. Checking it here turns a 20-minute hunt into one report line.
    from aiida.orm import Computer

    report["computers"] = [
        {
            "label": c.label,
            "configured": bool(_computer_configured(c)),
            # The configure command differs per transport, and printing the
            # local one for an SSH computer sends the reader to a command that
            # cannot work.
            "transport_type": getattr(c, "transport_type", "core.local"),
        }
        for c in Computer.collection.all()
    ]
    return report


def unconfigured_computer_hint(computer_row: dict) -> str:
    """What to run for a computer with no authinfo, in ITS transport.

    The command differs per transport, and printing the local one for an SSH
    computer sends the reader to something that cannot work — on a box they
    probably cannot reach yet, which is the moment they least need a wrong
    instruction.
    """
    transport = computer_row.get("transport_type") or "core.local"
    return (
        f"computer `{computer_row['label']}` is NOT configured for this user — authinfo is "
        f"per-user and does not migrate with nodes, and the first symptom is a "
        f"misleading `verdi code list` printing nothing. Fix:\n"
        f"  verdi computer configure {transport} {computer_row['label']} -n --safe-interval 0"
    )


def _computer_configured(computer) -> bool:
    try:
        return computer.is_configured
    except Exception:  # noqa: BLE001 - unreadable authinfo counts as unconfigured
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stamp", action="store_true", help="write the content_hash extra")
    parser.add_argument("--mark-lost", action="store_true",
                        help="record members_lost on collections whose members are gone")
    parser.add_argument("--rehash", action="store_true",
                        help="recompute _aiida_hash where a backend attribute write invalidated it")
    parser.add_argument("--skip-sources", action="store_true", help="do not stat source files")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    from aiida import load_profile

    load_profile(args.profile)
    if args.mark_lost:
        print("lost members:", mark_lost_members(apply=True))
    if args.rehash:
        print("rehash:", rehash_stale(apply=True))
    report = check_database(stamp=args.stamp, check_sources=not args.skip_sources)
    sims = report["simulations"]
    cols = report["collections"]
    print(f"simulations: {sims['total']}")
    print(f"  no creating process : {sims['without_creating_process']} (pre-2026-07-30 imports; not fixable)")
    print(f"  source file missing : {sims['missing_source_file']}")
    print(f"  byte-identical sets : {sims['byte_identical_groups']} {sims['byte_identical_pks'][:5]}")
    if sims["content_hash_stamped"]:
        print(f"  content_hash stamped: {sims['content_hash_stamped']}")
    for comp in report["computers"]:
        if not comp["configured"]:
            print(unconfigured_computer_hint(comp))
    print(f"collections with dangling members: {len(cols['dangling_members'])} "
          f"({cols['dangling_live']} live)")
    for row in cols["dangling_members"][:5]:
        print(f"  pk {row['pk']:>6} {row['label']:<50} {row['missing']}/{row['of']} missing"
              f"{' [LIVE]' if row['live'] else ''}")
    cov = report["coverage"]
    evals = report.get("evaluations") or {}
    print(
        "\ncoverage — re-measure, do not quote. `content_hash` keeps up on its own "
        "(the import parser stamps it) and so does `solver_version` on FLOW-3D\n"
        "calcjobs (their parser stamps it), but a *simulation* is created by a "
        "different job that never sees runhyd.log, so its coverage drifts with\nevery import:"
    )
    for label, have, fixer in (
        ("solver_version", cov["solver_version"], None),
        ("content_hash", cov["content_hash"], "python -m aiida_pino.utils.integrity --stamp"),
    ):
        missing = cov["simulations"] - have
        flag = "" if not missing else f"   <- {missing} missing" + (f": {fixer}" if fixer else "")
        print(f"  {label:16s} {have}/{cov['simulations']}{flag}")
    if evals:
        missing = evals["total"] - evals["with_units"]
        fixer = f"   <- {missing} without units" if missing else ""
        print(f"  {'metric_units':16s} {evals['with_units']}/{evals['total']} evaluations{fixer}")
    also = report["also_checked"]
    print(f"\nalso checked: evicted sims {also['evicted_simulations']} | "
          f"stale _aiida_hash {also['stale_hashes']}/{also['collections_with_stored_hash']} | "
          f"supersession chain breaks {also['supersession_chain_breaks']}")
    print(f"  the duplicate count is a LOWER BOUND: {also['duplicate_check_is_a_lower_bound_because']}")
    for row in also["chain_break_examples"]:
        print(f"  chain break: pk {row['pk']} — {row['why']}")
    print(f"\ncollections holding duplicate content: {len(cols['holding_duplicate_content'])}")
    for row in cols["holding_duplicate_content"][:5]:
        print(f"  pk {row['pk']:>6} {row['label']:<50} {row['duplicate_members']} repeat(s) "
              f"of {row['members']} members{' [LIVE]' if row['live'] else ''}")


if __name__ == "__main__":
    main()
