"""Dataset-directory validation, shared by every adapter that runs on a box.

This existed in THREE verbatim-ish copies -- `train_adapter`, `eval_adapter`,
`kfold_adapter` -- and they had already drifted before anyone noticed: train
grew a consolidated-per-case-NPZ branch the other two never got. When the
manifest member cross-check was added it went into train alone, so eval and
kfold went on accepting a directory holding nothing but `manifest.json`.

The consequences differ, and eval's is the worse one. A training on a partial
dataset produces a model with a quiet quality deficit, which someone can retrain
once suspicious. An **eval** on a partial dataset produces a NUMBER -- one that
goes into a comparison, a table, possibly a manuscript -- computed over fewer
cases than it claims. Nobody revisits a number that looks fine.

Standard library only, and staged beside every adapter by `stage_script`: a
compute box has no `aiida_pino` package to import this from.
"""

from __future__ import annotations

import json
from pathlib import Path


def missing_manifest_members(manifest_path: Path) -> list[str]:
    """Member directories a collection manifest names but that are absent.

    Empty for a manifest that lists no members (a per-simulation manifest), so
    this only tightens the collection case. Unreadable manifests return empty
    too: the caller's other checks already own a corrupt manifest, and this must
    not turn a readable problem into an unreadable one.
    """
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - other checks own a corrupt manifest
        return []
    members = payload.get("members") if isinstance(payload, dict) else None
    if not isinstance(members, list) or not members:
        return []
    root = manifest_path.parent
    missing = []
    for member in members:
        relative = (member or {}).get("relative_path") if isinstance(member, dict) else None
        if relative and not (root / str(relative)).exists():
            missing.append(str(relative))
    return missing


def validate_dataset_input_dir(
    path: Path,
    *,
    label: str,
    accept_consolidated_npz: bool = False,
    consolidated_npz_check=None,
) -> None:
    """Fail early, and clearly, when staged dataset input is not usable.

    `accept_consolidated_npz` is a parameter rather than always-on because only
    the training adapter accepted that layout before this consolidation.
    Switching it on for eval and k-fold would LOOSEN two validators as a side
    effect of removing duplication, which is not a change to make silently: a
    silent loosening is unbounded and has no failure signal, while staying
    strict fails loudly and is a two-line fix if it is ever wrong.

    Measured 2026-08-19, so this is not merely cautious: all 385 entries in the
    live runtime cache carry a manifest, the manifest branch is checked first
    and returns, and eval's dataset directory is always the staged
    `dataset_input` (`calculations/eval.py`) rather than a hand-pointed path.
    The consolidated branch is therefore unreachable for eval in practice, and
    default-off costs nothing.
    """
    try:
        from aiida_pino.utils.runtime_repair import repair_dataset_dir

        repair_dataset_dir(path)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - repair is best effort
        print(f"[runtime-repair] skipped ({type(exc).__name__}: {exc})", flush=True)

    if not path.exists():
        raise SystemExit(f"{label} path does not exist: {path}")
    if not path.is_dir():
        raise SystemExit(f"{label} path is not a directory: {path}")

    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        # A manifest is not a dataset. Returning on its mere presence let a
        # directory holding ONLY manifest.json through, and the job then ran on
        # a collection with no members -- silently. That state is reachable: a
        # staging failure clears the destination before copying and can leave
        # exactly this behind (transfer order is os.listdir order), and a remote
        # cache has no repair path, because there is no object store there.
        missing = missing_manifest_members(manifest_path)
        if missing:
            shown = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise SystemExit(
                f"{label} manifest names members that are not present: {shown}{more}. "
                f"{len(missing)} of the listed members are missing, so this dataset is "
                "incomplete and running on it would silently use less data than it claims. "
                "On a REMOTE computer re-run `aiida-pino stage-remote <collection-pk> "
                "--computer <label>` on the primary; there is no repair path on the box."
            )
        return

    if accept_consolidated_npz and consolidated_npz_check is not None and consolidated_npz_check(path):
        return

    has_vtk_payload = any(
        candidate.is_file() and candidate.suffix.lower() in {".vtk", ".vtu"} for candidate in path.rglob("*")
    )
    if has_vtk_payload:
        return

    npz_note = "contains no consolidated per-case NPZ files, " if accept_consolidated_npz else ""
    raise SystemExit(
        f"{label} is missing native runtime manifest, {npz_note}"
        f"and contains no VTK/VTU files: {path}. "
        "This usually means dataset staging failed in aiida-pino. On a REMOTE computer it "
        "most often means the runtime cache was never copied to this machine, or was copied "
        "and then removed: re-run `aiida-pino stage-remote <collection-pk> --computer <label>` "
        "on the primary and resubmit."
    )


def runtime_repair_command(python_executable: str) -> str:
    """Best-effort cache repair, quiet when it cannot run.

    `aiida_pino` is not installed on a compute box, so `-m
    aiida_pino.utils.runtime_repair` there prints a raw
    "Error while finding module specification" before `|| true` swallows the
    exit code. Harmless, but it is the FIRST error text in the log of an
    otherwise healthy run, and it sends a reader looking in the wrong place --
    it cost one reviewer a minute on a run whose real failure was elsewhere.

    stderr is dropped rather than the failure hidden: the in-process path
    already reports a genuine repair problem as `[runtime-repair] skipped
    (...)`, so nothing diagnostic is lost by silencing a subprocess that, off
    this machine, can only ever fail for one uninteresting reason.

    2026-08-25: silencing it also hid that the repair is PERMANENTLY inert on every
    remote computer, not merely noisy there. The path exists because five workchains died
    on `BadZipFile: Bad CRC-32` reading a cache nothing rebuilt -- and it has never once
    run on the boxes where staged-cache corruption actually occurs. Shipping it as an
    uploaded adapter would not help: repair rebuilds from the AiiDA object store, which
    lives on the SUBMITTING machine, so a compute box genuinely cannot do it. What it can
    do is SAY so once, instead of failing silently and reading as "repair ran, found
    nothing".
    """
    import shlex

    quoted = shlex.quote(python_executable)
    unavailable = (
        "[runtime-repair] unavailable on this computer: aiida_pino is not installed here, "
        "so a corrupt or missing staged cache cannot be rebuilt (the object store lives on "
        "the submitting machine). Re-stage from there if the data looks wrong."
    )
    return (
        f"if {quoted} -c 'import aiida_pino' 2>/dev/null; then "
        f"{quoted} -m aiida_pino.utils.runtime_repair 2>/dev/null || true; "
        f"else echo {shlex.quote(unavailable)}; fi"
    )
