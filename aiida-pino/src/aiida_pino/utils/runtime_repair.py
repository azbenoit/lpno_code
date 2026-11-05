"""Re-export a dataset runtime cache that vanished between upload and execution.

`stage_dataset_input` symlinks a job's dataset dir to
``~/.cache/aiida-pino/runtime/<node-uuid>`` at UPLOAD time, but a job can then
sit in the gpu_spool seat queue for hours. Anything that clears the cache in
that window (its own docstring advertises ``rm -rf`` as safe, which is true
only for future submits) leaves the symlink dangling, and the trainer dies
with a misleading "Direct VTK directory loading is deprecated" error — five
queued RIGNO ablation arms were lost that way on 2026-07-22/23.

The symlink target's basename IS the node uuid, so a dangling link carries
everything needed to rebuild itself.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
#: `<node-uuid>` or `<node-uuid>-<fieldhash>`. The discriminator arrived with
#: field-trimmed staging (a compact Temperature/fl export must not collide with
#: a full-field one) and silently broke repair: the anchored uuid regex rejected
#: every discriminated directory, so the gpu-seat shim's post-wait rebuild
#: no-opped for exactly the runs that trim fields. That took out the whole
#: varspeed vscan arm on 2026-07-31 -- five workchains dead on
#: `BadZipFile: Bad CRC-32`, reading a cache nothing rebuilt.
_CACHE_DIR_RE = re.compile(
    r"^(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"(?:-(?P<disc>[0-9a-f]+))?"
    # `-s<staging_code_version>`: a cache staged after the staging code changed, built
    # under a sibling name so the stale one is never deleted under a running job
    # (2026-08-23). Unambiguous against `disc` because that is [0-9a-f]+ and this
    # segment starts with a literal `s`. It is NOT part of the identity -- the node and
    # the recipe are the same, only the code that wrote the bytes is newer -- so a repair
    # rebuilds it exactly as it would any other cache. Without this the name failed to
    # parse, `cache_identity` returned None, and repair silently declined to rebuild it.
    r"(?:-s(?P<staging>[0-9a-f]+))?$"
)
#: Sidecar naming. Kept *outside* the cache directory on purpose: the whole
#: point is to survive that directory being evicted.
_SIDECAR_SUFFIX = ".fields.json"


def runtime_cache_root() -> Path:
    """Cache root, honouring the same override the CalcJob uses."""
    env = os.environ.get("AIIDA_PINO_RUNTIME_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "aiida-pino" / "runtime"


def cache_identity(name: str) -> tuple[str, str | None] | None:
    """Split a cache directory name into ``(node_uuid, discriminator)``.

    Returns ``None`` when the name is not a runtime cache. Be strict: a loose
    match would re-export over a directory belonging to something else.
    """
    match = _CACHE_DIR_RE.match(name)
    if match is None:
        return None
    return match.group("uuid"), match.group("disc")


def write_cache_sidecar(
    root: Path,
    cache_name: str,
    *,
    runtime_fields: list[str] | None,
    allow_pack_index: bool = True,
) -> None:
    """Record the recipe a discriminated cache was built with.

    The discriminator is a hash, so it cannot be inverted back into a recipe;
    without this, a repair of a field-trimmed cache would silently re-export
    every field and hand the job a different dataset than it trained on.

    `allow_pack_index=False` is recorded too, because a cache built for another
    machine carries no pack-index entries by design — those are absolute paths
    into THIS machine's object store — and a repair that re-exported it with an
    index would produce the same cache NAME holding payloads the remote box
    cannot resolve.

    Stays byte-identical to the historical bare-list form whenever there is
    nothing new to say, so every sidecar already on disk keeps its meaning and
    no rewrite is needed. Best-effort: a sidecar we cannot write must not fail
    the submit.
    """
    if not runtime_fields and allow_pack_index:
        return
    payload: object = list(runtime_fields) if runtime_fields else None
    if not allow_pack_index:
        payload = {"runtime_fields": list(runtime_fields) if runtime_fields else None, "pack_index": False}
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{cache_name}{_SIDECAR_SUFFIX}").write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _read_sidecar(root: Path, cache_name: str):
    """The raw sidecar payload: a legacy list, a dict, or None."""
    try:
        raw = (root / f"{cache_name}{_SIDECAR_SUFFIX}").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def read_cache_sidecar(root: Path, cache_name: str) -> list[str] | None:
    """Fields recorded for ``cache_name``; ``None`` means a full-field export."""
    parsed = _read_sidecar(root, cache_name)
    if isinstance(parsed, dict):
        parsed = parsed.get("runtime_fields")
    return [str(item) for item in parsed] if isinstance(parsed, list) and parsed else None


def read_cache_allows_pack_index(root: Path, cache_name: str) -> bool:
    """Whether this cache may be REBUILT with pack-index entries.

    True for a sidecar that says nothing — every sidecar written before this
    existed described a local cache, where an index is the normal fast path.
    Only an explicit `pack_index: false` withholds it.
    """
    parsed = _read_sidecar(root, cache_name)
    if isinstance(parsed, dict) and parsed.get("pack_index") is False:
        return False
    return True


def repair_dataset_dir(path: Path) -> bool:
    """Rebuild the runtime cache behind ``path`` if it went missing.

    Returns True when a repair was performed. No-ops (returning False) when the
    staged directory is already valid, is not a cache symlink, or the uuid does
    not resolve to a node — callers then fall through to their normal
    validation error.
    """
    if (path / "manifest.json").is_file():
        return False
    target = Path(os.path.realpath(path))
    if (target / "manifest.json").is_file():
        return False
    identity = cache_identity(target.name)
    if target.parent != runtime_cache_root() or identity is None:
        return False
    node_uuid, _discriminator = identity
    # Fields the cache was built with, so a trimmed export is rebuilt trimmed.
    # Rebuilding it full-field would hand the job a different dataset than the
    # one it was staged against, which is worse than not repairing at all.
    runtime_fields = read_cache_sidecar(target.parent, target.name)
    allows_index = read_cache_allows_pack_index(target.parent, target.name)

    from aiida import load_profile
    from aiida.orm import load_node

    load_profile()
    try:
        node = load_node(node_uuid)
    except Exception:  # noqa: BLE001 - unknown uuid: let normal validation speak
        return False

    from aiida_pino.data import PinoDatasetUnionData  # local: keep import cost off the fast path
    from aiida_pino.utils import export_dataset_union, normalize_dataset_collection_node

    print(f"[runtime-repair] cache for {target.name} is missing; re-exporting", flush=True)
    build = target.parent / f".repair-{target.name}-{os.getpid()}"
    if build.exists():
        shutil.rmtree(build)
    if isinstance(node, PinoDatasetUnionData):
        export_dataset_union(node, build)
    else:
        if not hasattr(node, "export_runtime_arraydir"):
            node = normalize_dataset_collection_node(node, label=target.name)
        import inspect

        signature = inspect.signature(node.export_runtime_arraydir).parameters
        kwargs = {}
        if runtime_fields and "runtime_fields" in signature:
            kwargs["runtime_fields"] = runtime_fields
        # A cache built for another machine must come back without an index, or
        # the repair hands that box absolute paths into this one's object store.
        if not allows_index and "allow_pack_index" in signature:
            kwargs["allow_pack_index"] = False
        node.export_runtime_arraydir(build, **kwargs)
    if not (build / "manifest.json").is_file():
        shutil.rmtree(build, ignore_errors=True)
        raise SystemExit(f"[runtime-repair] re-export produced no manifest for {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        build.replace(target)
    except OSError:
        # Another process republished it first; theirs is as good as ours.
        shutil.rmtree(build, ignore_errors=True)
    if not (target / "manifest.json").is_file():
        raise SystemExit(f"[runtime-repair] cache still missing after re-export: {target}")
    print(f"[runtime-repair] rebuilt {target}", flush=True)
    return True


def repair_staged_symlinks_in(cwd: Path) -> int:
    """Repair every dangling runtime-cache symlink directly inside ``cwd``.

    Used from the gpu-seat shim, which runs AFTER the (possibly hours-long)
    seat wait and immediately before the training reads its data. The
    adapter-time repair in ``validate_dataset_input_dir`` cannot cover that
    window — the cache can be pruned (disk-pressure cleanup evicted it to 5
    entries on 2026-07-23) between validation and the actual read. This scans
    the job dir for symlinks whose target is under the cache root and rebuilds
    any that dangle. Returns the number repaired. Never raises out — the
    training's own missing-data error is a fine fallback.
    """
    root = runtime_cache_root().resolve()
    repaired = 0
    for child in sorted(cwd.iterdir()):
        if not child.is_symlink():
            continue
        try:
            target = Path(os.path.realpath(child))
            if target.parent.resolve() != root or cache_identity(target.name) is None:
                continue
            if repair_dataset_dir(child):
                repaired += 1
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[runtime-repair] {child.name}: skipped ({type(exc).__name__}: {exc})", flush=True)
    return repaired


if __name__ == "__main__":
    import sys

    paths = [Path(a) for a in sys.argv[1:]]
    if paths:
        for p in paths:
            try:
                repair_dataset_dir(p)
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[runtime-repair] {p}: skipped ({type(exc).__name__}: {exc})", flush=True)
    else:
        repair_staged_symlinks_in(Path.cwd())
