"""Base helpers for script-backed CalcJobs."""

from __future__ import annotations

import ast
import contextlib
import fcntl
import inspect
import json
import os
import shutil
from importlib import resources
from pathlib import Path

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob, CalcJobProcessSpec

from aiida_pino.data import PinoDatasetCollectionData, PinoDatasetData, PinoDatasetUnionData
from aiida_pino.data.pino_dataset_collection import (
    assert_cache_usable_for_computer,
    pack_index_allowed_for_transport,
)
from aiida_pino.utils.runtime_cache import (
    cache_discriminator_for,
    export_kwargs_for,
    runtime_cache_name,
    runtime_cache_root,
)
from aiida_pino.utils import export_dataset_union, normalize_dataset_collection_node, validate_remote_data_for_code


def runtime_fields_from_training_profile(profile: dict) -> list[str] | None:
    """Return output VTK fields requested by the training profile, if explicit."""
    fields = profile.get("vtk_fields") or profile.get("runtime_fields")
    if fields is None:
        for override in profile.get("hydra_overrides", []) or []:
            text = str(override)
            if not text.startswith("custom.vtk_fields="):
                continue
            value = text.split("=", 1)[1].strip()
            try:
                fields = json.loads(value)
            except json.JSONDecodeError:
                try:
                    fields = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    fields = [part.strip().strip("\"'") for part in value.strip("[]").split(",") if part.strip()]
            break
    if fields is None:
        return None
    out = [str(field) for field in fields if str(field)]
    if profile.get("infer_velocities") and "Velocity" not in out:
        out.append("Velocity")
    return out or None


def _fresh_build_dir(root: Path, cache_name: str) -> Path:
    """A build directory nothing else can be writing to, with dead attempts of this cache swept.

    Named per ATTEMPT, not per pid. The pid-only name was reused by AiiDA's presubmit
    retry in the same worker process, so a worker left running by a broken pool (see
    `_run_derived_pool`) wrote into the retry's fresh directory and the retry's own
    sequential pass then found a file it had not written: `Duplicate case name
    mbc40_001` on 2026-09-10, on a collection with 61 distinct names.

    Called under `runtime_cache_build_lock`, so every existing `<cache>.build-*` sibling is
    by definition not a live attempt and is removed. A straggler still writing into one can
    recreate a partial dir; it can never collide with the path returned here, and the next
    build sweeps it again.
    """
    import uuid

    # Both spellings: the retired pid-only `.build-<cache>-<pid>` and the per-attempt one.
    for pattern in (f".build-{cache_name}-*", f"{cache_name}.build-*"):
        for stale in root.glob(pattern):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
    return root / f"{cache_name}.build-{os.getpid()}-{uuid.uuid4().hex[:8]}"


@contextlib.contextmanager
def runtime_cache_build_lock(root: Path, cache_name: str):
    """Serialise builders of ONE runtime cache, so concurrent submits share it.

    `_dataset_runtime_cache` had no lock, so every calcjob needing the same
    dataset built the whole export independently and all but one threw the result
    away at `build.replace(final)`. Measured 2026-08-05: FIVE `.build-*`
    directories for one dataset from five worker PIDs (342/431/90/342/268 MB,
    ~1.47 GB wasted), because two trainings needed the same dataset at once and daemon
    restarts kept re-entering. Each simulation there decompresses to 3.3 GB, so
    the duplication costs RAM and page cache, not only CPU.

    `flock` specifically: the kernel drops it when the holder exits, crash or
    kill, so a builder that dies mid-export cannot wedge every future submit --
    which matters on a box where workers get killed (see the same incident).

    The lock file sits BESIDE the cache directory, never inside it, so it
    survives the atomic `build.replace(final)` publish.
    """

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f"{cache_name}.lock"
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


@contextlib.contextmanager
def runtime_cache_materialise_slot(root: Path):
    """Serialise materialisation across DIFFERENT datasets, not just the same one.

    `runtime_cache_build_lock` is per CACHE NAME, so two builds of the same dataset share
    work but two builds of different datasets run side by side. That is what filled the disk
    on 2026-08-27: two datasets (~40 GB and ~42 GB) materialised concurrently, wrote
    82 GB in about three minutes and took root from 100 GB to 19 GB.

    No individual check was wrong. `refuse_build_that_cannot_fit` is per-build and stateless,
    so both builds read the same 100 GB free before either had written a byte, and both
    passed. A space check cannot see a sibling that has not allocated yet; only a lock can.

    This makes the "one dataset at a time" rule mechanical instead of
    manual. `flock` for the same reason as the sibling lock: the kernel drops it when the
    holder exits, so a killed builder cannot wedge every later submission.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".materialise.lock"
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


def remote_dataset_symlink_entry(node, computer, *, runtime_fields, target, is_union: bool = False):
    """The `remote_symlink_list` entry for a dataset already staged on `computer`.

    A remote job must never be handed the LOCAL cache path. That is what used to
    happen -- `stage_dataset_input` emitted the local path whatever the computer
    was -- and AiiDA dutifully created a symlink to a directory that does not
    exist on the compute box. The job then started and died at step 0 reporting
    a missing runtime manifest, which reads as a corrupt dataset rather than as
    a machine that was never given one.

    The copy is found by cache NAME, computed by the same function staging uses,
    so a field-set change misses loudly instead of silently training on the
    wrong fields. Nothing here opens a transport: this runs inside
    `prepare_for_submission`, and the error names the collection *actually
    resolved* at submit time, which is what closes the reuse-substitution gap.
    """
    from aiida_pino.utils.data_nodes import find_remote_runtime_cache
    from aiida_pino.utils.runtime_cache import cache_discriminator_for, runtime_cache_name

    if is_union:
        raise ValueError(
            f"Dataset union {getattr(node, 'uuid', '?')} cannot run on remote computer "
            f"`{computer.label}`: `export_dataset_union` exports its members without "
            "`allow_pack_index=False`, so the cache carries absolute pack paths into this "
            "machine's object store and resolves to nothing there. Run this union locally, "
            "or compose the members into a base/composed dataset and stage that."
        )

    transport_type = getattr(computer, "transport_type", "core.local")
    discriminator = cache_discriminator_for(node, runtime_fields=runtime_fields, transport_type=transport_type)
    cache_name = runtime_cache_name(node.uuid, discriminator)
    staged = find_remote_runtime_cache(computer, cache_name)
    if staged is None:
        fields = " ".join(runtime_fields) if runtime_fields else ""
        fields_hint = f" --fields {fields}" if fields else ""
        # Say WHICH collection this is. A training's inputs are often base
        # collections, but what the job actually consumes is a DERIVED
        # collection the workchain creates at submit time from the derivation
        # profile — it does not exist to be staged beforehand. Someone who
        # staged the dataset they passed, and is now reading a uuid they have
        # never seen, needs that sentence more than they need the command.
        origin = ""
        try:
            metadata = node.get_dataset_metadata() if hasattr(node, "get_dataset_metadata") else {}
            if str(metadata.get("dataset_role", "")) == "derived":
                parent = metadata.get("parent_collection_pk")
                origin = (
                    f"\nThis is a DERIVED collection"
                    + (f" (from collection pk {parent})" if parent else "")
                    + ", created at submit time from the derivation profile — so it is NOT "
                    "the dataset you passed, and it could not have been staged in advance."
                )
        except Exception:  # noqa: BLE001 - the message must not fail
            origin = ""
        raise ValueError(
            f"Dataset {node.uuid} is not staged on computer `{computer.label}` "
            f"(expected runtime cache `{cache_name}`). A remote job cannot build it at "
            f"submit time — the payloads are here, not there.{origin}\nRun:\n"
            # The PK, not the uuid: this line is copied verbatim by whoever
            # reads the error, so it has to be a command that runs.
            f"  aiida-pino stage-remote {getattr(node, 'pk', None) or node.uuid} "
            f"--computer {computer.label}{fields_hint}"
        )
    return (computer.uuid, staged.get_remote_path(), target)


#: Modules staged into every job sandbox beside the adapter itself. A compute box
#: has no `aiida_pino` installed, so an adapter can only import from these.
STAGED_HELPERS = ("live_tail.py", "staged_dataset.py")


class ScriptCalculation(CalcJob):
    """Base class for CalcJobs that stage and execute a bundled adapter script."""

    script_name: str = ""
    parser_entry_point: str = ""

    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        super().define(spec)
        spec.inputs["metadata"]["options"]["parser_name"].default = cls.parser_entry_point
        spec.inputs["metadata"]["options"]["input_filename"].default = "aiida.in"
        spec.inputs["metadata"]["options"]["output_filename"].default = "aiida.out"
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["withmpi"].default = False

    def stage_script(self, folder: Folder) -> str:
        """Write the bundled adapter script into the job sandbox."""
        if not self.script_name:
            raise ValueError("`script_name` must be defined on the calculation class.")
        script_text = resources.files("aiida_pino.scripts").joinpath(self.script_name).read_text(encoding="utf-8")
        with folder.open(self.script_name, "w", encoding="utf-8") as handle:
            handle.write(script_text)
        # Everything an adapter may import must be staged with it: a compute box
        # has no `aiida_pino` package. Adding a name here is the ONLY way to make
        # a helper importable there, which is why the metric parser had to move
        # into one of these files rather than stay in `utils/`.
        for helper_name in STAGED_HELPERS:
            if self.script_name == helper_name:
                continue
            helper_text = resources.files("aiida_pino.scripts").joinpath(helper_name).read_text(encoding="utf-8")
            with folder.open(helper_name, "w", encoding="utf-8") as handle:
                handle.write(helper_text)
        return self.script_name

    def build_codeinfo(self, cmdline_params: list[str]) -> CodeInfo:
        """Construct a `CodeInfo` for the calculation."""
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline_params
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.withmpi = False
        return codeinfo

    def build_calcinfo(
        self,
        codeinfo: CodeInfo,
        retrieve_list: list[str],
        retrieve_temporary_list: list[str] | None = None,
        local_copy_list: list[tuple[str, str, str | None]] | None = None,
        remote_symlink_list: list[tuple[str, str, str | None]] | None = None,
    ) -> CalcInfo:
        """Construct the `CalcInfo` for the calculation."""
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = retrieve_list
        calcinfo.retrieve_temporary_list = retrieve_temporary_list or []
        calcinfo.local_copy_list = local_copy_list or []
        calcinfo.remote_symlink_list = remote_symlink_list or []
        return calcinfo

    def stage_folderlike_input(
        self,
        node: orm.FolderData | orm.RemoteData,
        target: str,
    ) -> tuple[list[tuple[str, str, str | None]], list[tuple[str, str, str | None]]]:
        """Stage a `FolderData` or `RemoteData` input into the job sandbox."""
        if isinstance(node, orm.FolderData):
            return [(node.uuid, ".", target)], []

        validate_remote_data_for_code(node, self.inputs.code)
        return [], [(node.computer.uuid, node.get_remote_path(), target)]

    def _dataset_runtime_cache(self, node, exporter, *, cache_discriminator: dict | None = None) -> Path:
        """Export a dataset's runtime layout once into a persistent cache dir.

        Exporting into the CalcJob sandbox made AiiDA's upload duplicate the
        whole dataset into the work directory (measured 51 GB / ~9 min per
        submit for a 247-case train+val, left behind by failed jobs). The
        cache is keyed by node uuid (collections are immutable).

        **The cache is only cheap while the payloads are loose objects.**
        `_try_link_repository_object` hardlinks the payload, but bails out when
        `get_object_meta().type` is not `loose` (pino_simulation.py:575) — a
        packed object has no standalone file to link to, so the export falls
        back to a streamed copy. Packing the container therefore turns this
        cache from directory entries into real bytes: after the 2026-07-24 pack
        it held 51 GB of genuine copies (247 npz, mean 219 MB). Nothing evicts
        it either — entries accumulate per node uuid forever.

        **Only clear it when no job is queued or running.** This docstring used
        to say `rm -rf ~/.cache/aiida-pino/runtime` was simply safe; it is safe
        for *future* submits, which re-export, and destructive for every job
        already uploaded. `stage_dataset_input` symlinks the workdir here at
        upload time, and a job can then sit hours in the gpu_spool seat queue or
        read from it for hours while training. Clearing it mid-flight has cost
        seven runs (five queued ablation arms 2026-07-22, plus two RIGNO
        trainings killed at step 1500 and step 18000 with a `FileNotFoundError`
        from a DataLoader worker). `runtime_repair.repair_dataset_dir` rebuilds
        a vanished cache at job start, and `NpzDirStore` retries once mid-run,
        but neither helps a job reading a directory that is being deleted
        underneath it. Drain the queue first, then clear stale uuids — with a
        packed container that is real disk space, not just directory entries.
        """
        from aiida_pino.utils.runtime_cache import sidestep_cache_name, staged_cache_is_current

        root = runtime_cache_root()
        cache_name = runtime_cache_name(node.uuid, cache_discriminator)
        # Existence alone used to be the whole test, so a cache built by OLDER staging
        # code was reused forever: nothing downstream can tell, because a stale cache is
        # a valid cache of the wrong thing.
        #
        # A stale one is SIDESTEPPED, never deleted. Deleting it is what the warning
        # above this method describes -- a job reading the directory dies with
        # `FileNotFoundError` from a DataLoader worker -- and a staging fix landing while
        # jobs are running is exactly when this branch fires, which is the same day those
        # jobs are mid-flight. So the fresh build goes to a sibling name, and whoever is
        # still reading the old one keeps their files.
        #
        # Decided BEFORE the sidecar is written, so the sidecar lands against the name
        # actually used; writing it for the pre-rename name would leave the new cache
        # without one, which is what makes the gpu-seat shim rebuild it full-field.
        usable, reason = staged_cache_is_current(root / cache_name)
        if not usable and (root / cache_name).is_dir() and reason != "no manifest":
            cache_name = sidestep_cache_name(root, cache_name)
            print(
                f"[runtime-cache] {reason}; staging a fresh cache as {cache_name} and "
                f"leaving the old directory for any job still reading it",
                flush=True,
            )
        # Record the fields *before* returning, on the reuse path too: a cache
        # published by an older submit has no sidecar, and without one the
        # gpu-seat shim would rebuild it full-field if it ever went missing.
        from aiida_pino.utils.runtime_repair import write_cache_sidecar

        write_cache_sidecar(
            root,
            cache_name,
            runtime_fields=(cache_discriminator or {}).get("runtime_fields"),
            # The discriminator says `pack_index: False` for a cache built for
            # another machine. Recording it is what stops a later repair
            # rebuilding the same NAME with absolute pack paths into this
            # machine's object store, which resolve to nothing over there.
            allow_pack_index=(cache_discriminator or {}).get("pack_index") is not False,
        )
        final = root / cache_name
        if staged_cache_is_current(final)[0]:
            return final
        root.mkdir(parents=True, exist_ok=True)
        _refuse_if_it_cannot_fit(root, node, cache_discriminator)
        # Serialise builders of this cache. Without it, every calcjob needing the
        # same dataset ran the whole export concurrently and all but one threw the
        # result away -- five simultaneous builds of one dataset were measured
        # (~1.47 GB wasted, 2026-08-05). Waiting here is strictly cheaper than
        # duplicating a multi-hour, 3.3 GB-per-simulation export.
        with runtime_cache_materialise_slot(root), runtime_cache_build_lock(root, cache_name):
            # Re-check under the lock: whoever held it before us may have
            # published the cache while we waited.
            if staged_cache_is_current(final)[0]:
                return final
            # cache_name, not the raw uuid: two differently-discriminated builds
            # of the same node in one process would otherwise share a build
            # directory, and the second would delete the first's work mid-export.
            build = _fresh_build_dir(root, cache_name)
            # try/finally, not just the OSError path below: a builder that dies INSIDE
            # `exporter` -- ENOSPC mid-export, OOM, a killed worker -- used to leave its
            # partial `.build-` directory behind, which is how 159 GB sat on an already-full
            # disk after a crash on 2026-08-25. On the success path `replace` has already
            # moved the directory, so the rmtree is a no-op. Same reasoning as the flock:
            # the holder can die, so cleanup cannot live only on the happy path.
            try:
                exporter(build)
                if not (build / "manifest.json").is_file():
                    raise ValueError(f"Failed to build dataset runtime cache at `{build}`.")
                try:
                    build.replace(final)
                except OSError:
                    # Lost the build race: another submit already published the cache.
                    pass
            finally:
                shutil.rmtree(build, ignore_errors=True)
            if not (final / "manifest.json").is_file():
                raise ValueError(f"Dataset runtime cache missing at `{final}`.")
            return final

    def stage_dataset_input(
        self,
        folder: Folder,
        node: PinoDatasetCollectionData | PinoDatasetData | PinoDatasetUnionData | orm.FolderData,
        target: str,
        *,
        runtime_fields: list[str] | None = None,
    ) -> tuple[list[tuple[str, str, str | None]], list[tuple[str, str, str | None]]]:
        """Stage a dataset for a job: cached runtime export + work-dir symlink.

        Native nodes never enter the sandbox; the job gets a symlink to the
        per-uuid runtime cache (same mechanism the eval calc already uses for
        checkpoints). Legacy FolderData keeps the repository copy path.
        """
        computer = self.inputs.code.computer
        computer_uuid = computer.uuid
        # A remote job reads its own machine's disk. Deciding this ONCE, before
        # the union branch, is deliberate: that branch used to return early with
        # a local path, so it never reached the transport check at all and a
        # remote union staged a directory that does not exist there.
        remote_job = not pack_index_allowed_for_transport(getattr(computer, "transport_type", "core.local"))

        if isinstance(node, PinoDatasetUnionData):
            if remote_job:
                remote_dataset_symlink_entry(node, computer, runtime_fields=runtime_fields, target=target, is_union=True)
            cache = self._dataset_runtime_cache(node, lambda path: export_dataset_union(node, path))
            return [], [(computer_uuid, str(cache), target)]

        # Prefer capability-based native staging over strict class identity checks.
        # This guards against edge cases where daemon workers may load equivalent node
        # classes through different module objects.
        if not isinstance(node, orm.FolderData):
            if not hasattr(node, "export_runtime_arraydir"):
                node = normalize_dataset_collection_node(node, label=target)
            if remote_job:
                # Never build locally for a remote job: the export would take
                # tens of minutes and 51 GB to produce a directory the box
                # cannot read. `stage-remote` puts the copy there ahead of time.
                return [], [remote_dataset_symlink_entry(node, computer, runtime_fields=runtime_fields, target=target)]
            # Pack-index entries carry ABSOLUTE pack paths into this machine's
            # object store, so only a local transport can resolve them. The
            # runtime cache is keyed by node uuid, so without the discriminator
            # a cache built for a local job would be handed to a remote one and
            # every payload would resolve to a file that is not there.
            #
            # Both of these come from `utils.runtime_cache` rather than being
            # rebuilt here, because `aiida-pino stage-remote` must land its copy
            # under the name this function will later look up. Two
            # implementations that agree today drift silently, and the symptom
            # is not an error -- it is a job that stages nothing.
            transport_type = getattr(self.inputs.code.computer, "transport_type", "core.local")
            export_kwargs = export_kwargs_for(node, runtime_fields=runtime_fields, transport_type=transport_type)
            discriminator = cache_discriminator_for(node, runtime_fields=runtime_fields, transport_type=transport_type)
            cache = self._dataset_runtime_cache(
                node,
                lambda path: node.export_runtime_arraydir(path, **export_kwargs),
                cache_discriminator=discriminator or None,
            )
            # Defence in depth, and no longer THE guarantee — it used to be,
            # and the comment here said so, but this branch is now reached only
            # by LOCAL jobs (a remote one returns above without building), where
            # an index is always allowed and this can never refuse.
            #
            # The guarantee moved to where a cache is actually prepared for
            # another machine: `stage_collection_to_computer` asserts the same
            # thing on the built cache before it transfers anything. This line
            # stays because it costs nothing and would matter again the day the
            # early return above is removed — but do not read it as the check
            # that protects remote jobs, because it is not reached by them.
            assert_cache_usable_for_computer(cache, transport_type)
            return [], [(computer_uuid, str(cache), target)]

        return [(node.uuid, ".", target)], []


def _refuse_if_it_cannot_fit(root, node, discriminator) -> None:
    """Size gate, applied at BOTH build sites.

    `_dataset_runtime_cache` and `warm_dataset_runtime_cache` each build independently, so
    gating one leaves the other able to fill the filesystem -- the same split that let
    stale caches be served from one path after the other was fixed.

    Best-effort about reading the members: a node that cannot enumerate them is not a
    reason to block a build that would otherwise work.
    """
    from aiida_pino.utils.runtime_cache import refuse_build_that_cannot_fit

    try:
        members = list(node.iter_member_simulations())
    except Exception:  # noqa: BLE001 - unions and folder-backed nodes need not support it
        return
    if not members:
        return
    profile = {}
    try:
        profile = (node.get_dataset_metadata() or {}).get("derivation_profile") or {}
    except Exception:  # noqa: BLE001
        profile = {}
    refuse_build_that_cannot_fit(
        root, members, profile, label=f"{getattr(node, 'label', '')!r} (pk {getattr(node, 'pk', '?')})"
    )


def warm_dataset_runtime_cache(node, *, runtime_fields=None, transport_type: str = "core.local") -> Path:
    """Build a dataset's runtime cache NOW, off any calcjob's critical path.

    The export runs inside `do_upload`, so a training cannot reach the GPU until
    it finishes. Measured 2026-08-05: both GPUs sat idle for hours while one core
    decompressed a 120-simulation dataset (3.3 GB each) for two queued
    trainings. `_dataset_runtime_cache` short-circuits on an existing
    `manifest.json`, so building the cache BEFORE submitting turns that upload
    into a no-op and the job goes straight to the GPU.

    Safe to call while other builders are running: it takes the same lock, so at
    worst it waits and reuses their result.
    """

    from aiida_pino.data import PinoDatasetUnionData
    from aiida_pino.utils import export_dataset_union, normalize_dataset_collection_node

    root = runtime_cache_root()
    # Same naming function as `stage_dataset_input`, not a second copy of the
    # rule. `transport_type` defaults to local, which reproduces the historical
    # name exactly (`runtime_fields` only, never `pack_index`) so no cache
    # already on disk is orphaned; `stage-remote` passes the compute box's
    # transport so it builds and names the copy that box will actually look up.
    discriminator = (
        {"runtime_fields": list(runtime_fields)} if runtime_fields else {}
    )
    if not pack_index_allowed_for_transport(transport_type):
        discriminator["pack_index"] = False
    cache_name = runtime_cache_name(node.uuid, discriminator or None)

    # SECOND place the cache-reuse decision is made (the first is
    # `_dataset_runtime_cache`). Both must apply the staging-code check or the fix is
    # only half live -- caught by
    # `test_a_stale_cache_is_sidestepped_and_never_deleted`, which reused a stale cache
    # through THIS function while the other path was already correct. Same sidestep, and
    # same reason for not deleting: a running job may hold the old directory open.
    from aiida_pino.utils.runtime_cache import sidestep_cache_name, staged_cache_is_current

    usable, reason = staged_cache_is_current(root / cache_name)
    if not usable and (root / cache_name).is_dir() and reason != "no manifest":
        cache_name = sidestep_cache_name(root, cache_name)
        print(
            f"[runtime-cache] {reason}; warming a fresh cache as {cache_name} and "
            f"leaving the old directory for any job still reading it",
            flush=True,
        )

    final = root / cache_name
    if staged_cache_is_current(final)[0]:
        return final

    if isinstance(node, PinoDatasetUnionData):
        exporter = lambda path: export_dataset_union(node, path)  # noqa: E731
    else:
        target = node
        if not hasattr(target, "export_runtime_arraydir"):
            target = normalize_dataset_collection_node(target, label=cache_name)
        kwargs = export_kwargs_for(target, runtime_fields=runtime_fields, transport_type=transport_type)
        exporter = lambda path: target.export_runtime_arraydir(path, **kwargs)  # noqa: E731

    root.mkdir(parents=True, exist_ok=True)
    _refuse_if_it_cannot_fit(root, node, discriminator)
    with runtime_cache_materialise_slot(root), runtime_cache_build_lock(root, cache_name):
        if staged_cache_is_current(final)[0]:
            return final
        # cache_name, not the raw uuid: two differently-discriminated builds
        # of the same node in one process would otherwise share a build
        # directory, and the second would delete the first's work mid-export.
        build = _fresh_build_dir(root, cache_name)
        # try/finally, not just the OSError path below: a builder that dies INSIDE
        # `exporter` -- ENOSPC mid-export, OOM, a killed worker -- used to leave its
        # partial `.build-` directory behind, which is how 159 GB sat on an already-full
        # disk after a crash on 2026-08-25. On the success path `replace` has already
        # moved the directory, so the rmtree is a no-op. Same reasoning as the flock:
        # the holder can die, so cleanup cannot live only on the happy path.
        try:
            exporter(build)
            if not (build / "manifest.json").is_file():
                raise ValueError(f"Failed to warm dataset runtime cache at `{build}`.")
            try:
                build.replace(final)
            except OSError:
                pass
        finally:
            shutil.rmtree(build, ignore_errors=True)
        return final
