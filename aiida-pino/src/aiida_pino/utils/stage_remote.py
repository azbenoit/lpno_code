"""Copy a dataset's runtime cache onto a compute box, and prove it arrived.

The transport gives you nothing to go on. Measured against the first real
compute box (2026-08-19), `puttree` returns `None` — no byte count, no file
list, no success signal beyond not raising — creates a missing destination
parent silently, and on an existing destination behaves like `cp -r`, landing
the source *inside* it under its own basename. So a retry after a partial
failure leaves a NESTED duplicate rather than replacing anything.

That last one dictates the shape of the check here. A nested duplicate has MORE
files and MORE bytes than expected, so "the count matches", "the total size is
at least right" and "manifest.json is present" all pass on the single most
likely corruption. The only check that catches it compares an exact set of
relative paths with their sizes and treats an unexpected path as a failure.

Verification runs ON THE REMOTE, because the question is what is on that disk.
"""

from __future__ import annotations

import shutil

from pathlib import Path
from typing import Any

#: `%s %P` puts the size first so a path containing spaces survives the split, and
#: records are NUL-delimited because a newline in a filename can FORGE an entry.
#: Demonstrated on a real box: a file literally named `decoy\n11 manifest.json`
#: makes a newline-delimited listing emit `11 manifest.json` as its own line, so an
#: ABSENT manifest.json passes both the presence and the size check. NUL is the one
#: byte that cannot appear in a POSIX filename, so this is airtight where escaping
#: or quoting would not be.
REMOTE_LISTING_COMMAND = "cd {directory} && find . -type f -printf '%s %P\\0'"


def expected_cache_contents(local_dir: str | Path) -> dict[str, int]:
    """Every file under `local_dir` as `relative posix path -> size in bytes`.

    Follows the tree rather than trusting the manifest: the manifest lists
    members, not the sidecars and per-case files beside them, and a check that
    only looks at what the manifest names cannot see a file that failed to
    transfer alongside it.
    """
    root = Path(local_dir)
    contents: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            contents[path.relative_to(root).as_posix()] = path.stat().st_size
    return contents


def parse_remote_listing(stdout: str) -> dict[str, int]:
    """Parse NUL-delimited `find . -type f -printf '%s %P\\0'` output.

    Splitting on NUL rather than newline is what makes a filename unable to
    forge a record for a different file. Only the leading size is stripped; the
    rest of the record is the path verbatim, newlines and all.
    """
    contents: dict[str, int] = {}
    for record in (stdout or "").split("\0"):
        if not record:
            continue
        size, _, relative = record.partition(" ")
        if not relative or not size.isdigit():
            continue
        contents[relative] = int(size)
    return contents


# Why sizes, and not content hashes. Measured on `core.ssh_async` (asyncssh/SFTP)
# by killing a real 419 MB `puttree` mid-flight: the partial file is TRUNCATED
# (412,569,600 bytes for an expected 419,430,400), with no pre-allocation
# (805,800 blocks x 512 = exactly the apparent size) and no zero-filled or sparse
# tail (the last 16 bytes are real data). So an interruption can only ever
# produce a file SHORTER than the source; equality would mean it finished. There
# is no route by which a partial file lands on the expected size with wrong
# content, which is the one case a size check would be blind to.
#
# Hashing would also be guarding a channel that is already integrity-protected:
# SSH MACs every packet, so silent in-transit corruption does not reach the
# filesystem. At multi-GB cache sizes that cost buys nothing.
#
# Note the failure SHAPE this implies, and why a total-bytes check is not
# equivalent: files transfer sequentially, so an interruption leaves the earlier
# files complete and exactly one file short. A "total within tolerance" check
# absorbs that; an exact per-path map cannot.
#
# BOUNDARY: this is a fact about this transport, not about transfers. A
# resumable or pre-allocating protocol could leave a full-length file with a
# partial body, and this conclusion would have to be re-measured rather than
# assumed. (Measured 2026-08-19.)
def compare_cache_contents(expected: dict[str, int], actual: dict[str, int]) -> list[str]:
    """Everything wrong with `actual`, as human-readable lines. Empty means good.

    An UNEXPECTED path is a failure, not a curiosity. That is the nested-copy
    case, and it is the reason this returns problems rather than a boolean over
    counts or totals.
    """
    problems: list[str] = []
    for relative, size in sorted(expected.items()):
        if relative not in actual:
            problems.append(f"missing on the remote: {relative}")
        elif actual[relative] != size:
            problems.append(f"size mismatch: {relative} expected {size} bytes, found {actual[relative]}")
    # LOAD-BEARING, and not merely tidiness — do not downgrade this to a warning.
    # It catches two different things. A re-put nests a duplicate tree instead of
    # replacing it, which has MORE files and MORE bytes than expected and so
    # passes every count- or size-based check. And a filename containing a
    # newline can forge a listing record for a file that does not exist, which
    # passes both the presence and the size check — the forged name's real,
    # oddly-named file is the only trace, and it shows up here as an extra.
    for relative in sorted(set(actual) - set(expected)):
        # The hint names BOTH causes. It used to assert the nesting one, which is
        # a good lead for a re-put and a misleading one for a hostile filename --
        # and the hostile case is the one where a confident wrong explanation
        # costs the most, because the reader is already looking at something odd.
        problems.append(
            f"unexpected extra path on the remote: {relative!r} "
            "(either a re-put nested a duplicate tree instead of replacing it, or a "
            "file is named so as to look like another entry in the listing)"
        )
    return problems


#: Hash every file under a directory on the REMOTE box, dropping each file's page cache
#: first. The drop is not optional: a checksum taken shortly after writing is served the
#: writer's still-cached pages and passes while the on-disk bytes are bad -- the same
#: torn-file effect recorded for t64v_087, and the reason the reporter's first
#: post-transfer md5 was worthless. `posix_fadvise(DONTNEED)` needs no root.
#:
#: Base64 so nothing in the snippet has to survive shell quoting.
_REMOTE_CHECKSUM_SNIPPET = """
import hashlib, os, sys
root = sys.argv[1]
out = []
for dirpath, _dirs, files in os.walk(root):
    for name in sorted(files):
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, root)
        fd = os.open(path, os.O_RDONLY)
        try:
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            digest = hashlib.md5()
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(fd)
        out.append(digest.hexdigest() + " " + rel)
sys.stdout.write("\\0".join(out))
"""


def remote_checksum_command(remote_dir: str, python_executable: str = "python3") -> str:
    """The shell command that prints `md5<space>relpath` records, NUL-delimited."""
    import base64

    encoded = base64.b64encode(_REMOTE_CHECKSUM_SNIPPET.encode("utf-8")).decode("ascii")
    return (
        f"echo {encoded} | base64 -d | {_quote(python_executable)} - {_quote(remote_dir)}"
    )


def local_checksums(local_dir: Path) -> dict[str, str]:
    """`relative posix path -> md5` for every file under `local_dir`."""
    import hashlib

    root = Path(local_dir)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        out[path.relative_to(root).as_posix()] = digest.hexdigest()
    return out


def parse_checksum_records(stdout: str) -> dict[str, str]:
    """Parse the NUL-delimited `md5 relpath` records from the remote snippet."""
    out: dict[str, str] = {}
    for record in stdout.split("\0"):
        record = record.strip()
        if not record:
            continue
        digest, _, relative = record.partition(" ")
        if relative and len(digest) == 32:
            out[relative] = digest
    return out


def compare_checksums(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    """Content problems, in the same shape `verify_remote_cache` returns.

    A file the remote could not hash is a PROBLEM, never a pass. Silently treating an
    unverifiable file as fine is the failure this whole check exists to remove -- and is
    exactly how the staged-cache repair stayed inert on remote boxes for months.
    """
    problems: list[str] = []
    for relative in sorted(expected):
        if relative not in actual:
            problems.append(f"not checksummed on the remote: {relative}")
        elif actual[relative] != expected[relative]:
            problems.append(
                f"CONTENT mismatch: {relative} hashed {actual[relative]}, expected "
                f"{expected[relative]} (same size, different bytes -- a size check passes this)"
            )
    return problems


def verify_remote_cache(
    transport,
    remote_dir: str,
    expected: dict[str, int],
    *,
    expected_checksums: dict[str, str] | None = None,
) -> list[str]:
    """List `remote_dir` on the box and compare it to `expected`.

    With `expected_checksums`, also hash every remote file with its page cache dropped and
    compare CONTENT. Sizes alone passed a real corruption: `r150np_137.npz` arrived with
    the same byte count and different bytes, failed CRC, and killed training 67664 at ~87
    steps (2026-08-25). Opt-in because it re-reads the whole cache -- minutes for a 20 GB
    collection -- and because the remote half is not yet exercised against a live box.
    """
    retval, stdout, stderr = transport.exec_command_wait(REMOTE_LISTING_COMMAND.format(directory=_quote(remote_dir)))
    if retval != 0:
        return [f"could not list {remote_dir} on the remote (exit {retval}): {(stderr or '').strip()[:200]}"]
    problems = compare_cache_contents(expected, parse_remote_listing(stdout))
    if expected_checksums is None:
        return problems
    retval, stdout, stderr = transport.exec_command_wait(remote_checksum_command(remote_dir))
    if retval != 0:
        # Not a pass: an unverifiable copy must not be recorded as verified.
        return problems + [
            f"could not checksum {remote_dir} on the remote (exit {retval}): "
            f"{(stderr or '').strip()[:200]}"
        ]
    return problems + compare_checksums(expected_checksums, parse_checksum_records(stdout))


def _quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def destination_is_this_directory(transport, local_dir: Path, remote_dir: str) -> bool:
    """Whether `remote_dir` is `local_dir` — the same directory, not the same name.

    Writes a uniquely named probe into the local cache and asks the box whether
    it can see it at the destination. Same filesystem means same file, which is
    the property that actually matters: it catches a computer pointing back at
    this machine, and it clears a genuinely remote box that merely shares
    `/home/<user>` — the normal result of consistent provisioning, and a case a
    string comparison gets exactly backwards.

    Cheaper checks were considered and rejected: hostname or `/etc/machine-id`
    would distinguish the hosts, but two different hosts can share a mount, and
    then identical paths really are the same directory. The probe tests the
    dependency rather than a proxy for it.

    Fails OPEN on an unusable transport: an unanswerable probe must not block
    staging, because the guard protects against a rare misconfiguration while a
    false refusal breaks the ordinary path.
    """
    import shlex
    import uuid

    probe = local_dir / f".aiida-pino-stage-probe-{uuid.uuid4().hex}"
    try:
        probe.write_text("aiida-pino staging identity probe", encoding="utf-8")
    except OSError:
        return False
    try:
        target = shlex.quote(f"{str(remote_dir).rstrip('/')}/{probe.name}")
        retval, stdout, _ = transport.exec_command_wait(f"test -e {target} && echo YES || echo NO")
        return retval == 0 and "YES" in (stdout or "")
    except Exception:  # noqa: BLE001 - see "fails OPEN" above
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def stage_cache_to_computer(
    transport,
    *,
    local_dir: str | Path,
    remote_dir: str,
    replace: bool = True,
    verify_content: bool = True,
) -> dict[str, Any]:
    """Put `local_dir` at `remote_dir` and verify it, raising on any problem.

    `replace` removes the destination first. Without that a second run nests a
    duplicate inside the first, which every size-based check would accept.
    """
    local_dir = Path(local_dir)
    if not (local_dir / "manifest.json").is_file():
        raise ValueError(f"{local_dir} is not a runtime cache (no manifest.json); refusing to stage it.")
    expected = expected_cache_contents(local_dir)
    if not expected:
        raise ValueError(f"{local_dir} is empty; refusing to stage it.")

    # Is the destination THIS directory? Decided by observation, not by
    # comparing path strings. Consistent user provisioning gives every machine
    # `/home/<user>`, so a genuinely remote box normally resolves to the same
    # path as this one — string comparison refused exactly the deployment this
    # command was written for, while a localhost `core.ssh_async` computer (the
    # destructive case) has to stay refused. Only the filesystem can tell those
    # apart, so ask it.
    if destination_is_this_directory(transport, local_dir, remote_dir):
        raise ValueError(
            f"Refusing to stage {local_dir} onto itself: the destination resolves to the "
            "same directory on the same filesystem, so clearing it would delete the cache "
            "being copied. This happens with a computer that points back at this machine; "
            "for a genuinely remote box, pass --remote-root to place its caches elsewhere."
        )

    if replace:
        # Unconditional: `puttree` onto an existing directory nests rather than
        # replaces, so "only clean up if it looks wrong" leaves the one failure
        # mode that verification would otherwise have to untangle.
        try:
            transport.rmtree(remote_dir)
        except Exception:  # noqa: BLE001 - absent is the normal case
            pass
    # NO `makedirs` here. `puttree` creates missing parents itself, and creating
    # the destination first is what makes it nest: `cp -r` semantics branch on
    # whether the destination EXISTS, not on whether it holds anything. A
    # `makedirs` one line after the `rmtree` above therefore defeated the
    # replace entirely and nested on every call, clean or not (measured on a
    # real box, 2026-08-19).
    transport.puttree(str(local_dir), remote_dir)

    problems = verify_remote_cache(
        transport,
        remote_dir,
        expected,
        expected_checksums=local_checksums(local_dir) if verify_content else None,
    )
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise ValueError(
            f"Staging {local_dir.name} to {remote_dir} did not verify:\n  {shown}{more}\n"
            "The copy on the remote is NOT usable and nothing has been recorded for it. "
            "Re-running staging clears the destination first, so leftover state from a "
            "failed attempt does not accumulate — but if the same problem repeats, it is "
            "not leftover state and re-running will not fix it."
        )
    return {
        "remote_path": remote_dir,
        "file_count": len(expected),
        "total_bytes": sum(expected.values()),
    }


def remote_cache_root(transport, override: str | None = None) -> str:
    """Where runtime caches live on the compute box.

    Asks the box for its own `$HOME` rather than reusing this machine's. The
    two happen to be identical on the first compute box, which is exactly why
    assuming it is unsafe: the assumption would go untested until the first box
    that differs, and its failure mode is a path that looks right.
    """
    if override:
        return str(override).rstrip("/")
    retval, stdout, stderr = transport.exec_command_wait("printf %s \"$HOME\"")
    home = (stdout or "").strip()
    if retval != 0 or not home:
        raise ValueError(
            "Could not determine the remote home directory "
            f"(exit {retval}): {(stderr or '').strip()[:200]}. "
            "Pass --remote-root to say where runtime caches should live."
        )
    return f"{home}/.cache/aiida-pino/runtime"


def _pid_alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process, but a process
    return True


def _sweep_dead_stream_dirs(root: Path) -> list[Path]:
    """Remove `<cache>.stream-<pid>` build dirs whose builder is gone.

    The failure path deletes its own build dir, but a SIGTERM never reaches it: on
    2026-09-09 a killed staging left 62 GB under `.stream-800096` on a root at 96 %, found
    only because a disk monitor fired. The pid is in the name, so liveness is one syscall.
    """
    removed: list[Path] = []
    for path in root.glob("*.stream-*"):
        pid = path.name.rsplit("-", 1)[-1]
        if not path.is_dir() or not pid.isdigit() or _pid_alive(int(pid)):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
        print(f"[stage-remote] swept orphaned build dir {path.name}", flush=True)
    return removed


def stage_collection_to_computer(
    collection,
    computer,
    *,
    runtime_fields=None,
    remote_root: str | None = None,
    dry_run: bool = False,
    verify_content: bool = True,
) -> dict[str, Any]:
    """Build this collection's runtime cache and put it on `computer`.

    Named with the same function staging uses, so the copy lands under the name
    `prepare_for_submission` will look for. Verified on the remote before it is
    recorded: an unverified copy must not become a `RemoteData` that a later
    submission trusts.
    """
    from aiida_pino.data.pino_dataset_collection import assert_cache_usable_for_computer
    from aiida_pino.utils.data_nodes import register_remote_runtime_cache
    from aiida_pino.utils.runtime_cache import cache_discriminator_for, runtime_cache_name

    transport_type = getattr(computer, "transport_type", "core.local")
    if transport_type in ("core.local",):
        raise ValueError(
            f"Computer `{computer.label}` is local; its jobs read the runtime cache directly "
            "and nothing needs staging."
        )

    # From the shared function, NOT hand-rebuilt. The hand-built version added
    # `pack_index: False` unconditionally while `stage_dataset_input` only adds
    # it when the exporter actually accepts `allow_pack_index` — so for an
    # exporter without that parameter the two produced DIFFERENT names and the
    # lookup missed permanently. That is exactly the drift this module was
    # written to make impossible.
    discriminator = cache_discriminator_for(
        collection, runtime_fields=runtime_fields, transport_type=transport_type
    )
    cache_name = runtime_cache_name(collection.uuid, discriminator)

    # Announce the LOCAL cost before building. `stage-remote` reads as an operation about
    # the remote machine, so the operator checks the remote's disk and concludes it is safe:
    # on 2026-08-27 the target had 2.2 TB free and the LOCAL root went from 100 GB to 19 GB
    # in three minutes. The build cost is not in the command's contract anywhere, and the
    # two workchains that materialise already say this -- staging simply never did.
    #
    # Whether it materialises at all turns on read-time support for THIS computer:
    # `demote_for_code` demotes a stride/crop to the loader only when the target is local or
    # its label is listed in `AIIDA_PINO_REMOTE_READ_TIME`. A shell without that variable
    # stages the full materialised variant where a shell with it stages a pack index, and
    # nothing in the output distinguishes the two -- so name it here.
    try:
        from aiida_pino.utils.dataset_derivation import report_materialisation_cost

        _profile = (collection.get_dataset_metadata() or {}).get("derivation_profile") or {}
        report_materialisation_cost(lambda msg: print(f"[stage-remote] {msg}", flush=True),
                                    collection, _profile)
    except Exception:  # noqa: BLE001 - an advisory must never block a stage
        pass

    # STREAMING: build, upload and delete member by member, so the LOCAL transient is one
    # member (~370 MB observed) instead of the whole dataset. The bulk path built the
    # entire cache first and uploaded it after; two concurrent stagings of ~40 GB each took
    # root from 100 GB to 19 GB in three minutes on 2026-08-27, and the operator had
    # checked only the TARGET's disk because the local cost was invisible. Nothing here
    # needed decompressing -- stored and staged sizes matched at 1.0x -- so the transient
    # was pure copy, and bounding it costs no transport change: `puttree` per member.
    #
    # Verification is unchanged in strength: sizes and (with `verify_content`) md5s are
    # collected from each member BEFORE its local copy is deleted, and compared against
    # the remote listing/checksums at the end, exactly as the bulk path did.
    import os as _os

    from aiida_pino.calculations.base import (
        runtime_cache_build_lock,
        runtime_cache_materialise_slot,
    )
    from aiida_pino.utils.runtime_cache import runtime_cache_root as _local_root

    if dry_run:
        # Estimate WITHOUT building. The old dry run built the full local cache to count
        # its files, which is the very cost this path exists to avoid paying.
        from aiida_pino.utils.runtime_cache import estimated_derived_bytes

        try:
            members = list(collection.iter_member_simulations())
            estimated = estimated_derived_bytes(
                members, (collection.get_dataset_metadata() or {}).get("derivation_profile") or {}
            )
        except Exception:  # noqa: BLE001
            members, estimated = [], None
        return {
            "cache_name": cache_name,
            "member_count": len(members),
            "estimated_bytes": estimated,
            "staged": False,
            "note": "dry run: nothing built, nothing transferred; peak local use when "
                    "staging for real is about one member at a time",
        }

    root_local = Path(_local_root())
    _sweep_dead_stream_dirs(root_local)

    # The destination check below protects the DESTINATION; the source was unprotected, and
    # on 2026-09-09 a materialised stage took this root from 35 GB free to 22 GB -- under the
    # FLOW-3D free-space floor -- before it was killed. Same gate the CalcJob build uses, so
    # it waits on transient pressure and refuses only what could never fit.
    # ponytail: sized as the whole build; a streamed COMPRESSED build peaks nearer one member
    # x worker concurrency, so this over-refuses there. Size by peak if that ever bites.
    from aiida_pino.utils.runtime_cache import refuse_build_that_cannot_fit

    _members = list(collection.iter_member_simulations())
    _profile = (collection.get_dataset_metadata() or {}).get("derivation_profile") or {}
    refuse_build_that_cannot_fit(root_local, _members, _profile, label=f"stage-remote {cache_name}")

    build_dir = root_local / f"{cache_name}.stream-{_os.getpid()}"
    expected: dict[str, int] = {}
    sums: dict[str, str] = {}

    with computer.get_transport() as transport:
        root = remote_cache_root(transport, remote_root)
        remote_dir = f"{root}/{cache_name}"
        if transport.path_exists(remote_dir):
            # `puttree` onto an existing directory nests rather than replaces.
            transport.rmtree(remote_dir)

        def _ship(rel_path: str) -> None:
            member_root = build_dir / rel_path
            for key, size in expected_cache_contents(member_root).items():
                expected[f"{rel_path}/{key}"] = size
            if verify_content:
                for key, digest in local_checksums(member_root).items():
                    sums[f"{rel_path}/{key}"] = digest
            transport.puttree(str(member_root), f"{remote_dir}/{rel_path}")
            shutil.rmtree(member_root)

        # The slot serialises materialisation across DATASETS (two ~40 GB builds passing
        # one stateless space check is what filled the disk); the per-cache lock shares
        # work between builders of the SAME one.
        with runtime_cache_materialise_slot(root_local), runtime_cache_build_lock(root_local, cache_name):
            if build_dir.exists():
                shutil.rmtree(build_dir)
            # Uncompressed full-resolution staging trades ~722 MB per case for a 21x cheaper
            # read, and is allowed only where the DESTINATION can afford it with 50 GB still
            # spare. Sized with `estimated_arraydir_bytes`, the UNCOMPRESSED
            # estimator: `estimated_derived_bytes` deliberately scales from the stored
            # COMPRESSED size (58x measured) and would under-count this build by that ratio.
            allow_full_res = False
            try:
                from aiida_pino.utils.runtime_cache import (
                    estimated_arraydir_bytes,
                    room_for_uncompressed_stage,
                )

                members = list(collection.iter_member_simulations())
                profile = (collection.get_dataset_metadata() or {}).get("derivation_profile") or {}
                needed = estimated_arraydir_bytes(members, profile)
                free = remote_free_bytes(transport, root)
                allow_full_res = room_for_uncompressed_stage(free=free, needed=needed)
                print(
                    f"[stage] uncompressed full-resolution staging "
                    f"{'ALLOWED' if allow_full_res else 'not allowed'} on {computer.label}: "
                    f"needs ~{(needed or 0) / 1024**3:.1f} GB, "
                    f"{(free or 0) / 1024**3:.1f} GB free, 50 GB must remain spare.",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - unknown room means NOT allowed
                print(f"[stage] could not size the destination ({exc}); "
                      f"uncompressed full-resolution staging not allowed.", flush=True)
            try:
                collection.export_runtime_arraydir(
                    build_dir,
                    runtime_fields=runtime_fields,
                    allow_pack_index=False,
                    on_member_staged=_ship,
                    allow_full_resolution=allow_full_res,
                )
                assert_cache_usable_for_computer(build_dir, transport_type)
                # What remains after the members shipped: the dataset manifest and any
                # top-level sidecars, written by the export epilogue.
                for path in sorted(build_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(build_dir).as_posix()
                    expected[rel] = path.stat().st_size
                    if verify_content:
                        sums.update({rel: local_checksums(path.parent)[path.name]})
                    if "/" in rel:
                        transport.makedirs(
                            f"{remote_dir}/{rel.rsplit('/', 1)[0]}", ignore_existing=True
                        )
                    transport.put(str(path), f"{remote_dir}/{rel}")
            except Exception:
                # The destination was cleared before the copy, so a failure here has
                # already destroyed whatever was there. Retract the record; an honest
                # "not staged" is recoverable where a lying record is not.
                shutil.rmtree(build_dir, ignore_errors=True)
                retract_remote_runtime_cache(computer, cache_name)
                raise
            problems = verify_remote_cache(
                transport, remote_dir, expected,
                expected_checksums=sums if verify_content else None,
            )
            if problems:
                shutil.rmtree(build_dir, ignore_errors=True)
                retract_remote_runtime_cache(computer, cache_name)
                raise RuntimeError(
                    "remote cache failed verification after streaming: "
                    + "; ".join(problems[:5])
                    + ". The copy on the remote is NOT usable and nothing has been recorded for it."
                )
            shutil.rmtree(build_dir, ignore_errors=True)

    moved = {"file_count": len(expected), "total_bytes": sum(expected.values())}

    node = register_remote_runtime_cache(
        computer,
        cache_name=cache_name,
        remote_path=remote_dir,
        verified={"file_count": moved["file_count"], "total_bytes": moved["total_bytes"]},
    )

    # The copy existed to be transferred; the transfer is done and verified, so keeping it
    # is pure leak. A burst of ck300 stagings left 40 GB + 27 GB + 13 GB of these behind and
    # took root to 100%, killing FLOW-3D case mbs40_037 with error 52. The only backstop was
    # a human running `runtime_evict` by hand, which is what let it reach 100%.
    #
    # ONLY when the discriminator recorded `pack_index: False`, which is what makes this
    # name remote-specific: a local job would look up a different name and cannot want this
    # directory. Without that guard the delete is unsafe -- `pack_index` is only recorded
    # when the exporter accepts `allow_pack_index`, so for an exporter without it the remote
    # and local names COINCIDE and this would pull a cache out from under a running job.
    # Streaming never publishes a local copy under the cache name: members are deleted as
    # they ship and the build directory goes with the last of them, so there is nothing to
    # evict here any more -- the leak this cleanup existed for cannot form.

    return {
        "cache_name": cache_name,
        # No local_path: streaming leaves no local copy behind.
        "remote_path": remote_dir,
        "file_count": moved["file_count"],
        "total_bytes": moved["total_bytes"],
        "remote_data_pk": node.pk,
        "computer": computer.label,
        "staged": True,
    }


def remote_free_bytes(transport, path: str) -> int | None:
    """Free bytes on the remote filesystem holding `path`, or None if it cannot be read.

    None is the honest answer for an unreadable destination, and `room_for_uncompressed_stage`
    refuses on None rather than assuming space.
    """
    try:
        retval, stdout, _ = transport.exec_command_wait(
            f"df -PB1 {_quote(path)} | awk 'NR==2 {{print $4}}'"
        )
    except Exception:  # noqa: BLE001
        return None
    if retval != 0:
        return None
    try:
        return int(str(stdout).strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def retract_remote_runtime_cache(computer, cache_name: str) -> bool:
    """Forget a recorded remote cache, because what it named is gone.

    Marks the record superseded rather than deleting the node: provenance keeps
    the fact that a copy once existed there, while `find_remote_runtime_cache`
    stops returning it. Returns whether anything was retracted.
    """
    from aiida_pino.utils.data_nodes import find_remote_runtime_cache

    existing = find_remote_runtime_cache(computer, cache_name)
    if existing is None:
        return False
    existing.base.extras.set("superseded", True)
    existing.base.extras.set("aiida_pino_cache_retracted", "staging failed after the destination was cleared")
    return True


def collections_consumed_by(process_pk: int) -> list[tuple[str, object]]:
    """The dataset collections a past run's CalcJobs actually consumed.

    Not the collections passed as workchain inputs. A training's inputs are
    usually base collections, but the workchain DERIVES what the job reads from
    the derivation profile (stride, time_as_dim, field selection), and that
    derived collection does not exist until submit time — so "stage the dataset
    you passed" stages the wrong thing, and nothing tells you so in advance.

    Read from the stored links rather than re-derived from the profile: a second
    implementation of the derivation rule would drift from the workchain's, and
    the symptom would be staging a plausible cache under the wrong name.
    """
    from aiida import orm

    from aiida_pino.data import PinoDatasetCollectionData

    node = orm.load_node(process_pk)
    labels = ("train_dataset", "val_dataset", "eval_dataset", "dataset")
    found: dict[str, tuple[str, object]] = {}
    for member in [node] + list(getattr(node, "called_descendants", []) or []):
        if not isinstance(member, orm.CalcJobNode):
            continue
        for link in member.base.links.get_incoming().all():
            if link.link_label in labels and isinstance(link.node, PinoDatasetCollectionData):
                found.setdefault(str(link.node.uuid), (link.link_label, link.node))
    return list(found.values())


def staged_caches_on(computer, *, verify: bool = False) -> list[dict[str, Any]]:
    """What is staged on `computer`, from the records this repo wrote.

    Nothing could answer "is this dataset already there?", so the alternatives
    were to ssh in and `ls` a cache root, or to discover the gap as an excepted
    CalcJob at presubmit — which is what happened. The discriminator is shown
    because a copy staged under a different field set is invisible to a job that
    wants another: two caches for one collection is a correct state, not a
    duplicate.

    `verify=True` also asks the box whether each recorded path still exists,
    which is the difference between "we staged it" and "it is there".
    """
    from aiida import orm

    from aiida_pino.data import PinoDatasetCollectionData
    from aiida_pino.utils.data_nodes import RUNTIME_CACHE_KIND

    builder = orm.QueryBuilder()
    builder.append(
        orm.RemoteData,
        filters={
            "extras.aiida_pino_kind": RUNTIME_CACHE_KIND,
            "extras.superseded": {"!==": True},
        },
        project=["*"],
    )
    rows: list[dict[str, Any]] = []
    for node in builder.all(flat=True):
        if not node.computer or node.computer.uuid != computer.uuid:
            continue
        cache_name = str(node.base.extras.get("aiida_pino_cache_name", "") or "")
        uuid = cache_name.split("-")[0:5]
        collection_uuid = "-".join(uuid) if len(uuid) == 5 else cache_name
        discriminator = cache_name[len(collection_uuid) + 1:] or "(none)"
        row = {
            "cache_name": cache_name,
            "remote_path": node.get_remote_path(),
            "remote_data_pk": node.pk,
            "discriminator": discriminator,
            "collection_pk": None,
            "collection_label": None,
        }
        try:
            collection = orm.load_node(collection_uuid)
            if isinstance(collection, PinoDatasetCollectionData):
                row["collection_pk"] = collection.pk
                row["collection_label"] = collection.label or None
        except Exception:  # noqa: BLE001 - an unresolvable uuid is still worth listing
            pass
        verified = node.base.extras.get("aiida_pino_cache_verified", None)
        if isinstance(verified, dict):
            row["file_count"] = verified.get("file_count")
            row["total_bytes"] = verified.get("total_bytes")
        rows.append(row)

    if verify and rows:
        try:
            with computer.get_transport() as transport:
                for row in rows:
                    retval, _, _ = transport.exec_command_wait(
                        f"test -d {_quote(row['remote_path'])}/ && test -f {_quote(row['remote_path'])}/manifest.json"
                    )
                    row["present"] = retval == 0
        except Exception as exc:  # noqa: BLE001 - listing must not fail on an unreachable box
            for row in rows:
                row["present"] = None
            rows.append({"note": f"could not verify on {computer.label}: {type(exc).__name__}"})
    return rows
