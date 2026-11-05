"""Where a dataset's runtime cache lives, and what it is called.

`stage_dataset_input` builds this cache locally; `aiida-pino stage-remote`
copies one to a compute box. Staging then looks the remote copy up **by name**
at `prepare_for_submission`. Two implementations of that name would not fail
loudly when they drifted -- they would produce a job that finds nothing staged
and a training that dies at step 0 -- so the name is computed here, once, and
both paths call it.

The local name is load-bearing history: a cache with no discriminator is the
bare uuid, which is what every entry already on disk is called.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

#: Transports whose jobs read this machine's own filesystem.
from aiida_pino.data.pino_dataset_collection import pack_index_allowed_for_transport


def runtime_cache_root() -> Path:
    """Root of the local runtime cache (`AIIDA_PINO_RUNTIME_CACHE` overrides)."""
    return Path(os.environ.get("AIIDA_PINO_RUNTIME_CACHE") or (Path.home() / ".cache" / "aiida-pino" / "runtime"))


#: Source files whose contents determine the BYTES a staged runtime cache holds. Hashed
#: into every manifest as `staging_code_version`, so a cache built by older staging code
#: is detected instead of silently reused.
#:
#: Deliberately a short explicit list rather than the whole package: hashing everything
#: would invalidate every cache on any commit, and a full rebuild is 3.3 GB per
#: simulation. A file missing from this list is the failure mode this trades for -- its
#: change goes undetected, exactly as before -- so add to it when a change alters staged
#: output.
STAGING_CODE_FILES = (
    "data/pino_dataset_collection.py",
    "utils/runtime_export.py",
    "utils/flow3d_npz.py",
)

#: The same rule, for code this package IMPORTS. A staged steady runtime bakes in
#: `moving_frame_slices` computed by `_beam_following_slices_for` ->
#: `lpno.models.pipeline.frame_transfer` -> `npz_datasource._beam_path_over_time`,
#: and the trajectory channels come from the same function. So a fix THERE changes staged
#: output while every cache stays stamped "current" -- which is the failure the list above
#: already warns about, one package over.
#:
#: Found 2026-09-02 after an lpno fix to the beam-overrun clamp: the loader had frozen the beam at the declared path end for
#: `turn_off_at_end: false`, putting the moving-frame window 5-12 cells off for anchorbox
#: steady targets. Ten cached runtimes stayed valid-looking across that fix.
IMPORTED_STAGING_CODE_MODULES = (
    "lpno.models.pipeline.frame_transfer",
    "lpno.custom_utils.npz_datasource",
)

#: Stamps written by earlier staging code whose staged BYTES are identical to the current
#: code's -- accepted as current instead of forcing a rebuild. Added when the hash changes
#: for a reason that cannot alter output: the switch to the comment-insensitive digest
#: below, and the sizing/pool code leaving `pino_dataset_collection.py`. Both stamps below
#: were written on 2026-09-14 by code whose derivation functions are byte-identical to this
#: commit's (44f6338 and 5ad19d7 changed only worker sizing). Prune when no cache carries them.
#: `d5b3f964fdc4` is what a warmer STARTED under the byte digest stamps when it finishes
#: after this commit: the stamp is computed at manifest-write time from the files on disk,
#: and those files are exactly this commit's -- so that is a third spelling of the same bytes.
#: `bdad67243746`: the tree digest before `trimmed_fields` -- an export with no
#: `runtime_fields` still inflates every field, so its caches are byte-identical to that one.
#: `42a6e1e919cf`: the tree digest with the parent-side trim; an export with no
#: `runtime_fields` inflated every field then as now.
EQUIVALENT_STAGING_CODE_VERSIONS = frozenset({"7eecbd06157c", "6f31a4505743", "d5b3f964fdc4", "bdad67243746", "42a6e1e919cf"})


def _staging_source_digest_bytes(source: bytes) -> bytes:
    """What the digest sees of one source file: its syntax tree, not its text.

    Comments, blank lines, formatting and docstrings do not reach the staged arrays, and
    hashing them cost three full rebuilds on 2026-09-14 (a sizing constant, then a comment,
    then a docstring). Two files that parse to the same tree stamp the same version. A
    file that does not parse is hashed verbatim, so a broken checkout still stamps
    something rather than nothing.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            del body[0]  # a docstring
    return ast.dump(tree, include_attributes=False).encode()


def staging_code_version() -> str:
    """12-hex digest of the staging sources, or `"unknown"` if they cannot be read.

    2026-08-23: the cache-hit test was `manifest.json exists`, nothing more. After the
    `derived_fields` staging fix landed, a collection whose cache had been built 52
    minutes earlier was reused verbatim -- 17 arrays, no conduction_field -- and the arms
    consuming it would have died on the same KeyError the fix had just removed, reading as
    "the fix did not work". Caught by hand; there was no supported way to invalidate a
    cache short of `rm -rf`, and nothing to suggest you needed to.

    `"unknown"` never matches a real digest, but a MISSING stamp is grandfathered by the
    comparison rather than treated as stale -- see `staged_cache_is_current`.
    """
    import hashlib

    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha1()
    for relative in STAGING_CODE_FILES:
        path = root / relative
        try:
            digest.update(_staging_source_digest_bytes(path.read_bytes()))
        except OSError:
            return "unknown"
    # Imported sources are located through the module system rather than by a relative
    # path: `lpno` is an editable install from a sibling checkout, so its
    # location is not derivable from this file and hard-coding one would silently stop
    # digesting the day the checkout moves.
    import importlib.util

    for module in IMPORTED_STAGING_CODE_MODULES:
        try:
            spec = importlib.util.find_spec(module)
            origin = getattr(spec, "origin", None) if spec else None
            if not origin:
                # Absent means the staged output cannot depend on it in this environment;
                # "unknown" would invalidate every cache on a box that simply has no
                # lpno, which is a worse failure than the one being prevented.
                continue
            digest.update(_staging_source_digest_bytes(Path(origin).read_bytes()))
        except (ImportError, ValueError, OSError):
            return "unknown"
    return digest.hexdigest()[:12]


def sidestep_cache_name(root: Path, cache_name: str) -> str:
    """The `<name>-s<stamp>` sibling a stale cache is sidestepped to.

    Prefers a sibling that already exists and is usable under an EQUIVALENT stamp: the
    directory name carries the stamp that was current when its build started, so without
    this a cache finished under `6f31a4505743` sat unfound beside a rebuild named for the
    stamp of the day (2026-09-14). Falls back to the current stamp's name.
    """
    current = staging_code_version()
    for stamp in (current, *sorted(EQUIVALENT_STAGING_CODE_VERSIONS)):
        candidate = root / f"{cache_name}-s{stamp}"
        if candidate.is_dir() and staged_cache_is_current(candidate)[0]:
            return candidate.name
    return f"{cache_name}-s{current}"


def staged_cache_is_current(cache_dir: Path) -> tuple[bool, str]:
    """(usable, reason) for an existing staged cache directory.

    A cache with NO `staging_code_version` is accepted. It predates this mechanism, and
    refusing it would force a rebuild of every cache on the machine at once -- during a live
    campaign that is a worse failure than the one being prevented, and the stamp cannot
    tell a stale pre-existing cache from a fine one. Everything staged from now on carries
    the stamp and is checked.
    """
    import json

    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        return False, "no manifest"
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8")).get("staging_code_version")
    except (OSError, ValueError):
        return False, "unreadable manifest"
    if not recorded:
        return True, "no staging_code_version recorded (built before the stamp existed)"
    current = staging_code_version()
    if recorded == current:
        return True, "current"
    if recorded in EQUIVALENT_STAGING_CODE_VERSIONS:
        return True, f"staged by equivalent staging code ({recorded}, same bytes as {current})"
    return False, f"staged by different staging code ({recorded} != {current})"


#: Leave this much free after a build rather than filling the disk exactly. Root hitting
#: 100% does not fail cleanly here: on 2026-08-25 a build that could never fit took root to
#: 1.3 GB, truncated `u_network.best.pth` of an unrelated training to 0 bytes at step 13300,
#: hung a transport upload, and left four arms in `created` while the daemon reported 4%
#: worker use -- so it read as worker saturation rather than as a full disk.
RUNTIME_BUILD_FREE_MARGIN_BYTES = 20 * 1024**3

#: Free space a destination must still have AFTER an uncompressed full-resolution stage
#: Uncompressed staging trades disk for read speed -- ~722 MB per case
#: against 0.076 s to read versus 1.58 s compressed -- and the trade is only sane where the
#: disk can afford it. 50 GB admits a compute machine with a large free disk and refuses a nearly full
#: one, where a full root stalls FLOW-3D solves, without a per-machine list.
MATERIALIZE_SPARE_BYTES = 50 * 1024**3


def room_for_uncompressed_stage(*, free, needed) -> bool:
    """Whether a destination can hold an uncompressed stage and still keep the spare.

    Unknown on either side is REFUSED rather than assumed free: a size this gate cannot
    compute is exactly the build it must not wave through.
    """
    if not isinstance(free, (int, float)) or not isinstance(needed, (int, float)):
        return False
    if free <= 0 or needed < 0:
        return False
    return (free - needed) >= MATERIALIZE_SPARE_BYTES

#: How long to wait for room before giving up, and how often to look. Waiting is the point:
#: on 2026-08-27 two datasets staged CONCURRENTLY wrote 82 GB in three minutes (40 GB and
#: 42 GB) and took root from 100 GB to 19 GB. Each build passed the check on its own -- the
#: check is per-build and stateless, so both read the same 100 GB before either wrote a byte.
RUNTIME_BUILD_WAIT_S = int(os.environ.get("AIIDA_PINO_BUILD_WAIT_S", 30 * 60))
RUNTIME_BUILD_POLL_S = 20


def _derived_shape(shape: list, profile: dict) -> list:
    """Shape after the transforms `runtime_export` actually applies, in its order.

    Crop FIRST (it restricts the domain), then frame selection, then stride, then
    super-resolution -- matching `runtime_export.py:141-171`. Only canonical 5-D
    `(t, c, x, y, z)` arrays are reduced; coordinate vectors and scalars are left alone
    because they are negligible against the field arrays and their axis meaning is not
    positional.

    Modelling this is not an optimisation. Estimating from RAW shapes over-estimated a
    `stride: [2,2,1]` collection by 4x and refused an 8-member mb40 eval at "183 GB" that
    would really have written ~46 GB -- 4 GB short of a limit it was never near. An
    over-estimate is not the safe direction when it blocks real work; it just fails in a
    way that looks principled.
    """
    if len(shape) != 5:
        return shape
    t, c, x, y, z = (int(d) for d in shape)

    crop = int(profile.get("crop_z_top", 0) or 0)
    if crop > 0:
        z = max(z - crop, 0)

    if profile.get("moving_ref_frame"):
        # The beam-following route, in `apply_derivation`'s own order: super-resolution,
        # then the last `moving_frame_steps` frames cut to a window (steps - 1) * super_res
        # cells narrower than the domain, averaged to ONE frame when `time_avg_window` > 1,
        # then the stride. Ignoring it estimated a 295-member steady training set at
        # ~1585 GB (every raw frame kept) against ~4 GB written, and the size gate trusts
        # this figure (2026-10-07, reproduction test from the deposit).
        try:
            sr = max(int(profile.get("super_res_factor", 1) or 1), 1)
            steps = max(int(profile.get("moving_frame_steps", 1) or 1), 1)
            window = int(profile.get("time_avg_window", 1) or 1)
            sx, sy, sz = (max(int(v), 1) for v in (profile.get("stride") or [1, 1, 1]))
        except (TypeError, ValueError):
            sr, steps, window, sx, sy, sz = 1, 1, 1, 1, 1, 1
        x, y, z = x * sr, y * sr, z * sr
        x = max(x - (steps - 1) * sr, 1)
        t = 1 if window > 1 else min(t, steps)
        return [t, c, -(-x // sx), -(-y // sy), -(-z // sz)]

    frames = profile.get("time_frames")
    if frames:
        try:
            t = min(t, int(frames))
        except (TypeError, ValueError):
            pass

    stride = profile.get("stride") or [1, 1, 1]
    try:
        sx, sy, sz = (max(int(v), 1) for v in stride)
    except (TypeError, ValueError):
        sx = sy = sz = 1
    # ceil: a stride of 2 over 5 cells keeps 3, not 2.
    x, y, z = -(-x // sx), -(-y // sy), -(-z // sz)

    try:
        super_res = int(profile.get("super_res_factor", 1) or 1)
    except (TypeError, ValueError):
        super_res = 1
    if super_res > 1:
        x, y, z = x * super_res, y * super_res, z * super_res

    return [t, c, x, y, z]


def _stored_compressed_bytes(members) -> int:
    """Total COMPRESSED size of these members as they sit in the object store."""
    from aiida_pino.utils.runtime_export import payload_locator

    total = 0
    for member in members:
        try:
            locator = payload_locator(member)
        except Exception:  # noqa: BLE001 - a member without a locator contributes nothing
            continue
        if locator and locator.get("size"):
            total += int(locator["size"])
    return total


def estimated_derived_bytes(members, profile: dict | None = None) -> int:
    """What a derived build will really write, now that the writer COMPRESSES.

    The uncompressed element count over-estimates by the compression ratio -- measured 58x
    on the gascrop val split once `derived_member_worker` moved to `savez_compressed`
    (0.05 GB per member against 2.9 GB before). Left uncorrected the gate would refuse
    exactly the builds compression makes possible, which is the failure this same gate
    already had once today in its 4x form.

    So estimate from the STORED compressed size, scaled by how much of the data the
    derivation keeps. Both sides are compressed, so the ratio carries across; material
    channels make the real output smaller still (twelve `np.full` arrays compress to
    almost nothing), which is the safe direction.

    Falls back to the uncompressed count when no member has a locator -- an unknown size
    must not read as zero.
    """
    import numpy as np

    # `materialize: true` exists to write the store UNCOMPRESSED (the encoding is the whole
    # point, see the 2026-09-09 entry), so the compressed-scaled figure below is wrong by the
    # compression ratio for exactly that build. On 2026-09-09 an operator was told "about
    # 1.7 GB will be written" and watched the build directory pass 62 GB.
    if (profile or {}).get("materialize"):
        return estimated_arraydir_bytes(members, profile)

    stored = _stored_compressed_bytes(members)
    if not stored:
        return estimated_arraydir_bytes(members, profile)

    raw = 0
    derived = 0
    for member in members:
        try:
            shapes = (member.get_simulation_metadata() or {}).get("array_shapes") or {}
        except Exception:  # noqa: BLE001
            continue
        for shape in shapes.values():
            if not shape:
                continue
            try:
                dims = [int(d) for d in shape]
            except (TypeError, ValueError):
                continue
            raw += int(np.prod(dims))
            derived += int(np.prod(_derived_shape(dims, profile or {})))
    if raw <= 0:
        return stored
    return int(stored * (derived / raw))


def estimated_arraydir_bytes(members, profile: dict | None = None) -> int:
    """Bytes materialising these members will write, from stored metadata alone.

    Reads `array_shapes` out of each member's metadata -- no payload is touched, so this
    costs milliseconds for a build that would take hours -- and applies the derivation's
    own transforms so the figure tracks what the exporter will really write.
    """
    import numpy as np

    profile = profile or {}
    total = 0
    for member in members:
        try:
            shapes = (member.get_simulation_metadata() or {}).get("array_shapes") or {}
        except Exception:  # noqa: BLE001 - a member without metadata contributes nothing
            continue
        for shape in shapes.values():
            if not shape:
                continue
            try:
                dims = _derived_shape([int(d) for d in shape], profile)
                total += int(np.prod(dims)) * 4
            except (TypeError, ValueError):
                continue
    return total


def refuse_build_that_cannot_fit(root, members, profile: dict | None = None, *, label: str = "") -> None:
    """Raise before building a runtime cache that the filesystem cannot hold.

    Nothing gated this before: a derivation whose materialised form needed several hundred
    GB more than the disk had simply ran until the filesystem was full, taking unrelated
    jobs down with it. A build that CANNOT fit is refused rather than made to wait, which
    is the opposite of the FLOW-3D free-space floor and deliberately so -- that floor waits
    because pressure there is transient (another case is about to drop its flsgrf), whereas
    this is structural: no amount of waiting makes 259 GB fit in 174 GB.
    """
    import shutil
    import time

    needed = estimated_derived_bytes(members, profile)
    if not needed:
        return
    try:
        usage = shutil.disk_usage(str(root))
    except OSError:
        return
    free, total = usage.free, usage.total
    want = needed + RUNTIME_BUILD_FREE_MARGIN_BYTES
    if want <= free:
        return
    from aiida_pino.utils.guards import resource_guard, resource_guard_mode

    if resource_guard_mode() == "warn":
        # A warning, then the build goes ahead: no wait either, the operator was told.
        resource_guard(
            f"materialising {label or 'this dataset'} needs about {needed / 1e9:.0f} GB "
            f"({len(list(members))} members) plus a {RUNTIME_BUILD_FREE_MARGIN_BYTES / 1e9:.0f} GB "
            f"margin and {free / 1e9:.0f} GB is free; the build may fill the disk."
        )
        return

    # WAIT when the disk could hold this once something else finishes, REFUSE only when it
    # never could. The distinction is the one the FLOW-3D free-space floor is built on: a
    # refusing gate turns transient pressure into permanently dead work, and there it killed
    # a 300-case sweep in five minutes. Structural impossibility is different -- no amount of
    # waiting makes 259 GB fit in 174 GB -- and that still raises immediately.
    if want <= total:
        deadline = time.monotonic() + RUNTIME_BUILD_WAIT_S
        announced = False
        while time.monotonic() < deadline:
            try:
                free = shutil.disk_usage(str(root)).free
            except OSError:
                return
            if want <= free:
                return
            if not announced:
                print(
                    f"[runtime-cache] waiting for room to materialise {label or 'this dataset'}: "
                    f"needs about {needed / 1e9:.0f} GB plus a "
                    f"{RUNTIME_BUILD_FREE_MARGIN_BYTES / 1e9:.0f} GB margin, "
                    f"{free / 1e9:.0f} GB free. Waiting up to {RUNTIME_BUILD_WAIT_S / 60:.0f} min "
                    f"(F3D-style: pressure here is usually another build about to finish).",
                    flush=True,
                )
                announced = True
            # Never sleep past the deadline: a 20 s poll on a 1 s budget turned a fast
            # refusal into a 20 s one, and made the tests that exercise refusal slow.
            time.sleep(max(0.0, min(RUNTIME_BUILD_POLL_S, deadline - time.monotonic())))
    resource_guard(
        f"refusing to materialise {label or 'this dataset'}: it needs about "
        f"{needed / 1e9:.0f} GB uncompressed ({len(list(members))} members) and only "
        f"{free / 1e9:.0f} GB is free, keeping a {RUNTIME_BUILD_FREE_MARGIN_BYTES / 1e9:.0f} GB "
        f"margin. The stored form is compressed, so a materialised variant is far larger than "
        f"the archive: 31x measured on one collection (13.3 GB stored, 412.6 GB). Free space, "
        f"use a smaller "
        f"split, or stage to a machine with room."
        + (
            f" Waited {RUNTIME_BUILD_WAIT_S / 60:.0f} min for room and it did not come."
            if needed + RUNTIME_BUILD_FREE_MARGIN_BYTES <= total
            else " The disk could not hold this even empty, so waiting cannot help."
        )
    )


def runtime_cache_name(node_uuid: str, discriminator: dict[str, Any] | None) -> str:
    """`<uuid>`, or `<uuid>-<12 hex>` when the export was parameterised.

    Empty and `None` discriminators must both give the bare uuid: that is the
    name every pre-existing cache entry carries.
    """
    if not discriminator:
        return str(node_uuid)
    encoded = json.dumps(discriminator, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{node_uuid}-{hashlib.sha1(encoded).hexdigest()[:12]}"


def cache_discriminator_for(node, *, runtime_fields, transport_type: str | None) -> dict[str, Any]:
    """What distinguishes this export from the default one.

    Mirrors what is actually passed to the exporter, which is why it inspects
    the signature: if `export_runtime_arraydir` cannot take a flag, the flag is
    not passed and must not be recorded, or the name would claim a distinction
    the bytes do not have.

    A non-local transport always lands `pack_index: False` here. That is the
    point of the discriminator -- a locally-built cache holds absolute pack
    paths into this machine's object store, so a remote job handed the same
    cache name would stage entries that resolve to nothing on its own disk.
    """
    signature = inspect.signature(node.export_runtime_arraydir).parameters
    discriminator: dict[str, Any] = {}
    if runtime_fields and "runtime_fields" in signature:
        discriminator["runtime_fields"] = list(runtime_fields)
    if not pack_index_allowed_for_transport(transport_type) and "allow_pack_index" in signature:
        discriminator["pack_index"] = False
    return discriminator


def export_kwargs_for(node, *, runtime_fields, transport_type: str | None) -> dict[str, Any]:
    """The exporter kwargs matching `cache_discriminator_for`, built together.

    Kept beside the discriminator deliberately: the two must describe the same
    export, and splitting them across modules is how they drift.
    """
    signature = inspect.signature(node.export_runtime_arraydir).parameters
    kwargs: dict[str, Any] = {}
    if runtime_fields and "runtime_fields" in signature:
        kwargs["runtime_fields"] = list(runtime_fields)
    if not pack_index_allowed_for_transport(transport_type) and "allow_pack_index" in signature:
        kwargs["allow_pack_index"] = False
    return kwargs
