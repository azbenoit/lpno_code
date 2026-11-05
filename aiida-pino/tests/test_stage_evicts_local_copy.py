"""Remote staging must delete its local copy once the transfer is verified.

2026-08-25, reported by cleanup. The staging path built a full local copy of each collection
under ~/.cache/aiida-pino/runtime and never removed it: a burst of ck300 stagings left 40 GB
+ 27 GB + 13 GB behind, took root to 100% / 241 MB free, and killed FLOW-3D case mbs40_037
with error 52. The only backstop was a human running `runtime_evict` by hand.

The guard is the point. `pack_index: False` is what makes the cache name remote-specific, so
a local job looks up a DIFFERENT name and cannot want this directory. It is only recorded
when the exporter accepts `allow_pack_index`; without it the remote and local names coincide
and deleting would pull a cache out from under a running job.
"""

from __future__ import annotations



def test_staging_applies_the_guard_on_the_success_path():
    """Updated for streaming staging: the leak this guard existed for cannot form.

    The bulk path built the full local cache and evicted it after a verified transfer;
    this pinned the eviction and its ordering. Staging now uploads member by member and
    deletes each local copy as it ships (`test_stage_remote_streaming` asserts the peak
    is ONE member and that nothing survives), so there is no post-transfer local cache
    to evict and the guard has no object. What remains pinned here: staging must never
    leave a build directory behind on the failure path either.
    """
    from pathlib import Path

    import aiida_pino

    src = (Path(aiida_pino.__file__).parent / "utils" / "stage_remote.py").read_text()
    fn = src[src.index("def stage_collection_to_computer"):]
    assert "shutil.rmtree(member_root)" in fn, "members must be deleted as they ship"
    assert fn.count("shutil.rmtree(build_dir, ignore_errors=True)") >= 3, (
        "the build dir must go on success AND on both failure paths"
    )
