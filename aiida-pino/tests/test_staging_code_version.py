"""A staged runtime cache must not outlive the staging code that built it.

2026-08-23. The cache-hit test was `manifest.json exists`, nothing more. After the
`derived_fields` staging fix landed (0f2a2ee) and the daemon was restarted, a resubmission
resolved onto a collection whose cache had been built 52 minutes EARLIER -- 17 arrays, no
`conduction_field` -- and reused it verbatim. The arms would have died on the exact
KeyError the fix had just removed, and it would have read as "the fix did not work". It
was caught by hand; there was no supported way to invalidate a cache short of `rm -rf`,
and nothing anywhere to suggest you needed to.

`format` could not have caught it: it names the LAYOUT (`aiida_arraydir_v2`) and does not
change when the staging code does. It is also written in eight places and, before this,
compared in none.
"""

from __future__ import annotations

import pytest
import json

from aiida_pino.utils.runtime_cache import (
    STAGING_CODE_FILES,
    staged_cache_is_current,
    staging_code_version,
)


def _cache(tmp_path, **manifest):
    d = tmp_path / "cache"
    d.mkdir(exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


@pytest.mark.requires_lpno
def test_the_version_is_stable_and_short():
    assert staging_code_version() == staging_code_version()
    assert len(staging_code_version()) == 12


def test_every_named_staging_file_exists():
    """A typo'd path silently degrades the digest to "unknown" and disables the check."""
    from pathlib import Path

    import aiida_pino

    root = Path(aiida_pino.__file__).resolve().parent
    missing = [name for name in STAGING_CODE_FILES if not (root / name).is_file()]
    assert not missing, f"STAGING_CODE_FILES names files that do not exist: {missing}"


def test_a_cache_staged_by_different_code_is_refused(tmp_path):
    """The reported case."""
    usable, reason = staged_cache_is_current(_cache(tmp_path, staging_code_version="0" * 12))
    assert usable is False
    assert "different staging code" in reason


def test_a_cache_staged_by_this_code_is_kept(tmp_path):
    usable, _ = staged_cache_is_current(_cache(tmp_path, staging_code_version=staging_code_version()))
    assert usable is True


def test_a_cache_with_no_stamp_is_grandfathered(tmp_path):
    """Deliberate, and the alternative is worse.

    A cache predating the stamp cannot be judged: the stamp cannot distinguish a stale one
    from a fine one. Refusing them all would rebuild every cache on the box at once, and a
    rebuild is 3.3 GB per simulation -- during a live campaign that is a worse failure than
    the one being prevented. Everything staged from now on carries the stamp.
    """
    usable, reason = staged_cache_is_current(_cache(tmp_path, format="aiida_arraydir_v2"))
    assert usable is True
    assert "before the stamp existed" in reason


def test_a_missing_or_unreadable_manifest_is_not_a_cache(tmp_path):
    assert staged_cache_is_current(tmp_path / "nope")[0] is False
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    assert staged_cache_is_current(broken)[0] is False


@pytest.mark.requires_lpno
def test_touching_a_staging_file_changes_the_version(tmp_path, monkeypatch):
    """The mechanism must actually respond to a staging-code change, not just to itself."""
    import aiida_pino.utils.runtime_cache as rc

    before = rc.staging_code_version()
    monkeypatch.setattr(rc, "STAGING_CODE_FILES", STAGING_CODE_FILES[:1])
    assert rc.staging_code_version() != before, "a different file set must hash differently"


def test_a_really_staged_manifest_carries_the_stamp(aiida_profile_clean, pino_dataset_factory, tmp_path):
    """The pipeline, not the helper.

    Every other test here exercises `staged_cache_is_current` against a manifest the test
    wrote itself. That proves the comparison and nothing about whether the exporter
    actually records the field -- which is the half that has to work for any of it to
    matter, and the half that was missing for `format` (written in eight places, compared
    in none).
    """
    from aiida_pino.utils.runtime_cache import staging_code_version

    dataset = pino_dataset_factory("stamped")
    # Returns the manifest FILE, not the directory.
    staged = dataset.export_runtime_arraydir(tmp_path / "staged")

    manifest = json.loads(staged.read_text(encoding="utf-8"))
    assert manifest.get("staging_code_version") == staging_code_version()
    usable, reason = staged_cache_is_current(staged.parent)
    assert usable is True and reason == "current"


def test_a_stale_cache_is_sidestepped_and_never_deleted(aiida_profile_clean, pino_dataset_factory, tmp_path, monkeypatch):
    """A stale cache must be left on disk, because a running job may be reading it.

    Deleting it is the failure `_dataset_runtime_cache`'s own docstring warns about --
    trainings killed at step 1500 and 18000 with a `FileNotFoundError` from a DataLoader
    worker, because neither the repair path nor the store's retry helps a job reading a
    directory that is being removed underneath it. And a staging fix landing WHILE jobs
    run is exactly when this branch fires: that is the same situation that produced the
    bug (0f2a2ee landed at 20:24 with arms mid-flight). So the fresh build goes to a
    sibling name instead.
    """
    import aiida_pino.utils.runtime_cache as rc
    from aiida_pino.utils.runtime_cache import runtime_cache_root

    monkeypatch.setenv("AIIDA_PINO_RUNTIME_CACHE", str(tmp_path / "cache"))
    root = runtime_cache_root()
    dataset = pino_dataset_factory("sidestep")

    stale_dir = root / dataset.uuid
    stale_dir.mkdir(parents=True)
    (stale_dir / "manifest.json").write_text(
        json.dumps({"staging_code_version": "0" * 12, "marker": "do-not-delete"}), encoding="utf-8"
    )
    canary = stale_dir / "in_use_by_a_running_job.npy"
    canary.write_bytes(b"x")

    from aiida_pino.calculations.base import warm_dataset_runtime_cache

    fresh = warm_dataset_runtime_cache(dataset)

    assert canary.is_file(), "the stale cache was deleted under a job that may be reading it"
    assert fresh.resolve() != stale_dir.resolve(), "the stale cache must not be reused"
    assert staged_cache_is_current(fresh)[0] is True


def test_a_sidestepped_cache_name_is_still_repairable():
    """`-s<version>` must not make the cache unparseable, or repair silently declines.

    `repair_dataset_dir` rebuilds a cache that went missing under a running job -- it
    exists because five workchains died on `BadZipFile: Bad CRC-32` reading a cache
    nothing rebuilt. It gives up when `cache_identity` returns None, which is exactly what
    the suffix caused before the pattern was extended. The staging version is not part of
    the identity: same node, same recipe, newer code.
    """
    from aiida_pino.utils.runtime_repair import cache_identity

    uuid = "0e6f6b2a-1111-2222-3333-444455556666"
    assert cache_identity(f"{uuid}-s0123456789ab") == (uuid, None)
    assert cache_identity(f"{uuid}-abcdef123456-s0123456789ab") == (uuid, "abcdef123456")
    # unchanged for the names that already existed
    assert cache_identity(uuid) == (uuid, None)
    assert cache_identity(f"{uuid}-abcdef123456") == (uuid, "abcdef123456")
    assert cache_identity("not-a-cache") is None


def test_the_digest_covers_imported_loader_sources_too():
    """A staged steady runtime bakes in code from lpno, so the stamp must see it.

    `moving_frame_slices` is computed by `_beam_following_slices_for` ->
    `lpno.models.pipeline.frame_transfer` -> `npz_datasource._beam_path_over_time`,
    and the trajectory channels come from the same function. A fix THERE changes staged
    output while every cache stays stamped "current" -- the failure `STAGING_CODE_FILES`
    already warns about, one package over. Reported 2026-09-02 after `0b04707` fixed the
    beam-overrun clamp: ten cached runtimes survived it looking valid, with the moving-frame
    window 5-12 cells off.
    """
    import importlib.util
    import shutil
    from pathlib import Path

    from aiida_pino.utils.runtime_cache import (
        IMPORTED_STAGING_CODE_MODULES,
        staging_code_version,
    )

    assert "lpno.models.pipeline.frame_transfer" in IMPORTED_STAGING_CODE_MODULES

    spec = importlib.util.find_spec("lpno.models.pipeline.frame_transfer")
    if spec is None or not spec.origin:  # pragma: no cover - package absent
        import pytest

        pytest.skip("lpno not importable here")

    target = Path(spec.origin)
    before = staging_code_version()
    backup = target.with_suffix(".py.testprobe")
    shutil.copy2(target, backup)
    try:
        # A statement, not a comment: the digest hashes the syntax tree (2026-09-14), so
        # only a change that can reach staged bytes moves it.
        target.write_bytes(target.read_bytes() + b"\n_staging_probe = 1\n")
        assert staging_code_version() != before, (
            "editing an imported staging source did not change the digest, so a loader "
            "fix would leave every cache stamped current"
        )
    finally:
        shutil.move(str(backup), str(target))
    assert staging_code_version() == before, "the digest must be restored exactly"


def test_a_missing_imported_module_does_not_invalidate_every_cache():
    """Absent means the staged output cannot depend on it here.

    Returning "unknown" would mark every cache on such a box stale at once -- a worse
    failure than the one being prevented, and the same reasoning that makes an unstamped
    cache acceptable.
    """
    import aiida_pino.utils.runtime_cache as rc

    original = rc.IMPORTED_STAGING_CODE_MODULES
    rc.IMPORTED_STAGING_CODE_MODULES = ("definitely_not_a_real_module_xyz",)
    try:
        assert rc.staging_code_version() != "unknown"
    finally:
        rc.IMPORTED_STAGING_CODE_MODULES = original
