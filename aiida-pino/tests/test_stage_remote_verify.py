"""Verifying a transferred runtime cache, against a transport that tells you nothing.

Measured against the first real compute box (2026-08-19), `puttree`:

* returns `None` — no byte count, no file list, no success signal beyond "did
  not raise";
* creates a missing destination parent SILENTLY, so a mis-derived destination
  builds a fresh tree somewhere wrong;
* has `cp -r` semantics on an existing destination: a re-put lands the source
  *inside* it under its own basename, so a retry after a partial failure leaves
  a NESTED duplicate.

The third is why this module compares an exact set of relative paths and sizes
and treats any EXTRA path as a failure. A nested duplicate has *more* files and
*more* bytes than expected, so every "count matches" or "total size >= expected"
check passes on precisely the corruption most likely to occur.
"""

from __future__ import annotations

from pathlib import Path

from aiida_pino.utils.stage_remote import (
    compare_cache_contents,
    compare_cache_contents,
    expected_cache_contents,
    parse_remote_listing,
)


def _make_cache(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text('{"format": 1}')
    (root / "members").mkdir()
    (root / "members" / "a.npz").write_bytes(b"x" * 100)
    (root / "members" / "b.npz").write_bytes(b"y" * 250)
    return root


def test_expected_contents_are_relative_paths_and_sizes(tmp_path):
    contents = expected_cache_contents(_make_cache(tmp_path / "c"))
    assert contents == {
        "manifest.json": 13,
        "members/a.npz": 100,
        "members/b.npz": 250,
    }


def test_an_identical_copy_has_no_problems(tmp_path):
    expected = expected_cache_contents(_make_cache(tmp_path / "c"))
    assert compare_cache_contents(expected, dict(expected)) == []


def test_a_missing_file_is_reported_by_name(tmp_path):
    expected = expected_cache_contents(_make_cache(tmp_path / "c"))
    actual = dict(expected)
    del actual["members/b.npz"]
    problems = compare_cache_contents(expected, actual)
    assert any("members/b.npz" in p and "missing" in p.lower() for p in problems)


def test_a_truncated_file_is_reported_with_both_sizes(tmp_path):
    expected = expected_cache_contents(_make_cache(tmp_path / "c"))
    actual = dict(expected)
    actual["members/a.npz"] = 40
    problems = compare_cache_contents(expected, actual)
    assert any("members/a.npz" in p and "100" in p and "40" in p for p in problems)


def test_a_nested_duplicate_tree_fails_even_though_it_has_MORE_bytes(tmp_path):
    """The measured re-put failure. A size or count check would pass this."""
    expected = expected_cache_contents(_make_cache(tmp_path / "c"))
    actual = dict(expected)
    for name, size in list(expected.items()):
        actual[f"reput_049umvo4/{name}"] = size

    assert sum(actual.values()) > sum(expected.values()), "the corruption is BIGGER, not smaller"
    assert len(actual) > len(expected)

    problems = compare_cache_contents(expected, actual)
    assert problems, "extra paths must fail; this is the case a count check misses"
    assert any("unexpected" in p.lower() or "extra" in p.lower() for p in problems)
    assert any("reput_049umvo4" in p for p in problems)


def test_an_empty_remote_directory_fails_rather_than_passing_vacuously(tmp_path):
    expected = expected_cache_contents(_make_cache(tmp_path / "c"))
    problems = compare_cache_contents(expected, {})
    assert len(problems) >= 3


def test_the_remote_listing_parser_handles_paths_with_spaces_and_tabs():
    """Size first, so the path keeps its spaces; tabs are not separators either."""
    listing = "13 manifest.json\x00100 members/a b.npz\x00250 members/b\tc.npz\x00"
    assert parse_remote_listing(listing) == {
        "manifest.json": 13,
        "members/a b.npz": 100,
        "members/b\tc.npz": 250,
    }


def test_a_newline_in_a_filename_cannot_forge_a_record_for_an_absent_file():
    """Demonstrated on a real box, and it defeats presence AND size checks.

    A file literally named `decoy\n11 manifest.json` makes a NEWLINE-delimited
    listing emit `11 manifest.json` as its own line, so an absent manifest.json
    reads as present at the right size. NUL-delimiting removes the possibility:
    the odd name stays one record, and surfaces as an unexpected extra.
    """
    forged_name = "decoy\n11 manifest.json"
    listing = f"2 real.npz\x001 {forged_name}\x00"
    parsed = parse_remote_listing(listing)

    assert parsed == {"real.npz": 2, forged_name: 1}
    assert "manifest.json" not in parsed, "the forged record must not become a file"

    expected = {"real.npz": 2, "manifest.json": 11}
    problems = compare_cache_contents(expected, parsed)
    assert any("manifest.json" in p and "missing" in p.lower() for p in problems)
    assert any("unexpected extra path" in p for p in problems)


def test_the_parser_ignores_blank_and_malformed_records():
    assert parse_remote_listing("\x00\x0042 ok.bin\x00garbage\x00") == {"ok.bin": 42}


def test_the_parser_returns_nothing_for_an_empty_listing():
    assert parse_remote_listing("") == {}


class _FakeTransport:
    """A transport with the measured semantics of the real one.

    `puttree` nests on an existing destination, `rmtree` clears it, and nothing
    reports anything back. `corrupt` mutates the landed tree so a test can stage
    a specific failure.
    """

    def __init__(self, corrupt=None):
        self.tree: dict[str, int] = {}
        # md5 per landed path, so the fake can answer the content check too.
        self.digests: dict[str, str] = {}
        self.calls: list[str] = []
        self._corrupt = corrupt
        # Whether the DESTINATION DIRECTORY exists, which is what `cp -r`
        # semantics actually branch on. Modelling "the tree has content"
        # instead hid a real bug: `makedirs` recreated an empty destination
        # right after `rmtree`, so the real transport nested every time while
        # this stub did not (caught on a real box, 2026-08-19).
        self.destination_exists = False

    def rmtree(self, path):
        self.calls.append(f"rmtree:{path}")
        self.tree.clear()
        self.digests.clear()
        self.destination_exists = False

    def makedirs(self, path, ignore_existing=False):
        self.calls.append(f"makedirs:{path}")
        self.destination_exists = True

    def puttree(self, local, remote):
        self.calls.append(f"puttree:{local}->{remote}")
        landed = expected_cache_contents(local)
        from aiida_pino.utils.stage_remote import local_checksums

        landed_digests = local_checksums(Path(local))
        if self.destination_exists:
            # cp -r semantics: the source lands INSIDE the existing destination,
            # under its own basename. Depends on the directory EXISTING, not on
            # it holding anything.
            for name, size in landed.items():
                self.tree[f"{Path(local).name}/{name}"] = size
            for name, digest in landed_digests.items():
                self.digests[f"{Path(local).name}/{name}"] = digest
        else:
            self.tree.update(landed)
            self.digests.update(landed_digests)
        self.destination_exists = True
        if self._corrupt:
            self._corrupt(self.tree)
            # A corruption test mutates sizes; keep the digest map to the paths that
            # still exist so a dropped file reads as dropped rather than as unhashed.
            self.digests = {k: v for k, v in self.digests.items() if k in self.tree}
            for name in self.tree:
                self.digests.setdefault(name, "0" * 32)
        return None

    def exec_command_wait(self, command):
        # Staging verifies CONTENT as well as sizes since 2026-08-25, so the fake has to
        # answer both commands. Returning the size listing for a checksum request would
        # make every file read as "not checksummed" -- a fake that cannot model the
        # contract fails the tests rather than testing them.
        if "base64 -d" in command:
            self.calls.append("checksum")
            return 0, "\x00".join(f"{digest} {name}" for name, digest in sorted(self.digests.items())), ""
        self.calls.append("list")
        return 0, "".join(f"{size} {name}\x00" for name, size in sorted(self.tree.items())), ""


def test_a_clean_transfer_verifies_and_reports_what_it_moved(tmp_path):
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    cache = _make_cache(tmp_path / "cache")
    transport = _FakeTransport()
    result = stage_cache_to_computer(transport, local_dir=cache, remote_dir="/remote/cache")
    assert result["file_count"] == 3
    assert result["total_bytes"] == 363
    clear = next(i for i, c in enumerate(transport.calls) if c.startswith("rmtree:"))
    put = next(i for i, c in enumerate(transport.calls) if c.startswith("puttree:"))
    assert clear < put, "destination cleared BEFORE the put"


def test_the_destination_is_cleared_so_a_retry_cannot_nest(tmp_path):
    """Two runs in a row must leave one tree, not a tree inside a tree."""
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    cache = _make_cache(tmp_path / "cache")
    transport = _FakeTransport()
    stage_cache_to_computer(transport, local_dir=cache, remote_dir="/remote/cache")
    stage_cache_to_computer(transport, local_dir=cache, remote_dir="/remote/cache")
    assert set(transport.tree) == {"manifest.json", "members/a.npz", "members/b.npz"}


def test_without_clearing_the_nested_duplicate_is_caught(tmp_path):
    """Proves the check catches the corruption, not just that we avoid it."""
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    cache = _make_cache(tmp_path / "cache")
    transport = _FakeTransport()
    stage_cache_to_computer(transport, local_dir=cache, remote_dir="/remote/cache")
    with pytest_raises_value("unexpected extra path"):
        stage_cache_to_computer(transport, local_dir=cache, remote_dir="/remote/cache", replace=False)


def test_a_truncated_file_fails_staging(tmp_path):
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    def truncate(tree):
        tree["members/a.npz"] = 7

    cache = _make_cache(tmp_path / "cache")
    with pytest_raises_value("size mismatch"):
        stage_cache_to_computer(_FakeTransport(truncate), local_dir=cache, remote_dir="/remote/cache")


def test_a_dropped_file_fails_even_when_the_byte_total_is_made_plausible(tmp_path):
    """Drop one file and pad another so the total matches: still caught."""
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    def drop_and_pad(tree):
        dropped = tree.pop("members/b.npz")
        tree["members/a.npz"] += dropped

    cache = _make_cache(tmp_path / "cache")
    with pytest_raises_value("missing on the remote"):
        stage_cache_to_computer(_FakeTransport(drop_and_pad), local_dir=cache, remote_dir="/remote/cache")


def test_a_directory_without_a_manifest_is_refused_before_any_transfer(tmp_path):
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    (tmp_path / "notacache").mkdir()
    (tmp_path / "notacache" / "x.bin").write_bytes(b"z")
    transport = _FakeTransport()
    with pytest_raises_value("not a runtime cache"):
        stage_cache_to_computer(transport, local_dir=tmp_path / "notacache", remote_dir="/remote/c")
    assert transport.calls == [], "nothing may be transferred before the check"


def test_a_failed_remote_listing_is_a_failure_not_a_pass(tmp_path):
    from aiida_pino.utils.stage_remote import verify_remote_cache

    class _Broken(_FakeTransport):
        def exec_command_wait(self, command):
            return 1, "", "find: not found"

    problems = verify_remote_cache(_Broken(), "/remote/cache", {"manifest.json": 1})
    assert problems and "could not list" in problems[0]


def pytest_raises_value(fragment):
    import pytest

    class _Ctx:
        def __enter__(self):
            self._ctx = pytest.raises(ValueError)
            self._exc = self._ctx.__enter__()
            return self._exc

        def __exit__(self, *args):
            result = self._ctx.__exit__(*args)
            if result:
                assert fragment in str(self._exc.value), f"expected {fragment!r} in {self._exc.value}"
            return result

    return _Ctx()


def test_the_remote_root_comes_from_the_REMOTE_home_not_ours(monkeypatch, tmp_path):
    """Both boxes share /home/<user> today; assuming that is untested luck."""
    from aiida_pino.utils.stage_remote import remote_cache_root

    monkeypatch.setenv("HOME", str(tmp_path))

    class _Box(_FakeTransport):
        def exec_command_wait(self, command):
            return 0, "/home/someone_else\n", ""

    assert remote_cache_root(_Box()) == "/home/someone_else/.cache/aiida-pino/runtime"


def test_an_explicit_remote_root_wins_and_asks_the_box_nothing():
    from aiida_pino.utils.stage_remote import remote_cache_root

    class _Box(_FakeTransport):
        def exec_command_wait(self, command):
            raise AssertionError("must not query the box when told where to put it")

    assert remote_cache_root(_Box(), "/scratch/caches/") == "/scratch/caches"


def test_an_unreadable_remote_home_refuses_rather_than_guessing():
    from aiida_pino.utils.stage_remote import remote_cache_root

    class _Box(_FakeTransport):
        def exec_command_wait(self, command):
            return 1, "", "nope"

    with pytest_raises_value("--remote-root"):
        remote_cache_root(_Box())


def test_the_destination_is_not_recreated_between_the_clear_and_the_put(tmp_path):
    """The bug a stub transport hid, pinned so it cannot come back.

    `puttree` branches on whether the destination EXISTS. A `makedirs` after the
    `rmtree` -- which reads as harmless, and which every "ensure the directory is
    there" instinct asks for -- put the destination straight back and made the
    put nest on EVERY call, clean or not. Found only against a real box.
    """
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    cache = _make_cache(tmp_path / "cache")
    transport = _FakeTransport()
    stage_cache_to_computer(transport, local_dir=cache, remote_dir="/remote/cache")

    put_index = next(i for i, c in enumerate(transport.calls) if c.startswith("puttree:"))
    before_put = transport.calls[:put_index]
    assert not any(c.startswith("makedirs:") for c in before_put), (
        "nothing may recreate the destination between the clear and the put; "
        f"calls were {before_put}"
    )
    assert set(transport.tree) == {"manifest.json", "members/a.npz", "members/b.npz"}


def test_a_matching_path_alone_does_not_refuse(tmp_path):
    """REPLACES an earlier test that refused on the path STRING alone.

    That test encoded a false positive. Comparing `remote_dir` to `local_dir` as
    text refuses every machine provisioned with the same home layout — which is
    the normal case, and was the deployment `stage-remote` exists for. Identity
    is now decided by whether the box can see a probe written here, so a
    matching path is no longer sufficient grounds to refuse; see
    `test_the_same_directory_is_still_refused_when_it_really_is_the_same` for
    the case that must still fail.
    """
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    cache = _make_cache(tmp_path / "cache")
    result = stage_cache_to_computer(_SharedHomeBox(), local_dir=cache, remote_dir=str(cache))
    assert result["file_count"] == 3


def test_a_failed_restage_does_not_leave_a_record_pointing_at_the_wreckage(tmp_path, aiida_profile_clean):
    """The clear happens before the copy, so a failure destroys the good copy.

    If a RemoteData still names that path, the next submission stages the
    wreckage and the job dies at step 0 -- the exact silent failure this whole
    path exists to prevent. A failed staging must retract the record.
    """
    from aiida import orm

    from aiida_pino.utils.data_nodes import find_remote_runtime_cache, register_remote_runtime_cache
    from aiida_pino.utils.stage_remote import retract_remote_runtime_cache

    computer = orm.Computer(
        label="wreckage-box",
        hostname="localhost",
        transport_type="core.ssh_async",
        scheduler_type="core.direct",
        workdir="/tmp/aiida_run",
    )
    computer.store()
    register_remote_runtime_cache(computer, cache_name="cache-x", remote_path="/remote/cache-x")
    assert find_remote_runtime_cache(computer, "cache-x") is not None

    retract_remote_runtime_cache(computer, "cache-x")
    assert find_remote_runtime_cache(computer, "cache-x") is None


class _SharedHomeBox(_FakeTransport):
    """A genuinely remote machine that happens to use the same home path.

    Consistent user provisioning gives every box `/home/<user>`, so this is the
    normal case on a lab cluster, not an edge case. A probe file written here is
    NOT visible there.
    """

    def exec_command_wait(self, command):
        if "aiida-pino-stage-probe" in command:
            return 0, "NO\n", ""
        return super().exec_command_wait(command)


class _LoopbackBox(_FakeTransport):
    """A core.ssh_async computer pointed at this machine: same filesystem.

    A probe written locally IS visible at the destination, because they are the
    same directory. This is the case the guard exists for.
    """

    def exec_command_wait(self, command):
        if "aiida-pino-stage-probe" in command:
            return 0, "YES\n", ""
        return super().exec_command_wait(command)


def test_a_remote_box_sharing_the_home_path_is_staged_normally(tmp_path):
    """The false positive that made stage-remote fail on its own target deployment.

    Comparing path STRINGS refuses any machine with the same home layout, which
    is what consistent provisioning produces. The question is whether the two
    paths are the same DIRECTORY, and only the filesystem can answer that.
    """
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    cache = _make_cache(tmp_path / "cache")
    transport = _SharedHomeBox()
    result = stage_cache_to_computer(transport, local_dir=cache, remote_dir=str(cache))
    assert result["file_count"] == 3


def test_the_same_directory_is_still_refused_when_it_really_is_the_same(tmp_path):
    """The destructive case must stay refused: a loopback computer."""
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    cache = _make_cache(tmp_path / "cache")
    transport = _LoopbackBox()
    with pytest_raises_value("same directory"):
        stage_cache_to_computer(transport, local_dir=cache, remote_dir=str(cache))
    assert (cache / "manifest.json").is_file(), "the local cache must survive"
    assert not any(c.startswith("rmtree:") for c in transport.calls), "nothing may be removed"


def test_the_probe_never_survives_in_the_staged_payload(tmp_path):
    """A probe left behind would be transferred and then flagged as an extra."""
    from aiida_pino.utils.stage_remote import stage_cache_to_computer

    cache = _make_cache(tmp_path / "cache")
    stage_cache_to_computer(_SharedHomeBox(), local_dir=cache, remote_dir="/remote/cache")
    leftover = [p.name for p in cache.rglob("*aiida-pino-stage-probe*")]
    assert leftover == [], f"probe files left behind: {leftover}"
