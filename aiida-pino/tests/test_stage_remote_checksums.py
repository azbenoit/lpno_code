"""Staging must be able to prove CONTENT arrived, not just byte counts.

2026-08-25. `r150np_137.npz` reached the compute box with the SAME byte count as the local source
and a DIFFERENT md5; its `Fraction Of Fluid.npy` failed CRC and training 67664 died at ~87
steps on `zipfile.BadZipFile: Bad CRC-32`. `stage-remote` had reported `staged: True`.

The existing verification compares per-file paths and SIZES, which a same-size corruption
passes. Found only by md5-comparing all 120 files by hand.

**The page-cache caveat is load-bearing.** A checksum taken shortly after writing is served
the writer's still-cached pages and passes while the on-disk bytes are bad -- the
torn-file effect already recorded for t64v_087. The reporter's first post-transfer md5 was
worthless for that reason; dropping each file's pages with `posix_fadvise(DONTNEED)` (no
root needed) is what made the re-check meaningful.
"""

from __future__ import annotations

from pathlib import Path

from aiida_pino.utils.stage_remote import (
    compare_checksums,
    local_checksums,
    parse_checksum_records,
    remote_checksum_command,
)


def test_local_checksums_are_by_relative_path(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.npz").write_bytes(b"one")
    (tmp_path / "sub" / "b.npz").write_bytes(b"two")
    out = local_checksums(tmp_path)
    assert sorted(out) == ["a.npz", "sub/b.npz"]
    assert all(len(v) == 32 for v in out.values())


def test_a_same_size_different_content_file_is_caught(tmp_path):
    """The reported corruption exactly: identical byte count, different bytes."""
    (tmp_path / "r150np_137.npz").write_bytes(b"good-bytes")
    expected = local_checksums(tmp_path)
    (tmp_path / "r150np_137.npz").write_bytes(b"BAD!-bytes")  # same length
    actual = local_checksums(tmp_path)

    assert len(b"good-bytes") == len(b"BAD!-bytes"), "the test itself must be same-size"
    problems = compare_checksums(expected, actual)
    assert len(problems) == 1 and "CONTENT mismatch" in problems[0]


def test_identical_content_reports_nothing(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"same")
    expected = local_checksums(tmp_path)
    assert compare_checksums(expected, dict(expected)) == []


def test_a_file_the_remote_could_not_hash_is_a_problem_not_a_pass():
    """Unverifiable must never read as verified.

    Treating a missing result as fine is how the staged-cache repair stayed inert on
    remote boxes for months.
    """
    problems = compare_checksums({"a.npz": "0" * 32}, {})
    assert len(problems) == 1 and "not checksummed" in problems[0]


def test_the_remote_records_round_trip():
    records = "abc" + "0" * 29 + " a.npz\0" + "def" + "0" * 29 + " sub/b.npz"
    parsed = parse_checksum_records(records)
    assert parsed == {"a.npz": "abc" + "0" * 29, "sub/b.npz": "def" + "0" * 29}


def test_malformed_records_are_dropped_not_guessed():
    assert parse_checksum_records("") == {}
    assert parse_checksum_records("short a.npz") == {}      # not a 32-char digest
    assert parse_checksum_records("0" * 32 + " ") == {}      # no path


def test_the_remote_command_drops_the_page_cache_and_survives_quoting(tmp_path):
    """Run the real snippet locally: it is the same code the remote executes."""
    import subprocess

    (tmp_path / "sub").mkdir()
    (tmp_path / "a.npz").write_bytes(b"one")
    (tmp_path / "sub" / "b.npz").write_bytes(b"two")

    command = remote_checksum_command(str(tmp_path))
    assert "POSIX_FADV_DONTNEED" not in command, "the snippet is base64, not inlined"
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    assert parse_checksum_records(result.stdout) == local_checksums(tmp_path)


def test_a_path_with_spaces_survives(tmp_path):
    """`%s %P` splitting bit the size check once; the checksum parser must not repeat it."""
    import subprocess

    (tmp_path / "Fraction Of Fluid.npy").write_bytes(b"payload")
    command = remote_checksum_command(str(tmp_path))
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert parse_checksum_records(result.stdout) == local_checksums(tmp_path)
