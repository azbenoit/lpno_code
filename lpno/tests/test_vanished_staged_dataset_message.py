"""A staged dataset that vanished must say so, not blame VTK.

aiida-pino symlinks a job's dataset dir to `~/.cache/aiida-pino/runtime/<uuid>`
at *upload* time, but the trainer reads it at *execution* time -- and on a busy
GPU seat hours can pass in between. Clearing the cache in that window leaves the
symlink dangling, and the loader's fallthrough blamed the wrong thing:

    Direct VTK directory loading is deprecated: '<path>' is neither a
    consolidated-NPZ dataset nor an arraydir runtime. Convert the cases to
    consolidated NPZ ...

which sends you off converting datasets that were never the problem. Observed
twice: five RIGNO ablation arms on 2026-07-22 (8392/8397/8399/8401/8403), and
run 9163 on 2026-07-23, whose cache was pruned 2 min before it won the seat.
"""
from __future__ import annotations

import pytest

from lpno.models.pipeline.preprocessing_datasets import (
    _describe_missing_dataset_dir,
)


def test_dangling_symlink_names_the_runtime_cache(tmp_path):
    target = tmp_path / "cache" / "runtime" / "6298cd58-4fe1-4f69-b430-81b5d419e613"
    link = tmp_path / "train_dataset"
    link.symlink_to(target)  # target never created: the cache entry is gone

    message = _describe_missing_dataset_dir(link)

    assert message is not None
    assert "symlink" in message.lower()
    assert str(target) in message
    assert "VTK" not in message
    # Must point at the actual cause, not at dataset conversion.
    assert "cache" in message.lower()


def test_missing_path_is_reported_as_missing(tmp_path):
    message = _describe_missing_dataset_dir(tmp_path / "not_there")

    assert message is not None
    assert "does not exist" in message.lower()


def test_empty_directory_is_reported_as_empty(tmp_path):
    empty = tmp_path / "staged"
    empty.mkdir()

    message = _describe_missing_dataset_dir(empty)

    assert message is not None
    assert "empty" in message.lower()


def test_a_real_dataset_dir_is_not_flagged(tmp_path):
    """A populated directory is a genuine format problem -- keep the VTK error."""
    good = tmp_path / "staged"
    good.mkdir()
    (good / "manifest.json").write_text("{}")

    assert _describe_missing_dataset_dir(good) is None


def test_file_instead_of_directory_is_flagged(tmp_path):
    not_a_dir = tmp_path / "staged"
    not_a_dir.write_text("oops")

    message = _describe_missing_dataset_dir(not_a_dir)

    assert message is not None
    assert "not a directory" in message.lower()
