"""Folder paths are `/`-separated and every segment is a real folder.

Operations that ignored subfolders left them pointing at a parent that no
longer existed, which is what made moving a folder in the GUI behave like a
rename that lost the folder's contents.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from aiida_pino.console import folders


def _filed(factory, name, folder):
    node = factory(name, group_name=f"{name}-1")
    folders.set_folder(node, folder)
    return node


def test_renaming_a_folder_carries_its_subfolders(aiida_profile_clean, pino_simulation_factory):
    parent = _filed(pino_simulation_factory, "p", "Campaign")
    child = _filed(pino_simulation_factory, "c", "Campaign/Round one")
    deep = _filed(pino_simulation_factory, "d", "Campaign/Round one/Retries")

    folders.rename_folder("Campaign", "Archive/Campaign")

    assert parent.base.extras.get("folder") == "Archive/Campaign"
    assert child.base.extras.get("folder") == "Archive/Campaign/Round one"
    assert deep.base.extras.get("folder") == "Archive/Campaign/Round one/Retries"


def test_moving_a_folder_into_itself_is_refused(aiida_profile_clean, pino_simulation_factory):
    _filed(pino_simulation_factory, "s", "Sweep")
    with pytest.raises(ValueError, match="inside itself"):
        folders.rename_folder("Sweep", "Sweep/Inner")


def test_a_sibling_with_a_shared_prefix_is_not_moved(aiida_profile_clean, pino_simulation_factory):
    """`Campaign2` must not be dragged along by an operation on `Campaign` —
    the match is on the path segment, not the string prefix."""
    target = _filed(pino_simulation_factory, "a", "Campaign")
    sibling = _filed(pino_simulation_factory, "b", "Campaign2")

    folders.rename_folder("Campaign", "Done")

    assert target.base.extras.get("folder") == "Done"
    assert sibling.base.extras.get("folder") == "Campaign2"


def test_removing_a_folder_unfiles_its_subfolders_too(aiida_profile_clean, pino_simulation_factory):
    top = _filed(pino_simulation_factory, "t", "Temp")
    inner = _filed(pino_simulation_factory, "i", "Temp/Inner")

    assert folders.remove_folder("Temp") == 2
    assert top.base.extras.get("folder", None) is None
    assert inner.base.extras.get("folder", None) is None
