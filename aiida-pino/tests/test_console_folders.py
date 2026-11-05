"""Folders (AiiDA groups) for the dataset browser."""

from __future__ import annotations

import pytest
from aiida import orm

from aiida_pino.console import folders
from aiida_pino.data.dataset_hierarchy import create_base_dataset
from web.app.aiida_service import AiidaConsoleService
from web.app.config import Settings


@pytest.fixture
def console_service():
    return AiidaConsoleService(Settings())


def _dataset(factory, name, members=2):
    node, _ = create_base_dataset([factory(f"{name}_{i}") for i in range(members)], name=name)
    return node


def test_set_folder_moves_between_folders_and_prunes_empties(aiida_profile_clean, pino_simulation_factory) -> None:
    node = _dataset(pino_simulation_factory, "keyhole")
    assert folders.set_folder(node, "reference/keyhole") == "reference/keyhole"
    assert folders.folder_by_pk([node.pk]) == {node.pk: "reference/keyhole"}

    folders.set_folder(node, "reference/steady")
    assert folders.folder_by_pk([node.pk]) == {node.pk: "reference/steady"}
    assert folders.list_folders() == ["reference/steady"], "the emptied folder was left behind"

    folders.set_folder(node, None)
    assert folders.folder_by_pk([node.pk]) == {}
    assert folders.list_folders() == []


def test_unfiled_datasets_get_an_automatic_folder(aiida_profile_clean) -> None:
    assert folders.auto_folder({"simulation_count": 1}) == folders.ORPHANED
    assert folders.auto_folder({"simulation_count": 0}) == folders.ORPHANED
    assert folders.auto_folder({"simulation_count": 2}) is None
    # AL rounds mint datasets continuously; they file themselves.
    assert folders.auto_folder({"simulation_count": 9, "keywords": ["self-improve"]}) == folders.SELF_IMPROVE


def test_listing_reports_folder_and_orphans(aiida_profile_clean, pino_simulation_factory, console_service) -> None:
    curated = _dataset(pino_simulation_factory, "curated", members=3)
    stray = _dataset(pino_simulation_factory, "stray", members=1)
    console_service.set_folder([curated.pk], "reference")

    by_pk = {row["pk"]: row for row in console_service.list_datasets(role="all")}
    assert by_pk[curated.pk]["folder"] == "reference"
    assert by_pk[stray.pk]["folder"] == folders.ORPHANED


def test_listing_is_not_truncated_to_the_newest_page(aiida_profile_clean, pino_simulation_factory, console_service) -> None:
    """The bug that hid the core datasets: per-case collections outgrew the
    default page and pushed every older curated set off the end."""
    oldest = _dataset(pino_simulation_factory, "oldest", members=2)
    for i in range(5):
        _dataset(pino_simulation_factory, f"newer_{i}", members=2)

    listed = [row["pk"] for row in console_service.list_datasets(role="all")]
    assert oldest.pk in listed
    assert len(listed) >= 6


def test_aiida_groups_are_not_folders(aiida_profile_clean, pino_simulation_factory) -> None:
    """Folders are extras, not groups. An archive import puts every imported
    node in a `core.import` group; when folders were groups that one swallowed
    the whole browser into a single entry."""
    from aiida.orm import ImportGroup

    node = _dataset(pino_simulation_factory, "imported")
    imported = ImportGroup(label="20260721-132900").store()
    imported.add_nodes([node])

    assert folders.folder_by_pk([node.pk]) == {}
    assert folders.list_folders() == []


def test_filing_works_off_the_main_thread(aiida_profile_clean, pino_simulation_factory) -> None:
    """The console serves sync endpoints from a threadpool. Folders as AiiDA
    groups died there: the cached default User stays bound to the first
    thread's session, so storing a group from another thread raised
    InvalidRequestError and left db_dbgroup.user_id null."""
    import threading

    from aiida import orm

    orm.User.collection.get_default()  # bind the cached user to this thread first
    node = _dataset(pino_simulation_factory, "threaded")
    outcome = {}

    def file_it():
        try:
            outcome["folder"] = folders.set_folder(orm.load_node(node.pk), "worker/thread")
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=file_it)
    worker.start()
    worker.join()

    assert "error" not in outcome, outcome.get("error")
    assert outcome["folder"] == "worker/thread"
    assert folders.folder_by_pk([node.pk]) == {node.pk: "worker/thread"}


def test_rename_folder_keeps_its_datasets(aiida_profile_clean, pino_simulation_factory) -> None:
    a, b = _dataset(pino_simulation_factory, "a"), _dataset(pino_simulation_factory, "b")
    folders.set_folder(a, "refernce")  # typo, the reason rename exists
    folders.set_folder(b, "refernce")

    folders.rename_folder("refernce", "reference")

    assert folders.list_folders() == ["reference"]
    assert folders.folder_by_pk([a.pk, b.pk]) == {a.pk: "reference", b.pk: "reference"}


def test_rename_onto_an_existing_folder_merges(aiida_profile_clean, pino_simulation_factory) -> None:
    a, b = _dataset(pino_simulation_factory, "a"), _dataset(pino_simulation_factory, "b")
    folders.set_folder(a, "old")
    folders.set_folder(b, "keep")

    folders.rename_folder("old", "keep")

    assert folders.list_folders() == ["keep"]
    assert folders.folder_by_pk([a.pk, b.pk]) == {a.pk: "keep", b.pk: "keep"}


def test_remove_folder_unfiles_its_datasets(aiida_profile_clean, pino_simulation_factory) -> None:
    a = _dataset(pino_simulation_factory, "a")
    folders.set_folder(a, "scratch")

    folders.remove_folder("scratch")

    assert folders.list_folders() == []
    assert folders.folder_by_pk([a.pk]) == {}
    assert orm.load_node(a.pk).pk == a.pk, "removing a folder must not touch its datasets"


def test_rename_and_remove_reject_unknown_folders(aiida_profile_clean) -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError):
        folders.rename_folder("nope", "other")
    with _pytest.raises(ValueError):
        folders.remove_folder("nope")
