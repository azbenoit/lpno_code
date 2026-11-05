"""Folders for the dataset and model browsers.

A folder is a `folder` extra on the node, holding a slash-separated path. It
started life as an AiiDA group, which reads better on paper — `verdi group
list` would have seen them — but creating a group from the console's request
threadpool blows up: AiiDA caches the default `User` on its collection, that
DbUser stays bound to the session of whichever thread first touched it, and
storing a group from another thread raises `InvalidRequestError: Object
'<DbUser>' is already attached to session N`, then a NOT NULL violation on
`db_dbgroup.user_id`. Extras are how every other annotation here is stored
(keywords, split, superseded), they go through the node ORM, and they are
thread-safe on this backend — so folders are extras.

A node is in at most one folder. A folder exists exactly as long as something
is in it: filing the last node out makes it disappear, which is why there is no
separate create or delete step — typing a new name in the browser creates it.

Nodes nobody filed are placed automatically: a collection holding a single
simulation is an import artifact rather than a curated set, so it lands in
`ORPHANED`, which the browser keeps collapsed at the bottom of the list.
"""

from __future__ import annotations

from typing import Any

FOLDER_EXTRA = "folder"
# Not "orphans": these are single-case imports, and several are members of
# curated training sets. The old name read as "broken", which is why 640 of
# them sat unread at the bottom of the list.
ORPHANED = "Single-run imports"
SELF_IMPROVE = "Self-improvement (AL)"


def normalize(path: str | None) -> str:
    """Trim a user-typed folder path; empty means "in no folder"."""
    return (path or "").strip().strip("/").strip()


def folder_by_pk(pks: list[int]) -> dict[int, str]:
    """Map node pk -> folder, for the nodes that are in one."""
    from aiida import orm

    if not pks:
        return {}
    builder = orm.QueryBuilder()
    builder.append(
        orm.Node,
        filters={"id": {"in": list(pks)}, "extras": {"has_key": FOLDER_EXTRA}},
        project=["id", f"extras.{FOLDER_EXTRA}"],
    )
    return {int(pk): str(folder) for pk, folder in builder.all() if folder}


def list_folders(*, include_computed: bool = False) -> list[str]:
    """Every folder in use, sorted.

    `ORPHANED` and `SELF_IMPROVE` are computed by `auto_folder` at render time
    and are stored on no node, so a query over the `folder` extra cannot see
    them: the CLI listed 8 folders where the browser shows 10, and the two it
    omitted hold 722 of 836 datasets. They are marked, because they are not
    addressable — `nodes_in_folder` returns nothing for them and rename/remove
    refuse them.
    """
    from aiida import orm

    builder = orm.QueryBuilder()
    builder.append(orm.Node, filters={"extras": {"has_key": FOLDER_EXTRA}}, project=[f"extras.{FOLDER_EXTRA}"])
    filed = sorted({str(folder) for folder, in builder.all() if folder})
    if not include_computed:
        return filed
    computed = [f"{name} (computed, not filed)" for name in (ORPHANED, SELF_IMPROVE) if name not in filed]
    return filed + computed


def nodes_in_folder(label: str, *, include_subfolders: bool = False) -> list:
    """Every node filed under `label`.

    With `include_subfolders`, also everything in `label/...`. Folder paths are
    `/`-separated and every segment is a real folder, so an operation on a
    folder that ignores its subfolders silently leaves them behind — which is
    what made renaming (and therefore moving) a folder detach its contents.
    """
    from aiida import orm

    if not include_subfolders:
        filters = {f"extras.{FOLDER_EXTRA}": {"==": label}}
    else:
        filters = {
            "or": [
                {f"extras.{FOLDER_EXTRA}": {"==": label}},
                {f"extras.{FOLDER_EXTRA}": {"like": f"{label}/%"}},
            ]
        }
    builder = orm.QueryBuilder()
    builder.append(orm.Node, filters=filters, project=["*"])
    return list(builder.all(flat=True))


def set_folder(node, path: str | None) -> str | None:
    """Move `node` into folder `path`, or out of any folder when empty."""
    label = normalize(path)
    if not label:
        try:
            node.base.extras.delete(FOLDER_EXTRA)
        except AttributeError:
            pass  # it was not in a folder
        return None
    node.base.extras.set(FOLDER_EXTRA, label)
    return label


def rename_folder(old: str, new: str) -> str:
    """Rename or move a folder, **with its subfolders**.

    Renaming onto an existing folder merges into it, and moving is the same
    operation: `rename_folder("A", "B/A")` nests A inside B.

    The whole subtree moves. Renaming only the nodes filed at exactly `old`
    left everything in `old/child` pointing at a parent that no longer existed,
    so a move looked like a rename that lost the folder's contents.
    """
    label = normalize(new)
    if not label:
        raise ValueError("the new folder name is required")
    source = normalize(old)
    if not source:
        raise ValueError("the folder to rename is required")
    if label == source:
        return label
    if label.startswith(f"{source}/"):
        raise ValueError(f"cannot move {source!r} inside itself ({label!r})")
    members = nodes_in_folder(source, include_subfolders=True)
    if not members:
        raise ValueError(f"no folder {source!r}")
    for node in members:
        current = str(node.base.extras.get(FOLDER_EXTRA, "") or "")
        suffix = current[len(source):] if current.startswith(source) else ""
        node.base.extras.set(FOLDER_EXTRA, f"{label}{suffix}")
    return label


def remove_folder(label: str) -> int:
    """Delete a folder and its subfolders, unfiling their nodes.

    The nodes themselves are untouched. Subfolders are included for the same
    reason as in `rename_folder`: leaving them behind orphans them under a
    parent that no longer exists.
    """
    members = nodes_in_folder(normalize(label), include_subfolders=True)
    if not members:
        raise ValueError(f"no folder {label!r}")
    for node in members:
        node.base.extras.delete(FOLDER_EXTRA)
    return len(members)


def auto_folder(row: dict[str, Any]) -> str | None:
    """Folder for a node the user has not filed, or None to leave it top-level.

    Only for kinds that arrive continuously and would otherwise re-clutter the
    list every day: single-simulation import artifacts, and the datasets each
    active-learning round mints. Explicit filing always wins over this.
    """
    if "simulation_count" in row and int(row.get("simulation_count") or 0) <= 1:
        return ORPHANED
    if "self-improve" in (row.get("keywords") or []):
        return SELF_IMPROVE
    return None


def resolve_rows(rows: list[dict[str, Any]]) -> None:
    """Fill each row's ``folder`` in place: explicit, else inherited, else auto.

    A derived dataset lives where its base lives, by inheritance rather than by
    writing anything on it — so moving the base moves its children, and unfiling
    it takes them back out, with no second copy of the truth. Rows need
    ``pk``, ``_inherit_pk`` and ``_folder_own``.

    Inheritance follows ``_inherit_pk``, which the caller sets only for *recorded*
    lineage. A parent guessed from member-set inclusion must not relocate a
    dataset: that guess put every single-case import under whichever curated set
    happened to contain it.

    The walk is guarded: nothing should ever create a parent cycle, but a
    hand-edited attribute could, and a listing that never returns is worse than
    a wrong parent.
    """
    by_pk = {row["pk"]: row for row in rows}

    def walk(row, seen):
        if row.get("folder") is not None:
            return row["folder"]
        if row.get("_folder_own"):
            row["folder"] = row["_folder_own"]
        else:
            parent = by_pk.get(row.get("_inherit_pk"))
            if parent is not None and row["pk"] not in seen:
                row["folder"] = walk(parent, seen | {row["pk"]})
        if row.get("folder") is None:
            row["folder"] = auto_folder(row)
        return row["folder"]

    for row in rows:
        walk(row, frozenset())
