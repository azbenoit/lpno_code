"""Backfilled values are marked as such; native ones are not.

A value written at materialisation can only be wrong through a code bug. A
backfilled one is an inference made after the fact, and inference carries error
the original never had -- in one week a backfill read the wrong `[step: N]`
line, another mistook a peak-frame width for a section width, and a lineage
pointer claimed "derived from" on evidence the grids later refuted. All were
caught; nothing on the node would have told a later reader which values were
exposed to that class of mistake.
"""

from __future__ import annotations

from aiida import orm

from aiida_pino.utils.provenance_mode import (
    KEY,
    is_backfilled,
    mark_backfilled,
    provenance_of,
)


def test_marking_is_additive_and_idempotent(aiida_profile_clean):
    node = orm.Data().store()
    mark_backfilled(node, ["held_out"])
    mark_backfilled(node, ["held_out"])          # same key twice
    mark_backfilled(node, ["solidus_k"])         # a second key later
    assert node.base.extras.get(KEY) == ["held_out", "solidus_k"]
    assert is_backfilled(node, "held_out") and is_backfilled(node, "solidus_k")
    assert not is_backfilled(node, "num_members")


def test_a_dry_run_marks_nothing(aiida_profile_clean):
    node = orm.Data().store()
    planned = mark_backfilled(node, ["held_out"], apply=False)
    assert planned == ["held_out"]
    assert node.base.extras.get(KEY, None) is None


def test_provenance_of_separates_mode_from_source(aiida_profile_clean):
    """`material_source: manifest` says WHERE the value came from conceptually.

    It does not say whether it was written at creation -- all 3,894 sims
    carrying that tag were stamped by a backfill. Both facts are needed.
    """
    node = orm.Data()
    node.base.extras.set("solidus_k", 1873.15)
    node.base.extras.set("material_source", "manifest")
    node.store()

    before = provenance_of(node, "solidus_k")
    assert before["mode"] == "native" and before["source"] == "manifest"

    mark_backfilled(node, ["solidus_k"])
    after = provenance_of(node, "solidus_k")
    assert after["mode"] == "backfilled"
    assert after["source"] == "manifest", "the semantic source is unchanged by the mode"
    assert after["value"] == 1873.15


def test_an_unmarked_value_reads_native_and_that_is_a_default_not_a_promise(aiida_profile_clean):
    node = orm.Data()
    node.base.extras.set("num_members", 12)
    node.store()
    assert provenance_of(node, "num_members")["mode"] == "native"
    assert provenance_of(node, "absent_key")["present"] is False


def test_backfilled_values_are_findable_in_one_filter(aiida_profile_clean):
    marked, plain = orm.Data().store(), orm.Data().store()
    mark_backfilled(marked, ["held_out"])
    found = orm.QueryBuilder().append(
        orm.Data, filters={"extras.backfilled_keys": {"contains": ["held_out"]}},
        project=["id"],
    ).all(flat=True)
    assert marked.pk in found and plain.pk not in found
